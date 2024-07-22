import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'
import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
import random
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def str_to_float_list(arg):
    return [float(x) for x in arg.strip('[]').split(',')]

def add_dataset_args(parser):
    
    parser.add_argument("--env", type=str, required=True, help="Environment")
    
    parser.add_argument("--bustotal", type=int, required=False,
                        default=100, help="Total number of buses")

    parser.add_argument("--beta", type=float, required=False,
                        default=0.95, help="Beta")
    parser.add_argument("--theta", type=str_to_float_list, required=False, default="[1, 5, 1]", help="Theta values as a list of floats")
    
    parser.add_argument("--H", type=int, required=False,
                        default=100, help="Context horizon")
    
    parser.add_argument("--maxMileage", type=int, required=False,
                        default=10, help="Max mileage")
    parser.add_argument("--numTypes", type=int, required=False,
                        default=10, help="Number of bus types")
    parser.add_argument("--extrapolation", type=str, required=False,
                        default='False', help="Extrapolation")
    


def add_model_args(parser):
    parser.add_argument("--embd", type=int, required=False,
                        default=128, help="Embedding size")
    parser.add_argument("--head", type=int, required=False,
                        default=1, help="Number of heads")
    parser.add_argument("--layer", type=int, required=False,
                        default=8, help="Number of layers")
    parser.add_argument("--lr", type=float, required=False,
                        default=1e-3, help="Learning Rate")
    parser.add_argument("--dropout", type=float,
                        required=False, default=0, help="Dropout")
    parser.add_argument('--shuffle', default=False, action='store_true')


def add_train_args(parser):
    parser.add_argument("--num_epochs", type=int, required=False,
                        default=5000, help="Number of epochs")


def add_eval_args(parser):
    parser.add_argument("--epoch", type=int, required=False,
                        default=-1, help="Epoch to evaluate")
    parser.add_argument("--hor", type=int, required=False,
                        default=-1, help="Episode horizon (for mdp)")
    parser.add_argument("--n_eval", type=int, required=False,
                        default=100, help="Number of eval trajectories")





def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        query_states = []
        query_actions = []
        query_true_EPs = []
        query_true_Qs = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])

            query_states.append(traj['query_state'])
            query_actions.append(traj['query_action'])
            query_true_EPs.append(traj['query_true_EP'])
            query_true_Qs.append(traj['query_true_Q'])  

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        
        #if len(context_rewards.shape) < 3:
        #    context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states) #its dimension is (num_samples, state_dim)
        query_actions = np.array(query_actions)
        
        query_true_EPs = np.array(query_true_EPs)
        query_true_Qs = np.array(query_true_Qs)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'query_actions': convert_to_tensor(query_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'query_true_EPs': convert_to_tensor(query_true_EPs, store_gpu=self.store_gpu),
            'query_true_Qs': convert_to_tensor(query_true_Qs, store_gpu=self.store_gpu),
        }
        
        self.zeros = np.zeros(
            config['maxMileage'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'query_states': self.dataset['query_states'][index],
            'query_actions': self.dataset['query_actions'][index],
            'query_true_EPs': self.dataset['query_true_EPs'][index],
            'query_true_Qs': self.dataset['query_true_Qs'][index],
            'zeros': self.zeros
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]

        return res


class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = 1
        self.action_dim = 1
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=4,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config) # Model

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim, self.n_embd)
        
        self.pred_q_values = nn.Linear(self.n_embd, 2)

    def forward(self, x):
        query_states = x['query_states'][:, None] # (batch_size, 1, state_dim). #None and unsqueeze are equivalent
        zeros = x['zeros'][:, None] # (batch_size, 1, state_dim+1)
        
        state_seq = torch.cat([query_states, x['context_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        
        action_seq = torch.cat(
            [zeros[:, :, 1], x['context_actions']], dim=1) # (batch_size, 1+horizon, action_dim)
        next_state_seq = torch.cat(
            [zeros[:, :, 1], x['context_next_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        
        
        state_seq = state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
        action_seq = action_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, action_dim)
        next_state_seq = next_state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
       

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq], dim=2) 
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_q_values(transformer_outputs['last_hidden_state'])
        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]



def build_Zurcher_data_filename(env, config, mode):
    """
    Builds the filename for the Zurcher data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    if mode != 4:
        filename_template = 'datasets/trajs_{}.pkl'
    else: 
        filename_template = '{}'
    filename = env
    if mode != 2:
        filename += '_bustotal' + str(config['bustotal'])
    filename += '_beta' + str(config['beta'])
    filename += '_theta' + str(config['theta'])    
    filename += '_numTypes' + str(config['numTypes'])    
    filename += '_H' + str(config['horizon'])
    filename += '_' + config['rollin_type']
    #filename += '_extrapolation' + str(config['extrapolation'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_n_eval' + str(config['n_eval'])
        filename += '_eval'
        
    return filename_template.format(filename)



def build_Zurcher_model_filename(env, config):
    """
    Builds the filename for the Zurcher model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    #filename += '_Bustotal' + str(config['Bustotal'])
    #filename += '_beta' + str(config['beta'])
    #filename += '_theta' + str(config['theta'])
    #filename += '_numTypes' + str(config['numTypes'])
    filename += '_H' + str(config['horizon'])
    filename += '_seed' + str(config['seed'])
    return filename



###########################################  Main  ###########################################


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    add_model_args(parser)
    add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    bustotal = args['bustotal']
    theta = args['theta']
    beta = args['beta']
    horizon = args['H']
    xmax = args['maxMileage']
    state_dim = xmax
    action_dim = 2
    numTypes = args['numTypes']
    extrapolation = args['extrapolation']
    
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    num_epochs = args['num_epochs']
    seed = args['seed']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0


    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

   
    dataset_config = {
        'horizon': horizon,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'horizon': horizon,
        'seed': seed,
    }
    if env.startswith('Zurcher'):
        state_dim = 1
        action_dim = 2

        dataset_config.update({'bustotal': bustotal, 'maxMileage': xmax,'theta': theta, 'beta': beta, 'xmax': xmax, 
                               'numTypes': numTypes, 'extrapolation': extrapolation,'rollin_type': 'expert'})
        
        path_train = build_Zurcher_data_filename(
            env, dataset_config, mode=0)
        path_test = build_Zurcher_data_filename(
            env, dataset_config, mode=1)

        filename = build_Zurcher_model_filename(env, model_config)
        fig_filename = build_Zurcher_data_filename(env, dataset_config, mode=4)

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        #'bustotal': bustotal,
        #'bus_types': numTypes,
        #'theta': theta,
        #'beta': beta,
        'maxMileage': xmax,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
    }
    model = Transformer(config).to(device)

    params = {
        'batch_size': 64,
        'shuffle': True,
    }

    log_filename = f'figs/loss/{fig_filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)



    train_dataset = Dataset(path_train, config)
    test_dataset = Dataset(path_test, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    MSE_loss_fn = torch.nn.MSELoss(reduction='sum')

    
    train_loss = []
    test_loss = []
    test_full_loss = []
    test_Q_MSE_loss = []
    test_full_Q_MSE_loss = []
    
    best_epoch = -1
    best_Q_MSE_loss = float('inf')
    best_normalized_true_Qs = None
    best_normalized_full_pred_q_values = None

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))
 
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_CrossEntropy_loss = 0.0
            epoch_full_CrossEntropy_loss = 0.0
            epoch_Q_MSE_loss = 0.0
            
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                
                #query_action is one-hot encoded action for each query state
                true_actions = batch['query_actions'].long() #dimension is (batch_size)
                pred_q_values = model(batch) #dimension is (batch_size, horizon, action_dim)
                
                true_actions_unsqueezed = true_actions.unsqueeze(
                1).repeat(1, pred_q_values.shape[1]) #dimension is (batch_size, horizon)
                true_actions_reshaped = true_actions_unsqueezed.reshape(-1) #dimension is (batch_size*horizon,)
                
                query_states = batch['query_states'] #dimension is (batch_size, state_dim)
                
                #query_true_Q is true Q values of actions for each query state
                query_true_Qs = batch['query_true_Qs'] #dimension is (batch_size, action_dim)
            
                #For each period k, the model predicts the Q values of actions of query state depending
                #   only on period 1...period k inputs
                #GPT2 implements "causal masking", i.e., output at period k can only depend on inputs at period 0, ..., k
                
                pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
                
                #prediction depending on the whole context, not partial context
                full_pred_q_values = pred_q_values[:,-1,:] #dimension is (batch_size, action_dim)
            
                ####### Action CrossEntropy loss                
                cross_entropy_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped)
                epoch_CrossEntropy_loss += cross_entropy_loss.item()/horizon
                
                ####### (Full context) Action CrossEntropy loss
                full_cross_entropy_loss = CrossEntropy_loss_fn(full_pred_q_values, true_actions)
                epoch_full_CrossEntropy_loss += full_cross_entropy_loss.item()
                
                
                ####### Q value MSE loss
                #Normalized Q values
                min_true_Qs = torch.min(query_true_Qs, dim=1, keepdim=True)[0]
                normalized_true_Qs = query_true_Qs - min_true_Qs
            
                min_q_values = torch.min(full_pred_q_values, dim=1, keepdim=True)[0]
                normalized_full_pred_q_values = full_pred_q_values - min_q_values

                if i == 0: #i=0 means the first batch                   
                    print(normalized_true_Qs)
                    print(normalized_full_pred_q_values)
                
                
                Q_MSE_loss = MSE_loss_fn(normalized_true_Qs, normalized_full_pred_q_values)
                epoch_Q_MSE_loss += Q_MSE_loss.item()

            if epoch_Q_MSE_loss < best_Q_MSE_loss:
                best_Q_MSE_loss = epoch_Q_MSE_loss
                best_normalized_true_Qs = normalized_true_Qs
                best_normalized_full_pred_q_values = normalized_full_pred_q_values
                best_epoch = epoch + 1    

        test_loss.append(epoch_CrossEntropy_loss / len(test_dataset))
        test_full_loss.append(epoch_full_CrossEntropy_loss / len(test_dataset))
        test_Q_MSE_loss.append(epoch_Q_MSE_loss / len(test_dataset))
        
        end_time = time.time()
        printw(f"\tCross entropy test loss: {test_loss[-1]}")
        printw(f"\tMSE of Q-value: {test_Q_MSE_loss[-1]}")
        printw(f"\tEval time: {end_time - start_time}")


        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            #GPT2 implements "causal masking", i.e., output at period k can only depend on inputs at period 0, ..., k
            pred_q_values = model(batch) #dimension is (batch_size, horizon, action_dim)
            
            true_actions = batch['query_actions'].long() #dimension is (batch_size)
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_q_values.shape[1]) #dimension is (batch_size, horizon)
            true_actions_reshaped = true_actions.reshape(-1)  #dimension is (batch_size*horizon,)
            
            pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
            #pred_actions_reshaped = torch.softmax(pred_q_values_reshaped, dim=1) 
            
            optimizer.zero_grad() #clear previous gradients
            
            #The computed loss is a kind of regret of cross entropy loss until horizon 
            loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped)
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon

        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")


        # LOGGING
        if (epoch + 1) % 10000 == 0:
            torch.save(model.state_dict(),
                       f'models/{fig_filename}_epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 5 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Q value MSE Loss:        {test_Q_MSE_loss[-1]}")
            printw(f"Test action cross entropy Loss:        {test_loss[-1]}")
            printw(f"Train action cross entropy Loss:       {train_loss[-1]}")
            printw("\n")

            plt.figure()
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Cross-Entropy Loss')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{fig_filename}_CrossEntropy_loss.png")
            plt.clf()
            
            plt.figure()
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Q MSE Loss')
            plt.plot(test_Q_MSE_loss[1:], label="Test Q MSE loss")
            plt.legend()
            plt.savefig(f"figs/loss/{fig_filename}_Q_MSE_loss.png")
            plt.clf()
            

    torch.save(model.state_dict(), f'models/{filename}.pt')
    
    printw(f"Best epoch: {best_epoch}")
    printw(f"Best Q MSE loss: {best_Q_MSE_loss}")
    printw(f"Normalized true Qs: {best_normalized_true_Qs}")
    printw(f"Normalized full predicted Q values: {best_normalized_full_pred_q_values}")
    
    print("Done.")
