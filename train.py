import torch
import argparse
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
import json
import utils
import sys
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class Dataset(torch.utils.data.Dataset):
    """Dataset class."""
    

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['H']
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
        query_next_states = []
        query_next_actions = []
        query_true_EPs = []
        query_next_true_EPs = []
        query_true_Qs = []
        query_next_true_Qs = []

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])

            query_states.append(traj['query_state'])
            query_actions.append(traj['query_action'])
            query_next_states.append(traj['query_next_state'])
            query_next_actions.append(traj['query_next_action'])
            query_true_EPs.append(traj['query_true_EP'])
            query_next_true_EPs.append(traj['query_next_true_EP'])
            query_true_Qs.append(traj['query_true_Q'])  
            query_next_true_Qs.append(traj['query_next_true_Q'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        
        #if len(context_rewards.shape) < 3:
        #    context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states) #its dimension is (num_samples, state_dim)
        query_actions = np.array(query_actions)
        query_next_states = np.array(query_next_states)
        query_next_actions = np.array(query_next_actions)
        
        query_true_EPs = np.array(query_true_EPs)
        query_next_true_EPs = np.array(query_next_true_EPs)
        query_true_Qs = np.array(query_true_Qs)
        query_next_true_Qs = np.array(query_next_true_Qs)

        self.dataset = {
            'query_states': Dataset.convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'query_next_states': Dataset.convert_to_tensor(query_next_states, store_gpu=self.store_gpu),
            'query_actions': Dataset.convert_to_tensor(query_actions, store_gpu=self.store_gpu),
            'query_next_actions': Dataset.convert_to_tensor(query_next_actions, store_gpu=self.store_gpu),
            'context_states': Dataset.convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': Dataset.convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': Dataset.convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'query_true_EPs': Dataset.convert_to_tensor(query_true_EPs, store_gpu=self.store_gpu),
            'query_next_true_EPs': Dataset.convert_to_tensor(query_next_true_EPs, store_gpu=self.store_gpu),
            'query_true_Qs': Dataset.convert_to_tensor(query_true_Qs, store_gpu=self.store_gpu),
            'query_next_true_Qs': Dataset.convert_to_tensor(query_next_true_Qs, store_gpu=self.store_gpu)
        }
        
        self.zeros = np.zeros(
            config['maxMileage'] + 1
        )
        self.zeros = Dataset.convert_to_tensor(self.zeros, store_gpu=self.store_gpu)


    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        #'Generates one sample of data'. DataLoader constructs a batch using this.
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'query_states': self.dataset['query_states'][index],
            'query_next_states': self.dataset['query_next_states'][index],
            'query_actions': self.dataset['query_actions'][index],
            'query_next_actions': self.dataset['query_next_actions'][index],
            'query_true_EPs': self.dataset['query_true_EPs'][index],
            'query_next_true_EPs': self.dataset['query_next_true_EPs'][index],
            'query_true_Qs': self.dataset['query_true_Qs'][index],
            'query_next_true_Qs': self.dataset['query_next_true_Qs'][index],
            'zeros': self.zeros
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]

        return res
    def convert_to_tensor(x, store_gpu=True):
        if store_gpu:
            return torch.tensor(np.asarray(x)).float().to(device)
        else:
            return torch.tensor(np.asarray(x)).float()


class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.H = self.config['H']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = 1
        self.action_dim = 1
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.H),
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
        self.pred_r_values = nn.Linear(self.n_embd, 2)

    def forward(self, x):
        query_states = x['query_states'][:, None] # (batch_size, 1, state_dim). #None and unsqueeze are equivalent
        query_next_states = x['query_next_states'][:, None] # (batch_size, 1, state_dim)
        
        zeros = x['zeros'][:, None] # (batch_size, 1, state_dim+1)
        state_seq = torch.cat([query_states, x['context_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        state_seq_with_next = torch.cat([query_next_states, x['context_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        action_seq = torch.cat(
            [zeros[:, :, 1], x['context_actions']], dim=1) # (batch_size, 1+horizon, action_dim)
        next_state_seq = torch.cat(
            [zeros[:, :, 1], x['context_next_states']], dim=1) # (batch_size, 1+horizon, state_dim)
        
        
        state_seq = state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
        state_seq_with_next = state_seq_with_next.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
        action_seq = action_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, action_dim)
        next_state_seq = next_state_seq.unsqueeze(2) # (batch_size, 1+horizon, 1, state_dim)
       

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq], dim=2) 
        seq_next = torch.cat(
            [state_seq_with_next, action_seq, next_state_seq], dim=2)
        
        stacked_inputs = self.embed_transition(seq)
        stacked_next_inputs = self.embed_transition(seq_next)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        transformer_next_outputs = self.transformer(inputs_embeds=stacked_next_inputs)

        preds = self.pred_q_values(transformer_outputs['last_hidden_state'])
        preds_r = self.pred_r_values(transformer_outputs['last_hidden_state'])
        
        preds_next = self.pred_q_values(transformer_next_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :], preds_next[:, 1:, :], preds_r[:, 1:, :] #The first horizon input is spared for the query state




def build_data_filename(config, mode):
    """
    Builds the filename for the data.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_beta{config['beta']}_theta{config['theta']}"
                f"_numTypes{config['numTypes']}_H{config['H']}_{config['rollin_type']}")
    
    filename += f'_{mode}'
    
    return filename_template.format(filename)


def build_model_filename(config):
    """
    Builds the filename for the model.
    """
    filename = (f"{config['env']}_shuf{config['shuffle']}_lr{config['lr']}"
                f"_do{config['dropout']}_embd{config['n_embd']}"
                f"_layer{config['n_layer']}_head{config['n_head']}"
                f"_H{config['H']}_seed{config['seed']}")
    return filename

def build_log_filename(config):
    """
    Builds the filename for the log file.
    """
    filename = (f"zurcher_num_trajs{config['num_trajs']}"
                f"_beta{config['beta']}_theta{config['theta']}"
                f"_numTypes{config['numTypes']}_H{config['H']}_{config['rollin_type']}"
                f"_loss_ratio{config['loss_ratio']}"
                f"_infR{config['infR']}")
    return filename + ".log"

def printw(message, config):
    print(message)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = build_log_filename(config)
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "a") as log_file:
        print(message, file=log_file)

# Make sure to create the logs directory


def train(config):
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    if not os.path.exists('logs'):
        os.makedirs('logs', exist_ok=True)

    # Set random seeds
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Prepare dataset
    dataset_config = {
        'H': config['H'],
        'num_trajs': config['num_trajs'],
        'maxMileage': config['maxMileage'],
        'theta': config['theta'],
        'beta': config['beta'],
        'numTypes': config['numTypes'],
        'rollin_type': 'expert',
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env']
    }

    path_train = build_data_filename(dataset_config, mode='train')
    path_test = build_data_filename(dataset_config, mode='test')

    train_dataset = Dataset(path_train, dataset_config)
    test_dataset = Dataset(path_test, dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])

    # Prepare model
    model_config = {
        'lr': config['lr'],
        'n_layer': config['n_layer'],
        'n_embd': config['n_embd'],
        'n_head': config['n_head'],
        'shuffle': config['shuffle'],
        'seed': config['seed'],
        'dropout': config['dropout'],
        'test': False,
        'store_gpu': True,
        'H': config['H'],
        'loss_ratio': config['loss_ratio'],
        'infR': config['infR']
    }
    model = Transformer(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    MSE_loss_fn = torch.nn.MSELoss(reduction='sum')
    MAE_loss_fn = torch.nn.L1Loss(reduction='sum')

    # Training loop
    train_loss = []
    train_be_loss = []
    train_ce_loss = []
    test_loss = []
    test_full_loss = []
    test_Q_MSE_loss = []
    
    #Storing the best training epoch and its corresponding best Q MSE loss/Q values
    best_epoch = -1
    best_Q_MSE_loss = float('inf')
    best_normalized_true_Qs = torch.tensor([])
    best_normalized_full_pred_q_values = torch.tensor([])

    
    for epoch in tqdm(range(config['num_epochs']), desc="Training Progress"):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}", config)
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
                pred_q_values, pred_q_values_next, pred_r_values = model(batch) #dimension is (batch_size, horizon, action_dim)
                
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
                epoch_CrossEntropy_loss += cross_entropy_loss.item()/config['H']
                
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
                    #print(normalized_true_Qs)
                    printw(f"True Q values: {query_true_Qs}", config)
                    #print(normalized_full_pred_q_values)
                    printw(f"Predicted Q values: {full_pred_q_values}", config)
                
                
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
        printw(f"\tCross entropy test loss: {test_loss[-1]}", config)
        printw(f"\tMSE of Q-value: {test_Q_MSE_loss[-1]}", config)
        printw(f"\tEval time: {end_time - start_time}", config)
        
        # Training
        epoch_train_loss = 0.0
        epoch_train_be_loss = 0.0
        epoch_train_ce_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}

            ##(s,a,s',a') pair printout for sanity check
            
            
            #GPT2 implements "causal masking", i.e., output at period k can only depend on inputs at period 0, ..., k
            pred_q_values, pred_q_values_next, pred_r_values = model(batch) #dimension is (batch_size, horizon, action_dim)
            
            true_actions = batch['query_actions'].long() #dimension is (batch_size)
            #in torch, .long() converts the tensor to int64. CrossEntropyLoss requires the target to be int64.

            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_q_values.shape[1]) #dimension is (batch_size, horizon)
            true_actions_reshaped = true_actions.reshape(-1)  #dimension is (batch_size*horizon,)
            pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
            pred_r_values_reshaped = pred_r_values.reshape(-1, pred_r_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)

            ### Q(s,a) and r(s,a) for each batch
            chosen_q_values_reshaped = pred_q_values_reshaped[
                torch.arange(pred_q_values_reshaped.size(0)), true_actions_reshaped
            ]
            chosen_r_values_reshaped = pred_r_values_reshaped[
                torch.arange(pred_r_values_reshaped.size(0)), true_actions_reshaped
            ]

            

            ### Q(s',a') and p(s',a') for each batch
            pred_q_values_nextstate_reshaped = pred_q_values_next.reshape(-1, pred_q_values_next.shape[-1]) #dimension is (batch_size*horizon, action_dim)
            prob_nextstate = F.softmax(pred_q_values_nextstate_reshaped, dim=1) #dimension is (batch_size*horizon, action_dim)
            ## get a' 
            query_next_actions = batch['query_next_actions'].long() #dimension is (batch_size)
            query_next_actions = query_next_actions.unsqueeze(
                1).repeat(1, pred_q_values_next.shape[1]) #dimension is (batch_size, horizon)
            query_next_actions_reshaped = query_next_actions.reshape(-1) #dimension is (batch_size*horizon,)
            #Q(s',a') 
            chosen_q_values_nextstate_reshaped = pred_q_values_nextstate_reshaped[
                torch.arange(pred_q_values_nextstate_reshaped.size(0)), query_next_actions_reshaped
            ] #dimension is (batch_size*horizon,)
            #p(s',a')
            chosen_prob_nextstate_reshaped = prob_nextstate[
                torch.arange(prob_nextstate.size(0)), query_next_actions_reshaped
            ] #dimension is (batch_size*horizon,)
    
            #CrossEntropy loss of p(s,a)
            ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped) #The computed loss is a kind of regret of cross entropy loss until horizon 

            if config['infR']: # Direct inference of r function
                #computation of bellman error of the batch (Q(s,a)-r(s,a)-beta*(Q(s',a')-log(p(s',a'))))
                bellman_loss = MAE_loss_fn(
                    chosen_q_values_reshaped, chosen_r_values_reshaped + config['beta'] * (chosen_q_values_nextstate_reshaped + np.euler_gamma-  torch.log(chosen_prob_nextstate_reshaped))
                )
                #boundary condition loss (r(s,0)=0)+
                boundary_loss = MAE_loss_fn(pred_r_values_reshaped[:, 1], (-5)*torch.ones_like(pred_r_values_reshaped[:, 1]))
                loss = ce_loss + config['loss_ratio']*(bellman_loss + boundary_loss)
            else: # BE is applied only for regularization
                #Bellman error for batch size*horizon
                bellman_error = chosen_q_values_reshaped + 5 - config['beta'] * (chosen_q_values_nextstate_reshaped + np.euler_gamma - torch.log(chosen_prob_nextstate_reshaped))
                #Exclude the action 0 from computing the Bellman error. Only count action 1's Bellman error for the loss
                bellman_error_0 = torch.where(true_actions_reshaped == 0, 0, bellman_error)
                bellman_loss = MAE_loss_fn(bellman_error_0, torch.zeros_like(bellman_error_0))
                loss = ce_loss + config['loss_ratio']*bellman_loss
            
            

            
            optimizer.zero_grad() #clear previous gradients
            
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / config['H']
            epoch_train_be_loss += bellman_loss.item() / config['H']
            epoch_train_ce_loss += ce_loss.item() / config['H']

        train_loss.append(epoch_train_loss / len(train_dataset))
        train_be_loss.append(epoch_train_be_loss / len(train_dataset))
        train_ce_loss.append(epoch_train_ce_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}", config)
        printw(f"\tBE loss: {train_be_loss[-1]}", config)
        printw(f"\tCE loss: {train_ce_loss[-1]}", config)

        printw(f"\tTrain time: {end_time - start_time}", config)


        # Logging and plotting
        
        if (epoch + 1) % 10000 == 0:
            torch.save(model.state_dict(),
                       f'models/{build_log_filename(config)}_epoch{epoch+1}.pt')

        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1) #subplot in a 2x1 grid, and selects the first (top) subplot
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Loss')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(train_be_loss[1:], label="Bellman Error Loss")
            plt.plot(train_ce_loss[1:], label="Cross-Entropy Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.plot(test_full_loss[1:], label="Full Context Test Loss")
            plt.legend()
            
            plt.subplot(2, 1, 2) #subplot in a 2x1 grid, and selects the second (bottom) subplot
            plt.yscale('log')
            plt.xlabel('epoch')
            plt.ylabel('Q MSE Loss')
            plt.plot(test_Q_MSE_loss[1:], label="Test Q MSE loss")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"figs/loss/{build_log_filename(config)}_losses.png")
            plt.close()

    torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')
    
    printw(f"\nTraining completed.", config)
    printw(f"Best epoch: {best_epoch}", config)
    printw(f"Best Q MSE loss: {best_Q_MSE_loss}", config)
    
    if best_epoch > 0:
        printw(f"Sample of normalized true Qs: {best_normalized_true_Qs[:10]}", config)
        printw(f"Sample of normalized full predicted Q values: {best_normalized_full_pred_q_values[:10]}", config)
    else:
        printw("No best Q values were recorded during training.", config)
    
    printw("Done.", config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config)
