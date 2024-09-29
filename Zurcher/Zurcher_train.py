import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import torch.nn as nn
import json
import sys
from mlp import MLP, QtoVMLP
from datetime import datetime



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
        
        states_total = [] 
        actions_total = []
        next_states_total = []
        states_true_EPs_total = []
        next_states_true_EPs_total = []
        states_true_Qs_total = []
        next_states_true_Qs_total = []
        busTypes = []

        for traj in self.trajs:
            states_total.append(traj['states']) 
            actions_total.append(traj['actions'])
            next_states_total.append(traj['next_states'])
            
            states_true_EPs_total.append(traj['states_true_EPs'])
            states_true_Qs_total.append(traj['states_true_Qs'])
            next_states_true_EPs_total.append(traj['next_states_true_EPs'])
            next_states_true_Qs_total.append(traj['next_states_true_Qs'])

            busTypes.append(traj['busType'])
            
        states_total = np.array(states_total) #dimension of states_total is (num_trajs, H, state_dim)
                    #when a batch is called, the dimension of the batch is (batch_size, H, state_dim)
        actions_total = np.array(actions_total)
        next_states_total = np.array(next_states_total)

        states_true_EPs_total = np.array(states_true_EPs_total)
        states_true_Qs_total = np.array(states_true_Qs_total)
        next_states_true_EPs_total = np.array(next_states_true_EPs_total)
        next_states_true_Qs_total = np.array(next_states_true_Qs_total)
        

        busTypes = np.array(busTypes)

        self.dataset = {
            'states': Dataset.convert_to_tensor(states_total, store_gpu=self.store_gpu),
            'actions': Dataset.convert_to_tensor(actions_total, store_gpu=self.store_gpu),
            'next_states': Dataset.convert_to_tensor(next_states_total, store_gpu=self.store_gpu),
            'states_true_EPs': Dataset.convert_to_tensor(states_true_EPs_total, store_gpu=self.store_gpu),
            'states_true_Qs': Dataset.convert_to_tensor(states_true_Qs_total, store_gpu=self.store_gpu),
            'next_states_true_EPs': Dataset.convert_to_tensor(next_states_true_EPs_total, store_gpu=self.store_gpu),
            'next_states_true_Qs': Dataset.convert_to_tensor(next_states_true_Qs_total, store_gpu=self.store_gpu),
            'busTypes': Dataset.convert_to_tensor(busTypes, store_gpu=self.store_gpu)
        }

    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        #'Generates one sample of data'. DataLoader constructs a batch using this.
        res = {
            'states': self.dataset['states'][idx],
            'actions': self.dataset['actions'][idx],
            'next_states': self.dataset['next_states'][idx],
            'states_true_EPs': self.dataset['states_true_EPs'][idx],
            'states_true_Qs': self.dataset['states_true_Qs'][idx],
            'next_states_true_EPs': self.dataset['next_states_true_EPs'][idx],
            'next_states_true_Qs': self.dataset['next_states_true_Qs'][idx],
            'busType': self.dataset['busTypes'][idx]
        }
        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['states'] = res['states'][perm]
            res['actions'] = res['actions'][perm]
            res['next_states'] = res['next_states'][perm]
        
        return res


    def convert_to_tensor(x, store_gpu=True):
        if store_gpu:
            return torch.tensor(np.asarray(x)).float().to(device)
        else:
            return torch.tensor(np.asarray(x)).float()


def loss_ratio(x, start_value, end_value, transition_point=100):
    if x < 1:
        return start_value  # For x < 1, return the start value
    elif 1 <= x <= transition_point:
        # Linear interpolation between (1, start_value) and (transition_point, end_value)
        slope = (end_value - start_value) / (transition_point - 1)
        return start_value + slope * (x - 1)
    else:  # x > transition_point
        return end_value

def build_data_filename(config, mode):
    """
    Builds the filename for the data.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_dummies{config['num_dummies']}x{config['dummy_dim']}"
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
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_dummies{config['num_dummies']}x{config['dummy_dim']}"
                f"_beta{config['beta']}_theta{config['theta']}"
                f"_numTypes{config['numTypes']}_H{config['H']}_{config['rollin_type']}")
    filename += f'_{timestamp}'
    
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
        'env': config['env'],
        'num_dummies': config['num_dummies'],
        'dummy_dim': config['dummy_dim']
    }

    path_train = build_data_filename(dataset_config, mode='train')
    path_test = build_data_filename(dataset_config, mode='test')

    train_dataset = Dataset(path_train, dataset_config)
    test_dataset = Dataset(path_test, dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    
    states_dim = config['num_dummies']+1 #+1 is for the mileage
    actions_dim = 2
    # Prepare model
    model_config = {
        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], #layer normalization
    }
    model = MLP(states_dim, actions_dim, **model_config).to(device)
    QtoVmodel = QtoVMLP(actions_dim, **model_config).to(device)

    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
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
    best_Q_MSE_loss = 9999
    best_normalized_true_Qs = torch.tensor([])
    best_normalized_pred_q_values = torch.tensor([])

    alpha = 0.05  # Smoothing factor for moving average
    mu_ce_loss, var_ce_loss = 0.0, 0.5
    mu_be_loss, var_be_loss = 0.0, 0.5

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
                batch = {k: v.to(device) for k, v in batch.items()} #dimension is (batch_size, horizon, state_dim)

                pred_q_values, pred_q_values_next = model(batch) #dimension is (batch_size, horizon, action_dim)
                pred_vnext_values = QtoVmodel(pred_q_values)
                
                true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
                true_actions_reshaped = true_actions.reshape(-1) #dimension is (batch_size*horizon,)
                pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
                
                #prediction depending on the whole context, not partial context
                
                ####### Action CrossEntropy loss                
                cross_entropy_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped)
                epoch_CrossEntropy_loss += cross_entropy_loss.item()/config['H']
                
                ####### Q value MSE loss
                #Normalized Q values
                true_Qs_batch = batch['states_true_Qs'] #dimension is (batch_size, horizon, action_dim)
                last_true_Qs = true_Qs_batch[:,-1,:] #Just consider the last horizon state's Q values. dimensioni is (batch_size, action_dim)
                min_true_Qs = torch.min(last_true_Qs, dim=1, keepdim=True)[0]
                normalized_true_Qs = last_true_Qs - min_true_Qs
                
                last_pred_q_values = pred_q_values[:,-1,:] #dimension is (batch_size, action_dim)
                min_q_values = torch.min(last_pred_q_values, dim=1, keepdim=True)[0]
                normalized_last_pred_q_values = last_pred_q_values - min_q_values

                if i == 0: #i=0 means the first batch                   
                    #print(normalized_true_Qs)
                    printw(f"True Q values: {last_true_Qs}", config)
                    #print(normalized_last_pred_q_values)
                    printw(f"Predicted Q values: {last_pred_q_values}", config)
                
                
                Q_MSE_loss = MSE_loss_fn(normalized_true_Qs, normalized_last_pred_q_values)
                epoch_Q_MSE_loss += Q_MSE_loss.item()

            if epoch_Q_MSE_loss < best_Q_MSE_loss:
                best_Q_MSE_loss = epoch_Q_MSE_loss
                best_normalized_true_Qs = normalized_true_Qs
                best_normalized_pred_q_values = normalized_last_pred_q_values
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
        
        torch.autograd.set_detect_anomaly(True)
        
        
        for i, batch in enumerate(train_loader): #For batch i in the training dataset
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            pred_q_values, pred_q_values_next = model(batch) #dimension is (batch_size, horizon, action_dim)
            
           
            pred_vnext_values = QtoVmodel(pred_q_values.clone().detach()) 
            
            true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
            batch_size = true_actions.shape[0]
            #in torch, .long() converts the tensor to int64. CrossEntropyLoss requires the target to be int64.
            
            #count number of batches that satisfies true_actions == 1
            count_nonzero = torch.count_nonzero(true_actions == 1)
            count_nonzero_pos = torch.max(count_nonzero, torch.tensor(1))

            true_actions_reshaped = true_actions.reshape(-1)  #dimension is (batch_size*horizon,)
            pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
            pred_vnext_values_reshaped = pred_vnext_values.reshape(-1, pred_vnext_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
           
            ### Q(s,a) 
            chosen_q_values_reshaped = pred_q_values_reshaped[
                torch.arange(pred_q_values_reshaped.size(0)), true_actions_reshaped
            ]
            #E[V(s'|s,a)]
            chosen_vnext_values_reshaped = pred_vnext_values_reshaped[
                torch.arange(pred_vnext_values_reshaped.size(0)), true_actions_reshaped
            ]
            #dimension of chosen_q_values_reshaped is (batch_size*horizon,)

            #V(s') = logsumexp Q(s',a') + gamma
            pred_q_values_nextstate_reshaped = pred_q_values_next.reshape(-1, pred_q_values_next.shape[-1]) #dimension is (batch_size*horizon, action_dim)
            logsumexp_nextstate = torch.logsumexp(pred_q_values_nextstate_reshaped, dim=1) #dimension is (batch_size*horizon,)
            vnext_reshaped = np.euler_gamma + logsumexp_nextstate
            
            types = batch['busType'] #dimension is (batch_size)
            theta = config['theta']
            pivot_rewards = (-1)*(theta[2]*types+theta[1]) #dimension is (batch_size,)
            
            pivot_rewards = pivot_rewards.unsqueeze(1).repeat(1, pred_q_values_next.shape[1]) #dimension is (batch_size, horizon)
            pivot_rewards_reshaped = pivot_rewards.reshape(-1) #dimension is (batch_size*horizon,)
            
            
            if i %2 == 0: # update model paramters only
                
                #V(s')-E[V(s')] minimization loss
                D = MSE_loss_fn(vnext_reshaped.clone().detach(), chosen_vnext_values_reshaped)
                D.backward()
                vnext_optimizer.step() #we use separate optimizer for vnext
                vnext_optimizer.zero_grad() #clear gradients for the batch
                
                QtoVmodel.zero_grad() #clear gradients for the batch. This prevents the accumulation of gradients.
        
            else:     # QtoVmodel parameters only
                ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped) #shape  is (batch_size*horizon,)
                #printw(f"Cross entropy loss: {ce_loss.item()}", config)
                #td error for batch size*horizon
                
                #Non-pivot actions will be removed anyways, so I just add pivot_rewards for all cases here
                td_error = chosen_q_values_reshaped - pivot_rewards_reshaped - config['beta'] * vnext_reshaped
                #V(s')-E[V(s')|s,a]
                vnext_dev = (vnext_reshaped - chosen_vnext_values_reshaped.clone().detach())
                #Bi-conjugate trick to compute the Bellman error
                be_error_naive = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                #Exclude the action 0 from computing the Bellman error. Only count action 1's Bellman error for the loss
                be_error_0 = torch.where(true_actions_reshaped == 0, 0, be_error_naive)
                #be_loss is normalized by the number of nonzero true-action batch numbers
                be_loss = MAE_loss_fn(be_error_0, torch.zeros_like(be_error_0))/count_nonzero_pos *batch_size*config['H']
                
                #initialize mu losses and ce losses when i==0
                if i == 0:
                    mu_ce_loss = ce_loss.item()
                    #var_ce_loss = 0.5
                    mu_be_loss = be_loss.item()
                    #var_be_loss = 0.5
                
                ### Update the running means and variances
                mu_ce_loss = alpha * ce_loss.item() + (1 - alpha) * mu_ce_loss
                var_ce_loss = alpha * (ce_loss.item() - mu_ce_loss) ** 2 + (1 - alpha) * var_ce_loss
                
                mu_be_loss = alpha * be_loss.item() + (1 - alpha) * mu_be_loss
                var_be_loss = alpha * (be_loss.item() - mu_be_loss) ** 2 + (1 - alpha) * var_be_loss
                
                ### Compute dynamic lambda (loss_ratio) based on variance
                #lambda_dynamic = (var_ce_loss ** 0.5) / (var_be_loss ** 0.5) #S2, 
                #lambda_dynamic = (var_be_loss ** 0.5) / (var_ce_loss ** 0.5) #S1, Only when gradient of CE is relatively smaller than gradient of BE, we increase the ratio
                lambda_dynamic = mu_ce_loss / mu_be_loss #S4
                #loss = ce_loss + loss_ratio(epoch, 0, config['loss_ratio'], 5000) *be_loss #S3
                #loss = ce_loss + *loss_ratio(epoch, 0, lambda_dynamic, 2000) *be_loss
                loss = ce_loss + config['loss_ratio']*lambda_dynamic * be_loss
                loss.backward()
                q_optimizer.step()
                q_optimizer.zero_grad() #clear gradients for the batch
                model.zero_grad()
                
                epoch_train_loss += loss.item() / config['H']
                epoch_train_be_loss += be_loss.item() / config['H']
                epoch_train_ce_loss += ce_loss.item() / config['H']

            
            if i == 0: #i=0 means the first batch
                full_pred_r_values = pred_q_values[:,-1,:] - config['beta']*pred_vnext_values[:,-1,:]
                printw(f"Predicted r values: {full_pred_r_values}", config)
            
            

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

        if (epoch + 1) % 1 == 0:
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
        printw(f"Sample of normalized full predicted Q values: {best_normalized_pred_q_values[:10]}", config)
    else:
        printw("No best Q values were recorded during training.", config)
    
    printw("Done.", config)

