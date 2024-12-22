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
from mlp import MLP
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
        states_true_expVs_total = []
        busTypes = []

        for traj in self.trajs:
            states_total.append(traj['states']) 
            actions_total.append(traj['actions'])
            next_states_total.append(traj['next_states'])
            
            states_true_EPs_total.append(traj['states_true_EPs'])
            states_true_Qs_total.append(traj['states_true_Qs'])
            states_true_expVs_total.append(traj['states_true_expVs'])
            
            next_states_true_EPs_total.append(traj['next_states_true_EPs'])
            next_states_true_Qs_total.append(traj['next_states_true_Qs'])
           

            busTypes.append(traj['busType'])
            
        states_total = np.array(states_total) #dimension of states_total is (num_trajs, H, state_dim)
                    #when a batch is called, the dimension of the batch is (batch_size, H, state_dim)
        actions_total = np.array(actions_total)
        next_states_total = np.array(next_states_total)

        states_true_EPs_total = np.array(states_true_EPs_total)
        states_true_Qs_total = np.array(states_true_Qs_total)
        states_true_expVs_total = np.array(states_true_expVs_total)
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
            'states_true_expVs': Dataset.convert_to_tensor(states_true_expVs_total, store_gpu=self.store_gpu),
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
            'states_true_expVs': self.dataset['states_true_expVs'][idx],
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
    
    filename = (f"RL_"
                f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_dummies{config['num_dummies']}x{config['dummy_dim']}"
                f"_beta{config['beta']}_theta{config['theta']}"
                f"_H{config['H']}"
                f"_batch{config['batch_size']}"
                )
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

    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    MSE_loss_fn = torch.nn.MSELoss(reduction='sum')
    mean_MSE_loss_fn = torch.nn.MSELoss() #default is mean
    MAE_loss_fn = torch.nn.L1Loss(reduction='sum')
    
    train_loss = []
    train_be_loss = []
    train_ce_loss = []
    train_D_loss = []
    test_Q_MSE_loss = []
    
    #Storing the best training epoch and its corresponding best Q MSE loss/Q values
    best_epoch = -1
    best_Q_MSE_loss = 9999
    best_normalized_true_Qs = torch.tensor([])
    best_normalized_pred_q_values = torch.tensor([])

    
    for epoch in tqdm(range(config['num_epochs']), desc="Training Progress"):
        
        ############### Start of an epoch ##############
        
        ### EVALUATION ###
        printw(f"Epoch: {epoch + 1}", config)
        start_time = time.time()
        
        with torch.no_grad():
            epoch_Q_MSE_loss = 0.0
            
            ##### Test batch loop #####
            
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()} #dimension is (batch_size, horizon, state_dim)
                states = batch['states']
                pred_q_values, pred_q_values_next, _ = model(batch) #dimension is (batch_size, horizon, action_dim)
                
                true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
                true_actions_reshaped = true_actions.reshape(-1) #dimension is (batch_size*horizon,)
                pred_q_values_reshaped = pred_q_values.reshape(-1, pred_q_values.shape[-1]) #dimension is (batch_size*horizon, action_dim)
                
                ####### Q value MSE loss
                
                true_Qs_batch = batch['states_true_Qs'] #dimension is (batch_size, horizon, action_dim)
                chosen_true_Qs = torch.gather(true_Qs_batch, dim=2, index=true_actions.unsqueeze(-1)).squeeze(-1) #dimension is (batch_size, horizon)
                chosen_pred_Qs = torch.gather(pred_q_values, dim=2, index=true_actions.unsqueeze(-1)).squeeze(-1) #dimension is (batch_size, horizon)
                
                #Visualize the Q values of the last horizon state
                last_true_Qs = true_Qs_batch[:,-1,:] #Just consider the last horizon state's Q values. dimensioni is (batch_size, action_dim)
                last_states = states[:,-1,0].unsqueeze(1) #dimension is (batch_size, state_dim)
                #I only want the first element of the state, which is the mileage
                last_true_Qs_with_states = torch.cat((last_states, last_true_Qs), dim=1) #dimension is (batch_size, state_dim+action_dim) 
                last_pred_q_values = pred_q_values[:,-1,:] #dimension is (batch_size, action_dim)
                last_pred_q_values_with_states = torch.cat((last_states, last_pred_q_values), dim=1) #dimension is (batch_size, state_dim+action_dim)
                
                if i == 0: #i=0 means the first batch                   
                    printw(f"True Q values: {last_true_Qs_with_states[:10]}", config)
                    printw(f"Predicted Q values: {last_pred_q_values_with_states[:10]}", config)
                
                Q_MSE_loss = mean_MSE_loss_fn(chosen_true_Qs, chosen_pred_Qs) #it gives batch mean (total sum / (batch_size*horizon))
                epoch_Q_MSE_loss += Q_MSE_loss.item()
      
            ##### Finish of the batch loop for a single epoch #####
            ##### Back to epoch level #####
            # Note that epoch MSE losses are sum of all test batch means in the epoch
             
            if epoch_Q_MSE_loss/len(test_dataset) < best_Q_MSE_loss: #epoch_Q_MSE_loss is sum of all test batch means in the epoch
                        
                best_Q_MSE_loss = epoch_Q_MSE_loss/len(test_dataset)
                best_normalized_true_Qs = last_true_Qs #Last batch's true Q values
                best_normalized_pred_q_values = last_pred_q_values #Last batch's predicted Q values
                
                best_epoch = epoch    
        
        ############# Finish of an epoch's evaluation ############
        
        #test_loss.append(epoch_CrossEntropy_loss / len(test_dataset))
        test_Q_MSE_loss.append(epoch_Q_MSE_loss / len(test_dataset)) #len(test_dataset) is the number of batches in the test dataset
        
        end_time = time.time()
        #printw(f"\tCross entropy test loss: {test_loss[-1]}", config)
        printw(f"\tMSE of normalized Q-value: {test_Q_MSE_loss[-1]}", config)
        #printw(f"\tMSE of V(s',a'): {test_vnext_MSE_loss[-1]}", config)
        printw(f"\tEval time: {end_time - start_time}", config)
        
        
        ############# Start of an epoch's training ############
        
        epoch_train_loss = 0.0
        epoch_train_be_loss = 0.0
        epoch_train_ce_loss = 0.0
        epoch_train_D_loss = 0.0
        start_time = time.time()
        
        torch.autograd.set_detect_anomaly(True)
        
        
        for i, batch in enumerate(train_loader): #For batch i in the training dataset
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            pred_q_values, pred_q_values_next, pred_vnext_values = model(batch) #dimension is (batch_size, horizon, action_dim)
            
            states = batch['states'] #dimension is (batch_size, horizon, state_dim)
            true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
            types = batch['busType'] #dimension is (batch_size,)
            types_expanded = types.unsqueeze(1).expand(-1, true_actions.shape[1])
            theta = config['theta']
            state_mileages = states[:,:,0] #dimension is (batch_size, horizon)
            
            rewards = torch.zeros_like(true_actions, dtype=torch.float) #dimension is (batch_size, horizon)
            rewards[true_actions == 0] = -1 * theta[0]*state_mileages[true_actions == 0] 
            rewards[true_actions == 1] = -1 * (theta[2] * types_expanded[true_actions == 1] + theta[1])
            rewards_reshaped = rewards.reshape(-1) #dimension is (batch_size*horizon,)
        
            #in torch, .long() converts the tensor to int64. CrossEntropyLoss requires the target to be int64.
            
            #count number of batches that satisfies true_actions == 1
            #count_nonzero = torch.count_nonzero(true_actions == 1) #dimension is (batch_size, horizon)
            #count_nonzero_pos = torch.max(count_nonzero, torch.tensor(1)) #dimension is (batch_size, horizon)

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
            #vnext_reshaped = np.euler_gamma + logsumexp_nextstate
            vnext_reshaped = logsumexp_nextstate
        
            
            if i %2 == 0: # update model paramters only
                
                #V(s')-E[V(s')] minimization loss
                D = MSE_loss_fn(vnext_reshaped.clone().detach(), chosen_vnext_values_reshaped)
                D.backward()
                vnext_optimizer.step() #we use separate optimizer for vnext
                vnext_optimizer.zero_grad() #clear gradients for the batch
                epoch_train_D_loss += D.item() / config['H'] #per-sample loss
                model.zero_grad() #clear gradients for the batch. This prevents the accumulation of gradients.
        
            else:     # QtoVmodel parameters only
                ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped) #shape  is (batch_size*horizon,)
                #printw(f"Cross entropy loss: {ce_loss.item()}", config)
                #td error for batch size*horizon
                
                #First, compute td error for (s,a) pairs that appear in the data. 
                #Non-pivot actions will be removed anyways, so I just add pivot_rewards for all cases here

                td_error = chosen_q_values_reshaped - rewards_reshaped - config['beta'] * vnext_reshaped #\delta(s,a) = Q(s,a) - r(s,a) - beta*V(s')
                #V(s')-E[V(s')|s,a]
              
                vnext_dev = (vnext_reshaped - chosen_vnext_values_reshaped.clone().detach())
                #Bi-conjugate trick to compute the Bellman error
                be_error = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                #We call it naive because we just add pivot r for every actions we see in the batch
                be_loss = MAE_loss_fn(be_error, torch.zeros_like(be_error))#/count_nonzero_pos *batch_size*config['H']
                #count_nonzero_pos is the number of nonzero true-actions in batch_size*horizon
        
                loss = ce_loss + be_loss
                loss.backward()
                q_optimizer.step()
                q_optimizer.zero_grad() #clear gradients for the batch
               
                model.zero_grad()
                
                epoch_train_loss += loss.item() / config['H']
                epoch_train_be_loss += be_loss.item() / config['H']
                epoch_train_ce_loss += ce_loss.item() / config['H']
                
                print(f"Epoch_train_loss: {epoch_train_loss}", end='\r')

    
        
            
            
        #len(train_dataset) is the number of batches in the training dataset
        train_loss.append(epoch_train_loss / len(train_dataset)) 
        train_be_loss.append(epoch_train_be_loss / len(train_dataset))
        train_ce_loss.append(epoch_train_ce_loss / len(train_dataset))
        train_D_loss.append(epoch_train_D_loss / len(train_dataset))

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
                plt.figure(figsize=(12, 12))  # Increase the height to fit all plots
    
                # Plotting total train loss
                plt.subplot(4, 1, 1) # Adjust to 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Total Train Loss')
                plt.plot(train_loss[1:], label="Total Train Loss")
                plt.legend()

                # Plotting BE loss
                plt.subplot(4, 1, 2) # Second plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('BE Loss')
                plt.plot(train_be_loss[1:], label="Bellman Error Loss", color='red')
                plt.legend()

                # Plotting CE loss
                plt.subplot(4, 1, 3) # Third plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('CE Loss')
                plt.plot(train_ce_loss[1:], label="Cross-Entropy Loss", color='blue')
                plt.legend()

                # Plotting Q MSE loss
                plt.subplot(4, 1, 4) # Fourth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Q MSE Loss')
                plt.plot(test_Q_MSE_loss[1:], label="Test Q MSE Loss", color='green')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"figs/loss/{build_log_filename(config)}_losses.png")
                plt.close()

    torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')
    
    printw(f"\nTraining completed.", config)
    printw(f"Best epoch: {best_epoch}", config)
    printw(f"Best Q MSE loss: {best_Q_MSE_loss}", config)
    
    if best_epoch > 0:
        printw(f"Sample of true Qs: {best_normalized_true_Qs[:10]}", config)
        printw(f"Sample of predicted Q values: {best_normalized_pred_q_values[:10]}", config)
    else:
        printw("No best Q values were recorded during training.", config)
    
    printw("Done.", config)

