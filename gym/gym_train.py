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
        rewards_total = []

        for traj in self.trajs:
            states_total.append(traj['states']) 
            actions_total.append(traj['actions'])
            next_states_total.append(traj['next_states'])
            rewards_total.append(traj['rewards'])
    
        self.dataset = {
            'states': states_total,
            'actions': actions_total,
            'next_states': next_states_total,
            'rewards': rewards_total,
        }

    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        traj = {
            'states': torch.tensor(self.dataset['states'][idx]).float(),
            'actions': torch.tensor(self.dataset['actions'][idx]).long(),
            'next_states': torch.tensor(self.dataset['next_states'][idx]).float(),
            'rewards': torch.tensor(self.dataset['rewards'][idx]).float(),
        }

        if self.shuffle:
            traj_length = traj['states'].shape[0]  # Get actual trajectory length
            perm = torch.randperm(traj_length)  # Permute only within this trajectory
            traj['states'] = traj['states'][perm]
            traj['actions'] = traj['actions'][perm]
            traj['next_states'] = traj['next_states'][perm]
            traj['rewards'] = traj['rewards'][perm]

        return traj


    def convert_to_tensor(x, store_gpu=True):
        if store_gpu:
            return torch.tensor(np.asarray(x)).float().to(device)
        else:
            return torch.tensor(np.asarray(x)).float()


def collate_fn(batch):
    """
    Custom collate function to handle variable-length trajectories.
    Each batch is a list of dictionaries {'states': tensor, 'actions': tensor, ...}
    """
    batch_dict = {
        'states': [item['states'] for item in batch],
        'actions': [item['actions'] for item in batch],
        'next_states': [item['next_states'] for item in batch],
        'rewards': [item['rewards'] for item in batch],
    }
    return batch_dict


def build_data_filename(config, mode):
    """
    Builds the filename for the data.
    Mode is either 'train', 'test', or 'eval'.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}")
    
    filename += f'_{mode}'
    
    return filename_template.format(filename)


def build_model_filename(config):
    """
    Builds the filename for the model.
    """
    filename = (f"{config['env']}_shuf{config['shuffle']}_lr{config['lr']}"
                f"_decay{config['decay']}_Tik{config['Tik']}"
                f"_do{config['dropout']}_embd{config['n_embd']}"
                f"_layer{config['n_layer']}_head{config['n_head']}"
                f"_seed{config['seed']}")
    return filename

def build_log_filename(config):
    """
    Builds the filename for the log file.
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = (f"{config['env']}_num_trajs{config['num_trajs']}"
                f"_lr{config['lr']}"
                f"_batch{config['batch_size']}"
                f"_Tik{config['Tik']}"
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
        'num_trajs': config['num_trajs'],
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env']
    }

    path_train = build_data_filename(dataset_config, mode='train')
    path_test = build_data_filename(dataset_config, mode='test')

    train_dataset = Dataset(path_train, dataset_config)
    test_dataset = Dataset(path_test, dataset_config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'], collate_fn=collate_fn
    )

    
    states_dim = 8
    actions_dim = 4
    hidden_sizes = [64,64]
    # Prepare model
    model_config = {



        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], #layer normalization
    }
    model = MLP(states_dim, actions_dim, **model_config).to(device)

    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    MSE_loss_fn = torch.nn.MSELoss(reduction='mean')
    MAE_loss_fn = torch.nn.L1Loss(reduction='sum')
    
    repetitions = config['repetitions']  # Number of repetitions

    rep_test_Q_MSE_loss = []
    rep_test_r_MAPE_loss = []
    rep_best_r_MAPE_loss = []
    rep_best_Q_MSE_loss = []    

    
    for rep in range(repetitions):
        print(f"\nStarting repetition {rep+1}/{repetitions}")
        train_loss = []
        train_be_loss = []
        train_ce_loss = []
        train_D_loss = []
        test_Q_MSE_loss = []
        test_r_MAPE_loss = []
        test_vnext_MSE_loss = []
        
        #Storing the best training epoch and its corresponding best Q MSE loss/Q values
        best_epoch = -1
        best_Q_MSE_loss = 9999
        best_r_MAPE_loss = 9999
        best_normalized_true_Qs = torch.tensor([])
        best_normalized_pred_q_values = torch.tensor([])
      
        
        for epoch in tqdm(range(config['num_epochs']), desc="Training Progress"):
            
            ############### Start of an epoch ##############
            
            ### EVALUATION ###
            printw(f"Epoch: {epoch + 1}", config)
            start_time = time.time()
            with torch.no_grad():
                epoch_r_MAPE_loss = 0.0
                
                ##### Test batch loop #####
                
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')
                    batch = {k: [v.to(device) for v in batch[k]] for k in batch}  # Move lists of tensors to device
                    states = batch['states']
                    pred_q_values, pred_q_values_next, pred_vnext_values = model(batch) #dimension is (total_trans, action_dim)
                    
                    true_actions = torch.cat(batch['actions'], dim=0).long() #dimension is (total_trans)
        
                    ### Q(s,a) 
                    chosen_q_values_reshaped = pred_q_values[
                    torch.arange(pred_q_values.size(0)), true_actions
                    ]

                    #E[V(s'|s,a)]
                    chosen_vnext_values_reshaped = pred_vnext_values[
                        torch.arange(pred_vnext_values.size(0)), true_actions
                    ]
                    
                    
                    #V(s') = logsumexp Q(s',a') + gamma

                    logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1) 
                    #vnext_reshaped = np.euler_gamma + logsumexp_nextstate
                    vnext_reshaped = logsumexp_nextstate
                              
                
                    pred_r_values = pred_q_values - config['beta']*pred_vnext_values 
          
                    
                    chosen_pred_r_values = torch.gather(pred_r_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)

                    true_r_values = torch.cat(batch['rewards'], dim=0) #dimension
              
                    
                    #For computing r_MAPE (mean absolute percentage error)
                    diff = torch.abs(chosen_pred_r_values- true_r_values) #dimension is (batch_size, horizon)
                    denom = torch.abs(true_r_values) #dimension is (batch_size, horizon)
                   
                    r_MAPE = torch.mean(diff / denom)*100 #dimension is (1,), because it is the mean of all diff/denom values in the batch
                    epoch_r_MAPE_loss += r_MAPE.item()
                    
                    
                
                ##### Finish of the batch loop for a single epoch #####
                ##### Back to epoch level #####
                # Note that epoch MSE losses are sum of all test batch means in the epoch
                
                if epoch_r_MAPE_loss/len(test_loader) < best_r_MAPE_loss: #epoch_r_MAPE_loss is sum of all test batch means in the epoch
        
                    best_r_MAPE_loss = epoch_r_MAPE_loss/len(test_loader) #len(test_dataset) is the number of batches in the test dataset
                    best_epoch = epoch          
            
            ############# Finish of an epoch's evaluation ############
    
            test_r_MAPE_loss.append(epoch_r_MAPE_loss / len(test_loader)) #mean of all test batch means in the epoch
            
            end_time = time.time()
            #printw(f"\tCross entropy test loss:
            
            printw(f"\tMAPE of r(s,a): {test_r_MAPE_loss[-1]}", config)

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
                batch = {k: [v.to(device) for v in batch[k]] for k in batch}  # Move lists of tensors to device
                
                pred_q_values, pred_q_values_next, pred_vnext_values = model(batch) 
                true_actions = torch.cat(batch['actions'], dim=0).long() 
              
                
                true_rewards = torch.cat(batch['rewards'], dim=0)
                
            
                ### Q(s,a) 
                chosen_q_values = torch.gather(pred_q_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)
                chosen_vnext_values = torch.gather(pred_vnext_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)
                
                #V(s') = logsumexp Q(s',a') + gamma
                logsumexp_nextstate = torch.logsumexp(pred_q_values, dim=1) #dimension is (batch_size*horizon,)
                #vnext = np.euler_gamma + logsumexp_nextstate
                vnext = logsumexp_nextstate
                
                if i % 2 == 0: # update xi only, update xi every 2 batches

                    #V(s')-E[V(s')] minimization loss
                    D = MSE_loss_fn(vnext.clone().detach(), chosen_vnext_values)
                    D.backward()
                    
                    #Non-fixed lr part starts
                    current_lr_vnext = config['lr'] / (1 + config['decay']*epoch)
                    vnext_optimizer.param_groups[0]['lr'] = current_lr_vnext
                    #Non-fixed lr part ends
                    
                    vnext_optimizer.step() #we use separate optimizer for vnext
                    vnext_optimizer.zero_grad() #clear gradients for the batch
                    epoch_train_D_loss += D.item()  #per-sample loss
                    model.zero_grad() #clear gradients for the batch. This prevents the accumulation of gradients.
            
                else:  # update Q only, update Q every 2 batches
                    #ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped) #shape  is (batch_size*horizon,)
                    Mean_CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                    ce_loss = Mean_CrossEntropy_loss_fn(pred_q_values, true_actions) #shape  is (batch_size*horizon,)
                    #printw(f"Cross entropy loss: {ce_loss.item()}", config)
                    #td error for batch size*horizon
                    
                    #First, compute td error for (s,a) pairs that appear in the data. 
                    #Non-pivot actions will be removed anyways, so I just add pivot_rewards for all cases here
                    #Pivot action here is doing nothing (action 0)
                    #Setting pivot reward does not affect anything. However, for convenience, 
                    #to give the true reward for the pivot action so that we can compare the reward learning precisely. 
            
                    pivot_rewards = true_rewards
                        
                    td_error = chosen_q_values - pivot_rewards- config['beta'] * vnext #\delta(s,a) = Q(s,a) - r(s,a) - beta*V(s')
                    #V(s')-E[V(s')|s,a]
                 
                    
                    vnext_dev = (vnext - chosen_vnext_values.clone().detach())
                    #Bi-conjugate trick to compute the Bellman error
                    be_error_naive = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                    #We call it naive because we just add pivot r for every actions we see in the batch
                    
                    #Exclude the action 0 from computing the Bellman error, leaving pivot cases only.
                    be_error_0 = torch.where(true_actions != 2, 0, be_error_naive) #only consider the Bellman error for action 0
                    #be_loss is normalized by the number of nonzero true-action batch numbers
                    
                    mean_MAE_loss_fn = torch.nn.L1Loss(reduction='mean')
             
                    be_loss = mean_MAE_loss_fn(be_error_0, torch.zeros_like(be_error_0))
                    #count_nonzero_pos is the number of nonzero true-actions in batch_size*horizon
                    
                    #Tikhonov
                    if config['Tik'] == True:
                        loss = 100*(1/(1+epoch))*ce_loss + be_loss
                    else:               
                        loss = ce_loss + be_loss
                    #
                    
                    loss.backward()
                    
                    #Non-fixed lr part starts
                    current_lr_q = config['lr'] / (1 + config['decay']*epoch)
                    q_optimizer.param_groups[0]['lr'] = current_lr_q
                    #Non-fixed lr part ends
                    
                    if config['clip'] != False:                    
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip'])
                    q_optimizer.step()
                    q_optimizer.zero_grad() #clear gradients for the batch
                
                    model.zero_grad()
                    
                    epoch_train_loss += loss.item() 
                    epoch_train_be_loss += be_loss.item() 
                    epoch_train_ce_loss += ce_loss.item() 
                    
                    print(f"Epoch_train_loss: {epoch_train_loss}", end='\r')

                
                if i == 0: #i=0 means the first batch
                    pred_r_values_print = pred_q_values[:10,:] - config['beta']*pred_vnext_values[:10,:] #for print
                    chosen_r_values_print = torch.gather(pred_r_values_print, dim=1, index=true_actions[:10].unsqueeze(-1)) #for print
                    true_r_values_print = true_rewards[:10].unsqueeze(1) #for print
                    #states = batch['states']
                    #states = torch.cat(states, dim=0) #dimension is (batch_size*horizon, state_dim)
                    #last_states = states[:10,0].unsqueeze(1)#dimension is (batch_size, state_dim)
                    pred_r_values_with_true_r = torch.cat((true_r_values_print, chosen_r_values_print), dim=1) #dimension is (batch_size, state_dim+action_dim)
                    printw(f"Predicted r values: {pred_r_values_with_true_r[:10]}", config)
                
       
                
                
            #len(train_dataset) is the number of batches in the training dataset
            train_loss.append(epoch_train_loss / len(train_loader)) 
            train_be_loss.append(epoch_train_be_loss / len(train_loader))
            train_ce_loss.append(epoch_train_ce_loss / len(train_loader))
            train_D_loss.append(epoch_train_D_loss / len(train_loader))

            end_time = time.time()
            
            printw(f"\tTrain loss: {train_loss[-1]}", config)
            printw(f"\tBE loss: {train_be_loss[-1]}", config)
            printw(f"\tCE loss: {train_ce_loss[-1]}", config)
            printw(f"\tTrain time: {end_time - start_time}", config)


            # Logging and plotting
            
            #if (epoch + 1) % 10000 == 0:
            #    torch.save(model.state_dict(),
            #            f'models/{build_log_filename(config)}_rep{rep}_epoch{epoch+1}.pt')

            if (epoch + 1) % 1 == 0:
                plt.figure(figsize=(12, 12))  # Increase the height to fit all plots
    
                # Plotting total train loss
                plt.subplot(5, 1, 1) # Adjust to 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Total Train Loss')
                plt.plot(train_loss[1:], label="Total Train Loss")
                plt.legend()

                # Plotting BE loss
                plt.subplot(5, 1, 2) # Second plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Train BE Loss')
                plt.plot(train_be_loss[1:], label="Bellman Error Loss", color='red')
                plt.legend()

                # Plotting CE loss
                plt.subplot(5, 1, 3) # Third plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Train CE Loss')
                plt.plot(train_ce_loss[1:], label="Cross-Entropy Loss", color='blue')
                plt.legend()
                
                # Plotting r MAPE loss 
                plt.subplot(5, 1, 4) # Fifth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Test R MAPE Loss')
                plt.plot(test_r_MAPE_loss[1:], label="r MAPE Loss", color='purple')
                plt.legend()
                
                plt.subplot(5, 1, 5) # Sixth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('D Loss')
                plt.plot(train_D_loss[1:], label="D Loss", color='orange')
                plt.legend()
                
                
                plt.tight_layout()
                plt.savefig(f"figs/loss/{build_log_filename(config)}_rep{rep}_losses.png")
                plt.close()
            ############### Finish of an epoch ##############
        ##### Finish of all epochs #####
        
        printw(f"Best epoch for repetition {rep+1} : {best_epoch}", config)
        printw(f"Best R MAPE loss for repetition {rep+1}: {best_r_MAPE_loss}", config)
        
        ################## Finish of one repetition #########################      
        if best_epoch > 0:
            rep_best_r_MAPE_loss.append(best_r_MAPE_loss) 
        else:
            printw("No best r values were recorded during training.", config)  
            
        rep_test_r_MAPE_loss.append(test_r_MAPE_loss)
            
        torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')
        
        printw(f"\nTraining of repetition {rep+1} finished.", config)
        
    #### Finish of all repetitions ####    
    rep_test_r_MAPE_loss = np.array(rep_test_r_MAPE_loss) #dimension is (repetitions, num_epochs)
    
    mean_r_mape = np.mean(rep_test_r_MAPE_loss, axis=0) #dimension is (num_epochs,)
    std_r_mape = np.std(rep_test_r_MAPE_loss, axis=0)/np.sqrt(repetitions)
    
    epochs = np.arange(0, config['num_epochs'])
    
    plt.figure(figsize=(12, 6))  # Increase the height to fit all plots

    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('R MAPE Loss')
    plt.plot(mean_r_mape, label="Mean R MAPE Loss", color='blue')
    plt.fill_between(epochs, mean_r_mape - std_r_mape, mean_r_mape + std_r_mape, alpha=0.2, color='blue')
    plt.legend()

 
    plt.tight_layout()
    plt.savefig(f"figs/loss/Reps{repetitions}_{build_log_filename(config)}_losses.png")
    plt.close()
    
    printw(f"\nTraining completed.", config)
    mean_best_r_mape = np.mean(rep_best_r_MAPE_loss) 
    std_best_r_mape = np.std(rep_best_r_MAPE_loss)/np.sqrt(repetitions)
    ##separate logging for the final results
    printw(f"\nFinal results for {repetitions} repetitions", config)
    printw(f"Mean best R MAPE loss: {mean_best_r_mape}", config)
    printw(f"Standard error of best R MAPE loss: {std_best_r_mape}", config)

    