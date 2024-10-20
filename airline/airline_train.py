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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlp_module')))
from mlp import MLP
from datetime import datetime



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""
    #train_dataset = Dataset(path_train, dataset_config)

    def __init__(self, data, config):
        self.shuffle = config['shuffle']
        self.store_gpu = config['store_gpu']
        self.config = config
        '''
        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
        '''
        with open(data, "rb") as f:
            loaded_trajs = pickle.load(f)
        
        states_total = loaded_trajs['states']
        actions_total = loaded_trajs['actions']
        next_states_total = loaded_trajs['next_states']
            
        states_total = np.array(states_total) #dimension of states_total is (num_trajs, H, state_dim)
                    #when a batch is called, the dimension of the batch is (batch_size, H, state_dim)
        self.len_traj = len(states_total)
        actions_total = np.array(actions_total)
        next_states_total = np.array(next_states_total)
        
        states_total = np.nan_to_num(states_total, nan=0.0)
        actions_total = np.nan_to_num(actions_total, nan=0.0)
        next_states_total = np.nan_to_num(next_states_total, nan=0.0)

        states_total = np.asarray(states_total, dtype=np.float32).reshape(states_total.shape)
        actions_total = np.asarray(actions_total, dtype=np.float32).reshape(actions_total.shape)
        next_states_total = np.asarray(next_states_total, dtype=np.float32).reshape(next_states_total.shape)
        
        #print(states_total.shape)
        #print(type(states_total))
        #print(states_total.dtype)
        #print(states_total[0][254])
        #print(np.isnan(states_total).any())
        #exit(0)
        self.dataset = {
            'states': Dataset.convert_to_tensor(states_total, store_gpu=self.store_gpu),
            'actions': Dataset.convert_to_tensor(actions_total, store_gpu=self.store_gpu),
            'next_states': Dataset.convert_to_tensor(next_states_total, store_gpu=self.store_gpu),
        }
    def __len__(self):
        return self.len_traj
    
    def __getitem__(self, idx):
        #'Generates one sample of data'. DataLoader constructs a batch using this.
        res = {
            'states': self.dataset['states'][idx],
            'actions': self.dataset['actions'][idx],
            'next_states': self.dataset['next_states'][idx]
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

def loss_ratio3(x): #x is the epoch number
    if x < 1:
        return 1/5000  # For x < 1, return the start value
    else:  
        return 1/5000*x
    
def loss_ratio2(x): #x is the epoch number
    if x < 1:
        return 10000  # For x < 1, return the start value
    else:  
        return 10000/x

def build_data_filename(config, mode):
    """
    Builds the filename for the airline data.
    Mode is either 'train' or 'test'.
    """
    filename_template = 'datasets/airline_carr_id_{}_{}.pkl'
    filename = filename_template.format(config['carr_id'], mode)
    return filename


def build_model_filename(config):
    """
    Builds the filename for the model.
    """
    filename = (f"{config['env']}_shuf{config['shuffle']}_lr{config['lr']}"
                f"_do{config['dropout']}_embd{config['n_embd']}"
                f"_layer{config['n_layer']}_head{config['n_head']}"
                f"_seed{config['seed']}")
    return filename

def build_log_filename(config):
    """
    Builds the filename for the log file.
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = (f"{config['env']}"
                f"{config['carr_id']}"
                f"_beta{config['beta']}"
                f"_batch{config['batch_size']}"
                f"div{config['div']}"
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
        'beta': config['beta'],
        'store_gpu': True,
        'shuffle': config['shuffle'],
        'env': config['env'],
        'carr_id': config['carr_id'],
        'div': config['div']
    }

    path_train = build_data_filename(dataset_config, mode='train')
    path_test = build_data_filename(dataset_config, mode='test')

    train_dataset = Dataset(path_train, dataset_config)
    test_dataset = Dataset(path_test, dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    
    states_dim = 11 
    actions_dim = 2
    # Prepare model
    model_config = {
        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], #layer normalization
    }
    model = MLP(states_dim, actions_dim, **model_config).to(device)

    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    #CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss()
    MSE_loss_fn = torch.nn.MSELoss(reduction='sum')
    #MAE_loss_fn = torch.nn.L1Loss(reduction='sum')
    MAE_loss_fn = torch.nn.L1Loss()
    repetitions = config['repetitions']  # Number of repetitions

    rep_test_ce_loss = []
    rep_best_ce_loss = []    

    for rep in range(repetitions):
        print(f"\nStarting repetition {rep+1}/{repetitions}")
        train_loss = []
        train_be_loss = []
        train_ce_loss = []
        train_D_loss = []
        test_ce_loss = []
        test_vnext_MSE_loss = []
        
        #Storing the best training epoch and its corresponding best ce loss
        best_epoch = -1
        best_ce_loss = 9999

        for epoch in tqdm(range(config['num_epochs']), desc="Training Progress"):
            
            ############### Start of an epoch ##############
            
            ### EVALUATION ###
            printw(f"Epoch: {epoch + 1}", config)
            start_time = time.time()
            with torch.no_grad():
                epoch_ce_loss = 0.0
           
                ##### Test batch loop #####
                
                for i, batch in enumerate(test_loader):
                    
                    print(f"Batch {i} of {len(test_loader)}", end='\r')
                    batch = {k: v.to(device) for k, v in batch.items()} #dimension is (batch_size, horizon, state_dim)
                    states = batch['states']
                    pred_q_values, _ , pred_vnext_values = model(batch) #dimension is (batch_size, horizon, action_dim)
                    
                    batch_size = pred_q_values.shape[0]
                    H = pred_q_values.shape[1]
                    
                    
                    true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
                    true_actions_reshaped = true_actions.reshape(-1) #dimension is (batch_size*horizon,)
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
                    
                    
                    ####### Action CrossEntropy loss                
                    ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped)
                    epoch_ce_loss += ce_loss.item()/H

                    pred_r_values = pred_q_values - config['beta']*pred_vnext_values #dimension is (batch_size, horizon, action_dim)
                    chosen_pred_r_values = torch.gather(pred_r_values, dim=2, index=true_actions).squeeze(-1)
                    #dimension is (batch_size, horizon)
                    


                
                ##### Finish of the batch loop for a single epoch #####
                ##### Back to epoch level #####
                # Note that epoch MSE losses are sum of all test batch means in the epoch
                
                if epoch_ce_loss/len(test_dataset) < best_ce_loss: #epoch_r_MSE_loss is sum of all test batch means in the epoch
        
                    best_ce_loss = epoch_ce_loss/len(test_dataset) #len(test_dataset) is the number of batches in the test dataset
                    best_epoch = epoch          
                    best_ce_loss = epoch_ce_loss
                    
                    best_epoch = epoch    
            
            ############# Finish of an epoch's evaluation ############
            
            test_ce_loss.append(epoch_ce_loss / len(test_dataset)) #len(test_dataset) is the number of batches in the test dataset
            
            end_time = time.time()
            printw(f"\tCross entropy loss: {test_ce_loss[-1]}", config)
            printw(f"\tEval time: {end_time - start_time}", config)
            
            
            ############# Start of an epoch's training ############
            
            epoch_train_loss = 0.0
            epoch_train_be_loss = 0.0
            epoch_train_ce_loss = 0.0
            epoch_train_D_loss = 0.0
            start_time = time.time()
            
            torch.autograd.set_detect_anomaly(True)
            num_batches = len(train_loader)
            print(f"Number of batches: {num_batches}")
            
            for i, batch in enumerate(train_loader): #For batch i in the training dataset
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                
                
                pred_q_values, pred_q_values_next, pred_vnext_values = model(batch) #dimension is (batch_size, horizon, action_dim)
                
                batch_size = pred_q_values.shape[0]
                H = pred_q_values.shape[1]
                
                true_actions = batch['actions'].long() #dimension is (batch_size, horizon,)
                #in torch, .long() converts the tensor to int64. CrossEntropyLoss requires the target to be int64.
                
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
                
                if config['proj'] == True:
                    div = config['div']
                else:
                    div = 2
               
                if i % div == 0: # update D only, div=3 means update D every 3 batches
                    
                    #V(s')-E[V(s')] minimization loss
                    D = MSE_loss_fn(vnext_reshaped.clone().detach(), chosen_vnext_values_reshaped)
                    D.backward()
                    vnext_optimizer.step() #we use separate optimizer for vnext
                    vnext_optimizer.zero_grad() #clear gradients for the batch
                    epoch_train_D_loss += D.item() / H #per-sample loss
                    model.zero_grad() #clear gradients for the batch. This prevents the accumulation of gradients.
                    
                else:     # QtoVmodel parameters only
                    
                    ce_loss = CrossEntropy_loss_fn(pred_q_values_reshaped, true_actions_reshaped) #shape  is (batch_size*horizon,)
                    #print(f"CE loss: {ce_loss.item()}")
                    
                    
                    #First, compute td error for (s,a) pairs that appear in the data. 
                    #Non-pivot actions will be removed anyways, so I just add pivot_rewards for all cases here
                    
                    pivot_rewards = 0*torch.ones(batch_size, H) #non-entry(action 1), dimension is (batch_size, horizon) 
                    pivot_rewards_reshaped = pivot_rewards.reshape(-1) #dimension is (batch_size*horizon,)
                    pivot_rewards_reshaped = pivot_rewards_reshaped.to(chosen_q_values_reshaped.device)

                    td_error = chosen_q_values_reshaped - pivot_rewards_reshaped - config['beta'] * vnext_reshaped #\delta(s,a) = Q(s,a) - r(s,a) - beta*V(s')
                    #V(s')-E[V(s')|s,a]
                    '''
                    vnext_dev = (vnext_reshaped - chosen_vnext_values_reshaped.clone().detach())
                    #Bi-conjugate trick to compute the Bellman error
                    be_error_naive = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                    '''
                    
                    vnext_dev = (vnext_reshaped - chosen_vnext_values_reshaped.clone().detach())
                    #Bi-conjugate trick to compute the Bellman error
                    be_error_naive = td_error**2-config['beta']**2 * vnext_dev**2 #dimension is (batch_size*horizon,)
                    #We call it naive because we just add pivot r for every actions we see in the batch
                    
                    #Exclude the action 1 from computing the Bellman error, leaving pivot cases only.
                    be_error_0 = torch.where(true_actions_reshaped == 1, 0, be_error_naive) #only consider the Bellman error for action 0
                    #be_loss is normalized by the number of nonzero true-action batch numbers
                    be_loss = MAE_loss_fn(be_error_0, torch.zeros_like(be_error_0))
                    
                    if config['proj'] == True: 

                        if i % div == 1:
                            loss = ce_loss + config['loss_ratio']*loss_ratio(epoch, 0, 1, 50000) * be_loss
                        else: 
                            loss = ce_loss
                    else: #i.e. div=2
                        loss =  ce_loss + config['loss_ratio']*loss_ratio(epoch, 0, 1, 50000) * be_loss  
                 
                    
                         
                    loss.backward()
                    q_optimizer.step()
                    q_optimizer.zero_grad() #clear gradients for the batch
                
                    model.zero_grad()
                    
                    epoch_train_loss += loss.item() / H
                    epoch_train_be_loss += be_loss.item() / H
                    epoch_train_ce_loss += ce_loss.item() / H
                    
                    print(f"Epoch_train_loss: {epoch_train_loss}", end='\r')

                
                if i == 0: #i=0 means the first batch
                    pred_r_values_print = pred_q_values[:,-1,:] - config['beta']*pred_vnext_values[:,-1,:] #for print
                    states = batch['states']
                    last_states = states[:,-1,0].unsqueeze(1)#dimension is (batch_size, state_dim)
                    pred_r_values_with_states = torch.cat((last_states, pred_r_values_print), dim=1) #dimension is (batch_size, state_dim+action_dim)
                    printw(f"Predicted r values: {pred_r_values_with_states[:10]}", config)
                
                pred_r_values = pred_q_values - config['beta']*pred_vnext_values #dimension is (batch_size, horizon, action_dim)
                chosen_pred_r_values = torch.gather(pred_r_values, dim=2, index=true_actions).squeeze(-1)
                #dimension is (batch_size, horizon)
                
            ##### Finish of the batch loop for a single epoch #####
                
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
            
            #if (epoch + 1) % 10000 == 0:
            #    torch.save(model.state_dict(),
            #            f'models/{build_log_filename(config)}_rep{rep}_epoch{epoch+1}.pt')

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
                
                plt.subplot(4, 1, 4) # Sixth plot in a 6x1 grid
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
        printw(f"Best ce loss for repetition {rep+1}: {best_ce_loss}", config)
        
        ################## Finish of one repetition #########################      
        if best_epoch > 0:
            rep_best_ce_loss.append(best_ce_loss) 
        else:
            printw("No best ce values were recorded during training.", config)  
            
        rep_test_ce_loss.append(test_ce_loss)
            
        torch.save(model.state_dict(), f'models/{build_log_filename(config)}.pt')
        
        printw(f"\nTraining of repetition {rep+1} finished.", config)
        
    #### Finish of all repetitions ####    
    rep_test_ce_loss = np.array(rep_test_ce_loss) #dimension is (repetitions, num_epochs)
    mean_ce = np.mean(rep_test_ce_loss, axis=0) #dimension is (num_epochs,)
    std_ce = np.std(rep_test_ce_loss, axis=0)/np.sqrt(repetitions)
    
    epochs = np.arange(0, config['num_epochs'])
    
    plt.figure(figsize=(12, 12))  # Increase the height to fit all plots

    # Plotting BE loss
    #plt.subplot(2, 1, 2) # Second plot in a 2x1 grid
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('ce Loss')
    plt.plot(mean_ce, label="Mean ce Loss", color='red')
    plt.fill_between(epochs, mean_ce - std_ce, mean_ce + std_ce, alpha=0.2, color='red')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"figs/loss/Reps{repetitions}_{build_log_filename(config)}_losses.png")
    plt.close()
    
    printw(f"\nTraining completed.", config)
    mean_best_ce = np.mean(rep_best_ce_loss)
    std_best_ce = np.std(rep_best_ce_loss)/np.sqrt(repetitions)
    ##separate logging for the final results
    printw(f"\nFinal results for {repetitions} repetitions", config)
    printw(f"Mean best ce loss: {mean_ce_mse}", config)
    printw(f"Standard error of best ce loss: {std_ce_mse}", config)
        
    