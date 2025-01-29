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
import torch
import pickle
import numpy as np
import torch
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """Optimized Dataset class for storing and sampling (s, a, r, s') transitions."""

    def __init__(self, path, config):
        self.store_gpu = config.get('store_gpu', False)  # Store tensors on GPU if True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure path is a list
        if not isinstance(path, list):
            path = [path]

        # Load dataset efficiently
        states, actions, next_states, rewards = [], [], [], []
        
        for p in path:
            with open(p, 'rb') as f:
                trajs = pickle.load(f)

                # Extract and concatenate transitions in a single step (Vectorized)
                states.append(np.concatenate([traj['states'] for traj in trajs], axis=0))
                actions.append(np.concatenate([traj['actions'] for traj in trajs], axis=0))
                next_states.append(np.concatenate([traj['next_states'] for traj in trajs], axis=0))
                rewards.append(np.concatenate([traj['rewards'] for traj in trajs], axis=0))

        # Convert lists to single NumPy arrays
        states = np.concatenate(states, axis=0) #dimension is #transitions x state_dim
        actions = np.concatenate(actions, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)

        
        # Convert to PyTorch tensors
        self.dataset = {
            'states': self.convert_to_tensor(states, store_gpu=self.store_gpu),
            'actions': self.convert_to_tensor(actions, store_gpu=self.store_gpu),
            'next_states': self.convert_to_tensor(next_states, store_gpu=self.store_gpu),
            'rewards': self.convert_to_tensor(rewards, store_gpu=self.store_gpu),
        }

        # Shuffle dataset at initialization 
        if config.get('shuffle', False):
            self.shuffle_dataset()

    def __len__(self):
        """Return the number of transitions."""
        return len(self.dataset['states'])

    def __getitem__(self, idx):
        """Return a single (s, a, r, s') transition."""
        return {
            'states': self.dataset['states'][idx],
            'actions': self.dataset['actions'][idx],
            'next_states': self.dataset['next_states'][idx],
            'rewards': self.dataset['rewards'][idx]
        }

    def shuffle_dataset(self):
        """Shuffle all transitions."""
        indices = np.arange(len(self.dataset['states']))
        np.random.shuffle(indices)
        
        for key in self.dataset.keys():
            self.dataset[key] = self.dataset[key][indices]
        
    
    @staticmethod
    def convert_to_tensor(x, store_gpu):
        """Convert numpy array to tensor, optionally storing on GPU."""
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        return tensor.to("cuda") if store_gpu else tensor



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
    filename = (f"BC_{config['env']}_shuf{config['shuffle']}_lr{config['lr']}"
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
    
    filename = (f"BC_{config['env']}_num_trajs{config['num_trajs']}"
                f"_lr{config['lr']}"
                f"_batch{config['batch_size']}"
                f"_decay{config['decay']}"
                f"_clip{config['clip']}"
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
        
def corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    covariance = torch.mean((x - x_mean) * (y - y_mean))  # Cov(X, Y)
    std_x = torch.std(x, unbiased=False)  # Standard deviation of X
    std_y = torch.std(y, unbiased=False)  # Standard deviation of Y

    correlation = covariance / (std_x * std_y)  # Pearson correlation formula
    return correlation        

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
        train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=config['shuffle']
    )

    if config['env'] == 'LL':
        states_dim = 8
        actions_dim = 4
        init_b_value = 30
    elif config['env'] == 'CP':
        states_dim = 4
        actions_dim = 2
        init_b_value = 1
    elif config['env'] == 'AC':
        states_dim = 6
        actions_dim = 3
        init_b_value = -1
    else:
        print('Invalid environment')
        exit()
        
    def custom_output_b_init(bias):
        nn.init.constant_(bias, init_b_value)
    
    # Prepare model
    model_config = {
        'hidden_sizes' : [config['h_size']]*config['n_layer'],
        'layer_normalization': config['layer_norm'], #layer normalization
    }
    model = MLP(states_dim, actions_dim, output_b_init=custom_output_b_init, **model_config).to(device)
    
    q_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    vnext_optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    MSE_loss_fn = torch.nn.MSELoss(reduction='mean')
    MAE_loss_fn = torch.nn.L1Loss(reduction='mean')
    
    repetitions = config['repetitions']  # Number of repetitions

    rep_test_r_MAPE_loss = []
    rep_best_r_MAPE_loss = []

    
    for rep in range(repetitions):
        print(f"\nStarting repetition {rep+1}/{repetitions}")
        train_loss = []
        test_r_MAPE_loss = []
        
        #Storing the best training epoch and its corresponding best Q MSE loss/Q values
        best_epoch = -1
        best_r_MAPE_loss = 9999
      
        
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
                    batch = {k: v.to(device) for k, v in batch.items()} 
                    states = batch['states']
                    pred_q_values, pred_q_values_next, _  = model(batch) 

                    true_actions = batch['actions'].long() 
                    states = batch['states']
                
                    
                    #find the action with most frequent occurence in the batch                
                    true_rewards = batch['rewards']
                    
                    ### Q(s,a) 
                    chosen_q_values = torch.gather(pred_q_values, dim=1, index=true_actions.unsqueeze(-1)) #dimension is (batch_size, horizon)
                    
                    #Empirical V(s') = logsumexp Q(s',a') + gamma
                    logsumexp_nextstate = torch.logsumexp(pred_q_values_next, dim=1) #dimension is (batch_size*horizon,)
                    vnext = logsumexp_nextstate

                    #Terminating conditions    
                    
                    if config['env'] == 'LL':
                        terminal_TF = (states[:, 6] == 1) & (states[:, 7] == 1)
                    elif config['env'] == 'AC':
                        terminal_TF = (-torch.cos(states[:, 0]) - torch.cos(states[:, 0] + states[:, 1])) > 1.0
                    elif config['env'] == 'CP': 
                        x_threshold = 2.4
                        theta_threshold_radians = 12 * 2 * np.pi / 360  # 12 degrees in radians
                        terminal_TF = (torch.abs(states[:, 0]) > x_threshold) | (torch.abs(states[:, 2]) > theta_threshold_radians)
                    else: #retrun terminal_TF as all False
                        terminal_TF = torch.zeros_like(states[:, 0], dtype=torch.bool)
                    
                    vnext = torch.where(terminal_TF, torch.tensor(0.0, device=vnext.device), vnext)
                    
                    # First term of BC-learn
                    pred_r_values = chosen_q_values - config['beta']*vnext #dimension is (batch_size, horizon)
                   
                    ##############For computing r_MAPE (mean absolute percentage error)########################
                    diff = torch.abs(pred_r_values - true_rewards) #dimension is (batch_size, horizon)
                    #denom = torch.abs(true_r_values) #dimension is (batch_size, horizon)
                    #r_MAPE = torch.mean(diff / denom)*100 #dimension is (1,), because it is the mean of all diff/denom values in the batch
                    r_MAPE = torch.mean(diff)
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
            start_time = time.time()
            
            torch.autograd.set_detect_anomaly(True)
            
            
            for i, batch in enumerate(train_loader): #For batch i in the training dataset
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()} #dimension is (batch_size, horizon, state_dim)
                
                pred_q_values, pred_q_values_next, _  = model(batch) 

                true_actions = batch['actions'].long() 
                Mean_CrossEntropy_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = Mean_CrossEntropy_loss_fn(pred_q_values, true_actions)
                
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
             
                print(f"Epoch_train_loss: {epoch_train_loss}", end='\r')

            
                
       
                
                
            #len(train_dataset) is the number of batches in the training dataset
            train_loss.append(epoch_train_loss / len(train_loader)) 
        
            end_time = time.time()
            
            printw(f"\tTrain loss: {train_loss[-1]}", config)
            printw(f"\tTrain time: {end_time - start_time}", config)


            # Logging and plotting
            
            if (epoch + 1) % 1000 == 0:
                torch.save(model.state_dict(),
                    f'models/{build_log_filename(config)}_rep{rep}_epoch{epoch+1}.pt')

            if (epoch + 1) % 1 == 0:
                plt.figure(figsize=(12, 12))  # Increase the height to fit all plots
    
                # Plotting total train loss
                plt.subplot(2, 1, 1) # Adjust to 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Total Train Loss')
                plt.plot(train_loss[1:], label="Total Train Loss")
                plt.legend()

                # Plotting r MAPE loss 
                plt.subplot(2, 1, 2) # Fifth plot in a 6x1 grid
                plt.yscale('log')
                plt.xlabel('epoch')
                plt.ylabel('Test R MAPE Loss')
                plt.plot(test_r_MAPE_loss[1:], label="r MAPE Loss", color='purple')
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

    