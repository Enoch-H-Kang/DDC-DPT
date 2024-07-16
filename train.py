import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch

import numpy as np
import common_args
import random
from dataset import Dataset
from net import Transformer
from utils import (
    build_Zurcher_data_filename,
    build_Zurcher_model_filename,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    print("Args: ", args)

    #python3 train.py --env Zurcher --bustotal 100 --H 100 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1
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

    log_filename = f'figs/loss/{filename}_logs.txt'
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
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    MSEloss_fn = torch.nn.MSELoss(reduction='sum')
    KLDivloss_fn = torch.nn.KLDivLoss(reduction='sum')

    
    train_MSE_loss = []
    test_MSE_loss = []
    test_Q_MSE_loss = []

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))
 
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_MSE_loss = 0.0
            epoch_Q_MSE_loss = 0.0
            
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                
                true_actions = batch['query_actions'] #dimension is (batch_size, action_dim)
                query_states = batch['query_states'] #dimension is (batch_size, state_dim)
                query_true_EPs = batch['query_true_EPs'] #dimension is (batch_size, action_dim)
                query_true_Qs = batch['query_true_Qs'] #dimension is (batch_size, action_dim)
                
                
                pred_q_values = model(batch) #dimension is (batch_size, horizon, action_dim)
                full_pred_q_values = pred_q_values[:,-1,:] #dimension is (batch_size, action_dim)
                pred_actions = torch.softmax(full_pred_q_values, dim=1) #dimension is (batch_size, action_dim)


                #Print and compare normalized Q values
                
                min_true_Qs = torch.min(query_true_Qs, dim=1, keepdim=True)[0]
                normalized_true_Qs = query_true_Qs - min_true_Qs
            
                min_q_values = torch.min(full_pred_q_values, dim=1, keepdim=True)[0]
                normalized_full_pred_q_values = full_pred_q_values - min_q_values

                if i == 0:                    
                    print(normalized_true_Qs)
                    print(normalized_full_pred_q_values)
                    
                #true_actions = true_actions.unsqueeze(1).repeat(1, pred_q_values.shape[1], 1)
                
                cross_entropy_loss = loss_fn(pred_actions, query_true_EPs)
                Q_MSE_loss = MSEloss_fn(normalized_true_Qs, normalized_full_pred_q_values)
                epoch_MSE_loss += cross_entropy_loss.item() / horizon
                epoch_Q_MSE_loss += Q_MSE_loss.item() / horizon

        test_MSE_loss.append(epoch_MSE_loss / len(test_dataset))
        test_Q_MSE_loss.append(epoch_Q_MSE_loss / len(test_dataset))
        end_time = time.time()
        printw(f"\tCross entropy test loss: {test_MSE_loss[-1]}")
        printw(f"\tMSE of Q-value: {test_Q_MSE_loss[-1]}")
        printw(f"\tEval time: {end_time - start_time}")


        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            pred_q_values = model(batch) #dimension is (batch_size, horizon, action_dim)

            true_actions = batch['query_actions'] #dimension is (batch_size, action_dim)
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_q_values.shape[1], 1) #dimension is (batch_size, horizon, action_dim)
            true_actions = true_actions.reshape(-1, action_dim)  #dimension is (batch_size*horizon, action_dim)
            
            pred_q_values = pred_q_values.reshape(-1, action_dim) #dimension is (batch_size*horizon, action_dim)
            pred_actions = torch.softmax(pred_q_values, dim=1) 
            
            optimizer.zero_grad()
            
            loss = MSEloss_fn(pred_actions, true_actions)
           
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon

        train_MSE_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain MSE loss: {train_MSE_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")


        # LOGGING
        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(),
                       f'models/{filename}_epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 5 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Q value MSE Loss:        {test_Q_MSE_loss[-1]}")
            printw(f"Test action cross entropy Loss:        {test_MSE_loss[-1]}")
            printw(f"Train action cross entropy Loss:       {train_MSE_loss[-1]}")
            printw("\n")

            plt.yscale('log')
            plt.plot(train_MSE_loss[1:], label="Train prediction_MSE Loss")
            plt.plot(test_MSE_loss[1:], label="Test prediction_MSE Loss")
            plt.plot(test_Q_MSE_loss[1:], label="Test Q_MSE Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{filename}_train_MSE&QMSE_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'models/{filename}.pt')
    print("Done.")
