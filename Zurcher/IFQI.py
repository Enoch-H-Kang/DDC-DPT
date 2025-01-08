import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from functools import partial
from scipy.optimize import minimize
from mlp import MLP  # Import your multi-headed MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['H']
        self.store_gpu = config['store_gpu']
        self.config = config

        if not isinstance(path, list):
            path = [path]

        # load all trajectories
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

        states_total = np.array(states_total)
        actions_total = np.array(actions_total)
        next_states_total = np.array(next_states_total)

        states_true_EPs_total = np.array(states_true_EPs_total)
        states_true_Qs_total = np.array(states_true_Qs_total)
        states_true_expVs_total = np.array(states_true_expVs_total)
        next_states_true_EPs_total = np.array(next_states_true_EPs_total)
        next_states_true_Qs_total = np.array(next_states_true_Qs_total)
        busTypes = np.array(busTypes)

        self.dataset = {
            'states'            : self.convert_to_tensor(states_total, self.store_gpu),
            'actions'           : self.convert_to_tensor(actions_total, self.store_gpu),
            'next_states'       : self.convert_to_tensor(next_states_total, self.store_gpu),
            'states_true_EPs'   : self.convert_to_tensor(states_true_EPs_total, self.store_gpu),
            'states_true_Qs'    : self.convert_to_tensor(states_true_Qs_total, self.store_gpu),
            'states_true_expVs' : self.convert_to_tensor(states_true_expVs_total, self.store_gpu),
            'next_states_true_EPs': self.convert_to_tensor(next_states_true_EPs_total, self.store_gpu),
            'next_states_true_Qs': self.convert_to_tensor(next_states_true_Qs_total, self.store_gpu),
            'busTypes'          : self.convert_to_tensor(busTypes, self.store_gpu)
        }

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        res = {
            'states'            : self.dataset['states'][idx],
            'actions'           : self.dataset['actions'][idx],
            'next_states'       : self.dataset['next_states'][idx],
            'states_true_EPs'   : self.dataset['states_true_EPs'][idx],
            'states_true_Qs'    : self.dataset['states_true_Qs'][idx],
            'states_true_expVs' : self.dataset['states_true_expVs'][idx],
            'next_states_true_EPs': self.dataset['next_states_true_EPs'][idx],
            'next_states_true_Qs': self.dataset['next_states_true_Qs'][idx],
            'busType'           : self.dataset['busTypes'][idx]
        }
        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['states']      = res['states'][perm]
            res['actions']     = res['actions'][perm]
            res['next_states'] = res['next_states'][perm]
        return res

    @staticmethod
    def convert_to_tensor(x, store_gpu=True):
        t = torch.tensor(np.asarray(x), dtype=torch.float32)
        if store_gpu:
            t = t.to(device)
        return t


def sgd_fitted_q_iteration(config):
    """
    A mini-batch “SGD” style approach to Soft FQI.

    Pseudocode:
    for iteration in 1..n_iter:
        for each batch in train_loader:
            1) model.eval() -> get next_q_values, compute target
            2) model.train() -> do MSE step to match Q(s,a) to target

    We do not gather all states, actions, targets across the entire dataset
    in one big array. Instead, each mini-batch constructs its own target
    and we do an immediate gradient step.
    """
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    train_loader = config['train_loader']
    states_dim   = config['states_dim']
    actions_dim  = config['actions_dim']
    hidden_sizes = config.get('hidden_sizes', [64,64])
    layer_norm   = config.get('layer_norm', False)
    beta         = config['beta']
    lr           = config['lr']
    n_iter       = config['fqi_num_iterations']
    n_epochs     = config['fqi_epochs_per_iteration']

    # Suppose theta is [theta0, theta1]. If action=0 => r=theta0*mileage, else => r=theta1
    theta_np = config['theta']
    theta    = torch.tensor(theta_np, dtype=torch.float32, device=device)

    model = MLP(
        states_dim=states_dim,
        actions_dim=actions_dim,
        hidden_sizes=hidden_sizes,
        layer_normalization=layer_norm
    ).to(device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss_fn  = nn.MSELoss(reduction='mean')
    best_total_loss = 1e9
    itrDone = False
    for iteration in range(n_iter):
        if itrDone == True:
            break
        print(f"\n=== Soft FQI Iteration {iteration+1}/{n_iter} ===")
        
        # We'll do multiple epochs of going through entire dataset
        for ep in range(n_epochs): 
            total_loss = 0.0
            for batch in train_loader:
                # 1) Move everything to device
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                model.eval()
                with torch.no_grad():
                    # get Q(s,a), Q(s',a')
                    q_values, next_q_values, _ = model(batch) 
                    # shape => (B, H, actions_dim)

                    states  = batch['states']   # (B,H,state_dim)
                    next_states = batch['next_states']  # (B,H,state_dim)
                    actions = batch['actions']  # (B,H)

                    # mileage = states[:,:,0]
                    # reward = ...
                    mileage = states[:,:,0]
                    reward = torch.where(
                        actions==0,
                        theta[0]*mileage,
                        theta[1]*torch.ones_like(mileage)
                    )
                    # soft bellman target = r + beta * logsumexp(next_q_values)
                    logsumexp_nq = torch.logsumexp(next_q_values, dim=2)  # (B,H)
                    bellman_target = reward + beta*logsumexp_nq  # shape (B,H)

                # 2) model.train(), forward pass with Q(s,a) => do MSE vs target
                model.train() # set to training mode
                # Flatten
                B, H, actdim = q_values.size()
                states_flat  = states.reshape(B*H, -1)
                next_states_flat = next_states.reshape(B*H, -1)
                actions_flat = actions.reshape(B*H).long()
                target_flat  = bellman_target.reshape(B*H)
                

                # We'll feed single-step data
                s_batch_3d = states_flat.unsqueeze(1)  # (B*H,1,state_dim)
                # next_states not used for Q(s,a) training
                ns_batch_3d= next_states_flat.unsqueeze(1)  # (B*H,1,state_dim)

                train_dict = {
                    'states': s_batch_3d,      # shape (N,1, state_dim)
                    'actions': None,
                    'next_states': ns_batch_3d
                }
                pred_q, _, _ = model(train_dict)  # => (N,1,actions_dim)
                pred_q = pred_q.reshape(-1, pred_q.size(2))  # => (N, actions_dim)

                chosen_q = pred_q[torch.arange(len(actions_flat), device=device), actions_flat]
                loss = mse_loss_fn(chosen_q, target_flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
            print(f"   [Iter {iteration+1}, Epoch {ep+1}] MSE Loss: {total_loss:.3f}")
            print(f"   [Iter {iteration+1}, Epoch {ep+1}] Best MSE Loss: {best_total_loss:.3f}")
    
            if total_loss < best_total_loss:
                best_total_loss = total_loss
            else:
                if total_loss > 1.1*best_total_loss:
                    print("Early stopping!")
                    itrDone = True
    return model


def evaluate_pseudo_likelihood(q_model, config):
    """
    NLL = - sum_{(s,a)} log( softmax( Q(s,a) ) ).
    """
    test_loader = config['test_loader']
    q_model.eval()
    nll = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in test_loader:
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            q_values, _, _ = q_model(batch)
            actions = batch['actions']
            B, H, actdim = q_values.size()
            qvals_flat = q_values.reshape(B*H, actdim)
            acts_flat  = actions.reshape(B*H).long()

            lse = torch.logsumexp(qvals_flat, dim=1, keepdim=True)
            log_probs = qvals_flat - lse
            chosen_lp = log_probs[torch.arange(B*H, device=device), acts_flat]
            nll -= chosen_lp.sum().item()
            total_count += B*H
    if total_count==0:
        return 0.0
    return nll/ total_count


def objective_fn(theta_array, config):
    """
    A derivative-free objective.  We'll do mini-batch SGD in sgd_fitted_q_iteration
    for each guess of theta, then evaluate negative log-likelihood.
    """
    config['theta'] = theta_array
    q_model = sgd_fitted_q_iteration(config)
    nll = evaluate_pseudo_likelihood(q_model, config)
    return nll


###############################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    Example main function that:
      1. Sets up config
      2. Loads train/test data via your Dataset class
      3. Minimizes objective_fn with nelder-mead
    """
    # Example config
    config = {
        'H': 100,
        'beta': 0.95,
        'num_trajs': 50,
        'states_dim': 1,       # Suppose your state has dimension 5
        'actions_dim': 2,      # Suppose 2 possible actions
        'hidden_sizes': [64,64],
        'layer_norm': False,
        'lr': 1e-4,
        'fqi_num_iterations': 200,
        'fqi_epochs_per_iteration': 1,
        'batch_size': 128,
        'shuffle': True,
        'store_gpu': True,
        'seed': 42,

        'theta_init': [-1.0, -5.0],  # initial guess for [theta1,theta2]
    }

    # 1. Build paths
    train_path = 'datasets/trajs_train.pkl'
    test_path  = 'datasets/trajs_test.pkl'

    # 2. Build dataset => data loaders
    train_dataset = Dataset(train_path, config)
    test_dataset  = Dataset(test_path,  config)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False)
    config['train_loader'] = train_loader
    config['test_loader']  = test_loader
    
    sample_batch = next(iter(train_loader))
    # 3. Minimization approach
    #    We'll do a derivative-free search over [theta1,theta2], e.g. nelder-mead
    from functools import partial
    objective = partial(objective_fn, config=config)
    res = minimize(
        fun=objective,
        x0=config['theta_init'],
        method='nelder-mead',
        options={'maxiter': 1000, 'disp': True}
    )

    print("\n=== Optimization Complete ===")
    print("Best thetas:", res.x)
    print("Minimum negative log-likelihood:", res.fun)

    # If you want the final Q-model:
    #config['theta'] = res.x
    #final_q = sgd_fitted_q_iteration(config)
    #final_nll = evaluate_pseudo_likelihood(final_q, config)
    #print(f"Final Q-model negative log-likelihood: {final_nll:.4f}")

if __name__ == '__main__':
    main()
