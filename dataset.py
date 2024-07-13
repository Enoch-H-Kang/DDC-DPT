import pickle

import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# The `Dataset` class extends `torch.utils.data.Dataset` and is used to load, preprocess, and manage the data 
# for training machine learning models. It provides a standard way to handle the dataset, making it compatible with 
# PyTorch's DataLoader, which can be used for batching, shuffling, and parallel loading of the data.

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

