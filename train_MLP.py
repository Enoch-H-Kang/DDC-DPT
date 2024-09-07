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
        
        states_total = []
        actions_total = []
        next_states_total = []
        states_true_EPs_total = []
        next_states_true_EPs_total = []
        states_true_Qs_total = []
        next_states_true_Qs_total = []
        busTypes = []

        for traj in self.trajs:
            states_total.append(traj['states']) #dimension of states_total is (num_trajs, H, state_dim)
            actions_total.append(traj['actions'])
            next_states_total.append(traj['next_states'])
            
            states_true_EPs_total.append(traj['states_true_EPs'])
            states_true_Qs_total.append(traj['states_true_Qs'])
            next_states_true_EPs_total.append(traj['next_states_true_EPs'])
            next_states_true_Qs_total.append(traj['next_states_true_Qs'])

            busTypes.append(traj['busType'])

        states_total = np.array(states_total)
        actions_total = np.array(actions_total)
        next_states_total = np.array(next_states_total)

        states_true_EPs_total = np.array(states_true_EPs_total)
        states_true_Qs_total = np.array(states_true_Qs_total)
        next_states_true_EPs_total = np.array(next_states_true_EPs_total)
        next_states_true_Qs_total = np.array(next_states_true_Qs_total)

        busTypes = np.array(busTypes)

        self.dataset = {
            'states_total': Dataset.convert_to_tensor(states_total, store_gpu=self.store_gpu),
            'actions_total': Dataset.convert_to_tensor(actions_total, store_gpu=self.store_gpu),
            'next_states_total': Dataset.convert_to_tensor(next_states_total, store_gpu=self.store_gpu),
            'states_true_EPs_total': Dataset.convert_to_tensor(states_true_EPs_total, store_gpu=self.store_gpu),
            'states_true_Qs_total': Dataset.convert_to_tensor(states_true_Qs_total, store_gpu=self.store_gpu),
            'next_states_true_EPs_total': Dataset.convert_to_tensor(next_states_true_EPs_total, store_gpu=self.store_gpu),
            'next_states_true_Qs_total': Dataset.convert_to_tensor(next_states_true_Qs_total, store_gpu=self.store_gpu),
            'busTypes': Dataset.convert_to_tensor(busTypes, store_gpu=self.store_gpu)
        }

    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self, idx):
        #'Generates one sample of data'. DataLoader constructs a batch using this.
        res = {
            'states': self.dataset['states_total'][idx],
            'actions': self.dataset['actions_total'][idx],
            'next_states': self.dataset['next_states_total'][idx],
            'states_true_EPs': self.dataset['states_true_EPs_total'][idx],
            'states_true_Qs': self.dataset['states_true_Qs_total'][idx],
            'next_states_true_EPs': self.dataset['next_states_true_EPs_total'][idx],
            'next_states_true_Qs': self.dataset['next_states_true_Qs_total'][idx],
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