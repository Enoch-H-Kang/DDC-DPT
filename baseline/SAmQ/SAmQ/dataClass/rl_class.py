## A class for giving data in a batch mode. 

from torch.utils.data import Dataset, DataLoader 
import random
import numpy as np
import torch 
import os
import pickle
import joblib

class traj_online_rl_dataclass():
    def __init__(self, data):
        self.r = data['r']
        self.s = data['s']
        self.a = data['a']
    def __len__(self):
        return len(self.a)
    def __getitem__(self,idx):
        r = torch.FloatTensor([self.r[idx]])
        s = torch.FloatTensor(self.s[idx])
        a = torch.LongTensor([self.a[idx]])
        s_next = torch.FloatTensor(self.s[idx+1])
        return r, s, a, s_next

