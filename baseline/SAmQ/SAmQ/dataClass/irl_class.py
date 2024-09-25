# dataClass for PQR

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
import random
import numpy as np
import torch 
import os
import pickle
import joblib
import glob
from pathlib import Path


class pqr_dataclass():
    def __init__(self, data_path,aggregate_desc=None):
        self.aggregate_desc = aggregate_desc
        self.s = np.load(data_path / 's.npy')
        self.s_test = np.load(data_path / 's_test.npy')
        self.a_test = np.load(data_path / 'a_test.npy')
        self.s_next = np.load(data_path / 's_next.npy')
        self.a = np.load(data_path / 'a.npy')
        self.p = np.load(data_path / 'p.npy')
        self.q = np.load(data_path / 'q.npy')
        self.r = np.load(data_path / 'r.npy')
        self.s_real = np.load(data_path / 's.npy')
        if aggregate_desc is not None:
            '''
            self.s = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_value.npy'))
            self.s_test = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_test_value.npy'))
            self.s_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_next_value.npy'))
            '''
            self.s = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
            self.s_test = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_test.npy'))[:,None]
            self.s_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_next.npy'))[:,None]
            
            #self.s_agg = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
            #self.s_agg_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
    def __len__(self):
        return len(self.s)
    def __getitem__(self,idx):
        s = torch.FloatTensor(self.s[idx])
        q = torch.FloatTensor(self.q[idx])
        r = torch.FloatTensor(self.r[idx])
        s_next = torch.FloatTensor(self.s_next[idx])
        a = torch.LongTensor([self.a.reshape(len(self.a))[idx]])
        q_next = torch.FloatTensor(self.q[idx+1])
        p = torch.FloatTensor(self.p[idx])
        p_next = torch.FloatTensor(self.p[idx+1])
        return s,a, p,p_next,s_next,q,q_next,r 

class airline_dataclass():
    def __init__(self, data_path,aggregate_desc=None):    
        self.s = np.load(data_path / 's.npy')
        self.s_next = np.load(data_path / 's_next.npy')
        self.a = np.load(data_path / 'a.npy')
        self.s_test = np.load(data_path / 's_test.npy')
        self.s_real = np.load(data_path / 's.npy')
        if aggregate_desc is not None:
            '''
            self.s = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_value.npy'))
            self.s_test = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_test_value.npy'))
            self.s_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_next_value.npy'))
            '''
            #self.s_agg = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
            #self.s_agg_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
            self.s = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s.npy'))[:,None]
            self.s_test = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_test.npy'))[:,None]
            self.s_next = np.load(data_path/'..'/'train'/(aggregate_desc+'aggregated_s_next.npy'))[:,None]
            



    def __len__(self):
        return len(self.s)
    def __getitem__(self,idx):
        s = torch.FloatTensor(self.s[idx,:])
        s_next = torch.FloatTensor(self.s_next[idx,:])
        a = torch.LongTensor(self.a[idx,:])

        q = torch.FloatTensor([0])
        r = torch.FloatTensor([0])
        q_next = torch.FloatTensor([0])
        p = torch.FloatTensor([0])
        p_next = torch.FloatTensor([0])

        return s,a, p,p_next,s_next,q,q_next,r 

if __name__ == "__main__":
    test_env = airline_dataclass(Path('/home/ubuntu/github/cS/ailrline_data'), 'all')
    print(test_env[0])
    print(len(test_env))