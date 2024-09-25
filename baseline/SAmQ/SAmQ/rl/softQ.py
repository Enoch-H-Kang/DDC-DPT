"""
Main playground module for soft q iteration RL
"""
import numpy as np
import os
import torch
import torch.nn
from torch.nn import functional as F  # NOQA
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from scipy.special import softmax
import datetime
import sys
from dowel import logger, tabular
import pickle
from SAmQ.helper.functions import _Default, make_optimizer, makePath,setupTensorboard, dumpAllTensorboard
from SAmQ.estimation.qEstimation import qEstimation

from multiprocessing import Pool
from scipy import stats
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import time

import pickle
import joblib
import pandas as pd

from SAmQ.dataClass.rl_class import traj_online_rl_dataclass
from SAmQ.env.bus_env import syn_bus_env



class soft_q_iteration():
    def __init__(self, discount, optimizer, l_r, hidden_size, env, log):        
        self.env = env
        self.discount = discount        
        self.log = log
        self.path = log.dir / 'data_generation'
        os.mkdir(self.path)

        # Setting up the training parameters for deep models. 
        self.q = qEstimation(self.env.d_s ,self.env.n_a, hidden_sizes=[hidden_size,hidden_size], hidden_nonlinearity=torch.relu)
        self.l_r = l_r
        self.optimizer = make_optimizer(optimizer, self.q, lr=self.l_r, amsgrad = True)

    def train(self, n_epochs, batch_size, epoch_size, writer, num_workers = 0,patience = 5):
        # soft DQN 
        lossF = torch.nn.MSELoss()
        min_loss = np.inf
        waiting = 0
        result_dict = {}
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0
            count = 0
            # Each iteration, re generate data. Thus no need for testing.
            self.data = traj_online_rl_dataclass(self.env.get_data(epoch_size))
            dataloaders = DataLoader(self.data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            
            for r, s, a, s_next in dataloaders:
                v =  torch.log(torch.exp(self.q(s_next)).sum(1))
                y = r + self.discount *v.reshape((len(r),1))
                est = torch.gather(self.q(s), 1, a)
                lossV = lossF(est, y)
                count += 1
                self.optimizer.zero_grad()
                lossV.backward()
                self.optimizer.step()
                epoch_loss += lossV.item()
                
            train_dict = {'Mean Training Bellman Loss':epoch_loss/count, 'waiting':waiting}
            for k, value in train_dict.items():
                writer.add_scalar('softq/'+k, value, epoch)
            notStop = epoch_loss < min_loss
            if notStop:
                min_loss =epoch_loss
                waiting = 0
                torch.save(self.q.state_dict(), self.path/'best_softq_model.pth')
            else:
                waiting+=1
            if waiting >patience:
                break
        
        self.log(f'Final Bellman Error for softQ: {epoch_loss/count}. Epochs: {epoch}')

        return train_dict

    def getAction(self, s):
        q_value = self.q(torch.FloatTensor(s)).detach().numpy()
        p = softmax(q_value)
        a = np.random.choice(self.env.n_a, 1, p=p).item()
        return a,p,q_value
    def generate_data(self, N ):
        self.N = N
        self.env.reset()
        r,s,a,p,q = [],[self.env.s],[],[],[]
        for i in range(N):
            temp_a, temp_p, temp_q = self.getAction(s[i])
            r.append(self.env.step(temp_a))
            s.append(self.env.s)
            a.append(temp_a)
            p.append(temp_p)
            q.append(temp_q)
        r = np.array(r).reshape((len(a),1))
        a = np.array(a).reshape((len(a),1))
        s = np.array(s)
        p = np.array(p)
        q = np.array(q)
        print(f'The portion of replacement in training data is {np.array(a).mean()}')

        np.save(self.path/'s.npy', s[:-2,:])
        np.save(self.path/'s_next.npy', s[1:-1,:])
        np.save(self.path/'a.npy', a[:-1,:])
        np.save(self.path/'p.npy', p)
        np.save(self.path/'r.npy', r)
        np.save(self.path/'q.npy', q)

        # do it again for testing
        self.env.reset()
        s,a = [self.env.s],[]
        for i in range(N):
            temp_a, _,_ = self.getAction(s[i])
            self.env.step(temp_a)
            s.append(self.env.s)
            a.append(temp_a)
        a = np.array(a).reshape((len(a),1))
        s = np.array(s)
        print(f'The portion of replacement in test data is {np.array(a).mean()}')

        np.save(self.path/'s_test.npy', s[:-1,:])
        np.save(self.path/'a_test.npy', a)

        self.log(f'Number of diferent states:{len(np.unique(s, axis=0))}')
        return len(np.unique(s, axis=0))



if __name__ == "__main__":
    dS = int(sys.argv[1]) # Dimension of the state variable. Useful ones are only 2. 
    testEnv = rlRunner(N=50000, dS=dS, nA=2, b=5, theta=2)
    testEnv.getData()
    testEnv.setUp(discount = 0.8, qOptimizer = torch.optim.Adam,  qLr = 0.003,)
    testEnv.trainQEstimation(100,100)
    testEnv.generateData(10000)
    
    