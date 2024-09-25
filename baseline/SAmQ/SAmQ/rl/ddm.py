# Main pg for iteration-based DDM
"""
Main playground module for IRL
"""
import numpy as np
import os
import torch
import torch.nn
from torch.nn import functional as F  # NOQA
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

import datetime
import pickle
import joblib
from SAmQ.helper.functions import _Default, make_optimizer
from SAmQ.estimation.qEstimation import qEstimation

from multiprocessing import Pool
from scipy import stats

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import pandas as pd
import glob

class rust_ddm():
    def __init__(self, discount, log, env, delta, r_f, aggregation_method):
        self.log = log
        self.discount = discount
        self.env = env
        self.n_a = env.n_a
        self.d_s = env.d_s #theta = [1,0,0] and args.d_s = len(args.theta). Therefore d_s = 3
        file_name = str(delta)+aggregation_method+'aggregated_s.npy'
        self.aggregated_s = np.load(log.dir/'train'/file_name)
        self.s = np.load(log.dir/'data_generation'/'s.npy')
        self.a = np.load(log.dir/'data_generation'/'a.npy')
        self.s_next = np.load(log.dir/'data_generation'/'s_next.npy')
        #self.a_next = np.load(log.dir/'data_generation'/'a_next.npy')
        self.s_test = np.load(log.dir/'data_generation'/'s_test.npy')
        self.a_test = np.load(log.dir/'data_generation'/'a_test.npy')
        file_name = str(delta)+aggregation_method+'aggregated_s_test.npy'
        self.aggregated_s_test = np.load(log.dir/'train'/file_name)

        self.n = 1+max(self.aggregated_s)
        self.q = np.ones(shape = (self.n, self.n_a))
        self.r_f = r_f
        
        
    def get_r(self,b, theta):
        self.r = self.r_f(b, theta, self.s)
        return self.r
        

    def get_q(self, n_steps, alpha):
        for i in range(n_steps):
            for j in range(len(self.s)-1):
                action = self.a[j,0]
                s = self.aggregated_s[j]
                if action== 0:
                    reward = 0
                else:
                    reward = self.r[j]
                s_next = self.aggregated_s[j+1]
                v_next = np.log(np.exp(self.q[s_next, :]).sum())
                self.q[ s, action] = self.q[ s, action] + alpha*(reward + self.discount*v_next -self.q[ s, action])
        
        a = np.load(self.log.dir/'data_generation'/'a.npy')

        try:
            q = np.load(self.log.dir/'data_generation'/'q.npy')
            p = np.load(self.log.dir/'data_generation'/'p.npy')
        except:
            q = np.zeros((len(a),self.n_a))
            p = np.zeros((len(a),self.n_a))
        
        q_est = np.ones(len(a))
        q_est_matrix = np.ones((len(a), self.n_a))
        q_target = np.ones(len(a))
        p_target = np.ones(len(a))
        for i in range(len(a)):
            q_est[i] = self.q[self.aggregated_s[i], a[i]]
            q_target[i] = q[i, a[i]]
            p_target[i] = p[i, a[i]]
            q_est_matrix[i,:] = self.q[self.aggregated_s[i],:]
        p_est = q_est - np.log(np.exp(q_est_matrix).sum(1))
        
        re_dict = {'MSE for the current Q':((q_target-q_est)**2).mean(), 
            'MSE for the current likelihood':((p_target-p_est)**2).mean()}
        return re_dict

    def get_p(self):
        p = self.q-np.log(np.exp(self.q).sum(1)).reshape(len(self.q),1)
        return (p[self.aggregated_s[0:(len(self.aggregated_s))], self.a[:,0]]).mean()

    def get_p_test(self):
        p = self.q-np.log(np.exp(self.q).sum(1)).reshape(len(self.q),1)
        return (p[self.aggregated_s_test[0:(len(self.aggregated_s_test))], self.a_test[:,0]]).mean()



if __name__ == "__main__":
    nIterations = 5000
    thetaL = 1.8+np.arange(60)*0.01
    deltaL = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1]
    testEnv = iDDMRunner(discount=0.8, thetaList = thetaL, dataId = None)
    testEnv.runAll(squashMethod='', deltaList = deltaL, nIterations = nIterations)
    
    

    