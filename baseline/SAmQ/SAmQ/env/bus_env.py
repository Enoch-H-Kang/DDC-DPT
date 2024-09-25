# Environment for the bus engine. 

# Only two variables are used for reward. 
# Transition is uniform. 

import gym
from gym import spaces
import numpy as np
import random
from math import floor
from math import exp
import pickle
import os

class syn_bus_env():

    def __init__(self,  n_a,  d_s, b, theta, r_f):

        self.n_a = n_a
        self.t = 0
        self.d_s = d_s
        self.s = np.zeros(d_s)
        self.b = b
        self.theta = theta
        self.r_f = r_f
        self.fake_state_index = (np.array(theta)==0) #theta = [1,0,0]. Therefore fake_state_index = [False, True, True]
    def reward_fn(self, s, a):        
        if a==0:
            return 0
        else:
            return self.r_f(self.b, self.theta, s)
    def transit_fn(self,s,a):        
        if a==0:
            s_next = s*0
        else:
            s_next = s + np.random.uniform(1,2,self.d_s) # True state updated by uniform.
        s_next[self.fake_state_index] = np.random.uniform(-5,5,np.sum(self.fake_state_index)) # Fake state updated by more noise. 
        return s_next
    def reset(self):
        self.t = 0
        self.s = np.zeros(self.d_s)
        self.s[self.fake_state_index] = np.random.uniform(-5,5,np.sum(self.fake_state_index)) # Fake state reset with noise. 
        return 
    def step(self, a):
        reward = self.reward_fn(self.s, a)
        self.s = self.transit_fn(self.s,a)
        return reward
    def get_data(self, N, policy = None):
        self.reset()
        re = dict(a=[], s=[self.s], r=[])
        for _ in range(N):
            if policy is None:
                temp_a = np.random.randint(0, self.n_a)
            else:
                tmep_a = policy(s)
            re['r'].append(self.step(temp_a))
            re['s'].append(self.s)
            re['a'].append(temp_a)
        return re
