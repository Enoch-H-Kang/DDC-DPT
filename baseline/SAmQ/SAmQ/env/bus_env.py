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
from scipy.special import logsumexp

class syn_bus_env():

    def __init__(self,  n_a,  d_s, b, theta, r_f, discount):

        self.n_a = n_a
        self.t = 0
        self.d_s = d_s
        self.s = np.zeros(d_s)
        self.b = b
        self.theta = theta
        self.r_f = r_f
        self.beta = discount
        self.xmax = 10
        self.fake_state_index = (np.array(theta)==0) #theta = [1,0,0]. Therefore fake_state_index = [False, True, True]
        self.mileages = np.arange(self.xmax+1) #mileages = [0,1,2,3,4,5,6,7,8,9,10]
        self.EP, self.Q = self.calculate_EP_Q()
        self.U = self._get_util()
        
    def _get_util(self):
        '''
        Input: None
        
        '''
        theta1 = self.theta[0]
        theta2 = self.b
        u1 = -theta1 * self.mileages  # an array of length s with each element being -theta[0] * s
        u2 = -theta2 * np.ones(self.mileages.shape)  # an array of length s with each element being -theta[1]
        U = np.column_stack((u1, u2))
        # an example of a row of U is [-1, -5, -1], 
        # where the first column is the cost of period maintenance at mileage 5, 
        # the second column is the cost of replacement, and the third column is the cost of mileage
        return U
    
    def vfi(self, tol=1e-8):
        '''
        Q-Value iteration. 
        Input: None
        Output: Approximation of Q function. 
        '''
    
        U = self._get_util() #Dimension is (states, actions)
        gamma = np.euler_gamma
        Q = np.zeros((len(self.mileages), 2)) #Dimension is (states, actions)
        dist = 1
        iter = 0
        while dist > tol:
            V = gamma + logsumexp(Q, axis=1) #dimension is (stats,). This is V(s)
            
            #expV = np.append(V[1:], V[-1])  # This is E[V(s')|s,a]. Transition to s+1. As mileage does not increase after it reaches max, the last element is repeated. Dimension is (states,)
            
            expV = np.zeros_like(V) #Dimension is (states,)
            for i in range(4):
                EVi = np.append(V[i+1:], [V[-1]] * (i+1))  # Transition to s+i+1 with prob 1/4, last state repeats for boundary
                expV += EVi / 4
            
            Q1 = np.zeros_like(Q)  # initialize Q
            # Compute value function for maintenance (not replacing)
            Q1[:, 0] = U[:, 0] + self.beta * expV  # stochastic transitiion to state s+1, s+2, s+3, s+4 with prob 1/4 each
            # Compute value function for replacement
            Q1[:, 1] = U[:, 1] + self.beta * V[1]  # deterministic transition to state 1
        
            dist = np.linalg.norm(Q1 - Q)
            Q=Q1
            iter += 1
        return Q
    
    def calculate_EP_Q(self):
        '''
        Returns EP=Expert Policy
        '''
        Q = self.vfi()
        EP1 = np.exp(Q[:, 1]) / (np.exp(Q[:, 0]) + np.exp(Q[:, 1])) 
        EP = np.array([1-EP1, EP1]).T #self.EP's dimension is (states, actions)
        #given states being a vector of states, self.EP[states] is a matrix of optimal choice probability vectors at states with dimension (states, actions)
        return EP, Q
    
    def reward_fn(self, s, a):        
        if a==0:
            return -self.b
        else:
            return self.r_f(self.theta, s)
    
    def opt_action(self, s0): #The optimal decision policy
        #make s0 an integer
        s0 = int(s0)
        return self.EP[s0]
    
    def transit_fn(self,s,a):        
        if a==1:
            s_next = np.ones(self.d_s) # True state updated to 1.
        else:
            s0_next = min(s[0] + np.random.randint(1,5), self.xmax) # largest value of s is xmax
            s_next = np.copy(s)
            s_next[0] = s0_next
        s_next[self.fake_state_index] = np.random.randint(-5,6,np.sum(self.fake_state_index)) # Fake state updated by more noise. 
        return s_next
    
    def reset(self):
        self.t = 0
        self.s = np.zeros(self.d_s) #dimension of s is (d_s,)
        self.s[self.fake_state_index] = np.random.randint(-5,6,np.sum(self.fake_state_index)) # Fake state reset with noise. 
        return 

    
    def step(self, a):
        reward = self.reward_fn(self.s, a)
        self.s = self.transit_fn(self.s,a)
        return reward
    def get_data(self, N, policy):
        self.reset()
        re = dict(a=[], s=[self.s], r=[])
        for _ in range(N):
            if policy == 'expert':
                temp_a = np.random.choice([0,1], p=self.opt_action(self.s[0]))
                re['r'].append(self.step(temp_a))
                re['s'].append(self.s)
                re['a'].append(temp_a)
            else:
                temp_a = np.random.randint(0, self.n_a)
            
        return re
