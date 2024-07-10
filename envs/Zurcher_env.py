import itertools

import gym
import numpy as np
import torch
from scipy.special import logsumexp

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ZurcherEnv(BaseEnv):
    def __init__(self, theta, beta, horizon, xmax, type):
        self.theta = theta
        self.beta = beta
        self.horizon = horizon
        self.xmax = xmax
        self.states = np.arange(self.xmax+1)
        self.type = type #Type can be 0, 1, 2... numTypes-1
        self.current_step = 0
        self.EP = self.calculate_EP()
        
    def get_util(self):
        theta1 = self.theta[0]
        theta2 = self.theta[1]
        theta3 = self.theta[2]
        u1 = -theta1 * self.states  # an array of length s with each element being -theta[0] * s
        u2 = -theta2 * np.ones(self.states.shape)  # an array of length s with each element being -theta[1]
        u3 = -theta3 * np.ones(self.states.shape)  # an array of length s with each element being -theta[2]
        U = np.column_stack((u1, u2, u3))
        # an example of a row of U is [-5, -9, -4.5], 
        # where the first column is the cost of period maintenance at mileage 5, 
        # the second column is the cost of replacement, and the third column is the cost of mileage
        return U
    
    def vfi(self):
        U = self.get_util()
        gamma = np.euler_gamma
        Vtil = np.zeros((len(self.states), 2)) #row: state, column: action
        dist = 1
        iter = 0
        while dist > 1e-8:
            V = gamma + logsumexp(Vtil, axis=1)
            # Ensure expV corresponds to V but shifts the last element for replacement decision logic
            expV = np.append(V[1:], V[-1])  # As mileage does not increase after it reaches max, the last element is repeated
            Vtil1 = np.zeros_like(Vtil)  # initialize Vtil1
            # Compute value function for maintenance (not replacing)
            Vtil1[:, 0] = U[:, 0] + self.beta * expV  # action-specific value function of not replacing
            # Compute value function for replacement
            Vtil1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV[0]  # action-specific value function of replacing
        
            
            dist = np.linalg.norm(Vtil1 - Vtil)
            Vtil = Vtil1
            iter += 1

        return Vtil, expV
    
    def calculate_EP(self):
        Vtil = self.vfi()[0]
        EP = np.exp(Vtil[:, 1]) / (np.exp(Vtil[:, 0]) + np.exp(Vtil[:, 1])) 
        return EP

    def sample_state(self):
        return np.random.randint(0, self.xmax+1)

    def sample_action(self):
        #choose either 0 or 1
        return np.random.choice([0, 1])

    def reset(self):
        self.current_step = 0
        self.state = 0
        return self.state

    def transit(self, state, action):
        action = np.argmax(action) # convert one-hot to integer
        state = np.array(state) #
        if action == 0:
            state = min(state + 1, self.xmax)
            reward = -self.theta[0] * state - self.type*self.theta[1]
        elif action == 1:
            state = 1
            reward = -self.theta[2]
        else: 
            raise ValueError("Invalid action")

        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        return self.EP[state]



class DarkroomEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews
