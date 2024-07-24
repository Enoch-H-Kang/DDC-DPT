import argparse
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
from scipy.special import logsumexp
import utils


class Environment(object):
    def __init__(self, H, beta):
        self.H = H
        self.beta = beta

    def reset(self):
        self.current_step = 0
        self.state = 0
        return self.state
    

    def build_filepaths(self):
        """
        Builds the filename for the Zurcher data.
        Mode is either 0: train, 1: test, 2: eval.
        """
        filename_template = 'datasets/trajs_{}.pkl'
        filename = (f"{self.env}{'_bustotal' + str(self.bustotal)}"
                    f"_beta{self.beta}_theta{self.theta}"
                    f"_numTypes{self.numTypes}_H{self.H}_{self.rollin_type}")
        
        train_filepath += '_train'
        test_filepath += '_test'
        eval_filepath += '_eval'
            
        return filename_template.format(filename)

class ZurcherEnv(Environment):
    def __init__(self, theta, beta, horizon, xmax, type):
        super(ZurcherEnv, self).__init__(horizon, beta)
        self.env_name = 'Zurcher'        
        self.theta = theta
        self.xmax = xmax
        self.states = np.arange(self.xmax+1)
        self.type = type #Type can be 0, 1, 2... numTypes-1
        self.current_step = 0
        self.EP, self.Vtil = self.calculate_EP_Vtil()
        self.U = self._get_util()
        
    def _get_util(self):
        '''
        Input: None
        
        '''
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
    
    def vfi(self, tol=1e-8):
        '''
        Q-Value iteration. 
        Input: None
        Output: Approximation of Q function. 
        '''
        U = self.U
        gamma = np.euler_gamma
        Q = np.zeros((len(self.states), 2)) #row: state, column: action
        dist = 1
        iter = 0
        while dist > tol:
            V = gamma + logsumexp(Q, axis=1)
            # Ensure expV corresponds to V but shifts the last element for replacement decision logic
            expV = np.append(V[1:], V[-1])  # As mileage does not increase after it reaches max, the last element is repeated
            Q1 = np.zeros_like(Q)  # initialize Vtil1
            # Compute value function for maintenance (not replacing)
            Q1[:, 0] = U[:, 0] + self.beta * expV  # action-specific value function of not replacing
            # Compute value function for replacement
            Q1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV[0]  # action-specific value function of replacing
        
            dist = np.linalg.norm(Q1 - Q)
            iter += 1

        return Q
    
    def calculate_EP_Vtil(self):
        '''
        Returns EP=Expert Policy
        '''
        Q = self.vfi()
        EP1 = np.exp(Q[:, 1]) / (np.exp(Q[:, 0]) + np.exp(Q[:, 1])) 
        EP = np.array([1-EP1, EP1]).T.tolist()
        return EP, Q

    def sample_state(self):
        return np.random.randint(0, self.xmax+1)

    def sample_action(self):
        #choose either 0 or 1
        randchoice = random.choice([[0, 1], [1, 0]])
        return randchoice

    def transit(self, state, action_prob):
        #action_prob is an two-dimensional array of probs, where the first element is
        # the prob of action 0 and the second element is the prob of action 1
        action = np.random.choice([0, 1], p=action_prob)
        if action == 0:
            next_state = min(state + 1, self.xmax)
            #reward = -self.theta[0] * state - self.type*self.theta[1]
        elif action == 1:
            next_state = 1
            #reward = -self.theta[2]
        else: 
            raise ValueError("Invalid action")
        self.state = next_state

        return next_state, action

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        return self.EP[state]




def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    #rewards = []

    state = env.reset()
    for _ in range(env.H):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
            
            #action_prob=one-hot encoding of action
            action_prob = np.zeros(2)
            action_prob[action] = 1
            
        elif rollin_type == 'expert':
            action_prob = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, action = env.transit(state, action_prob)
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    
    #construct a concatanated matrix that consists of (state, action, next_state)
    #print(np.column_stack((states, actions, next_states)))
    #exit(0)

    return states, actions, next_states

def generate_Zurcher_histories(buses, theta, beta, H, xmax, rollin_type, **kwargs):
    envs = [ZurcherEnv(theta, beta, H, xmax, busType) for busType in buses]
    
    trajs = []
    for env in tqdm(envs):
        (
            context_states,
            context_actions,
            context_next_states,
        ) = rollin_mdp(env, rollin_type=rollin_type)
        
        query_state = context_next_states[-1] #query_state is the state after the last context state
        query_true_EP = env.opt_action(query_state) #True optimal choice prob vector at the query state
        query_action = np.random.choice([0, 1], p=query_true_EP) #The target action chosen according to true optimal choice prob 
        query_true_Q = env.Vtil[query_state] #Q value at the query state

        
        
        traj = {
            'query_state': query_state,
            'query_action': query_action,
            'context_states': context_states,
            'context_actions': context_actions,
            'context_next_states': context_next_states,
            'busType': env.type,
            'query_true_EP': query_true_EP, #True EP at the query state, of dimension 2
            'query_true_Q': query_true_Q, #True Q at the query state, of dimension 2
        }

        trajs.append(traj)

    return trajs


def save_data(train_trajs, test_trajs, eval_trajs, 
              config, train_filepath, test_filepath, eval_filepath):
    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)

    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")



def generate(config):
    np.random.seed()
    random.seed()

    env = config['env']
    beta = config['beta']
    theta = config['theta']
    H = config['H']
    trajectories = config['generation']['trajectories']
    xmax = config['env_params']['maxMileage']
    numTypes = config['env_params']['numTypes']
    n_eval = config['generation']['eval_trajectories']

    if env == 'Zurcher':
        config.update({'rollin_type': 'expert'})
        
        bus_types = np.random.choice(numTypes, trajectories)
        train_test_split = int(.8 * trajectories)
        train_buses = bus_types[:train_test_split] 
        test_buses = bus_types[train_test_split:]
        eval_buses = np.random.choice(numTypes, n_eval)


        train_trajs = generate_Zurcher_histories(train_buses, **config)
        test_trajs = generate_Zurcher_histories(test_buses, **config)
        eval_trajs = generate_Zurcher_histories(eval_buses, **config)
    else:
        raise NotImplementedError
    
    train_filepath, test_filepath, eval_filepath = env.generate_filepaths

    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")

