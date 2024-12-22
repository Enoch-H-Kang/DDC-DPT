import argparse
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
from scipy.special import logsumexp
import json

class Environment(object):
    def __init__(self, env, num_trajs, H, beta, theta, rollin_type):
        self.H = H
        self.beta = beta
        self.num_trajs = num_trajs
        self.env = env
        self.theta = theta
        
        self.rollin_type = rollin_type
        self.current_step = 0

class ZurcherEnv(Environment):
    def __init__(self, type, **config):
        super().__init__(
            env=config['env'],
            num_trajs=config['num_trajs'],
            H=config['H'],
            beta=config['beta'],
            theta=config['theta'],
            rollin_type=config['rollin_type']
        )
        self.env_name = 'Zurcher'        
        self.xmax = config['maxMileage']
        self.states = np.arange(self.xmax+1) #dimension is (states,), where states are 0,1,2,...,xmax
        self.numTypes = config['numTypes']
        self.type = type
        self.current_step = 0
        self.EP, self.Q, self.expV = self.calculate_EP_Q()
        self.U = self._get_util()
        self.count = 0
        self.num_dummies = config['num_dummies']
        self.dummy_dim = config['dummy_dim']
    
    def build_filepaths(self, mode):
        """
        Builds the filename for the Zurcher data.
        Mode is either 'train', 'test', or 'eval'.
        """
        filename_template = 'datasets/trajs_{}.pkl'
        filename = (f"{self.env}_num_trajs{self.num_trajs}"
                    f"_dummies{self.num_dummies}x{self.dummy_dim}"
                    f"_beta{self.beta}_theta{self.theta}"
                    f"_numTypes{self.numTypes}_H{self.H}_{self.rollin_type}")
        
        filename += f'_{mode}'
        
        return filename_template.format(filename)
        
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
        Q = np.zeros((len(self.states), 2)) #Dimension is (states, actions), which is (xmax+1, 2)
        dist = 1
        iter = 0
        while dist > tol:
            ##V = gamma + logsumexp(Q, axis=1) #dimension is (stats,), which is (xmax+1,). This is V(s)
            V = logsumexp(Q, axis=1)
            #expV0 = np.append(V[1:], V[-1])  # Transition to s+1 with prob 1, last state repeats for boundary
            
            
            
            
            expV0 = np.zeros_like(V) #Dimension is (states,)
            for i in range(4):  #range(4) is [0,1,2,3]
                Vi = np.append(V[i+1:], [V[-1]] * (i+1))  # Transition to s+i+1 with prob 1/4, last state repeats for boundary
                expV0 += Vi / 4
            
            Q1 = np.zeros_like(Q)  # initialize Q
            
            # Compute value function for maintenance (not replacing)
            Q1[:, 0] = U[:, 0] + self.beta * expV0  # stochastic transitiion to state s+1, s+2, s+3, s+4 with prob 1/4 each
            
            # Compute value function for replacement
            expV1 = V[1]*np.ones_like(V) #Dimension is (states,). When replaced, the mileage is 1
            #Q1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV1  # deterministic transition to state 1. Type is 0 (default setting)
            Q1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV1
            expV = np.column_stack((expV0, expV1)) #Dimension is (states, actions)
            dist = np.linalg.norm(Q1 - Q)
            Q=Q1
            iter += 1
            
        return Q, expV
    
    def calculate_EP_Q(self):
        '''
        Returns EP=Expert Policy
        '''
        Q, expV = self.vfi()
        EP1 = np.exp(Q[:, 1]) / (np.exp(Q[:, 0]) + np.exp(Q[:, 1])) 
        EP = np.array([1-EP1, EP1]).T #self.EP's dimension is (states, actions)
        #given states being a vector of states, self.EP[states] is a matrix of optimal choice probability vectors at states with dimension (states, actions)
        return EP, Q, expV

    def sample_state(self):
        return np.random.randint(0, self.xmax+1)

    def sample_action(self):
        #choose either 0 or 1
        randchoice = random.choice([[0, 1], [1, 0]])
        return randchoice
    
    def reset(self):
        self.current_step = 0
        self.state = 0
        return self.state

    def transit(self, state, action_prob):
        #action_prob is an two-dimensional array of probs, where the first element is
        # the prob of action 0 and the second element is the prob of action 1
        action = np.random.choice([0, 1], p=action_prob)
        if action == 0:
            #next state is sampled among 1,2,3,4 + current state
            next_state = min(state + np.random.randint(1, 5), self.xmax) #transit to s+1, s+2, s+3, s+4 with prob 1/4 each, respecting the boundary
            #next_state = min(state + 1, self.xmax) #transit to s+1 with prob 1
        
        elif action == 1:
            next_state = 1
    
        else: 
            raise ValueError("Invalid action")
        self.state = next_state
        return next_state, action

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        return self.EP[state]




def rollin_mdp(env, n_dummies, dummy_dim, rollin_type):
    full_states = []
    actions = []
    full_next_states = []

    state = env.reset()
    full_state = np.concatenate(([state],np.random.randint(-dummy_dim,dummy_dim+1,n_dummies)))
    #full_state is a vector of states and dummies, where the first element is the mileage and the rest are dummies
    for _ in range(env.H): #Remember: I made H+1 to H.
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
        
        full_states.append(full_state)
        actions.append(action)
        full_next_state = np.concatenate(([next_state],np.random.randint(-dummy_dim,dummy_dim+1,n_dummies)))
        full_next_states.append(full_next_state)
        
        state = next_state
        full_state = full_next_state

    full_states = np.array(full_states) #index 0 to H-1 are context states, index H is the first query state
    actions = np.array(actions) #index 0 to H-1 are context actions, index H is the first query action
    full_next_states = np.array(full_next_states) #index H-1 is the first query state, index H is the second query state
    return full_states, actions, full_next_states

def generate_Zurcher_histories(config):
    
    num_Types = config['numTypes']
    num_trajs = config['num_trajs']
    n_dummies = config['num_dummies']
    dummy_dim = config['dummy_dim']
    
    Types = np.random.choice(range(num_Types), num_trajs)
    envs = [ZurcherEnv(type = Type, **config) for Type in Types]
    
    trajs = []
    for env in tqdm(envs):
        (
            full_states, #dimension is (H, state_dim)
            actions, #dimension is (H,)
            full_next_states, #dimension is (H, state_dim)
        ) = rollin_mdp(env, n_dummies, dummy_dim, rollin_type=config['rollin_type'])
        #revert full_states to states
        states = full_states[:,0] #Recover back states without dummies. Dimension is (H,)
        next_states = full_next_states[:,0] #Recover back next_states without dummies
       
        states_true_EPs = env.EP[states] #True EP at the query state 1, of dimension 2
        states_true_Qs = env.Q[states] #True Q at the query state 1, of dimension 2
        states_true_expVs = env.expV[states] #True E[V(s',a')|s,a] at the query state 1, of dimension 2
        
        next_states_true_EPs = env.EP[next_states] #True EP at the query state 2, of dimension 2
        next_states_true_Qs = env.Q[next_states] #True Q at the query state 2, of dimension 2
        
        traj = {
            'states': full_states, #dimension is (H,) (if state dim is not 1, then it is (H, state_dim))
            'actions': actions, #dimension is (H,)
            'next_states': full_next_states, #dimension is (H,)
            'busType': env.type, #dimension is (1,)
            'states_true_EPs': states_true_EPs, #True EP at the query state 1, of dimension 2
            'next_states_true_EPs': next_states_true_EPs, #True EP at the query state 2, of dimension 2
            'states_true_Qs': states_true_Qs, #True Q at the query state 1, of dimension 2
            'next_states_true_Qs': next_states_true_Qs, #True Q at the query state 2, of dimension 2
            'states_true_expVs': states_true_expVs, #True E[V(s',a')|s,a] at the query state 1, of dimension 2
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
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    env = config['env']
    if env != 'zurcher':
        raise NotImplementedError("Only Zurcher environment is implemented")
    
    num_train_trajs = int(0.8 * config['num_trajs'])
    config_train = {**config, 'num_trajs': num_train_trajs}
    config_test = {**config, 'num_trajs': config['num_trajs'] - num_train_trajs}
    
    env_instance = ZurcherEnv(type = 0, **config)
    train_filepath = env_instance.build_filepaths('train')
    test_filepath = env_instance.build_filepaths('test')

    if os.path.exists(train_filepath) and os.path.exists(test_filepath):
        print(f"Data files already exist for the current configuration:")
        print(f"Train file: {train_filepath}")
        print(f"Test file: {test_filepath}")
        print("Skipping data generation.")
        return

    print("Generating new data...")
    train_trajs = generate_Zurcher_histories(config_train)
    test_trajs = generate_Zurcher_histories(config_test)
   

    
    
    

    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)

    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    generate(config)
 
    