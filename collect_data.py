import argparse
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from IPython import embed

from envs import Zurcher_env
from utils import (
    build_Zurcher_data_filename
)

def str_to_float_list(arg):
    return [float(x) for x in arg.strip('[]').split(',')]

def add_dataset_args(parser):
    
    parser.add_argument("--env", type=str, required=True, help="Environment")
    
    parser.add_argument("--bustotal", type=int, required=False,
                        default=100, help="Total number of buses")

    parser.add_argument("--beta", type=float, required=False,
                        default=0.95, help="Beta")
    parser.add_argument("--theta", type=str_to_float_list, required=False, default=[1, 2, 9], help="Theta values as a list of floats")
    
    parser.add_argument("--H", type=int, required=False,
                        default=100, help="Context horizon")
    
    parser.add_argument("--maxMileage", type=int, required=False,
                        default=200, help="Max mileage")
    parser.add_argument("--numTypes", type=int, required=False,
                        default=10, help="Number of bus types")
    parser.add_argument("--extrapolation", type=str, required=False,
                        default='False', help="Extrapolation")
    


def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards



def generate_Zurcher_histories(buses, theta, beta, horizon, xmax, rollin_type, **kwargs):
    envs = [Zurcher_env.ZurcherEnv(theta, beta, horizon, xmax, busType) for busType in buses]
    
    trajs = []
    for env in tqdm(envs):
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = rollin_mdp(env, rollin_type=rollin_type)
        for k in range(len(context_states)):  
            query_state = context_states[k]
            optimal_action = env.opt_action(query_state)
            traj = {
                'query_state': query_state,
                'optimal_action': optimal_action,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
                'busType': env.type,
            }

            trajs.append(traj)

    return trajs



if __name__ == '__main__':
    #python3 collect_data.py --env Zurcher --bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --maxMileage 200 --numTypes 4 --extrapolation False
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser() #creates an ArgumentParser object
    add_dataset_args(parser) #define what command-line arguments your script can accept
    args = vars(parser.parse_args()) #you parse the command line. 
                                     #This converts the command-line input into a dictionary of options that you can use in your script.
    print("Args: ", args)

    env = args['env']
    beta = args['beta']
    theta = args['theta']
    horizon = args['H']
    bustotal = args['bustotal']
    xmax = args['maxMileage']
    numTypes = args['numTypes']
    extrapolation = args['extrapolation']
    
    n_eval = 100

    config = {
        'horizon': horizon,
        'n_eval': n_eval,
    }

    # Main data collection part
    
    if env == 'Zurcher':

        config.update({'bustotal': bustotal, 'maxMileage': xmax,'theta': theta, 'beta': beta, 'xmax': xmax, 
                       'numTypes': numTypes, 'extrapolation': extrapolation,'rollin_type': 'expert'})
        #We use uniform for RL objective. For IRL objective, we use expert.
        
        bus_types = np.random.choice(numTypes, bustotal) #for n_envs number of environments, randomly choose a bus type from numTypes
        train_test_split = int(.8 * bustotal) #Calculates index that splits train/test data
        train_buses = bus_types[:train_test_split] 
        test_buses = bus_types[train_test_split:]

        
        
        if extrapolation == 'False':
            eval_buses = np.random.choice(numTypes, n_eval)
            
        else:
            raise NotImplementedError
        
        train_trajs = generate_Zurcher_histories(train_buses, **config)
        test_trajs = generate_Zurcher_histories(test_buses, **config)
        eval_trajs = generate_Zurcher_histories(eval_buses, **config)

        train_filepath = build_Zurcher_data_filename(
            env, config, mode=0)
        test_filepath = build_Zurcher_data_filename(
            env, config, mode=1)
        eval_filepath = build_Zurcher_data_filename(env, config, mode=2)

    else:
        raise NotImplementedError


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

