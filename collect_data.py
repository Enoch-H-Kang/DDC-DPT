import argparse
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from IPython import embed

import common_args
from envs import Zurcher_env
from utils import (
    build_Zurcher_data_filename
)


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


def rand_pos_and_dir(env):
    pos_vec = np.random.uniform(0, env.size, size=3)
    pos_vec[1] = 0.0
    dir_vec = np.random.uniform(0, 2 * np.pi)
    return pos_vec, dir_vec





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
                'context_states': context_states[:k],
                'context_actions': context_actions[:k],
                'context_next_states': context_next_states[:k],
                'context_rewards': context_rewards[:k],
                'busType': env.type,
            }

            trajs.append(traj)

    return trajs



if __name__ == '__main__':
    #python3 collect_data.py --env Zurcher --Bustotal 100 --beta 0.95 --theta [1,2,9] --H 100 --maxMileage 200 --numTypes 4 --extrapolation False
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser() #creates an ArgumentParser object
    common_args.add_dataset_args(parser) #define what command-line arguments your script can accept
    args = vars(parser.parse_args()) #you parse the command line. 
                                     #This converts the command-line input into a dictionary of options that you can use in your script.
    print("Args: ", args)

    env = args['env']
    beta = args['beta']
    theta = args['theta']
    horizon = args['H']
    Bustotal = args['Bustotal']
    xmax = args['maxMileage']
    numTypes = args['numTypes']
    extrapolation = args['extrapolation']
    

    config = {
        'horizon': horizon,
    }

    # Main data collection part
    
    if env == 'Zurcher':

        config.update({'Bustotal': Bustotal, 'maxMileage': xmax,'theta': theta, 'beta': beta, 'xmax': xmax, 'numTypes': numTypes, 'extrapolation': extrapolation,'rollin_type': 'uniform'})
        #We use uniform for RL objective. For IRL objective, we use expert.
        
        bus_types = np.random.choice(numTypes, Bustotal) #for n_envs number of environments, randomly choose a bus type from numTypes
        train_test_split = int(.8 * Bustotal) #Calculates index that splits train/test data
        train_buses = bus_types[:train_test_split] 
        test_buses = bus_types[train_test_split:]

        if extrapolation == 'False':
            eval_buses = np.random.choice(numTypes, 100)
            
        else:
            raise NotImplementedError
        
        train_trajs = generate_Zurcher_histories(train_buses, **config)
        test_trajs = generate_Zurcher_histories(test_buses, **config)
        eval_trajs = generate_Zurcher_histories(eval_buses, **config)

        train_filepath = build_Zurcher_data_filename(
            env, Bustotal, config, mode=0)
        test_filepath = build_Zurcher_data_filename(
            env, Bustotal, config, mode=1)
        eval_filepath = build_Zurcher_data_filename(env, 100, config, mode=2)

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
