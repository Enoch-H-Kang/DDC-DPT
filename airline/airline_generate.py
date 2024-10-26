import argparse
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import json

class AirlineEnv:
    def __init__(self, carr_id):
        self.current_step = 0
        
        # Load the data for 2013, 2014, 2015 from the airline_data folder
        data_path = os.path.join(os.path.dirname(__file__), 'airline_data')
        rawData13 = pd.read_csv(os.path.join(data_path, 'airline13.csv'))
        rawData14 = pd.read_csv(os.path.join(data_path, 'airline14.csv'))
        rawData15 = pd.read_csv(os.path.join(data_path, 'airline15.csv'))
        rawData15_hard = pd.read_csv(os.path.join(data_path, 'airline15_hard.csv'))
        
        # Filter data by carrier ID
        if carr_id != 100:
            rawData13 = rawData13[rawData13['CarrID'] == carr_id]
            rawData14 = rawData14[rawData14['CarrID'] == carr_id]
            rawData15 = rawData15[rawData15['CarrID'] == carr_id]
            rawData15_hard = rawData15_hard[rawData15_hard['CarrID'] == carr_id]
        
        # Define state variables by dropping unnecessary columns
        self.s13 = rawData13.drop(columns=['csa_code_origin', 'csa_code_dest', 'CarrID', 
                                           'Segid', 'yearq', 'Unnamed: 0',
                                           'fstseats', 'busseats', 'ecoseats', 'scheduled_aircraft_max_take_off_weight',
                                           'scheduled_flights', 'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles',
                                           'capacity_t', 
                                           #'pop_origin', 'pop_dest', 
                                           'capacity_tplus1', 'seg_entry_tplus1']).to_numpy()
        
        self.s14 = rawData14.drop(columns=['csa_code_origin', 'csa_code_dest', 'CarrID', 
                                           'Segid', 'yearq', 'Unnamed: 0',
                                           'fstseats', 'busseats', 'ecoseats', 'scheduled_aircraft_max_take_off_weight',
                                           'scheduled_flights', 'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles',
                                           'capacity_t',
                                           #'pop_origin', 'pop_dest',
                                           'capacity_tplus1', 'seg_entry_tplus1']).to_numpy()
        self.s15 = rawData15.drop(columns=['csa_code_origin', 'csa_code_dest','CarrID', 
                                           'Segid', 'yearq', 'Unnamed: 0',
                                           'fstseats', 'busseats', 'ecoseats', 'scheduled_aircraft_max_take_off_weight',
                                           'scheduled_flights', 'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles',
                                           'capacity_t', 
                                           #'pop_origin', 'pop_dest', 
                                           'capacity_tplus1', 'seg_entry_tplus1']).to_numpy()
        self.s15_hard = rawData15_hard.drop(columns=['csa_code_origin', 'csa_code_dest','CarrID', 
                                           'Segid', 'yearq', 'Unnamed: 0',
                                           'fstseats', 'busseats', 'ecoseats', 'scheduled_aircraft_max_take_off_weight',
                                           'scheduled_flights', 'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles',
                                           'capacity_t', 
                                           #'pop_origin', 'pop_dest', 
                                           'capacity_tplus1', 'seg_entry_tplus1']).to_numpy()
        # Define actions
        self.a13 = rawData13[['seg_entry_tplus1']].astype(int).to_numpy()
        self.a14 = rawData14[['seg_entry_tplus1']].astype(int).to_numpy()
        self.a15 = rawData15[['seg_entry_tplus1']].astype(int).to_numpy()
        self.a15_hard = rawData15_hard[['seg_entry_tplus1']].astype(int).to_numpy()
        
        # Stack states and actions to create training and test sets
        
def build_filepaths(carr_id, mode):
    """
    Builds the filename for the airline data.
    Mode is either 'train' or 'test'.
    """
    filename_template = 'datasets/airline_carr_id_{}_{}.pkl'
    filename = filename_template.format(carr_id, mode)
    return filename


def generate_airline_histories(carr_id, mode='train'):
    env = AirlineEnv(carr_id)    
    print(f"env.s13.shape: {env.s13.shape}")
    
    if mode == 'train': 
        # shape of env.s13 is (n, 10) where n is the number of rows in the dataset
        s13_expanded = np.expand_dims(env.s13, axis=1)  # shape is (n, 1, 10)
        s14_expanded = np.expand_dims(env.s14, axis=1)  # shape is (n, 1, 10)
        states = np.concatenate((s13_expanded, s14_expanded), axis=1)  # shape is (n, 2, 10)
        
        a13_expanded = np.expand_dims(env.a13, axis=1)  # shape is (n, 1, 1)
        a14_expanded = np.expand_dims(env.a14, axis=1)  # shape is (n, 1, 1)
        actions = np.concatenate((a13_expanded, a14_expanded), axis=1)  # shape is (n, 2, 1)
        
        next_states = np.expand_dims(env.s15, axis=1)  # shape is (n, 1, 10)
        
        trajs = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
        }
    elif mode == 'test':
        states = np.expand_dims(env.s15, axis=1)  # shape is (n, 1, 10)
        actions = np.expand_dims(env.a15, axis=1)  # shape is (n, 1, 1)
        next_states = np.zeros_like(states)  # shape is (n, 1, 10)
        trajs = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
        }
    else: #mode == 'hard_test'
        states = np.expand_dims(env.s15_hard, axis=1)  # shape is (n, 1, 10)
        actions = np.expand_dims(env.a15_hard, axis=1)
        next_states = np.zeros_like(states)
        trajs = {
            'states': states,
            'actions': actions,
            'next_states': next_states,
        }
        
    return trajs


def generate():
    
    print("Generating new data...")
    data_path = os.path.join(os.path.dirname(__file__), 'airline_data')
    rawData13 = pd.read_csv(os.path.join(data_path, 'airline13.csv'))
    unique_carr_ids = rawData13['CarrID'].unique() #type of unique_carr_ids is numpy.ndarray with shape (12,)
    
    test = 0
    for carr_id in tqdm(unique_carr_ids):
        
        trajs_train = generate_airline_histories(carr_id, mode='train')
        trajs_test = generate_airline_histories(carr_id, mode='test')
        trajs_hard_test = generate_airline_histories(carr_id, mode='hard_test')
        
        if not os.path.exists('datasets'):
            os.makedirs('datasets', exist_ok=True)
        train_filepath = build_filepaths(carr_id,'train')
        test_filepath = build_filepaths(carr_id,'test')
        test_hard_filepath = build_filepaths(carr_id,'hard_test')
        
        with open(train_filepath, 'wb') as file:
            pickle.dump(trajs_train, file)
        print(f"Saved to {train_filepath}.")

        with open(test_filepath, 'wb') as file:
            pickle.dump(trajs_test, file)
        print(f"Saved to {test_filepath}.")
        
        with open(test_hard_filepath, 'wb') as file:
            pickle.dump(trajs_hard_test, file)
        print(f"Saved to {test_hard_filepath}.")


if __name__ == "__main__":

    generate()
