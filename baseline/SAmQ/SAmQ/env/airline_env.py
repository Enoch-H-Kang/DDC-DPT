# Environment for the bus engine. 

# Only two variables are used for reward. 
# Transition is uniform. 

import gym
from gym import spaces
import numpy as np
import random
from math import floor
from math import exp
import pandas as pd

import pickle
import os

class airline_env():

    def __init__(self,  data_path, carr_id, r_f, log_path):

        self.r_f = r_f
        rawData13 = pd.read_csv(data_path/'airline13.csv')
        rawData14 = pd.read_csv(data_path/'airline14.csv')
        rawData15 = pd.read_csv(data_path/'airline15.csv')


        data_all = pd.concat([rawData13])
        if carr_id != 100:
            rawData13 = rawData13[rawData13['CarrID'].to_numpy()==carr_id]
            rawData14 = rawData14[rawData14['CarrID'].to_numpy()==carr_id]
            rawData15 = rawData15[rawData15['CarrID'].to_numpy()==carr_id]   

        
        s13 = rawData13.drop(columns = ['csa_code_origin', 'csa_code_dest','CarrID','Segid', 'yearq','Unnamed: 0',  
            'fstseats', 'busseats', 'ecoseats','scheduled_aircraft_max_take_off_weight','scheduled_flights',
            'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles','capacity_t',
            'pop_origin','pop_dest','capacity_tplus1','seg_entry_tplus1']).to_numpy()
        s14 = rawData14.drop(columns = ['csa_code_origin', 'csa_code_dest','CarrID','Segid', 'yearq','Unnamed: 0',  
            'fstseats', 'busseats', 'ecoseats','scheduled_aircraft_max_take_off_weight','scheduled_flights',
            'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles','capacity_t',
            'pop_origin','pop_dest','capacity_tplus1','seg_entry_tplus1']).to_numpy()
        s15 = rawData15.drop(columns = ['csa_code_origin', 'csa_code_dest','CarrID','Segid', 'yearq','Unnamed: 0',  
            'fstseats', 'busseats', 'ecoseats','scheduled_aircraft_max_take_off_weight','scheduled_flights',
            'scheduled_aircraft_hold_volume', 'scheduled_aircraft_range_stat_miles','capacity_t',
            'pop_origin','pop_dest','capacity_tplus1','seg_entry_tplus1']).to_numpy()
        a13 = rawData13[['seg_entry_tplus1']].astype(int).to_numpy()
        a14 = rawData13[['seg_entry_tplus1']].astype(int).to_numpy()
        a15 = rawData13[['seg_entry_tplus1']].astype(int).to_numpy()
        
        s = np.vstack((s13, s14))
        s_next = np.vstack((s14, s15))
        a = np.vstack((a13, a14))
        s_test = s15
        a_test = a15

        path = log_path / 'data_generation'
        os.mkdir(path)
        np.save(path/'s.npy', s)
        np.save(path/'s_next.npy', s_next)
        np.save(path/'a.npy', a)
        np.save(path/'a_test.npy', a_test)
        np.save(path/'s_test.npy', s_test)
        

        self.n_a = 2
        self.d_s = s.shape[1]
        self.n_states = len(s)
        
