import numpy as np
from scipy.special import logsumexp
import time
import pandas as pd

# Set the seed using current time to ensure randomness
np.random.seed(int(time.time()))
#np.random.seed(1234)

#path is current directory + data/
path = ""+ "data/"
# Define parameters
beta = 0.95
gamma = np.euler_gamma
theta1 = 1
theta2 = 5

Horizon = 100
Bustotal = 50  #5000 transitions, because each bus has 100 transitions

# Set up state-space
xmax = 10
states = np.arange(xmax + 1) # generates an array starting from 0 up to xmax, inclusive




def get_util(theta, states):
    #states is an array starting from 0 up to xmax, inclusive
    #theta = [theta1, theta2, theta3]
    u1 = -theta[0] * states  # an array of length s with each element being -theta[0] * s
    u2 = -theta[1] * np.ones(states.shape)  # an array of length s with each element being -theta[1]
    U = np.column_stack((u1, u2))

    return U

def vfi(theta, beta, s):
    U = get_util(theta, s)
    gamma = np.euler_gamma

    Q = np.zeros((len(s), 2)) #row: state, column: action
    dist = 1
    iter = 0
    while dist > 1e-8:
        V = logsumexp(Q, axis=1)
        # Ensure expV corresponds to V but shifts the last element for replacement decision logic
        expV0 = np.zeros_like(V) #Dimension is (states,)
        for i in range(4):  #range(4) is [0,1,2,3]
            Vi = np.append(V[i+1:], [V[-1]] * (i+1))  # Transition to s+i+1 with prob 1/4, last state repeats for boundary
            expV0 += Vi / 4
        
        Q1 = np.zeros_like(Q)  # initialize Q
        
        # Compute value function for maintenance (not replacing)
        Q1[:, 0] = U[:, 0] + beta * expV0  # stochastic transitiion to state s+1, s+2, s+3, s+4 with prob 1/4 each
        
        # Compute value function for replacement
        expV1 = V[1]*np.ones_like(V) #Dimension is (states,). When replaced, the mileage is 1
        Q1[:, 1] = U[:, 1]+ beta * expV1
        expV = np.column_stack((expV0, expV1)) #Dimension is (states, actions)
        dist = np.linalg.norm(Q1 - Q)
        Q=Q1
        iter += 1

    return Q, expV


Vtil, expV = vfi([theta1, theta2], beta, states)



def gen_data(Vtil, expV):
    EP = np.exp(Vtil[:, 1]) / (np.exp(Vtil[:, 0]) + np.exp(Vtil[:, 1])) 
    data = []
    data_test = [] 

    for bus in range(Bustotal):

        mileage = 1
       

        for t in range(Horizon):
            d = np.random.binomial(1, EP[mileage]) #true decision
            data.append([bus, t, 0, mileage, Vtil[mileage, 0], int(d == 0)])
            data.append([bus, t, 1, mileage, Vtil[mileage, 1], int(d == 1)])
            
            data_test.append([bus, t, 0, mileage, int(d == 0)])
            data_test.append([bus, t, 1, mileage, int(d == 1)])
            
            #if not d==1, +1, +2, +3, +4 mileage with 1/4 probability each
            mileage = 1 if d == 1 else min(np.random.choice([mileage + 1, mileage + 2, mileage + 3, mileage + 4], p=[0.25, 0.25, 0.25, 0.25]), xmax)

    df = pd.DataFrame(data, columns=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
    df_test = pd.DataFrame(data_test, columns=['v1', 'v2', 'v3', 'v4', 'v5'])
    
    return df, df_test

df, df_test = gen_data(Vtil, expV)
df.to_csv(path + 'data.csv', index=False, header = False)
df.to_csv(path + 'data.txt', sep=',', index=False, header = False)

df_test.to_csv(path + f'data_bus{Bustotal}_test.txt', sep=',', index=False, header = False)
