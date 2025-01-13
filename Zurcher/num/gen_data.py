import numpy as np
from scipy.special import logsumexp
import time
import pandas as pd

# Set the seed using current time to ensure randomness
np.random.seed(int(time.time()))
#np.random.seed(1234)

#path = "/home/ehwkang/DDC_DPT/Zurcher/num/data/"
#path is current directory + data/
path = ""+ "data/"
# Define parameters
beta = 0.95
gamma = np.euler_gamma
theta1 = 1
theta2 = 5

Horizon = 100
Bustotal = 1000

# Set up state-space
xmax = 200
states = np.arange(xmax + 1) # generates an array starting from 0 up to xmax, inclusive




def get_util(theta, states):
    #states is an array starting from 0 up to xmax, inclusive
    #theta = [theta1, theta2, theta3]
    u1 = -theta[0] * states  # an array of length s with each element being -theta[0] * s
    u2 = -theta[1] * np.ones(states.shape)  # an array of length s with each element being -theta[1]
    U = np.column_stack((u1, u2))
    # an example of a row of U is [-5, -9, -4.5], 
    # where the first column is the cost of period maintenance at mileage 5, 
    # the second column is the cost of replacement, and the third column is the cost of mileage
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
        #Q1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV1  # deterministic transition to state 1. Type is 0 (default setting)
        Q1[:, 1] = U[:, 1]+ beta * expV1
        expV = np.column_stack((expV0, expV1)) #Dimension is (states, actions)
        dist = np.linalg.norm(Q1 - Q)
        Q=Q1
        iter += 1

    return Q, expV

#The following function gen_data creates HW4_gen_data.csv with columns v1, v2, v3, v4, v5, v6 with "Bustotal" number or buses' data, 
#where each of bus is with horizon length of "Horizon"
#v1: busnumber that starts from 0 to Bustotal-1
#v2: type of the bus
#v3: timeperiod that starts from 0, 0, 1, 1, ..., to Horizon-1, Horizon-1
#v4: DecisionNo indicates the decision that can be made, so it iterates between 0 and 1
#v5: Mileage of the bus, which starts from 0
#v6: Value function associated with the mileage and the decision that can be made
#v7: If true Decision made by the bus matches the DecisionNo, 1; otherwise, 0

Vtil, expV = vfi([theta1, theta2], beta, states)


#typedist is a vector that has length Bustotal and each element is 0 or 1, where exactly 0.4 portion among Bustotal number of buses are type 0
#For example, if Bustotal is 100, then exactly 40 buses are type 0 and exactly 60 buses are type 1
#np.random.shuffle(typedist)

def gen_data(Vtil, expV):
    EP = np.exp(Vtil[:, 1]) / (np.exp(Vtil[:, 0]) + np.exp(Vtil[:, 1])) 
    #pi is the probability of type 0
    data = []
    data_test = [] #data for HW5 testing

    for bus in range(Bustotal):
        #type = np.random.binomial(1, 1-pi)
        #type = int(typedist[bus])
        mileage = 1
       

        for t in range(Horizon):
            d = np.random.binomial(1, EP[mileage]) #true decision
            data.append([bus, t, 0, mileage, Vtil[mileage, 0], int(d == 0)])
            data.append([bus, t, 1, mileage, Vtil[mileage, 1], int(d == 1)])
            
            data_test.append([bus, t, 0, mileage, int(d == 0)])
            data_test.append([bus, t, 1, mileage, int(d == 1)])
            
            #if not d==1, +1, +2, +3, +4 mileage with 1/4 probability each
            mileage = 1 if d == 1 else np.random.choice([mileage + 1, mileage + 2, mileage + 3, mileage + 4], p=[0.25, 0.25, 0.25, 0.25])

    df = pd.DataFrame(data, columns=['v1', 'v2', 'v3', 'v4', 'v5', 'v6'])
    df_test = pd.DataFrame(data_test, columns=['v1', 'v2', 'v3', 'v4', 'v5'])
    
    return df, df_test

df, df_test = gen_data(Vtil, expV)
df.to_csv(path + 'data.csv', index=False, header = False)
df.to_csv(path + 'data.txt', sep=',', index=False, header = False)

#dataset generation for HW1 testing
#df_test.to_csv(path + 'data_assg1_test.txt', sep=',', index=False, header = False)
df_test.to_csv(path + f'data_bus{Bustotal}_test.txt', sep=',', index=False, header = False)
#the following variable "countType0" counts number of type 0 rows from data4.csv
#countType0 = int(len(df[df['v2'] == 0])/2)
#print(f"countType0: {countType0}")