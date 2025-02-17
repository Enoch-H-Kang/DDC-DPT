import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv
from scipy.special import logsumexp
import os
import numdifftools as nd
import time
import sys
import time

# Set the seed using current time to ensure randomness
np.random.seed(int(time.time()))
#np.random.seed(1234)

path = ""+ "data/"

# Define parameters
beta = 0.95

# Set up state-space
xmax = 10
states = np.arange(xmax + 1) # generates an array starting from 0 up to xmax, inclusive

def get_util(theta, s):
    u1 = -theta[0] * s  # Period maintenance cost
    u2 = -theta[1] * np.ones(s.shape)  # Replacement cost
    U = np.column_stack((u1, u2))
    return U


def vfi(theta, beta, s):
    U = get_util(theta, s)

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
    return Q

def vfi_expV(theta, beta, s):
    U = get_util(theta, s)

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
    return expV


def CCPandTKEst(data):

    CCPs = {}
    choicestorage = {} #choicestorage is a dictionary of counts of the occurances of each state-action pair

    for i in range(data.shape[0]):
        current_state = data.iloc[i, 3] #current state is the mileage of the current row
        #CCPsCount[current_state, data.iloc[i, 4]] += 1 
        if current_state not in choicestorage:
            choicestorage[current_state] = [0,0]
        choicestorage[current_state][data.iloc[i, 4]] += 1   
    for key in choicestorage:
        CCPs[key] = choicestorage[key] / np.sum(choicestorage[key])
    
    return CCPs, choicestorage

def forward(theta, beta, CCPs, choicestorage): 
    #For each row of data, we calculate the log-likelihood of the data given the CCPs and P  
    llh = 0
    for state in choicestorage:
        for stepsize in range(1, 5):  # Transitions to state+1, state+2, state+3, state+4
            next_state = state + stepsize
            if next_state in CCPs: 
                Qdiff = -theta[0]*state + theta[1]
                for stepsize in range(1, 5):  #next state integration part for maintenance action
                    next_state2 = state + stepsize
                    if next_state2 in CCPs:
                        Qdiff += 0.25*beta*(-np.log(CCPs[next_state2][1]))
                Qdiff = Qdiff + beta*np.log(CCPs[1][1])
                rProb = 1/(1+np.exp(Qdiff))
                cProb = 1-rProb
                Prob = [cProb, rProb] #Prob is the probability of each action given the current state
                for action in [0,1]: 
                    llh += choicestorage[state][action] * np.log(Prob[action]) 
    return -llh

def nfxp(theta, beta, s, mileage, d):

    value = vfi(theta, beta, s) # Action-specific value function given theta and beta
    
    EP = np.exp(value[:, 1]) / (np.exp(value[:, 0]) + np.exp(value[:, 1])) #The shape of EP is (len(s),)
    Index = mileage + 1
    llh = np.sum(np.log(EP[Index[d == 1] - 1])) + np.sum(np.log(1 - EP[Index[d == 0] - 1]))
    return -llh


def llh_forward(theta):
    return forward(theta, beta, CCPs, choicestorage) 

def llh_nfxp(theta):
    return nfxp(theta, beta, states, mileage, d) #beta, states, mileage, d are all global variables




# Load data
filepath = os.path.join(path, "data_bus50_test.txt") #os-independent path construction
save_path_nfxp = os.path.join(path, "results", f"NFXP_estimates_bus50.csv")
save_path_forward = os.path.join(path, "results", f"Forward_estimates_bus50.csv")
data = pd.read_csv(filepath, header=None) # Load data as pandas dataframe
data_new = data.iloc[1::2, :].reset_index(drop=True) #data.iloc[::2, :] selects every other row starting from the first row

data_reformat = data[data.iloc[:, 4] == 1]  
mileage = data_reformat.iloc[:, 3].values 
d = data_reformat.iloc[:, 2].values 

CCPs, choicestorage = CCPandTKEst(data_new)

theta0 = np.array([0, 0]) # Initial guess





# Optimize


theta_true = np.array([1,5])
Q_true = vfi(theta_true, beta, states)
expV_true = vfi_expV(theta_true, beta, states)
print("Q_true:", Q_true) 
print("expV_true:", expV_true)

start_forward = time.perf_counter()

res_forward = minimize(lambda x: llh_forward(x), theta0)
theta_hat_forward = res_forward.x
llh_forward_val = -res_forward.fun
print("Estimated thetas_forward:", theta_hat_forward)
print("Log-likelihood_forward:", llh_forward_val)
print("Q_forward:", vfi(theta_hat_forward, beta, states))
print("expV_forward:", vfi_expV(theta_hat_forward, beta, states))
end_forward = time.perf_counter()

print(f"Time taken for forward: {end_forward - start_forward}")


start_nfxp = time.perf_counter()

res_nfxp = minimize(lambda x: llh_nfxp(x), theta0)
theta_hat_nfxp = res_nfxp.x
llh_nfxp_val = -res_nfxp.fun
print("Estimated thetas_nfxp:", theta_hat_nfxp)
print("Log-likelihood_nfxp:", llh_nfxp_val)
print("Q_nfxp:", vfi(theta_hat_nfxp, beta, states))
print("expV_nfxp:", vfi_expV(theta_hat_nfxp, beta, states))
end_nfxp = time.perf_counter()

print(f"Time taken for nfxp: {end_nfxp - start_nfxp}")

# Standard Errors, Numerically
def neg_llh_forward(theta):
    return -llh_forward(theta)
def neg_llh_nfxp(theta):
    return -llh_nfxp(theta)

hessian_func_forward = nd.Hessian(neg_llh_forward)
hessian_func_nfxp = nd.Hessian(neg_llh_nfxp)
hessian_matrix_forward = hessian_func_forward(theta_hat_forward)
hessian_matrix_nfxp = hessian_func_nfxp(theta_hat_nfxp)
se_est_forward = np.sqrt(np.diag(inv(-hessian_matrix_forward)))
se_est_nfxp = np.sqrt(np.diag(inv(-hessian_matrix_nfxp)))
print("Standard errors_forward:", se_est_forward)
print("Standard errors_nfxp:", se_est_nfxp)


def compute_predicted_rewards(mileage, d, theta_hat):
    # Reward prediction based on the estimated theta1 and theta2
    predicted_rewards = np.zeros_like(mileage, dtype=float)
    
    # If action is not replace (d=0), use theta1 * mileage
    predicted_rewards[d == 0] = -theta_hat[0] * mileage[d == 0]
    
    # If action is replace (d=1), use theta2
    predicted_rewards[d == 1] = -theta_hat[1]
    
    return predicted_rewards

# Assuming the actual reward can be directly inferred from the dataset:
# If no replacement (d=0), actual reward is -theta1 * mileage
# If replacement (d=1), actual reward is -theta2 (replacement cost)
def compute_actual_rewards(mileage, d, theta_true):
    # The true reward would use the true parameter values (theta_true) to compute the actual reward
    actual_rewards = np.zeros_like(mileage, dtype=float)
    
    # True reward when no replacement
    actual_rewards[d == 0] = -theta_true[0] * mileage[d == 0]
    #d==0 is a boolean array where the value is True if the decision is not to replace, and False otherwise
    #This operation is equivalent to:
    #for i in range(len(mileage)):
    #    if d[i] == 0:
    #        actual_rewards[i] = -theta_true[0] * mileage[i]
    
    
    # True reward when replacement
    actual_rewards[d == 1] = -theta_true[1]
    
    return actual_rewards

# Assuming we are comparing the predictions using theta_hat with the true rewards generated by theta_true
theta_true = np.array([1,5])  # You can replace this with known ground truth thetas if available
theta_hat = theta_hat_forward 
print("theta_hat_forward:", theta_hat)
# Compute predicted and actual rewards
predicted_rewards = compute_predicted_rewards(mileage, d, theta_hat)
print("Predicted rewards:", predicted_rewards)
actual_rewards = compute_actual_rewards(mileage, d, theta_true)
print("Actual rewards:", actual_rewards)
#print the minimum of actual rewards
print("Maximum of actual rewards:", np.max(actual_rewards))
# Compute MSE
mse = np.mean((actual_rewards - predicted_rewards) ** 2)
#mean absolute percentage error (MAPE) is a measure used to estimate the accuracy of a model
mape = np.mean(np.abs((actual_rewards - predicted_rewards) / actual_rewards)) * 100
print("CCP MSE for reward prediction:", mse)
print("CCP MAPE for reward prediction:", mape)

data_size = data_reformat.shape[0]



for boots in [100]:
    thetas_forward = []
    for b in range(1, boots + 1):
        # Random sampling of bus numbers for each bootstrap
        randnum = np.random.randint(0, data_size, size=(data_size, 1)) # Randomly sample data_size bus numbers

        # Collect data for random buses
        randdata = pd.DataFrame()
        for bus in range(data_size):
            random_data = data[data.iloc[:, 0] == randnum[bus][0]] # Select data for the random bus number
            randdata = pd.concat([randdata, random_data], ignore_index=True) # Concatenate the data for the random bus number
            
        randdata_new1 = randdata.iloc[1::2, :].reset_index(drop=True) #data.iloc[::2, :] selects every other row starting from the first row
        
        # Estimate CCPs and transition matrix for the forward model
        CCPs, choicestorage = CCPandTKEst(randdata_new1)
        #print(f"CCPs: {CCPs}")
        #print(f"choice storage: {choicestorage}")
        
        # Minimization for forward using bootstrap CCPs
        def llh_forward_bootstrap(theta):
            return forward(theta, beta, CCPs, choicestorage)

        res_forward = minimize(llh_forward_bootstrap, theta0)
        MAPE = np.mean(np.abs((theta_true - res_forward.x) / theta_true)) * 100
        thetas_forward.append(MAPE)

        print(f"Bootstrap: {b} of {boots}, Estimated thetas_forward: {res_forward.x}")

    # Save the bootstrapped parameter estimates
    
    thetas_df_forward = pd.DataFrame(thetas_forward, columns=['MAPE'])
    #filepath_forward = os.path.join(path, "results", f"forward_estimates_{boots}.csv")
    thetas_df_forward.to_csv(save_path_forward, index=False)



# Load the bootstrapped parameter estimates

bootstrapped_data = pd.read_csv(save_path_forward)

# Calculate the bootstrapped mean and standard deviation
bootstrapped_mean = bootstrapped_data.mean()
bootstrapped_std = bootstrapped_data.std()

print("Bootstrapped Mean:")
print(bootstrapped_mean)

print("Bootstrapped Standard Deviation:")
print(bootstrapped_std)




theta_true = np.array([1,5])  # You can replace this with known ground truth thetas if available
theta_hat = theta_hat_nfxp
print("theta_hat_nfxp:", theta_hat)
# Compute predicted and actual rewards
predicted_rewards = compute_predicted_rewards(mileage, d, theta_hat)
print("Predicted rewards:", predicted_rewards)
actual_rewards = compute_actual_rewards(mileage, d, theta_true)
print("Actual rewards:", actual_rewards)
#print the minimum of actual rewards
print("Maximum of actual rewards:", np.max(actual_rewards))
# Compute MSE
mse = np.mean((actual_rewards - predicted_rewards) ** 2)
#mean absolute percentage error (MAPE) is a measure used to estimate the accuracy of a model
mape = np.mean(np.abs((actual_rewards - predicted_rewards) / actual_rewards)) * 100
print("NFXP MSE for reward prediction:", mse)
print("NFXP MAPE for reward prediction:", mape)



for boots in [100]:
    thetas = []
    for b in range(1, boots + 1):
        #in bootstrapping, the number of resampling is the same as the number of the original data
        data_size = data_reformat.shape[0]
        #randnum = np.random.randint(0, data_size, size=(data_size, 1)) 
        randnum = np.random.randint(0, data_size, size=(data_size, 1)) # Randomly sample sample-size number of bus numbers
        randdata_mileage = []
        randdata_d = []
        for bus in range(data_size):
            random_mileage = data_reformat[data_reformat.iloc[:, 0] == randnum[bus][0]].iloc[:, 3].values 
            random_d = data_reformat[data_reformat.iloc[:, 0] == randnum[bus][0]].iloc[:, 2].values

            randdata_mileage.append(random_mileage)
            randdata_d.append(random_d)

        randdata_mileage = np.concatenate(randdata_mileage)
        randdata_d = np.concatenate(randdata_d)

        res = minimize(lambda x: nfxp(x, beta, states, randdata_mileage, randdata_d), theta0)
        MAPE = np.mean(np.abs((theta_true - res.x) / theta_true)) * 100
        thetas.append(MAPE)
        print(f"Bootstrap: {b} of {boots}, Estimated thetas_nfxp: {res.x}")

    thetas_df = pd.DataFrame(thetas, columns=['MAPE'])
    filepath = os.path.join(path, "results", f"nfxp_estimates_{boots}.csv")
    thetas_df.to_csv(save_path_nfxp, index=False)

