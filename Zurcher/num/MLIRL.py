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
gamma = np.euler_gamma

# Set up state-space
xmax = 200
states = np.arange(xmax + 1) # generates an array starting from 0 up to xmax, inclusive

def get_util(theta, s):
    u1 = -theta[0] * s  # Period maintenance cost
    u2 = -theta[1] * np.ones(s.shape)  # Replacement cost
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
        #Q1[:, 1] = U[:, 1]+ self.type*U[:, 2] + self.beta * expV1  # deterministic transition to state 1. Type is 0 (default setting)
        Q1[:, 1] = U[:, 1]+ beta * expV1
        expV = np.column_stack((expV0, expV1)) #Dimension is (states, actions)
        dist = np.linalg.norm(Q1 - Q)
        Q=Q1
        iter += 1

    return Q


def CCPandTKEst(data, max_state=20):
    """
    Estimate CCPs and the empirical transition probabilities P (for action 0) 
    nonparametrically from the given data. Transitions are only counted when 
    the bus index (column 0) is the same for consecutive rows.

    Args:
        data (pd.DataFrame): The input data with columns [bus_id, ..., current_state, decision].
        max_state (int): Maximum state value to consider (default is 20).

    Returns:
        CCPs (dict): Conditional Choice Probabilities for each state.
        choicestorage (dict): Raw counts of state-action occurrences.
        P (np.ndarray): Transition probability matrix (size max_state+1 x max_state+1) for action 0.
    """
    # Initialize CCPs and choice storage
    CCPs = {}
    choicestorage = {}

    # Initialize transition counts for action 0
    P_count = np.zeros((max_state + 1, max_state + 1))  # Count of transitions for action 0

    for i in range(data.shape[0] - 1):  # Iterate until the second-to-last row
        current_bus = data.iloc[i, 0]  # Bus index of the current row
        next_bus = data.iloc[i + 1, 0]  # Bus index of the next row
        current_state = int(data.iloc[i, 3])  # Current state (mileage)
        action = int(data.iloc[i, 4])  # Action taken (0 or 1)

        # Record state-action counts
        if current_state not in choicestorage:
            choicestorage[current_state] = [0, 0]
        choicestorage[current_state][action] += 1

        # Record state-to-state transitions for action 0, only if the bus index matches
        if action == 0 and current_bus == next_bus:
            next_state = int(data.iloc[i + 1, 3])  # Next state (mileage of next row)
            if current_state <= max_state and next_state <= max_state:
                P_count[current_state, next_state] += 1

    # Calculate CCPs by normalizing choicestorage counts
    for state in choicestorage:
        CCPs[state] = choicestorage[state] / np.sum(choicestorage[state])

    # Calculate transition probabilities P for action 0 by normalizing P_count
    P = np.nan_to_num(P_count / np.sum(P_count, axis=1)[:, None])

    return CCPs, choicestorage, P


def forward(theta, beta, CCPs, choicestorage, P): 
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
                        prob = P[state][next_state2]
                        Qdiff += prob*beta*(-np.log(CCPs[next_state2][1]))
                Qdiff = Qdiff + beta*np.log(CCPs[1][1])
                rProb = 1/(1+np.exp(Qdiff))
                cProb = 1-rProb
                Prob = [cProb, rProb] #Prob is the probability of each action given the current state
                for action in [0,1]: 
                    llh += choicestorage[state][action] * np.log(Prob[action]) 
    #print(llh)
    return -llh

def nfxp(theta, beta, s, mileage, d):

    value = vfi(theta, beta, s) # Action-specific value function given theta and beta
    EP = np.exp(value[:, 1]) / (np.exp(value[:, 0]) + np.exp(value[:, 1])) #The shape of EP is (len(s),)
    Index = mileage + 1
    llh = np.sum(np.log(EP[Index[d == 1] - 1])) + np.sum(np.log(1 - EP[Index[d == 0] - 1]))
    return -llh


def llh_forward(theta):
    return forward(theta, beta, CCPs, choicestorage, P) 




# Load data
filepath = os.path.join(path, "data_bus10000_test.txt") #os-independent path construction
save_path = os.path.join(path, "results", f"forward_estimates_bus10000.csv")
#filepath = os.path.join(path, "data_assg3.txt") #os-independent path construction
#data = pd.read_csv(filepath) # Load data as pandas dataframe
data = pd.read_csv(filepath, header=None) # Load data as pandas dataframe
#data_new is even rows of data
data_new = data.iloc[1::2, :].reset_index(drop=True) #data.iloc[::2, :] selects every other row starting from the first row

data_reformat = data[data.iloc[:, 4] == 1]  
mileage = data_reformat.iloc[:, 3].values 
d = data_reformat.iloc[:, 2].values 


CCPs, choicestorage, P = CCPandTKEst(data_new)

theta0 = np.array([0, 0]) # Initial guess





# Optimize

start_forward = time.perf_counter()

res_forward = minimize(lambda x: llh_forward(x), theta0)
theta_hat_forward = res_forward.x
print("Estimated thetas_forward:", theta_hat_forward)

end_forward = time.perf_counter()

print(f"Time taken for forward: {end_forward - start_forward}")


# Standard Errors, Numerically
def neg_llh_forward(theta):
    return -llh_forward(theta)


hessian_func_forward = nd.Hessian(neg_llh_forward)
hessian_matrix_forward = hessian_func_forward(theta_hat_forward)
se_est_forward = np.sqrt(np.diag(inv(-hessian_matrix_forward)))
print("Standard errors_forward:", se_est_forward)


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

'''

for boots in [20]:
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
    thetas_df_forward.to_csv(save_path, index=False)



# Load the bootstrapped parameter estimates

bootstrapped_data = pd.read_csv(save_path)

# Calculate the bootstrapped mean and standard deviation
bootstrapped_mean = bootstrapped_data.mean()
bootstrapped_std = bootstrapped_data.std()

print("Bootstrapped Mean:")
print(bootstrapped_mean)

print("Bootstrapped Standard Deviation:")
print(bootstrapped_std)




theta_true = np.array([1,5])  # You can replace this with known ground truth thetas if available
theta_hat = theta_hat_nfxp
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



for boots in [20]:
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
    thetas_df.to_csv(save_path, index=False)


import matplotlib.pyplot as plt

# Load the bootstrapped parameter estimates
filepath_forward = os.path.join(path, "results", f"forward_estimates_{boots}_bus500.csv")
bootstrapped_data = pd.read_csv(filepath_forward)


# Plot the probability distribution of theta1
plt.hist(bootstrapped_data['theta1_forward'], bins=100, density=True, alpha=0.5)
plt.xlabel('theta1')
plt.ylabel('Probability')
plt.title('Probability Distribution of theta1')
plt.savefig(os.path.join(path, "results", "theta1_distribution.png"))
plt.show()
plt.close()

# Plot the probability distribution of theta2
plt.hist(bootstrapped_data['theta2_forward'], bins=100, density=True, alpha=0.5)
plt.xlabel('theta2')
plt.ylabel('Probability')
plt.title('Probability Distribution of theta2')
plt.savefig(os.path.join(path, "results", "theta2_distribution.png"))
plt.show()
plt.close()

'''
