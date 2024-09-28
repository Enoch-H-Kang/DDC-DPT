import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv
from scipy.special import logsumexp
import os
import numdifftools as nd
import time

np.random.seed(1234)

path = "/home/ehwkang/folder_Class/DDCM_Hema/HW7_data/"

# Define parameters
beta = 0.9
states_TF = np.array([1, 0, 0, 0, 0]) #dimension of s is 5, index 0 is mileage and index 2-4 are dummy variables
# Set up state-space
xmax = 200
states = np.arange(xmax + 1) # generates an array starting from 0 up to xmax, inclusive

def get_util(theta, states):
    s1 = states[:,0] # Mileage
    
    u1 = -theta[0] * s1  # Period maintenance cost
    u2 = -theta[1] * np.ones(s1.shape)  # Replacement cost
    U = np.column_stack((u1, u2))
    return U

def statespace_gen(states_TF, xmax, maxdummy):
    num_dummies = (states_TF == 0).sum()  # Number of dummy variables
    first_index = np.arange(xmax + 1)

    # Define the range for the second, third, and fourth indices (-10 to 10)
    dummy_indices = np.arange(-maxdummy, maxdummy+1)
    
    index_list = [first_index] + [dummy_indices] * k

    # Generate all combinations of the second, third, and fourth indices
    grid = np.meshgrid(*index_list, indexing='ij')

    # Now, for each value in first_index, create a full state by combining it with the combinations of the other indices
    states = np.array([g.ravel() for g in grid]).T # ravel() flattens the array
    # number of states = (xmax + 1) * (2 * maxdummy + 1)^k
    # states.shape = (xmax + 1 *(2 * maxdummy + 1)^k, k+1)
    
    return states

def vfi(theta, beta, states):
    U = get_util(theta, states)
    gamma = np.euler_gamma

    Q = np.zeros((len(states), 2)) #row: state, column: action
    dist = 1
    iter = 0
    while dist > 1e-8:
        V = gamma + logsumexp(Q, axis=1)
        # Ensure expV corresponds to V but shifts the last element for replacement decision logic
        #expV = np.append(V[1:], V[-1])  # As mileage does not increase after it reaches max, the last element is repeated
        
        expV = np.zeros_like(V)
        for i in range(4):
            Vi = np.append(V[i+1:], [V[-1]] * (i+1))  # Transition to s+i+1 with prob 1/4, last state repeats for boundary
            expV += Vi / 4
        
        Q1 = np.zeros_like(Q)  # initialize Q1
        # Compute value function for maintenance (not replacing)
        Q1[:, 0] = U[:, 0] + beta * expV  # action-specific value function of not replacing
        # Compute value function for replacement
        Q1[:, 1] = U[:, 1] + beta * V[1]  # action-specific value function of replacing
        
        dist = np.linalg.norm(Q1 - Q)
        Q = Q1
        iter += 1

    return Q


def nfxp(theta, beta, s, mileage, d):

    value = vfi(theta, beta, s) # Action-specific value function given theta and beta
    EP = np.exp(value[:, 1]) / (np.exp(value[:, 0]) + np.exp(value[:, 1]))
    Index = mileage + 1
    llh = np.sum(np.log(EP[Index[d == 1] - 1])) + np.sum(np.log(1 - EP[Index[d == 0] - 1]))
    return -llh

def llh(theta):
    return nfxp(theta, beta, states, mileage, d) #beta, states, mileage, d are all global variables

# Load data
filepath = os.path.join(path, "data7_test.txt") #os-independent path construction
data = pd.read_csv(filepath) # Load data as pandas dataframe
'''
The third column of the data is just repetition of 0, 1, 0, 1.... 
and the fifth column of data indicates whether the decision (0 no replacement, 1 replacement) actually matches with the third column value (if matches, 0; othewise, 1). 
'''
data_reformat = data[data.iloc[:, 4] == 1]  # Filter rows where the decision matches with the third column value
mileage = data_reformat.iloc[:, 3].values # Extract mileage data
d = data_reformat.iloc[:, 2].values #

theta0 = np.array([0, 0])

start_nfxp = time.perf_counter()

# Optimize
res = minimize(lambda x: llh(x), theta0)
theta_hat = res.x
print("Estimated thetas:", theta_hat)

end_nfxp = time.perf_counter()

print(f"Time taken for NFXP: {end_nfxp - start_nfxp} seconds")

# Standard Errors, Numerically
def neg_llh(theta):
    return -llh(theta)


hessian_func = nd.Hessian(neg_llh)
hessian_matrix = hessian_func(theta_hat)
se_est = np.sqrt(np.diag(inv(-hessian_matrix)))
print("Standard errors:", se_est)
'''
# Bootstrapping
for boots in [50, 125, 250, 500, 1000, 2000]:
    thetas = []
    for b in range(1, boots + 1):
        randnum = np.random.randint(0, 1000, size=(1000, 1))

        randdata_mileage = []
        randdata_d = []
        for bus in range(1000):
            random_mileage = data_reformat[data_reformat.iloc[:, 0] == randnum[bus][0]].iloc[:, 3].values
            random_d = data_reformat[data_reformat.iloc[:, 0] == randnum[bus][0]].iloc[:, 2].values

            randdata_mileage.append(random_mileage)
            randdata_d.append(random_d)

        randdata_mileage = np.concatenate(randdata_mileage)
        randdata_d = np.concatenate(randdata_d)

        res = minimize(lambda x: nfxp(x, beta, states, randdata_mileage, randdata_d), theta0)
        thetas.append(res.x)
        print(f"Bootstrap: {b} of {boots}, Estimated thetas: {res.x}")

    thetas_df = pd.DataFrame(thetas, columns=['theta1', 'theta2'])
    filepath = os.path.join(path, "results", f"estimates_{boots}.csv")
    thetas_df.to_csv(filepath, index=False)
'''