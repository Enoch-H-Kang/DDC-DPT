# Differet reward functions 
import numpy as np
# Linear reward 
def linear_reward(theta, s):
	theta = np.array(theta)
	if s.ndim == 1:
		return -(theta*s).sum()
	elif s.ndim == 2:
		return - (theta*s).sum(1)
	else:
		raise NotImplementedError

