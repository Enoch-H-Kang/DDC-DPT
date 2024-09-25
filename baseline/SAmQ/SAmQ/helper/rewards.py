# Differet reward functions 
import numpy as np
# Linear reward 
def linear_reward(b, theta, s):
	theta = np.array(theta)
	if s.ndim == 1:
		return b-(theta*s).sum()
	elif s.ndim == 2:
		return b - (theta*s).sum(1)
	else:
		raise NotImplementedError

