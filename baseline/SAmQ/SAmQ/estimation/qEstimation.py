"""This modules estimates a continuous Q-function network with discrete actions"""

import torch

from SAmQ.estimation.mlp import MLPModule
from torch.nn import functional as F  # NOQA
from torch.nn import CrossEntropyLoss
import numpy as np

class qEstimation(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, dState, nAction, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._obs_dim = dState
        #self._action_dim = env_spec.action_space.flat_dim
        self._action_dim = nAction
        self.IRLObses=None
        self.IRLActions = None
        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)
        
    def forward(self, observations):
        """Return Q-value(s)."""
        return torch.clamp(super().forward(observations), min=-50, max=50)
