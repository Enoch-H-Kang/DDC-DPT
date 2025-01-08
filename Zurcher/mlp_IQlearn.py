import torch
from torch import nn
from torch.nn import functional as F

from multiHeadedMLPModule import MultiHeadedMLPModule


class MLP(MultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 states_dim,
                 actions_dim,
                 hidden_sizes = [10,10],
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=lambda x: nn.init.constant_(x, -55),
                 #output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(2, states_dim, [actions_dim, actions_dim], hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization) #fix n_heads to 2 (q and v heads) with 2 output dimensions each (actions_dim)
    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            Return Q-values and E[V(s',a')|s,a]-values.

        """
        states = x['states'] #dimension is (batch_size, horizon, state_dim)
        batch_size, horizon, state_dim = states.shape
        #
        states = states.reshape(-1, state_dim) #dim is (batch_size*horizon, state_dim)
        q_values, _ = super().forward(states) #dim is (batch_size*horizon, action_dim)
        q_values = q_values.reshape(batch_size, horizon, -1) #dim is (batch_size, horizon, action_dim)
        
        #states_p1 adds 1 to the mileage, which is the first element of the state
        states_p1 = states.clone() #dim is (batch_size*horizon, state_dim)
        #Add 1 to the first element of states_p1 if that element is smaller than 20 using one line of code
        states_p1[:,0] = torch.where(states_p1[:,0] + 1 < 20, states_p1[:,0] + 1, torch.tensor(20))
        q_values_p1, _ = super().forward(states_p1) #dim is (batch_size*horizon, action_dim)
        #vnext is logsumexp of q_values_p1
        vnext_p1 = torch.logsumexp(q_values_p1, dim=1) #dim is (batch_size*horizon)
        
        #states_p2 adds 2 to the mileage, which is the first element of the state
        states_p2 = states.clone()
        states_p2[:,0] = torch.where(states_p2[:,0] + 2 < 20, states_p2[:,0] + 2, torch.tensor(20))
        q_values_p2, _ = super().forward(states_p2) #dim is (batch_size*horizon, action_dim)
        vnext_p2 = torch.logsumexp(q_values_p2, dim=1) #dim is (batch_size*horizon)
        
        #states_p3 adds 3 to the mileage, which is the first element of the state
        states_p3 = states.clone()
        states_p3[:,0] = torch.where(states_p3[:,0] + 3 < 20, states_p3[:,0] + 3, torch.tensor(20))
        q_values_p3, _ = super().forward(states_p3) #dim is (batch_size*horizon, action_dim)
        vnext_p3 = torch.logsumexp(q_values_p3, dim=1) #dim is (batch_size*horizon)
        
        states_p4 = states.clone()
        states_p4[:,0] = torch.where(states_p4[:,0] + 3 < 20, states_p4[:,0] + 3, torch.tensor(20))
        q_values_p4, _ = super().forward(states_p4) #dim is (batch_size*horizon, action_dim)
        vnext_p4 = torch.logsumexp(q_values_p4, dim=1) #dim is (batch_size*horizon)
        
        #mean of p1-p3 is the nextv_0, the expected next state value function of not replacing the engine now
        vnext_0 = (vnext_p1 + vnext_p2 + vnext_p3 + vnext_p4)/4 #dim is (batch_size*horizon)
        
        #states_r1 returns to the mileage 1
        states_r1 = states.clone()
        states_r1[:,0] = 1
        q_values_r1, _ = super().forward(states_r1) #dim is (batch_size*horizon, action_dim)
        vnext_1 = torch.logsumexp(q_values_r1, dim=1) #dim is (batch_size*horizon)
        
        #stack vnext_0 and vnext_1
        vnext_values = torch.stack([vnext_0, vnext_1], dim=1) #dim is (batch_size*horizon, 2)
        vnext_values = vnext_values.reshape(batch_size, horizon, -1) #dim is (batch_size, horizon, 2)
          
        next_states = x['next_states']
        next_states = next_states.reshape(-1, state_dim) # dim is (batch_size*horizon, state_dim)
        next_q_values, _ = super().forward(next_states)
        next_q_values = next_q_values.reshape(batch_size, horizon, -1)
        
        return q_values, next_q_values, vnext_values