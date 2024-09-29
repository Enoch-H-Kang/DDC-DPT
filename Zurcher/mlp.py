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
                 output_b_init=lambda x: nn.init.constant_(x, -40),
                 #output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(1, states_dim, actions_dim, #single-headed MLP with input dimensions states_dim and output dimensions actions_dim
                         hidden_sizes,
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
        states = x['states'] #dimension is (batch_size, horizon, state_dim), (batch_size, horizon) if state_dim is 1
        
        
        batch_size = states.shape[0]
        horizon = states.shape[1]
        states = states.reshape(-1, states.shape[-1]) #dim is (batch_size*horizon, state_dim)
   
        q_values = super().forward(states)[0] #dim is (batch_size*horizon, action_dim)
        q_values = q_values.reshape(batch_size, horizon, -1) #dim is (batch_size, horizon, action_dim)
        
        next_states = x['next_states']
        next_states = next_states.reshape(-1, next_states.shape[-1]) #dim is (batch_size*horizon, state_dim)
        
        next_q_values = super().forward(next_states)[0]
        next_q_values = next_q_values.reshape(batch_size, horizon, -1)
        
        return q_values, next_q_values
    
    

class QtoVMLP(MultiHeadedMLPModule):
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
                 actions_dim,
                 hidden_sizes = [10],
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=lambda x: nn.init.constant_(x, -40),
                 #output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__(1, actions_dim, #Input dimensions is actions_dim (Q-values for actions)
                         actions_dim, #fix n_heads to 1 (q head) with output dimensions actions_dim. If multiple heads [actions_dim, actions_dim] format
                         hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization) #fix n_heads to 2 (q and v heads) with 2 output dimensions each (actions_dim)
    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward method.

        Args:
            input_value (torch.Tensor):  Q-values with (N, *, input_dim)
                shape.

        Returns:
            Return and E[V(s',a')|s,a]-values.

        """
        
        pred_Q_values = x ## Dimension of x is (batch_size, horizon, action_dim)
        pred_Q_values_reshaped = pred_Q_values.reshape(-1, pred_Q_values.shape[-1]) #dim is (batch_size*horizon, action_dim)
        
        batch_size = pred_Q_values.shape[0]
        horizon = pred_Q_values.shape[1]
   
        vnext_values = super().forward(pred_Q_values_reshaped)[0] #dim is (batch_size*horizon, action_dim)
        vnext_values = vnext_values.reshape(batch_size, horizon, -1) #dim is (batch_size, horizon, action_dim)
        
        return vnext_values