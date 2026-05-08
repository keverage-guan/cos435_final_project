# DQN and BiasLayer, based on cell 7

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    '''
    Deep Q Network class
    '''

    def __init__(self, K: int, n_actions: int, device: torch.device, zero_init: bool = False, input_dim : int = 2):
        '''
        INPUT
        K: length of input message (zero for the teacher)
        n_actions: number of possible actions
        device: torch device specification
        zero_init: initialize all weights to zero?
        '''
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_dim + K, 10, device=device) # input size is 2(gridworld coordinates) + length of message
        self.lin2 = nn.Linear(10, 20, device=device)
        self.lin3 = nn.Linear(20, 20, device=device)
        self.lin4 = nn.Linear(20, n_actions, device=device)

        # Initialize all weights (and biases) to zero
        if zero_init:
            torch.nn.init.zeros_(self.lin1.weight)
            torch.nn.init.zeros_(self.lin1.bias)
            torch.nn.init.zeros_(self.lin2.weight)
            torch.nn.init.zeros_(self.lin2.bias)
            torch.nn.init.zeros_(self.lin3.weight)
            torch.nn.init.zeros_(self.lin3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        forward pass of the network
        ---
        INPUT
        x - network input, i.e. combination of state and potentially message
        ---
        OUTPUT
        y - the four Q-values, i.e. torch.tensor([Q(s,0),Q(s,1),Q(s,2),Q(s,3)])
        '''
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x


class BiasLayer(nn.Module):
    '''
    Bias Layer (add bias to individual network nodes/filter positions)
    '''
    def __init__(self, shape: tuple, device: torch.device):
        '''
        Initialise parameters of bias layer
        ---
        INPUT
        shape: Requisite shape of bias layer
        device: torch device specification
        '''
        super(BiasLayer, self).__init__()
        init_bias = torch.zeros(shape, device=device)
        self.bias = nn.Parameter(init_bias, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass
        ---
        INPUT
        x: Input features
        ---
        OUTPUT
        y: Output of bias layer
        '''
        y = x + self.bias
        return y
