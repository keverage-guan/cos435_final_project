import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.experiment_config import ExperimentConfig


class DQN(nn.Module):
    '''
    Deep Q-Network used by both teacher (K=0, no message) and student (K=5, with message).
    Input is a 2D state coordinate concatenated with a message vector of length K.
    '''

    def __init__(self, K: int, config: ExperimentConfig, device: torch.device, zero_init: bool = False):
        '''
        INPUT
        K:         length of input message (0 for teacher, config.K for student)
        config:    experiment configuration
        device:    torch device
        zero_init: if True, initializes all weights and biases of layers 1-3 to zero
        '''
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(2 + K, 10, device=device)
        self.lin2 = nn.Linear(10, 20, device=device)
        self.lin3 = nn.Linear(20, 20, device=device)
        self.lin4 = nn.Linear(20, config.n_actions, device=device)

        if zero_init:
            nn.init.zeros_(self.lin1.weight)
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.weight)
            nn.init.zeros_(self.lin2.bias)
            nn.init.zeros_(self.lin3.weight)
            nn.init.zeros_(self.lin3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        ---
        INPUT
        x - state coordinate (and optionally message) tensor
        ---
        OUTPUT
        Q-values for each action: tensor of shape (n_actions,)
        '''
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x


class BiasLayer(nn.Module):
    '''
    Adds a learnable bias tensor of a given shape to its input.
    Used in ConvAutoEncoder encoder/decoder and student network.
    '''

    def __init__(self, shape: tuple, device: torch.device):
        '''
        INPUT
        shape:  shape of the bias tensor (must match input shape)
        device: torch device
        '''
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape, device=device), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias
