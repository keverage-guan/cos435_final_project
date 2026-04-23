import random
import torch
import torch.nn as nn

from configs.experiment_config import ExperimentConfig
from src.env.gridworld import state_int_to_tuple
from src.models.dqn import DQN


def select_action_epsgreedy(agent: DQN, state: torch.Tensor, message: torch.Tensor,
                            epsilon: float, config: ExperimentConfig) -> int:
    '''
    Epsilon-greedy action selection.
    ---
    INPUT
    agent   - DQN representing the agent
    state   - current state in coordinate representation
    message - message received by the agent
    epsilon - exploration probability
    config  - experiment configuration
    ---
    OUTPUT
    action (0:right, 1:up, 2:left, 3:down)
    '''
    if random.random() > epsilon:
        with torch.no_grad():
            input = torch.cat((state[0], message), 0)
            return agent(input).argmax().item()
    else:
        return random.choice(range(config.n_actions))


def select_action_optimism(agent: DQN, state_int: int, message: torch.Tensor,
                           alpha: float, sa_counts: 'dict[tuple[int, int], int]',
                           config: ExperimentConfig, device: torch.device) -> int:
    '''
    Optimism-in-face-of-uncertainty action selection.
    Adds an exploration bonus alpha/sqrt(visit_count) to each Q-value.
    ---
    INPUT
    agent      - DQN representing the agent
    state_int  - current state in integer representation
    message    - message received by the agent
    alpha      - exploration bonus constant
    sa_counts  - visit counts per (state, action) pair
    config     - experiment configuration
    device     - compute device
    ---
    OUTPUT
    action (0:right, 1:up, 2:left, 3:down)
    '''
    state = state_int_to_tuple(state_int, config, device)
    current_sa_counts = torch.tensor(
        [sa_counts[(state_int, a)] for a in range(config.n_actions)], device=device
    )
    with torch.no_grad():
        input = torch.cat((state[0], message), 0)
        qvals = agent(input)
        qvals = qvals + alpha / torch.sqrt(current_sa_counts)
        return qvals.argmax().item()


def select_action_eps_trust(agent: DQN, state_int: int, probas_message_matrix: torch.Tensor,
                            softy: nn.Softmax, iota: float, trust: float,
                            current_ep: int, total_eps: int,
                            config: ExperimentConfig, device: torch.device) -> 'tuple[torch.Tensor, torch.Tensor, int]':
    '''
    Epsilon-greedy action selection with global teacher trust parameter.
    ---
    INPUT
    agent                 - DQN representing the student agent
    state_int             - current state in integer representation
    probas_message_matrix - action probability matrix from the teacher message
    softy                 - softmax function
    iota                  - scaling constant for student Q-values before softmax
    trust                 - teacher trust weight in [0, 1]
    current_ep            - current episode number (for decaying epsilon)
    total_eps             - total number of episodes
    config                - experiment configuration
    device                - compute device
    ---
    OUTPUT
    probas_message  - action probabilities from teacher message at current state
    probas_student  - action probabilities from student Q-values at current state
    action          - selected action
    '''
    state = state_int_to_tuple(state_int, config, device)
    with torch.no_grad():
        # 1. message
        probas_message = probas_message_matrix[:, state_int // config.grid_dim, state_int % config.grid_dim]
        # 2. student
        qvals_student = agent(state[0])
        qvals_student -= qvals_student.min() * torch.ones(config.n_actions, device=device)
        probas_student = softy(iota * qvals_student)
        # 3. combination
        probas = trust * probas_message + (1 - trust) * probas_student
        # 4. decaying epsilon-greedy
        r = random.random()
        if r < 0.4 * (1 - current_ep / total_eps) + 0.4 * (current_ep / total_eps):
            action = random.choice(range(config.n_actions))
        else:
            action = torch.argmax(probas).item()
    return probas_message, probas_student, action


def select_action_eps_trust_uninfo(agent: DQN, state_int: int, softy: nn.Softmax,
                                   iota: float, current_ep: int, total_eps: int,
                                   config: ExperimentConfig, device: torch.device) -> 'tuple[torch.Tensor, int]':
    '''
    Epsilon-greedy action selection without a teacher (uninformed baseline).
    ---
    INPUT
    agent      - DQN representing the student agent
    state_int  - current state in integer representation
    softy      - softmax function
    iota       - scaling constant for student Q-values before softmax
    current_ep - current episode number (for decaying epsilon)
    total_eps  - total number of episodes
    config     - experiment configuration
    device     - compute device
    ---
    OUTPUT
    probas_student - action probabilities from student Q-values at current state
    action         - selected action
    '''
    state = state_int_to_tuple(state_int, config, device)
    with torch.no_grad():
        # 1. student
        qvals_student = agent(state[0])
        qvals_student -= qvals_student.min() * torch.ones(config.n_actions, device=device)
        probas = softy(iota * qvals_student)
        # 2. decaying epsilon-greedy
        r = random.random()
        if r < 0.4 * (1 - current_ep / total_eps) + 0.4 * (current_ep / total_eps):
            action = random.choice(range(config.n_actions))
        else:
            action = torch.argmax(probas).item()
    return probas, action


def select_action_opt_trust(agent: DQN, state_int: int, probas_message_matrix: torch.Tensor,
                            softy: nn.Softmax, iota: float, trust: float, alpha: float,
                            sa_counts: 'dict[tuple[int, int], int]',
                            config: ExperimentConfig, device: torch.device) -> 'tuple[torch.Tensor, torch.Tensor, int]':
    '''
    Optimism-in-face-of-uncertainty action selection with global teacher trust parameter.
    ---
    INPUT
    agent                 - DQN representing the student agent
    state_int             - current state in integer representation
    probas_message_matrix - action probability matrix from the teacher message
    softy                 - softmax function
    iota                  - scaling constant for student Q-values before softmax
    trust                 - teacher trust weight in [0, 1]
    alpha                 - exploration bonus constant
    sa_counts             - visit counts per (state, action) pair
    config                - experiment configuration
    device                - compute device
    ---
    OUTPUT
    probas_message  - action probabilities from teacher message at current state
    probas_student  - action probabilities from student Q-values at current state
    action          - selected action
    '''
    state = state_int_to_tuple(state_int, config, device)
    current_sa_counts = torch.tensor(
        [sa_counts[(state_int, a)] for a in range(config.n_actions)], device=device
    )
    with torch.no_grad():
        # 1. message
        probas_message = probas_message_matrix[:, state_int // config.grid_dim, state_int % config.grid_dim]
        # 2. student
        qvals_student = agent(state[0])
        qvals_student -= qvals_student.min() * torch.ones(config.n_actions, device=device)
        probas_student = softy(iota * qvals_student)
        # 3. combination with optimism bonus
        probas = trust * probas_message + (1 - trust) * (probas_student + alpha / (current_sa_counts + 1))
        action = torch.argmax(probas).item()
    return probas_message, probas_student, action


def select_action_opt_trust_uninfo(agent: DQN, state_int: int, softy: nn.Softmax,
                                   iota: float, alpha: float,
                                   sa_counts: 'dict[tuple[int, int], int]',
                                   config: ExperimentConfig, device: torch.device) -> 'tuple[torch.Tensor, int]':
    '''
    Optimism-in-face-of-uncertainty action selection without a teacher (uninformed baseline).
    ---
    INPUT
    agent      - DQN representing the student agent
    state_int  - current state in integer representation
    softy      - softmax function
    iota       - scaling constant for student Q-values before softmax
    alpha      - exploration bonus constant
    sa_counts  - visit counts per (state, action) pair
    config     - experiment configuration
    device     - compute device
    ---
    OUTPUT
    probas - action probabilities from student Q-values at current state
    action - selected action
    '''
    state = state_int_to_tuple(state_int, config, device)
    current_sa_counts = torch.tensor(
        [sa_counts[(state_int, a)] for a in range(config.n_actions)], device=device
    )
    with torch.no_grad():
        # 1. student
        qvals_student = agent(state[0])
        qvals_student -= qvals_student.min() * torch.ones(config.n_actions, device=device)
        probas = softy(iota * qvals_student)
        # 2. optimism bonus
        probas += alpha / (current_sa_counts + 1)
        action = torch.argmax(probas).item()
    return probas, action
