import math as mt
import os
import pickle
import random
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from configs.experiment_config import ExperimentConfig
from src.env.gridworld import SquareGridworld, state_int_to_tuple
from src.models.dqn import DQN
from src.rl.memory import ReplayMemory, Transition
from src.rl.policies import select_action_optimism
from src.utils.graph_utils import graph_from_walls


def transition_memories(init_state: int, goal_state: int, wall_states: 'list[int]',
                        config: ExperimentConfig, device: torch.device) -> 'tuple[dict, list]':
    '''
    Precompute a compact memory of all transitions in a gridworld for efficient DQN training.
    ---
    INPUT
    init_state, goal_state, wall_states - task specification
    config  - experiment configuration
    device  - compute device
    ---
    OUTPUT
    transition_index_dict - maps (state, action) to its index in the transition batch
    batches_list          - [concatenated states+nextstates tensor, rewards tensor]
    '''
    env = SquareGridworld(init_state, goal_state, wall_states, config)
    outcomes = env.get_outcomes()

    transition_memory = ReplayMemory(capacity=config.n_actions * config.grid_dim ** 2)
    transition_index_dict = {}
    i = 0
    for s_int in range(config.grid_dim ** 2):
        if s_int in wall_states or s_int == goal_state:
            continue
        s = state_int_to_tuple(s_int, config, device)
        for a in range(config.n_actions):
            transition_index_dict[(s_int, a)] = i
            i += 1
            ns_int, r = outcomes[(s_int, a)]
            ns = state_int_to_tuple(ns_int, config, device)
            transition_memory.push(s, torch.tensor([[a]], device=device), ns, torch.tensor([[r]], device=device))
    # add goal state transitions at the end
    gs = state_int_to_tuple(goal_state, config, device)
    for a in range(config.n_actions):
        transition_memory.push(gs, torch.tensor([[a]], device=device), gs, torch.tensor([[config.goal_reward]], device=device))
        transition_index_dict[(goal_state, a)] = i
        i += 1

    transitions = transition_memory.memory
    batch = Transition(*zip(*transitions))
    nextstates = torch.cat(batch.next_state)
    rewards = torch.cat(batch.reward)

    # second memory for states only (avoids running each state through the network four times)
    transition_memory2 = ReplayMemory(capacity=config.n_actions * config.grid_dim ** 2)
    for s_int in range(config.grid_dim ** 2):
        if s_int in wall_states or s_int == goal_state:
            continue
        s = state_int_to_tuple(s_int, config, device)
        transition_memory2.push(s, torch.tensor([[0]], device=device), s, torch.tensor([[0.]], device=device))
    gs = state_int_to_tuple(goal_state, config, device)
    transition_memory2.push(gs, torch.tensor([[0]], device=device), gs, torch.tensor([[0.]], device=device))

    transitions2 = transition_memory2.memory
    batch2 = Transition(*zip(*transitions2))
    states = torch.cat(batch2.state)
    batches_list = [torch.cat([states, nextstates]), rewards]

    return transition_index_dict, batches_list


def optimize_dqn(network: DQN, optimizer: torch.optim.Optimizer, memory: torch.Tensor,
                 batches: 'list[torch.Tensor]', goal_found: bool, loss_norm: nn.Module,
                 config: ExperimentConfig) -> float:
    '''
    Perform one gradient step on the teacher DQN using the Bellman equation.
    ---
    INPUT
    network    - DQN being optimized
    optimizer  - optimizer for the DQN
    memory     - tensor of transition indices (short + long term memory merged)
    batches    - [states+nextstates tensor, rewards tensor] from transition_memories
    goal_found - whether the goal has been reached yet
    loss_norm  - loss function (e.g. nn.MSELoss)
    config     - experiment configuration
    ---
    OUTPUT
    loss - scalar loss value for this step
    '''
    states_nextstates, rewards = batches
    n = config.n_actions

    # compute Q-values for all states, next states, and all actions
    qvalues = network(states_nextstates)
    state_action_values = torch.flatten(qvalues[:int(len(rewards) / n)])
    # max_a' Q(s', a') for all next states
    next_state_values = qvalues[int(len(rewards) / n):].max(dim=1, keepdim=True).values

    # Bellman targets
    expected_state_action_values = torch.flatten(config.gamma_bellman * next_state_values + rewards)

    # weight transitions by visit count (sqrt for MSELoss)
    trans_factors = torch.sqrt(torch.bincount(memory, minlength=len(next_state_values)))

    if goal_found:
        expected_state_action_values[-n:] = config.goal_reward
        trans_factors[-n:] = mt.sqrt(2)  # goal actions counted in both short- and long-term memory
        loss = 1 / mt.sqrt(len(memory) + 2 * n) * loss_norm(
            state_action_values * trans_factors,
            expected_state_action_values * trans_factors
        )
    if not goal_found:
        expected_state_action_values[-n:] = 0.
        state_action_values[-n:] = 0.
        loss = 1 / mt.sqrt(len(memory)) * loss_norm(
            state_action_values * trans_factors,
            expected_state_action_values * trans_factors
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def q_matrix_from_network(network: nn.Module, message: torch.Tensor, wall_states: 'list[int]',
                          config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    '''
    Extract the full Q-matrix from a trained network by forward-passing every state.
    ---
    INPUT
    network     - trained teacher network
    message     - message received (empty tensor for teacher)
    wall_states - wall states of the gridworld
    config      - experiment configuration
    device      - compute device
    ---
    OUTPUT
    q_matrix - tensor of shape (n_actions, grid_dim, grid_dim)
    '''
    q_matrix = torch.zeros(size=(config.n_actions, config.grid_dim, config.grid_dim))
    for s in range(config.grid_dim ** 2):
        indx, indy = s // config.grid_dim, s % config.grid_dim
        if (not (s in wall_states)) or config.lava:
            q_matrix[:, indx, indy] = network(torch.cat((state_int_to_tuple(s, config, device)[0], message), 0))
        else:
            q_matrix[:, indx, indy] = torch.zeros(config.n_actions, device=device)
    return q_matrix


def q_matrix_generator_deterministic(label_dict: 'dict[int, list]',
                                     wall_state_dict: 'dict[int, list[int]]',
                                     config: ExperimentConfig,
                                     device: torch.device) -> 'dict[int, torch.Tensor]':
    '''
    Generate perfect Q-matrices analytically using Dijkstra shortest paths.
    ---
    INPUT
    label_dict      - maps task index to [wall_index, init_state, goal_state]
    wall_state_dict - maps wall index to list of wall states
    config          - experiment configuration
    device          - compute device
    ---
    OUTPUT
    qdict - maps task index to perfect Q-matrix tensor of shape (n_actions, grid_dim, grid_dim)
    '''
    qdict = {}
    for task, (wall_index, init_state, goal_state) in label_dict.items():
        wall_states = wall_state_dict[wall_index]
        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        env = SquareGridworld(init_state, goal_state, wall_states, config)
        outcomes = env.get_outcomes()

        q_matrix = torch.zeros(size=(config.n_actions, config.grid_dim, config.grid_dim))
        v_value_dict = {goal_state: config.goal_reward}

        # compute V(s) for all non-wall, non-goal states via Dijkstra
        for s in range(config.grid_dim ** 2):
            if (config.lava or not (s in wall_states)) and s != goal_state:
                path = nx.dijkstra_path(G, s, goal_state, weight='weight')
                path.reverse()
                v_value = config.goal_reward
                for k in range(len(path) - 1):
                    edge_reward = -G.edges[(path[k], path[k + 1])]['weight']
                    v_value = config.gamma_bellman * v_value + edge_reward
                v_value_dict[s] = v_value

        # derive Q(s, a) from V(s) via one-step lookahead
        for s in range(config.grid_dim ** 2):
            indx, indy = s // config.grid_dim, s % config.grid_dim
            if s in wall_states:
                q_matrix[:, indx, indy] = torch.zeros(config.n_actions, device=device)
            elif s == goal_state:
                q_matrix[:, indx, indy] = torch.tensor(
                    [config.goal_reward] * config.n_actions, device=device
                )
            else:
                for a in range(config.n_actions):
                    next_state, reward = outcomes[(s, a)]
                    q_matrix[a, indx, indy] = config.gamma_bellman * v_value_dict[next_state] + reward

        qdict[task] = q_matrix

    print("Completed calculation of deterministic Q-matrices")
    return qdict


def q_matrix_generator(label_dict: 'dict[int, list]',
                       wall_state_dict: 'dict[int, list[int]]',
                       perfect_qdict: 'dict[int, torch.Tensor]',
                       config: ExperimentConfig,
                       device: torch.device,
                       num_eps: int = 200,
                       loss_norm: nn.Module = None,
                       max_steps: int = 50,
                       alpha: float = 16.0) -> 'dict[int, torch.Tensor]':
    '''
    Train a DQN teacher for each task and collect near-optimal Q-matrices.
    Retries up to config.max_attempts times if the result diverges from perfect_qdict.
    ---
    INPUT
    label_dict      - maps task index to [wall_index, init_state, goal_state]
    wall_state_dict - maps wall index to list of wall states
    perfect_qdict   - perfect Q-matrices from q_matrix_generator_deterministic
    config          - experiment configuration
    device          - compute device
    num_eps         - number of training episodes per task
    loss_norm       - loss function for optimize_dqn (defaults to nn.MSELoss)
    max_steps       - maximum steps per episode
    alpha           - exploration bonus constant for optimism policy
    ---
    OUTPUT
    qdict - maps task index to trained Q-matrix tensor of shape (n_actions, grid_dim, grid_dim)
    '''
    if loss_norm is None:
        loss_norm = nn.MSELoss()

    qdict = {}
    for task, (wall_index, init_state, goal_state) in label_dict.items():
        wall_states = wall_state_dict[wall_index]
        print(f"task is {task}")
        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        env = SquareGridworld(init_state, goal_state, wall_states, config)
        outcomes = env.get_outcomes()

        transition_index_dict, batches_list = transition_memories(init_state, goal_state, wall_states, config, device)

        good_qmatrix = False
        counter = 0
        while counter < config.max_attempts and not good_qmatrix:
            ep_steps = np.zeros(num_eps)
            teacher = DQN(K=0, n_actions=config.n_actions, device=device)
            message = torch.tensor([], device=device)
            optimizer = torch.optim.Adam(teacher.parameters(), lr=config.lr_teacher, weight_decay=0)

            memory_short = deque([], maxlen=config.L)
            memory_long = deque([], maxlen=config.n_actions * config.grid_dim ** 2)
            sa_counts = {(s, a): 0 for s in range(config.grid_dim ** 2) for a in range(config.n_actions)}
            goalfound = False

            for ep in range(num_eps):
                # start each episode at a random non-wall, non-goal state
                state_int = random.choice([
                    s for s in range(config.grid_dim ** 2)
                    if s not in wall_states and s != goal_state
                ])
                for t in range(max_steps):
                    action = select_action_optimism(teacher, state_int, message, alpha, sa_counts, config, device)
                    next_state_int, reward = outcomes[(state_int, action)]

                    reward = torch.tensor([[reward]], device=device)
                    action = torch.tensor([[action]], device=device)

                    if next_state_int is None:
                        if not goalfound:
                            goalfound = True
                        ep_steps[ep] = t
                        break
                    else:
                        memory_short.append(transition_index_dict[(state_int, action.item())])
                        if sa_counts[(state_int, action.item())] == 0:
                            memory_long.append(transition_index_dict[(state_int, action.item())])

                        if t == max_steps - 1:
                            ep_steps[ep] += max_steps
                        sa_counts[(state_int, action.item())] += 1
                        state_int = next_state_int

                    memory = torch.tensor(list(memory_short) + list(memory_long), device=device)
                    optimize_dqn(teacher, optimizer, memory, batches_list, goalfound, loss_norm, config)

            q_matrix = q_matrix_from_network(teacher, message, wall_states, config, device)
            difference_to_perfect = torch.linalg.vector_norm(q_matrix - perfect_qdict[task], ord=1)
            print(f"difference to perfect Q-matrix: {difference_to_perfect}")

            if difference_to_perfect < config.accuracy:
                good_qmatrix = True
                qdict[task] = q_matrix
                q_array = np.round(np.flip(q_matrix.detach().numpy(), 1).copy(), 3)
                print(f"final q-matrix of teacher is {q_array}")
            elif counter == config.max_attempts - 1:
                qdict[task] = perfect_qdict[task]
                counter += 1
            else:
                counter += 1

        plt.show()

    return qdict


def generate_q_matrices(config: ExperimentConfig,
                        device: torch.device) -> 'tuple[dict, dict, dict]':
    '''
    Top-level entry point for Q-matrix generation (called by main.py).
    Loads task definitions from disk, then either loads or regenerates Q-matrices
    depending on config.qmat_gen.
    ---
    INPUT
    config - experiment configuration
    device - compute device
    ---
    OUTPUT
    q_matrices      - maps task index to Q-matrix tensor
    label_dict      - maps task index to [wall_index, init_state, goal_state]
    wall_state_dict - maps wall index to list of wall states
    '''
    label_path = f"data/teacher/label dictionaries/q_matrices_labels{config.qmat_read_code}.pkl"
    wall_path = f"data/teacher/wall state dictionaries/wall_states{config.qmat_read_code}.pkl"

    with open(label_path, 'rb') as f:
        label_dict = pickle.load(f)
    with open(wall_path, 'rb') as f:
        wall_state_dict = pickle.load(f)

    if not config.qmat_gen:
        qmat_path = f"data/teacher/q matrix dictionaries/q_matrices{config.qmat_read_code}.pkl"
        with open(qmat_path, 'rb') as f:
            q_matrices = pickle.load(f)
        return q_matrices, label_dict, wall_state_dict

    perfect_qdict = q_matrix_generator_deterministic(label_dict, wall_state_dict, config, device)
    q_matrices = q_matrix_generator(label_dict, wall_state_dict, perfect_qdict, config, device)

    os.makedirs("data/teacher/q matrix dictionaries", exist_ok=True)
    save_path = f"data/teacher/q matrix dictionaries/q_matrices{config.qmat_read_code}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(q_matrices, f)

    return q_matrices, label_dict, wall_state_dict
