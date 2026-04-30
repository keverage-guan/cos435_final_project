# student performance evaluator and random walker

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
from src.rl.teacher import transition_memories, optimize_dqn

def train_student(label_dict, wall_state_dict, sopt_dict, message_dict, config, device,
                  num_eps=750, max_steps=50, alpha=16.0):
    
    student = DQN(K=config.K, n_actions=config.n_actions, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr_teacher)
    loss_norm = nn.MSELoss()

    for task, (wall_index, init_state, goal_state) in label_dict.items():
        wall_states = wall_state_dict[wall_index]
        print(f"training student on task {task}")

        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        env = SquareGridworld(init_state, goal_state, wall_states, config)
        outcomes = env.get_outcomes()

        transition_index_dict, batches_list = transition_memories(init_state, goal_state, wall_states, config, device)

        # get message from trained SAE
        message = message_dict[(wall_index, init_state, goal_state)]

        # concatenate message to all states and next states in batches_list
        states_nextstates = batches_list[0]  # shape: (n_states*2, 2)
        message_repeated = message.unsqueeze(0).expand(len(states_nextstates), -1)  # shape: (n_states*2, K)
        states_nextstates_with_message = torch.cat([states_nextstates, message_repeated], dim=1)  # shape: (n_states*2, 7)
        batches_list_student = [states_nextstates_with_message, batches_list[1]]

        memory_short = deque([], maxlen=config.L)
        memory_long = deque([], maxlen=config.n_actions * config.grid_dim ** 2)
        sa_counts = {(s, a): 0 for s in range(config.grid_dim ** 2) for a in range(config.n_actions)}
        goalfound = False

        for ep in range(num_eps):
            state_int = init_state
            for i in range(max_steps):
                action = select_action_optimism(student, state_int, message, alpha, sa_counts, config, device)
                next_state_int, reward = outcomes[(state_int, action)]

                if next_state_int is None:
                    if not goalfound:
                        goalfound = True
                    break
                else:
                    memory_short.append(transition_index_dict[(state_int, action)])
                    if sa_counts[(state_int, action)] == 0:
                        memory_long.append(transition_index_dict[(state_int, action)])
                    sa_counts[(state_int, action)] += 1
                    state_int = next_state_int

                memory = torch.tensor(list(memory_short) + list(memory_long), device=device)
                optimize_dqn(student, optimizer, memory, batches_list_student, goalfound, loss_norm, config)

    return student
    
def run_evaluations(student, label_dict, wall_state_dict, message_dict, sopt_dict, config, device):
    
    student.eval()
    softy = nn.Softmax(dim=0)
    
    informed_rates = []
    misinformed_rates = []
    
    with torch.no_grad():
        for task, (wall_index, init_state, goal_state) in label_dict.items():
            wall_states = wall_state_dict[wall_index]
            
            G = graph_from_walls(wall_states, config)
            if not nx.is_connected(G):
                continue
            
            env = SquareGridworld(init_state, goal_state, wall_states, config)
            outcomes = env.get_outcomes()
            next_states_dict = {s: [outcomes[(s,a)][0] for a in range(config.n_actions)] for s in range(config.grid_dim**2)}
            
            sopt = sopt_dict[task]
            max_steps = 2 * sopt
            
            correct_message = message_dict[(wall_index, init_state, goal_state)]
            wrong_task = random.choice([t for t in label_dict if t != task])
            wrong_wall, wrong_init, wrong_goal = label_dict[wrong_task]
            wrong_message = message_dict[(wrong_wall, wrong_init, wrong_goal)]
            
            for message, rates_list in [(correct_message, informed_rates), (wrong_message, misinformed_rates)]:
                
                # get action probabilities for every state
                action_probas = torch.zeros(config.n_actions, config.grid_dim, config.grid_dim)
                for s in range(config.grid_dim**2):
                    state = state_int_to_tuple(s, config, device)
                    q_values = student(torch.cat((state[0], message), 0))
                    action_probas[:, s // config.grid_dim, s % config.grid_dim] = softy(q_values)
                
                # build transition matrix
                matrix = torch.zeros(config.grid_dim**2, config.grid_dim**2)
                for s in range(config.grid_dim**2):
                    if s == goal_state:
                        matrix[s, s] = 1
                    else:
                        for a, ns in enumerate(next_states_dict[s]):
                            matrix[ns, s] += action_probas[a, s // config.grid_dim, s % config.grid_dim]
                
                # compute state occupancy probabilities
                probas = torch.zeros(config.grid_dim**2)
                probas[init_state] = 1
                for _ in range(max_steps):
                    probas = matrix @ probas
                
                rates_list.append(probas[goal_state].item())
    
    print(f"Informed student:    {100*sum(informed_rates)/len(informed_rates):.2f}%")
    print(f"Misinformed student: {100*sum(misinformed_rates)/len(misinformed_rates):.2f}%")
    
    return informed_rates, misinformed_rates

#######################################  HELPER FUNCTIONS  ##########################################

def run_episode(student, message, init_state, outcomes, max_steps, config, device):
    state_int = init_state
    with torch.no_grad():
        for t in range(max_steps):
            state = state_int_to_tuple(state_int, config, device)
            input = torch.cat((state[0], message), 0)
            action = student(input).argmax().item()
            next_state_int, reward = outcomes[(state_int, action)]
            if next_state_int is None:
                return 1  # reached goal
            state_int = next_state_int
    return 0  # did not reach goal

def run_random_episode(init_state,  outcomes, max_steps, config, smart=False):
    state_int = init_state
    for t in range(max_steps):
        if smart:
            # only pick actions that don't hit walls
            valid_actions = [a for a in range(config.n_actions) 
                           if outcomes[(state_int, a)][0] != state_int]
            action = random.choice(valid_actions) if valid_actions else random.choice(range(config.n_actions))
        else:
            action = random.choice(range(config.n_actions))
        next_state_int, reward = outcomes[(state_int, action)]
        if next_state_int is None:
            return 1
        state_int = next_state_int
    return 0