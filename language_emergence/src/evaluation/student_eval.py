# src/evaluation/student_eval.py
import math as mt
import random
from typing import Callable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from configs.experiment_config import ExperimentConfig
from src.models.dqn import DQN
from src.rl.teacher import q_matrix_from_network
from src.utils.graph_utils import graph_from_walls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_probas_transformer(
    env,
    goal_state: int,
    n_states: int,
    n_actions: int,
    device: torch.device,
    n_samples: int = 200,
) -> torch.Tensor:
    """
    Build the (n_actions * n_states, n_states²) transformer used to convert a
    flat action-probability vector into a state-to-state transition matrix.
    """
    trans_probs = env.get_transition_probs(n_samples=n_samples)
    T = torch.zeros(n_actions * n_states, n_states * n_states, device=device)

    for s in range(n_states):
        if s == goal_state:
            for a in range(n_actions):
                T[a * n_states + s, s * n_states + s] = 1.0
        else:
            for a in range(n_actions):
                for ns in range(n_states):
                    p = trans_probs[s, a, ns].item()
                    if p > 0.0:
                        T[a * n_states + s, s * n_states + ns] = p
    return T


def _build_all_states_tensor(
    env,
    n_states: int,
    config: ExperimentConfig,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([
        env.state_int_to_tuple(s, config, device)[0]
        for s in range(n_states)
    ])


# ---------------------------------------------------------------------------
# Training without feedback
# ---------------------------------------------------------------------------

def train_student(
    label_dict: dict,
    wall_state_dict: dict,
    sopt_dict: dict,
    message_dict: dict,
    config: "ExperimentConfig",
    device: torch.device,
    make_env: Callable,
    num_eps: int = 750,
    max_steps: int = 50,
    alpha: float = 16.0,
    n_transition_samples: int = 200,
) -> nn.Module:
    # Initialize Student Model (DQN)
    student = DQN(K=config.K, n_actions=config.n_actions, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr_teacher)
    
    kappa = getattr(config, 'kappa', 0.05)
    n_states = config.grid_dim ** 2

    print("Precomputing task environments...")
    tasks_data = []
    
    for task_id, task_params in tqdm(label_dict.items(), desc="Precomputing tasks"):
        # Flexible unpacking: wall_index, init, goal, [order_mode]
        wall_index, init_state, goal_state = task_params[:3]
        
        env = make_env(task_params, wall_state_dict, config)
        
        T = _build_probas_transformer(
            env, goal_state, n_states, config.n_actions, device,
            n_samples=n_transition_samples,
        )
        
        all_states = _build_all_states_tensor(env, n_states, config, device)

        # Flexible key matching for message_dict
        msg_key = tuple(task_params) # Matches either 3-tuple or 4-tuple keys
        if msg_key not in message_dict:
            # Fallback for safety to first 3 elements
            msg_key = tuple(task_params[:3])
            if msg_key not in message_dict:
                continue
                
        message = message_dict[msg_key]

        tasks_data.append({
            'init_state': init_state,
            'goal_state': goal_state,
            'opt_steps': sopt_dict.get(task_id, max_steps),
            'message': message,
            'all_states': all_states,
            'probas_transformer': T,
        })

    print(f"Training student on {len(tasks_data)} tasks...")

    for ep in tqdm(range(num_eps), desc="Student epochs"):
        random.shuffle(tasks_data)
        optimizer.zero_grad(set_to_none=True)

        for d in tasks_data:
            # --- ADAPTER BLOCK: Handle 1D vs 2D States ---
            states = d['all_states']
            if states.shape[1] == 1:
                grid_dim = config.grid_dim
                indices = states.view(-1).long()
                x = (indices % grid_dim).float() / (grid_dim - 1)
                y = (indices // grid_dim).float() / (grid_dim - 1)
                states = torch.stack([x, y], dim=1)
            # ---------------------------------------------

            msg_rep = d['message'].unsqueeze(0).expand(states.shape[0], -1)
            Q = student(torch.cat([states, msg_rep], dim=1))

            action_probas_flat = (
                F.softmax(Q, dim=1)
                .t().flatten().unsqueeze(0)
            )
            
            matrix_big = (
                (action_probas_flat @ d['probas_transformer'])
                .view(n_states, n_states).t()
            )
            
            goal_proba = torch.linalg.matrix_power(
                matrix_big, d['opt_steps']
            )[d['goal_state'], d['init_state']]

            loss = (
                (1 - kappa) * (1 - goal_proba) ** 4
                + (kappa / mt.sqrt(n_states * config.n_actions)) * torch.norm(Q, 2)
            )
            loss.backward()

        optimizer.step()
        
        if (ep + 1) % 50 == 0:
            print(f"    Epoch {ep + 1}/{num_eps} complete")

    return student


# ---------------------------------------------------------------------------
# Joint training with feedback
# ---------------------------------------------------------------------------

def train_student_with_feedback(
    label_dict: dict,
    wall_state_dict: dict,
    sopt_dict: dict,
    q_matrix_dict: dict,
    autoencoder: nn.Module,
    student: DQN,
    config: ExperimentConfig,
    device: torch.device,
    make_env: Callable,
    num_eps: int = 750,
    max_steps: int = 50,
    n_transition_samples: int = 200,
) -> tuple:
    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(student.parameters()),
        lr=config.lr_teacher,
    )
    
    kappa = getattr(config, 'kappa', 0.05)
    gamma_sparse = getattr(config, 'gamma_sparse', 0.1)
    zeta_std = getattr(config, 'zeta_std', 5.0) 
    n_states = config.grid_dim ** 2

    autoencoder.train()
    student.train()

    print("Precomputing task environments...")
    tasks_data = []

    for task_key, task_params in tqdm(label_dict.items(), desc="Precomputing tasks"):
        wall_index, init_state, goal_state = task_params[:3]
        wall_states = wall_state_dict[wall_index]

        try:
            G = graph_from_walls(wall_states, config)
            if not nx.is_connected(G):
                continue
        except Exception:
            pass

        env = make_env(task_params, wall_state_dict, config)
        T = _build_probas_transformer(
            env, goal_state, n_states, config.n_actions, device,
            n_samples=n_transition_samples,
        )
        all_states = _build_all_states_tensor(env, n_states, config, device)
        
        # Use the direct dictionary key to fetch Q-matrix
        original_q = q_matrix_dict[task_key].unsqueeze(0).to(device)

        tasks_data.append({
            'init_state': init_state,
            'goal_state': goal_state,
            'opt_steps': sopt_dict.get(task_key, max_steps),
            'original_q': original_q,
            'all_states': all_states,
            'probas_transformer': T,
        })

    print(f"Joint training on {len(tasks_data)} tasks...")

    for ep in tqdm(range(num_eps), desc="Joint training epochs"):
        random.shuffle(tasks_data)
        optimizer.zero_grad(set_to_none=True)

        for d in tasks_data:
            message, q_recon = autoencoder(d['original_q'])
            flat_msg = message.view(1, -1) 

            # --- ADAPTER BLOCK: Handle 1D vs 2D States ---
            states = d['all_states']
            if states.shape[1] == 1:
                grid_dim = config.grid_dim
                indices = states.view(-1).long()
                x = (indices % grid_dim).float() / (grid_dim - 1)
                y = (indices // grid_dim).float() / (grid_dim - 1)
                states = torch.stack([x, y], dim=1)
            # ---------------------------------------------

            msg_rep = flat_msg.expand(n_states, -1)
            Q = student(torch.cat([states, msg_rep], dim=1))

            action_probas_flat = (
                torch.nn.functional.softmax(Q, dim=1)
                .t().flatten().unsqueeze(0)
            )
            matrix_big = (
                (action_probas_flat @ d['probas_transformer'])
                .view(n_states, n_states).t()
            )
            
            goal_proba = torch.linalg.matrix_power(
                matrix_big, d['opt_steps']
            )[d['goal_state'], d['init_state']]

            student_loss = (
                (1 - kappa) * (1 - goal_proba) ** 4
                + (kappa / mt.sqrt(n_states * config.n_actions)) * torch.norm(Q, 2)
            )
            
            recon_loss = torch.norm(d['original_q'] - q_recon, 2)
            sparse_loss = torch.norm(message, 1)

            batch_loss = (
                (1 - gamma_sparse) * recon_loss
                + gamma_sparse * sparse_loss
                + zeta_std * student_loss
            )
            
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (ep + 1) % 50 == 0:
            print(f"    Epoch {ep + 1}/{num_eps} completed")

    return autoencoder, student


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluations(
    student: DQN,
    label_dict: dict,
    wall_state_dict: dict,
    message_dict: dict,
    sopt_dict: dict,
    config: ExperimentConfig,
    device: torch.device,
    make_env: Callable,
    n_transition_samples: int = 200,
) -> tuple:
    student.eval()
    n_states = config.grid_dim ** 2
    informed_rates, misinformed_rates = [], []
    
    with torch.no_grad():
        for task_key, task_params in label_dict.items():
            wall_index, init_state, goal_state = task_params[:3]
            wall_states = wall_state_dict[wall_index]

            try:
                G = graph_from_walls(wall_states, config)
                if not nx.is_connected(G):
                    continue
            except Exception:
                pass

            env = make_env(task_params, wall_state_dict, config)
            trans_probs = env.get_transition_probs(n_samples=n_transition_samples)

            sopt = sopt_dict[task_key]
            max_steps = 2 * sopt

            # Standardized key lookup based on full task params
            msg_key = tuple(task_params)
            if msg_key not in message_dict:
                msg_key = tuple(task_params[:4])
            
            correct_msg = message_dict[msg_key].to(device)
            
            # Select a random message from the pool of available messages
            wrong_msg_key = random.choice(list(message_dict.keys()))
            while wrong_msg_key == msg_key:
                wrong_msg_key = random.choice(list(message_dict.keys()))
            wrong_msg = message_dict[wrong_msg_key].to(device)

            for message, rate_list in ((correct_msg, informed_rates),
                                       (wrong_msg, misinformed_rates)):
                action_probs = torch.zeros(n_states, config.n_actions)
                for s in range(n_states):
                    state_vec = env.state_int_to_tuple(s, config, device)[0]
                    
                    # Coordinate adapter for consistency with forward logic
                    if state_vec.shape[0] == 1:
                        grid_dim = config.grid_dim
                        idx = state_vec.long().item()
                        state_vec = torch.tensor([idx % grid_dim, idx // grid_dim], 
                                                device=device).float() / (grid_dim - 1)
                    
                    q_vals = student(
                        torch.cat((state_vec, message), dim=0).unsqueeze(0)
                    ).squeeze()
                    action_probs[s] = torch.softmax(q_vals, dim=0)

                matrix = torch.zeros(n_states, n_states)
                for s in range(n_states):
                    if s == goal_state:
                        matrix[s, s] = 1.0
                    else:
                        for a in range(config.n_actions):
                            for ns in range(n_states):
                                matrix[ns, s] += (
                                    action_probs[s, a] * trans_probs[s, a, ns]
                                )

                probas = torch.zeros(n_states)
                probas[init_state] = 1.0
                for _ in range(max_steps):
                    probas = matrix @ probas

                rate_list.append(probas[goal_state].item())

    print(f"Informed student:     {100 * sum(informed_rates) / len(informed_rates) if informed_rates else 0:.2f}%")
    print(f"Misinformed student: {100 * sum(misinformed_rates) / len(misinformed_rates) if misinformed_rates else 0:.2f}%")
    return informed_rates, misinformed_rates

def run_evaluations_inventory(
    student: DQN,
    label_dict: dict,
    wall_state_dict: dict,
    message_dict: dict,
    sopt_dict: dict,
    config: ExperimentConfig,
    device: torch.device,
    make_env: Callable,
    n_transition_samples: int = 200,
) -> tuple:
    student.eval()
    n_states = config.grid_dim ** 2
    n_actions = config.n_actions
    informed_rewards, misinformed_rewards = [], []
    
    with torch.no_grad():
        # Pre-build a batch of "grid-adapted" state vectors for the student network
        # This avoids the slow Python loop for coordinate conversion during evaluation
        raw_states = torch.arange(n_states, device=device)
        grid_dim = config.grid_dim
        x = (raw_states % grid_dim).float() / (grid_dim - 1)
        y = (raw_states // grid_dim).float() / (grid_dim - 1)
        batched_state_vecs = torch.stack([x, y], dim=1) # (n_states, 2)

        for task_key, task_params in tqdm(label_dict.items(), desc="Evaluating Inventory"):
            env = make_env(task_params, wall_state_dict, config)
            
            # Use the environment's built-in tensor methods
            trans_probs = env.get_transition_probs(n_samples=n_transition_samples).to(device)
            horizon = sopt_dict.get(task_key, 50)
            
            # Message lookup
            msg_key = tuple(task_params)
            if msg_key not in message_dict:
                msg_key = tuple(task_params[:4])
            
            correct_msg = message_dict[msg_key].to(device)
            
            # Select random wrong message
            all_keys = list(message_dict.keys())
            wrong_msg_key = random.choice(all_keys)
            while wrong_msg_key == msg_key and len(all_keys) > 1:
                wrong_msg_key = random.choice(all_keys)
            wrong_msg = message_dict[wrong_msg_key].to(device)

            for message, reward_list in ((correct_msg, informed_rewards),
                                         (wrong_msg, misinformed_rewards)):
                
                # 1. Vectorized Policy Generation
                # Shape: (n_states, message_dim)
                msg_rep = message.unsqueeze(0).expand(n_states, -1)
                # Single forward pass for all states
                q_vals = student(torch.cat([batched_state_vecs, msg_rep], dim=1))
                policy = torch.softmax(q_vals, dim=1) # (n_states, n_actions)

                # 2. Build Expected Reward Vector and Transition Matrix (Vectorized)
                # Precompute expected rewards for all (s, a) pairs
                # Note: For speed, we use the env's method. 
                # If this is still slow, precompute this inside get_transition_probs
                r_sa = torch.tensor([[env.get_expected_reward(s, a) 
                                     for a in range(n_actions)] 
                                     for s in range(n_states)], device=device)
                
                # E[r|s] = sum_a pi(a|s) * R(s,a)
                expected_reward_vec = (policy * r_sa).sum(dim=1)

                # Transition Matrix M: M[ns, s] = sum_a P(ns | s, a) * pi(a | s)
                # trans_probs: (s, a, ns) -> policy: (s, a)
                # Using einsum for fast contraction: (s, a, ns), (s, a) -> (ns, s)
                matrix = torch.einsum('san,sa->ns', trans_probs, policy)

                # 3. Accumulated Reward Projection
                total_expected_reward = 0.0
                state_dist = torch.zeros(n_states, device=device)
                init_state = task_params[1] 
                state_dist[init_state] = 1.0
                
                for _ in range(horizon):
                    total_expected_reward += torch.dot(state_dist, expected_reward_vec)
                    state_dist = matrix @ state_dist

                reward_list.append(total_expected_reward.item())

    avg_inf = sum(informed_rewards) / len(informed_rewards) if informed_rewards else 0
    avg_mis = sum(misinformed_rewards) / len(misinformed_rewards) if misinformed_rewards else 0
    print(f"Informed Reward: {avg_inf:.2f} | Misinformed: {avg_mis:.2f}")
    
    return informed_rewards, misinformed_rewards

# ---------------------------------------------------------------------------
# Telephone-game helpers
# ---------------------------------------------------------------------------

def extract_student_messages(
    student: DQN,
    autoencoder: nn.Module,
    label_dict: dict,
    wall_state_dict: dict,
    message_dict: dict,
    config: ExperimentConfig,
    device: torch.device,
    make_env: Callable,
) -> dict:
    student.eval()
    autoencoder.eval()
    student_message_dict = {}

    with torch.no_grad():
        for task_key, task_params in label_dict.items():
            wall_index = task_params[0]
            wall_states = wall_state_dict[wall_index]
            env = make_env(task_params, wall_state_dict, config)
            
            # Standardized key lookup
            msg_key = tuple(task_params)
            if msg_key not in message_dict:
                msg_key = tuple(task_params[:4])
            
            message = message_dict[msg_key].to(device)

            q_student = q_matrix_from_network(
                student, message, wall_states, config, device, env
            )
            new_msg, _ = autoencoder(q_student.unsqueeze(0).to(device))
            student_message_dict[msg_key] = new_msg.squeeze(0).detach()

    return student_message_dict


def close_the_loop(
    student: DQN,
    autoencoder: nn.Module,
    label_dict: dict,
    wall_state_dict: dict,
    message_dict: dict,
    sopt_dict: dict,
    config: ExperimentConfig,
    device: torch.device,
    make_env: Callable,
    num_eps: int = 750,
) -> tuple:
    print("--- Closing the Loop: extracting student messages ---")
    student_message_dict = extract_student_messages(
        student, autoencoder, label_dict, wall_state_dict,
        message_dict, config, device, make_env,
    )
    print("--- Closing the Loop: training 2nd-generation student ---")
    student_gen2 = train_student(
        label_dict, wall_state_dict, sopt_dict, student_message_dict,
        config, device, make_env, num_eps=num_eps,
    )
    return student_gen2, student_message_dict


# ---------------------------------------------------------------------------
# Episode utilities
# ---------------------------------------------------------------------------

def run_episode(student, message, init_state, outcomes, max_steps, config, device, env):
    state_int = init_state
    student.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            state_vec = env.state_int_to_tuple(state_int, config, device)[0]
            
            # Coordinate adapter
            if state_vec.shape[0] == 1:
                grid_dim = config.grid_dim
                idx = state_vec.long().item()
                state_vec = torch.tensor([idx % grid_dim, idx // grid_dim], 
                                        device=device).float() / (grid_dim - 1)

            action = student(
                torch.cat((state_vec, message), dim=0).unsqueeze(0)
            ).argmax().item()
            
            next_state_int, _ = outcomes[(state_int, action)]
            if next_state_int is None:
                return 1
            state_int = next_state_int
    return 0


def run_random_episode(init_state, outcomes, max_steps, config, smart=False):
    state_int = init_state
    for _ in range(max_steps):
        if smart:
            valid = [a for a in range(config.n_actions)
                     if outcomes[(state_int, a)][0] != state_int]
            action = random.choice(valid) if valid else random.randrange(config.n_actions)
        else:
            action = random.randrange(config.n_actions)
        next_state_int, _ = outcomes[(state_int, action)]
        if next_state_int is None:
            return 1
        state_int = next_state_int
    return 0