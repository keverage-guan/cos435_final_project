import os
import sys
import pickle
import torch
import numpy as np
from dataclasses import dataclass
from scipy.stats import poisson, binom, nbinom

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.experiment_config import ExperimentConfig

# ---------------------------------------------------------------------------
# Demand distribution descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemandDistributionConfig:
    dist_type: str
    params: dict
    category: str

    def __repr__(self):
        param_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{self.dist_type}({param_str})'

def _d(dist_type, category, **params):
    return DemandDistributionConfig(dist_type=dist_type, params=params, category=category)

DEMAND_DISTRIBUTIONS = [
    _d('poisson', 'low', lam=1), _d('poisson', 'low', lam=2),
    _d('uniform', 'low', low=0, high=2), _d('binomial', 'low', n=4, p=0.2),
    _d('poisson', 'medium', lam=3), _d('poisson', 'medium', lam=4),
    _d('binomial', 'medium', n=8, p=0.4), _d('uniform', 'medium', low=1, high=5),
    _d('binomial', 'medium', n=6, p=0.5), _d('poisson', 'high', lam=6),
    _d('poisson', 'high', lam=8), _d('uniform', 'high', low=3, high=8),
    _d('uniform', 'high', low=4, high=10), _d('negative_binomial', 'bursty', n=3, p=0.4),
    _d('negative_binomial', 'bursty', n=2, p=0.3), _d('zero_inflated_poisson', 'bursty', lam=4, pi=0.3),
    _d('zero_inflated_poisson', 'bursty', lam=2, pi=0.4), _d('binomial', 'symmetric', n=10, p=0.5),
    _d('uniform', 'symmetric', low=2, high=6), _d('uniform', 'symmetric', low=0, high=8),
]

ORDER_MODES = {
    'medium': [2, 4, 6],     # ID 0
    'big': [6, 9, 12],       # ID 1
    'small_big': [2, 6, 10], # ID 2
    'fine': [1, 2, 3],       # ID 3
    'coarse': [4, 8, 12],    # ID 4
    'skewed': [1, 3, 8],     # ID 5
}

ORDER_MODE_TO_ID = {name: i for i, name in enumerate(ORDER_MODES.keys())}

GRID_SIZE = 4
NUM_STATES = GRID_SIZE ** 2
INIT_STATES = list(range(NUM_STATES))
GOAL_STATE = -1

# ---------------------------------------------------------------------------
# Optimized Solver
# ---------------------------------------------------------------------------

def get_demand_pmf(dist_cfg, max_demand=32):
    d_vals = np.arange(max_demand + 1)
    p = dist_cfg.params
    if dist_cfg.dist_type == 'poisson': probs = poisson.pmf(d_vals, p['lam'])
    elif dist_cfg.dist_type == 'uniform':
        mask = (d_vals >= p['low']) & (d_vals <= p['high'])
        probs = np.where(mask, 1.0 / (p['high'] - p['low'] + 1), 0.0)
    elif dist_cfg.dist_type == 'binomial': probs = binom.pmf(d_vals, p['n'], p['p'])
    elif dist_cfg.dist_type == 'negative_binomial': probs = nbinom.pmf(d_vals, p['n'], p['p'])
    elif dist_cfg.dist_type == 'zero_inflated_poisson':
        probs = (1 - p['pi']) * poisson.pmf(d_vals, p['lam'])
        probs[0] += p['pi']
    else: probs = np.zeros_like(d_vals); probs[0] = 1.0
    return torch.from_numpy(probs / probs.sum()).float()

def solve_environment_q_matrix(demand_cfg, order_quantities, goal_state=-1, num_states=16, gamma=0.99, tol=1e-6):
    config = ExperimentConfig()
    pmf = get_demand_pmf(demand_cfg)
    max_d = len(pmf) - 1
    
    order_sizes = torch.tensor([0] + order_quantities).view(-1, 1, 1) 
    states = torch.arange(num_states).view(1, -1, 1)
    demands = torch.arange(max_d + 1).view(1, 1, -1)
    
    available = states + order_sizes
    next_states = torch.clamp(available - demands, 0, num_states - 1)
    unmet = torch.clamp(demands - available, min=0)
    
    rewards = config.step_reward - (0.5 * next_states.float()) - (2.0 * unmet.float())
    
    V = torch.zeros(num_states)
    has_goal = (0 <= goal_state < num_states)

    for _ in range(1000):
        # If no goal, v_next is always gamma * V. If goal, use terminal reward.
        if has_goal:
            is_goal_mask = (next_states == goal_state)
            v_next = torch.where(is_goal_mask, torch.tensor(float(config.goal_reward)), gamma * V[next_states])
        else:
            v_next = gamma * V[next_states]

        q_sa = torch.sum(pmf * (rewards + v_next), dim=2)
        V_new = torch.max(q_sa, dim=0)[0]
        
        if has_goal:
            V_new[goal_state] = config.goal_reward
            
        if torch.dist(V, V_new) < tol:
            V = V_new
            break
        V = V_new

    # Final Q assembly
    if has_goal:
        final_v_next = torch.where(next_states == goal_state, torch.tensor(float(config.goal_reward)), gamma * V[next_states])
    else:
        final_v_next = gamma * V[next_states]
        
    q_matrix = torch.sum(pmf * (rewards + final_v_next), dim=2)
    return q_matrix.view(1, 8, 8)

# ---------------------------------------------------------------------------
# Data Saving & Execution
# ---------------------------------------------------------------------------

def is_train_task(demand_cfg, order_mode) -> bool:
    train_dist_types = {'poisson', 'uniform'}
    train_categories = {'low', 'medium', 'high'}
    train_modes = {'medium', 'fine'}
    return (demand_cfg.dist_type in train_dist_types and 
            demand_cfg.category in train_categories and
            order_mode in train_modes)

def save_dataset(tasks: list, split: str):
    base_data_path = os.path.join(PROJECT_ROOT, 'src', 'data', 'inventory')
    dirs = {
        'lab': os.path.join(base_data_path, 'label dictionaries'),
        'env': os.path.join(base_data_path, 'env dictionaries'),
        'qma': os.path.join(base_data_path, 'q matrix dictionaries'),
        'wal': os.path.join(base_data_path, 'wall state dictionaries')
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    label_dict, env_dict, qmat_dict = {}, {}, {}
    env_cache = {} 

    for new_id, t in enumerate(tasks):
        # Add a 0 at the beginning to act as the wall_index
        label_dict[new_id] = (
            0,                                     # wall_index (placeholder)
            t['init_state'],                       # init_state
            t['goal_state'],                       # goal_state
            ORDER_MODE_TO_ID[t['order_mode_name']] # order_mode_id
        )
        
        elbl = t['env_label']
        if elbl not in env_cache:
            q_mat = solve_environment_q_matrix(t['demand_dist'], t['order_quantities'])
            env_cache[elbl] = q_mat
            env_dict[elbl] = {
                'demand_dist': t['demand_dist'], 
                'order_quantities': t['order_quantities'],
                'category': t['demand_dist'].category
            }
        
        qmat_dict[new_id] = env_cache[elbl]

    # Matching the requested code pattern: {config.qmat_read_code}
    code = f"{split}_{GRID_SIZE}x{GRID_SIZE}"
    
    # Save Labels
    with open(os.path.join(dirs['lab'], f"q_matrices_labels{code}.pkl"), 'wb') as f:
        pickle.dump(label_dict, f)

    # Save Wall States (Empty dummy to satisfy file-loading requirements)
    with open(os.path.join(dirs['wal'], f"wall_states{code}.pkl"), 'wb') as f:
        pickle.dump({0: []}, f)

    # Save Q-Matrices
    with open(os.path.join(dirs['qma'], f"q_matrices{code}.pkl"), 'wb') as f:
        pickle.dump(qmat_dict, f)

    # Metadata
    with open(os.path.join(dirs['env'], f"inventory_envs{code}.pkl"), 'wb') as f:
        pickle.dump(env_dict, f)
            
    print(f"Saved {split} dataset with {len(tasks)} tasks.")

def generate_and_save_all():
    train_tasks, test_tasks = [], []
    for d_idx, demand_cfg in enumerate(DEMAND_DISTRIBUTIONS):
        for o_mode, o_quantities in ORDER_MODES.items():
            env_label = f"d{d_idx}_{o_mode}"
            for init_state in INIT_STATES:
                task = {
                    'env_label': env_label, 
                    'demand_dist': demand_cfg, 
                    'order_quantities': list(o_quantities), 
                    'order_mode_name': o_mode,
                    'init_state': init_state, 
                    'goal_state': GOAL_STATE
                }
                if is_train_task(demand_cfg, o_mode):
                    train_tasks.append(task)
                else:
                    test_tasks.append(task)

    save_dataset(train_tasks, 'training')
    save_dataset(test_tasks, 'test')

if __name__ == '__main__':
    generate_and_save_all()