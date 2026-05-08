"""
Procedurally generates the teacher data PKLs (wall state dictionaries and
label dictionaries) from scratch, removing any dependency on the upstream
repo's data.zip.

Training set (225 tasks):
  - 16 wall configs: 1 empty + 15 single-wall (state 1 through 15)
  - Init state: always 0 (config.student_init)
  - Goal state: any state that is not 0, not a wall, and reachable from 0

Test set (1,313 tasks):
  - 105 wall configs: all C(15, 2) two-wall pairs from states {1 ... 15}
  - Same init/goal validity rules as training
"""

import os
import pickle
from itertools import combinations

import networkx as nx

from configs.experiment_config import ExperimentConfig
from src.utils.graph_utils import graph_from_walls


def generate_wall_configs(split: str, grid_dim: int = 4) -> 'dict[int, list[int]]':
    """
    Return a dict mapping wall_index → list of blocked states.

    split='training': index 0 = no walls, indices 1-15 = single wall at state 1-15
    split='test':     indices 0-104 = all C(15,2) two-wall pairs from states 1-15
    """
    states = list(range(1, grid_dim ** 2))  # [1, 2, ..., 15]

    if split == 'training':
        configs = {0: []}
        for i, s in enumerate(states, start=1):
            configs[i] = [s]
        return configs

    if split == 'test':
        return {i: list(pair) for i, pair in enumerate(combinations(states, 2))}

    raise ValueError(f"split must be 'training' or 'test', got {split!r}")


def generate_label_dict(
    wall_state_dict: 'dict[int, list[int]]',
    config: ExperimentConfig,
) -> 'dict[int, list]':
    """
    Enumerate all valid tasks given a wall config dict.

    A task (wall_idx, init_state, goal_state) is valid when:
      - goal_state != student_init (init and goal differ)
      - goal_state not in wall_states (goal is accessible)
      - the gridworld graph is connected (no isolated regions)

    Returns {task_idx: [wall_idx, init_state, goal_state]}
    """
    label_dict = {}
    task_idx   = 0
    init_state = config.student_init  # always 0

    for wall_idx, wall_states in wall_state_dict.items():
        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        for goal_state in range(config.grid_dim ** 2):
            if goal_state == init_state:
                continue
            if goal_state in wall_states:
                continue
            label_dict[task_idx] = [wall_idx, init_state, goal_state]
            task_idx += 1

    return label_dict


def save_teacher_data(base_dir: str, config: ExperimentConfig) -> None:
    """
    Generate and save the 4 non-Q-matrix PKLs to their canonical locations
    under data/teacher/.
    """
    wall_dir  = os.path.join(base_dir, 'data', 'teacher', 'wall state dictionaries')
    label_dir = os.path.join(base_dir, 'data', 'teacher', 'label dictionaries')
    os.makedirs(wall_dir,  exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for split, code in [('training', 'training_4x4'), ('test', 'test_4x4')]:
        wall_state_dict = generate_wall_configs(split, grid_dim=config.grid_dim)
        label_dict      = generate_label_dict(wall_state_dict, config)

        wall_path  = os.path.join(wall_dir,  f'wall_states{code}.pkl')
        label_path = os.path.join(label_dir, f'q_matrices_labels{code}.pkl')

        with open(wall_path, 'wb') as f:
            pickle.dump(wall_state_dict, f)
        with open(label_path, 'wb') as f:
            pickle.dump(label_dict, f)

        print(f"  [{split}] {len(wall_state_dict)} wall configs, "
              f"{len(label_dict)} tasks → saved to data/teacher/")
