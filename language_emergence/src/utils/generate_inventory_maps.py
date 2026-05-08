import os
import pickle
import random
from collections import deque
from itertools import combinations


# map configuration ---------------------------------------------------
UNITS_PER_ORDER = 4
LAVA = True
NUM_STATES = 16
NUM_ACTIONS = 4
DEMANDS = [1, 2, 3]
DATASET_ID = 'v1'
SEED = 42

# map generation ------------------------------------------------------
def check_reachability(initial_state, goal_state, wall_states, num_states, num_actions, demand, lava):

    wall_set = set(wall_states)
    visited_states = set()
    queue = deque([(initial_state, 0)])

    while queue:
        current_state, steps_taken = queue.popleft()

        if current_state == goal_state:
            return steps_taken
        
        if current_state in visited_states:
            continue
        
        visited_states.add(current_state)

        for action in range(num_actions):
            order_quantity = action * UNITS_PER_ORDER
            next_state_candidate = current_state + order_quantity - demand

            if lava: 
                if next_state_candidate < 0:
                    next_state = 0
                elif next_state_candidate >= num_states:
                    next_state = num_states - 1
                elif next_state_candidate in wall_set:
                    continue
                else:
                    next_state = next_state_candidate
            else:
                if next_state_candidate < 0 or next_state_candidate >= num_states or next_state_candidate in wall_set:
                    continue
                else:
                    next_state = next_state_candidate
                
            if next_state not in visited_states:
                queue.append((next_state, steps_taken + 1))    

    return None


def generate_wall_configs(num_states, max_walls, forbidden_states, min_walls=0):
    available_states = [state for state in range(num_states) if state not in forbidden_states]
    wall_configurations = []

    for num_walls in range(min_walls, max_walls + 1):
        if num_walls == 0:
            wall_configurations.append([])
        else:
            for combination in combinations(available_states, num_walls):
                wall_configurations.append(list(combination))

    return wall_configurations

def generate_inventory_maps(num_states, num_actions, min_walls, max_walls, demands, lava, dataset_id, seed):
    random.seed(seed)
 
    if demands is None:
        demands = [1, 2, 3, 4]
 
    initial_state = 0
    forbidden_states = {initial_state}
    all_wall_configurations = generate_wall_configs(num_states, max_walls, forbidden_states, min_walls)
 
    wall_state_dict = {}
    label_dict = {}
    wall_config_index = 0
    task_id = 0
    num_skipped = 0
 
    for wall_states in all_wall_configurations:
        wall_set = set(wall_states)
 
        possible_goal_states = [
            state for state in range(1, num_states)
            if state != initial_state and state not in wall_set
        ]
 
        config_has_valid_task = False
 
        for demand in demands:
            for goal_state in possible_goal_states:
                shortest_path_length = check_reachability(
                    initial_state, goal_state, wall_states,
                    num_states, num_actions, demand, lava
                )
 
                if shortest_path_length is None:
                    num_skipped += 1
                    continue
 
                if not config_has_valid_task:
                    wall_state_dict[wall_config_index] = wall_states
                    config_has_valid_task = True
 
                label_dict[task_id] = [wall_config_index, initial_state, goal_state, demand]
                task_id += 1
 
        if config_has_valid_task:
            wall_config_index += 1
 
    print(f"Generated {task_id} tasks across {wall_config_index} wall configurations")
    print(f"Skipped {num_skipped} unreachable tasks")
 
    return label_dict, wall_state_dict


def generate_and_save_all(num_states=16, num_actions=4, demands=None, lava=True, dataset_id='v1', seed=42):
    if demands is None:
        demands = [1, 2, 3]

    print("Generating training set (0 and 1 walls)...")
    train_label_dict, train_wall_state_dict = generate_inventory_maps(
        num_states=num_states, num_actions=num_actions, min_walls=0, max_walls=1,
        demands=demands, lava=lava, dataset_id=dataset_id, seed=seed
    )

    print("\nGenerating test set (exactly 2 walls)...")
    test_label_dict, test_wall_state_dict = generate_inventory_maps(
        num_states=num_states, num_actions=num_actions, min_walls=2, max_walls=2,
        demands=demands, lava=lava, dataset_id=dataset_id, seed=seed
    )

    train_label_dict, test_label_dict = validate_maps(
        train_label_dict, train_wall_state_dict,
        test_label_dict, test_wall_state_dict
    )

    save_maps(train_label_dict, train_wall_state_dict, dataset_id, split='train')
    save_maps(test_label_dict, test_wall_state_dict, dataset_id, split='test')

def save_maps(label_dict, wall_state_dict, dataset_id, split):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
    os.makedirs(os.path.join(base_path, 'inventory_labels'), exist_ok=True)

    label_path = os.path.join(base_path, 'inventory_labels', f'inventory_labels_{dataset_id}_{split}.pkl')
    wall_path = os.path.join(base_path, 'inventory_labels', f'wall_states_{dataset_id}_{split}.pkl')

    with open(label_path, 'wb') as f:
        pickle.dump(label_dict, f)
    with open(wall_path, 'wb') as f:
        pickle.dump(wall_state_dict, f)

    print(f"Saved label_dict -> {label_path}")
    print(f"Saved wall_state_dict -> {wall_path}")


def load_maps(dataset_id, split):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
    label_path = os.path.join(base_path, 'inventory_labels', f'inventory_labels_{dataset_id}_{split}.pkl')
    wall_path = os.path.join(base_path, 'inventory_labels', f'wall_states_{dataset_id}_{split}.pkl')

    with open(label_path, 'rb') as f:
        label_dict = pickle.load(f)
    with open(wall_path, 'rb') as f:
        wall_state_dict = pickle.load(f)

    return label_dict, wall_state_dict

# map validation ------------------------------------------------------
def validate_maps(train_label_dict, train_wall_state_dict, test_label_dict, test_wall_state_dict):
    print("\n--- Validation ---")
    any_warnings = False

    # check to make sure no wall config appears in both train and test
    train_wall_configs = {
        tuple(sorted(walls)) for walls in train_wall_state_dict.values()
    }
    test_wall_configs = {
        tuple(sorted(walls)) for walls in test_wall_state_dict.values()
    }
    overlapping_wall_configs = train_wall_configs & test_wall_configs
    if overlapping_wall_configs:
        print(f"{len(overlapping_wall_configs)} wall configs appear in both train and test sets")
        any_warnings = True

    # check no duplicate tasks within train set
    train_task_signatures = {}
    train_duplicates = []
    for task_id, (wall_config_index, initial_state, goal_state, demand) in train_label_dict.items():
        wall_config = tuple(sorted(train_wall_state_dict[wall_config_index]))
        signature = (wall_config, initial_state, goal_state, demand)
        if signature in train_task_signatures:
            train_duplicates.append((task_id, train_task_signatures[signature]))
        else:
            train_task_signatures[signature] = task_id
    if train_duplicates:
        print(f"{len(train_duplicates)} duplicate tasks found in train set. removing")
        for duplicate_task_id, _ in train_duplicates:
            del train_label_dict[duplicate_task_id]
        any_warnings = True

    # check to make sure no duplicate tasks within test set
    test_task_signatures = {}
    test_duplicates = []
    for task_id, (wall_config_index, initial_state, goal_state, demand) in test_label_dict.items():
        wall_config = tuple(sorted(test_wall_state_dict[wall_config_index]))
        signature = (wall_config, initial_state, goal_state, demand)
        if signature in test_task_signatures:
            test_duplicates.append((task_id, test_task_signatures[signature]))
        else:
            test_task_signatures[signature] = task_id
    if test_duplicates:
        print(f"({len(test_duplicates)} duplicate tasks found in test set. removing")
        for duplicate_task_id, _ in test_duplicates:
            del test_label_dict[duplicate_task_id]
        any_warnings = True

    # check that. no task appears in both train and test
    overlapping_tasks = set(train_task_signatures.keys()) & set(test_task_signatures.keys())
    if overlapping_tasks:
        print(f"{len(overlapping_tasks)} task appear in both train and test. removing from test")
        for signature in overlapping_tasks:
            duplicate_task_id = test_task_signatures[signature]
            del test_label_dict[duplicate_task_id]
        any_warnings = True

    # chack that all wall config indices in label_dict exist in wall_state_dict
    for split_name, label_dict, wall_state_dict in [
        ('train', train_label_dict, train_wall_state_dict),
        ('test',  test_label_dict,  test_wall_state_dict)
    ]:
        missing_indices = {
            wall_config_index
            for wall_config_index, _, _, _ in label_dict.values()
            if wall_config_index not in wall_state_dict
        }
        if missing_indices:
            print(f"{split_name} label_dict references {len(missing_indices)} missing wall configindices")
            any_warnings = True

    if not any_warnings:
        print("All checks passed")

    print(f"Train: {len(train_label_dict)} tasks, {len(train_wall_state_dict)} wall configs")
    print(f"Test:  {len(test_label_dict)} tasks, {len(test_wall_state_dict)} wall configs")
    print(f"Ratio: 1:{len(test_label_dict) / len(train_label_dict):.1f} tasks, "
          f"1:{len(test_wall_state_dict) / len(train_wall_state_dict):.1f} wall configs")

    return train_label_dict, test_label_dict


if __name__ == '__main__':
    generate_and_save_all()