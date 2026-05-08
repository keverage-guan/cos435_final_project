import os
import pickle
import random
from collections import deque
from itertools import combinations

UNITS_PER_ORDER = 4

def check_reachability(init_state, goal_state, wall_states, n_states, n_actions, demand, lava):

    wall_set = set(wall_states)
    visited = set()
    queue = deque([(init_state, 0)])

    while queue:
        state, steps = queue.popleft()

        if state == goal_state:
            return steps
        
        if state in visited:
            continue
        
        visited.add(state)

        for action in range(n_actions):
            order_quantity = action * UNITS_PER_ORDER
            candidate = state + order_quantity - demand

            if lava: 
                if candidate < 0:
                    next_state = 0
                elif candidate >= n_states:
                    next_state = n_states -1
                else:
                    next_state = candidate
            else:
                if candidate < 0 or candidate >= n_states or candidate in wall_set:
                    next_state = state
                    continue
                else:
                    next_state = state
                
        if next_state not in visited:
            queue.append((next_state, steps + 1))    

    return None

def generate_wall_configs(n_states, max_walls, forbidden):
    return

def generate_inventory_maps(n_states, n_actions, max_walls, demands, lava, id, seed):
    return