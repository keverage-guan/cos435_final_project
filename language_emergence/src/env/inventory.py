import torch
from typing import Optional

from configs.experiment_config import ExperimentConfig


class InventoryManagement():
    """
    grid_dim**2 states (inventory levels 0..grid_dim²-1)
    n_actions actions (discrete order quantities 0..n_actions-1)

    State s represents having s units of stock on hand.

    Actions 0..n_actions-1 represent order quantities:
        action a orders a * order_step units, where order_step = grid_dim² // n_actions

    - Agent starts at init_state (initial stock level)
    - Reaching goal_state (target stock level) gives goal_reward and ends the episode
    - wall_states are forbidden inventory levels (e.g., overstock thresholds)
    - lava=False: invalid transitions (stockout/overflow/wall) bounce back to current state
    - lava=True: wall_states are accessible; only the map boundary (0 and n_states-1) cannot be crossed
    - Fixed demand per period: config.demand units are consumed each step after ordering
    - Each step incurs step_reward (e.g. holding cost); hitting a wall incurs wall_reward (e.g. stockout penalty)
    ---
    NOTE: requires config.demand (int) — a new field not present in the gridworld config.
    """

    def __init__(self, init_state: int, goal_state: int, wall_states: 'list[int]', config: ExperimentConfig):
        self.init_state: int = init_state
        self.goal_state: int = goal_state
        self.wall_states: 'list[int]' = wall_states
        self.n_states: int = config.grid_dim ** 2   # total inventory levels
        self.n_actions: int = config.n_actions       # number of distinct order sizes
        self.lava: bool = config.lava
        self.step_reward: float = config.step_reward
        self.goal_reward: float = config.goal_reward
        self.wall_reward: float = config.wall_reward
        self.demand: int = config.demand             # fixed units demanded per period
        # action a orders a * order_step units; step sized so max order reaches ~full capacity
        self.order_step: int = max(1, self.n_states // self.n_actions)

    def get_outcome(self, state: int, action: int) -> 'tuple[Optional[int], float]':
        '''
        Given an inventory level and an order action, returns the next inventory level and reward.

        Transition: next_state = state + order_qty - demand
            - order arrives before demand is realised (order-up-to convention)
            - demand is fixed and always fully realised (no partial fulfilment)
        ---
        INPUT
        state:  current inventory level (integer, 0..n_states-1)
        action: order quantity index (0=order nothing, a=order a*order_step units)
        ---
        OUTPUT
        next_state - next inventory level (None if goal reached)
        reward     - immediate reward for this transition
        '''
        if state == self.goal_state:
            return None, self.goal_reward

        order_qty = action * self.order_step
        candidate = state + order_qty - self.demand
        reward = self.step_reward

        out_of_bounds = candidate < 0 or candidate >= self.n_states

        if self.lava:
            # Boundary violations are blocked; wall_states are accessible but penalised
            if out_of_bounds:
                reward += self.wall_reward
                next_state = max(0, min(self.n_states - 1, candidate))
            else:
                next_state = candidate
                if next_state in self.wall_states or state in self.wall_states:
                    reward += self.wall_reward
        else:
            # Any invalid transition (out of bounds or into a wall) bounces back
            if out_of_bounds or candidate in self.wall_states:
                next_state = state
                reward += self.wall_reward
            else:
                next_state = candidate

        return int(next_state), reward

    def get_outcomes(self) -> 'dict[tuple[int, int], tuple[Optional[int], float]]':
        '''
        Returns a dictionary mapping every (state, action) pair to (next_state, reward).
        '''
        return {(s, a): self.get_outcome(s, a)
                for s in range(self.n_states)
                for a in range(self.n_actions)}


def state_int_to_tuple(state_int: Optional[int], config: ExperimentConfig, device: torch.device) -> Optional[torch.Tensor]:
    '''
    Converts an inventory level integer to a (1, 1) scalar tensor, centred at 0.

    The 1D analogue of the gridworld's (1, 2) centred coordinate tensor.
    ---
    INPUT
    state_int - integer inventory level (0..n_states-1)
    ---
    OUTPUT
    state - tensor of shape (1, 1) with centred inventory level, or None
    '''
    if state_int is None:
        return None
    n_states = config.grid_dim ** 2
    cval = (n_states - 1) / 2
    return torch.tensor([[state_int - cval]], dtype=torch.float32, device=device)


def state_tuple_to_int(state: Optional[torch.Tensor], config: ExperimentConfig) -> Optional[int]:
    '''
    Converts a (1, 1) centred scalar tensor back to an inventory level integer.
    ---
    INPUT
    state - tensor of shape (1, 1) with centred inventory level
    ---
    OUTPUT
    state_int - integer inventory level, or None
    '''
    if state is None:
        return None
    n_states = config.grid_dim ** 2
    cval = (n_states - 1) / 2
    return int((state[0, 0] + cval).round().item())


def get_state_tensors(m_len: int, config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    '''
    Builds a (n_states, m_len, 1) tensor of centred inventory level scalars,
    tiled m_len times per state — used to batch student network forward passes
    over all states at once.

    Shape convention mirrors the gridworld's (grid_dim, grid_dim*m_len, 2):
    the product of the first two dims equals n_states * m_len in both cases.
    ---
    INPUT
    m_len - number of messages in the current batch
    ---
    OUTPUT
    state_tensors - tensor of shape (n_states, m_len, 1)
    '''
    n_states = config.grid_dim ** 2
    state_tensors = torch.zeros(size=(n_states, m_len, 1), dtype=torch.float32, device=device)
    for s in range(n_states):
        s_scalar = state_int_to_tuple(s, config, device)[0]  # shape (1,)
        for b in range(m_len):
            state_tensors[s, b] = s_scalar
    return state_tensors