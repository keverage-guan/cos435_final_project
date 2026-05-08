import math
import torch
from typing import Optional

from configs.experiment_config import ExperimentConfig


class SquareGridworld():
    """
    n**2 states (n-by-n gridworld) -> n is the grid dimension

    The mapping from state to the grid is as follows:
    n(n-1)  n(n-1)+1  ...  n^2-1
    ...     ...       ...  ...
    n       n+1       ...  2n-1
    0       1         ...  n-1

    Actions 0, 1, 2, 3 correspond to right, up, left, down (always exactly one step)

    - Agent starts at init_state
    - Landing in goal_state gets goal_reward and ends the episode
    - Bumping into a wall state or the map border incurs wall_reward
    - lava=False: agent bounces back from walls
    - lava=True: wall states are accessible (outside boundary can never be crossed)
    - Each step also incurs step_reward
    """

    def __init__(self, init_state: int, goal_state: int, wall_states: 'list[int]', config: ExperimentConfig):
        self.init_state: int = init_state
        self.goal_state: int = goal_state
        self.wall_states: 'list[int]' = wall_states
        self.grid_dim: int = config.grid_dim
        self.n_states: int = config.grid_dim ** 2
        self.n_actions: int = config.n_actions
        self.lava: bool = config.lava
        self.step_reward: float = config.step_reward
        self.goal_reward: float = config.goal_reward
        self.wall_reward: float = config.wall_reward

    def get_outcome(self, state: int, action: int) -> 'tuple[Optional[int], float]':
        '''
        Given a state and an action, returns the next state and immediate reward.
        ---
        INPUT
        state:  current state (integer)
        action: action taken (0=right, 1=up, 2=left, 3=down)
        ---
        OUTPUT
        next_state - the next state (None if goal reached)
        reward     - immediate reward for the action taken
        '''
        # arriving at the goal ends the episode
        if state == self.goal_state:
            return None, self.goal_reward

        n = self.grid_dim
        next_state_dict = {0: state + 1, 1: state + n, 2: state - 1, 3: state - n}
        cross_boundary_dict = {
            0: state % n == n - 1,       # right edge
            1: state >= n * (n - 1),     # top edge
            2: state % n == 0,           # left edge
            3: state < n,               # bottom edge
        }

        if self.lava:
            reward = self.step_reward
            next_state = next_state_dict[action]
            if cross_boundary_dict[action]:
                reward += self.wall_reward
                next_state = state
            elif next_state in self.wall_states or state in self.wall_states:
                reward += self.wall_reward
        else:
            reward = self.step_reward
            next_state = next_state_dict[action]
            if next_state in self.wall_states or cross_boundary_dict[action]:
                next_state = state
                reward += self.wall_reward

        return int(next_state), reward

    def get_outcomes(self) -> 'dict[tuple[int, int], tuple[Optional[int], float]]':
        '''
        Returns a dictionary mapping every (state, action) pair to (next_state, reward).
        '''
        return {(s, a): self.get_outcome(s, a)
                for s in range(self.n_states)
                for a in range(self.n_actions)}


    def state_int_to_tuple(self, state_int: Optional[int], config: ExperimentConfig, device: torch.device) -> Optional[torch.Tensor]:
        '''
        Converts a state integer to a (1, 2) coordinate tensor centered at (0, 0).
        ---
        INPUT
        state_int - integer state representation
        ---
        OUTPUT
        state - tensor of shape (1, 2) with (x, y) coordinates, or None
        '''
        if state_int is None:
            return None
        n = config.grid_dim
        cval = (n - 1) / 2
        sx = state_int % n - cval
        sy = math.floor(state_int / n) - cval
        return torch.tensor([[sx, sy]], device=device)


    def state_tuple_to_int(self, state: Optional[torch.Tensor], config: ExperimentConfig) -> Optional[int]:
        '''
        Converts a (1, 2) coordinate tensor back to a state integer.
        ---
        INPUT
        state - tensor of shape (1, 2) with (x, y) coordinates
        ---
        OUTPUT
        state_int - integer state representation, or None
        '''
        if state is None:
            return None
        n = config.grid_dim
        cval = (n - 1) / 2
        sx, sy = state[0]
        return int((sx + cval).item() + n * (sy + cval).item())


    def get_state_tensors(self, m_len: int, config: ExperimentConfig, device: torch.device) -> torch.Tensor:
        '''
        Builds a (grid_dim, grid_dim * m_len, 2) tensor of state coordinates,
        tiled m_len times per state — used to batch student network forward passes
        over all states at once.
        ---
        INPUT
        m_len - number of messages in the current batch
        ---
        OUTPUT
        state_tensors - tensor of shape (grid_dim, grid_dim * m_len, 2)
        '''
        n = config.grid_dim
        state_tensors = torch.zeros(size=(n, n * m_len, 2))
        for j in range(n):
            for i in range(n):
                s = i * n + j
                s_tup = state_int_to_tuple(s, config, device)[0]
                for b in range(m_len):
                    state_tensors[j, i * m_len + b] = s_tup
        return state_tensors

    def get_transition_probs(self, n_samples: int = 1) -> torch.Tensor:
        """
        Exact (deterministic) transition probabilities.
        Returns a tensor of shape (n_states, n_actions, n_states) with 0/1 entries.
        n_samples is accepted for API compatibility with stochastic envs but ignored.
        """
        probs = torch.zeros(self.n_states, self.n_actions, self.n_states)
        for s in range(self.n_states):
            if s == self.goal_state:
                probs[s, :, s] = 1.0          # absorbing
            else:
                for a in range(self.n_actions):
                    ns, _ = self.get_outcome(s, a)
                    if ns is None:
                        ns = self.goal_state
                    probs[s, a, ns] = 1.0
        return probs