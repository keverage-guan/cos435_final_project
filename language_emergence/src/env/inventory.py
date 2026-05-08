import math
import torch
import numpy as np
from typing import Optional, List
from collections import deque

from configs.experiment_config import ExperimentConfig


class DemandDistribution:
    """
    Configurable stochastic demand distribution.

    Supported types and their keyword parameters (all have sensible defaults):

        'poisson'               lam=3.0
        'uniform'               low=0, high=5
        'binomial'              n=10, p=0.3
        'negative_binomial'     n=5, p=0.5
        'zero_inflated_poisson' lam=3.0, pi=0.3   (pi = P(zero spike))

    Examples
    --------
    DemandDistribution()                                     # Poisson(3)
    DemandDistribution('poisson', lam=5)
    DemandDistribution('uniform', low=1, high=8)
    DemandDistribution('binomial', n=20, p=0.25)
    DemandDistribution('negative_binomial', n=3, p=0.4)
    DemandDistribution('zero_inflated_poisson', lam=4, pi=0.2)
    """

    POISSON               = "poisson"
    UNIFORM               = "uniform"
    BINOMIAL              = "binomial"
    NEG_BINOMIAL          = "negative_binomial"
    ZERO_INFLATED_POISSON = "zero_inflated_poisson"

    _DEFAULTS = {
        POISSON:               {"lam": 3.0},
        UNIFORM:               {"low": 0, "high": 5},
        BINOMIAL:              {"n": 10, "p": 0.3},
        NEG_BINOMIAL:          {"n": 5,  "p": 0.5},
        ZERO_INFLATED_POISSON: {"lam": 3.0, "pi": 0.3},
    }

    def __init__(self, dist_type: str = "poisson", **kwargs):
        if dist_type not in self._DEFAULTS:
            raise ValueError(
                f"Unknown distribution '{dist_type}'. "
                f"Choose from: {list(self._DEFAULTS)}"
            )
        self.dist_type = dist_type
        self.params = {**self._DEFAULTS[dist_type], **kwargs}

    def sample(self) -> int:
        p = self.params
        if self.dist_type == self.POISSON:
            return int(np.random.poisson(p["lam"]))
        elif self.dist_type == self.UNIFORM:
            return int(np.random.randint(p["low"], p["high"] + 1))
        elif self.dist_type == self.BINOMIAL:
            return int(np.random.binomial(p["n"], p["p"]))
        elif self.dist_type == self.NEG_BINOMIAL:
            return int(np.random.negative_binomial(p["n"], p["p"]))
        elif self.dist_type == self.ZERO_INFLATED_POISSON:
            if np.random.random() < p["pi"]:
                return 0
            return int(np.random.poisson(p["lam"]))

    @property
    def mean(self) -> float:
        p = self.params
        if self.dist_type == self.POISSON:
            return float(p["lam"])
        elif self.dist_type == self.UNIFORM:
            return (p["low"] + p["high"]) / 2.0
        elif self.dist_type == self.BINOMIAL:
            return float(p["n"] * p["p"])
        elif self.dist_type == self.NEG_BINOMIAL:
            return p["n"] * (1.0 - p["p"]) / p["p"]
        elif self.dist_type == self.ZERO_INFLATED_POISSON:
            return (1.0 - p["pi"]) * p["lam"]

    def __repr__(self) -> str:
        return f"DemandDistribution(type={self.dist_type!r}, params={self.params})"


class InventoryManagement:
    """
    Inventory management MDP.

    States
    ------
    Integers 0 .. grid_dim²-1, representing units of stock on hand.

    Actions
    -------
    4 fixed actions:
        0 - order nothing (always)
        1 - order order_quantities[0] units
        2 - order order_quantities[1] units
        3 - order order_quantities[2] units

    Transition (lost-sales convention, each step)
    ---------------------------------------------
    1. Place the order for `action`; it enters a pipeline queue.
    2. Receive the order placed exactly `lead_time` steps ago
       (with lead_time=0 the order arrives immediately).
    3. Realise stochastic demand d ~ demand_dist.
    4. next_inventory = clamp(on_hand + received - d, 0, n_states-1)
       Inventory is never negative (lost sales) and never exceeds capacity.

    Reward per step
    ---------------
    r = step_reward
        - holding_cost    * next_inventory
        - stockout_penalty * max(0, d - (on_hand + received))

    If a goal_state is provided and reached, goal_reward is added and the
    episode ends (next_state = None). Set goal_state=None for an open-ended
    continuing task.

    Parameters
    ----------
    init_state       : starting inventory level
    goal_state       : target inventory level, or None for no terminal state
    order_quantities : list of exactly 3 integers — order sizes for actions 1, 2, 3
    config           : ExperimentConfig (provides grid_dim, step_reward, goal_reward)
    holding_cost     : cost per unit held per step               (default 1.0)
    stockout_penalty : cost per unit of unmet demand per step    (default 5.0)
    lead_time        : periods between placing and receiving an order (default 0)
    demand_dist      : DemandDistribution instance               (default Poisson(lam=3))
    """

    def __init__(
        self,
        init_state: int,
        goal_state: Optional[int],
        order_quantities: List[int],
        config: ExperimentConfig,
        holding_cost: float = 1.0,
        stockout_penalty: float = 5.0,
        lead_time: int = 0,
        demand_dist: Optional[DemandDistribution] = None,
    ):
        if len(order_quantities) != 3:
            raise ValueError(
                "order_quantities must contain exactly 3 values "
                "(order sizes for actions 1, 2, and 3)."
            )

        self.init_state      = init_state
        self.goal_state      = goal_state
        self.n_states        = config.grid_dim ** 2
        self.n_actions       = 4
        self.step_reward     = config.step_reward
        self.goal_reward     = config.goal_reward
        self.holding_cost    = holding_cost
        self.stockout_penalty = stockout_penalty
        self.lead_time       = lead_time
        self.demand_dist     = demand_dist if demand_dist is not None else DemandDistribution()
        # Index 0 is always "order nothing"; indices 1-3 come from order_quantities
        self.order_sizes: List[int] = [0] + list(order_quantities)
        # Pipeline: FIFO queue of pending order quantities, length = lead_time
        self._init_pipeline()

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _init_pipeline(self):
        """Creates a fresh, empty pipeline of the correct length."""
        self.pipeline: deque = deque(
            [0] * self.lead_time,
            maxlen=max(self.lead_time, 1)
        )

    def reset(self) -> int:
        """Resets to init_state and clears the order pipeline. Call at episode start."""
        self._init_pipeline()
        return self.init_state

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def get_outcome(self, state: int, action: int) -> 'tuple[Optional[int], float]':
        """
        Simulates one inventory step and returns (next_state, reward).

        Side effect: for lead_time > 0 the pipeline deque is updated.
        Call reset() between independent episodes.
        ---
        INPUT
        state  : current on-hand inventory (0 .. n_states-1)
        action : 0 = order nothing, 1-3 = order_quantities[0-2]
        ---
        OUTPUT
        next_state : next on-hand inventory, or None if goal reached
        reward     : immediate scalar reward
        """
        # Terminal check at the START of a step (agent is already at goal)
        if self.goal_state is not None and state == self.goal_state:
            return None, self.goal_reward

        order_qty = self.order_sizes[action]

        # Determine how many units arrive this period
        if self.lead_time == 0:
            received = order_qty
        else:
            received = self.pipeline.popleft()
            self.pipeline.append(order_qty)

        demand = self.demand_dist.sample()

        available    = state + received
        unmet_demand = max(0, demand - available)
        raw_next     = available - demand

        # Clamp: lost sales (floor 0) and capacity ceiling (cap n_states-1)
        next_state = max(0, min(self.n_states - 1, raw_next))

        reward = (
            self.step_reward
            - self.holding_cost * next_state
            - self.stockout_penalty * unmet_demand
        )

        # Optional goal reached mid-transition
        if self.goal_state is not None and next_state == self.goal_state:
            reward += self.goal_reward
            return None, reward

        return int(next_state), reward

    def get_outcomes(self) -> 'dict[tuple[int, int], tuple[Optional[int], float]]':
        """
        Returns a dictionary of sampled (next_state, reward) for every
        (state, action) pair by drawing one demand sample per pair.

        Each pair is evaluated with a fresh, empty pipeline (lead_time > 0
        effects are therefore single-step approximations). Mainly useful for
        tabular methods or debugging with lead_time=0.
        """
        backup = deque(self.pipeline)
        results = {}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self._init_pipeline()
                results[(s, a)] = self.get_outcome(s, a)
        self.pipeline = backup
        return results

    # ------------------------------------------------------------------
    # State representation helpers (mirrors gridworld interface)
    # ------------------------------------------------------------------

    def state_int_to_tuple(
        self,
        state_int: Optional[int],
        config: ExperimentConfig,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Converts an inventory level integer to a (1, 1) centred scalar tensor.

        The inventory level is linearly shifted so that the midpoint of
        [0, n_states-1] maps to 0, matching the gridworld's centred coordinates.
        ---
        OUTPUT: tensor of shape (1, 1), or None
        """
        if state_int is None:
            return None
        cval = (self.n_states - 1) / 2.0
        return torch.tensor([[state_int - cval]], dtype=torch.float32, device=device)

    def state_tuple_to_int(
        self,
        state: Optional[torch.Tensor],
        config: ExperimentConfig,
    ) -> Optional[int]:
        """
        Converts a (1, 1) centred scalar tensor back to an inventory level integer.
        ---
        OUTPUT: integer inventory level, or None
        """
        if state is None:
            return None
        cval = (self.n_states - 1) / 2.0
        return int((state[0, 0] + cval).round().item())

    def get_state_tensors(
        self,
        m_len: int,
        config: ExperimentConfig,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Builds an (n_states, m_len, 1) tensor of centred inventory scalars,
        tiled m_len times per state — used to batch student network forward
        passes over all states at once.
        ---
        INPUT
        m_len : number of messages in the current batch
        ---
        OUTPUT
        state_tensors : tensor of shape (n_states, m_len, 1)
        """
        state_tensors = torch.zeros(
            size=(self.n_states, m_len, 1),
            dtype=torch.float32,
            device=device,
        )
        for s in range(self.n_states):
            s_scalar = self.state_int_to_tuple(s, config, device)[0]  # shape (1,)
            for b in range(m_len):
                state_tensors[s, b] = s_scalar
        return state_tensors
    
    def get_transition_probs(self, n_samples: int = 100) -> torch.Tensor:
        """
        Returns stochastic transition probabilities by sampling demand.
        Output shape: (n_states, n_actions, n_states)
        """
        probs = torch.zeros(self.n_states, self.n_actions, self.n_states)
        for s in range(self.n_states):
            if self.goal_state is not None and s == self.goal_state:
                probs[s, :, s] = 1.0
                continue
            
            for a in range(self.n_actions):
                order_qty = self.order_sizes[a]
                received = order_qty if self.lead_time == 0 else 0 # Simplification
                
                for _ in range(n_samples):
                    demand = self.demand_dist.sample()
                    ns = max(0, min(self.n_states - 1, (s + received) - demand))
                    probs[s, a, ns] += 1.0
                
                probs[s, a, :] /= n_samples
        return probs
    
    def get_expected_reward(self, state: int, action: int) -> float:
        """
        Calculates the expected reward for a (state, action) pair 
        by marginalizing over the demand distribution.
        """
        if self.goal_state is not None and state == self.goal_state:
            return float(self.goal_reward)

        order_qty = self.order_sizes[action]
        
        # Approximation: Lead time 0 arrival logic for evaluation 
        # (matching get_outcome's single-step logic)
        received = order_qty if self.lead_time == 0 else (self.pipeline[0] if self.pipeline else 0)
        available = state + received
        
        expected_r = 0.0
        
        # We sample the demand distribution to estimate the expectation
        # For a more precise analytical version, you'd sum over the PMF
        n_samples = 100 
        for _ in range(n_samples):
            demand = self.demand_dist.sample()
            unmet_demand = max(0, demand - available)
            raw_next = available - demand
            next_state = max(0, min(self.n_states - 1, raw_next))

            reward = (
                self.step_reward
                - self.holding_cost * next_state
                - self.stockout_penalty * unmet_demand
            )
            
            if self.goal_state is not None and next_state == self.goal_state:
                reward += self.goal_reward
                
            expected_r += reward

        return expected_r / n_samples