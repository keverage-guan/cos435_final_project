import math as mt

import networkx as nx
import numpy as np

from configs.experiment_config import ExperimentConfig


def graph_from_walls(wall_states: 'list[int]', config: ExperimentConfig) -> nx.Graph:
    '''
    Creates a graph representing the gridworld.
    Edges have weights corresponding to movement costs (step reward or wall reward).
    ---
    INPUT
    wall_states - list of wall state positions
    config      - experiment configuration
    ---
    OUTPUT
    G - undirected graph representing the gridworld
    '''
    G = nx.Graph()
    n = config.grid_dim

    # add all states as nodes with their (x, y) grid positions
    for s in range(n ** 2):
        G.add_node(s, pos=(s % n, s // n))

    # add horizontal (s, s+1) and vertical (s, s+grid_dim) edges between adjacent states
    # edge weights are positive (= -step_reward) so shortest-path algorithms can be applied
    for s1 in range(n ** 2):
        for s2 in [s1 + 1, s1 + n]:
            if s2 not in G.nodes:
                continue
            # skip edges that cross the right or top boundary
            if s2 == s1 + 1 and s1 % n == n - 1:
                continue
            if s2 == s1 + n and s1 >= n * (n - 1):
                continue
            G.add_edge(s1, s2, weight=-config.step_reward)

    if config.lava:
        # wall states are accessible but incur an extra penalty on adjacent edges
        for s in wall_states:
            for s_i, s_j in [[s - 1, s], [s, s + 1], [s - n, s], [s, s + n]]:
                if (s_i, s_j) in G.edges:
                    G.edges[s_i, s_j]['weight'] -= config.wall_reward
    else:
        # wall states are inaccessible — remove them from the graph entirely
        for s in wall_states:
            G.remove_node(s)

    return G


def max_dist_pair(G: nx.Graph) -> 'tuple[int, int]':
    '''
    Compute the pair of states with maximum shortest-path distance in a gridworld.
    ---
    INPUT
    G - gridworld graph from graph_from_walls
    ---
    OUTPUT
    (s1, s2) - the two states that are furthest apart
    '''
    all_pairs = dict(nx.all_pairs_dijkstra(G))
    all_pairs_dist = [(n, all_pairs[n][0]) for n in all_pairs]
    max_dist_per_node = [
        (n, list(d.keys())[np.argmax(np.array(list(d.values())))], max(d.values()))
        for n, d in all_pairs_dist
    ]
    best_index = np.argmax([k for _, _, k in max_dist_per_node])
    s1, s2 = max_dist_per_node[best_index][0], max_dist_per_node[best_index][1]
    return s1, s2


def dead_end_goals(wall_states: 'list[int]', config: ExperimentConfig) -> 'list[int]':
    '''
    Find all dead-end states (exactly one open neighbor) in a maze.
    These make good goal locations since they require deliberate navigation to reach.
    ---
    INPUT
    wall_states - list of wall state positions
    config      - experiment configuration
    ---
    OUTPUT
    dead_ends - list of dead-end state indices
    '''
    n = config.grid_dim
    dead_ends = []
    for s in range(n ** 2):
        left  = -1 if s % n == 0       else s - 1
        right = -1 if s % n == n - 1   else s + 1
        up    = -1 if s >= n * (n - 1) else s + n
        down  = -1 if s < n            else s - n
        neighbors = [right, up, left, down]
        open_neighbors = [nb for nb in neighbors if nb != -1 and nb not in wall_states]
        if len(open_neighbors) == 1 and s not in wall_states:
            dead_ends.append(s)
    return dead_ends


def node_dist(G: nx.Graph, s1: int, s2: int) -> int:
    '''
    Compute the shortest path length (in steps) between two states.
    ---
    INPUT
    G       - gridworld graph from graph_from_walls
    s1, s2  - integer state indices
    ---
    OUTPUT
    dist - number of steps on the shortest path
    '''
    return nx.dijkstra_path_length(G, s1, s2, weight=None)
