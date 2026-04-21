import random
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    '''
    Fixed-capacity circular buffer for experience replay.
    Stores Transition namedtuples and supports random sampling for DQN training.
    '''

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition.'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> 'list[Transition]':
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
