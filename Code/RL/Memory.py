from collections import deque
from collections import namedtuple
import random
import torch
import numpy
import _codecs

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:

    # torch.serialization.add_safe_globals([Transition, deque, numpy.core.multiarray._reconstruct, numpy.ndarray, _codecs.encode, numpy.dtype])

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def save(self):
        return self.memory

    def load(self, memory):
        self.memory = memory