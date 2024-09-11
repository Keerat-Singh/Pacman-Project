from collections import deque, namedtuple
import random
import torch
import numpy
import _codecs
import numpy as np
from typing import Dict, List, Tuple, Deque
from SegmentTree import SumSegmentTree, MinSegmentTree
import NN_Constants


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



# Memory buffer using numpy array 
# https://github.com/Curt-Park/rainbow-is-all-you-need/tree/master

# transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, n_step: int, batch_size: int = 32, discount_factor: float = 0.99):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen = self.n_step)
        self.discount_factor = discount_factor

    def store(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, rew: float, done: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        
        transition = Transition(obs, act, next_obs, rew, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.discount_factor)
        obs, act = self.n_step_buffer[0].state, self.n_step_buffer[0].action

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.done_buf[self.ptr] = done
        self.rews_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:

        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs= self.obs_buf[idxs], next_obs= self.next_obs_buf[idxs], acts= self.acts_buf[idxs], rews= self.rews_buf[idxs],
                    done=self.done_buf[idxs],  indices=idxs) 
    
    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(obs= self.obs_buf[idxs], next_obs= self.next_obs_buf[idxs], acts= self.acts_buf[idxs], rews= self.rews_buf[idxs], 
                    done = self.done_buf[idxs])
    
    # Once the deque contains n_step transitions, the _get_n_step_info method is called to compute the N-step reward, next state, and done flag
    def _get_n_step_info(self, n_step_buffer: Deque, discount_factor: float) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        #  ('state', 'action', 'next_state', 'reward', 'done')
        last_transition = n_step_buffer[-1]
        rew, next_obs, done = last_transition.reward, last_transition.next_state, last_transition.done
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition.reward, transition.next_state, transition.done

            rew = r + discount_factor * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size
    



class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, alpha: float = 0.6, n_step: int = 1, discount_factor: float = 0.99):
        """Initialization."""
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim= obs_dim, size= size, batch_size= batch_size, n_step= n_step, discount_factor= discount_factor)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        print(f"Tree Capacity: {tree_capacity} and Max Capacity given: {self.max_size}")
        print(n_step)
        print(self.n_step)
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, rew: float, done: bool,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, next_obs, rew, done)
        
        if transition:
            self.sum_tree.tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree.tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        print(f"Getting divide by zero encountered for beta value: {beta}")
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(obs= obs,next_obs=next_obs,acts=acts,rews=rews,done=done,weights=weights,indices=indices)
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            # assert priority > 0
            # assert 0 <= idx < len(self)

            self.sum_tree.tree[idx] = priority ** self.alpha
            self.min_tree.tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            # print(f'Inside _sample_proportional; Upperbound value: {upperbound}, a: {a}, b: {b}, segment: {segment}, p_total: {p_total}')
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree.tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight