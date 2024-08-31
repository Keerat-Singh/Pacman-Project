# Constants

# DQN Constants
MEMORY_SIZE = 20000
BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99  # value ranges from 0-1 where 0 means agent only cares about immediate rewards and vice versa
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.996
EPSILON_MIN = 0.01
NUM_EPISODES = 20000
EPISODE = 200        # Saving a new model after how many episodes

REWARDS = {'Food' : 0.2,
           'Power Up': 0.5,
           'Ghost Kill': 3,
           'Death' : -10}