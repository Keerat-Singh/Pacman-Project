# NN Constants
MEMORY_SIZE = 100000
# MEMORY_SIZE = 131072
BATCH_SIZE = 128
DISCOUNT_FACTOR = 0.99  # value ranges from 0-1 where 0 means agent only cares about immediate rewards and vice versa
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
NUM_EPISODES = 20000
EPISODE = 200        # Saving a new model after how many episodes

# C51 
"""The main difference between C51 and DQN is that rather than simply predicting the Q-value for each state-action pair, 
    C51 predicts a histogram model for the probability distribution of the Q-value"""
ATOM_SIZE: int = 51
V_MIN: int = -25     # Minimum possible reward in an episode
V_MAX: int = 400     # Maximum possible reward in an episode TODO this could break as we don't know the exact value of max reward for an episode yet

# n_step
N_STEPS: int = 3

# PER parameters
BETA: float = 0.1      
"""
beta is a parameter used to compensate for the bias introduced by prioritized sampling.
this tends from 0 - 1 generally
When **beta = 1**, the experiences from the prioritized buffer are weighted just as they would be in a uniform experience replay, 
meaning all transitions are treated equally, regardless of their priority.
"""
PRIORITIZED_EPSILON: float = 1e-5            
"""
guarantees every transition can be sampled
A larger value for prior_eps ensures a higher minimum probability of sampling low-priority transitions, which could increase exploration.
"""
alpha = 0.6
"""
is a parameter that controls the degree of prioritization used when sampling experiences
it ranges from [0,1], where 1 indicates PER uses full prioritization, where transitions with larger TD-errors are strongly preferred.
"""


# Pacman game reward
REWARDS = {'Food' : 0.4,
           'Power Up': 0.8,
           'Ghost Kill': 6,
           'Death' : -15,
           'Wall Collision' : -0.0005}
