import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchviz import make_dot
import numpy as np
import random
import pygame as PG
from PacmanGame import PacmanGame
import NN_Constants
from Memory import ReplayMemory

# we define are dqn algo with the following formula:
"""
Q(s,a) ← Q(s,a)+ alpha[reward + discount_factor * maxQ(s',a') - Q(s,a)]
where: maxQ(s',a') means the action that maximizes Q value in the next state s' 
 """
# Q(s,a): Q value for a current state 's' for current action 'a'
# alpha is learning rate
# reward is the reward received after taking action a
# s' is the next state 
# a' is the next action



# Classs for DQN 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Agent:
    def __init__(self, state_dim, action_dim, memory_size, batch_size, discount_factor, lr, epsilon, epsilon_decay, epsilon_min):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = len(state_dim)
        self.action_dim = len(action_dim)
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.total_reward = []
        self.episode_reward = 0     # This will keep track of rewards for previous NN_Constants.EPISODE
        self.loss = []
        self.q_values = []
           
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return np.argmax(q_values.cpu().data.numpy()).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # print(len(self.memory))

        batch = self.memory.sample(self.batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([[x[1]] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.discount_factor * next_q_values

        loss = F.mse_loss(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        # test_print(self.epsilon, self.epsilon_decay, self.episode_reward, loss, q_values)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def test_print(epsilon, epsilon_decay, episode_reward, loss, q_values):
    print('Epsilon: ', epsilon)
    # print('Epsilon Decay: ', epsilon_decay)
    # print('Episode Reward: ', episode_reward)
    # print('Loss: ', loss)
    # print('Q Values: ', q_values)


def train_dqn(env, agent, episode_count, num_episodes, target_update):
    
    for episode in range(episode_count, num_episodes):
        env.reset_game()  # Reset environment and get initial state
        state = env.get_state_space()
        done = False
        current_episode_reward = 0.0

        while not done:
            for event in PG.event.get():
                if event.type == PG.QUIT:
                    PG.quit()
                    return

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            env.display()
            agent.memory.push(state, action, next_state, reward, done)
            agent.train()
            state = next_state
            current_episode_reward += reward

        # Saving last reward from each episode
        agent.total_reward.append(current_episode_reward)

        # Decay epsilon after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % target_update == 0:
            agent.update_target_net()
        
        # Saving model after some episodes have been passed
        if episode % NN_Constants.EPISODE == 0:
            model_name = "Model_at_episode_" + str(episode) + '.pth'
            agent.episode_reward = current_episode_reward
            save_model(agent, model_name, episode)

        print(f"Episode {episode}, Total Reward: {current_episode_reward}, Epsilon: {agent.epsilon}")
        print(state)

# Showing network architecture 
def network_architecture(state_dim, action_dim):
    # Example input tensor
    x = torch.randn(1, (state_dim))
    model = DQN(input_dim=(state_dim), output_dim=(action_dim))
    # Forward pass
    y = model(x)
    # Visualize the network
    make_dot(y, params=dict(model.named_parameters())).render("network_architecture", format="png")


# Saving the model
def save_model(agent, filename, episode_count):
    model_path = os.path.join('Model/DQN', filename)
    torch.save({        
        'episode_count': episode_count,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'total_reward' : agent.total_reward,
        'episode_reward' : agent.episode_reward,
        'memory' : agent.memory
        }, model_path)
    print(f"Model saved to {model_path}")

# Loading the model
def load_model(agent, filename, env):
    model_path = os.path.join('Model/DQN', filename)
    # memory_ = 

    state_dim = env.get_state_space()
    action_dim = env.get_action_space()
    agent = Agent(state_dim, action_dim, memory_size= NN_Constants.MEMORY_SIZE, batch_size=NN_Constants.BATCH_SIZE, 
            discount_factor=NN_Constants.DISCOUNT_FACTOR, lr=NN_Constants.LEARNING_RATE, 
            epsilon=NN_Constants.EPSILON, epsilon_decay=NN_Constants.EPSILON_DECAY, epsilon_min=NN_Constants.EPSILON_MIN, 
            )

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cuda', weights_only= False)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.epsilon_decay = checkpoint['epsilon_decay']
        agent.epsilon_min = checkpoint['epsilon_min']
        agent.total_reward = checkpoint['total_reward']
        # agent.episode_reward = checkpoint['episode_reward']
        episode_count = checkpoint['episode_count']
        agent.memory = checkpoint['memory']
        print(f"Loaded model: {model_path}")
    else:
        episode_count = 0
        print(f"No model found at: {model_path}. Initializing new agent.")

    return agent, episode_count





# Main working
agent = None
# Will update model name to continue training
model_name = "Model_at_episode_5800.pth"
env = PacmanGame()
agent, episode_count = load_model(agent, model_name, env)

network_architecture(agent.state_dim, agent.action_dim)
train_dqn(env, agent, episode_count, num_episodes = NN_Constants.NUM_EPISODES, target_update = 10)