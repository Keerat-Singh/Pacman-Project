# import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from Game.PacmanGame import PacmanGame


# Classs for DQN 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class Agent:
    def __init__(self, state_dim, action_dim, memory_size, batch_size, gamma, lr, epsilon, epsilon_decay, epsilon_min):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return np.argmax(q_values.cpu().data.numpy())

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch])
        action_batch = torch.LongTensor([[x[1]] for x in batch])
        reward_batch = torch.FloatTensor([x[2] for x in batch])
        next_state_batch = torch.FloatTensor([x[3] for x in batch])
        done_batch = torch.FloatTensor([x[4] for x in batch])

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = F.mse_loss(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_dqn(env, agent, num_episodes, target_update):
    for episode in range(num_episodes):
        env.reset_game()  # Reset environment and get initial state
        state = env.get_state_space()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)  # Select action
            next_state, reward, done = env.step(action)  # Execute action
            agent.memory.push((state, action, reward, next_state, done))  # Store transition in memory
            agent.train()  # Train the agent
            state = next_state
            total_reward += reward

        if episode % target_update == 0:
            agent.update_target_net()

        print(f"Episode {episode}, Total Reward: {total_reward}")


# Example usage
env = PacmanGame()
state_dim = env.get_state_space()  # Define this based on your game
action_dim = env.get_action_space()  # Define this based on your game
agent = Agent(state_dim, action_dim, memory_size=10000, batch_size=64, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

train_dqn(env, agent, num_episodes=500, target_update=10)
