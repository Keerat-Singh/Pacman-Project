import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchviz import make_dot
import numpy as np
import random
import pygame as PG
from PacmanGame import PacmanGame
import NN_Constants
import Memory
import math
from typing import Dict
import pandas as pd
import time
# from math import floor, ceil

# torch.serialization.add_safe_globals([ReplayMemory])


class NoisyLinear(nn.Module):
    # Noisy layer for Noisy Networks (Rainbow)
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))


    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)  # Noise for input features
        epsilon_out = self.scale_noise(self.out_features)  # Noise for output features

        # Outer product for weight noise and directly apply epsilon for bias noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # Apply outer product noise to weights
        self.bias_epsilon.copy_(epsilon_out)  # Apply noise to biases

    def forward(self, x):
        # if self.training:
        #     epsilon_in = torch.randn_like(self.weight_epsilon)
        #     epsilon_out = torch.randn_like(self.bias_epsilon)
        # else:
        #     weight = self.weight_mu
        #     bias = self.bias_mu

        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)
    
    @staticmethod
    def scale_noise(size):
        x = torch.randn(size)  # Generate Gaussian noise
        return x.sign().mul(x.abs().sqrt())  # Apply factorized noise



# Rainbow net 
class Network(nn.Module):
    
    """Initialization."""
    def __init__(self, in_dim: int, out_dim: int, atom_size: int, support: torch.Tensor):
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        """Forward method to compute Q-values from the distribution."""
        dist = self.dist(game_state)
        q = torch.sum(dist * self.support, dim=2)  # Compute expected Q-values by summing over the atoms
        return q
    
    def dist(self, game_state: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        # Pass the input through the fully connected layers
        game_state = F.relu(self.fc1(game_state))
        game_state = F.relu(self.fc2(game_state))
        # game_state shape is [1,128]

        adv_hid = F.relu(self.advantage_hidden_layer(game_state))
        val_hid = F.relu(self.value_hidden_layer(game_state))
        
        advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
        # advantage shape is [1,4,51]
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        # value shape is [1,1,51]
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        # q_atoms shape is [1,4,51]
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        # dist shape is [1,4,51]
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
    

class Agent:
    """""Initializing values"""
    def __init__(self, memory_size, batch_size, discount_factor, lr, v_min, v_max,
                 atom_size, n_step: int, env = PacmanGame()):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.state_dim = len(self.env.get_state_space())
        self.action_dim = len(self.env.get_action_space())
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # Prioritized memory buffer
        self.beta = NN_Constants.BETA
        self.prioritized_epsilon = NN_Constants.PRIORITIZED_EPSILON
        self.prioritized_memory = Memory.PrioritizedReplayBuffer(obs_dim= self.state_dim, size= NN_Constants.MEMORY_SIZE,
                                                                 batch_size= NN_Constants.BATCH_SIZE, alpha= 0.2,
                                                                 discount_factor= discount_factor, n_step= 1)

        # N step memory buffer
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.n_step_memory = Memory.ReplayBuffer(obs_dim= self.state_dim, size= memory_size, 
                                                     n_step= n_step, batch_size= batch_size, discount_factor= discount_factor)

        # C51/Categorical DQN Parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.policy_net = Network(in_dim = self.state_dim, out_dim= self.action_dim, atom_size= self.atom_size, support= self.support).to(self.device)
        self.target_net = Network(in_dim = self.state_dim, out_dim= self.action_dim, atom_size= self.atom_size, support= self.support).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Episode couter to keep reload the model to continue training
        self.episode = 0

           
    def select_action(self, state):    

        # NoisyNet: no epsilon greedy action selection
        selected_action = self.policy_net(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action
    

    def step(self, action: np.ndarray):
        """Take an action and return the response of the env."""
        # next_state, reward, terminated, truncated, _ = self.env.step(action)
        # env.get_state_space return a state = np.concatenate((board_state, game_state, pacman_position, ghost_positions, ghost_state))
        next_state, reward, done = self.env.step(action)
        
        if not self.is_test:
            self.transition += [next_state, reward, done]
           
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.n_step_memory.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.prioritized_memory.store(*one_step_transition)
    
        return next_state, reward, done

    # updating model
    def train(self):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.prioritized_memory.sample_batch(self.beta)   
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.discount_factor)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and Sn-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            discount_factor = self.discount_factor ** self.n_step
            samples = self.n_step_memory.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, discount_factor)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prioritized_epsilon
        self.prioritized_memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()
 

    # Hard saving model
    def update_target_net(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


    # Training our agent
    def train_dqn(self, num_episodes:int, df, df_path, target_update:int = 10):
        """Train the agent."""
        # self.is_test = False
        
        update_cnt = 0
        for episode in range(self.episode, num_episodes):
            self.env.reset_game()      # Reset environment and get initial state
            state = self.env.get_state_space()
            done = False
            current_episode_reward = 0.0
            current_episode_loss = 0.0
            current_score = 0
            start_time = time.time()

            while not done:
                for event in PG.event.get():
                    if event.type == PG.QUIT:
                        PG.quit()
                        return

                action = self.select_action(state)
                """"Storing state info during self.step()"""
                next_state, reward, done = self.step(action= action)
                self.env.display()
                state = next_state
                current_episode_reward += reward            
                current_score = self.env.current_score()

                # PER: increase beta
                fraction = min(episode / num_episodes, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if training is ready
                if len(self.prioritized_memory) >= self.batch_size:
                    current_episode_loss = self.train()
                    update_cnt += 1
                    
                    # if hard update is needed
                    """This will only start once our memory has >= states than batch_size"""
                    if update_cnt % target_update == 0:
                        # self._target_hard_update()
                        self.update_target_net()


            # Calculating time
            end_time = time.time()
            current_time = end_time - start_time

            # Saving model after some episodes have been passed
            if episode % NN_Constants.EPISODE == 0:
                model_name = "Model_at_episode_" + str(episode) + '.pth'
                self.episode_reward = current_episode_reward
                save_model(self, subdir, model_name, episode)

            # Saving csv
            if episode in df['Episode'].values:
                df.loc[df['Episode'] == episode, ['Reward', 'Loss', 'Score', 'Time Taken']] = [current_episode_reward, current_episode_loss, current_score, current_time]
            else:
                new_row = pd.DataFrame({'Episode': [episode], 'Reward': [current_episode_reward], 'Loss': [current_episode_loss],
                                        'Score' : [current_score], 'Time Taken' : [current_time]})
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(df_path, index=False)
            

            print(f"Episode {episode}, Total Reward: {current_episode_reward}, Loss: {current_episode_loss}, Score: {current_score}, Time Taken: {current_time}")
        
            """"Current Epsisode Loss value will be update once we have higher memory than stated batch size"""


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], discount_factor: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.policy_net(next_state).argmax(1)
            next_dist = self.target_net.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * discount_factor * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device= self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.policy_net.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

# Showing network architecture 
def network_architecture(state_dim, support, policy_net):
    # Example input tensor
    x = torch.randn(1, (state_dim)).to(torch.device("cuda"))
    # Forward pass
    y = policy_net(x)
    # Visualize the network
    make_dot(y, params=dict(policy_net.named_parameters())).render("network_architecture", format="png")


# Saving the model
def save_model(agent, subdir, filename, episode_count):
    model_path = os.path.join('Model/Rainbow', subdir, filename)
    dir_path = os.path.dirname(model_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'episode' : episode_count,
        'prioritized_memory' : agent.prioritized_memory,
        'n_step_memory' : agent.n_step_memory,
        'discount_factor' : agent.discount_factor,
        'v_min' : agent.v_min,
        'v_max' : agent.v_max,
        'atom_size' : agent.atom_size,
        'support' : agent.support,
        'n_step' : agent.n_step,
        'optimizer' : agent.optimizer,
        'is_test' : agent.is_test,
        'beta' : agent.beta
        }, model_path)
    print(f"Model saved to {model_path}")

# Loading the model
def load_model(agent, subdir, filename):
    model_path = os.path.join('Model/Rainbow', subdir, filename)
    dir_path = os.path.dirname(model_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    agent = Agent(memory_size= NN_Constants.MEMORY_SIZE, batch_size=NN_Constants.BATCH_SIZE, 
            discount_factor=NN_Constants.DISCOUNT_FACTOR, lr=NN_Constants.LEARNING_RATE, 
            v_min = NN_Constants.V_MIN, v_max = NN_Constants.V_MAX, atom_size = NN_Constants.ATOM_SIZE, n_step = NN_Constants.N_STEPS)

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location='cuda', weights_only= False)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.episode = checkpoint['episode']
        agent.prioritized_memory = checkpoint['prioritized_memory']
        agent.n_step_memory = checkpoint['n_step_memory']
        agent.discount_factor = checkpoint['discount_factor']
        agent.v_min = checkpoint['v_min']
        agent.v_max = checkpoint['v_max']
        agent.atom_size = checkpoint['atom_size']
        agent.support = checkpoint['support']
        agent.n_step = checkpoint['n_step']
        agent.optimizer = checkpoint['optimizer']
        agent.is_test = checkpoint['is_test']
        agent.beta = checkpoint['beta']
        print(f"Loaded model: {model_path}")
    else:
        print(f"No model found at: {model_path}. Initializing new agent.")

    return agent


# Main working
agent = None
# Will update model name to continue training
subdir = 'Second'
model_name = "Model_at_episode_0.pth"
agent = load_model(agent, subdir, model_name)


# Checking and loading the episode/reward df csv file
df_path = os.path.join('Model/Rainbow', subdir, "reward.csv")
# Check if the CSV file exists
if not os.path.isfile(df_path):
    # If it doesn't exist, create an empty DataFrame or with specific columns, and save it
    df = pd.DataFrame(columns=["Episode", "Reward", "Loss", "Score", "Time Taken"])  # Define your columns
    df.to_csv(df_path, index=False)
    print(f"Created new CSV file at {df_path}")
else:
    # If it exists, read the CSV file
    df = pd.read_csv(df_path)
    print(f"Loaded existing CSV file from {df_path}")


network_architecture(agent.state_dim, agent.support, agent.policy_net)
agent.train_dqn(num_episodes= NN_Constants.NUM_EPISODES, df_path= df_path, df= df)