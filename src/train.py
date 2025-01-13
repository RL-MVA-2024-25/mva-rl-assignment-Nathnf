from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


# I implemented a simple DQN agent with target network (with replace update)

# Libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
from evaluate import evaluate_HIV


# Replay buffer from class
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


# Agent class
class ProjectAgent:
    def __init__(self):
        # Hard coded configuration
        config = {
            'learning_rate': 0.001,
            'gamma': 0.98,  
            'buffer_size': 1000000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.,
            'epsilon_decay_period': 20000,
            'epsilon_delay_decay': 100,
            'batch_size': 1024,
            'gradient_steps': 3,
            'update_target_freq': 400,
            'max_episode': 200,
            'criterion': torch.nn.SmoothL1Loss(),
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            'monitoring_nb_trials': 1
        }
        # print(config)

        # Model and target
        device = config['device']
        print('Using device:', device)
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons=512
        self.model = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
            ).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.update_target_freq = config['update_target_freq']
        self.max_episode = config['max_episode']
        self.monitoring_nb_trials = config['monitoring_nb_trials']
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']

        # Epsilon greedy strategy
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/epsilon_stop

        # Memory buffer
        self.memory = ReplayBuffer(config['buffer_size'], device)

        # Other hyperparameters
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']

    def act_greedy(self, model, state):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        return self.act_greedy(self.model, observation)

    def save(self):
        torch.save(self.model.state_dict(), "dqn_fredholm.pt")        # hard coded path
        return

    def load(self):
        path = "dqn_fredholm.pt"                                      # hard coded path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        self.model.eval()
        return

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self, epsilon, step):
        if step > self.epsilon_delay:
            epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
        return epsilon

    def train(self):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_val_cum_reward = 0

        while episode < self.max_episode:
            epsilon = self.update_epsilon(epsilon, step)

            # Epsilon greedy strategy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act_greedy(self.model, state)

            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode += 1

                val_cum_reward = evaluate_HIV(agent=self, nb_episode=self.monitoring_nb_trials)
                print("Episode ", '{:3d}'.format(episode), 
                      ", Epsilon ", '{:6.2f}'.format(epsilon), 
                      ", Batch size ", '{:5d}'.format(len(self.memory)), 
                      ", Episode return ", '{:.2e}'.format(episode_cum_reward),
                      ", Validation score ", '{:.2e}'.format(val_cum_reward), sep='')
                state, _ = env.reset()

                # Update and save the best model
                if val_cum_reward > best_val_cum_reward:
                    best_val_cum_reward = val_cum_reward
                    self.best_model = deepcopy(self.model)
                    self.save()
                episode_return.append(episode_cum_reward)

                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        self.save()
        return episode_return



if __name__ == "__main__":
    agent = ProjectAgent()
    scores = agent.train()
    plt.plot(scores)
    plt.show()
    env.close()

