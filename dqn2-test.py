import time
import gym
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gym_snake
from wrappers import *

BATCH_SIZE = 64  # Q-learning batch size

ACTIONS = 4

gpu = False

#%% DQN NETWORK ARCHITECTURE
class Network(nn.Module):
    
    def __init__(self, in_channels=4, num_actions=ACTIONS):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


model = Network()
model.load_state_dict(torch.load("checkpoint.pth", map_location=torch.device('cpu')))
if gpu:
    model.to('cuda:0')

#%% SELECT ACTION USING GREEDY ALGORITHM
steps_done = 0
def select_action(state):
    global steps_done
    data = Variable(state).type(torch.FloatTensor)
    if gpu:
        data = data.to('cuda:0')
    return model(data).data.max(1)[1].view(1, 1)
    
#%% MEMORY REPLAY
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#%%
memory = ReplayMemory(10000)
episode_durations = []

def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    start = time.time()
    while True:
        environment.render()
        action = select_action(torch.FloatTensor([state]))
        if gpu:
            action = action.cpu()
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])
        # negative reward when attempt ends
        if done:
            reward = -10

        memory.push((torch.FloatTensor([state]),
                     action,  # action is already a tensor
                     torch.FloatTensor([next_state]),
                     torch.FloatTensor([reward])))

        state = next_state
        steps += 1

        if done:
            #print("{2} Episode {0} finished after {1} steps".format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            print("Episode {0} finished after {1} steps, time used {2}".format(e, steps, time.time() - start))
            episode_durations.append(steps)
            break
            

EPISODES = 10000000  # number of episodes
#establish the environment
env = gym.make('snake-v0')
env = MaxAndSkipEnv(env)
env = CropFrame(env)
env = ImageToPyTorch(env) 
env = BufferWrapper(env, 4)
env = ScaledFloatFrame(env)

for e in range(EPISODES):
    print("episode", e)
    run_episode(e, env)


print('Complete')
