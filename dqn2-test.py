import time
import gym
import random
import math
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gym_snake
from wrappers1d import *

BATCH_SIZE = 64  # Q-learning batch size

ACTIONS = 3

gpu = False

#%% DQN NETWORK ARCHITECTURE
class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(32, 60)
        self.l2 = nn.Linear(60, 60)
        self.l3 = nn.Linear(60, ACTIONS)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


model = Network()
model.load_state_dict(torch.load("checkpoints/" + sys.argv[1] + ".pth", map_location=torch.device('cpu')))
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

#%%
episode_durations = []

def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    start = time.time()
    while True:
        # environment.render(mode='view_only')
        action = select_action(torch.FloatTensor([state]))
        if gpu:
            action = action.cpu()
        next_state, reward, done, _ = environment.step(action.numpy()[0, 0])
        # negative reward when attempt ends
        # if done:
        #     reward = -10

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
env = BufferWrapper(env, 4)

for e in range(EPISODES):
    print("episode", e)
    run_episode(e, env)


print('Complete')
