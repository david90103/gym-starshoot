
from __future__ import print_function
import random

import gym
import tkinter
import gym_snake # snake-v0
from wrappers import *
from matplotlib import pyplot as plt
from gym.envs.classic_control import rendering

env = gym.make('snake-v0')
env = MaxAndSkipEnv(env)
env = CropFrame(env)
env = ImageToPyTorch(env)
env = BufferWrapper(env, 4)
env = ScaledFloatFrame(env)

ACTIONS = env.action_space.n
ROLLOUT_TIME = 1000

env.render()
# viewer = rendering.SimpleImageViewer() 
# fig = plt.figure()
# viewer = fig.add_subplot(111)
# viewer.get_xaxis().set_visible(False)
# viewer.get_yaxis().set_visible(False)
# plt.ion()
# fig.show()

def rollout(env):
    obser = env.reset()
    for t in range(ROLLOUT_TIME):
        obser, r, done, info = env.step(random.randrange(ACTIONS))
        # viewer.clear()
        # viewer.imshow(obser[0], cmap='gray')
        # plt.pause(0.01)
        env.render()
        if done: 
            break

print("ACTIONS={}".format(ACTIONS))
try:
    while 1:
        rollout(env)
except tkinter.TclError:
    print("Terminated by user.")