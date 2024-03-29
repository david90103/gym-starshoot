import os, subprocess, time, signal

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_snake.envs.snake import Controller, Discrete

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'view_only', 'rgb_array']}

    def __init__(self, grid_size=[50,80], unit_size=10, unit_gap=1, random_init=True):
        self.grid_size = grid_size
        self.unit_size = unit_size
        self.unit_gap = unit_gap
        self.viewer = None
        self.action_space = Discrete(3)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(grid_size[1], grid_size[0], 3), dtype=np.uint8)
        self.observation_space = spaces.Discrete(8)
        self.random_init = random_init

    def step(self, action):
        self.last_obs, self.last_grid_obs, rewards, done, info = self.controller.step(action)
        return self.last_obs, rewards, done, info

    def reset(self):
        self.controller = Controller(self.grid_size, self.unit_size, self.unit_gap)
        self.last_grid_obs = self.controller.grid.grid.copy()
        self.last_obs = np.zeros(8) # 44
        return self.last_obs

    def render(self, mode='rgb_array', close=False, frame_speed=.01):
        # mode='view_only'
        if mode == 'human':
            if self.viewer is None:
                self.fig = plt.figure()
                self.viewer = self.fig.add_subplot(111)
                self.viewer.get_xaxis().set_visible(False)
                self.viewer.get_yaxis().set_visible(False)
                plt.ion()
                self.fig.show()
            else:
                self.viewer.clear()
                self.viewer.imshow(self.last_grid_obs)
                plt.pause(frame_speed)
            self.fig.canvas.draw()
        elif mode == 'view_only':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            else:
                self.viewer.imshow(self.last_grid_obs)
            time.sleep(0.1)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return self.last_grid_obs

    def seed(self, x):
        pass
