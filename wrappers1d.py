import gym
import collections
import numpy as np

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    def step(self, action):
            total_reward = 0.0
            done = None
            for _ in range(self._skip):
                obs, reward, done, info = self.env.step(action)
                self._obs_buffer.append(obs)
                total_reward += reward
                if done:
                    break
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
            return max_frame, total_reward, done, info
    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        # FIXME I HAVE NO IDEA WHAT I AM DOING... HEHE
        # old_space = env.observation_space
        # self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),old_space.high.repeat(n_steps, axis=0), dtype=dtype)
        self.observation_space = gym.spaces.Discrete(8 * n_steps)
    def reset(self):
        # self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        self.buffer = np.zeros(8 * 4, dtype=self.dtype)
        return self.observation(self.env.reset())
    def observation(self, observation):
        self.buffer = self.buffer[8:]
        self.buffer = np.append(self.buffer, observation)
        return self.buffer
