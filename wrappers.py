import gym
import collections
import numpy as np
import cv2

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

class CropFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
         super(CropFrame, self).__init__(env)
         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(40, 40, 1), dtype=np.uint8)
    def observation(self, obs):
         return CropFrame.process(obs)
    @staticmethod
    def process(frame):
        # if frame.size == 210 * 160 * 3:
        #     img = np.reshape(frame, [210, 160,  3]).astype(np.float32)
        # elif frame.size == 250 * 160 * 3:
        #     img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        # else:
        #     assert False, "Unknown resolution."
        img = frame
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [40, 40, 1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),old_space.high.repeat(n_steps, axis=0), dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
     def observation(self, obs):
         return np.array(obs).astype(np.float32) / 255.0