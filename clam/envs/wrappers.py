from collections import deque

import cv2
import gym
import gymnasium
import gymnasium.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from PIL import Image
from r3m import load_r3m
from torchvision import transforms as T

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.65, 0.0]),
}
DEFAULT_SIZE = 224


class ImageObservationWrapper(gymnasium.Wrapper):
    def __init__(self, env, image_shape=(64, 64)):
        super(ImageObservationWrapper, self).__init__(env)
        self.image_shape = image_shape
        # Set observation space to match the rendered image dimensions
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(image_shape[0], image_shape[1], 3), dtype=np.uint8
        )
        mujoco_renderer = MujocoRenderer(
            env.model, env.data, DEFAULT_CAMERA_CONFIG, DEFAULT_SIZE, DEFAULT_SIZE
        )
        self.env.unwrapped.mujoco_renderer = mujoco_renderer

    def reset(self, **kwargs):
        # Reset the environment and capture the initial image observation
        res = self.env.reset()

        if isinstance(res, tuple):
            return self._get_image_obs(), res[1]

        return self._get_image_obs()

    def step(self, action):
        # Take a step in the environment
        step_res = self.env.step(action)

        # Return the image observation instead of the state

        if len(step_res) == 4:
            obs, reward, done, info = step_res
            return self._get_image_obs(), reward, done, info
        else:
            obs, reward, done, terminated, info = step_res
            return self._get_image_obs(), reward, done, terminated, info

    def _get_image_obs(self):
        # Render the environment to capture the current image
        image = self.env.render()

        # Resize the image if needed
        if self.image_shape is not None and image.shape[:2] != self.image_shape:
            # from PIL import Image

            # image = Image.fromarray(image).resize(self.image_shape)
            image = cv2.resize(image, self.image_shape)
            image = np.array(image)

        return image


class FlipImageObsWrapper(gymnasium.Wrapper):
    """
    Flips the image upside down
    """

    def __init__(self, env):
        super(FlipImageObsWrapper, self).__init__(env)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, terminated, info

    def _process_obs(self, obs):
        return np.flipud(obs)


class ActionRepeatWrapper(gymnasium.Wrapper):
    """
    Repeat the action for a fixed number of steps
    """

    def __init__(self, env, repeat=2):
        super(ActionRepeatWrapper, self).__init__(env)
        self.repeat = repeat

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        step_reward = 0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, terminated, info = self.env.step(action)
            step_reward += reward

        return obs, step_reward, done, terminated, info


class R3MWrapper(gymnasium.Wrapper):
    def __init__(self, env, r3m_id: str):
        super().__init__(env)
        self.env = env
        self.r3m = load_r3m(r3m_id).eval()
        self.r3m.to(self.r3m.module.device)
        self.transforms = T.Compose([T.Resize(256), T.CenterCrop(224)])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2048,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, terminated, info

    def _process_obs(self, obs):
        """
        Assumes input is (H, W, C) numpy array with values in [0, 255]

        Returns:
            embedding (np.ndarray): (2048,) numpy array
        """
        # make sure it is channel first
        # R3M expects between [0, 255]
        img = torch.from_numpy(obs.copy()).permute(2, 0, 1)
        img = self.transforms(img).unsqueeze(0).to(self.r3m.module.device)
        with torch.no_grad():
            embedding = self.r3m(img)
        embedding = embedding.squeeze().cpu().numpy()
        return embedding


class FrameStackWrapper(gymnasium.Wrapper):
    """
    Stack multiple frames together
    """

    def __init__(self, env, num_frames=4):
        super(FrameStackWrapper, self).__init__(env)
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

        # Update the observation space to include the stacked frames
        low = np.repeat(self.observation_space.low, num_frames, axis=-1)
        high = np.repeat(self.observation_space.high, num_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, terminated, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=-1)
