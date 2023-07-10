import configparser

import gym
from gym import spaces
import numpy as np


from uav_camera.uav_camera_blender import UAVCameraBlender


class CustomEnv(gym.Env):
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('../../config.ini')
        self.x_res = config.getint('Blender Camera', 'x_res')
        self.y_res = config.getint('Blender Camera', 'y_res')
        self.rotation_mode = config.getstr('Simulation Environment', 'rotation_mode')
        self.scene_path = 'PATH_TO_SCENE'
        self.uav_camera = UAVCameraBlender(self.scene_path)
        image_shape = (self.y_res, self.x_res, 3)  # Shape of RGB image
        depth_shape = (self.y_res, self.x_res, 1)  # Shape of depth map
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),  # RGB image
            spaces.Box(low=0, high=100000, shape=depth_shape, dtype=np.float32),  # Depth map
        ))
        pose_shape = None
        if self.rotation_mode == 'XYZ':
            pose_shape = (6,)  # Shape of camera position and orientation (XYZ + XYZ Euler angles), or (XYZ + Quaternion)
        elif self.rotation_mode == 'QUATERNION':
            pose_shape = (7,)
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1000, high=1000, shape=pose_shape, dtype=np.float32),  # Camera Pose
        ))

    def reset(self):
        obs = None, None
        reward = 0
        done = False
        info = None
        return obs, reward, done, info

    def step(self, action):
        cam_info = self.uav_camera.capture(action)
        obs = (cam_info[0], cam_info[1])
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def render(self):
        self.uav_camera.view()
