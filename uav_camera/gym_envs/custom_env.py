import gym
from gym import spaces
import numpy as np
from uav_camera.uav_camera_blender import UAVCameraBlender


class CustomEnv(gym.Env):
    def __init__(self):
        self.x_res = 1920
        self.y_res = 1080
        self.focal_length = 50
        self.blender_path = '/home/acis/blender-3.5.1-linux-x64'
        self.uav_camera = UAVCameraBlender(self.x_res, self.y_res, self.focal_length, 10, "airplane_0029.obj", self.blender_path)
        image_shape = (self.y_res, self.x_res, 3)  # Shape of RGB image
        depth_shape = (self.y_res, self.x_res, 1)  # Shape of depth map
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),  # RGB image
            spaces.Box(low=0, high=100000, shape=depth_shape, dtype=np.float32),  # Depth map
        ))

        pose_shape = (6,)  # Shape of camera position and orientation (XYZ + XYZ Euler angles), or (XYZ + Quaternion)
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
