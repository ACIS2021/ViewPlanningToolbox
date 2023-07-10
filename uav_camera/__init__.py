from gym.envs.registration import register

register(
    id="uav_camera-v0",
    entry_point="uav_camera.gym_envs:CustomEnv",
)
