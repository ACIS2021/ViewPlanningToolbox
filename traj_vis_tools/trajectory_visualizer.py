from pathlib import Path
import configparser

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import blendtorch.btt as btt


class TrajectoryVisualizer:
    def __init__(self, rgb_images, poses, cam_pose, res, scene_path, image_panel_dims, image_panel_scale,
                 thickness=0.01, focal_length=50, rotation_mode=None, collision_checking=False):
        self.scene_path = scene_path
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.blender_path = config.getstr['Simulation Environment']['Blender_path']

        if rotation_mode is None:
            self.rototation_mode = config.getstr['Simulation Environment']['rotation_mode']

        else:
            self.rotation_mode = rotation_mode

        if self.rotation_mode == 'QUATERNION':
            for i in range(len(poses)):
                poses[i] = R.from_quat(poses[i]).as_euler('xyz')

        self.res_x, self.res_y = res
        self.focal_length = focal_length
        self.image_panel_dims = image_panel_dims
        self.thickness = thickness

        # convet the list of poses into a numpy array
        self.poses = np.array(poses)
        self.rgb_images = np.array(rgb_images)
        self.cam_pose = np.array(cam_pose)

        if collision_checking:
            # find the minimum distance between any two poses
            min_dist = np.min(np.linalg.norm(self.poses[1:, 0:3] - self.poses[:-1, 0:3], axis=1))
            self.size = image_panel_scale * min_dist
        else:
            self.size = image_panel_scale

    def visualize_trajectory(self, blend_save_path=None):
        additional_args = [
            "--x_render_res", str(self.res_x),
            "--y_render_res", str(self.res_y),
            "--focal_length", str(self.focal_length),
            '--scene_path', str(self.scene_path),
        ]
        launch_args = dict(
            scene=Path(__file__).parent / "trajectory_gen.blend",
            script=Path(__file__).parent / "trajectory_gen.blend.py",
            num_instances=1,
            named_sockets=["DATA", "COMMAND"],
            instance_args=[additional_args],
            background=False,
            blend_path=self.blender_path
        )

        # Launch Blender
        with btt.BlenderLauncher(**launch_args) as bl:
            addr_datas = bl.launch_info.addresses["DATA"]
            data_remotes = [btt.DuplexChannel(a) for a in addr_datas]
            addr_commands = bl.launch_info.addresses["COMMAND"]
            command_remotes = [btt.DuplexChannel(a) for a in addr_commands]
            if blend_save_path is not None:
                command_remotes[0].send(rgb_images=self.rgb_images, poses=self.poses, cam_pose=self.cam_pose,
                                        image_panel_dims=self.image_panel_dims, size=self.size, thickness=self.thickness, blend_save_path=blend_save_path)
            else:
                command_remotes[0].send(rgb_images=self.rgb_images, poses=self.poses, cam_pose=self.cam_pose,
                                        image_panel_dims=self.image_panel_dims, size=self.size, thickness=self.thickness, blend_save_path=None)
            _ = data_remotes[0].recv()

            read_path = Path(__file__).parent / 'tmp' / 'trajectory_visualization.png'
            scene_image_disk = cv2.imread(str(read_path))
            return scene_image_disk
