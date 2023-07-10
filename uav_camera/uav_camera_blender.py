import configparser


import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


from .PoseRenderer.render_queue_manager import RenderQueueManager


class UAVCameraBlender:
    def __init__(self, starting_shape_path, x_res=None, y_res=None, focal_length=None, depth_max_range=None, rotation_mode=None):
        if x_res is None:
            config = configparser.ConfigParser()
            config.read('../config.ini')
            self.x_res = config.getint('Blender Camera', 'x_res')
            self.y_res = config.getint('Blender Camera', 'y_res')
            self.focal_length = config.getfloat('Blender Camera', 'focal_length')
            self.depth_max_range = config.getfloat('Blender Camera', 'depth_max_range')
            self.rotation_mode = config.getstr('Simulation Environment', 'rotation_mode')
        else:
            self.x_res = x_res
            self.y_res = y_res
            self.focal_length = focal_length
            self.shape_path = starting_shape_path
            self.rotation_mode = rotation_mode

        self.rqm = RenderQueueManager(x_res, y_res, focal_length, starting_shape_path)
        self.depth_map = None
        self.rgb_image = None
        self.k = None
        self.depth_max_range = depth_max_range  # Set this value to whatever you want
        self.invalid_depth_value = 65504.

        self.sensor_width = 36
        self.sensor_height = 24

        self.max_display_width = 1080
        self.max_display_height = 1080

    def capture(self, action):
        """

        :param action: A vector where the first 3 elements are the x, y, z position of the camera in the world frame and
        the following elements are the rotation in either XYZ rotation or quaternion rotation.
        :return:
        """
        if self.rotation_mode == 'QUATERNION':
            rotation_action = action[3:]
            # convert the quaternion to euler angles using rotation transform library
            rotation_action = R.from_quat(rotation_action).as_euler('xyz')
            action = np.concatenate((action[:3], rotation_action))

        self.rgb_image, self.depth_map, self.k = self.rqm.get_rgb_image_and_depth_map(action)
        return self.rgb_image, self.depth_map, self.k

    def get_depth_image(self):
        depth_map = self.depth_map
        if np.max(depth_map) > self.depth_max_range:
            depth_img = np.clip(depth_map, 0, self.depth_max_range)
            depth_img = depth_img * 255 / self.depth_max_range
        else:
            depth_img = depth_map * 255 / np.max(depth_map)
        depth_img = depth_img.astype(np.uint8)
        rgb_depth_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        rgb_depth_img[:, :, 0] = depth_img[:, :, 0]
        rgb_depth_img[:, :, 1] = depth_img[:, :, 0]
        rgb_depth_img[:, :, 2] = depth_img[:, :, 0]
        return rgb_depth_img

    def view(self):  # If I want to show what is going on in the environment, then that code would go here
        # display the image and depth map in a side-by-side in opencv
        rgb_img = self.rgb_image
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        rgb_depth_img = self.get_depth_image()
        display_img = np.concatenate((bgr_img, rgb_depth_img), axis=0)

        # check if the display image is too large for the screen, and if it is, rescale it so it is just right
        # make the display image as large as possible while still fitting on the screen with the same aspect ratio
        if display_img.shape[0] > self.max_display_height:
            scale_factor = self.max_display_height / display_img.shape[0]
            display_img = cv2.resize(display_img, None, fx=scale_factor, fy=scale_factor)
        if display_img.shape[1] > self.max_display_width:
            scale_factor = self.max_display_width / display_img.shape[1]
            display_img = cv2.resize(display_img, None, fx=scale_factor, fy=scale_factor)

        cv2.namedWindow('RGB and Depth Images', cv2.WINDOW_NORMAL)
        # Move the window to the top-left corner of the primary display
        cv2.moveWindow('RGB and Depth Images', 0, 0)
        cv2.imshow('RGB and Depth Images', display_img)
        cv2.waitKey(1)

    def depth_to_image(self, depth_map):
        # check what is the max depth seen in the depth map not including the self.invalid_depth_value
        max_visible_depth = np.max(depth_map[depth_map != self.invalid_depth_value])
        if max_visible_depth > self.depth_max_range:
            depth_img = np.clip(depth_map, 0, self.depth_max_range)
            depth_img = depth_img * 255 / self.depth_max_range
        else:
            depth_img = np.clip(depth_map, 0, max_visible_depth)
            depth_img = depth_img * 255 / max_visible_depth
        depth_img = depth_img.astype(np.uint8)
        rgb_depth_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        rgb_depth_img[:, :, 0] = depth_img[:, :, 0]
        rgb_depth_img[:, :, 1] = depth_img[:, :, 0]
        rgb_depth_img[:, :, 2] = depth_img[:, :, 0]
        return rgb_depth_img

    def load_new_shape(self, shape_path):
        self.shape_path = shape_path
        self.rqm.load_new_shape(shape_path)

    def close(self):
        self.rqm.close()

