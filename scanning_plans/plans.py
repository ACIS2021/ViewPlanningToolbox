import math
import numpy as np
import cv2


class ScanningPlan:
    def __init__(self, uav_camera, visualize_scan):
        self.poses = []
        self.rgb_images = []
        self.depth_maps = []
        self.uav_camera = uav_camera
        self.visualize_scan = visualize_scan

    def query_camera(self, pose):
        rgb, depth, k = self.uav_camera.capture(pose)
        if self.visualize_scan:
            # convert to bgr
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # show the bgr image
            cv2.imshow('uav_camera', bgr)
            cv2.waitKey(1)

        self.rgb_images.append(rgb)
        self.depth_maps.append(depth)
        self.poses.append(pose)


class HemisphereScan(ScanningPlan):
    def __init__(self, side_overlap, end_overlap, radius, center, uav_camera, visualize_scan=False):
        super().__init__(uav_camera, visualize_scan)
        self.radius = radius
        self.center = center
        self.side_overlap = side_overlap
        self.end_overlap = end_overlap

    def start(self):
        # Calculate the horizontal field of view (FOV)
        horizontal_fov = 2 * math.atan(self.uav_camera.sensor_width / (2 * self.uav_camera.focal_length))

        # Calculate the vertical field of view (FOV)
        vertical_fov = 2 * math.atan(self.uav_camera.sensor_height / (2 * self.uav_camera.focal_length))

        r = self.radius
        center = self.center
        latitude_separation = 2 * r * np.tan(vertical_fov / 2) * (1 - self.side_overlap)
        theta_step = np.arctan(latitude_separation / r)

        longitude_separation = 2 * r * np.tan(horizontal_fov / 2) * (1 - self.end_overlap)

        i = theta_step
        j = 0
        while i < np.pi / 2:
            while j < 2 * np.pi:
                theta = i
                phi = j
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                cam_pose = -1 * np.array([x, y, z])
                z_angle = np.arctan2(cam_pose[1], cam_pose[0]) - np.pi / 2
                cam_pose *= -1

                x_angle = np.pi / 2 + np.arctan(
                    (center[2] - cam_pose[2]) / np.sqrt(cam_pose[0] ** 2 + cam_pose[1] ** 2))

                self.query_camera([x, y, z, x_angle, 0, z_angle])

                j += longitude_separation / np.sqrt(r ** 2 - z ** 2)
            j = 0
            i += theta_step
        if self.visualize_scan:
            cv2.destroyAllWindows()

        return self.rgb_images, self.depth_maps, self.poses


class CircularScan(ScanningPlan):
    def __init__(self, end_overlap, radius, height, center, uav_camera, visualize_scan=False):
        super().__init__(uav_camera, visualize_scan)
        self.r = radius
        self.center = center
        self.z = height
        self.end_overlap = end_overlap

    def start(self):
        # Calculate the horizontal field of view (FOV)
        horizontal_fov = 2 * math.atan(self.uav_camera.sensor_width / (2 * self.uav_camera.focal_length))
        center = self.center

        longitude_separation = 2 * self.r * np.tan(horizontal_fov / 2) * (1 - self.end_overlap)
        j = 0
        while j < 2 * np.pi:
            theta = j
            x = self.r * np.cos(theta)
            y = self.r * np.sin(theta)
            z = self.z

            cam_pose = -1 * np.array([x, y, z])
            z_angle = np.arctan2(cam_pose[1], cam_pose[0]) - np.pi / 2
            cam_pose *= -1

            x_angle = np.pi / 2 + np.arctan(
                (center[2] - cam_pose[2]) / np.sqrt(cam_pose[0] ** 2 + cam_pose[1] ** 2))

            self.query_camera([x, y, z, x_angle, 0, z_angle])

            j += longitude_separation / np.sqrt(self.r ** 2 - z ** 2)
            if self.visualize_scan:
                cv2.destroyAllWindows()

        return self.rgb_images, self.depth_maps, self.poses

