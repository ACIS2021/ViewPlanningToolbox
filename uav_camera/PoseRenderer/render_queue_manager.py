import torch.multiprocessing as mp
import multiprocessing
import psutil
from .pose_renderer import render_pose


class RenderQueueManager:
    def __init__(self, x_res, y_res, focal_length, initial_shape_path):
        self.command_queue = mp.Queue(maxsize=1)
        self.data_queue = mp.Queue(maxsize=1)
        self.msg = None
        # def render_pose(object_path, dataset_path, device, pose_queue, rgb_image_queue):
        self.process = multiprocessing.Process(target=render_pose,
                                               args=(
                                               x_res, y_res, focal_length, self.command_queue, self.data_queue, initial_shape_path))
        self.process.start()

    def get_rgb_image_and_depth_map(self, pose):
        self.command_queue.put((pose, self.msg))
        self.msg = None

        rgb_image, depth_map, k = self.data_queue.get()
        return rgb_image, depth_map, k

    def load_new_shape(self, shape_path):
        self.msg = shape_path

    def close(self):
        if self.process.is_alive():
            self.process.kill()

        for proc in psutil.process_iter():
            if proc.name() == "blender":
                proc.kill()
