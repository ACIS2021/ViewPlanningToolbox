# Copyright (c) 2023 Matthew Tucsok
"""
This script acts as a wrapper for the TSDFReconstructor class. It is used to reconstruct a TSDF volume from a list of poses, rgb_images, and depth maps in the format
of this toolbox
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
import cv2
from skimage import measure


def run_reconstruction(bgr_images, depth_maps, cam_poses, voxel_size, cam_k, return_queue, cwd):
    reconstruction = TSDFInterface(voxel_size=voxel_size, cam_k=cam_k, cwd=cwd)
    reconstruction.construct(bgr_images, depth_maps, cam_poses)

    vol_origin = reconstruction.vol_origin
    return_queue.put((reconstruction.get_volume(), vol_origin))
    reconstruction.clear()




def blender_camera_pose_conversion(pose, rotation_mode):
    """
    Convert the camera pose from Blender to OpenCV
    """
    if rotation_mode == 'XYZ':
        cam_rot_mat = R.from_euler('xyz', pose[3:]).as_matrix()
    elif rotation_mode == 'QUATERNION':
        cam_rot_mat = R.from_quat(pose[3:]).as_matrix()
    else:
        raise ValueError(f"Invalid rotation mode: {rotation_mode}")
    cam_loc = np.array(pose[:3])
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = cam_rot_mat
    cam_pose[:3, 3] = cam_loc

    # Compute the adjustment matrices for rotation and translation
    R_adjust = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Apply the rotation and translation adjustments to the camera pose matrix
    cam_pose[:3, :3] = cam_pose[:3, :3] @ R_adjust

    return cam_pose


class TSDFInterface:
    def __init__(self, cam_k, voxel_size, cwd):
        self.cam_k = cam_k
        self.voxel_size = voxel_size
        self.vol_bnds = None
        self.verts = None
        self.faces = None
        self.norms = None
        self.colors = None
        self.view_frust_pts = None
        self.tsdf_vol = None
        import sys
        # add cwd to path
        sys.path.append(cwd)
        from . import fusion
        self.fusion = fusion
        self.vol_origin = None

    def construct(self, bgr_images, depth_maps, cam_poses):
        print("Estimating voxel volume bounds...")
        self.vol_bnds = np.zeros((3, 2))
        # iterate over reconstruction info, which is a tuple of (rgb, depth, loc, rot)
        for i in range(len(cam_poses)):
            depth_map = depth_maps[i]
            cam_pose = cam_poses[i]

            # Compute camera view frustum and extend convex hull
            self.view_frust_pts = self.fusion.get_view_frustum(depth_map, self.cam_k, cam_pose)
            self.vol_bnds[:, 0] = np.minimum(self.vol_bnds[:, 0], np.amin(self.view_frust_pts, axis=1))
            self.vol_bnds[:, 1] = np.maximum(self.vol_bnds[:, 1], np.amax(self.view_frust_pts, axis=1))
        self.vol_origin = self.vol_bnds[:, 0].copy(order='C').astype(np.float32)

        print("Initializing voxel volume...")
        self.tsdf_vol = self.fusion.TSDFVolume(self.vol_bnds, voxel_size=self.voxel_size)
        for i in range(len(cam_poses)):
            bgr_image = bgr_images[i]
            depth_map = depth_maps[i]
            cam_pose = cam_poses[i]

            print(f"Fusing frame {i}")

            # Integrate observation into voxel volume (assume color aligned with depth)
            self.tsdf_vol.integrate(bgr_image, depth_map, self.cam_k, cam_pose, obs_weight=1.)

    def get_volume(self):
        return self.tsdf_vol.get_volume()

    def clear(self):
        del self.tsdf_vol, self.verts, self.faces, self.norms, self.colors, self.view_frust_pts, self.vol_bnds
        self.tsdf_vol = None
        self.vol_bnds = None


class TSDF3DReconstructor:
    def __init__(self, cam_k, voxel_size, bgr_images, depth_maps, invalid_depth_value, cam_poses, rotation_mode, cwd):
        self.cam_k = cam_k
        self.voxel_size = voxel_size
        self.bgr_images = bgr_images
        self.depth_maps = depth_maps
        self.cam_poses = cam_poses
        self.rotation_mode = rotation_mode
        self.cwd = cwd
        self.reconstruction = None
        self.vol_origin = None
        self.color_const = 256 * 256
        self.process = None

        for i in range(len(cam_poses)):
            self.cam_poses[i] = blender_camera_pose_conversion(cam_poses[i], rotation_mode)  # translation then rotation
            depth = depth_maps[i]
            depth[depth == invalid_depth_value] = 0  # 65504. for Blender
            depth = depth.squeeze()
            self.depth_maps[i] = depth
        self.reconstruction_process = None

    def reconstruct(self):
        # put the torch model on the cpu
        queue = mp.Queue()
        self.process = mp.Process(target=run_reconstruction,
                       args=(self.bgr_images, self.depth_maps, self.cam_poses, self.voxel_size, self.cam_k, queue, self.cwd))
        self.process.start()
        self.reconstruction, self.vol_origin = queue.get()
        print('reconstruction done')
        if self.process.is_alive():
            self.process.kill()
        print('process closed')

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol = self.reconstruction

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + self.vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self.color_const)
        colors_g = np.floor((rgb_vals - colors_b * self.color_const) / 256)
        colors_r = rgb_vals - colors_b * self.color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol = self.reconstruction

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + self.vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self.color_const)
        colors_g = np.floor((rgb_vals - colors_b * self.color_const) / 256)
        colors_r = rgb_vals - colors_b * self.color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    def save_mesh(self, filename):
        """Save a 3D mesh to a polygon .ply file.
        """
        verts, faces, norms, colors = self.get_mesh()
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (verts.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(verts.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                verts[i, 0], verts[i, 1], verts[i, 2],
                norms[i, 0], norms[i, 1], norms[i, 2],
                colors[i, 0], colors[i, 1], colors[i, 2],
            ))

        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

        ply_file.close()

    def save_pointcloud(self, filename):
        """Save a point cloud to a polygon .ply file.
        """
        xyzrgb = self.get_point_cloud()
        xyz = xyzrgb[:, :3]
        rgb = xyzrgb[:, 3:].astype(np.uint8)

        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (xyz.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(xyz.shape[0]):
            ply_file.write("%f %f %f %d %d %d\n" % (
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
