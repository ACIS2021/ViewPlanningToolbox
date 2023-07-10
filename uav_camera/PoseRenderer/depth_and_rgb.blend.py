from math import radians, atan2, pi, atan, sqrt

import numpy as np
import blendtorch.btb as btb
import bpy
from mathutils import Vector, Euler, Quaternion, Matrix


def get_calibration_matrix_k_from_blender(cam_data, mode='simple'):
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    if mode == 'simple':
        aspect_ratio = width / height
        k = np.zeros((3, 3), dtype=np.float32)
        k[0][0] = width / 2 / np.tan(cam_data.angle / 2)
        k[1][1] = height / 2. / np.tan(cam_data.angle / 2) * aspect_ratio
        k[0][2] = width / 2.
        k[1][2] = height / 2.
        k[2][2] = 1.
        k.transpose()

    elif mode == 'complete':

        focal = cam_data.lens  # mm
        sensor_width = cam_data.sensor_width  # mm
        sensor_height = cam_data.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if cam_data.sensor_fit == 'VERTICAL':
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels
        # print('fx', alpha_u)
        # print('fy', alpha_v)
        # print('cx', u_0)
        # print('cy', v_0)
        k = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        k = None
    return k


def delete_any_existing_objects():
    # Get a list of all objects in the scene
    all_objects = bpy.context.scene.objects

    # Loop through all objects and remove any that are not the camera or a light
    for obj in all_objects:
        if obj.type != 'CAMERA' and obj.type != 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)


def load_object(source):
    # Split the source path at the extension
    source_split = source.split(".")
    # Get the extension
    extension = source_split[-1]
    # If the extension is .obj
    if extension == "obj":
        bpy.ops.wm.obj_import(filepath=source)
    if extension == "gltf":
        bpy.ops.import_scene.gltf(filepath=source, import_pack_images=True, guess_original_bind_pose=True)
    if extension == "fbx":
        bpy.ops.import_scene.fbx(filepath=source)

    # Get selected objects (so)
    so = bpy.context.selected_objects  # Keep in mind this returns a list, and needs to be converted into one object
    obj = so[0]  # Getting just one object
    return obj


def object_scaling_and_centering(obj=None):
    obj.select_set(True)
    name = obj.name
    bpy.data.objects[name].rotation_euler[0] = 0
    bpy.data.objects[name].rotation_euler[1] = 0
    bpy.data.objects[name].rotation_euler[2] = 0
    max_dim = float(max(obj.dimensions))
    scale_factor = 1 / max_dim
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor), orient_type='GLOBAL',
                             orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                             orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1,
                             use_proportional_connected=False,
                             use_proportional_projected=False)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # center the object based on center of mass
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    obj.location = [0, 0, 0]
    # bbox = obj.bound_box
    # bbox = np.array(bbox)
    # bbox_center = np.sum(bbox, axis=0) / 8
    # bbox_center = Vector((bbox_center[0], bbox_center[1], bbox_center[2]))
    # # Find where the bounding box center is relative to the global origin
    # translation_vector = obj.matrix_world @ bbox_center
    # obj.location = obj.location - translation_vector
    # # Shift the object up so it sits on the XY plane
    # obj.location[2] = obj.location[2] + obj.dimensions[2] / 2



def configure_camera(bt_args, env_args):
    # Get the first camera in the scene
    cam = bpy.context.scene.camera
    cam_data = cam.data
    cam_data.lens = env_args.focal_length
    cam.rotation_mode = 'XYZ'
    cam.location = (0, 0, 0)
    cam.rotation_euler = (0, 0, 0)
    k = get_calibration_matrix_k_from_blender(cam_data)


    render_cam = btb.Camera()
    render = btb.CompositeRenderer(
        [
            btb.CompositeSelection("image", "Out1", "Image", "RGB"),
            btb.CompositeSelection("depth", "Out1", "Depth", "V"),
        ],
        btid=bt_args.btid,
        camera=render_cam,
    )
    return cam, k, render


def render_specified_pose(cam, k, command_duplex, data_duplex, env_args, render):
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    while True:
        command_msg = command_duplex.recv()

        if command_msg['msg'] is not None:
            delete_any_existing_objects()
            obj = load_object(command_msg['msg'])
            object_scaling_and_centering(obj)

        pose = command_msg['cam_pose']
        location = pose[0:3]
        orientation = pose[3:6]
        cam.location = location
        cam.rotation_euler = orientation

        render_results = render.render()
        rgb_image = render_results["image"]
        depth_map = render_results["depth"]

        k = np.array(k)

        data_duplex.send(rgb_image=rgb_image, depth_map=depth_map, k=k)


def main():
    import argparse
    # Parse script arguments passed via blendtorch launcher
    bt_args, remainder = btb.parse_blendtorch_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_render_res", default=224, type=int)
    parser.add_argument("--y_render_res", default=224, type=int)
    parser.add_argument('--focal_length', type=float, default=50)
    parser.add_argument('--initial_shape_path', type=str)
    env_args = parser.parse_args(remainder)

    bpy.context.scene.render.resolution_x = env_args.x_render_res
    bpy.context.scene.render.resolution_y = env_args.y_render_res

    delete_any_existing_objects()
    cam, k, render = configure_camera(bt_args, env_args)
    obj = load_object(env_args.initial_shape_path)

    object_scaling_and_centering(obj)

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    # Image duplex for sending images back to pytorch script
    data_duplex = btb.DuplexChannel(bt_args.btsockets["DATA"], bt_args.btid)

    # Pose duplex for receiving poses for camera placement
    command_duplex = btb.DuplexChannel(bt_args.btsockets["COMMAND"], bt_args.btid)

    render_specified_pose(cam, k, command_duplex, data_duplex, env_args, render)


main()
