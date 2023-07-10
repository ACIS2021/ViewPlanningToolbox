import time
from pathlib import Path

import numpy as np
import blendtorch.btb as btb
import bpy
from mathutils import Vector


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
    # center based on center of mass
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

    render_cam = btb.Camera()
    render = btb.CompositeRenderer(
        [
            btb.CompositeSelection("image", "Out1", "Image", "RGB"),
            btb.CompositeSelection("depth", "Out1", "Depth", "V"),
        ],
        btid=bt_args.btid,
        camera=render_cam,
    )
    return cam, render


def create_blender_image(numpy_image):
    numpy_image = np.concatenate((numpy_image, np.ones((numpy_image.shape[0], numpy_image.shape[1], 1)) * 255), axis=2)
    # flip the image vertically
    numpy_image = np.flip(numpy_image, 0)
    # normalize the image
    numpy_image_normalized = numpy_image / 255.0

    # Create a Blender image and assign the normalized numpy image as its pixels
    image = bpy.data.images.new(f"CustomImage", width=numpy_image_normalized.shape[1],
                                height=numpy_image_normalized.shape[0])
    image.pixels = numpy_image_normalized.flatten()
    # Pack the image into the .blend file
    image.pack()

    return image


def create_image_panels(numpy_image, image_id, pose, panel_dims, scale):
    bpy.ops.mesh.primitive_plane_add(size=scale)

    # Set the location and scale of the plane
    plane = bpy.context.object
    plane.location = pose[0:3]
    plane.rotation_euler = pose[3:]
    plane.name = 'ImagePanel' + str(image_id)
    plane.scale[0], plane.scale[1] = panel_dims[0]/1000, panel_dims[1]/1000

    # Create a material for the plane
    material = bpy.data.materials.new("PlaneMaterial")
    material.use_nodes = True

    # Clear existing nodes
    material.node_tree.nodes.clear()

    # Create a new image texture node
    image_texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    image_texture_node.image = create_blender_image(numpy_image)

    bsdf_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    # make the Specular slider 0
    bsdf_node.inputs[7].default_value = 0
    output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    # link the image_texture_node to the Principled BSDF input base color
    material.node_tree.links.new(image_texture_node.outputs[0], bsdf_node.inputs[0])
    # link the Principled BSDF output to the Material Output input surface
    material.node_tree.links.new(bsdf_node.outputs[0], output_node.inputs[0])
    # Assign the material to the plane
    plane.data.materials.append(material)


def draw_trajectory(thickness=0.001):
    # Get a list of all image panel objects in the scene
    image_panels = [obj for obj in bpy.context.scene.objects if
                    obj.type == 'MESH' and obj.name.startswith("ImagePanel")]

    # Sort the image panels based on their name (as string integers)
    sorted_panels = sorted(image_panels, key=lambda obj: int(obj.name[10:]))

    # Get the locations of the image panels
    locations = [panel.location for panel in sorted_panels]
    # convert the locations to a list of numpy arrays
    locations = [np.array(location) for location in locations]

    for i in range(len(locations) - 1):
        # calculate the distance between two locations
        length = np.linalg.norm(locations[i + 1] - locations[i])
        # calculate the halfway point between the two locations
        center = (locations[i] + locations[i + 1]) / 2
        center = center - thickness * center  # Shifting the tubes so they don't go through the image panels
        direction = (locations[i + 1] - locations[i])/(length + 1e-7)  # Small error term if there are two poses on eachother

        direction_vector = Vector(direction)
        rotation_euler = direction_vector.to_track_quat('-Z', 'Y').to_euler()
        bpy.ops.mesh.primitive_cylinder_add(
            location=center,
            depth=length,
            radius=thickness,
            rotation=rotation_euler
        )

        # Set the tube object
        obj = bpy.context.active_object

        # Set the tube color to green
        mat = bpy.data.materials.new(name="TubeMaterial")
        # use nodes
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        bsdf_node = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.inputs[0].default_value = (1-(i/len(locations)), i / len(locations), 0, 1)  # Color fade

        bsdf_node.inputs[7].default_value = 0
        output_node = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf_node.outputs[0], output_node.inputs[0])
        obj.data.materials.append(mat)


        # Make the tube object visible to the renderer
        obj.hide_render = False


def generate_trajectory_visualization(cam, command_duplex, data_duplex, env_args, render):
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # get the rgb_images list of numpy arrays from the command_duplex
    command_msg = command_duplex.recv()
    rgb_images = command_msg["rgb_images"]
    poses = command_msg["poses"]
    cam_pose = command_msg["cam_pose"]
    panel_dims = command_msg["image_panel_dims"]
    scale = command_msg["size"]
    thickness = command_msg["thickness"]

    for i in range(poses.shape[0]):
        rgb_image = rgb_images[i]
        pose = poses[i]
        # create a reference image
        create_image_panels(rgb_image, i, pose, panel_dims, scale)

    draw_trajectory(thickness)

    cam.location = cam_pose[0:3]
    cam.rotation_euler = cam_pose[3:6]
    # set the world background color to white
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 0)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1

    # save the scene with the textures of the image panels embedded
    blend_save_path = command_msg["blend_save_path"]
    if blend_save_path is not None:
        scene_path = Path(env_args.scene_path)
        scene_name = scene_path.stem
        bpy.ops.wm.save_as_mainfile(filepath=blend_save_path + f"/{scene_name}.blend")
    render_results = render.render()
    scene_image = render_results["image"]

    # save the scene to disk
    # save the scene to disk in a folder called tmp located at the same level as the blend file
    save_path = Path(__file__).parent / "tmp"
    bpy.context.scene.render.filepath = str(save_path / "trajectory_visualization.png")  # Used to fix color issue
    bpy.ops.render.render(write_still=True)
    # Below code kept for synchronizing process
    data_duplex.send(scene_image=scene_image)

    # doing it twice to fix bug with data_duplex not sending the image
    render_results = render.render()
    scene_image = render_results["image"]
    data_duplex.send(scene_image=scene_image)
def main():
    import argparse
    # Parse script arguments passed via blendtorch launcher
    bt_args, remainder = btb.parse_blendtorch_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_render_res", default=224, type=int)
    parser.add_argument("--y_render_res", default=224, type=int)
    parser.add_argument('--focal_length', type=float, default=50)
    parser.add_argument('--scene_path', type=str)
    # add the rgb_images list of numpy arrays as an argument

    env_args = parser.parse_args(remainder)

    bpy.context.scene.render.resolution_x = env_args.x_render_res
    bpy.context.scene.render.resolution_y = env_args.y_render_res

    delete_any_existing_objects()
    cam, render = configure_camera(bt_args, env_args)
    obj = load_object(env_args.scene_path)

    object_scaling_and_centering(obj)

    command_duplex = btb.DuplexChannel(bt_args.btsockets["COMMAND"], bt_args.btid)
    data_duplex = btb.DuplexChannel(bt_args.btsockets["DATA"], bt_args.btid)

    generate_trajectory_visualization(cam, command_duplex, data_duplex, env_args, render)


main()
