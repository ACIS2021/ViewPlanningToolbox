def render_pose(x_render_res, y_render_res, focal_length, command_queue, data_queue, initial_shape_path, blender_path):
    from pathlib import Path

    import numpy as np
    import blendtorch.btt as btt

    additional_args = [
        "--x_render_res", str(x_render_res),
        "--y_render_res", str(y_render_res),
        "--focal_length", str(focal_length),
        '--initial_shape_path', str(initial_shape_path)
    ]
    launch_args = dict(
        scene=Path(__file__).parent / "depth_and_rgb.blend",
        script=Path(__file__).parent / "depth_and_rgb.blend.py",
        num_instances=1,
        named_sockets=["DATA", "COMMAND"],
        instance_args=[additional_args],
        background=False,
        blend_path=blender_path
    )

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        addr_datas = bl.launch_info.addresses["DATA"]
        data_remotes = [btt.DuplexChannel(a) for a in addr_datas]
        addr_commands = bl.launch_info.addresses["COMMAND"]
        command_remotes = [btt.DuplexChannel(a) for a in addr_commands]

        while True:
            pose, msg = command_queue.get()
            command_remotes[0].send(cam_pose=pose, msg=msg)
            msg_data = data_remotes[0].recv()
            rgb_image = msg_data['rgb_image']
            rgb_image = np.power(rgb_image, 1 / 2.2)
            rgb_image = rgb_image * 255
            rgb_image = rgb_image.astype(np.uint8)
            depth_map = msg_data['depth_map']
            k = msg_data['k']
            data_queue.put((rgb_image, depth_map, k))
