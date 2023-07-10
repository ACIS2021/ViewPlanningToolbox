import os
import re
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path

import bpy
import minexr
import numpy as np

from .camera import Camera


@dataclass
class CompositeSelection:
    """Name of given to this element, serves as key in the result dict."""

    key: str
    """Name of the target FileOutput node in the compositing tree."""
    node: str
    """Name of the slot to select."""
    slot: str
    """Selection of channels to extract from slot. 
    Note, use 'RGBA', 'RGB', 'BGR' for color information and 'V' for depth information."""
    channels: str


@dataclass
class _EXRSelection:
    key: str
    channels: list


class CompositeRenderer:
    """Provides composite rendering support for EEVEE.

    Rendering with compositor support. This implementation currently assumes that at least one FileOutput node is present in the compositing tree. Each FileOutput node has to be configured as follows:
     - Filename: has include hashes placeholders #### for frame index.
     - Format: OPEN_EXR_MULTILAYER,
     - Codec: None
     - Precision: Half
    The initialization is passed an array of `CompositeSelection` elements that basically define which elements should be returned after a call to `render`.

    Note, color images are in linear color spaces and may appear darker than
    expected when visualized. Use `btt.colors.gamma` to gamma encode before
    visualizing.

    Params
    ------
    slots: Iterable[CompositeSelection]
        Slots to be selected and returned by `render`.
    camera: btb.Camera, None
        Camera view to be rendered. When None, default camera is used.
    delete_render_files: bool, optional
        Delete intermediate render files. Defaults to true.
    btid: int, optional
        Blendtorch worker index. Required to avoid file collisions for multiple workers.
    """

    def __init__(self, slots, camera=None, delete_render_files=True, btid=1):
        self.btid = btid
        self.camera = camera or Camera()
        self.delete_render_files = delete_render_files
        self._scene = bpy.context.scene
        assert self._scene.use_nodes, "CompositeRenderer requires compositing nodes"
        assert len(slots) > 0, "No slots selected"

        self.outnodes = self._find_output_nodes()
        self.outnodes = self._update_output_paths(self.outnodes)
        self._outpath_re = re.compile(r"((#)+)")

        def node_from_name(name):
            nodes = [n for n in self.outnodes if n.name == name]
            assert len(nodes) > 0, f"Could not find output node {name}"
            return nodes[0]

        def exr_from_slots(slots):
            exr = []
            for s in slots:
                channels = [f"{s.slot}.{c}" for c in s.channels]
                exr.append(_EXRSelection(s.key, channels))
            return exr

        self.mapped_slots = {
            node_from_name(k): exr_from_slots(g)
            for k, g in groupby(slots, key=lambda s: s.node)
        }

    def _find_output_nodes(self):
        tree = self._scene.node_tree
        outnodes = [n for n in tree.nodes if n.type == "OUTPUT_FILE"]

        def is_compatible(n):
            return (
                n.format.file_format == "OPEN_EXR_MULTILAYER"
                and n.format.exr_codec == "NONE"
                and n.format.color_depth == "16"  # currently, minexr assumes fp16
            )

        outnodes_ok = [n for n in outnodes if is_compatible(n)]
        # outnodes_dropped = [n for n in outnodes if not is_compatible(n)]
        assert (
            len(outnodes_ok) > 0
        ), "Could not find a single compatible output filenode"
        return outnodes_ok

    def _update_output_paths(self, outnodes):
        for n in outnodes:
            path = Path(bpy.path.abspath(n.base_path))
            path = path.parent / f"{path.stem}_{self.btid:02d}"
            path = path.with_suffix(".exr")
            n.base_path = str(path)
        return outnodes

    def _actual_path(self, fidx, base_path):
        def _replicate(g, fidx):
            len = g.span()[1] - g.span()[0]
            return str(fidx).zfill(len)

        newpath, cnt = self._outpath_re.subn(partial(_replicate, fidx=fidx), base_path)
        assert (
            cnt > 0
        ), "Composite renderer requires hash placeholders in output paths to identify frame number."
        path = Path(bpy.path.abspath(newpath))
        assert path.exists(), f"Could not find output file {path}"
        return path

    def render(self):
        """Render the scene and return data of selected slots.

        Returns
        -------
        data: dict
            Mapping from name to np.array
        """

        fidx = self._scene.frame_current
        self.camera.set_as_active_camera(self._scene)
        bpy.ops.render.render(animation=False, write_still=False, use_viewport=True)

        key_data = {}
        for node, exrsel in self.mapped_slots.items():
            path = self._actual_path(fidx, node.base_path)
            with open(path, "rb") as fp:
                reader = minexr.load(fp)
                # print(reader.attrs)
                for exr in exrsel:
                    data = reader.select(exr.channels).astype(np.float32)
                    key_data[exr.key] = data
            if self.delete_render_files:
                os.remove(path)

        return key_data
