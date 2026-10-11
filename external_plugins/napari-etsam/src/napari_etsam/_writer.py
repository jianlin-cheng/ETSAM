"""Save napari layers as MRC files (File ▸ Save Selected Layers… ▸ name.mrc)."""

import mrcfile
import numpy as np


def save_mrc(path, data, metadata=None, scale=(1, 1, 1)):
    """Write ``data`` with the source tomogram's voxel size (or the layer scale)."""
    if not path.lower().endswith((".mrc", ".rec", ".map")):
        path += ".mrc"
    voxel_size = (metadata or {}).get("voxel_size") or tuple(scale)[::-1]
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
    return path


def write_labels(path, data, attributes):
    return [
        save_mrc(
            path,
            np.asarray(data).astype(np.uint8),
            attributes.get("metadata"),
            attributes.get("scale"),
        )
    ]


def write_image(path, data, attributes):
    data = np.asarray(data)
    if data.dtype not in (np.int8, np.int16, np.uint16, np.uint8, np.float32):
        data = data.astype(np.float32)
    return [save_mrc(path, data, attributes.get("metadata"), attributes.get("scale"))]
