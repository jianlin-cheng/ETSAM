"""Open MRC tomograms (and ETSAM masks) in napari."""

from pathlib import Path

import mrcfile
import numpy as np

MRC_SUFFIXES = (".mrc", ".rec", ".map", ".mrcs", ".st", ".ali")


def get_reader(path):
    paths = path if isinstance(path, list) else [path]
    if all(str(p).lower().endswith(MRC_SUFFIXES) for p in paths):
        return read_mrc_files
    return None


def read_mrc_files(path):
    paths = path if isinstance(path, list) else [path]
    layers = []
    for p in paths:
        with mrcfile.open(p, permissive=True) as mrc:
            data = mrc.data.copy()
            vs = mrc.voxel_size
            voxel_size = tuple(float(v) if v > 0 else 1.0 for v in (vs.x, vs.y, vs.z))
        name = Path(p).name
        name = name[: name.rfind(".")]
        kwargs = {
            "name": name,
            "scale": voxel_size[::-1],  # (Z, Y, X)
            "metadata": {"voxel_size": voxel_size, "source_path": str(Path(p).resolve())},
        }
        # A few small integer values: a saved mask, shown as labels.
        sample = data[:: max(1, data.shape[0] // 16)]
        if np.issubdtype(data.dtype, np.integer) and len(np.unique(sample)) <= 16:
            layers.append((data.astype(np.uint8), kwargs, "labels"))
        else:
            # Robust contrast: cryo-ET tomograms have a few extreme voxels.
            kwargs["contrast_limits"] = tuple(np.percentile(sample, (0.5, 99.5)).tolist())
            layers.append((data, kwargs, "image"))
    return layers
