"""
postprocess.py
==============

Post-process 3D binary segmentation masks with selectable steps:

1. ``--postprocess-remove-parallel-membrane-misconnections`` — sever small 2D
   misconnections between parallel membranes.
2. ``--postprocess-remove-thin-noise`` — remove thin 3D blobs that do not span
   enough slices along Z.

With ``--post-process`` or no method flags (standalone ``postprocess.py``), all steps run.
With one or more method flags, only the selected steps run.

Example
-------
    python postprocess.py mask_in.mrc mask_out.mrc
    python postprocess.py mask_in.mrc mask_out.mrc --postprocess-remove-thin-noise
    python postprocess.py mask_in.mrc mask_out.mrc --postprocess-remove-parallel-membrane-misconnections --post-process-max-bridge-len 8
"""

from __future__ import annotations

DEFAULT_POST_PROCESS_MIN_SLICES = 10
DEFAULT_MAX_BRIDGE_LEN = 5
DEFAULT_MAX_SPUR_LEN = 0

import argparse
import os
import sys
from collections import defaultdict

import mrcfile
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import find_objects
from scipy.ndimage import label as cc_label

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None


def add_postprocess_arguments(parser: argparse.ArgumentParser) -> None:
    """Register shared postprocessing CLI flags on ``parser``."""
    parser.add_argument(
        "--post-process",
        help="run all postprocessing steps",
        action="store_true",
    )
    parser.add_argument(
        "--postprocess-remove-thin-noise",
        help="remove thin 3D blobs that do not span enough Z slices",
        action="store_true",
    )
    parser.add_argument(
        "--postprocess-remove-parallel-membrane-misconnections",
        help="sever small 2D misconnections between parallel membranes per Z slice",
        action="store_true",
    )
    parser.add_argument(
        "--post-process-min-slices",
        help="minimum number of Z slices a blob (connected component) must span to be kept",
        type=int,
        default=DEFAULT_POST_PROCESS_MIN_SLICES,
    )
    parser.add_argument(
        "--post-process-max-bridge-len",
        help="max skeleton length (px) of a rung between two distinct junctions to remove",
        type=int,
        default=DEFAULT_MAX_BRIDGE_LEN,
    )
    parser.add_argument(
        "--post-process-max-spur-len",
        help="if > 0, also prune dead-end spurs up to this length (px)",
        type=int,
        default=DEFAULT_MAX_SPUR_LEN,
    )


def postprocess_requested(args: argparse.Namespace) -> bool:
    """Return whether any postprocessing was requested on the CLI."""
    return bool(
        getattr(args, "post_process", False)
        or getattr(args, "postprocess_remove_thin_noise", False)
        or getattr(args, "postprocess_remove_parallel_membrane_misconnections", False)
    )


def _resolve_postprocess_steps(
    *,
    post_process_all: bool = False,
    remove_small_noise: bool = False,
    remove_parallel_membrane_misconnections: bool = False,
    default_all: bool = False,
) -> tuple[bool, bool]:
    """Decide which postprocessing steps to run from CLI flags."""
    if remove_small_noise or remove_parallel_membrane_misconnections:
        return remove_small_noise, remove_parallel_membrane_misconnections
    if post_process_all or default_all:
        return True, True
    return False, False


def postprocess_cli(
    mask: np.ndarray,
    args: argparse.Namespace,
    *,
    default_all: bool = False,
) -> np.ndarray:
    """
    Resolve postprocessing steps from ``args`` and run ``postprocess`` on ``mask``.
    """
    remove_small_noise, remove_parallel = _resolve_postprocess_steps(
        post_process_all=getattr(args, "post_process", False),
        remove_small_noise=getattr(args, "postprocess_remove_thin_noise", False),
        remove_parallel_membrane_misconnections=getattr(
            args, "postprocess_remove_parallel_membrane_misconnections", False
        ),
        default_all=default_all,
    )
    if not remove_small_noise and not remove_parallel:
        if default_all:
            raise ValueError(
                "No postprocessing steps selected. Omit method flags to run all steps, "
                "or pass --postprocess-remove-thin-noise and/or "
                "--postprocess-remove-parallel-membrane-misconnections."
            )
        return mask

    return postprocess(
        mask,
        remove_small_noise=remove_small_noise,
        remove_parallel_membrane_misconnections=remove_parallel,
        min_z_span=getattr(
            args, "post_process_min_slices", DEFAULT_POST_PROCESS_MIN_SLICES
        ),
        max_bridge_len=getattr(
            args, "post_process_max_bridge_len", DEFAULT_MAX_BRIDGE_LEN
        ),
        max_spur_len=getattr(args, "post_process_max_spur_len", DEFAULT_MAX_SPUR_LEN),
    )


def _full_structure_3d() -> np.ndarray:
    """Return a 3x3x3 structuring element for full (26) connectivity."""
    return np.ones((3, 3, 3), dtype=int)


def _shift_zero(arr: np.ndarray, offset: tuple[int, ...]) -> np.ndarray:
    """Shift ``arr`` by ``offset`` filling exposed borders with zeros."""
    result = np.zeros_like(arr)
    src_slices, dst_slices = [], []
    for o, size in zip(offset, arr.shape):
        if o > 0:
            src_slices.append(slice(0, size - o))
            dst_slices.append(slice(o, size))
        elif o < 0:
            src_slices.append(slice(-o, size))
            dst_slices.append(slice(0, size + o))
        else:
            src_slices.append(slice(0, size))
            dst_slices.append(slice(0, size))
    result[tuple(dst_slices)] = arr[tuple(src_slices)]
    return result


_OFFSETS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def sever_connections_bridge(
    mask: np.ndarray, max_bridge_len: int, max_spur_len: int = 0
) -> np.ndarray:
    """
    Remove short "rung" interconnections between lines (2D, topological).

    Skeletonise each slice, detect short junction-to-junction rungs, and carve them
    at full mask width. Optional spur pruning when ``max_spur_len > 0``.
    """
    if skeletonize is None:
        raise RuntimeError(
            "Connection removal requires scikit-image: pip install scikit-image"
        )
    if not mask.any():
        return mask.copy()

    skel = skeletonize(mask)
    if not skel.any():
        return mask.copy()

    s8 = np.ones((3, 3), dtype=np.uint8)
    skel_u8 = skel.astype(np.uint8)
    degree = ndi.convolve(skel_u8, s8, mode="constant") - skel_u8
    degree = degree * skel_u8

    junctions = skel & (degree >= 3)
    endpoints = skel & (degree == 1)
    segments = skel & ~junctions

    seg_lbl, n_seg = ndi.label(segments, structure=s8)
    if n_seg == 0:
        return mask.copy()

    jun_lbl, _ = ndi.label(junctions, structure=s8)
    seg_len = np.bincount(seg_lbl.ravel(), minlength=n_seg + 1)

    seg_junctions: dict[int, set[int]] = defaultdict(set)
    for dy, dx in _OFFSETS_8:
        shifted_jun = _shift_zero(jun_lbl, (dy, dx))
        touch = (seg_lbl > 0) & (shifted_jun > 0)
        if not touch.any():
            continue
        for sid, jid in zip(seg_lbl[touch].tolist(), shifted_jun[touch].tolist()):
            seg_junctions[sid].add(jid)

    seg_has_endpoint: set[int] = set()
    if max_spur_len > 0 and endpoints.any():
        for dy, dx in _OFFSETS_8:
            shifted_end = _shift_zero(endpoints, (dy, dx))
            touch = (seg_lbl > 0) & shifted_end
            seg_has_endpoint.update(seg_lbl[touch].tolist())
        seg_has_endpoint.update(seg_lbl[endpoints].tolist())

    bridge_ids: list[int] = []
    for sid in range(1, n_seg + 1):
        njun = len(seg_junctions.get(sid, ()))
        if seg_len[sid] <= max_bridge_len and njun >= 2:
            bridge_ids.append(sid)
        elif (
            max_spur_len > 0
            and seg_len[sid] <= max_spur_len
            and njun == 1
            and sid in seg_has_endpoint
        ):
            bridge_ids.append(sid)

    if not bridge_ids:
        return mask.copy()

    nearest_idx = ndi.distance_transform_edt(
        ~skel, return_distances=False, return_indices=True
    )
    owner = seg_lbl[tuple(nearest_idx)]
    carve = mask & np.isin(owner, bridge_ids)
    return mask & ~carve


def _apply_per_slice(mask: np.ndarray, axis: int, fn) -> np.ndarray:
    """Run a 2D operation ``fn(slice)`` independently on every slice."""
    out = np.empty_like(mask)
    moved = np.moveaxis(mask, axis, 0)
    moved_out = np.moveaxis(out, axis, 0)
    for i in range(moved.shape[0]):
        moved_out[i] = fn(moved[i])
    return out


def sever_mask_connections(
    mask: np.ndarray,
    *,
    max_bridge_len: int = DEFAULT_MAX_BRIDGE_LEN,
    max_spur_len: int = DEFAULT_MAX_SPUR_LEN,
    axis: int = 0,
) -> np.ndarray:
    """
    Sever small 2D interconnections in each slice along ``axis`` (default Z).
    """
    foreground = mask > 0 if mask.dtype != np.bool_ else mask
    cleaned = _apply_per_slice(
        foreground,
        axis,
        lambda s: sever_connections_bridge(s, max_bridge_len, max_spur_len),
    )
    if mask.dtype == np.bool_:
        return cleaned
    return cleaned.astype(mask.dtype)


def identify_3d_blobs_and_remove_thin_blobs(
    ip_mrc_data: np.ndarray, min_z_span: int = DEFAULT_POST_PROCESS_MIN_SLICES
) -> np.ndarray:
    """
    Identify 3D connected components (blobs) in a 3D mask and remove those that
    do not span at least ``min_z_span`` slices along the Z-axis.
    """
    if ip_mrc_data is None:
        raise ValueError("ip_mrc_data is None")
    if ip_mrc_data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {ip_mrc_data.shape}")
    if min_z_span < 1:
        raise ValueError("min_z_span must be >= 1")

    mask = ip_mrc_data.astype(bool)
    if not mask.any():
        return np.zeros_like(ip_mrc_data)

    structure = _full_structure_3d()
    labels, num = cc_label(mask, structure=structure)
    if num == 0:
        return np.zeros_like(ip_mrc_data)

    slices = find_objects(labels)
    keep = np.zeros(num + 1, dtype=bool)
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            keep[idx] = False
            continue
        z_slice = slc[0]
        z_span = (z_slice.stop - z_slice.start) if z_slice is not None else 0
        keep[idx] = z_span >= min_z_span

    filtered_mask = keep[labels]

    if ip_mrc_data.dtype == np.bool_:
        return filtered_mask
    return filtered_mask.astype(ip_mrc_data.dtype)


def postprocess(
    mask: np.ndarray,
    *,
    remove_small_noise: bool = True,
    remove_parallel_membrane_misconnections: bool = True,
    min_z_span: int = DEFAULT_POST_PROCESS_MIN_SLICES,
    max_bridge_len: int = DEFAULT_MAX_BRIDGE_LEN,
    max_spur_len: int = DEFAULT_MAX_SPUR_LEN,
) -> np.ndarray:
    """Apply selected mask postprocessing steps."""
    result = mask
    if remove_small_noise:
        print(f"==> Post-processing: remove small noise (min_z_span={min_z_span})")
        result = identify_3d_blobs_and_remove_thin_blobs(result, min_z_span=min_z_span)

    if remove_parallel_membrane_misconnections:
        print(
            "==> Post-processing: remove parallel membrane misconnections "
            f"(max_bridge_len={max_bridge_len}, max_spur_len={max_spur_len})"
        )
        result = sever_mask_connections(
            result,
            max_bridge_len=max_bridge_len,
            max_spur_len=max_spur_len,
        )

    return result


def save_mask(mask: np.ndarray, voxel_size, output_map_file_path: str) -> None:
    with mrcfile.new(output_map_file_path, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.uint8))
        mrc.voxel_size = voxel_size


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Post-process a 3D binary mask MRC.")
    parser.add_argument(
        "input_mask_file_path",
        help="Input binary mask MRC file path",
        type=str,
    )
    parser.add_argument(
        "output_mask_file_path",
        help="Output postprocessed mask MRC file path",
        type=str,
    )
    add_postprocess_arguments(parser)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_mask_file_path)
    output_path = os.path.abspath(args.output_mask_file_path)

    print("==> Reading input mask:", input_path)
    with mrcfile.open(input_path) as mrc:
        voxel_size = mrc.voxel_size
        mask = mrc.data
    print(f"Input mask shape: {mask.shape}, voxel size: {voxel_size}")

    post_processed_mask = postprocess_cli(mask, args, default_all=True)

    print("==> Saving post-processed mask:", output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_mask(post_processed_mask, voxel_size, output_path)
    print("Done.")


if __name__ == "__main__":
    try:
        _run_cli()
    except Exception as e:
        print("Error: " + str(e), file=sys.stderr)
        sys.exit(1)
