"""
Helper run with the Python of the ETSAM conda environment (not in the host app).

Shared, identical copy in the ChimeraX plugin (chimerax.etsam) and the napari
plugin (napari_etsam); keep them in sync. It drives ETSAM's Python API
(``etsam.segment`` and friends) and reports to the plugin with one JSON object
per line on stdout; everything ETSAM prints goes to stderr. Modes::

    python etsam_worker.py segment-array --input volume.npy --output logits.npy [options]
    python etsam_worker.py postprocess --input mask.npy --output cleaned.npy

Events: start, progress, done; any mode may end with fatal.
"""

import sys

# This file lives inside a plugin package next to modules such as cmd.py;
# keep them from shadowing standard-library modules in this process.
sys.path.pop(0)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

if sys.platform == "win32":
    # The plugins start this Python directly, without "conda activate". Put the
    # environment's DLL folders on the PATH as activation does: cuDNN loads its
    # parts (cudnn_graph64_9.dll, ...) from Library\bin through the PATH, and
    # without it the first GPU inference crashes the process.
    _dll_dirs = [
        os.path.join(sys.prefix, *parts)
        for parts in ((), ("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"))
    ]
    os.environ["PATH"] = os.pathsep.join([*_dll_dirs, os.environ.get("PATH", "")])

_events = sys.stdout
sys.stdout = sys.stderr  # ETSAM's own prints must not mix with the events


def emit(event, **fields):
    _events.write(json.dumps({"event": event, **fields}) + "\n")
    _events.flush()


def _device():
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", f"{props.name} ({props.total_memory / 1024**3:.0f} GB)", True
    return "cpu", "CPU", False


GPU_BUSY_TEXTS = (
    "busy or unavailable",
    "CUDA_ERROR_ALREADY_ACQUIRED",
    "cudaErrorDevicesUnavailable",
)
GPU_BUSY_MESSAGE = (
    "The GPU is busy: another program is keeping ETSAM from using it. Restart the "
    "application (ChimeraX or napari) and try again; if that does not help, close other "
    "programs that use the GPU."
)


def _friendly(e):
    """Turn common low-level failures into actionable messages."""
    text = str(e)
    if isinstance(e, ModuleNotFoundError) and e.name in (
        "etsam",
        "postprocess",
        "inference",
        "sam2",
    ):
        return (
            f"ETSAM is not installed in the environment {sys.prefix}. In the ETSAM folder, with "
            "this environment activated, run: pip install --no-deps -e .  (step 3 of "
            "https://jianlin-cheng.github.io/ETSAM/installation/#method-2-conda-environment-from-source)"
        )
    if any(t in traceback.format_exc() for t in GPU_BUSY_TEXTS):
        return GPU_BUSY_MESSAGE
    if "out of memory" in text.lower() or type(e).__name__ == "OutOfMemoryError":
        return (
            "The GPU ran out of memory. Close other programs that use the GPU, use "
            "split processing, or bin the tomogram, and try again."
        )
    if isinstance(e, MemoryError):
        return (
            "Not enough RAM to segment this tomogram (ETSAM needs about 15-25 GB for a "
            "1000 x 1000 x 500 tomogram). Close other programs or bin the tomogram."
        )
    if isinstance(e, PermissionError):
        return f"{text}. Choose another output folder."
    return text or type(e).__name__


def segment_array(args):
    """Segment a (Z, Y, X) array saved with numpy.save; write the logits the same way."""
    import etsam
    import numpy as np

    device, name, accelerated = _device()
    emit("start", device=device, device_name=name, accelerated=accelerated)
    data = np.load(args.input)
    if data.ndim != 3:
        raise ValueError(f"ETSAM needs a 3D tomogram (Z, Y, X), but got shape {data.shape}.")
    last = [0.0]

    def progress(fraction, message):
        now = time.monotonic()
        if now - last[0] >= 0.1:
            last[0] = now
            emit("progress", fraction=round(fraction, 4), message=message)

    started = time.monotonic()
    logits = etsam.segment(
        data,
        normalize_method=args.normalize_method,
        split_processing=args.split_processing,
        stage1_prompt=args.stage1_prompt,
        stage2_prompt=args.stage2_prompt,
        progress=progress,
    )
    del data
    emit("progress", fraction=1.0, message="Saving result")
    np.save(args.output, np.asarray(logits, dtype=np.float32))
    emit("done", output=args.output, seconds=round(time.monotonic() - started, 1))


def postprocess_array(args):
    """Clean up a binary mask saved with numpy.save (no PyTorch needed)."""
    import numpy as np
    import postprocess

    emit("start")
    mask = np.load(args.input)
    emit("progress", fraction=0.0, message="Cleaning up the mask")
    np.save(args.output, postprocess.postprocess(mask).astype(np.uint8))
    emit("done", output=args.output)


def _add_model_options(parser):
    parser.add_argument("--split-processing", action="store_true")
    parser.add_argument("--normalize-method", default="min_max_positive_values")
    parser.add_argument("--stage1-prompt", default="grid_zero")
    parser.add_argument("--stage2-prompt", default="zero")


def main():
    parser = argparse.ArgumentParser("etsam_worker")
    sub = parser.add_subparsers(dest="mode", required=True)
    arr = sub.add_parser("segment-array")
    arr.add_argument("--input", required=True)
    arr.add_argument("--output", required=True)
    _add_model_options(arr)
    post = sub.add_parser("postprocess")
    post.add_argument("--input", required=True)
    post.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "segment-array":
            segment_array(args)
        else:
            postprocess_array(args)
    except Exception as e:  # noqa: BLE001 - e.g. ETSAM or PyTorch failed to import; reported to the plugin
        traceback.print_exc()
        emit(
            "fatal",
            error=_friendly(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
