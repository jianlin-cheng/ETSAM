"""Find the PyTorch operations that are inaccurate on a device (diagnostics).

    python tests/e2e/replay_ops.py TOMOGRAM.mrc [--device mps] [--slices 3]

Runs ETSAM's stage-1 model on the CPU, on the last slices of the tomogram, and
replays every PyTorch operation on the device with the CPU's exact inputs. Each
operation's own error is then known, without the errors of the operations
before it: a correct float32 operation differs from the CPU by about 1e-6
(relative to its largest value). Prints the call sites with the largest errors.
"""

import argparse
import collections
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import mrcfile  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._pytree import tree_flatten, tree_map  # noqa: E402

import etsam  # noqa: E402
import inference  # noqa: E402
from sam2.utils.volume import NormalizedTomogram  # noqa: E402

# Moving data between devices, or reading it: nothing to compare.
SKIP = {"to", "cpu", "cuda", "numpy", "item", "tolist", "detach", "clone", "contiguous",
        "pin_memory", "copy_", "__repr__", "__format__", "__len__", "__bool__", "__index__",
        "__int__", "__float__", "data_ptr", "storage_offset", "stride", "size", "dim", "numel"}


def location():
    """The first frame in ETSAM's own code: file:line (function)."""
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename
        if path.startswith(ROOT) and "replay_ops.py" not in path:
            return f"{os.path.relpath(path, ROOT)}:{frame.f_lineno} ({frame.f_code.co_name})"
        frame = frame.f_back
    return "?"


class Replay(TorchFunctionMode):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.errors = collections.defaultdict(lambda: [0.0, 0, None, 0.0])  # max, calls, shape, max abs
        self.failures = collections.Counter()

    def move(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(self.device)
        if isinstance(value, torch.device) and value.type == "cpu":
            return self.device
        return value

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        if name in SKIP or name.startswith("__get") or name.startswith("__set"):
            return func(*args, **kwargs)
        # Copied before the CPU runs the operation, which may change its inputs in place.
        moved = tree_map(self.move, args), tree_map(self.move, kwargs)
        out = func(*args, **kwargs)
        expected = [t for t in tree_flatten(out)[0]
                    if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())]
        if not expected:
            return out
        try:
            replayed = func(*moved[0], **moved[1])
        except Exception as e:  # noqa: BLE001 - an operation the device lacks is reported, not fatal
            self.failures[f"{name} at {location()}: {type(e).__name__}: {str(e)[:100]}"] += 1
            return out
        actual = [t for t in tree_flatten(replayed)[0]
                  if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())]
        for a, b in zip(expected, actual):
            a = a.detach().to(torch.complex128 if a.is_complex() else torch.float64)
            b = b.detach().cpu().to(a.dtype)
            if a.shape != b.shape:
                self.failures[f"{name} at {location()}: shape {tuple(a.shape)} != {tuple(b.shape)}"] += 1
                continue
            finite = torch.isfinite(a) & torch.isfinite(b)
            if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
                self.failures[f"{name} at {location()}: inf/nan differ"] += 1
            if not finite.any():
                continue
            scale = a[finite].abs().max().item()
            error = (a[finite] - b[finite]).abs().max().item() / max(scale, 1e-12)
            entry = self.errors[(name, location())]
            entry[1] += 1
            if error >= entry[0]:
                entry[0], entry[2], entry[3] = error, tuple(a.shape), scale
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tomogram")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--show", type=int, default=40)
    args = parser.parse_args()
    print("PyTorch", torch.__version__)
    data = mrcfile.read(args.tomogram)
    volume = NormalizedTomogram(np.ascontiguousarray(data[len(data) - args.slices:]))
    ckpt, _ = etsam.ensure_checkpoints()
    replay = Replay(args.device)
    track = inference.vos_inference

    def replayed_tracking(*a, **k):  # not while the model is built (on the meta device)
        with replay:
            return track(*a, **k)

    inference.vos_inference = replayed_tracking
    inference.etsam_inference(
            volume, config_path=etsam.CONFIG, ckpt_path=ckpt, prompt_method="grid_zero",
        grid_interval=50, device="cpu", output=np.empty(volume.shape, dtype=np.float32),
    )
    ranked = sorted(replay.errors.items(), key=lambda item: -item[1][0])
    print(f"== {len(ranked)} call sites, largest relative error on {args.device} first ==")
    for (name, where), (error, calls, shape, scale) in ranked[:args.show]:
        print(f"{error:9.2e}  {name:32s} {where}  shape {shape}  max |value| {scale:.3g}  ({calls} calls)")
    print(f"== operations that failed or differ in shape or inf/nan ({len(replay.failures)}) ==")
    for text, count in replay.failures.most_common(30):
        print(f"{count:5d}  {text}")


if __name__ == "__main__":
    main()
