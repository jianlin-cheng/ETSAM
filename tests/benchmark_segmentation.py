"""Measure the time and peak memory of segmenting a tomogram with this checkout.

    python tests/benchmark_segmentation.py TOMOGRAM.mrc [--split] [--output-type mask]
        [--save result.npy] [--compare reference.npy]

Uses the installed checkpoints; downloads nothing. Prints one JSON line with the
segmentation time (excluding CUDA start-up), the process's peak RAM and the
peak CUDA memory. --compare checks that the result is byte identical to a
result saved earlier with --save. ETSAM is imported from the current folder, so
a copy of this script can measure other revisions too.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import mrcfile
import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # the ETSAM checkout to measure
import etsam  # noqa: E402


def peak_ram_mib():
    if os.name != "nt":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10

    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters(cb=ctypes.sizeof(Counters))
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tomogram")
    parser.add_argument("--slices", type=int, help="only segment the first N slices")
    parser.add_argument("--split", action="store_true", help="split processing")
    parser.add_argument("--normalize-method", default="min_max_positive_values")
    parser.add_argument("--stage1-prompt", default="grid_zero")
    parser.add_argument("--stage2-prompt", default="zero")
    parser.add_argument("--output-type", choices=("logits", "mask"), default="logits")
    parser.add_argument("--save", help="save the result to this .npy file")
    parser.add_argument("--compare", help="compare the result with this .npy file")
    args = parser.parse_args()

    checkpoints = Path(sys.prefix) / "checkpoints"
    etsam.ensure_checkpoints = lambda progress=None: [str(checkpoints / f"etsam_stage{i}_v1.pt") for i in (1, 2)]
    if torch.cuda.is_available():
        torch.ones(1, device="cuda").sum().item()  # exclude CUDA start-up

    with mrcfile.mmap(args.tomogram, mode="r") as mrc:
        tomogram = mrc.data[: args.slices]
        started = time.perf_counter()
        result = etsam.segment(
            tomogram,
            normalize_method=args.normalize_method,
            split_processing=args.split,
            stage1_prompt=args.stage1_prompt,
            stage2_prompt=args.stage2_prompt,
            **({"output_type": "mask"} if args.output_type == "mask" else {}),
        )
        seconds = time.perf_counter() - started

    report = {"seconds": round(seconds, 2), "peak_ram_mib": round(peak_ram_mib(), 1)}
    if torch.cuda.is_available():
        report["peak_cuda_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if args.compare:
        reference = np.load(args.compare, mmap_mode="r")
        report["identical"] = bool(
            reference.shape == result.shape and reference.dtype == result.dtype
            and all(reference[i].tobytes() == result[i].tobytes() for i in range(len(result)))
        )
        del reference
    if args.save:
        np.save(args.save, result)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
