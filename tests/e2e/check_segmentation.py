"""Check an ETSAM result on a real tomogram (end-to-end tests in CI).

    python tests/e2e/check_segmentation.py TOMOGRAM.mrc [--mask MASK] [--logits LOGITS]
        [--threshold -0.25] [--reference LOGITS] [--min-membrane 0.01]

MASK and LOGITS are .mrc files (from the etsam command) or .npy files (from the
plugins' worker); without --mask, the mask is the logits thresholded. Checks
that they match the tomogram's shape (and voxel size), that the mask is the
logits thresholded (unless --postprocessed), and that the last quarter of the slices contains membrane: ETSAM finds membranes only after
tracking for about 20 slices, so the first slices of a small test tomogram stay
empty. --reference compares the logits with another run's (for example MPS
with the CPU): they must be strongly correlated and their masks must overlap.
The correlation is of the logits clipped to [-20, 20]: on slices where the
model sees no object, all logits are -1024, and a run that starts finding
membranes a slice earlier would otherwise dominate it.
"""

import argparse
import sys

import mrcfile
import numpy as np


def load(path):
    """Array and voxel size (None for .npy files)."""
    if path.endswith(".npy"):
        return np.load(path), None
    with mrcfile.open(path, permissive=True) as mrc:
        return mrc.data.copy(), mrc.voxel_size.copy()


def dice(a, b):
    total = np.count_nonzero(a) + np.count_nonzero(b)
    return 1.0 if total == 0 else 2 * np.count_nonzero(a & b) / total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tomogram")
    parser.add_argument("--mask")
    parser.add_argument("--logits")
    parser.add_argument("--threshold", type=float, default=-0.25)
    parser.add_argument("--postprocessed", action="store_true", help="the mask was cleaned up")
    parser.add_argument("--reference", help="logits of another run to compare with")
    parser.add_argument("--min-membrane", type=float, default=0.01)
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--min-dice", type=float, default=0.9)
    args = parser.parse_args()

    failures = []

    def check(condition, message):
        print(("ok:   " if condition else "FAIL: ") + message)
        if not condition:
            failures.append(message)

    if not (args.mask or args.logits):
        parser.error("give --mask, --logits or both")
    tomogram, voxel_size = load(args.tomogram)
    if args.mask:
        mask, mask_voxel_size = load(args.mask)
    else:
        mask, mask_voxel_size = (load(args.logits)[0] > args.threshold).astype(np.uint8), None
    check(mask.shape == tomogram.shape, f"mask shape {mask.shape} == tomogram shape {tomogram.shape}")
    if mask_voxel_size is not None:
        check(mask_voxel_size == voxel_size, f"mask voxel size {mask_voxel_size} == {voxel_size}")
    check(set(np.unique(mask)) <= {0, 1}, f"mask values {np.unique(mask)[:5]} are 0 and 1")
    mask = mask.astype(bool)
    tail = mask[-max(1, len(mask) // 4):]
    fraction = np.count_nonzero(tail) / tail.size
    check(fraction >= args.min_membrane,
          f"membrane in the last quarter of the slices: {100 * fraction:.2f}% >= {100 * args.min_membrane:.2f}%")
    print(f"      membrane per slice (%): {np.round(100 * mask.mean(axis=(1, 2)), 2).tolist()}")

    if args.logits:
        logits, logits_voxel_size = load(args.logits)
        check(logits.shape == tomogram.shape, f"logits shape {logits.shape} == {tomogram.shape}")
        check(logits.dtype == np.float32, f"logits dtype {logits.dtype} is float32")
        check(bool(np.isfinite(logits).all()), "logits are finite")
        if logits_voxel_size is not None:
            check(logits_voxel_size == voxel_size, f"logits voxel size {logits_voxel_size} == {voxel_size}")
        if not args.postprocessed:
            check(np.array_equal(mask, logits > args.threshold),
                  f"mask == logits > {args.threshold}")
        if args.reference:
            reference, _ = load(args.reference)
            clipped = [np.clip(a, -20, 20).ravel() for a in (logits, reference)]
            correlation = float(np.corrcoef(*clipped)[0, 1])
            check(correlation >= args.min_correlation,
                  f"correlation with the reference logits (clipped to ±20): {correlation:.6f} "
                  f">= {args.min_correlation}")
            overlap = dice(logits > args.threshold, reference > args.threshold)
            check(overlap >= args.min_dice,
                  f"Dice overlap with the reference mask: {overlap:.4f} >= {args.min_dice}")
            print("      Dice per slice:", [round(float(dice(a > args.threshold, b > args.threshold)), 3)
                                         for a, b in zip(logits, reference)])
            no_object = [i for i, (a, b) in enumerate(zip(logits, reference))
                         if (a.max() <= -1000) != (b.max() <= -1000)]
            print(f"      slices where only one run sees no object: {no_object}")

    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("All checks passed.")


if __name__ == "__main__":
    main()
