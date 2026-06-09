"""Convert ETSAM binary mask + paper multi-label segmentation → remapped multi-label MRC.

ETSAM produces a binary (membrane / background) mask that does not differentiate
organelle types. This script:
  1. Resamples the paper segmentation (13.30 Å/px) onto the ETSAM grid (9.98 Å/px)
     via nearest-neighbor interpolation.
  2. Assigns each ETSAM membrane voxel an organelle label using a per-voxel
     EDT nearest-label approach: each voxel is assigned to the paper-labeled
     organelle whose segmented voxels are closest (within NEAREST_LABEL_MAX_NM).
     Voxels farther than that threshold are marked UNASSIGNED (255).
  3. Writes a remapped multi-label MRC at the ETSAM grid with the correct voxel
     size and origin so all downstream scripts work without modification.
  4. Writes a summary diagnostic CSV.

This per-voxel approach is necessary because ETSAM often detects the OMM and IMM as
a single connected binary blob (the membranes are only ~20 nm apart). A per-blob
majority-vote would assign the entire mito membrane system to one class; the EDT
approach correctly splits each voxel to its nearest organelle.

Usage:
    python etsam_to_multilabel.py <etsam_mask.mrc> <paper_seg.mrc> <output_dir> \\
        [--nearest-nm 30] [--mito-tag Mito_ER]

    If output_dir/<tomo_base>_etsam_to_<mito-tag>.labels.mrc already exists and passes
    sanity checks, the script exits cleanly (idempotent). Corrupt outputs are renamed
    to *.corrupted_<UTC-timestamp>.mrc — never deleted.

IMPORTANT: DO NOT DELETE ANY FILES. Use mv to rename corrupted outputs.
"""
import argparse
import os
import sys
import csv
import datetime
from pathlib import Path

import numpy as np
import mrcfile
from scipy.ndimage import distance_transform_edt

NEAREST_NM_DEFAULT = 30   # max distance (nm) from paper label for voxel assignment
LABEL_MAP = {0: "bg", 1: "OMM", 2: "IMM", 3: "ER"}


def _utc_tag():
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def read_mrc(path):
    """Return (data ndarray, voxel_size_angstroms, origin_angstroms_xyz)."""
    for open_fn, kwargs in [
        (mrcfile.open, {"permissive": True}),
        (mrcfile.mmap, {"mode": "r", "permissive": True}),
    ]:
        try:
            with open_fn(path, **kwargs) as mrc:
                data   = np.array(mrc.data)
                voxsz  = float(mrc.voxel_size.x)
                origin = (float(mrc.header.origin.x),
                          float(mrc.header.origin.y),
                          float(mrc.header.origin.z))
            return data, voxsz, origin
        except Exception:
            continue
    raise RuntimeError(f"Could not open MRC: {path}")


def sanity_check_output(path, expected_voxsize=9.98):
    if not os.path.exists(path):
        return False, "does not exist"
    if os.path.getsize(path) == 0:
        return False, "zero bytes"
    try:
        data, vs, _ = read_mrc(path)
        if abs(vs - expected_voxsize) > 0.5:
            return False, f"voxel_size={vs:.2f} Å (expected ~{expected_voxsize})"
        unique = set(np.unique(data).tolist())
        allowed = {0, 1, 2, 3, 255}
        if not unique.issubset(allowed):
            return False, f"unexpected labels: {unique - allowed}"
    except Exception as e:
        return False, str(e)
    return True, "ok"


def resample_paper_onto_etsam(paper, paper_vox, paper_origin,
                               etsam_shape, etsam_vox, etsam_origin):
    """Map paper-label volume onto the ETSAM grid via nearest-neighbor indexing."""
    Sz, Sy, Sx = etsam_shape
    ratio = etsam_vox / paper_vox
    oz = (etsam_origin[0] - paper_origin[0]) / paper_vox
    oy = (etsam_origin[1] - paper_origin[1]) / paper_vox
    ox = (etsam_origin[2] - paper_origin[2]) / paper_vox

    iz = np.clip(np.round(oz + np.arange(Sz) * ratio).astype(int), 0, paper.shape[0] - 1)
    iy = np.clip(np.round(oy + np.arange(Sy) * ratio).astype(int), 0, paper.shape[1] - 1)
    ix = np.clip(np.round(ox + np.arange(Sx) * ratio).astype(int), 0, paper.shape[2] - 1)

    resampled = paper[np.ix_(iz, iy, ix)]

    # Physical extent sanity check
    physical_e = np.array(etsam_shape) * etsam_vox
    physical_p = np.array(paper.shape)  * paper_vox
    rel_err = np.abs(physical_e - physical_p) / physical_p
    if np.any(rel_err > 0.05):
        print(f"  WARNING: physical extents differ by >{rel_err.max()*100:.1f}% "
              f"(ETSAM={physical_e} Å  paper={physical_p} Å) — check origins/voxsizes")
    return resampled


def nearest_label_assignment(mask_binary, paper_resampled, etsam_vox_angstrom,
                              nearest_nm=30, er_nearest_nm=None):
    """
    Per-voxel nearest-label assignment using EDT.

    For each ETSAM membrane voxel, find the paper-labeled organelle whose segmented
    voxels are spatially closest (Euclidean distance). If the nearest paper-labeled
    voxel is within the per-label threshold, assign that label; otherwise UNASSIGNED.

    er_nearest_nm: separate threshold for ER (label 3). If None, uses nearest_nm.
    A tighter ER threshold (e.g. 15 nm) reduces over-segmentation of distal ER voxels
    that inflate the ETSAM ER area and bias the dual-Gaussian thickness fit downward.

    Returns: out (uint8 array, same shape as mask_binary)
    """
    vox_nm = etsam_vox_angstrom / 10.0
    # Per-label max voxel distances
    label_max_vox = {
        1: nearest_nm / vox_nm,        # OMM
        2: nearest_nm / vox_nm,        # IMM
        3: (er_nearest_nm if er_nearest_nm is not None else nearest_nm) / vox_nm,  # ER
    }

    out     = np.zeros(mask_binary.shape, dtype=np.uint8)
    min_dist = np.full(mask_binary.shape, np.inf, dtype=np.float32)

    for lbl in [1, 2, 3]:  # OMM, IMM, ER
        binary_lbl = (paper_resampled == lbl)
        if not binary_lbl.any():
            print(f"  NOTE: label {lbl} ({LABEL_MAP[lbl]}) absent from paper seg — skipping")
            continue
        max_vox = label_max_vox[lbl]
        # EDT: distance (in voxels) from every grid point to nearest lbl voxel
        dist = distance_transform_edt(~binary_lbl).astype(np.float32)

        # Update assignment for ETSAM membrane voxels where this label is closer
        # AND within this label's distance threshold
        improve = mask_binary & (dist < min_dist) & (dist <= max_vox)
        out[improve]      = lbl
        min_dist[improve] = dist[improve]

    # Mark UNASSIGNED where nearest paper label is beyond its per-label threshold.
    # A voxel is UNASSIGNED if it was never assigned (min_dist still inf) or its
    # assigned label's threshold was exceeded (tracked via min_dist at assignment time).
    unassigned = mask_binary & (out == 0)
    out[unassigned] = 255

    return out


def write_multilabel_mrc(out_path, out_arr, voxsize, origin):
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(out_arr)
        mrc.voxel_size = voxsize
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]


def write_summary_csv(out_path, tomo_base, mito_tag, out_arr, mask_binary,
                      etsam_vox, nearest_nm):
    total_vox = int(mask_binary.sum())
    rows = []
    for lbl, name in [(1, "OMM"), (2, "IMM"), (3, "ER"), (255, "UNASSIGNED")]:
        count = int(np.sum(out_arr == lbl))
        rows.append({"label": lbl, "name": name, "voxel_count": count,
                     "fraction": round(count / max(1, total_vox), 4)})
    with open(out_path, "w", newline="") as f:
        f.write(f"# tomogram={tomo_base} mito_tag={mito_tag} "
                f"nearest_nm={nearest_nm} etsam_vox_angstrom={etsam_vox}\n")
        w = csv.DictWriter(f, fieldnames=["label", "name", "voxel_count", "fraction"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("etsam_mask",  help="Post-processed ETSAM binary mask MRC")
    ap.add_argument("paper_seg",   help="Paper multi-label segmentation MRC (~13.30 Å)")
    ap.add_argument("output_dir",  help="Directory for output files")
    ap.add_argument("--nearest-nm", type=float, default=NEAREST_NM_DEFAULT,
                    help="Max distance (nm) from paper label for OMM and IMM assignment")
    ap.add_argument("--er-nearest-nm", type=float, default=None,
                    help="Separate threshold for ER (default: same as --nearest-nm). "
                         "Use 15 nm to reduce ER over-segmentation from distal boundary voxels.")
    ap.add_argument("--mito-tag",   default="Mito_ER",
                    help="Tag for the mito segmentation (used in output filename)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract clean tomogram base name — everything before "_etsam_predicted"
    mask_name = Path(args.etsam_mask).name
    if "_etsam_predicted" in mask_name:
        tomo_base = mask_name.split("_etsam_predicted")[0]
    else:
        # Fallback: strip .mrc suffixes
        tomo_base = mask_name.split(".mrc")[0]

    out_mrc = os.path.join(args.output_dir,
                           f"{tomo_base}_etsam_to_{args.mito_tag}.labels.mrc")
    out_csv = os.path.join(args.output_dir,
                           f"{tomo_base}_etsam_to_{args.mito_tag}.classify.csv")

    # Idempotency check
    ok, reason = sanity_check_output(out_mrc)
    if ok:
        print(f"SKIP (already exists, passes sanity): {Path(out_mrc).name}")
        return

    if os.path.exists(out_mrc):
        tag = _utc_tag()
        corrupted = out_mrc + f".corrupted_{tag}.mrc"
        print(f"  Moving suspect output: {Path(out_mrc).name} → {Path(corrupted).name}  ({reason})")
        os.rename(out_mrc, corrupted)

    print(f"\n{'='*60}")
    print(f"Processing: {tomo_base}  mito_tag={args.mito_tag}")
    print(f"{'='*60}")

    # Load ETSAM mask
    print(f"Loading ETSAM mask: {Path(args.etsam_mask).name}")
    mask_data, etsam_vox, etsam_origin = read_mrc(args.etsam_mask)
    if abs(etsam_vox - 9.98) > 0.5:
        print(f"  ERROR: Unexpected ETSAM voxel size {etsam_vox:.2f} Å (expected ~9.98).")
        sys.exit(1)
    mask_binary = (mask_data > 0)
    del mask_data
    print(f"  Shape: {mask_binary.shape}  voxsize: {etsam_vox:.3f} Å  "
          f"membrane voxels: {mask_binary.sum():,}")

    # Load paper segmentation
    print(f"Loading paper seg: {Path(args.paper_seg).name}")
    paper_data, paper_vox, paper_origin = read_mrc(args.paper_seg)
    paper_labels = np.round(paper_data).astype(np.uint8)
    del paper_data
    unique_labels = set(np.unique(paper_labels).tolist())
    print(f"  Shape: {paper_labels.shape}  voxsize: {paper_vox:.3f} Å  labels: {unique_labels}")
    unexpected = unique_labels - {0, 1, 2, 3}
    if unexpected:
        print(f"  WARNING: unexpected labels {unexpected} — setting to 0")
        paper_labels[~np.isin(paper_labels, [0, 1, 2, 3])] = 0

    # Resample paper labels onto ETSAM grid
    print(f"Resampling paper seg ({paper_vox:.2f} → {etsam_vox:.2f} Å/px) ...")
    paper_resampled = resample_paper_onto_etsam(
        paper_labels, paper_vox, paper_origin,
        mask_binary.shape, etsam_vox, etsam_origin
    )
    del paper_labels

    # Per-voxel EDT nearest-label assignment
    er_nm = args.er_nearest_nm
    print(f"Per-voxel EDT assignment (nearest_nm={args.nearest_nm}"
          f"{f', er_nearest_nm={er_nm}' if er_nm is not None else ''}) ...")
    out = nearest_label_assignment(mask_binary, paper_resampled, etsam_vox,
                                   args.nearest_nm, er_nearest_nm=er_nm)
    del paper_resampled

    # Summary
    total_vox = int(mask_binary.sum())
    for lbl, name in [(1, "OMM"), (2, "IMM"), (3, "ER"), (255, "UNASSIGNED")]:
        count = int(np.sum(out == lbl))
        pct   = count / max(1, total_vox) * 100
        print(f"  {name}: {count:,} voxels ({pct:.1f}%)")

    unassigned_frac = int(np.sum(out == 255)) / max(1, total_vox)
    print(f"  UNASSIGNED fraction: {unassigned_frac*100:.1f}%  (threshold: <15%)")

    # Write outputs
    print(f"Writing: {Path(out_mrc).name}")
    write_multilabel_mrc(out_mrc, out, etsam_vox, etsam_origin)
    write_summary_csv(out_csv, tomo_base, args.mito_tag, out, mask_binary,
                      etsam_vox,
                      f"{args.nearest_nm}(er={er_nm})" if er_nm else args.nearest_nm)
    print(f"Summary CSV: {Path(out_csv).name}")

    # Final sanity
    ok, reason = sanity_check_output(out_mrc)
    if not ok:
        print(f"  ERROR: Output failed sanity check: {reason}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
