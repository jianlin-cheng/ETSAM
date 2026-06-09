"""Classify ETSAM membrane voxels as Vesicle using the paper otherOrg segmentation.

ETSAM produces a single binary membrane mask per tomogram covering ALL membranes
(mito + ER + vesicles). The mito classifier (etsam_to_multilabel.py) extracts the
OMM/IMM/ER voxels. This companion script extracts the *vesicle* voxels using the
paper's `*_otherOrg.labels.mrc` segmentation as the labelling oracle.

Per config_mef_other.yml, the otherOrg segmentation uses label 1 = small vesicles
(VesicleSm) and label 2 = larger vesicle/membrane structures (Vesicle); labels >=3
denote OTHER (non-vesicle) organelles and are ignored. We merge labels {1, 2} into a
single "vesicle present" oracle (matching the paper's single "vesicle" thickness
category in Fig 2). A tomogram whose otherOrg file has no label 1 or 2 has no vesicle
and is skipped (e.g. *_lam1_ts_004 has only labels {3, 4}). Every ETSAM membrane voxel
within NEAREST_NM of a vesicle-oracle voxel is labelled Vesicle (4); all other voxels
become background (0) — the mito membranes are handled by the separate mito file, so
they are not marked UNASSIGNED here. Produces a clean single-label MRC at the ETSAM grid.

Output: <output_dir>/<tomo_base>_etsam_to_Vesicle.labels.mrc  (labels {0, 4})

Usage:
    python etsam_to_vesicle.py <etsam_mask.mrc> <otherOrg_seg.mrc> <output_dir> \
        [--nearest-nm 30] [--vesicle-labels 1,2]

IMPORTANT: DO NOT DELETE ANY FILES. Suspect outputs are renamed to
*.corrupted_<UTC-timestamp>.mrc — never deleted.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# Reuse the validated helpers from the Phase-2 mito classifier.
from etsam_to_multilabel import (
    read_mrc, resample_paper_onto_etsam, write_multilabel_mrc, _utc_tag,
)

VESICLE_LABEL = 4
NEAREST_NM_DEFAULT = 30


def sanity_check_vesicle(path, expected_voxsize=9.98):
    if not os.path.exists(path):
        return False, "does not exist"
    if os.path.getsize(path) == 0:
        return False, "zero bytes"
    try:
        data, vs, _ = read_mrc(path)
        if abs(vs - expected_voxsize) > 0.5:
            return False, f"voxel_size={vs:.2f} A (expected ~{expected_voxsize})"
        unique = set(np.unique(data).tolist())
        if not unique.issubset({0, VESICLE_LABEL}):
            return False, f"unexpected labels: {unique - {0, VESICLE_LABEL}}"
    except Exception as e:
        return False, str(e)
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("etsam_mask", help="Post-processed ETSAM binary mask MRC (~9.98 A)")
    ap.add_argument("other_seg",  help="Paper otherOrg segmentation MRC (~13.30 A)")
    ap.add_argument("output_dir", help="Directory for output files")
    ap.add_argument("--nearest-nm", type=float, default=NEAREST_NM_DEFAULT,
                    help="Max distance (nm) from a vesicle-oracle voxel for assignment")
    ap.add_argument("--vesicle-labels", default="1,2",
                    help="Comma-separated otherOrg label values that are vesicles "
                         "(default 1,2 per config_mef_other.yml; >=3 are other organelles)")
    ap.add_argument("--mito-segs", nargs="*", default=[],
                    help="Paper Mito_ER segmentation MRC(s) for this tomogram. Voxels "
                         "closer to a mito membrane than to the vesicle oracle are NOT "
                         "claimed as vesicle (prevents mito contamination of the vesicle "
                         "surface). Pass all mito segs for multi-mito tomograms.")
    args = ap.parse_args()
    vesicle_labels = {int(x) for x in args.vesicle_labels.split(",") if x.strip()}

    os.makedirs(args.output_dir, exist_ok=True)

    mask_name = Path(args.etsam_mask).name
    tomo_base = mask_name.split("_etsam_predicted")[0] if "_etsam_predicted" in mask_name \
        else mask_name.split(".mrc")[0]

    out_mrc = os.path.join(args.output_dir, f"{tomo_base}_etsam_to_Vesicle.labels.mrc")
    out_csv = os.path.join(args.output_dir, f"{tomo_base}_etsam_to_Vesicle.classify.csv")

    ok, reason = sanity_check_vesicle(out_mrc)
    if ok:
        print(f"SKIP (already exists, passes sanity): {Path(out_mrc).name}")
        return
    if os.path.exists(out_mrc):
        corrupted = out_mrc + f".corrupted_{_utc_tag()}.mrc"
        print(f"  Moving suspect output: {Path(out_mrc).name} -> {Path(corrupted).name}  ({reason})")
        os.rename(out_mrc, corrupted)

    print(f"\n{'='*60}\nVesicle classify: {tomo_base}\n{'='*60}")

    print(f"Loading ETSAM mask: {Path(args.etsam_mask).name}")
    mask_data, etsam_vox, etsam_origin = read_mrc(args.etsam_mask)
    if abs(etsam_vox - 9.98) > 0.5:
        print(f"  ERROR: Unexpected ETSAM voxel size {etsam_vox:.2f} A (expected ~9.98).")
        sys.exit(1)
    mask_binary = (mask_data > 0)
    del mask_data
    print(f"  Shape: {mask_binary.shape}  voxsize: {etsam_vox:.3f} A  membrane voxels: {mask_binary.sum():,}")

    print(f"Loading otherOrg seg: {Path(args.other_seg).name}")
    other_data, other_vox, other_origin = read_mrc(args.other_seg)
    other_labels = np.round(other_data).astype(np.uint8)
    del other_data
    present = set(np.unique(other_labels).tolist()) - {0}
    vesicle_present = sorted(present & vesicle_labels)
    print(f"  Shape: {other_labels.shape}  voxsize: {other_vox:.3f} A  "
          f"otherOrg labels={sorted(present)}  vesicle labels used={vesicle_present}")
    if not vesicle_present:
        print(f"  NOTE: no vesicle labels {sorted(vesicle_labels)} present "
              f"(otherOrg has {sorted(present)}) — no vesicle in this tomogram, skipping.")
        sys.exit(0)
    # Merge the vesicle label(s) {1,2} into a single vesicle oracle.
    other_binary = np.isin(other_labels, list(vesicle_labels)).astype(np.uint8)
    del other_labels

    print(f"Resampling otherOrg seg ({other_vox:.2f} -> {etsam_vox:.2f} A/px) ...")
    other_resampled = resample_paper_onto_etsam(
        other_binary, other_vox, other_origin,
        mask_binary.shape, etsam_vox, etsam_origin,
    )
    del other_binary

    vox_nm = etsam_vox / 10.0
    max_vox = args.nearest_nm / vox_nm
    print(f"Per-voxel EDT vesicle assignment (nearest_nm={args.nearest_nm}, max_vox={max_vox:.1f}) ...")
    dist_ves = distance_transform_edt(other_resampled == 0).astype(np.float32)
    del other_resampled

    # Competition against mito membranes: build the union of all mito-membrane voxels
    # (any non-zero label in any Mito_ER seg) on the ETSAM grid, and only claim a voxel
    # as Vesicle when it is strictly closer to the vesicle oracle than to a mito membrane.
    dist_mito = np.full(mask_binary.shape, np.inf, dtype=np.float32)
    for mseg in args.mito_segs:
        if not os.path.exists(mseg):
            print(f"  WARNING: mito seg not found, skipping competition for: {mseg}")
            continue
        mdata, mvox, morigin = read_mrc(mseg)
        mbin = (np.round(mdata).astype(np.uint8) > 0).astype(np.uint8)
        del mdata
        mres = resample_paper_onto_etsam(mbin, mvox, morigin,
                                         mask_binary.shape, etsam_vox, etsam_origin)
        del mbin
        d = distance_transform_edt(mres == 0).astype(np.float32)
        del mres
        np.minimum(dist_mito, d, out=dist_mito)
        del d
    if args.mito_segs:
        print(f"  Mito competition using {len(args.mito_segs)} mito seg(s)")

    out = np.zeros(mask_binary.shape, dtype=np.uint8)
    claim = mask_binary & (dist_ves <= max_vox) & (dist_ves < dist_mito)
    out[claim] = VESICLE_LABEL
    dist = dist_ves  # for downstream prints/csv naming compatibility

    n_ves = int(np.sum(out == VESICLE_LABEL))
    total = int(mask_binary.sum())
    print(f"  Vesicle: {n_ves:,} voxels ({n_ves/max(1,total)*100:.1f}% of ETSAM membrane)")
    if n_ves == 0:
        print("  WARNING: no ETSAM voxels within threshold of vesicle oracle.")

    print(f"Writing: {Path(out_mrc).name}")
    write_multilabel_mrc(out_mrc, out, etsam_vox, etsam_origin)
    with open(out_csv, "w") as f:
        f.write(f"# tomogram={tomo_base} mode=vesicle nearest_nm={args.nearest_nm} "
                f"etsam_vox_angstrom={etsam_vox} vesicle_oracle_labels={vesicle_present} "
                f"otherOrg_all_labels={sorted(present)}\n")
        f.write("label,name,voxel_count,fraction\n")
        f.write(f"{VESICLE_LABEL},Vesicle,{n_ves},{round(n_ves/max(1,total),4)}\n")

    ok, reason = sanity_check_vesicle(out_mrc)
    if not ok:
        print(f"  ERROR: Output failed sanity check: {reason}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
