"""Compute per-surface area-weighted median membrane thickness -> CSV.

This is the ONLY script that reads the per-triangle data from the refined PyCurv
graphs (*_refined.gt produced by measure_thickness.py). For each surface it reduces
the per-triangle `thickness` (weighted by triangle `area`) to a single area-weighted
median — the paper's per-surface metric (Medina et al. JCB 2026, Materials & Methods,
"Statistical inference": "the area-weighted median of each quantification was used as
the overall measurement for that surface").

The output CSV is consumed by generate_default_report.py, which does NO computation —
it only reads this file. Re-run this script whenever the underlying graphs change.

Usage:
    python compute_thickness_medians.py

Output:
    results/thickness_default_persurface.csv
        columns: group, organelle, surface_key, surface_median_nm, n_triangles
"""
import csv
import os
from glob import glob
from pathlib import Path

import numpy as np
from graph_tool import load_graph
from morphometrics_stats import weighted_median

REPO = Path("/home/joel/bml_drive/surface_morphometrics")
RH = 9                                  # radius_hit of the refined graphs
ORGS = ["OMM", "IMM", "ER"]

# (group label, work directory holding the *_refined.gt graphs)
GROUPS = [
    ("Paper segmentation", REPO / "results/mef_newpipe_default"),
    ("ETSAM segmentation", REPO / "results/mef_etsam_newpipe_default"),
]
OUT_CSV = REPO / "results/thickness_default_persurface.csv"


def surface_key(path):
    """Tomogram/mito key shared by the paper and ETSAM filenames.

    paper : MIM019_2_lam1_ts_002_Mito_ER.labels_OMM.AVV_rh9_refined.gt
    etsam : MIM019_2_lam1_ts_002_etsam_to_Mito_ER.labels_OMM.AVV_rh9_refined.gt
    both -> MIM019_2_lam1_ts_002_Mito_ER
    """
    name = os.path.basename(path).replace("_etsam_to_", "_")
    return name.split(".labels_")[0]


def area_weighted_median(gt_file):
    """Area-weighted median per-triangle thickness for one refined graph.

    Returns (median_nm, n_triangles_used), or (None, 0) if there is no usable data.
    Excludes the thickness == 4.0 fit-failure placeholder and non-positive values.
    """
    graph = load_graph(str(gt_file))
    if "thickness" not in graph.vp or "area" not in graph.vp:
        return None, 0
    thickness = graph.vp["thickness"].get_array().astype(float)
    area = graph.vp["area"].get_array().astype(float)
    keep = (np.isfinite(thickness) & (thickness > 0) & (thickness != 4.0)
            & np.isfinite(area) & (area > 0))
    if not keep.any():
        return None, 0
    return float(weighted_median(thickness[keep], area[keep])), int(keep.sum())


def main():
    rows = []
    for group, work_dir in GROUPS:
        for org in ORGS:
            for gt_file in sorted(glob(f"{work_dir}/*_{org}.AVV_rh{RH}_refined.gt")):
                median, n = area_weighted_median(gt_file)
                key = surface_key(gt_file)
                if median is None:
                    print(f"  SKIP (no thickness): {os.path.basename(gt_file)}")
                    continue
                rows.append((group, org, key, median, n))
                print(f"  {group:20s} {org:3s} {key:42s} {median:.4f} nm  ({n} tri)")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "organelle", "surface_key", "surface_median_nm", "n_triangles"])
        for group, org, key, median, n in rows:
            writer.writerow([group, org, key, f"{median:.6f}", n])

    print(f"\nWrote {len(rows)} per-surface medians -> {OUT_CSV}")


if __name__ == "__main__":
    main()
