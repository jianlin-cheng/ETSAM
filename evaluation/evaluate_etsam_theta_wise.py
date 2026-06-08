import argparse
import os

import mrcfile
import numpy as np
import pandas as pd
import pymeshlab as pm


def extract_point_cloud_xyz(mask, voxel_size):
    """Extract all label-1 voxels as a physical XYZ point cloud (nm), following mrc2xyz.py.

    Returns:
        points_xyz: (N, 3) float64 array in nm, columns are [x, y, z] where z = beam direction
        voxel_indices: (N, 3) int array of (z, y, x) voxel indices for mapping back
    Returns (None, None) if mask is empty.
    """
    data = np.where(mask == 1)
    if len(data[0]) == 0:
        return None, None
    vs = float(voxel_size.x) / 10.0  # angstrom → nm, isotropic (matches mrc2xyz.py)
    points_xyz = np.column_stack([
        data[2] * vs,  # x  (numpy axis 2 → physical x)
        data[1] * vs,  # y  (numpy axis 1 → physical y)
        data[0] * vs,  # z  (numpy axis 0 → physical z = beam direction)
    ]).astype(np.float64)
    voxel_indices = np.column_stack([data[0], data[1], data[2]])  # (N, 3) as (z, y, x)
    return points_xyz, voxel_indices


def compute_normals_pymeshlab(points_xyz, k_neighbors=30):
    """Compute per-point normals using pymeshlab's k-NN normal estimation.

    Returns (N, 3) normal matrix where column 2 = nz (beam-direction component).
    Returns None if the point cloud has fewer than k_neighbors points.
    """
    if len(points_xyz) < k_neighbors:
        return None
    ms = pm.MeshSet()
    ms.add_mesh(pm.Mesh(vertex_matrix=points_xyz))
    ms.compute_normal_for_point_clouds(k=k_neighbors, smoothiter=1)
    return ms.current_mesh().vertex_normal_matrix()  # (N, 3): columns are [nx, ny, nz]


def compute_theta_degrees(normals_xyz):
    """Compute theta = angle between each normal and the Z-axis (beam direction) in degrees.

    Column 2 of pymeshlab normals (nz) corresponds to the beam-direction (z = axis 0 in numpy).
    theta = arccos(|nz|) for unit normals → range [0, 90].
    """
    return np.degrees(np.arccos(np.clip(np.abs(normals_xyz[:, 2]), 0.0, 1.0)))


def save_theta_heatmap(mask_shape, voxel_indices, bin_indices, voxel_size, output_path):
    """Save a theta-bin heatmap MRC for visualization.

    Voxel values: 0 = background, 1 = first bin, 2 = second bin, ...
    """
    heatmap = np.zeros(mask_shape, dtype=np.uint8)
    zs, ys, xs = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    heatmap[zs, ys, xs] = (bin_indices + 1).astype(np.uint8)
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(heatmap)
        mrc.voxel_size = voxel_size


def compute_theta_binned_metrics(gt_mask, pred_mask, voxel_size, bins, k_neighbors=30):
    """Compute precision and recall per theta bin using independent binning.

    For Recall: normals are computed from the GT mask; GT voxels are binned by theta;
        recall = (GT voxels in bin also predicted) / (Total GT voxels in bin).
    For Precision: normals are computed from the Predicted mask; predicted voxels are binned;
        precision = (Predicted voxels in bin overlapping GT) / (Total Predicted voxels in bin).

    Args:
        gt_mask: binary 3D numpy array (uint8)
        pred_mask: binary 3D numpy array (uint8)
        voxel_size: mrcfile voxel_size object
        bins: list of bin edges in degrees, e.g. [0, 30, 60, 90]
        k_neighbors: number of neighbors for pymeshlab normal estimation

    Returns:
        metrics: dict of metric values keyed by e.g. "theta_0_30_recall"
        theta_info: dict with "gt" and "pred" tuples of (voxel_indices, bin_indices)
    """
    bins = np.array(bins, dtype=float)
    n_bins = len(bins) - 1

    # --- RECALL side: normals from GT mask ---
    gt_pts_xyz, gt_vox_idx = extract_point_cloud_xyz(gt_mask, voxel_size)
    if gt_pts_xyz is not None:
        gt_normals = compute_normals_pymeshlab(gt_pts_xyz, k_neighbors)
        if gt_normals is not None:
            gt_theta = compute_theta_degrees(gt_normals)
            gt_bin_idx = np.clip(np.digitize(gt_theta, bins) - 1, 0, n_bins - 1)
        else:
            gt_bin_idx = None
    else:
        gt_bin_idx = None

    # --- PRECISION side: normals from Predicted mask ---
    pred_pts_xyz, pred_vox_idx = extract_point_cloud_xyz(pred_mask, voxel_size)
    if pred_pts_xyz is not None:
        pred_normals = compute_normals_pymeshlab(pred_pts_xyz, k_neighbors)
        if pred_normals is not None:
            pred_theta = compute_theta_degrees(pred_normals)
            pred_bin_idx = np.clip(np.digitize(pred_theta, bins) - 1, 0, n_bins - 1)
        else:
            pred_bin_idx = None
    else:
        pred_bin_idx = None

    metrics = {}
    for b in range(n_bins):
        lo, hi = int(bins[b]), int(bins[b + 1])
        label = f"theta_{lo}_{hi}"

        # Recall
        if gt_bin_idx is not None:
            gt_in_bin = gt_vox_idx[gt_bin_idx == b]
            gt_count = len(gt_in_bin)
            if gt_count > 0:
                zs, ys, xs = gt_in_bin[:, 0], gt_in_bin[:, 1], gt_in_bin[:, 2]
                tp_recall = int(pred_mask[zs, ys, xs].sum())
                recall = tp_recall / gt_count
            else:
                recall = float("nan")
        else:
            gt_in_bin = np.empty((0, 3), dtype=int)
            gt_count = 0
            recall = float("nan")

        # Precision
        if pred_bin_idx is not None:
            pred_in_bin = pred_vox_idx[pred_bin_idx == b]
            pred_count = len(pred_in_bin)
            if pred_count > 0:
                zs, ys, xs = pred_in_bin[:, 0], pred_in_bin[:, 1], pred_in_bin[:, 2]
                tp_prec = int(gt_mask[zs, ys, xs].sum())
                precision = tp_prec / pred_count
            else:
                precision = float("nan")
        else:
            pred_in_bin = np.empty((0, 3), dtype=int)
            pred_count = 0
            precision = float("nan")

        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            dice = float("nan")
        else:
            dice = 2 * precision * recall / (precision + recall)

        denom = precision + recall - (precision * recall)
        if np.isnan(precision) or np.isnan(recall) or denom == 0:
            iou = float("nan")
        else:
            iou = (precision * recall) / denom

        metrics[f"{label}_dice"] = dice
        metrics[f"{label}_iou"] = iou
        metrics[f"{label}_gt_count"] = gt_count
        metrics[f"{label}_pred_count"] = pred_count

    theta_info = {
        "gt": (gt_vox_idx, gt_bin_idx) if gt_bin_idx is not None else (None, None),
        "pred": (pred_vox_idx, pred_bin_idx) if pred_bin_idx is not None else (None, None),
    }
    return metrics, theta_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_etsam_theta_wise.py")
    parser.add_argument(
        "--csv", help="CSV file containing the test dataset", default="data/testset.csv", type=str
    )
    parser.add_argument(
        "--collection-dir",
        help="Collection directory containing the test dataset",
        default="data/collection",
        type=str,
    )
    parser.add_argument(
        "--predictions-dir",
        help="Predictions directory containing the stage2 logits",
        default="results/etsam_testset_predictions",
        type=str,
    )
    parser.add_argument(
        "--results",
        help="CSV file to store the results",
        default="results/etsam_testset_predictions/etsam_testset_theta_wise_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--stage2-logit-threshold",
        help="Stage 2 logit threshold",
        default=-0.25,
        type=float,
    )
    parser.add_argument(
        "--theta-bins",
        help="Comma-separated bin edges in degrees (e.g. '0,30,60,90')",
        default="0,30,60,90",
        type=str,
    )

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = os.path.abspath(args.collection_dir)
    predictions_dir = os.path.abspath(args.predictions_dir)
    results_file = os.path.abspath(args.results)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    theta_bins = [float(v) for v in args.theta_bins.split(",")]
    k_neighbors = 30
    stage2_logit_threshold = args.stage2_logit_threshold

    results_df = pd.DataFrame()
    df = pd.read_csv(csv_file, dtype={"dataset_id": str, "run_id": str})
    for index, row in df.iterrows():
        dataset_id = row["dataset_id"]
        run_id = row["run_id"]

        print(f"==> Processing {dataset_id}:{run_id}")

        results_df.loc[index, "dataset_id"] = dataset_id
        results_df.loc[index, "run_id"] = run_id

        run_dir = os.path.join(collection_dir, dataset_id, run_id)
        gt_mask_file_path = os.path.join(run_dir, "target_mask", f"{run_id}_target_mask.mrc")

        stage2_output_logits_file_path = os.path.join(
            predictions_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage2_seg_logits.mrc"
        )

        try:
            if not os.path.exists(stage2_output_logits_file_path):
                print("==> No cached predictions found, run evaluate_etsam.py to generate the predictions.")
                continue

            print("==> Found cached predictions, computing metrics.")
            with mrcfile.open(gt_mask_file_path) as mrc:
                gt_mask_shape = mrc.data.shape
                gt_mask = mrc.data.copy().astype(np.uint8)
                voxel_size = mrc.voxel_size

            with mrcfile.open(stage2_output_logits_file_path) as mrc:
                etsam_stage2_seg_logits = mrc.data.copy()
                etsam_stage2_seg_mask = (etsam_stage2_seg_logits > stage2_logit_threshold).astype(
                    np.uint8
                )

            if etsam_stage2_seg_mask.shape != gt_mask_shape:
                print(
                    "Error: Cached prediction shape does not match target mask shape "
                    f"{gt_mask_shape}"
                )
                continue

            print(f"==> Computing theta-binned metrics (bins={theta_bins})...")
            metrics, _ = compute_theta_binned_metrics(
                gt_mask, etsam_stage2_seg_mask, voxel_size, theta_bins, k_neighbors
            )
            for key, value in metrics.items():
                results_df.loc[index, key] = value

            for b in range(len(theta_bins) - 1):
                lo, hi = int(theta_bins[b]), int(theta_bins[b + 1])
                label = f"theta_{lo}_{hi}"
                print(
                    f"[{lo}-{hi}°] => "
                    f"Precision: {metrics[f'{label}_precision']:.4f}, "
                    f"Recall: {metrics[f'{label}_recall']:.4f}, "
                    f"Dice: {metrics[f'{label}_dice']:.4f}, "
                    f"IoU: {metrics[f'{label}_iou']:.4f}, "
                    f"GT count: {metrics[f'{label}_gt_count']}, "
                    f"Pred count: {metrics[f'{label}_pred_count']}"
                )

            results_df.to_csv(results_file, index=False)
            print("-------------------------------------------------------------------------------------------------")

        except Exception as e:
            print("Error: " + str(e))

    print(f"Successfully evaluated ETSAM with theta-wise metrics. Results stored in {results_file}")
