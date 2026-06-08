import argparse
import glob
import os
import sys

import mrcfile
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.classification import binary_f1_score
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional.classification import binary_precision
from torchmetrics.functional.classification import binary_recall

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEMBRANE_OBJECT_ID = 1

PLANE_AXIS_MAP = {
    "xy": 0,  # fix z, slice shape is (y, x)
    "xz": 1,  # fix y, slice shape is (z, x)
    "yz": 2,  # fix x, slice shape is (z, y)
}

def save_mask(mask, voxel_size, output_map_file_path):
    with mrcfile.new(output_map_file_path, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.uint8))
        mrc.voxel_size = voxel_size


def save_logits(logits, voxel_size, output_map_file_path):
    with mrcfile.new_mmap(
        output_map_file_path, shape=logits.shape, mrc_mode=2, fill=0, overwrite=True
    ) as mrc:
        mrc.set_data(logits)
        mrc.voxel_size = voxel_size


def compute_slice_metrics(mask_slice, gt_slice):
    mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).to("cuda")
    dice = binary_f1_score(mask_tensor, gt_tensor, zero_division=0).item()
    iou = binary_jaccard_index(mask_tensor, gt_tensor, zero_division=0).item()
    precision = binary_precision(mask_tensor, gt_tensor, zero_division=0).item()
    recall = binary_recall(mask_tensor, gt_tensor, zero_division=0).item()

    del mask_tensor, gt_tensor
    torch.cuda.empty_cache()
    return dice, iou, precision, recall


def compute_plane_wise_average_metrics(mask_3d, gt_mask_3d, plane):
    axis = PLANE_AXIS_MAP[plane]
    slice_metrics = []

    num_slices = mask_3d.shape[axis]
    for slice_idx in range(num_slices):
        pred_slice = np.take(mask_3d, indices=slice_idx, axis=axis).astype(np.uint8)
        gt_slice = np.take(gt_mask_3d, indices=slice_idx, axis=axis).astype(np.uint8)
        slice_metrics.append(compute_slice_metrics(pred_slice, gt_slice))

    return np.mean(np.array(slice_metrics), axis=0)


def compute_all_plane_metrics(mask_3d, gt_mask_3d):
    metrics = {}
    for plane in ("xy", "xz", "yz"):
        dice, iou, precision, recall = compute_plane_wise_average_metrics(
            mask_3d, gt_mask_3d, plane
        )
        metrics[f"{plane}_dice"] = dice
        metrics[f"{plane}_iou"] = iou
        metrics[f"{plane}_precision"] = precision
        metrics[f"{plane}_recall"] = recall
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_etsam_plane_wise.py")
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
        help="Directory containing precomputed ETSAM prediction logits",
        default="results/etsam_testset_predictions",
        type=str,
    )
    parser.add_argument(
        "--results",
        help="CSV file to store the results",
        default="results/etsam_testset_predictions/etsam_testset_plane_wise_metrics.csv",
        type=str,
    )
    parser.add_argument(
        "--stage2-logit-threshold",
        help="Stage 2 logit threshold",
        default=-0.25,
        type=float,
    )

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = os.path.abspath(args.collection_dir)
    results_file = os.path.abspath(args.results)
    predictions_dir = os.path.abspath(args.predictions_dir)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

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
        input_tomogram_file_path = glob.glob(os.path.join(run_dir, "tomogram", "*.mrc"))[0]
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
                gt_mask = mrc.data.copy().astype(np.uint8)

            with mrcfile.open(stage2_output_logits_file_path) as mrc:
                etsam_stage2_seg_logits = mrc.data.copy()
                etsam_stage2_seg_mask = (etsam_stage2_seg_logits > stage2_logit_threshold).astype(np.uint8)

            metrics = compute_all_plane_metrics(etsam_stage2_seg_mask, gt_mask)

            for key, value in metrics.items():
                results_df.loc[index, key] = value

            for plane in ("xy", "xz", "yz"):
                print(
                    f"{plane.upper()} => Dice: {metrics[f'{plane}_dice']}, "
                    f"IoU: {metrics[f'{plane}_iou']}, "
                    f"Precision: {metrics[f'{plane}_precision']}, "
                    f"Recall: {metrics[f'{plane}_recall']}"
                )
            results_df.to_csv(results_file, index=False)
            print("-------------------------------------------------------------------------------------------------")
        except Exception as e:
            print("Error: " + str(e))

    print(f"Successfully evaluated ETSAM with plane-wise metrics. Results stored in {results_file}")
