import argparse
import mrcfile
import os
import numpy as np
import pandas as pd
import glob
import torch
from monai.metrics import DiceMetric
from monai.metrics import compute_iou
from torchmetrics.functional.classification import binary_precision
from torchmetrics.functional.classification import binary_recall
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference

MEMBRANE_OBJECT_ID = 1

def min_max_normalize(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array

def min_max_postive_values_normalize(array):
    array[array < 0] = 0
    array = array / np.max(array)
    return array

def save_mask(mask, voxel_size, output_map_file_path):
    with mrcfile.new(output_map_file_path, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.uint8))
        mrc.voxel_size = voxel_size

def save_logits(logits, voxel_size, output_map_file_path):
    with mrcfile.new_mmap(output_map_file_path, shape=logits.shape, mrc_mode=2, fill = 0, overwrite=True) as mrc:
        mrc.set_data(logits)
        mrc.voxel_size = voxel_size

def compute_dice_iou_precision_recall(mask, gt_mask):
    # Create a fresh DiceMetric instance for each computation to avoid state accumulation
    dice_metric = DiceMetric(include_background=True, reduction="mean", num_classes=1)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to("cuda")
    gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).to("cuda")
    dice = dice_metric(mask_tensor, gt_mask_tensor).item()
    iou = compute_iou(mask_tensor, gt_mask_tensor).item()
    precision = binary_precision(mask_tensor, gt_mask_tensor).item()
    recall = binary_recall(mask_tensor, gt_mask_tensor).item()
    # Explicitly delete tensors and clear CUDA cache
    del mask_tensor, gt_mask_tensor, dice_metric
    torch.cuda.empty_cache()
    return dice, iou, precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_etsam.py")
    parser.add_argument("--csv", help="CSV file containing the test dataset", default="data/testset.csv", type=str)
    parser.add_argument("--collection-dir", help="Collection directory containing the test dataset", default="data/collection", type=str)
    parser.add_argument("--output-dir", help="Output directory to store the results", default="results/etsam_testset_predictions", type=str)
    parser.add_argument("--results", help="CSV file to store the results", default="results/etsam_testset_predictions/results.csv", type=str)
    parser.add_argument("--stage1-prompt", help="Stage 1 prompt method", default="grid_zero", type=str)
    parser.add_argument("--stage2-prompt", help="Stage 2 prompt method", default="zero", type=str)
    parser.add_argument("--stage2-logit-threshold", help="Stage 2 logit threshold", default=-0.25, type=float)

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = os.path.abspath(args.collection_dir)
    results_file = os.path.abspath(args.results)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stage1_ckpt_path = "checkpoints/etsam_stage1_v1.pt"
    stage2_ckpt_path = "checkpoints/etsam_stage2_v1.pt"
    config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    stage1_prompt = args.stage1_prompt # "zero", "grid" or "grid_zero"
    stage2_prompt = args.stage2_prompt # "zero", "grid", "grid_zero" or "etsam_stage1_partial"
    stage1_logit_threshold = 0.0 # threshold for converting predicted logits (likelihood scores) to binary mask
    stage2_logit_threshold = args.stage2_logit_threshold # threshold for converting predicted logits (likelihood scores) to binary mask

    results_df = pd.DataFrame()
    df = pd.read_csv(csv_file, dtype={'dataset_id': str, 'run_id': str})
    for index, row in df.iterrows():
        dataset_id = row['dataset_id']
        run_id = row['run_id']

        results_df.loc[index, "dataset_id"] = dataset_id
        results_df.loc[index, "run_id"] = run_id

        run_dir = os.path.join(collection_dir, dataset_id, run_id)
        input_tomogram_file_path = glob.glob(os.path.join(run_dir, 'tomogram', '*.mrc'))[0]
        gt_mask_file_path = os.path.join(run_dir, 'target_mask', f'{run_id}_target_mask.mrc')

        try:
            print("==> Reading input tomogram:", input_tomogram_file_path)
            with mrcfile.open(input_tomogram_file_path) as mrc:
                voxel_size = mrc.voxel_size
                input_tomogram_shape = mrc.data.shape
                tomogram_data = mrc.data.copy()
            print(f"Input tomogram shape: {input_tomogram_shape}, voxel size: {voxel_size}")

            with mrcfile.open(gt_mask_file_path) as mrc:
                gt_mask_shape = mrc.data.shape
                gt_mask = mrc.data.copy().astype(np.uint8)

            print(f"Target mask shape: {gt_mask_shape}, voxel size: {voxel_size}")

            if input_tomogram_shape != gt_mask_shape:
                print(f"Error: Input tomogram shape {input_tomogram_shape} does not match target mask shape {gt_mask_shape}")
                continue

            # Normalize tomogram data
            tomogram_data = min_max_postive_values_normalize(tomogram_data)
            
            # STAGE 1 Prediction
            print(f"==> Running Stage 1 Prediction with Prompt Method: {stage1_prompt}")
            per_obj_output_mask_full = inference.etsam_inference(tomogram_data, config_path=config, ckpt_path=stage1_ckpt_path, prompt_method=stage1_prompt)
            etsam_stage1_seg_logits = per_obj_output_mask_full[MEMBRANE_OBJECT_ID]
            etsam_stage1_seg_mask = (etsam_stage1_seg_logits > stage1_logit_threshold).astype(np.uint8)

            # Normalize stage 1 logits and add to tomogram data
            etsam_stage1_seg_logits[etsam_stage1_seg_logits < -5] = -5
            etsam_stage1_seg_logits[etsam_stage1_seg_logits > 5] = 5
            etsam_stage1_seg_logits = min_max_normalize(etsam_stage1_seg_logits)
            combined_input_data = tomogram_data + etsam_stage1_seg_logits
            
            # STAGE 2 Prediction
            print(f"==> Running Stage 2 Prediction with Prompt Method: {stage2_prompt}")
            per_obj_output_mask_full = inference.etsam_inference(combined_input_data, etsam_stage1_seg_mask, config_path=config, ckpt_path=stage2_ckpt_path, prompt_method=stage2_prompt)
            etsam_stage2_seg_logits = per_obj_output_mask_full[MEMBRANE_OBJECT_ID]
            etsam_stage2_seg_mask = (etsam_stage2_seg_logits > stage2_logit_threshold).astype(np.uint8)

            stage1_dice, stage1_iou, stage1_precision, stage1_recall = compute_dice_iou_precision_recall(etsam_stage1_seg_mask, gt_mask)
            stage2_dice, stage2_iou, stage2_precision, stage2_recall = compute_dice_iou_precision_recall(etsam_stage2_seg_mask, gt_mask)

            results_df.loc[index, "stage1_dice"] = stage1_dice
            results_df.loc[index, "stage1_iou"] = stage1_iou
            results_df.loc[index, "stage1_precision"] = stage1_precision
            results_df.loc[index, "stage1_recall"] = stage1_recall
            results_df.loc[index, "final_stage2_dice"] = stage2_dice
            results_df.loc[index, "final_stage2_iou"] = stage2_iou
            results_df.loc[index, "final_stage2_precision"] = stage2_precision
            results_df.loc[index, "final_stage2_recall"] = stage2_recall

            print(f"Stage 1 => Dice: {stage1_dice}, IoU: {stage1_iou}, Precision: {stage1_precision}, Recall: {stage1_recall}")
            print(f"Final Stage 2 => Dice: {stage2_dice}, IoU: {stage2_iou}, Precision: {stage2_precision}, Recall: {stage2_recall}")
            results_df.to_csv(results_file, index=False)
            print("-------------------------------------------------------------------------------------------------")
            
            # Clear intermediate variables before saving files
            del stage1_dice, stage1_iou, stage1_precision, stage1_recall
            del stage2_dice, stage2_iou, stage2_precision, stage2_recall

            stage1_output_mask_file_path = os.path.join(output_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage1_seg_mask.mrc")
            stage1_output_logits_file_path = os.path.join(output_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage1_seg_logits.mrc")
            stage2_output_mask_file_path = os.path.join(output_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage2_seg_mask.mrc")
            stage2_output_logits_file_path = os.path.join(output_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage2_seg_logits.mrc")
            os.makedirs(os.path.dirname(stage1_output_mask_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(stage1_output_logits_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(stage2_output_mask_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(stage2_output_logits_file_path), exist_ok=True)
            save_mask(etsam_stage1_seg_mask, voxel_size, stage1_output_mask_file_path)
            save_logits(etsam_stage1_seg_logits, voxel_size, stage1_output_logits_file_path)
            save_mask(etsam_stage2_seg_mask, voxel_size, stage2_output_mask_file_path)
            save_logits(etsam_stage2_seg_logits, voxel_size, stage2_output_logits_file_path)

            # Explicitly delete all intermediate variables and clear CUDA cache
            del tomogram_data, gt_mask, etsam_stage1_seg_logits, etsam_stage1_seg_mask, etsam_stage2_seg_logits, etsam_stage2_seg_mask, combined_input_data, per_obj_output_mask_full
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception as e:
            print("Error: "+str(e))

    print(f"Successfully evaluated ETSAM on the test dataset. Results stored in {results_file}")
