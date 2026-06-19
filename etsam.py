import argparse
import os

import mrcfile
import numpy as np
import torch

import inference
import postprocess

MEMBRANE_OBJECT_ID = 1


def min_max_normalize(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array


def min_max_postive_values_normalize(array):
    array[array < 0] = 0
    array = array / np.max(array)
    return array


def softplus_minmax_normalize(array):
    array = torch.nn.functional.softplus(torch.from_numpy(array)).cpu().numpy()
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array


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


def split_into_quadrants(tomogram_data):
    """
    Split a (Z, Y, X) tomogram into 4 sub-tomograms by halving Y and X.

    Returns a list of (sub_tomogram, (y_start, y_end, x_start, x_end)) tuples.
    """
    _, H, W = tomogram_data.shape
    y_mid = H // 2
    x_mid = W // 2

    quadrants = [
        (tomogram_data[:, :y_mid, :x_mid], (0, y_mid, 0, x_mid)),
        (tomogram_data[:, :y_mid, x_mid:], (0, y_mid, x_mid, W)),
        (tomogram_data[:, y_mid:, :x_mid], (y_mid, H, 0, x_mid)),
        (tomogram_data[:, y_mid:, x_mid:], (y_mid, H, x_mid, W)),
    ]
    return quadrants


def run_two_stage_inference_on_tomogram(
    tomogram_data,
    config,
    stage1_ckpt_path,
    stage2_ckpt_path,
    stage1_prompt,
    stage2_prompt,
    grid_interval,
    stage1_logit_threshold,
):
    """
    Run the two-stage ETSAM pipeline on a (Z, Y, X) volume (full tomogram or a sub-region).

    Normalizes the volume, runs stage 1, fuses logits with the normalized tomogram, then runs stage 2.

    Returns the stage-2 logits array matching the input spatial shape.
    """
    print(f"==> Stage 1 prediction (prompt: {stage1_prompt})")
    stage1_logits = inference.etsam_inference(
        tomogram_data,
        config_path=config,
        ckpt_path=stage1_ckpt_path,
        prompt_method=stage1_prompt,
        grid_interval=grid_interval,
    )[MEMBRANE_OBJECT_ID]

    stage1_mask = (stage1_logits > stage1_logit_threshold).astype(np.uint8)
    stage1_logits[stage1_logits < -5] = -5
    stage1_logits[stage1_logits > 5] = 5
    stage1_logits = min_max_normalize(stage1_logits)
    combined_input = tomogram_data + stage1_logits

    print(f"==> Stage 2 prediction (prompt: {stage2_prompt})")
    stage2_logits = inference.etsam_inference(
        combined_input,
        stage1_mask,
        config_path=config,
        ckpt_path=stage2_ckpt_path,
        prompt_method=stage2_prompt,
        grid_interval=grid_interval,
    )[MEMBRANE_OBJECT_ID]

    return stage2_logits


def combine_quadrant_logits(quadrant_results, full_shape):
    """
    Assemble per-quadrant logit arrays into a single (Z, Y, X) volume.

    quadrant_results: list of (logits, (y_start, y_end, x_start, x_end))
    full_shape: (Z, Y, X) of the original tomogram
    """
    combined = np.zeros(full_shape, dtype=np.float32)
    for logits, (y_start, y_end, x_start, x_end) in quadrant_results:
        combined[:, y_start:y_end, x_start:x_end] = logits
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser("etsam.py")
    parser.add_argument(
        "input_tomogram_file_path", help="input tomogram file path", type=str
    )
    parser.add_argument("--output-dir", help="Output directory", type=str, default="./")
    parser.add_argument(
        "--logit-threshold",
        help="threshold for converting preditcted logits (likelihood scores) to binary mask",
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--store-logits",
        help="store predicted logits (likelihood scores) in separate file along with binary mask",
        action="store_true",
    )
    postprocess.add_postprocess_arguments(parser)
    parser.add_argument(
        "--split-processing",
        help="split the tomogram into four smaller tomograms (Y/X), run etsam on them independently, then merge the results; use for complex tomograms with closely apposed cell membranes",
        action="store_true",
    )
    parser.add_argument(
        "--stage1-prompt",
        help="prompt for stage 1 (grid or zero)",
        type=str,
        default="grid_zero",
        choices=["zero", "grid", "grid_zero"],
    )
    parser.add_argument(
        "--stage2-prompt",
        help="prompt for stage 2 (zero, grid, grid_zero or etsam_stage1_partial)",
        type=str,
        default="zero",
        choices=["zero", "grid", "grid_zero", "etsam_stage1_partial"],
    )

    parser.add_argument(
        "--normalize-method",
        help="method for normalizing the tomogram data",
        type=str,
        default="min_max_positive_values",
        choices=["min_max_positive_values", "softplus_minmax"],
    )

    args = parser.parse_args()
    stage1_ckpt_path = "checkpoints/etsam_stage1_v1.pt"
    stage2_ckpt_path = "checkpoints/etsam_stage2_v1.pt"
    config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    stage1_prompt = args.stage1_prompt
    stage2_prompt = args.stage2_prompt
    store_logits = args.store_logits
    stage1_logit_threshold = -0.25
    stage2_logit_threshold = args.logit_threshold
    split_processing = args.split_processing
    grid_interval = 10 if split_processing else 50
    input_tomogram_file_path = os.path.abspath(args.input_tomogram_file_path)
    output_dir = os.path.abspath(args.output_dir)
    normalize_method = args.normalize_method

    try:
        print("==> Reading input tomogram:", input_tomogram_file_path)
        with mrcfile.open(input_tomogram_file_path) as mrc:
            voxel_size = mrc.voxel_size
            input_tomogram_shape = mrc.data.shape
            tomogram_data = mrc.data.astype(np.float32)
        print(f"Input tomogram shape: {input_tomogram_shape}, voxel size: {voxel_size}")

        print(f"==> Normalizing tomogram data with {normalize_method} method")
        if normalize_method == "min_max_positive_values":
            normalized_tomogram_data = min_max_postive_values_normalize(tomogram_data)
        elif normalize_method == "softplus_minmax":
            normalized_tomogram_data = softplus_minmax_normalize(tomogram_data)

        if split_processing:
            output_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_mask.mrc",
            )
            output_logits_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_logits.mrc",
            )
            output_post_processed_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_post_processed_mask.mrc",
            )

            quadrants = split_into_quadrants(normalized_tomogram_data)
            quadrant_labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
            print(
                f"==> Split tomogram into {len(quadrants)} sub-tomograms (split-processing mode, grid_interval={grid_interval})"
            )
            for label, (sub, (y0, y1, x0, x1)) in zip(quadrant_labels, quadrants):
                print(f"    {label}: shape {sub.shape}, Y[{y0}:{y1}] X[{x0}:{x1}]")

            quadrant_results = []
            for label, (sub_tomogram, coords) in zip(quadrant_labels, quadrants):
                print(f"\n==> Processing quadrant: {label}")
                q_logits = run_two_stage_inference_on_tomogram(
                    sub_tomogram,
                    config,
                    stage1_ckpt_path,
                    stage2_ckpt_path,
                    stage1_prompt,
                    stage2_prompt,
                    grid_interval=grid_interval,
                    stage1_logit_threshold=stage1_logit_threshold,
                )
                quadrant_results.append((q_logits, coords))

            print("\n==> Combining quadrant logits into full volume")
            logits = combine_quadrant_logits(quadrant_results, input_tomogram_shape)
        else:
            output_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_mask.mrc",
            )
            output_logits_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_logits.mrc",
            )
            output_post_processed_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_post_processed_mask.mrc",
            )

            logits = run_two_stage_inference_on_tomogram(
                normalized_tomogram_data,
                config,
                stage1_ckpt_path,
                stage2_ckpt_path,
                stage1_prompt,
                stage2_prompt,
                grid_interval=grid_interval,
                stage1_logit_threshold=stage1_logit_threshold,
            )

        mask = (logits > stage2_logit_threshold).astype(np.uint8)

        print("==> Saving predicted mask")
        os.makedirs(output_dir, exist_ok=True)
        save_mask(mask, voxel_size, output_mask_file_path)

        if store_logits:
            print("==> Saving segmentation logits")
            os.makedirs(os.path.dirname(output_logits_file_path), exist_ok=True)
            save_logits(logits, voxel_size, output_logits_file_path)

        if postprocess.postprocess_requested(args):
            post_processed_mask = postprocess.postprocess_cli(mask, args)
            print("==> Saving post-processed mask")
            os.makedirs(
                os.path.dirname(output_post_processed_mask_file_path), exist_ok=True
            )
            save_mask(
                post_processed_mask, voxel_size, output_post_processed_mask_file_path
            )

        print(
            "-------------------------------------------------------------------------------------------------"
        )

    except Exception as e:
        print("Error: " + str(e))
