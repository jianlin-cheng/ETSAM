import argparse
import mrcfile
import os
import numpy as np
import inference
from scipy.ndimage import label as cc_label
from scipy.ndimage import find_objects

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

def _full_structure_3d() -> np.ndarray:
    """Return a 3x3x3 structuring element for full (26) connectivity."""
    return np.ones((3, 3, 3), dtype=int)

def identify_3d_blobs_and_remove_thin_blobs(ip_mrc_data: np.ndarray, min_z_span: int = 3) -> np.ndarray:
    """
    Identify 3D connected components (blobs) in a 3D mask and remove those that
    do not span at least `min_z_span` slices along the Z-axis.

    Inputs
    - ip_mrc_data: 3D numpy array (Z, Y, X). Non-zero voxels are treated as foreground.
    - min_z_span: minimum number of Z slices a component must span to be kept (>= 1).
    - connectivity: fixed to full 26-connectivity.

    Output
    - 3D numpy array mask of the same shape and dtype as input, where thin blobs are removed.
    """
    if ip_mrc_data is None:
        raise ValueError("ip_mrc_data is None")
    if ip_mrc_data.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {ip_mrc_data.shape}")
    if min_z_span < 1:
        raise ValueError("min_z_span must be >= 1")

    # Treat any non-zero value as foreground
    mask = ip_mrc_data.astype(bool)
    if not mask.any():
        # Nothing to do
        return np.zeros_like(ip_mrc_data)

    structure = _full_structure_3d()
    labels, num = cc_label(mask, structure=structure)
    if num == 0:
        return np.zeros_like(ip_mrc_data)

    # Get bounding boxes for each labeled component
    slices = find_objects(labels)
    # Build an array indicating which labels to keep based on Z-span
    keep = np.zeros(num + 1, dtype=bool)  # index 0 is background
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            keep[idx] = False
            continue
        z_slice = slc[0]  # assuming array order is (Z, Y, X)
        z_span = (z_slice.stop - z_slice.start) if z_slice is not None else 0
        keep[idx] = z_span >= min_z_span

    filtered_mask = keep[labels]

    # Cast back to input dtype (preserving mask semantics)
    if ip_mrc_data.dtype == np.bool_:
        return filtered_mask
    else:
        return filtered_mask.astype(ip_mrc_data.dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("eval.py")
    parser.add_argument("input_tomogram_file_path", help="input tomogram file path", type=str)
    parser.add_argument("--output-dir", help="Output directory", type=str, default="./")
    parser.add_argument("--logit-threshold", help="threshold for converting preditcted logits (likelihood scores) to binary mask", type=float, default=-0.25)
    parser.add_argument("--store-logits", help="store predicted logits (likelihood scores) in separate file along with binary mask", action="store_true")
    parser.add_argument("--post-process", help="post process the predicted binary mask and store it in a separate file", action="store_true")
    parser.add_argument("--post-process-min-slices", help="minimum number of Z slices a blob (connected component) must span to be kept", type=int, default=10)

    args = parser.parse_args()
    stage1_ckpt_path = "checkpoints/etsam_stage1_v1.pt"
    stage2_ckpt_path = "checkpoints/etsam_stage2_v1.pt"
    config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    stage1_prompt = "grid_zero" # "zero", "grid"
    stage2_prompt = "zero" # "zero", "grid" or "etsam_stage1_partial"
    store_logits = args.store_logits
    stage1_logit_threshold = 0.0
    stage2_logit_threshold = args.logit_threshold
    post_process = args.post_process
    post_process_min_z_span = args.post_process_min_slices
    input_tomogram_file_path = os.path.abspath(args.input_tomogram_file_path)
    output_dir = os.path.abspath(args.output_dir)

    output_mask_file_path = os.path.join(output_dir, f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_mask.mrc")
    output_logits_file_path = os.path.join(output_dir, f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_logits.mrc")
    output_post_processed_mask_file_path = os.path.join(output_dir, f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_post_processed_mask.mrc")

    try:
        print("==> Reading input tomogram:", input_tomogram_file_path)
        with mrcfile.open(input_tomogram_file_path) as mrc:
            voxel_size = mrc.voxel_size
            input_tomogram_shape = mrc.data.shape
            tomogram_data = mrc.data.copy()
        print(f"Input tomogram shape: {input_tomogram_shape}, voxel size: {voxel_size}")

        # Normalize tomogram data
        tomogram_data = min_max_postive_values_normalize(tomogram_data)
        
        # STAGE 1 Prediction
        print(f"==> Running Stage 1 Prediction with Prompt Method: {stage1_prompt}")
        logits = inference.etsam_inference(tomogram_data, config_path=config, ckpt_path=stage1_ckpt_path, prompt_method=stage1_prompt)[MEMBRANE_OBJECT_ID]
        # mask = (logits > stage1_logit_threshold).astype(np.uint8)
        mask = None

        # Normalize stage 1 logits and add to tomogram data
        logits[logits < -5] = -5
        logits[logits > 5] = 5
        logits = min_max_normalize(logits)
        combined_input_data = tomogram_data + logits
        
        # STAGE 2 Prediction
        print(f"==> Running Stage 2 Prediction with Prompt Method: {stage2_prompt}")
        logits = inference.etsam_inference(combined_input_data, mask, config_path=config, ckpt_path=stage2_ckpt_path, prompt_method=stage2_prompt)[MEMBRANE_OBJECT_ID]
        mask = (logits > stage2_logit_threshold).astype(np.uint8)

        print("==> Saving predicted mask")
        os.makedirs(output_dir, exist_ok=True)
        save_mask(mask, voxel_size, output_mask_file_path)
        
        if store_logits:
            print("==> Saving segmentation logits")
            os.makedirs(os.path.dirname(output_logits_file_path), exist_ok=True)
            save_logits(logits, voxel_size, output_logits_file_path)
        
        if post_process:
            print("==> Post-processing the predicted mask")
            post_processed_mask = identify_3d_blobs_and_remove_thin_blobs(mask, min_z_span=post_process_min_z_span)
            print("==> Saving post-processed mask")
            os.makedirs(os.path.dirname(output_post_processed_mask_file_path), exist_ok=True)
            save_mask(post_processed_mask, voxel_size, output_post_processed_mask_file_path)

        print("-------------------------------------------------------------------------------------------------")

    except Exception as e:
        print("Error: "+str(e))
