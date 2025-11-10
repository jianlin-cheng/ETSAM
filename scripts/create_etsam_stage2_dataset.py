import pandas as pd
import numpy as np
import torch
import cryoet_utils as cu
import shutil
import os
import argparse
import glob
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

def etsam_stage2_create_dataset(images_output_dir, input_array, labels_output_dir, target_array):
    print(f"==> Slicing and saving the fused input data and annotation data")
    input_array = input_array.transpose(2, 1, 0)
    input_array = np.expand_dims(input_array, axis=-1)
    input_array = np.repeat(input_array, 3, axis=-1)  # Repeat the last dimension 3 times
    target_array = target_array.transpose(2, 1, 0)
    target_array = np.expand_dims(target_array, axis=-1)

    for i in range(input_array.shape[0]):
        input_array_slice = input_array[i, :, :, :]
        target_array_slice = target_array[i, :, :, :]

        input_array_slice = input_array_slice.squeeze()
        input_file_path = os.path.join(images_output_dir, f"{i:05d}.npy")
        np.save(input_file_path, input_array_slice)

        target_array_slice = target_array_slice.squeeze()
        target_file_path = os.path.join(labels_output_dir, f"{i:05d}.npy")
        np.save(target_file_path, target_array_slice)

    del input_array, target_array

def create_dataset(row_data, collection_dir, dataset_dir, etsam_stage1_checkpoint):
    dataset_id = row_data['dataset_id']
    run_id = row_data['run_id']
    split = row_data['split']

    run_dir = os.path.join(collection_dir, dataset_id, run_id)

    status_output_file = os.path.join(dataset_dir, '.status', str(run_id)+'.completed')
    if os.path.exists(status_output_file):
        print(f"Skipping {run_id} because it already processed")
        return

    try:
        tomogram_file = glob.glob(os.path.join(run_dir, 'tomogram', '*.mrc'))[0]
        tomo = cu.load_tomogram(tomogram_file)

        tomo.data = min_max_postive_values_normalize(tomo.data)

        print(f"==> Running Stage 1 Prediction")
        etsam_stage1_seg_logits = inference.etsam_inference(tomo.data, ckpt_path=etsam_stage1_checkpoint, prompt_method="grid_zero")[MEMBRANE_OBJECT_ID]
        etsam_stage1_seg_logits[etsam_stage1_seg_logits < -5] = -5
        etsam_stage1_seg_logits[etsam_stage1_seg_logits > 5] = 5
        etsam_stage1_seg_logits = min_max_normalize(etsam_stage1_seg_logits)
        combined_input_data = tomo.data + etsam_stage1_seg_logits

        final_target_mask_file = os.path.join(run_dir, 'target_mask', f"{run_id}_target_mask.mrc")
        final_target_mask = cu.load_tomogram(final_target_mask_file)

        images_output_dir = os.path.join(dataset_dir, split, 'inputs', str(run_id))
        labels_output_dir = os.path.join(dataset_dir, split, 'labels', str(run_id), '1') # 1 is the membrane object id
        shutil.rmtree(images_output_dir, ignore_errors=True)
        shutil.rmtree(os.path.dirname(labels_output_dir), ignore_errors=True)
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)
        etsam_stage2_create_dataset(images_output_dir, combined_input_data, labels_output_dir, final_target_mask.data)

        os.makedirs(os.path.dirname(status_output_file), exist_ok=True)
        with open(status_output_file, 'w') as f:
            f.write('completed')

        print("Done")
        del tomo, combined_input_data, final_target_mask, etsam_stage1_seg_logits
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(status_output_file):
            os.remove(status_output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for ETSAM Stage 2')
    parser.add_argument('--dataset-csv', type=str, default='data/dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--collection-dir', type=str, default='data/collection',
                        help='Path to the collection directory')
    parser.add_argument('--dataset-dir', type=str, default='data/etsam_stage2_dataset',
                        help='Output directory for the dataset')
    parser.add_argument('--etsam-stage1-checkpoint', type=str, default='checkpoints/etsam_stage1_v1.pt',
                        help='Path to the ETSAM stage 1 checkpoint')

    args = parser.parse_args()

    collection_dir = os.path.abspath(args.collection_dir)
    dataset_dir = os.path.abspath(args.dataset_dir)
    etsam_stage1_checkpoint = os.path.abspath(args.etsam_stage1_checkpoint)

    df = pd.read_csv(os.path.abspath(args.dataset_csv), dtype={'dataset_id': str, 'run_id': str})
    df = df[df["split"] == "train"].reset_index(drop=True)

    for index, row in df.iterrows():
        print(f"==> {index + 1} of {len(df)} - Creating dataset for Dataset: {row['dataset_id']} Run: {row['run_id']}")
        create_dataset(row, collection_dir, dataset_dir, etsam_stage1_checkpoint)
