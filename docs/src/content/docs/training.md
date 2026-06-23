---
title: Training
description: Fine-tune or train ETSAM Stage 1 and Stage 2 using the provided SAM2-based training pipeline.
prev:
  label: Advanced Usage
  link: /ETSAM/advanced/
next:
  label: Evaluation
  link: /ETSAM/evaluation/
---

ETSAM's training code is based on the SAM2 training framework with a modified data loading pipeline. Tomogram slices and annotations are loaded as floating-point `.npy` arrays rather than standard RGB images.

Training is a sequential two-stage process:

1. Stage 1 is trained on raw tomogram slices and learns to predict membrane locations from grid-point prompts at first slice.
2. Stage 2 is trained on fused inputs combining raw tomogram slices with Stage 1 predictions, learning to refine the initial output.

:::note
Both stages start from the SAM2.1 Hiera Base Plus checkpoint, `sam2.1_hiera_base_plus.pt`, as initial weights.
:::

## Training Requirements

- A100 GPUs with 80 GB VRAM are recommended.
- The examples below assume a 4-GPU training setup.
- Install the development dependencies before training:

Follow installation instruction for the initial conda environment setup.

```bash
conda activate etsam
pip install -e ".[dev]"
```

## Setup: Data and Checkpoints

### 1. Download the Training Dataset

Collect the ETSAM training tomograms from the CryoET Data Portal and PolNet simulations:

```bash
python scripts/collect_data.py \
    --csv data/dataset.csv \
    --output-dir data/collection
```

### 2. Fetch the SAM2 Base Checkpoint

```bash
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

## Training Stage 1

### 1. Create the Stage 1 Dataset

This script normalizes the collected tomograms and annotations, then splits them into individual Z-slices stored as `.npy` arrays.

```bash
python scripts/create_etsam_stage1_dataset.py \
    --csv data/dataset.csv \
    --collection-dir data/collection \
    --output-dir data/etsam_stage1_dataset
```

Output structure:

```text
data/etsam_stage1_dataset/
|-- train/
    |-- inputs/    # Normalized tomogram slices
    |-- labels/    # Ground-truth membrane annotations
```

### 2. Configure Paths

If you placed the dataset in a non-default location, update the paths in `sam2/configs/etsam_training/etsam_stage1.yaml`:

```yaml
dataset:
  train_img_folder: ./data/etsam_stage1_dataset/train/inputs
  train_gt_folder: ./data/etsam_stage1_dataset/train/labels

experiment_log_dir: ./etsam_logs/etsam_stage1_training
```

### 3. Run Stage 1 Training

```bash
python training/train.py \
    -c configs/etsam_training/etsam_stage1.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

Training artifacts are saved to `etsam_logs/etsam_stage1_training/`:

- `checkpoints/` stores model checkpoints saved during training. Use `last.ckpt` for Stage 2 dataset creation.
- `tensorboard/` stores TensorBoard logs for monitoring the training loss.

### 4. Monitor Training

```bash
tensorboard --logdir etsam_logs/etsam_stage1_training/tensorboard/
```

## Training Stage 2

### 1. Create the Stage 2 Dataset

Stage 2 requires a fused dataset where each input slice combines the normalized tomogram and the Stage 1 prediction.

```bash
python scripts/create_etsam_stage2_dataset.py \
    --etsam-stage1-checkpoint ./etsam_logs/etsam_stage1_training/checkpoints/last.ckpt \
    --csv data/dataset.csv \
    --collection-dir data/collection \
    --output-dir data/etsam_stage2_dataset
```

Output structure:

```text
data/etsam_stage2_dataset/
|-- train/
    |-- inputs/    # Fused tomogram + Stage 1 prediction slices
    |-- labels/    # Ground-truth membrane annotations
```

### 2. Configure Paths

Update `sam2/configs/etsam_training/etsam_stage2.yaml` if needed:

```yaml
dataset:
  train_img_folder: ./data/etsam_stage2_dataset/train/inputs
  train_gt_folder: ./data/etsam_stage2_dataset/train/labels

experiment_log_dir: ./etsam_logs/etsam_stage2_training
```

### 3. Run Stage 2 Training

```bash
python training/train.py \
    -c configs/etsam_training/etsam_stage2.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

Trained Stage 2 checkpoints are saved under `etsam_logs/etsam_stage2_training/checkpoints/`.

## Using Your Own Checkpoint

After training, run inference with custom checkpoints by pointing ETSAM to the new stage 1 and stage 2 weight files. The checkpoints follow the PyTorch Lightning format.
