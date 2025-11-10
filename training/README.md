# Training ETSAM
This folder contains the training code for ETSAM (which is largely based on SAM 2). The data loading pipeline of sam2 is modified to load tomogram and annotation slices as floating point numpy files (.npy) instead of image files (.jpg/.png).

## Getting Started
#### Requirements:
- We assume training on A100 GPUs with **80 GB** of memory.
- Install the packages required for training by running `pip install -e ".[dev]"` in the repo directory.

#### Download the data used to train ETSAM
```
python scripts/collect_data.py --csv data/dataset.csv --output-dir data/collection
```

#### Fetch SAM2 base checkpoints
Download the original SAM2.1 base checkpoint. We will use them as the initial weights for both ETSAM Stage 1 and 2. 
```
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

## Steps to fine-tune/train ETSAM Stage 1:
- Create the ETSAM Stage 1 dataset - Normalizes and store the tomograms and annotations as slices in numpy .npy format
    ```
    python scripts/create_etsam_stage1_dataset.py \
        --csv data/dataset.csv \
        --collection-dir data/collection \
        --output-dir data/etsam_stage1_dataset
    ```

- If needed, adjust the paths for ETSAM Stage 1 dataset and experiment log directory in `sam2/configs/etsam_training/etsam_stage1.yaml`.
    ```yaml
    dataset:
        # PATHS to Dataset
        train_img_folder: ./data/etsam_stage1_dataset/train/inputs
        train_gt_folder: ./data/etsam_stage1_dataset/train/labels
    ...
    ...
    experiment_log_dir: ./etsam_logs/etsam_stage1_training # Path to log directory, defaults to ./sam2_logs/${config_name}
    ```

- To fine-tune ETSAM Stage 1 model using 4 GPUs, run 

    ```python
    python training/train.py \
        -c configs/etsam_training/etsam_stage1.yaml \
        --use-cluster 0 \
        --num-gpus 4
    ```

    The training losses can be monitored using `tensorboard` logs stored under `tensorboard/` in the experiment log directory.
    After training/fine-tuning, you can then use the new checkpoint (saved in `checkpoints/` in the experiment log directory).

## Steps to fine-tune/train ETSAM Stage 2
- Create the ETSAM Stage 2 dataset - Runs ETSAM Stage 1 on training data and fuses the prediction with normalized tomogram and stores the them as slices in numpy .npy format
    ```
    python scripts/create_etsam_stage2_dataset.py \
        --etsam-stage1-checkpoint ./etsam_logs/etsam_stage1_training/checkpoints/last.ckpt \
        --csv data/dataset.csv \
        --collection-dir data/collection \
        --output-dir data/etsam_stage2_dataset
    ```

- If needed, adjust the paths for ETSAM Stage 2 dataset and experiment log directory in `sam2/configs/etsam_training/etsam_stage2.yaml`.
    ```yaml
    dataset:
        # PATHS to Dataset
        train_img_folder: ./data/etsam_stage2_dataset/train/inputs
        train_gt_folder: ./data/etsam_stage2_dataset/train/labels
    ...
    ...
    experiment_log_dir: ./etsam_logs/etsam_stage2_training # Path to log directory, defaults to ./sam2_logs/${config_name}
    ```

- To fine-tune ETSAM Stage 2 model using 4 GPUs, run 

    ```python
    python training/train.py \
        -c configs/etsam_training/etsam_stage2.yaml \
        --use-cluster 0 \
        --num-gpus 4
    ```

    The training losses can be monitored using `tensorboard` logs stored under `tensorboard/` in the experiment log directory.
    After training/fine-tuning, you can then use the new checkpoint (saved in `checkpoints/` in the experiment log directory).
