# ETSAM: Effectively Segmenting Membranes in cryo-Electron Tomograms
Cryogenic Electron Tomography (cryo-ET) is a core experimental technique used to visualizethe cell structures and macromolecules in their native cellular environment. Accuratesegmentation of cell structures, such as cell membranes, is crucial for advancing ourunderstanding of cellular organization and function. However, several inherent limitations incryo-ET, including the very low signal-to-noise ratio of reconstructed 3D tomograms, missingwedge artifacts from limited tilt angles, and other noise artifacts, collectively hinder the reliableidentification and delineation of these structures. In this study, we introduce ETSAM - a two-stage, SAM2-based fine-tuned model that effectively segments cell membranes in cryo-ETtomograms. It is trained on a diverse dataset comprising 83 experimental tomograms fromthe CryoET Data Portal (CDP) database and 28 simulated tomograms generated using thePolNet tool. ETSAM achieves state-of-the-art performance compared to other deep learningmethods when evaluated on an independent test set comprising 10 simulated tomogramsand 15 experimental tomograms from the CDP database, for which ground-truth annotationsare available. ETSAM robustly segments cell membranes with high sensitivity to membraneregions with less noise in the predictions, thereby achieving a better precision-recall trade-offthan other deep learning methods.

# ETSAM Pipeline 
![ETSAM Pipeline](<assets/etsam-pipeline.png>)

(A) shows the pipeline of the two-stage ETSAM. (B) shows the network architecture of the ETSAM block used in both stage 1 and stage 2.

# Installation

### Clone the project
```
git clone https://github.com/jianlin-cheng/ETSAM
cd ETSAM
```

### Download ETSAM Stage 1 and Stage 2 weights
```
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage1_v1.pt
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage2_v1.pt
```

### Setup Conda Environment
```
conda env create -f environment.yml
conda activate etsam
```

# Usage
## Run ETSAM on a sample cryo-ET tomogram:
```
python etsam.py <input_tomogram> --output-dir <output directory to store the predictions>
```
Stores the ETSAM predicted membranes as binary masks in the output directory.

**Example:**

Download the experimental tomogram of prokaryotic cell of Hylemonella gracilis (CDP Dataset: 10160, Run: 8354):
```
wget -P data/ https://files.cryoetdataportal.cziscience.com/10160/ycw2013-01-03-15/Reconstructions/VoxelSpacing16.145/Tomograms/100/ycw2013-01-03-15.mrc
```
Run ETSAM on the experimental tomogram to segment the membranes and store the predicted output in the `results/cdp_run_8354/` directory:
```
user@host$ python etsam.py data/ycw2013-01-03-15.mrc --output-dir results/cdp_run_8354/
==> Reading input tomogram: /run/media/joel/My_Passport/ETSAM/data/ycw2013-01-03-15.mrc
Input tomogram shape: (500, 924, 956), voxel size: (16.145, 16.145, 16.145)
==> Running Stage 1 Prediction with Prompt Method: grid_zero
==> Loading ETSAM with checkpoint:  checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:25<00:00, 19.83it/s]
==> Running Stage 2 Prediction with Prompt Method: etsam_stage1_partial
==> Loading ETSAM with checkpoint:  checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using ETSAM stage 1 predicted mask at every 50th slice as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:31<00:00, 15.66it/s]
==> Saving predicted mask
-------------------------------------------------------------------------------------------------
```
Visualizing the tomogram and predicted output stored `results/cdp_run_8354/ycw2013-01-03-15_etsam_predicted_mask.mrc`:

![Example-1](<assets/github-example-1.png>)

## Run ETSAM membrane prediction with post-processing:
```
python etsam.py <input_tomogram> \
    --post-process \
    --output-dir <output directory to store the predictions>
```
Stores both the unprocessed and post-processed binary masks in the output directory.

**Example:**

Download the experimental tomogram of vegetative cell of Schizosaccharomyces pombe 972h- (CDP Dataset: 10000 - Run 247):
```
wget -P data/ https://files.cryoetdataportal.cziscience.com/10000/TS_037/Reconstructions/VoxelSpacing13.480/Tomograms/100/TS_037.mrc
```
Run ETSAM on the experimental tomogram to segment the membranes, post-process the predictions and store both the unprocessed and post-processed output in the `results/cdp_run_247/` directory:
```
user@host$ python etsam.py data/TS_037.mrc --post-process --output-dir results/cdp_run_247/
==> Reading input tomogram: /run/media/joel/My_Passport/ETSAM/data/TS_037.mrc
Input tomogram shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Running Stage 1 Prediction with Prompt Method: grid_zero
==> Loading ETSAM with checkpoint:  checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:24<00:00, 20.19it/s]
==> Running Stage 2 Prediction with Prompt Method: etsam_stage1_partial
==> Loading ETSAM with checkpoint:  checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using ETSAM stage 1 predicted mask at every 50th slice as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:31<00:00, 15.70it/s]
==> Saving predicted mask
==> Post-processing the predicted mask
==> Saving post-processed mask
-------------------------------------------------------------------------------------------------
```

Visualizing the tomogram, unprocessed (`TS_037_etsam_predicted_mask.mrc`) and post-processed (`TS_037_etsam_predicted_post_processed_mask.mrc`) predicted segmentation stored in `results/cdp_run_247/`:

![Example-1](<assets/github-example-2.png>)

## Advanced Options
### Custom logit threshold for mask
For each pixel in a given tomogram slice, a logit value is predicted by the final layer of ETSAM's mask decoder. Logits can range from negative to positive values, with larger values indicating a higher likelihood of a cell membrane being present in that pixel. The final ETSAM Stage 2 predicted logits are converted into a binary membrane segmentation mask using a logit threshold, where a logit score > than the logit threshold is considered a membrane in the final predicted binary mask. From our ablation studies, we observed that although a default logit threshold of -0.5 overall performed the best, in certain tomograms, using a different logit threshold yields better results. Therefore, we also provide users with the option to override the default logit threshold (--logit-threshold) for ETSAM prediction, allowing them to achieve the preferred precision-recall trade-off on a per-tomogram basis.

```
python etsam.py <input_tomogram> \
    --logit-threshold -0.25 \
    --output-dir <output directory to store the predictions> 
```

### Store the predicted logit (likelihood) scores
We also provide the users with option to store the ETSAM predicted logit (likelihood) scores along with the binary mask. This allows users to visualize the predicted logit scores and select an optimal logit threshold.

```
python etsam.py <input_tomogram> \
    --store-logits \
    --output-dir <output directory to store the predictions> 
```

### Adjust the Post-process Parameter
We developed a post-processing technique that can significantly reduce the visual artifacts by identifying 3D blobs in the predicted segmentation mask and removing those that do not extend beyond a specific number (default: 10) consecutive slices. Because the size of a noise blob may vary across tomograms of different sizes, we provide users with an option (`--post-process-min-slices`) to adjust the number of slices used to filter the noise blobs as follows:
```
python etsam.py <input_tomogram> \
    --post-process \
    --post-process-min-slices 5 \
    --output-dir <output directory to store the predictions> 
```

Reducing the number of slices will remove only smaller noise blobs and increasing it will remove larger noise blobs. 

While we recommend using the post-processing technique to improve the visual clarity of ETSAM membrane segmentation, caution must be exercised when dealing with thin and small membrane regions in tomogram, as they may be removed during post-processing. In this case, we recommend visually comparing unprocessed and post-processed segmentation results to ensure biologically relevant membrane regions are not removed.

# Training ETSAM

Information on how to setup dataset and train ETSAM can be found in [training/README.md](training/README.md).

# Evaluating ETSAM on the test dataset
Download the test dataset
```
python scripts/collect_data.py --csv data/testset.csv --collection-dir data/collection
```
Run the script to evaluate ETSAM on the test dataset and store the results.
```
python scripts/evaluate_etsam.py \
    --results etsam_testset_results.csv \
    --csv data/testset.csv \
    --collection-dir data/collection
```
