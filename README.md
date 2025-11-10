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
## Run ETSAM on a sample cryo-ET tomogram
#### Raw ETSAM membrane prediction without any post-processing:
```
python etsam.py <input_tomogram> --output_dir <output directory to store the predictions>
```
Stores the ETSAM predicted membranes as binary masks in the output directory.

#### Post-processed ETSAM membrane prediction:
```
python etsam.py <input_tomogram> \
    --post-process \
    --output_dir <output directory to store the predictions>
```
Stores both the unprocessed and post-processed binary masks in the output directory.

## Advanced Options
### Custom logit threshold for mask
For each pixel in a given tomogram slice, a logit value is predicted by the final layer of ETSAM's mask decoder. Logits can range from negative to positive values, with larger values indicating a higher likelihood of a cell membrane being present in that pixel. The final ETSAM Stage 2 predicted logits are converted into a binary membrane segmentation mask using a logit threshold, where a logit score > than the logit threshold is considered a membrane in the final predicted binary mask. From our ablation studies, we observed that although a default logit threshold of -0.5 overall performed the best, in certain tomograms, using a different logit threshold yields better results. Therefore, we also provide users with the option to override the default logit threshold (--logit-threshold) for ETSAM prediction, allowing them to achieve the preferred precision-recall trade-off on a per-tomogram basis.

```
python etsam.py <input_tomogram> \
    --logit-threshold -0.25 \
    --output_dir <output directory to store the predictions> 
```

### Store the predicted logit (likelihood) scores
We also provide the users with option to store the ETSAM predicted logit (likelihood) scores along with the binary mask. This allows users to visualize the predicted logit scores and select an optimal logit threshold.

```
python etsam.py <input_tomogram> \
    --store-logits \
    --output_dir <output directory to store the predictions> 
```

### Adjust the Post-process Parameter
We developed a post-processing technique that can significantly reduce the visual artifacts by identifying 3D blobs in the predicted segmentation mask and removing those that do not extend beyond a specific number (default: 10) consecutive slices. Because the size of a noise blob may vary across tomograms of different sizes, we provide users with an option (`--post-process-min-slices`) to adjust the number of slices used to filter the noise blobs as follows:
```
python etsam.py <input_tomogram> \
    --post-process \
    --post-process-min-slices 5 \
    --output_dir <output directory to store the predictions> 
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
