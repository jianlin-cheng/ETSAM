---
title: Evaluation
description: Reproduce ETSAM benchmark results and understand the reported segmentation metrics.
prev:
  label: Training
  link: /ETSAM/training/
next:
  label: Citation
  link: /ETSAM/citation/
---

The ETSAM test set contains 17 experimental tomograms with ground-truth membrane annotations. The steps below reproduce the evaluation reported in the paper.

## Evaluate on the Test Dataset

### 1. Download the Test Dataset

Use the data collection script with the test set manifest:

```bash
python scripts/collect_data.py \
    --csv data/testset.csv \
    --collection-dir data/collection
```

### 2. Run ETSAM Evaluation

This runs ETSAM inference on all test tomograms and computes Dice, IoU, Precision, and Recall against the ground-truth annotations.

```bash
python evaluation/evaluate_etsam.py
```

Results are saved to `results/etsam_testset_predictions/3d_metrics.csv`.

### 3. Plane-Wise Evaluation

This evaluates ETSAM prediction consistency by averaging per-slice Dice, IoU, Precision, and Recall separately across the XY, XZ, and YZ planes.

```bash
python evaluation/evaluate_etsam_plane_wise.py
```

Results are saved to `results/etsam_testset_predictions/etsam_testset_plane_wise_metrics.csv`.

### 4. Theta-Wise Evaluation

This evaluates ETSAM prediction consistency across membrane orientations by binning membrane voxels by surface-normal angle and computing Dice, IoU, Precision, and Recall for each theta bin.

```bash
python evaluation/evaluate_etsam_theta_wise.py
```

Results are saved to `results/etsam_testset_predictions/etsam_testset_theta_wise_metrics.csv`.

### 5. Statistical Significance

Run paired t-tests comparing ETSAM against TARDIS and MemBrain-Seg:

```bash
python evaluation/t-test.py
```

:::note
TARDIS and MemBrain-Seg were evaluated previously, and their results are stored in the `results/` directory for comparison.
:::

### 6. Precision-Recall Curves and AUPRC

Compute the Area Under the Precision-Recall Curve (AUPRC) and generate PR curve plots:

```bash
python evaluation/pr_curve_auprc.py
```

## Evaluation Metrics

ETSAM is benchmarked with standard segmentation metrics computed against ground-truth binary membrane annotations.

| Metric | Interpretation |
| --- | --- |
| Dice | Harmonic mean of precision and recall; overall segmentation quality. |
| IoU | Measures how much the predicted membrane region overlaps the ground truth relative to their combined area |
| Precision | Fraction of predicted membrane pixels that are correct. |
| Recall | Fraction of true membrane pixels that are detected. |

The AUPRC, or Area Under the Precision-Recall Curve, provides a threshold-independent assessment of model discrimination.

## Baseline Comparison

ETSAM was compared against two leading cryo-ET membrane segmentation methods TARDIS and MemBrain-Seg.

ETSAM achieves a more favorable precision-recall trade-off than both baselines on the independent test set, demonstrating robust membrane detection across simulated and experimental tomograms.

:::tip
For full quantitative results, precision-recall curves, and statistical significance tests, see the [ETSAM preprint on bioRxiv](https://doi.org/10.1101/2025.11.23.689996).
:::
