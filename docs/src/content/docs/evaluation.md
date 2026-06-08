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

The ETSAM test set contains 20 tomograms: 10 simulated PolNet tomograms and 10 experimental CryoET Data Portal tomograms with ground-truth membrane annotations. The steps below reproduce the benchmark metrics reported in the paper.

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

Results are saved to `results/etsam_testset_predictions/results.csv`.

### 3. Statistical Significance

Run paired t-tests comparing ETSAM against TARDIS and MemBrain-Seg:

```bash
python evaluation/t-test.py
```

:::note
TARDIS and MemBrain-Seg were evaluated previously, and their results are stored in the `results/` directory for comparison.
:::

### 4. Precision-Recall Curves and AUPRC

Compute the Area Under the Precision-Recall Curve (AUPRC) and generate PR curve plots:

```bash
python evaluation/pr_curve_auprc.py
```

## Evaluation Metrics

ETSAM is benchmarked with standard segmentation metrics computed against ground-truth binary membrane annotations.

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Dice | `2 * TP / (2 * TP + FP + FN)` | Harmonic mean of precision and recall; overall segmentation quality. |
| IoU | `TP / (TP + FP + FN)` | Intersection over union; stricter than Dice. |
| Precision | `TP / (TP + FP)` | Fraction of predicted membrane pixels that are correct. |
| Recall | `TP / (TP + FN)` | Fraction of true membrane pixels that are detected. |

The AUPRC, or Area Under the Precision-Recall Curve, provides a threshold-independent assessment of model discrimination. It is especially useful for membrane segmentation because membrane pixels are sparse relative to background.

## Baseline Comparison

ETSAM was compared against two leading cryo-ET membrane segmentation methods:

- TARDIS: a distance-based instance segmentation approach designed for cryo-ET.
- MemBrain-Seg: a 3D U-Net based membrane segmentation model for cryo-ET.

ETSAM achieves a more favorable precision-recall trade-off than both baselines on the independent test set, demonstrating robust membrane detection across simulated and experimental tomograms.

:::tip
For full quantitative results, precision-recall curves, and statistical significance tests, see the [ETSAM preprint on bioRxiv](https://doi.org/10.1101/2025.11.23.689996).
:::
