import os
import numpy as np
import pandas as pd
import torch
import mrcfile
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAveragePrecision
import matplotlib.pyplot as plt


THRESHOLDS = 1000  # fixed threshold grid: linspace(0, 1, THRESHOLDS)

testset_real_df = pd.read_csv("data/testset.csv", dtype={"dataset_id": str, "run_id": str})
collection_dir = "data/collection"
etsam_dir = "results/etsam_testset_predictions"
output_dir = "results/etsam_testset_prcurve"


# Constant-memory streaming metrics. `thresholds` switches them from storing all
# predictions to accumulating a (THRESHOLDS, 2, 2) confusion matrix per update.
prc = BinaryPrecisionRecallCurve(thresholds=THRESHOLDS)
ap = BinaryAveragePrecision(thresholds=THRESHOLDS)  # == AUPRC (step integration)

for index, row in testset_real_df.iterrows():
    dataset_id = row["dataset_id"]
    run_id = row["run_id"]
    print(f"==> Processing {dataset_id}:{run_id}")
    exp_ann_file = os.path.join(etsam_dir, dataset_id, run_id, f"{dataset_id}_{run_id}_etsam_stage2_seg_logits.mrc")
    target_ann_file = os.path.join(collection_dir, dataset_id, run_id, "target_mask", f"{run_id}_target_mask.mrc")
    if not os.path.exists(target_ann_file):
        raise ValueError(f"Target map file {target_ann_file} does not exist")
    if not os.path.exists(exp_ann_file):
        raise ValueError(f"Exp tomogram file {exp_ann_file} does not exist")

    exp_ann = mrcfile.open(exp_ann_file).data.copy().astype(np.float32)
    target_ann = mrcfile.open(target_ann_file).data.copy().astype(np.uint8)

    exp_ann[exp_ann < -exp_ann.max()] = -exp_ann.max()

    preds = torch.sigmoid(torch.from_numpy(exp_ann).to("cuda")).ravel()  # float, [0, 1]
    targets = torch.from_numpy(target_ann).to("cuda").ravel().long()   # int {0, 1}

    prc.update(preds, targets)
    ap.update(preds, targets)

    # free this sample before loading the next one
    del exp_ann, target_ann, preds, targets

precision, recall, thresholds = prc.compute()
auprc = ap.compute()

print("ETSAM AUPRC: ", auprc)

os.makedirs(output_dir, exist_ok=True)
# Plot precision-recall curve
plt.figure(figsize=(10, 5))
plt.plot(recall[::1000], precision[::1000], label='ETSAM', color='blue')
plt.xlabel('Recall', fontsize=24, labelpad=20)
plt.ylabel('Precision', fontsize=24, labelpad=20)
plt.title('Precision-Recall (PR) Curve', fontsize=24, pad=20, weight='bold')
plt.text(0.5, 0.5, f'AUPRC: {auprc:.4f}', fontsize=24, ha='center', va='center')
plt.legend(fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.grid(False)  # Disable gridlines
plt.savefig(f"{output_dir}/pr_curve.png")
