import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro

def merge_results(etsam_csv_file, tardis_csv_file, memseg_csv_file, etsam_logit_threshold):
    etsam_results_df = pd.read_csv(etsam_csv_file, dtype={"run_id": str, "dataset_id": str})
    etsam_results_df = etsam_results_df[["dataset_id", "run_id", f"etsam_stage2_dice_{etsam_logit_threshold}", f"etsam_stage2_iou_{etsam_logit_threshold}", f"etsam_stage2_precision_{etsam_logit_threshold}", f"etsam_stage2_recall_{etsam_logit_threshold}"]]
    etsam_results_df.rename(columns={
        f"etsam_stage2_dice_{etsam_logit_threshold}": "ETSAM_Dice",
        f"etsam_stage2_iou_{etsam_logit_threshold}": "ETSAM_IoU",
        f"etsam_stage2_precision_{etsam_logit_threshold}": "ETSAM_Precision",
        f"etsam_stage2_recall_{etsam_logit_threshold}": "ETSAM_Recall"
    }, inplace=True)
    tardis_results_df = pd.read_csv(tardis_csv_file, dtype={"run_id": str, "dataset_id": str})
    tardis_results_df.rename(columns={
        "dice_score": "TARDIS_Dice",
        "mean_iou_score": "TARDIS_IoU",
        "precision": "TARDIS_Precision",
        "recall": "TARDIS_Recall"
    }, inplace=True)
    memseg_results_df = pd.read_csv(memseg_csv_file, dtype={"run_id": str, "dataset_id": str})
    memseg_results_df.rename(columns={
        "dice_score": "Membrain-Seg_Dice",
        "mean_iou_score": "Membrain-Seg_IoU",
        "precision": "Membrain-Seg_Precision",
        "recall": "Membrain-Seg_Recall"
    }, inplace=True)
    # merge on dataset_id and run_id
    df = pd.merge(etsam_results_df, tardis_results_df, on=["dataset_id", "run_id"], how="inner", suffixes=("", "_tardis"))
    df = pd.merge(df, memseg_results_df, on=["dataset_id", "run_id"], how="inner", suffixes=("", "_memseg"))
    return df

def perform_t_test(df):
    methods = ['ETSAM', 'Membrain-Seg', 'TARDIS']
    metrics = ['Dice', 'IoU', 'Precision', 'Recall']

    scores = {
        'ETSAM': {
            'Dice': df["ETSAM_Dice"].values,
            'IoU': df["ETSAM_IoU"].values,
            'Precision': df["ETSAM_Precision"].values,
            'Recall': df["ETSAM_Recall"].values
        },
        'Membrain-Seg': {
            'Dice': df["Membrain-Seg_Dice"].values,
            'IoU': df["Membrain-Seg_IoU"].values,
            'Precision': df["Membrain-Seg_Precision"].values,
            'Recall': df["Membrain-Seg_Recall"].values
        },
        'TARDIS': {
            'Dice': df["TARDIS_Dice"].values,
            'IoU': df["TARDIS_IoU"].values,
            'Precision': df["TARDIS_Precision"].values,
            'Recall': df["TARDIS_Recall"].values
        }
    }

    n_samples = len(df)

    comparisons = [
        ('ETSAM', 'Membrain-Seg'),
        ('ETSAM', 'TARDIS'),
        ('Membrain-Seg', 'TARDIS')
    ]

    # Number of comparisons for Bonferroni correction (3 comparisons per metric)
    n_comparisons = 3
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print("=" * 80)
    print("3-WAY PAIRED T-TEST WITH BONFERRONI CORRECTION FOR ALL METRICS")
    print("=" * 80)
    print(f"\nNumber of samples (paired observations): {n_samples}")
    print(f"Number of methods: {len(methods)}")
    print(f"Metrics analyzed: {', '.join(metrics)}")
    print(f"Original significance level (α): {alpha}")
    print(f"Bonferroni-corrected α (per metric): {bonferroni_alpha:.4f}")
    print(f"Number of pairwise comparisons per metric: {n_comparisons}")

    # ============================================================================
    # PART 1: CHECK ASSUMPTIONS FOR PAIRED T-TEST
    # ============================================================================
    print("\n" + "=" * 80)
    print("PART 1: SHAPIRO-WILK TEST FOR NORMALITY OF DIFFERENCES")
    print("=" * 80)

    print("Null Hypothesis (H0): The differences are normally distributed")
    print("If p > 0.05, we cannot reject the null hypothesis (normality assumption is satisfied)\n")

    normality_results = {}
    for metric in metrics:
        normality_results[metric] = {}
        print(f"\n{metric}:")
        for m1, m2 in comparisons:
            diff = scores[m1][metric] - scores[m2][metric]
            stat, p_val = shapiro(diff)
            normality_met = p_val > 0.05
            normality_results[metric][(m1, m2)] = {'stat': stat, 'p_value': p_val, 'met': normality_met}
            status = "MET" if normality_met else "NOT MET"
            print(f"  {m1} vs {m2}: W = {stat:.4f}, p = {p_val:.4f} -> {status}")

    print("\n" + "=" * 80)
    print("PART 2: PAIRED T-TEST RESULTS WITH BONFERRONI CORRECTION")
    print("=" * 80)

    all_results = {}
    for metric in metrics:
        all_results[metric] = []
        print(f"\n{'='*40}")
        print(f"{metric.upper()}")
        print(f"{'='*40}")

        print("\nStatistics:")
        for method in methods:
            data = scores[method][metric]
            print(f"  {method}: Mean = {np.mean(data):.4f}, SD = {np.std(data, ddof=1):.4f}, "
                f"Min = {np.min(data):.4f}, Max = {np.max(data):.4f}")
        
        print("\nPairwise Comparisons:")
        for m1, m2 in comparisons:
            scores1 = scores[m1][metric]
            scores2 = scores[m2][metric]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            mean_diff = np.mean(scores1) - np.mean(scores2)

            sig_uncorrected = p_value < alpha
            sig_bonferroni = p_value < bonferroni_alpha
            
            result = {
                'comparison': f"{m1} vs {m2}",
                'm1': m1,
                'm2': m2,
                'mean_diff': mean_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'sig_uncorrected': sig_uncorrected,
                'sig_bonferroni': sig_bonferroni,
                'normality_met': normality_results[metric][(m1, m2)]['met']
            }
            all_results[metric].append(result)
            
            print(f"\n ===== {m1} vs {m2} =====\n")
            print(f"--> Mean difference: {mean_diff:.4f}")
            print(f"--> t-statistic: {t_stat:.4f}")
            print(f"--> p-value: {p_value:.6f}")
            print(f"--> Significant (α=0.05): {'Yes' if sig_uncorrected else 'No'}")
            print(f"--> Significant (Bonferroni-corrected α={bonferroni_alpha:.4f}): {'Yes' if sig_bonferroni else 'No'}")
            if not result['normality_met']:
                print(f"--> Warning: Normality assumption not met for this comparison")

if __name__ == "__main__":
    etsam_csv_file = "results/etsam_testset_predictions/results.csv"
    tardis_csv_file = "results/tardis_testset_predictions/results.csv"
    memseg_csv_file = "results/membrain_seg_testset_predictions/results.csv"
    etsam_logit_threshold = -0.25

    df = merge_results(etsam_csv_file, tardis_csv_file, memseg_csv_file, etsam_logit_threshold)
    print(df.head())

    perform_t_test(df)
