import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

myDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.split(myDir)[0]
if parentDir not in sys.path:
    sys.path.append(parentDir)

from modules.glm import GeneralLinearModel
from modules.load import load_synonymous_muts

# ----------------------------
# DEFAULT DATASETS
# ----------------------------
datasets_default = {
    "DENV1 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv1_E/curated/curated_mut_counts.csv",
    "DENV2 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv2_E/curated/curated_mut_counts.csv",
    "DENV3 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv3_E/curated/curated_mut_counts.csv",
    "DENV4 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv4_E/curated/curated_mut_counts.csv",
    "DENV1 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv1_genome/curated/curated_mut_counts.csv",
    "DENV2 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv2_genome/curated/curated_mut_counts.csv",
    "DENV3 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv3_genome/curated/curated_mut_counts.csv",
    "DENV4 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv4_genome/curated/curated_mut_counts.csv",
    "RSV A": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_rsv_a_241125/curated/curated_mut_counts.csv",
    "RSV B": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_rsv_b_251125/curated/curated_mut_counts.csv",
    "HIV-1 pol": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_hiv_pol_141125/curated/curated_mut_counts.csv",
}

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

# ----------------------------
# FUNCTIONS
# ----------------------------
def compute_r2_local_context(df):
    model = GeneralLinearModel(included_factors=['local_context'])
    model.train(df_train=df)
    mean_sq_err = model.test(df_test=df)

    base_model = GeneralLinearModel(included_factors=[])
    base_model.train(df_train=df)
    base_err = base_model.test(df_test=df)

    return {mut: 1 - mean_sq_err[mut] / base_err[mut] for mut in mut_types if mut in mean_sq_err}

# ----------------------------
# ARGPARSE
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Plot R² heatmap of local context for selected viruses")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(datasets_default.keys()),
        help="Subset of dataset names to include (default: all)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="R2_local_context_heatmap.pdf",
        help="Output path for heatmap PDF"
    )
    return parser.parse_args()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    selected_datasets = {k: datasets_default[k] for k in args.datasets if k in datasets_default}

    r2_all = {}
    for virus, path in selected_datasets.items():
        if not os.path.exists(path):
            print(f"Skipping {virus}, file not found: {path}")
            continue
        df = load_synonymous_muts(path)
        r2_all[virus] = compute_r2_local_context(df)

    if not r2_all:
        print("No valid datasets to plot.")
        sys.exit(1)

    viruses = sorted(r2_all.keys())
    r2_matrix = np.array([[r2_all[v].get(mt, np.nan) for mt in mut_types] for v in viruses])

    plt.figure(figsize=(12, len(viruses)+2))
    sns.heatmap(
        r2_matrix,
        xticklabels=mut_types,
        yticklabels=viruses,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.title("Relative R² by mutation type")
    plt.xlabel("Mutation Type")
    plt.ylabel("Virus")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.show()