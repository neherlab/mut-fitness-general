import os, sys
myDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.split(myDir)[0]
if not (sys.path.__contains__(parentDir)):
    sys.path.append(parentDir)
from modules.glm import GeneralLinearModel
from modules.load import load_synonymous_muts
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

datasets_default = {
    # Dengue E gene
    "DENV1 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv1_E/curated/curated_mut_counts.csv",
    "DENV2 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv2_E/curated/curated_mut_counts.csv",
    "DENV3 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv3_E/curated/curated_mut_counts.csv",
    "DENV4 E": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv4_E/curated/curated_mut_counts.csv",
    # Dengue genome
    "DENV1 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv1_genome/curated/curated_mut_counts.csv",
    "DENV2 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv2_genome/curated/curated_mut_counts.csv",
    "DENV3 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv3_genome/curated/curated_mut_counts.csv",
    "DENV4 genome": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_denv4_genome/curated/curated_mut_counts.csv",
    # Other viruses
    "RSV A": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_rsv_a_241125/curated/curated_mut_counts.csv",
    "RSV B": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_rsv_b_251125/curated/curated_mut_counts.csv",
    "HIV-1 pol": "/scicore/home/neher/kuznet0001/mut-fitness-general/results_hiv_pol_141125/curated/curated_mut_counts.csv",
}

precomputed_default = {
    "SARS-CoV-2": "/scicore/home/neher/kuznet0001/data/sars_coefs_dict.pkl"
}

colors = {
    # Dengue E gene
    "DENV1 E": "#E74C3C",  # red
    "DENV2 E": "#3498DB",  # blue
    "DENV3 E": "#2ECC71",  # green
    "DENV4 E": "#F39C12",  # orange
    # Dengue genome
    "DENV1 genome": "#C0392B",  # dark red
    "DENV2 genome": "#2980B9",  # dark blue
    "DENV3 genome": "#27AE60",  # dark green
    "DENV4 genome": "#D68910",  # dark orange
    # Other viruses
    "RSV A": "#9B59B6",     # purple
    "RSV B": "#E67E22",     # orange
    "HIV-1 pol": "#1ABC9C", # teal
    "SARS-CoV-2": "#7F8C8D" # gray
}

# ----------------------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------------------


def load_models(datasets, precomputed_models):
    """Return dict mapping virus -> mut_type -> coefficients"""
    coefs = {}

    # Train models from CSVs
    for name, path in datasets.items():
        df = load_synonymous_muts(path)
        model = GeneralLinearModel(included_factors=['local_context'])
        model.train(df_train=df)
        coefs[name] = model.W

    # Load precomputed pickled models
    for name, path in precomputed_models.items():
        with open(path, 'rb') as f:
            coefs[name] = pickle.load(f)

    return coefs


def compute_oe_intercepts(datasets):
    """
    Compute O/E for each dataset:
    O = log(counts) = beta_0 for each mutation type
    E = log(sum(counts) / 12)
    O/E = O - E (in log space)
    
    Returns dict: virus -> mut_type -> O/E value
    """
    oe_dict = {}
    
    for name, path in datasets.items():
        df = load_synonymous_muts(path)
        
        # Get observed log counts for each mutation type (from beta_0 values)
        model = GeneralLinearModel(included_factors=['local_context'])
        model.train(df_train=df)
        
        # Extract beta_0 (intercept) for each mutation type
        observed = {mut_type: model.W[mut_type][0] for mut_type in mut_types}
        
        # Calculate expected: sum of actual counts / 12, then log
        # observed values are log(counts), so counts = exp(observed)
        total_counts = sum(np.exp(observed[mt]) for mt in mut_types)
        expected_count = total_counts / len(mut_types)
        expected_log = np.log(expected_count)
        
        # O/E in log space = O - E
        oe_dict[name] = {mut_type: observed[mut_type] - expected_log 
                         for mut_type in mut_types}
    
    return oe_dict


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
def plot_mut_coefs(coefs_dict, oe_dict, colors, mut_types, savepath=None):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=200)
    axes = axes.flatten(order='F')

    bar_width = 0.8 / len(coefs_dict)  # dynamic width
    all_names = list(coefs_dict.keys())
    min_bar, max_bar = 0, 0

    for i, mut_type in enumerate(mut_types):
        ax = axes[i]
        indices = np.arange(7)  # O/E + 6 context positions

        for j, name in enumerate(all_names):
            W = coefs_dict[name][mut_type]
            vals = np.array(W).flatten()
            
            # Replace intercept (beta_0) with O/E
            oe_value = oe_dict.get(name, {}).get(mut_type, vals[0])
            # O/E + 6 context terms
            vals = np.concatenate(([oe_value], vals[-6:]))

            # track global min/max
            min_bar = min(min_bar, np.min(vals))
            max_bar = max(max_bar, np.max(vals))

            ax.bar(indices + (j - len(all_names)/2 + 0.5)*bar_width, vals,
                   width=bar_width, color=colors[name], alpha=0.8,
                   label=name if i == 0 else "")

        # Formatting
        ax.set_title(mut_type[0] + r'$\rightarrow$' + mut_type[1])
        ax.grid(True)
        if i < 3:
            ax.set_ylabel('coefficient')

        x_labels = ["O/E"] + [
            rf"$\beta^{{{b},{pos}}}$" for b, pos in zip(
                ['C', 'G', 'T', 'C', 'G', 'T'],
                ["5'", "5'", "5'", "3'", "3'", "3'"]
            )
        ]
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

    # Uniform y-limits
    for ax in axes:
        ax.set_ylim(min_bar - 0.2, max_bar + 0.2)

    axes[0].legend(ncol=len(all_names), loc='upper right')
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)    
        plt.savefig(savepath)
    plt.show()

# ----------------------------------------------------------------------
# MAIN + ARGPARSE
# ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mutation context coefficients with O/E normalization.")

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the PDF (optional)."
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(datasets_default.keys()),
        help="Which datasets to include (default: all)."
    )

    parser.add_argument(
        "--only-precomputed",
        action="store_true",
        help="Skip training, only load precomputed models."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selected_datasets = {k: datasets_default[k]
                         for k in args.datasets if k in datasets_default}
    selected_precomputed = precomputed_default if args.only_precomputed else precomputed_default.copy()

    if not args.only_precomputed:
        coefs = load_models(selected_datasets, selected_precomputed)
        oe_dict = compute_oe_intercepts(selected_datasets)
    else:
        coefs = load_models({}, selected_precomputed)
        oe_dict = {}  # Cannot compute O/E for precomputed models without raw data

    plot_mut_coefs(
        coefs,
        oe_dict,
        colors,
        mut_types,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
