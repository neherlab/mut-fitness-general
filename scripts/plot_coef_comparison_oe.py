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
    "DENV1 E": "#cc0000",  # red (poster color)
    "DENV2 E": "#3498DB",  # blue
    "DENV3 E": "#2ECC71",  # green
    "DENV4 E": "#F39C12",  # orange
    # Dengue genome
    "DENV1 genome": "#C0392B",  # dark red
    "DENV2 genome": "#2980B9",  # dark blue
    "DENV3 genome": "#27AE60",  # dark green
    "DENV4 genome": "#D68910",  # dark orange
    # Other viruses - poster colors
    "RSV A": "#f1c232",     # yellow/gold (poster color)
    "RSV B": "#6aa84f",     # green (poster color)
    "HIV-1 pol": "#3d85c6", # blue (poster color)
    "HIV-1 pol local": "#3d85c6", # blue (same as HIV-1 pol)
    "SARS-CoV-2": "#8e7cc3" # purple (complementary to poster palette)
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
        
        # Extract beta_0 (intercept) for each mutation type - ensure scalar
        observed = {mut_type: float(np.array(model.W[mut_type]).flatten()[0]) 
                    for mut_type in mut_types}
        
        # Calculate expected: sum of actual counts / 12, then log
        # observed values are log(counts), so counts = exp(observed)
        total_counts = sum(np.exp(observed[mt]) for mt in mut_types)
        expected_count = total_counts / len(mut_types)
        expected_log = np.log(expected_count)
        
        # O/E in log space = O - E
        oe_dict[name] = {mut_type: float(observed[mut_type] - expected_log) 
                         for mut_type in mut_types}
    
    return oe_dict


def compute_oe_from_coefs(coefs_dict):
    """
    Compute O/E from already-trained coefficients (for precomputed models).
    O = beta_0 for each mutation type
    E = log(sum(exp(beta_0)) / 12)
    O/E = O - E (in log space)
    
    Returns dict: virus -> mut_type -> O/E value
    """
    oe_dict = {}
    
    for name, W_dict in coefs_dict.items():
        # Extract beta_0 (intercept) for each mutation type
        observed = {}
        for mut_type in mut_types:
            if mut_type in W_dict:
                vals = np.array(W_dict[mut_type]).flatten()
                observed[mut_type] = float(vals[0])
        
        if len(observed) == 0:
            continue
            
        # Calculate expected: sum of actual counts / 12, then log
        # observed values are log(counts), so counts = exp(observed)
        total_counts = sum(np.exp(observed[mt]) for mt in mut_types if mt in observed)
        expected_count = total_counts / len(mut_types)
        expected_log = np.log(expected_count)
        
        # O/E in log space = O - E
        oe_dict[name] = {mut_type: float(observed[mut_type] - expected_log) 
                         for mut_type in mut_types if mut_type in observed}
    
    return oe_dict


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
def plot_mut_coefs(coefs_dict, oe_dict, colors, mut_types, savepath=None):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=200)
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
        ax.set_title(mut_type[0] + r'$\rightarrow$' + mut_type[1], fontsize=14)
        ax.grid(True, alpha=0.3)
        if i < 3:
            ax.set_ylabel('coefficient', fontsize=13)

        x_labels = ["O/E"] + [
            rf"$\beta^{{{b},{pos}}}$" for b, pos in zip(
                ['C', 'G', 'T', 'C', 'G', 'T'],
                ["5'", "5'", "5'", "3'", "3'", "3'"]
            )
        ]
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=12)
        ax.tick_params(axis='y', labelsize=11)

    # Uniform y-limits
    for ax in axes:
        ax.set_ylim(min_bar - 0.2, max_bar + 0.2)

    # Legend below the plots
    fig.legend(all_names, loc='lower center', ncol=len(all_names), 
               bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)    
        plt.savefig(savepath, bbox_inches='tight')
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
    
    # Filter precomputed models: only include those that exist
    selected_precomputed = {}
    for k, v in precomputed_default.items():
        if k in args.datasets and os.path.exists(v):
            selected_precomputed[k] = v

    if not args.only_precomputed:
        coefs = load_models(selected_datasets, selected_precomputed)
        # Compute O/E for datasets loaded from CSV
        oe_dict = compute_oe_intercepts(selected_datasets)
        # Also compute O/E for precomputed models from their coefficients
        oe_precomputed = compute_oe_from_coefs({k: coefs[k] for k in selected_precomputed.keys() if k in coefs})
        oe_dict.update(oe_precomputed)
    else:
        coefs = load_models({}, selected_precomputed)
        # Compute O/E from precomputed coefficients
        oe_dict = compute_oe_from_coefs(coefs)

    plot_mut_coefs(
        coefs,
        oe_dict,
        colors,
        mut_types,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
