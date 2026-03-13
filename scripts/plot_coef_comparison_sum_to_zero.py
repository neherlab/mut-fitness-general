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
    # Local test data
    "HIV-1 pol local": "results_test_hiv_pol_with_rna/curated/curated_mut_counts.csv",
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
# TRANSFORMATION FUNCTIONS
# ----------------------------------------------------------------------

def transform_to_sum_to_zero(W_ref):
    """
    Transform coefficients from reference encoding to sum-to-zero encoding.
    
    Reference: y = β₀ + β_C·I(C) + β_G·I(G) + β_T·I(T)  [A=0 is reference]
    Sum-to-zero: y = β₀' + β_A'·I(A) + β_C'·I(C) + β_G'·I(G) + β_T'·I(T)
                 with constraint: β_A' + β_C' + β_G' + β_T' = 0
    
    Args:
        W_ref: Array with structure:
            - 8 coefs: [intercept, paired, C_l, G_l, T_l, C_r, G_r, T_r]
            - 9 coefs: [intercept, switch, paired, C_l, G_l, T_l, C_r, G_r, T_r]
        
    Returns:
        Array [intercept', A_l, C_l, G_l, T_l, A_r, C_r, G_r, T_r, ...extra factors]
    """
    w = np.array(W_ref).flatten()
    
    # Extract reference encoding coefficients from last 6 positions
    intercept = w[0]
    C_l, G_l, T_l = w[-6], w[-5], w[-4]
    C_r, G_r, T_r = w[-3], w[-2], w[-1]
    
    # Compute mean effect (A=0 in reference)
    mean_l = (C_l + G_l + T_l) / 4
    mean_r = (C_r + G_r + T_r) / 4
    
    # Transform to sum-to-zero encoding
    A_l_new = -mean_l
    C_l_new = C_l - mean_l
    G_l_new = G_l - mean_l
    T_l_new = T_l - mean_l
    
    A_r_new = -mean_r
    C_r_new = C_r - mean_r
    G_r_new = G_r - mean_r
    T_r_new = T_r - mean_r
    
    # Adjust intercept to grand mean
    intercept_adj = intercept + mean_l + mean_r
    
    # Build new coefficient array
    w_new = [intercept_adj, A_l_new, C_l_new, G_l_new, T_l_new, 
             A_r_new, C_r_new, G_r_new, T_r_new]
    
    # Add any additional factors (structure, etc.)
    if len(w) > 7:
        w_new.extend(w[1:len(w)-6])  # factors between intercept and context
    
    return np.array(w_new)


def compute_oe_intercepts(coefs_dict):
    """
    Compute O/E for intercepts to normalize for sequencing depth.
    
    O = log(counts) = intercept (after sum-to-zero transform = grand mean)
    E = log(mean(counts across all 12 mutation types))
    O/E = O - E (in log space)
    
    Args:
        coefs_dict: Dict mapping virus -> mut_type -> coefficients (sum-to-zero)
        
    Returns:
        Dict mapping virus -> mut_type -> O/E value
    """
    oe_dict = {}
    
    for name, W_dict in coefs_dict.items():
        # Extract intercepts for all mutation types
        intercepts = {mut_type: float(W_dict[mut_type][0]) 
                      for mut_type in mut_types if mut_type in W_dict}
        
        if len(intercepts) == 0:
            continue
        
        # Calculate expected: mean of counts across all mutation types
        # intercepts are log(counts), so counts = exp(intercept)
        total_counts = sum(np.exp(intercepts[mt]) for mt in intercepts)
        expected_count = total_counts / len(intercepts)
        expected_log = np.log(expected_count)
        
        # O/E in log space = O - E
        oe_dict[name] = {mut_type: intercepts[mut_type] - expected_log 
                         for mut_type in intercepts}
    
    return oe_dict


# ----------------------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------------------

def load_models(datasets, precomputed_models):
    """Load and transform models to sum-to-zero encoding"""
    coefs = {}

    # Train models from CSVs and transform
    for name, path in datasets.items():
        df = load_synonymous_muts(path)
        model = GeneralLinearModel(included_factors=['local_context'])
        model.train(df_train=df)
        
        # Transform each mutation type to sum-to-zero
        coefs[name] = {}
        for mut_type, w_ref in model.W.items():
            coefs[name][mut_type] = transform_to_sum_to_zero(w_ref)

    # Load precomputed pickled models and transform
    for name, path in precomputed_models.items():
        with open(path, 'rb') as f:
            W_ref_dict = pickle.load(f)
        
        coefs[name] = {}
        for mut_type, w_ref in W_ref_dict.items():
            coefs[name][mut_type] = transform_to_sum_to_zero(w_ref)

    return coefs


# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
def plot_mut_coefs(coefs_dict, oe_dict, colors, mut_types, savepath=None):
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), dpi=200)
    axes = axes.flatten(order='F')

    bar_width = 0.8 / len(coefs_dict)  # dynamic width
    all_names = list(coefs_dict.keys())
    min_bar, max_bar = 0, 0

    for i, mut_type in enumerate(mut_types):
        ax = axes[i]
        indices = np.arange(9)  # O/E + 8 context positions (4 left + 4 right)

        for j, name in enumerate(all_names):
            W = coefs_dict[name][mut_type]
            
            # Get O/E for this mutation type
            oe_value = oe_dict.get(name, {}).get(mut_type, W[0])
            
            # Extract 8 context coefficients (A, C, G, T for left and right)
            # Ignore any additional factors (e.g., RNA structure for SARS-CoV-2)
            context_vals = W[1:9]
            
            # Combine O/E + 8 context terms
            vals = np.concatenate(([oe_value], context_vals))

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

        # X-axis labels with all 4 nucleotides
        x_labels = ["O/E"] + [
            rf"$\beta^{{{b},{pos}}}$" for b, pos in zip(
                ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T'],
                ["5'", "5'", "5'", "5'", "3'", "3'", "3'", "3'"]
            )
        ]
        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=11)
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
        savedir = os.path.dirname(savepath)
        if savedir:
            os.makedirs(savedir, exist_ok=True)    
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------
# MAIN + ARGPARSE
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mutation context coefficients in sum-to-zero encoding with O/E normalization.")

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
    else:
        coefs = load_models({}, selected_precomputed)

    # Compute O/E intercepts
    oe_dict = compute_oe_intercepts(coefs)

    plot_mut_coefs(
        coefs,
        oe_dict,
        colors,
        mut_types,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
