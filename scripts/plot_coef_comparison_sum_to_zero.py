import os, sys
myDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.split(myDir)[0]
if not (sys.path.__contains__(parentDir)):
    sys.path.append(parentDir)
from modules.coef_plotting import (
    MUT_TYPES, load_dataset_registry, load_models,
    build_arg_parser, select_datasets,
)
import numpy as np
import matplotlib.pyplot as plt


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
                      for mut_type in MUT_TYPES if mut_type in W_dict}

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

if __name__ == "__main__":
    parser = build_arg_parser("Plot mutation context coefficients in sum-to-zero encoding with O/E normalization.")
    args = parser.parse_args()
    datasets_default, precomputed_default, colors = load_dataset_registry(args.datasets_file)

    selected_datasets, selected_precomputed = select_datasets(
        args, datasets_default, precomputed_default)

    if not args.only_precomputed:
        coefs = load_models(selected_datasets, selected_precomputed, transform=transform_to_sum_to_zero)
    else:
        coefs = load_models({}, selected_precomputed, transform=transform_to_sum_to_zero)

    # Compute O/E intercepts
    oe_dict = compute_oe_intercepts(coefs)

    plot_mut_coefs(
        coefs,
        oe_dict,
        colors,
        MUT_TYPES,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
