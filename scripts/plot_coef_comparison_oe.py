import os, sys
myDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.split(myDir)[0]
if not (sys.path.__contains__(parentDir)):
    sys.path.append(parentDir)
from modules.coef_plotting import (
    MUT_TYPES, load_dataset_registry, load_models,
    build_arg_parser, select_datasets,
)
from modules.glm import GeneralLinearModel
from modules.load import load_synonymous_muts
import numpy as np
import matplotlib.pyplot as plt


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
                    for mut_type in MUT_TYPES}

        # Calculate expected: sum of actual counts / 12, then log
        # observed values are log(counts), so counts = exp(observed)
        total_counts = sum(np.exp(observed[mt]) for mt in MUT_TYPES)
        expected_count = total_counts / len(MUT_TYPES)
        expected_log = np.log(expected_count)

        # O/E in log space = O - E
        oe_dict[name] = {mut_type: float(observed[mut_type] - expected_log)
                         for mut_type in MUT_TYPES}

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
        for mut_type in MUT_TYPES:
            if mut_type in W_dict:
                vals = np.array(W_dict[mut_type]).flatten()
                observed[mut_type] = float(vals[0])

        if len(observed) == 0:
            continue

        # Calculate expected: sum of actual counts / 12, then log
        # observed values are log(counts), so counts = exp(observed)
        total_counts = sum(np.exp(observed[mt]) for mt in MUT_TYPES if mt in observed)
        expected_count = total_counts / len(MUT_TYPES)
        expected_log = np.log(expected_count)

        # O/E in log space = O - E
        oe_dict[name] = {mut_type: float(observed[mut_type] - expected_log)
                         for mut_type in MUT_TYPES if mut_type in observed}

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


if __name__ == "__main__":
    parser = build_arg_parser("Plot mutation context coefficients with O/E normalization.")
    args = parser.parse_args()
    datasets_default, precomputed_default, colors = load_dataset_registry(args.datasets_file)

    selected_datasets, selected_precomputed = select_datasets(
        args, datasets_default, precomputed_default)

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
        MUT_TYPES,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
