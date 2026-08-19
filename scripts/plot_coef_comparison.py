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
# PLOT
# ----------------------------------------------------------------------
def plot_mut_coefs(coefs_dict, colors, mut_types, savepath=None):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=200)
    axes = axes.flatten(order='F')

    bar_width = 0.8 / len(coefs_dict)  # dynamic width
    all_names = list(coefs_dict.keys())
    min_bar, max_bar = 0, 0

    for i, mut_type in enumerate(mut_types):
        ax = axes[i]
        indices = np.arange(7)  # intercept + 6 context positions

        for j, name in enumerate(all_names):
            W = coefs_dict[name][mut_type]
            vals = np.array(W).flatten()

            # intercept + 6 context terms
            vals = np.concatenate(([vals[0]], vals[-6:]))

            # track global min/max
            min_bar = min(min_bar, np.min(vals[1:]))
            max_bar = max(max_bar, np.max(vals[1:]))

            ax.bar(indices + (j - len(all_names)/2 + 0.5)*bar_width, vals,
                   width=bar_width, color=colors[name], alpha=0.8,
                   label=name if i == 0 else "")

        # Formatting
        ax.set_title(mut_type[0] + r'$\rightarrow$' + mut_type[1])
        ax.grid(True)
        if i < 3:
            ax.set_ylabel('coefficient')

        x_labels = [r"$\beta_0$"] + [
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

    axes[0].legend(ncol=len(all_names), loc='upper right', fontsize=8)
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    plt.show()

# ----------------------------------------------------------------------
# MAIN + ARGPARSE
# ----------------------------------------------------------------------


if __name__ == "__main__":
    parser = build_arg_parser("Plot mutation context coefficients.")
    args = parser.parse_args()
    datasets_default, precomputed_default, colors = load_dataset_registry(args.datasets_file)

    selected_datasets, selected_precomputed = select_datasets(
        args, datasets_default, precomputed_default)

    if not args.only_precomputed:
        coefs = load_models(selected_datasets, selected_precomputed)
    else:
        coefs = load_models({}, selected_precomputed)

    plot_mut_coefs(
        coefs,
        colors,
        MUT_TYPES,
        savepath=args.out
    )

    if args.no_show:
        plt.close()
