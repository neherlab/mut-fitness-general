import numpy as np
import matplotlib.pyplot as plt
import pickle
from glm import GeneralLinearModel
from load import load_synonymous_muts

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

datasets = {
    "RSV A": "/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_rsv_a/curated/curated_mut_counts.csv",
    "RSV B": "/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_rsv_b/curated/curated_mut_counts.csv",
    "HIV-1 pol": "/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_hiv_pol_071125/curated/curated_mut_counts.csv",
}

precomputed_models = {
    "SARS-CoV-2": "/scicore/home/neher/kuznet0001/sars_coefs_dict.pkl"
}

colors = {
    "RSV A": "blue",
    "RSV B": "orange",
    "HIV-1 pol": "green",
    "SARS-CoV-2": "gray"
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
            vals = np.concatenate(([vals[0]], vals[-6:]))  # intercept + 6 context terms

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

    axes[0].legend(ncol=len(all_names), loc='upper right')
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    plt.show()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == '__main__':
    coefs_dict = load_models(datasets, precomputed_models)
    plot_mut_coefs(
        coefs_dict,
        colors,
        mut_types,
        savepath='/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_hiv_pol_071125/exploratory_figures/model_coefs_barplot_all.pdf'
    )
