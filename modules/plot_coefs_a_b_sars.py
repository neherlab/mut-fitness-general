import numpy as np
import matplotlib.pyplot as plt
import pickle
from glm import GeneralLinearModel
from load import load_synonymous_muts

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

# Paths
path_a = "/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_rsv_a/curated/curated_mut_counts.csv"
path_b = "/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results_rsv_b/curated/curated_mut_counts.csv"
path_sars = "/scicore/home/neher/kuznet0001/sars_coefs_dict.pkl"

if __name__ == '__main__':
    # Load data
    df_a = load_synonymous_muts(path_a)
    df_b = load_synonymous_muts(path_b)
    with open(path_sars, 'rb') as f:
        sars_W_full = pickle.load(f)

    # Train models
    model_a = GeneralLinearModel(included_factors=['local_context'])
    model_a.train(df_train=df_a)
    model_b = GeneralLinearModel(included_factors=['local_context'])
    model_b.train(df_train=df_b)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=200)
    axes = axes.flatten(order='F')

    min_bar, max_bar = 0, 0
    bar_width = 0.25

    for i, mut_type in enumerate(mut_types):
        ax = axes[i]

        # Get coefficients
        w_a = model_a.W[mut_type].flatten()
        w_b = model_b.W[mut_type].flatten()
        sars_full = np.array(sars_W_full[mut_type]).flatten()
        vals_sars = np.concatenate(([sars_full[0]], sars_full[-6:]))  # 7 bars
        vals_a = np.concatenate(([w_a[0]], w_a[-6:]))
        vals_b = np.concatenate(([w_b[0]], w_b[-6:]))

        # Determine min/max for uniform y-limits (exclude intercept)
        min_bar = min(min(vals_a[1:]), min(vals_b[1:]), min(vals_sars[1:]), min_bar)
        max_bar = max(max(vals_a[1:]), max(vals_b[1:]), max(vals_sars[1:]), max_bar)

        indices = np.arange(len(vals_a))

        # Plot bars
        ax.bar(indices - bar_width, vals_a, width=bar_width, color='blue', alpha=0.8, label='RSV A' if i==0 else "")
        ax.bar(indices, vals_b, width=bar_width, color='orange', alpha=0.8, label='RSV B' if i==0 else "")
        ax.bar(indices + bar_width, vals_sars, width=bar_width, color='gray', alpha=0.8, label='SARS-CoV-2' if i==0 else "")

        # Titles and grid
        ax.set_title(mut_type[0]+r'$\rightarrow$'+mut_type[1])
        ax.grid(True)
        if i < 3:
            ax.set_ylabel('coefficient')

        # X-axis labels (keep original β style)
        bases = ['C', 'G', 'T']
        # Only keep 6 labels after intercept (3 left + 3 right)
        x_labels = [r"$\beta_0$"] + [rf"$\beta^{{{b},{pos}}}$" for b,pos in zip(['C','G','T','C','G','T'], ["5'","5'","5'","3'","3'","3'"])]

        ax.set_xticks(indices)
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

    # Uniform y-limits
    for ax in axes:
        ax.set_ylim(min_bar - 0.2, max_bar + 0.2)

    # Legend
    axes[0].legend(ncol=3, loc='upper right')

    plt.tight_layout()
    plt.savefig('/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results/exploratory_figures/model_coefs_barplot_a_b_sars.pdf')
    plt.show()
