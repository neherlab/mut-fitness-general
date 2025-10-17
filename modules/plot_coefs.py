import numpy as np
import matplotlib.pyplot as plt
from glm import GeneralLinearModel
from load import load_synonymous_muts

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

if __name__ == '__main__':

    # Load mutation counts across full tree
    df_syn = load_synonymous_muts("/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results/curated/curated_mut_counts.csv")

    # Train the model (ONLY local_context)
    model = GeneralLinearModel(included_factors=['local_context'])
    model.train(df_train=df_syn)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=200)
    axes = axes.flatten(order='F')

    min_bar, max_bar = 0, 0

    for i, mut_type in enumerate(mut_types):
        w = model.W[mut_type].flatten()

        # Determine bar limits (exclude intercept for scaling)
        min_bar = min(np.min(w[1:]), min_bar)
        max_bar = max(np.max(w[1:]), max_bar)

        indices = np.arange(len(w))
        bar_width = 0.6
        ax = axes[i]

        # Plot coefficients excluding intercept
        ax.bar(indices[1:], w[1:], width=bar_width, color='blue')
        ax.grid(True)
        if i < 3:
            ax.set_ylabel('coefficient')
        ax.set_title(mut_type[0]+r'$\rightarrow$'+mut_type[1])

        # Generate X-axis labels dynamically based on local context size
        context_len = len(w) - 1  # excluding intercept
        # If your encoding is one-hot per base (A,C,G,T), determine positions:
        # Assuming positions around mutated site (e.g., k-mer size)
        num_bases = 3  # if you exclude A because it's reference or encoded differently
        flank = context_len // num_bases // 2  # approximate for symmetric context
        pos_labels = [f"-{i}" for i in range(flank, 0, -1)] + [f"+{i}" for i in range(1, flank+1)]
        bases = ['C', 'G', 'T']  # adjust if A is included
        x_labels = []
        for pos in pos_labels:
            for b in bases:
                x_labels.append(rf"$\beta^{{{b},{pos}}}$")

        ax.set_xticks(indices[1:])
        ax.set_xticklabels(x_labels, rotation=0, ha="center")

    # Uniform y-limits across all subplots
    for ax in axes:
        ax.set_ylim((min_bar - 0.2, max_bar + 0.2))

    plt.tight_layout()
    plt.savefig('/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/results/exploratory_figures/model_coefs_barplot.pdf')
    plt.show()
