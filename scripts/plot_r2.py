import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
myDir = os.path.dirname(os.path.abspath(__file__))
parentDir = os.path.split(myDir)[0]
if not (sys.path.__contains__(parentDir)):
    sys.path.append(parentDir)
    print(sys.path)
from modules.glm import GeneralLinearModel
from modules.load import load_synonymous_muts


feature_labels = [
    "intercept",
    "L=C", "L=G", "L=T",
    "R=C", "R=G", "R=T"
]


# ------------------------------------------------------------
# ARGPARSE
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GLM models and plot weights/R2 sequential models."
    )

    parser.add_argument(
        "--path",
        required=True,
        help="Path to the result folder containing `curated/curated_mut_counts.csv`."
    )

    return parser.parse_args()


# ------------------------------------------------------------
# UTIL
# ------------------------------------------------------------
def collect_weight_matrix(model, mutation_types, feature_labels):
    weight_matrix = []

    for mut_type in mutation_types:
        if mut_type in model.W:
            weights = model.W[mut_type].flatten()
            weight_matrix.append(weights)
        else:
            weight_matrix.append(np.full(len(feature_labels), np.nan))

    return np.array(weight_matrix)


def plot_weight_heatmap(model, feature_labels, outdir):
    mutation_types = sorted(model.W.keys())
    weight_matrix = [model.W[m].flatten() for m in mutation_types]
    weight_matrix = np.array(weight_matrix)

    plt.figure(figsize=(10, 0.6 * len(mutation_types)))
    sns.heatmap(
        weight_matrix,
        xticklabels=feature_labels,
        yticklabels=mutation_types,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )
    plt.title("Model Weights Across Mutation Types")
    plt.xlabel("Feature")
    plt.ylabel("Mutation Type")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_weights.pdf"))
    plt.show()


def plot_sequential_r_squared(mean_sq_err_dic, outdir):
    model_types = list(mean_sq_err_dic.keys())
    mut_types = sorted(mean_sq_err_dic[model_types[0]].keys())

    r2s = {m: [] for m in mut_types}
    for mut in mut_types:
        for i in range(len(model_types)):
            r2s[mut].append(
                1 - (mean_sq_err_dic[model_types[i]][mut] /
                     mean_sq_err_dic[model_types[0]][mut])
            )

    plt.figure(figsize=(8, 4.5), dpi=150)

    bar_positions = np.arange(len(mut_types))
    bar_width = 0.6
    base_bar = np.zeros(len(mut_types))

    for i in range(1, len(model_types)):
        increment = np.array(
            [r2s[m][i] - r2s[m][i - 1] for m in mut_types]
        )
        plt.bar(
            bar_positions,
            increment,
            bottom=base_bar,
            zorder=3,
            label=model_types[i],
            width=bar_width,
            color=['#0072B2', '#E69F00', '#009E73'][i - 1]
        )
        base_bar += increment

    plt.ylabel('$R^{2}$')
    plt.xticks(
        ticks=bar_positions,
        labels=[f"{m[0]}→{m[1]}" for m in mut_types],
        rotation=45,
        ha="center"
    )
    plt.grid(True, axis='y', zorder=1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "R2_sequentially.pdf"))
    plt.show()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    res_path = args.path.rstrip("/")
    curated_csv = os.path.join(res_path, "curated", "curated_mut_counts.csv")
    figs_out = os.path.join(res_path, "exploratory_figures")
    os.makedirs(figs_out, exist_ok=True)

    df = load_synonymous_muts(curated_csv)

    factors = ['local_context']
    names = ['base', 'local context']

    mean_squared_errs = {}

    for i in range(2):
        current_factors = factors[:i]
        print(f"--- Model {i} ---")
        print(f"Included factors: {current_factors}")

        model = GeneralLinearModel(included_factors=current_factors)

        model.train(df_train=df)

        for mut_type, w in model.W.items():
            print(f"Mutation type: {mut_type}, Weights: {w.flatten()}")

        mean_sq_err = model.test(df_test=df)
        mean_squared_errs[names[i]] = mean_sq_err

    plot_sequential_r_squared(mean_squared_errs, figs_out)
    plot_weight_heatmap(model, feature_labels, figs_out)
