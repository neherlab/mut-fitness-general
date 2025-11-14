import numpy as np
import matplotlib.pyplot as plt
import os
from glm import GeneralLinearModel
from load import load_synonymous_muts
import seaborn as sns
feature_labels = [
    "intercept",
    "L=C", "L=G", "L=T",
    "R=C", "R=G", "R=T"
]
res_path = "results_hiv_pol_071125"

def collect_weight_matrix(model, mutation_types, feature_labels):
    weight_matrix = []

    for mut_type in mutation_types:
        if mut_type in model.W:
            weights = model.W[mut_type].flatten()
            weight_matrix.append(weights)
        else:
            weight_matrix.append(np.full(len(feature_labels), np.nan))  # missing fit

    return np.array(weight_matrix)
def plot_weight_heatmap(model, feature_labels):
    mutation_types = sorted(model.W.keys())  # list of all fitted mutation types
    weight_matrix = []

    for mut_type in mutation_types:
        weights = model.W[mut_type].flatten()
        weight_matrix.append(weights)

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
    plt.savefig(f'/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/{res_path}/exploratory_figures/model_weights.pdf')
    plt.show()


def plot_sequential_r_squared(mean_sq_err_dic):

    model_types = list(mean_sq_err_dic.keys())
    mut_types = sorted(list(mean_sq_err_dic[model_types[0]].keys()))

    r2s = {mut_type: [] for mut_type in mut_types}
    for mut_type in mut_types:
        for i in range(len(model_types)):
            r2s[mut_type].append(1 - (mean_sq_err_dic[model_types[i]][mut_type] / mean_sq_err_dic[model_types[0]][mut_type]))

    plt.figure(figsize=(8, 4.5), dpi=150)

    bar_positions = np.arange(len(mut_types))
    bar_width = 0.6

    base_bar = np.zeros(len(mut_types))

    for i in range(1, len(model_types)):
        increment_values = np.array([r2s[mut_type][i] - r2s[mut_type][i-1] for mut_type in mut_types])
        plt.bar(bar_positions, increment_values, bottom=base_bar, zorder=3,
                label=model_types[i], width=bar_width, color=['#0072B2', '#E69F00', '#009E73'][i-1])
        base_bar += increment_values

    plt.ylabel('$R^{2}$')
    plt.xticks(ticks=bar_positions, labels=[rf"{mut_type[0]}$\rightarrow${mut_type[1]}" for mut_type in mut_types], rotation=45, ha="center")
    plt.grid(True, axis='y', zorder=1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)

    plt.tight_layout()
    if not os.path.isdir(f'/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/{res_path}/exploratory_figures/'):
        os.makedirs(f'/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/{res_path}/exploratory_figures/')

    plt.savefig(f'/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/{res_path}/exploratory_figures/R2_sequentially.pdf')
    plt.show()

df = load_synonymous_muts(f"/scicore/home/neher/kuznet0001/rsv_code/RSV-mut-fitness/{res_path}/curated/curated_mut_counts.csv")

    # factors = ['global_context', 'rna_structure', 'local_context']
factors = ['local_context']
# names = ['base', 'genomic position', 'RNA secondary structure', 'local context']
names = ['base', 'local context']

mean_squared_errs = {}

# for i in range(4):
for i in range(2):
    current_factors = factors[:i]
    print(f"--- Model {i} ---")
    print(f"Included factors: {current_factors}")

    model = GeneralLinearModel(included_factors=current_factors)

    # Train
    model.train(df_train=df)

    # Optionally inspect weights
    for mut_type, w in model.W.items():
        print(f"Mutation type: {mut_type}, Weights: {w.flatten()}")

    # Test
    mean_sq_err = model.test(df_test=df)
    mean_squared_errs[names[i]] = mean_sq_err

    print(f"MSEs for model {names[i]}:")
    for mut, mse in mean_sq_err.items():
        print(f"  {mut}: {mse:.4f}")

    print()

# Final result
print("All mean squared errors:")
print(mean_squared_errs)

# Plot
plot_sequential_r_squared(mean_squared_errs)
plot_weight_heatmap(model, feature_labels)
