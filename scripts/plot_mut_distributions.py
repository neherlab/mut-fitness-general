#!/usr/bin/env python3
"""
Plot distributions of mutation counts.

This script generates plots showing:
- Count distributions per mutation type
- Variance-to-mean ratios
- Inter-quantile ranges
"""

import argparse
import os
import pandas as pd
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(font_scale=1.0, style='ticks', palette='colorblind')


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = scipy.stats.pearsonr(x, y)
    r2 = math.pow(r, 2)
    ax = ax or plt.gca()
    ax.annotate('$R^2$' + f' = {r2:.2f}', xy=(.05, .875), xycoords=ax.transAxes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mutation count distributions and statistics"
    )
    parser.add_argument(
        "--counts",
        required=True,
        help="Path to curated mutation counts CSV file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for figures"
    )
    return parser.parse_args()


def plot_count_distributions(counts_df, output_dir):
    """Plot count distributions, variance ratios for each mutation type."""
    
    counts_var = 'truncated_actual_count'
    
    # Calculate mean and variance statistics per mutation type
    mean_counts_df = counts_df[
        counts_df['mut_class'] == 'synonymous'
    ].groupby('mut_type', as_index=False).agg(
        mean=(counts_var, 'mean'),
        var=(counts_var, 'var'),
        std=(counts_var, 'std'),
        n=('mut_type', 'count')
    )
    mean_counts_df['var_over_mean'] = mean_counts_df['var'] / mean_counts_df['mean']
    mean_counts_df['var_over_mean2'] = mean_counts_df['var'] / np.power(mean_counts_df['mean'], 2)
    mean_counts_df['mut_type_arrow'] = mean_counts_df['mut_type'].apply(
        lambda x: x[0] + r'$\rightarrow$' + x[1]
    )
    mean_counts_df['wt_nt'] = mean_counts_df['mut_type'].apply(lambda x: x[0])
    mut_order = mean_counts_df['mut_type_arrow'].unique()
    
    # Prepare data for plotting
    data = counts_df[counts_df['mut_class'] == 'synonymous'].copy()
    data['mut_type_arrow'] = data['mut_type'].apply(
        lambda x: x[0] + r'$\rightarrow$' + x[1]
    )
    data['wt_nt'] = data['mut_type'].apply(lambda x: x[0])
    
    # Create figure with 3 subplots
    (fig, axs) = plt.subplots(nrows=3, sharex=True, figsize=[5, 6])
    mut_type_pal = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    
    # Plot count distributions
    sns.boxplot(
        x='mut_type_arrow', y=counts_var, data=data, whis=(5, 95), showfliers=False,
        order=mut_order, ax=axs[0],
        hue='wt_nt', dodge=False, palette=mut_type_pal
    )
    axs[0].set(
        xlabel='', ylabel=f'synonymous mut.\ncounts per site', yscale='log',
        ylim=[0.5, max(data[counts_var]) * 2], yticks=[1, 1e1, 1e2, 1e3]
    )
    
    # Plot variance over mean
    sns.barplot(
        x='mut_type_arrow', y='var_over_mean', data=mean_counts_df,
        order=mut_order, ax=axs[1],
        hue='wt_nt', dodge=False, palette=mut_type_pal,
    )
    axs[1].axhline(1, ls='--', c='0.25')
    axs[1].set(
        xlabel='mutation type', ylabel=r'$\frac{{\sigma^2}}{{\mu}}$', yscale='log',
        ylim=[0.5, max(mean_counts_df['var_over_mean']) * 2]
    )
    axs[1].minorticks_off()
    axs[1].yaxis.label.set(rotation='horizontal', va='center', ha='right', fontsize=19)
    
    # Plot variance over mean squared
    sns.barplot(
        x='mut_type_arrow', y='var_over_mean2', data=mean_counts_df,
        order=mut_order, ax=axs[2],
        hue='wt_nt', dodge=False, palette=mut_type_pal,
    )
    axs[2].axhline(1, ls='--', c='0.25')
    axs[2].set(
        xlabel='mutation type', ylabel=r'$\frac{{\sigma^2}}{{\mu^2}}$',
        yscale='log', ylim=[0.1, max(mean_counts_df['var_over_mean2']) * 2]
    )
    axs[2].minorticks_off()
    axs[2].yaxis.label.set(rotation='horizontal', va='center', ha='right', fontsize=19)
    axs[2].tick_params('x', labelrotation=90)
    
    for i in [0, 1, 2]:
        axs[i].grid(axis='y')
        axs[i].get_legend().remove()
    
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(output_dir, 'mut_rates.png'), dpi=300)
    plt.close()
    print(f"Saved mutation rate distributions to {output_dir}/mut_rates.png")


def plot_inter_quantile_ranges(counts_df, output_dir):
    """Plot inter-quantile ranges for each mutation type."""
    
    counts_var = 'actual_count'
    (fig, axs) = plt.subplots(
        ncols=3, nrows=4, sharex=True, sharey=True, figsize=[10, 8]
    )
    axs = axs.reshape(-1)
    
    for i, (mut_type, data) in enumerate(counts_df.groupby('mut_type')):
        mut_type_arrow = mut_type[0] + r'$\rightarrow$' + mut_type[1]
        
        # Get synonymous data
        syn_data = data[data['mut_class'] == 'synonymous'].copy()
        
        step = 0.01
        diffs = np.arange(step, 0.5 + step, step)
        diffs_with_real_values = []
        fold_changes = []
        for diff in diffs:
            lower_bound = syn_data[counts_var].quantile(0.5 - diff)
            upper_bound = syn_data[counts_var].quantile(0.5 + diff)
            if lower_bound > 0:
                diffs_with_real_values.append(diff)
                fold_changes.append(upper_bound / lower_bound)
        
        sns.scatterplot(x=diffs_with_real_values, y=fold_changes, ax=axs[i])
        sns.lineplot(x=diffs_with_real_values, y=fold_changes, ax=axs[i])
        axs[i].set(title=mut_type_arrow, yscale='log')
        axs[i].grid()
    
    fig.text(0.5, -0.02, 'quantile pairs of 0.5 $+/-$X', ha='center')
    fig.text(-0.02, 0.5, 'fold change in counts between the indicated pair of quantiles',
             va='center', rotation='vertical')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inter_quantile_ranges.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved inter-quantile ranges to {output_dir}/inter_quantile_ranges.png")


def plot_mutation_type_correlations(counts_df, output_dir):
    """Plot correlations between mutation types at the same sites."""
    
    counts_var = 'actual_count'
    syn_df = counts_df[counts_df['mut_class'] == 'synonymous'].copy()
    
    # Aggregate by site and mutation type (sum across any duplicates like different clades)
    agg_df = syn_df.groupby(['nt_site', 'mut_type'])[counts_var].sum().reset_index()
    data = agg_df.pivot(index='nt_site', columns='mut_type', values=counts_var)
    
    mut_sets = [
        ['AC', 'AG', 'AT'],
        ['CA', 'CG', 'CT'],
        ['GA', 'GC', 'GT'],
        ['TA', 'TC', 'TG']
    ]
    
    for (mut_set_i, mut_set) in enumerate(mut_sets, 1):
        # Filter to mutation types that exist in the data
        available_muts = [m for m in mut_set if m in data.columns]
        if len(available_muts) < 2:
            continue
            
        g = sns.pairplot(
            data[available_muts].dropna().rename(columns={x: x[0] + r'$\rightarrow$' + x[1] for x in available_muts}),
            corner=True, kind='hist',
            plot_kws={'bins': 15},
            height=2
        )
        g.map_lower(corrfunc)
        for ax in g.diag_axes:
            ax.set_visible(False)
        for i, y_var in enumerate(g.y_vars):
            for j, x_var in enumerate(g.x_vars):
                if x_var == y_var:
                    g.axes[i, j].set_visible(False)
        plt.savefig(os.path.join(output_dir, f'mut_type_corr_{mut_set_i}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved mutation type correlations {mut_set_i} to mut_type_corr_{mut_set_i}.png")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading counts from {args.counts}")
    counts_df = pd.read_csv(args.counts)
    
    # Truncate counts at 98th percentile
    counts_df['truncated_actual_count'] = counts_df.groupby('mut_type')['actual_count'].transform(
        lambda x: x.clip(upper=x.quantile(0.98))
    )
    
    # Generate plots
    print("Generating count distribution plots...")
    plot_count_distributions(counts_df, args.output_dir)
    
    print("Generating inter-quantile range plots...")
    plot_inter_quantile_ranges(counts_df, args.output_dir)
    
    print("Generating mutation type correlation plots...")
    plot_mutation_type_correlations(counts_df, args.output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
