#!/usr/bin/env python3
"""
Plot correlations between reverse-complement mutation type pairs and summary plot.
Directly adapted from SARS2-synonymous-mut-rate notebook.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq

sns.set_theme(font_scale=1.0, style='ticks', palette='colorblind')


def parse_args():
    parser = argparse.ArgumentParser(description="Plot motif correlation analysis")
    parser.add_argument('--counts', required=True, help='Path to curated counts CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory for plots')
    parser.add_argument('--min-sites', type=int, default=10, help='Minimum number of sites per motif (default: 10)')
    return parser.parse_args()


def get_ref_motif(mut_type, motif):
    """Reference motif handling - reverse complement if mutation is from A or G."""
    if mut_type[0] in ['A', 'G']:
        return str(Seq(motif).reverse_complement())
    else:
        return motif


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading counts from {args.counts}...")
    counts_df = pd.read_csv(args.counts)
    
    # Truncate counts at 98th percentile like the notebook
    counts_df['truncated_actual_count'] = counts_df.groupby('mut_type')['actual_count'].transform(
        lambda x: x.clip(upper=x.quantile(0.98))
    )
    
    # Calculate motif medians
    counts_var = 'truncated_actual_count'
    global_medians = (
        counts_df[counts_df['mut_class'] == 'synonymous']
        .groupby(['mut_type'], as_index=False)
        .agg(global_median_counts=pd.NamedAgg(column=counts_var, aggfunc='median'))
    )
    
    motif_medians = (
        counts_df[counts_df['mut_class'] == 'synonymous']
        .groupby(['mut_type', 'motif'], as_index=False)
        .agg(
            median_counts=pd.NamedAgg(column=counts_var, aggfunc='mean'),
            n=pd.NamedAgg(column='motif', aggfunc='count'),
        )
        .query(f'n >= {args.min_sites}')
        .merge(global_medians, on='mut_type')
    )
    motif_medians['fold_change_from_global'] = (
        motif_medians['median_counts'] / motif_medians['global_median_counts']
    )
    motif_medians['ref_motif'] = motif_medians.apply(
        lambda row: get_ref_motif(row['mut_type'], row['motif']), axis=1
    )
    motif_medians['mut_type_arrow'] = motif_medians['mut_type'].apply(
        lambda x: x[0] + r'$\rightarrow$' + x[1]
    )
    
    # Plot RC mutation type correlations
    print("Generating reverse-complement mutation type correlation plots...")
    mut_type_pairs = [
        ['TG', 'AC'],
        ['TA', 'AT'],
        ['TC', 'AG'],
        ['CT', 'GA'],
        ['CA', 'GT'],
        ['CG', 'GC'],
    ]
    
    mirror_metadata = {}
    for mut_type_pair in mut_type_pairs:
        data = (
            motif_medians[motif_medians['mut_type'].isin(mut_type_pair)]
            .pivot(index='ref_motif', columns='mut_type', values='median_counts')
            .dropna()
        )
        
        if len(data) < 2:
            print(f"Skipping {mut_type_pair}: insufficient data")
            continue
        
        x = mut_type_pair[1]
        y = mut_type_pair[0]
        
        valid_data = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) < 2 or valid_data[x].nunique() <= 1 or valid_data[y].nunique() <= 1:
            print(f"Skipping {mut_type_pair}: insufficient or constant data")
            continue
        
        plt.figure(figsize=[2.5, 2.5])
        sns.scatterplot(x=x, y=y, data=data, s=85, alpha=1.0)
        r = data[x].corr(data[y])
        (m, b) = np.polyfit(data[x], data[y], 1)
        plt.annotate(f'R = {round(r, 2)}', [0.05, 0.9], xycoords='axes fraction')
        mirror_metadata[mut_type_pair[0]] = [round(r, 4), round(m, 2)]
        print(f"{mut_type_pair}: r = {round(r, 3)}, m = {round(m, 3)}, b = {round(b, 3)}")
        
        min_val = data.min().min()
        max_val = data.max().max()
        if mut_type_pair[0][0] == 'T':
            plt.plot([min_val, max_val], [min_val, max_val], ls='--', c='0.25')
        
        plt.xlabel(mut_type_pair[1][0] + r'$\rightarrow$' + mut_type_pair[1][1])
        plt.ylabel(mut_type_pair[0][0] + r'$\rightarrow$' + mut_type_pair[0][1])
        plt.title('median counts per motif', y=1.05)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'local_context_corr_{y}_{x}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot summary stripplot
    print("Generating summary plot...")
    mut_types_order = []
    for mut_type_pair in mut_type_pairs:
        mut_types_order += mut_type_pair
    mut_type_indices = {mut_type: i for (i, mut_type) in enumerate(mut_types_order)}
    motif_medians['mut_type_index'] = motif_medians['mut_type'].apply(
        lambda x: mut_type_indices.get(x, 99)
    )
    motif_medians.sort_values('mut_type_index', inplace=True)
    
    plt.figure(figsize=[12, 3])
    sns.stripplot(x='mut_type_arrow', y='fold_change_from_global', data=motif_medians, s=10, alpha=0.5)
    plt.axhline(1, ls='--', c='0.25')
    
    for (i, mut_type) in enumerate(mut_types_order):
        if mut_type in mirror_metadata:
            (r, m) = mirror_metadata[mut_type]
            r = f'R = {round(r, 2)}'
            plt.plot([i, i + 1], [8, 8], c='k')
            plt.annotate(f'{r}', [i + 0.5, 9.5], ha='center', va='bottom')
    
    plt.tick_params('x', labelrotation=90)
    plt.xlabel('mutation type')
    plt.ylabel('motif median / global median')
    ax = plt.gca()
    ax.set_yscale('log', base=2)
    plt.yticks([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8])
    plt.grid(axis='y')
    plt.minorticks_off()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'local_context_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved local_context_summary.png")
    
    # Plot RC correlations by pairing state if RNA structure available
    has_rna_structure = 'unpaired' in counts_df.columns and counts_df['unpaired'].nunique() > 1
    if has_rna_structure:
        print("Generating RC correlations by pairing state...")
        for pairing_state in ['paired', 'unpaired']:
            pairing_val = pairing_state == 'unpaired'
            
            # Calculate motif medians for this pairing state
            pairing_motif_medians = (
                counts_df[
                    (counts_df['mut_class'] == 'synonymous') &
                    (counts_df['unpaired'] == pairing_val)
                ]
                .groupby(['mut_type', 'motif'], as_index=False)
                .agg(
                    median_counts=pd.NamedAgg(column=counts_var, aggfunc='mean'),
                    n=pd.NamedAgg(column='motif', aggfunc='count'),
                )
                .query(f'n >= {args.min_sites}')
            )
            pairing_motif_medians['ref_motif'] = pairing_motif_medians.apply(
                lambda row: get_ref_motif(row['mut_type'], row['motif']), axis=1
            )
            
            for mut_type_pair in mut_type_pairs:
                # Check if both mutation types exist in this pairing state
                if not all(mt in pairing_motif_medians['mut_type'].values for mt in mut_type_pair):
                    continue
                    
                data = (
                    pairing_motif_medians[pairing_motif_medians['mut_type'].isin(mut_type_pair)]
                    .pivot(index='ref_motif', columns='mut_type', values='median_counts')
                    .dropna()
                )
                
                if len(data) < 2:
                    continue
                
                x = mut_type_pair[1]
                y = mut_type_pair[0]
                
                valid_data = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) < 2 or valid_data[x].nunique() <= 1 or valid_data[y].nunique() <= 1:
                    continue
                
                plt.figure(figsize=[2.5, 2.5])
                sns.scatterplot(x=x, y=y, data=data, s=85, alpha=1.0)
                r = data[x].corr(data[y])
                (m, b) = np.polyfit(data[x], data[y], 1)
                plt.annotate(f'R = {round(r, 2)}', [0.05, 0.9], xycoords='axes fraction')
                
                min_val = data.min().min()
                max_val = data.max().max()
                if mut_type_pair[0][0] == 'T':
                    plt.plot([min_val, max_val], [min_val, max_val], ls='--', c='0.25')
                
                plt.xlabel(mut_type_pair[1][0] + r'$\rightarrow$' + mut_type_pair[1][1])
                plt.ylabel(mut_type_pair[0][0] + r'$\rightarrow$' + mut_type_pair[0][1])
                plt.title(f'median counts per motif ({pairing_state})', y=1.05)
                sns.despine()
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'local_context_corr_{y}_{x}_{pairing_state}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved local_context_corr_{y}_{x}_{pairing_state}.png")
    


if __name__ == '__main__':
    main()
