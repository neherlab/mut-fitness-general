#!/usr/bin/env python3
"""
Plot local sequence context effects - stripplots by mutation type.
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

mut_class_colors = {
    'synonymous': sns.color_palette('colorblind')[0],
    'nonsynonymous': sns.color_palette('colorblind')[1],
    'stop': sns.color_palette('colorblind')[2]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot local context effects")
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
    counts_df['mut_type_arrow'] = counts_df['mut_type'].apply(
        lambda x: x[0] + r'$\rightarrow$' + x[1]
    )
    
    # Calculate global medians and motif medians
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
    
    # Plot local context for each mutation type
    mut_classes = ['synonymous']
    mut_types = ['TG', 'AC', 'TA', 'AT', 'TC', 'AG', 'CT', 'GA', 'CA', 'GT', 'CG', 'GC']
    
    # Check if RNA structure data is available
    has_rna_structure = 'unpaired' in counts_df.columns and counts_df['unpaired'].nunique() > 1
    
    for mut_type in mut_types:
        print(f"Plotting {mut_type}...")
        
        # Get data for mutation type
        data = counts_df[
            (counts_df['mut_class'].isin(mut_classes)) &
            (counts_df['mut_type'] == mut_type)
        ].copy()
        
        if len(data) == 0:
            print(f"No data for {mut_type}, skipping")
            continue
        
        global_median = data[data['mut_class'] == 'synonymous'][counts_var].median()
        global_mean = data[data['mut_class'] == 'synonymous'][counts_var].mean()
        motif_medians_data = motif_medians[motif_medians['mut_type'] == mut_type]
        
        # Sort motifs by their median, then plot the distribution of counts for each motif
        data = data.merge(motif_medians_data, on=['mut_type', 'motif'], how='right')
        data.sort_values('median_counts', inplace=True)
        
        plt.figure(figsize=[8, 3])
        sns.boxplot(
            x='motif', y=counts_var, data=data,
            hue='mut_class', hue_order=mut_classes, palette=mut_class_colors,
            showfliers=False, whis=0,
            linewidth=0,
            medianprops={"linewidth": 2},
            boxprops={'alpha': .3},
            zorder=10,
            showmeans=True,
        )
        sns.stripplot(
            x='motif', y=counts_var, data=data,
            hue='mut_class', hue_order=mut_classes, palette=mut_class_colors,
            dodge=True, alpha=0.5, s=7
        )
        plt.axhline(global_median, c='0.75', ls='--')
        plt.axhline(global_mean, c='red', ls='--', alpha=0.5)
        plt.legend().remove()
        sns.despine()
        plt.title(mut_type[0] + r'$\rightarrow$' + mut_type[1])
        plt.ylabel('synonymous mutation counts')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'local_context_{mut_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # If RNA structure available, also plot by pairing state
    if has_rna_structure:
        print("Generating plots separated by RNA structure...")
        for mut_type in mut_types:
            for pairing_state in ['paired', 'unpaired']:
                pairing_val = pairing_state == 'unpaired'
                data = counts_df[
                    (counts_df['mut_class'].isin(mut_classes)) &
                    (counts_df['unpaired'] == pairing_val) &
                    (counts_df['mut_type'] == mut_type)
                ].copy()
                
                if len(data) == 0:
                    continue
                
                global_median = data[data['mut_class'] == 'synonymous'][counts_var].median()
                medians_df = (
                    data[data['mut_class'] == 'synonymous']
                    .groupby(['mut_type', 'motif'], as_index=False)
                    .agg(
                        median_counts=pd.NamedAgg(column=counts_var, aggfunc='median'),
                        n=pd.NamedAgg(column='motif', aggfunc='count'),
                    )
                    .query(f'n >= {args.min_sites}')
                    .merge(global_medians, on='mut_type')
                )
                
                if len(medians_df) == 0:
                    continue
                
                data = data.merge(medians_df, on=['mut_type', 'motif'])
                data.sort_values('median_counts', inplace=True)
                
                plt.figure(figsize=[8, 3])
                sns.boxplot(
                    x='motif', y=counts_var, data=data,
                    hue='mut_class', hue_order=mut_classes, palette=mut_class_colors,
                    showfliers=False, whis=0,
                    linewidth=0,
                    medianprops={"linewidth": 2},
                    boxprops={'alpha': .3},
                    zorder=10,
                    showmeans=True,
                )
                sns.stripplot(
                    x='motif', y=counts_var, data=data,
                    hue='mut_class', hue_order=mut_classes, palette=mut_class_colors,
                    dodge=True, alpha=0.5, s=7
                )
                plt.axhline(global_median, c='0.75', ls='--')
                plt.legend().remove()
                sns.despine()
                plt.title(mut_type[0] + r'$\rightarrow$' + mut_type[1] + f' ({pairing_state})')
                plt.ylabel('synonymous mutation counts')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'local_context_{mut_type}_{pairing_state}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved local_context_{mut_type}_{pairing_state}.png")
        
        # Add pairing effect facet plot
        print("Generating pairing effect plot...")
        data = (
            counts_df[counts_df['mut_class'] == 'synonymous']
            .groupby(['mut_type_arrow', 'motif', 'unpaired'], as_index=False)
            .agg(
                mean_counts=pd.NamedAgg(column=counts_var, aggfunc='median'),
                n=pd.NamedAgg(column='motif', aggfunc='count'),
            )
            .query(f'n >= {args.min_sites}')
            .assign(ss_prediction=lambda x: x['unpaired'].map({True: 'unpaired', False: 'paired'}))
            .pivot_table(index=['mut_type_arrow', 'motif'], columns='ss_prediction', values='mean_counts')
            .reset_index()
            .dropna()
        )
        
        nobs = data['mut_type_arrow'].value_counts()
        mut_types_to_plot = list(nobs[nobs > 1].index.values)
        data = data[data['mut_type_arrow'].isin(mut_types_to_plot)]
        
        if len(data) > 0:
            g = sns.FacetGrid(data, col='mut_type_arrow', col_wrap=4, sharex=False, sharey=False, height=2, aspect=1)
            g.map(sns.scatterplot, 'paired', 'unpaired', alpha=0.5, s=85)
            g.set_titles("{col_name}")
            plt.suptitle('median counts per motif', y=1.05)
            plt.savefig(os.path.join(args.output_dir, 'local_context_pairing_effect.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved local_context_pairing_effect.png")
    
    print("Done!")


if __name__ == '__main__':
    main()
