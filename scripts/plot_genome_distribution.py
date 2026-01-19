#!/usr/bin/env python3
"""
Plot mutation distribution along the genome using sliding windows.
Directly adapted from user's analyze_counts_hiv_pol.ipynb notebook.
"""

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(font_scale=1.0, style='ticks', palette='colorblind')

mut_class_colors = {
    'synonymous': sns.color_palette('colorblind')[0],
    'nonsynonymous': sns.color_palette('colorblind')[1],
    'nonsense': sns.color_palette('colorblind')[2],
    'stop': sns.color_palette('colorblind')[2]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot genome distribution")
    parser.add_argument('--counts', required=True, help='Path to curated counts CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory for plots')
    parser.add_argument('--window-size', type=int, default=1000, help='Window flank size (default: 1000)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading counts from {args.counts}...")
    counts_df = pd.read_csv(args.counts)
    
    # Truncate counts at 98th percentile like the notebook
    counts_df['truncated_actual_count'] = counts_df.groupby('mut_type')['actual_count'].transform(
        lambda x: x.clip(upper=x.quantile(0.98))
    )
    
    # Add site column (same as nt_site)
    counts_df['site'] = counts_df['nt_site']
    
    counts_var = 'actual_count'
    window_flank_size = args.window_size
    print(f'Window size: {window_flank_size * 2 + 1}')
    
    mut_types = sorted(list(counts_df['mut_type'].unique()))
    
    for (mut_type, mut_type_data) in counts_df.groupby('mut_type'):
        print(f"Plotting {mut_type}...")
        
        # Plot data for syn, nonsyn, stop muts
        fig, axs = plt.subplots(
            nrows=3, ncols=2, figsize=[7, 5], sharex='col', sharey='row',
            height_ratios=[1, 3, 3], width_ratios=[7, 1]
        )
        axs = axs.reshape(-1)
        max_window_median = 0
        
        for (i, (mut_class, data)) in enumerate(mut_type_data.groupby('mut_class')):
            if mut_class in ['noncoding']:
                continue
            
            # Compute sliding-window means
            window_dict = defaultdict(list)
            for site in data['site']:
                data_i = data[data['site'].between(site - window_flank_size, site + window_flank_size)]
                window_dict['site'].append(site)
                if len(data_i) == 0:
                    window_dict[counts_var].append(np.nan)
                else:
                    window_dict[counts_var].append(data_i[counts_var].mean())
            
            rolling_data = pd.DataFrame(window_dict)
            rolling_data['mut_type'] = mut_type
            rolling_data['mut_class'] = mut_class
            
            if rolling_data[counts_var].max() > max_window_median:
                max_window_median = rolling_data[counts_var].max()
            
            # Plot sliding-window means and scatter plot
            sns.lineplot(
                x='site', y=counts_var, data=rolling_data, ax=axs[2],
                color=mut_class_colors[mut_class], lw=2
            )
            
            if mut_class == 'stop':
                zorder = 10
                alpha = 1
            else:
                zorder = 1
                alpha = 0.5
            
            if mut_class in ['synonymous']:
                sns.scatterplot(
                    x='site', y=counts_var, data=data, ax=axs[4],
                    alpha=alpha, color=mut_class_colors[mut_class], zorder=zorder
                )
        
        # Format axes
        axs[4].set(ylabel='synonymous\nmutation counts', xlabel='site')
        axs[2].set(ylabel='sliding-window\nmean (2 kb)')
        axs[2].grid()
        
        axs[0].set(ylim=[-0.55, 1.25], yticks=[])
        sns.despine(left=True, bottom=True, ax=axs[0])
        axs[0].xaxis.set_tick_params(which='both', bottom=False, labelbottom=False, labeltop=False)
        
        # Hide unused axes
        for ax in [axs[1], axs[3], axs[5]]:
            ax.set_visible(False)
        
        plt.suptitle(mut_type[0] + r'$\rightarrow$' + mut_type[1], y=0.98)
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'genome_dist_{mut_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Done!")


if __name__ == '__main__':
    main()
