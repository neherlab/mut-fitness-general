#!/usr/bin/env python3
"""
Plot histogram of synonymous mutation counts by RNA structure (paired/unpaired) and mutation type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def plot_counts_histogram(df, output_file):
    """
    Create stacked histogram of mutation counts by structure and mutation type.
    """
    # Extract mutation type (e.g., A→C from A123C)
    df['mut_type'] = df['nt_mutation'].str[0] + '→' + df['nt_mutation'].str[-1]
    
    # Filter to synonymous mutations (assuming this is already done, but could add checks)
    
    # Create figure (4 rows x 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), sharex=True)
    axes = axes.flatten()
    
    # Get unique mutation types sorted
    mut_types = sorted(df['mut_type'].unique())
    
    # Define bin edges for counts (log scale, excluding zero)
    max_count = df[df['actual_count'] > 0]['actual_count'].max()
    bins = np.logspace(0, np.log10(max_count + 1), 30)
    
    for idx, mut_type in enumerate(mut_types):
        ax = axes[idx]
        
        # Get data for this mutation type
        mut_data = df[df['mut_type'] == mut_type]
        
        # Separate by paired/unpaired
        paired_data = mut_data[mut_data['unpaired'] == 0]
        unpaired_data = mut_data[mut_data['unpaired'] == 1]
        
        # Get non-zero counts for histogram
        paired_counts_nonzero = paired_data[paired_data['actual_count'] > 0]['actual_count']
        unpaired_counts_nonzero = unpaired_data[unpaired_data['actual_count'] > 0]['actual_count']
        
        # Calculate statistics including zeros
        n_paired = len(paired_data)
        n_unpaired = len(unpaired_data)
        n_paired_zero = (paired_data['actual_count'] == 0).sum()
        n_unpaired_zero = (unpaired_data['actual_count'] == 0).sum()
        frac_paired_zero = n_paired_zero / n_paired if n_paired > 0 else 0
        frac_unpaired_zero = n_unpaired_zero / n_unpaired if n_unpaired > 0 else 0
        mean_paired = paired_data['actual_count'].mean() if n_paired > 0 else 0
        mean_unpaired = unpaired_data['actual_count'].mean() if n_unpaired > 0 else 0
        
        # Create stacked histogram (only non-zero counts)
        ax.hist([paired_counts_nonzero, unpaired_counts_nonzero], 
                bins=bins,
                label=['Paired', 'Unpaired'],
                stacked=True,
                color=['#0072B2', '#E69F00'],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5)
        
        ax.set_xscale('log')
        ax.set_title(f'{mut_type}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        ax.text(0.05, 0.95, 
                f'Paired: n={n_paired}, μ={mean_paired:.1f}, {frac_paired_zero:.1%} zeros\n'
                f'Unpaired: n={n_unpaired}, μ={mean_unpaired:.1f}, {frac_unpaired_zero:.1%} zeros',
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        if idx >= 9:  # Bottom row
            ax.set_xlabel('Actual count (log scale, zeros excluded)', fontsize=10)
    
    # Add legend to first plot
    axes[0].legend(loc='upper right', fontsize=10)
    
    plt.suptitle('Synonymous Mutation Counts by RNA Structure and Mutation Type', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def plot_counts_boxplot(df, output_file):
    """
    Create boxplot comparison of mutation counts by structure and mutation type.
    """
    df['mut_type'] = df['nt_mutation'].str[0] + '→' + df['nt_mutation'].str[-1]
    
    # Filter out zero counts for better visualization
    df_nonzero = df[df['actual_count'] > 0].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create boxplot
    sns.boxplot(data=df_nonzero, 
                x='mut_type', 
                y='actual_count',
                hue='ss_prediction',
                palette=['#0072B2', '#E69F00'],
                ax=ax,
                showfliers=False)
    
    ax.set_yscale('log')
    ax.set_xlabel('Mutation Type', fontsize=12)
    ax.set_ylabel('Actual Count (log scale)', fontsize=12)
    ax.set_title('Synonymous Mutation Counts by RNA Structure', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='RNA Structure', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot synonymous mutation counts by RNA structure'
    )
    parser.add_argument(
        '--counts',
        required=True,
        help='Input mutation counts CSV file with unpaired column'
    )
    parser.add_argument(
        '--output-hist',
        default='rna_structure_counts_histogram.pdf',
        help='Output file for histogram'
    )
    parser.add_argument(
        '--output-box',
        default='rna_structure_counts_boxplot.pdf',
        help='Output file for boxplot'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.counts}...")
    df = pd.read_csv(args.counts)
    
    # Check for required columns
    required_cols = ['nt_mutation', 'actual_count', 'unpaired', 'ss_prediction']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} mutations")
    print(f"Paired sites: {(df['unpaired'] == 0).sum()}")
    print(f"Unpaired sites: {(df['unpaired'] == 1).sum()}")
    
    # Create plots
    print("\nCreating histogram...")
    plot_counts_histogram(df, args.output_hist)
    
    print("\nCreating boxplot...")
    plot_counts_boxplot(df, args.output_box)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
