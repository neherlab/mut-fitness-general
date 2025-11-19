#!/usr/bin/env python3
"""
Curate mutation counts dataset.

This script creates a training dataset to infer the General Linear Model for mutations
by filtering and aggregating mutation counts.
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Curate mutation counts dataset for GLM training"
    )
    parser.add_argument(
        "--mut-counts",
        required=True,
        help="Input CSV file with mutation counts by clade"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file for curated counts"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    
    args = parser.parse_args()
    
    # Read in data
    print(f"Reading mutation counts from {args.mut_counts}")
    counts_df = pd.read_csv(args.mut_counts, low_memory=False)
    
    # Determine conserved sites
    # Identify sites where the codon and motif are conserved across all clade founders
    conserved_sites = counts_df['nt_site'].value_counts().index
    
    # Ignore sites that are annotated as being masked in any clade of the UShER tree
    # or are annotated for exclusion
    sites_to_ignore = list(counts_df[
        (counts_df['masked_in_usher'] == True) |
        (counts_df['exclude'] == True)
    ]['nt_site'].unique())
    
    # Retain only non-excluded and conserved sites
    curated_counts_df = counts_df[
        counts_df['nt_site'].isin(conserved_sites) &
        ~(counts_df['nt_site'].isin(sites_to_ignore))
    ]
    
    # Check that motifs are conserved
    assert sum(curated_counts_df['motif'] != curated_counts_df['ref_motif']) == 0
    
    # Aggregate counts across all clades
    ignore_cols = [
        'expected_count', 'actual_count', 'count_terminal', 'count_non_terminal', 'mean_log_size',
        'clade', 'pre_omicron_or_omicron'
    ]
    groupby_cols = [
        col for col in curated_counts_df.columns.values
        if col not in ignore_cols
    ]
    curated = curated_counts_df.groupby(groupby_cols, as_index=False).agg('sum', numeric_only=True)
    
    # Summary statistics
    print('Number of unique muts:')
    print('In the full dataset:', len(counts_df['nt_mutation'].unique()))
    print('In the curated dataset:', len(curated['nt_mutation'].unique()))
    print('\nNumber of curated mutations per category:')
    print(curated['mut_class'].value_counts())
    
    # Drop columns for site exclusions and masking
    curated.drop(columns=['exclude', 'masked_in_usher'], inplace=True)
    
    # Write curated dataframes to file
    if args.overwrite or not os.path.isfile(args.output):
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        curated.to_csv(args.output, index=False)
        print(f"\nSaved curated counts to {args.output}")
    else:
        print(f"\nOutput file {args.output} already exists. Use --overwrite to overwrite.")


if __name__ == "__main__":
    main()

