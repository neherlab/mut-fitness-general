#!/usr/bin/env python3
"""
Estimate probabilistic fitness by aggregating counts for clades in cluster.

This script creates tables for each subset of sequences (clades, groups of clades, etc.)
that contain the actual and predicted counts.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate mutation counts for clades in a cluster"
    )
    parser.add_argument(
        "--counts-df",
        required=True,
        help="Input CSV file with clade-wise mutation counts"
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="Name of the cluster"
    )
    parser.add_argument(
        "--clades",
        required=True,
        nargs='+',
        help="List of clades to include in the cluster"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file for cluster counts"
    )
    
    args = parser.parse_args()
    
    # Import cladewise mutations table
    print(f"Reading mutation counts from {args.counts_df}")
    muts_by_clade = pd.read_csv(args.counts_df, low_memory=False)
    
    # Ignore sites that are annotated as being masked or excluded
    muts_by_clade = muts_by_clade.query('not exclude').query('not masked_in_usher')
    
    # Aggregate counts for clades in cluster
    group_cols = ['nt_mutation', 'gene', 'codon_site', 'aa_mutation', 'synonymous', 'noncoding']
    nucleotides = ['A', 'C', 'G', 'T']
    
    print(f"Aggregating counts for cluster '{args.cluster}' with clades: {args.clades}")
    muts_by_clade_cluster = (muts_by_clade
        .query("clade.isin(@args.clades)")              # Selecting clades
        .groupby(group_cols, as_index=False)    # Columns not be aggregated
        .aggregate(                             # Aggregating counts
            expected_count = pd.NamedAgg('expected_count', 'sum'),
            predicted_count = pd.NamedAgg('predicted_count', 'sum'),
            actual_count = pd.NamedAgg('actual_count', 'sum'),
            tau_squared = pd.NamedAgg('tau_squared', 'mean')
        )
    )
    muts_by_clade_cluster.insert(0, 'nt_site', muts_by_clade_cluster['nt_mutation'].apply(lambda x: int(x[1:-1])))
    muts_by_clade_cluster.insert(0, 'cluster', args.cluster)
    muts_by_clade_cluster['wt'] = pd.CategoricalIndex(muts_by_clade_cluster['nt_mutation'].apply(lambda x: x[0]), ordered=True, categories=nucleotides)
    muts_by_clade_cluster['mut'] = pd.CategoricalIndex(muts_by_clade_cluster['nt_mutation'].apply(lambda x: x[-1]), ordered=True, categories=nucleotides)
    muts_by_clade_cluster = muts_by_clade_cluster.sort_values(['nt_site', 'wt', 'mut']).reset_index(drop=True) # Ordering by site, wildtype and mutant nucleotides
    muts_by_clade_cluster.drop(columns=['wt', 'mut'], inplace=True)
    
    # Save dataframe
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    muts_by_clade_cluster.to_csv(args.output, index=False)
    print(f"Saved cluster counts to {args.output}")


if __name__ == "__main__":
    main()

