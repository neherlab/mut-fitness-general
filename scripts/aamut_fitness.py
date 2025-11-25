#!/usr/bin/env python3
"""
Compute amino acid fitness mutations.

This script calculates estimates of the fitness effects of each amino acid substitution
by aggregating nucleotide mutation fitness effects.
"""

import argparse
import ast
import os
import sys
import pandas as pd

# Add module folder to system paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from modules import aamutfit


def main():
    parser = argparse.ArgumentParser(
        description="Compute amino acid fitness mutations"
    )
    parser.add_argument(
        "--ntmut-fit",
        required=True,
        help="Input CSV file with nucleotide mutation fitness estimates"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file with amino acid fitness estimates"
    )
    parser.add_argument(
        "--gene-overlaps-retain",
        type=str,
        help="Python dict string with gene overlap configuration. Gene overlaps to retain."
    )
    parser.add_argument(
        "--gene-overlaps-exclude",
        type=str,
        help="Python dict string with gene overlap configuration. Gene overlaps to exclude."
    )
    parser.add_argument(
        "--genes",
        nargs='+',
        required=True,
        help="List of genes in order on the viral genome"
    )
    parser.add_argument(
        "--fitness-pseudocount",
        type=float,
        default=0.5,
        help="Pseudocount for calculating amino-acid fitnesses (default: 0.5)"
    )
    
    args = parser.parse_args()
    print(args.genes)
    # Parse gene overlaps
    if args.gene_overlaps_retain and args.gene_overlaps_exclude:
        gene_overlaps_retain = ast.literal_eval(args.gene_overlaps_retain)
        gene_overlaps_exclude = ast.literal_eval(args.gene_overlaps_exclude)
        gene_overlaps = {'exclude': gene_overlaps_exclude, 'retain': gene_overlaps_retain}
    else:
        gene_overlaps = {'exclude': [], 'retain': []}
        print("Warning: No gene overlap configuration provided. Using empty lists for both exclude and retain.")
    
    # Columns to be exploded
    explode_cols = [
        "gene",
        "clade_founder_aa",
        "mutant_aa",
        "codon_site",
        "aa_mutation",
    ]
    
    # Read-in fitness of nucleotide mutations
    print(f"Reading nucleotide mutation fitness from {args.ntmut_fit}")
    ntmut_fit = pd.read_csv(args.ntmut_fit)
    
    # Get only coding mutations
    print("Extracting coding mutations...")
    ntmut_fit_coding = aamutfit.get_coding(ntmut_fit, gene_overlaps, explode_cols)
    # Aggregate counts for amino acid mutations
    print("Aggregating counts for amino acid mutations...")
    aa_counts = aamutfit.aggregate_counts(ntmut_fit_coding, explode_cols)
    # Adding naive fitness estimates
    print("Computing naive fitness estimates...")
    aamutfit.naive_fitness(aa_counts, fitness_pseudocount=args.fitness_pseudocount)
    # Dataframe with refined fitness estimates
    print("Computing refined fitness estimates...")
    aa_fit = aamutfit.aa_fitness(ntmut_fit_coding, explode_cols)
    # Merge counts and fitness dataframes
    aamut_fitness = aamutfit.merge_aa_df(aa_fit, aa_counts, explode_cols)
    # Order dataframe according to: genes order, site within the gene
    aamut_fitness['gene'] = pd.CategoricalIndex(aamut_fitness['gene'], ordered=True, categories=args.genes)
    aamut_fitness = aamut_fitness.sort_values(['gene', 'aa_site']).reset_index(drop=True)

    # Write to file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    aamut_fitness.to_csv(args.output, index=False)
    print(f"Saved amino acid fitness estimates to {args.output}")


if __name__ == "__main__":
    main()

