#!/usr/bin/env python3
"""
Add predicted counts to clade-wise dataframe.

This script adds predicted counts, based on the inferred mutation rate model,
to the table with observed mutation counts.
"""

import argparse
import os
import sys
import pandas as pd
import yaml

# Add module folder to system paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from modules import load
from modules import rates


def main():
    parser = argparse.ArgumentParser(
        description="Add predicted counts to clade-wise mutation counts"
    )
    parser.add_argument(
        "--counts-df",
        required=True,
        help="Input CSV file with clade-wise mutation counts"
    )
    parser.add_argument(
        "--curated-counts-df",
        required=True,
        help="Input CSV file with curated counts for training"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file with predicted counts added"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    use_rna_structure = config.get('use_rna_structure', True)
    
    # Load dataframes
    print(f"Loading clade-wise counts from {args.counts_df}")
    counts_by_clade = pd.read_csv(args.counts_df, low_memory=False)
    
    print(f"Loading curated counts from {args.curated_counts_df}")
    counts = load.load_synonymous_muts(args.curated_counts_df)
    
    # Populate with predicted counts
    print(f"Adding predicted counts (RNA structure: {'enabled' if use_rna_structure else 'disabled'})...")
    rates.add_predicted_count_all_clades(counts, counts_by_clade, use_rna_structure=use_rna_structure)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    counts_by_clade.to_csv(args.output, index=False)
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()

