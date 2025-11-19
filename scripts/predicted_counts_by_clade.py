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
    
    args = parser.parse_args()
    
    # Load dataframes
    print(f"Loading clade-wise counts from {args.counts_df}")
    counts_by_clade = pd.read_csv(args.counts_df, low_memory=False)
    
    print(f"Loading curated counts from {args.curated_counts_df}")
    counts = load.load_synonymous_muts(args.curated_counts_df)
    
    # Populate with predicted counts
    print("Adding predicted counts...")
    rates.add_predicted_count_all_clades(counts, counts_by_clade)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    counts_by_clade.to_csv(args.output, index=False)
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()

