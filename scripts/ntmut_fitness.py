#!/usr/bin/env python3
"""
Compute probabilistic fitness estimates for nucleotide mutations.

This script calculates estimates of the fitness effects of each nucleotide mutation
using a Bayesian probabilistic framework.
"""

import argparse
import os
import sys
import pandas as pd

# Add module folder to system paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import module for probabilistic fitness estimates
from modules import probfit


def main():
    parser = argparse.ArgumentParser(
        description="Compute probabilistic fitness estimates for nucleotide mutations"
    )
    parser.add_argument(
        "--cluster-counts",
        required=True,
        help="Input CSV file with cluster-specific mutation counts"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file with fitness estimates"
    )
    parser.add_argument(
        "--N-f",
        type=int,
        default=300,
        help="Number of fitness values to sample (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Read-in cluster specific counts dataframe
    print(f"Reading cluster counts from {args.cluster_counts}")
    counts_df = pd.read_csv(args.cluster_counts)
    
    # Compute posterior fitness estimates
    print(f"Computing probabilistic fitness estimates (N_f={args.N_f})...")
    probfit.add_probabilistic_estimates(counts_df, N_f=args.N_f)
    
    # Write to file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    counts_df.to_csv(args.output, index=False)
    print(f"Saved fitness estimates to {args.output}")


if __name__ == "__main__":
    main()

