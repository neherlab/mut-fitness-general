#!/usr/bin/env python3
"""
Create master tables with predicted mutation rates.

This script creates tables with predicted mutation rates for each mutation
in each of its contexts (motif, pairing state, etc.).
"""

import argparse
import numpy as np
import os
import sys
import yaml

# Add module folder to system paths
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from modules import rates
from modules import load


def main():
    parser = argparse.ArgumentParser(
        description="Create master tables with predicted mutation rates"
    )
    parser.add_argument(
        "--counts",
        required=True,
        help="Input CSV file with curated counts"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file for master table"
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
    
    # Load training dataframes
    print(f"Loading curated counts from {args.counts}")
    counts = load.load_synonymous_muts(args.counts)
    
    # Initialize rates objects
    rate = rates.Rates(use_rna_structure=use_rna_structure)
    print(f"RNA structure: {'enabled' if use_rna_structure else 'disabled'}")
    
    # Populate rates and add predicted counts
    print("Populating rates...")
    rate.populate_rates(counts)
    
    rate.rates["cond_count"] = rate.genome_composition(counts)
    
    # Computing residual variance
    print("Computing predicted counts and residual variance...")
    counts['predicted_count'] = rate.predicted_counts_by_clade(counts)
    
    tau = counts.groupby("mut_type", group_keys=False).apply(
        lambda x: np.mean(
            (np.log(x.actual_count + 0.5) - np.log(x.predicted_count + 0.5)) ** 2
        )
    )
    
    rate.residual_variance(counts, tau)
    
    # Formatting master tables
    # Adding lightswitch boundaries: a placeholder for global context
    rate.rates['nt_site_boundary'] = np.zeros(rate.rates.shape[0], int)
    
    # Add nt_site_before_boundary column if it doesn't exist (for backwards compatibility)
    if 'nt_site_before_boundary' not in rate.rates.columns:
        rate.rates['nt_site_before_boundary'] = False
    
    # Save master tables - columns depend on whether RNA structure is used
    cols = ['mut_type', 'motif'] + (['unpaired'] if use_rna_structure else []) \
        + ['nt_site_boundary', 'nt_site_before_boundary', 'rate', 'predicted_count', 'residual']

        
    rate.rates.drop(columns=['condition'], inplace=True)
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    rate.rates[cols].to_csv(args.output, index=False)
    print(f"Saved master table to {args.output}")


if __name__ == "__main__":
    main()

