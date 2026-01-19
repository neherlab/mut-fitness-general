#!/usr/bin/env python
"""
Add RNA secondary structure annotations to mutation counts file.
"""

import pandas as pd
import argparse

def parse_rna_structure(rna_file):
    """
    Parse RNA structure file and extract unpaired status for each position.
    
    Expected format: Two-column file (tab or comma separated)
    position    unpaired
    1           0
    2           1
    3           1
    ...
    
    Where 0 = paired (low SHAPE reactivity), 1 = unpaired (high SHAPE reactivity)
    """
    structure = {}
    
    with open(rna_file, 'r') as f:
        # Skip header line
        _ = f.readline().strip()
        
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Handle both tab and comma separators
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split(',')
            
            if len(parts) >= 2:
                pos = int(parts[0])
                unpaired = int(parts[1])
                structure[pos] = unpaired
    
    return structure

def main():
    parser = argparse.ArgumentParser(
        description='Add RNA secondary structure to mutation counts'
    )
    parser.add_argument(
        '--counts',
        required=True,
        help='Input mutation counts CSV file'
    )
    parser.add_argument(
        '--structure',
        required=True,
        help='RNA structure file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file with RNA structure'
    )
    
    args = parser.parse_args()
    
    # Parse RNA structure
    print(f"Reading RNA structure from {args.structure}")
    structure = parse_rna_structure(args.structure)
    print(f"Loaded structure for {len(structure)} positions")
    print(f"Paired sites: {sum(1 for v in structure.values() if v == 0)}")
    print(f"Unpaired sites: {sum(1 for v in structure.values() if v == 1)}")
    
    # Load mutation counts
    print(f"\nReading mutation counts from {args.counts}")
    df = pd.read_csv(args.counts)
    print(f"Loaded {len(df)} mutations")
    
    # Add unpaired column based on nt_site
    df['unpaired'] = df['nt_site'].map(structure)
    
    # Update ss_prediction column based on unpaired status
    df['ss_prediction'] = df['unpaired'].map({0: 'paired', 1: 'unpaired'})
    
    # Check for missing values
    missing = df['unpaired'].isna().sum()
    if missing > 0:
        print(f"\nWarning: {missing} mutations have no structure annotation")
        print("Setting missing values to unpaired (1)")
        df['unpaired'] = df['unpaired'].fillna(1)
        df['ss_prediction'] = df['ss_prediction'].fillna('unpaired')
    
    # Report statistics
    print(f"\nRNA structure statistics:")
    print(df['ss_prediction'].value_counts())
    
    # Save output
    print(f"\nSaving to {args.output}")
    df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
