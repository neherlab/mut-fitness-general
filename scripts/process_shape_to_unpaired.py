#!/usr/bin/env python3
"""
Convert SHAPE-MaP Norm_profile values to binary unpaired status for RNA structure.

Input: shapemapper-output.txt files with Norm_profile column
Output: Tab-separated files with position and unpaired columns
"""

import pandas as pd
import numpy as np
import argparse
import os


def process_shape_file(input_file, output_file, threshold=0.4):
    """
    Convert SHAPE Norm_profile to binary unpaired status.
    
    Args:
        input_file: Path to shapemapper-output.txt file
        output_file: Path to output file with position and unpaired columns
        threshold: SHAPE reactivity threshold (default 0.4)
                   Values > threshold are unpaired (1), <= threshold are paired (0)
    """
    print(f"\nProcessing {input_file}...")
    
    # Read the SHAPE file
    df = pd.read_csv(input_file, sep='\t')
    
    # Extract position and Norm_profile
    structure_df = df[['Nucleotide', 'Norm_profile']].copy()
    structure_df.columns = ['position', 'norm_profile']
    
    # Convert Norm_profile to binary unpaired
    # High SHAPE reactivity (> threshold) = unpaired (1)
    # Low SHAPE reactivity (<= threshold) = paired (0)
    # NaN values = set to unpaired (1) as default
    structure_df['unpaired'] = structure_df['norm_profile'].apply(
        lambda x: 1 if pd.isna(x) or x > threshold else 0
    )
    
    # Create output with position and unpaired columns
    output_df = structure_df[['position', 'unpaired']]
    
    # Save to file
    output_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"  Total positions: {len(output_df)}")
    print(f"  Paired: {(output_df['unpaired'] == 0).sum()}")
    print(f"  Unpaired: {(output_df['unpaired'] == 1).sum()}")
    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SHAPE Norm_profile to unpaired status'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input shapemapper-output.txt file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output file with position and unpaired columns'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.4,
        help='SHAPE reactivity threshold for unpaired classification (default: 0.4)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    process_shape_file(args.input, args.output, args.threshold)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
