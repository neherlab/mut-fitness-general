#!/usr/bin/env python3
"""
Annotate clade-wise mutation counts.

This script reads in clade founder sequences and mutation counts,
and adds metadata columns including motifs, mutation types, etc.
"""

import argparse
import os
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Annotate clade-wise mutation counts with metadata"
    )
    parser.add_argument(
        "--clade-founder",
        required=True,
        help="Input CSV file with clade founder sequences"
    )
    parser.add_argument(
        "--counts",
        required=True,
        help="Input CSV file with mutation counts"
    )
    parser.add_argument(
        "--rna-struct",
        required=False,
        default=None,
        help="Input file with RNA structure predictions (optional)"
    )
    parser.add_argument(
        "--founder-output",
        required=True,
        help="Output CSV file for annotated clade founders"
    )
    parser.add_argument(
        "--counts-output",
        required=True,
        help="Output CSV file for annotated counts"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist"
    )
    
    args = parser.parse_args()
    
    # Read in clade founder sequences
    print(f"Reading clade founders from {args.clade_founder}")
    founder_df = pd.read_csv(args.clade_founder)
    founder_df.sort_values(['clade', 'site'], inplace=True)
    
    # Get founder sequences
    founder_seq_dict = {}
    for (clade, data) in founder_df.groupby('clade'):
        founder_seq_dict[clade] = ''.join(data['nt'])
    
    # For each row, get the site's 3mer motif in the corresponding founder sequence
    def get_motif(site, clade):
        founder_seq = founder_seq_dict[clade]
        return founder_seq[site-2:site+1]
    
    min_and_max_sites = [founder_df['site'].min(), founder_df['site'].max()]
    founder_df['motif'] = founder_df.apply(
        lambda row: np.nan if row['site'] in min_and_max_sites \
            else get_motif(row['site'], row['clade']),
        axis=1
    )
    
    # Add columns giving the reference codon and motif
    founder_df = founder_df.merge(
        (
            founder_df[founder_df['clade'] == '19A']
            .rename(columns={'codon' : 'ref_codon', 'motif' : 'ref_motif'})
        )[['site', 'ref_codon', 'ref_motif']], on='site', how='left'
    )
    
    founder_df.rename(columns={'site': 'nt_site'}, inplace=True)
    
    # Save annotated founder dataframe
    if args.overwrite or not os.path.isfile(args.founder_output):
        os.makedirs(os.path.dirname(args.founder_output) if os.path.dirname(args.founder_output) else '.', exist_ok=True)
        founder_df.to_csv(args.founder_output, index=False)
        print(f"Saved annotated clade founders to {args.founder_output}")
    else:
        print(f"Output file {args.founder_output} already exists. Use --overwrite to overwrite.")
    
    # Read in and annotate counts data
    print(f"Reading counts from {args.counts}")
    counts_df = pd.read_csv(args.counts)
    
    # Filter by subset if column exists
    if 'subset' in counts_df.columns:
        counts_df = counts_df.query("subset == 'all'")
    
    # Add metadata
    counts_df[['wt_nt', 'mut_nt']] = counts_df['nt_mutation'].str.extract(r'(\w)\d+(\w)')
    counts_df['mut_type'] = counts_df['wt_nt'] + counts_df['mut_nt']
    
    def get_mut_class(row):
        if row['synonymous']:
            return 'synonymous'
        elif row['noncoding']:
            return 'noncoding'
        elif '*' in row['mutant_aa']:
            return 'nonsense'
        elif row['mutant_aa'] != row['clade_founder_aa']:
            return 'nonsynonymous'
        else:
            raise ValueError(row['mutant_aa'], row['clade_founder_aa'])
    
    counts_df['mut_class'] = counts_df.apply(lambda row: get_mut_class(row), axis=1)
    
    # Add column indicating if clade is pre-Omicron or Omicron
    pre_omicron_clades = [
        '20A', '20B', '20C', '20E', '20G', '20H', '20I', '20J', '21C','21I', '21J'
    ]
    counts_df['pre_omicron_or_omicron'] = counts_df['clade'].apply(
        lambda x: 'pre_omicron' if x in pre_omicron_clades else 'omicron'
    )
    
    # Add column indicating if a site is before the light switch boundary
    def light_switch(mut, site, lb1=13467, lb2=21562):
        if mut in ["AT", "CG", "GC"]:
            pos_bool = True if site < lb2 else False
        elif mut == "CT":
            pos_bool = True if site < lb1 else False
        else:
            pos_bool = False
        return pos_bool
    
    counts_df['nt_site_before_boundary'] = counts_df.apply(lambda x: light_switch(x.mut_type, x.nt_site), axis=1)
    
    # Add column indicating whether RNA sites are predicted to be paired
    if args.rna_struct:
        with open(args.rna_struct) as f:
            lines = [line.rstrip().split() for line in f]
        paired = np.array([[int(x[0]),int(x[4])] for x in lines[1:]])
        paired_dict = dict(zip(paired[:,0], paired[:,1]))
        
        def assign_ss_pred(site):
            if site not in paired_dict:
                return 'nd'
            elif paired_dict[site] == 0:
                return 'unpaired'
            else:
                return 'paired'
        
        counts_df['ss_prediction'] = counts_df['nt_site'].apply(lambda x: assign_ss_pred(x))
        counts_df['unpaired'] = counts_df['ss_prediction'].apply(lambda x: 1 if x == 'unpaired' else 0)
    else:
        # If no RNA structure file provided, mark all as unpaired=0 (paired)
        counts_df['ss_prediction'] = 'nd'
        counts_df['unpaired'] = 0
    
    # Add columns giving a site's motif relative to the clade founder and the reference sequence
    if 'motif' not in counts_df.columns or 'ref_motif' not in counts_df.columns:
        counts_df = counts_df.merge(
            founder_df[['nt_site', 'clade', 'motif', 'ref_motif']],
            on = ['nt_site', 'clade'], how='left',
        )
    
    # Assign motif to genome edges using actual sequence context from codons
    max_site = counts_df['nt_site'].max()
    
    # First site: use "A" + first two nucleotides of codon
    counts_df.loc[counts_df.nt_site == 1, ["motif", "ref_motif"]] = counts_df.loc[
        counts_df.nt_site == 1, "clade_founder_codon"
    ].apply(lambda x: "A" + x[:2])
    
    # Last site: use last two nucleotides of codon + "A"
    counts_df.loc[counts_df.nt_site == max_site, ["motif", "ref_motif"]] = counts_df.loc[
        counts_df.nt_site == max_site, "clade_founder_codon"
    ].apply(lambda x: x[1:] + "A")
    
    # Save to file
    if 'subset' in counts_df.columns:
        counts_df.drop(columns=['subset'], inplace=True)
    if args.overwrite or not os.path.isfile(args.counts_output):
        os.makedirs(os.path.dirname(args.counts_output) if os.path.dirname(args.counts_output) else '.', exist_ok=True)
        counts_df.to_csv(args.counts_output, index=False)
        print(f"Saved annotated counts to {args.counts_output}")
    else:
        print(f"Output file {args.counts_output} already exists. Use --overwrite to overwrite.")


if __name__ == "__main__":
    main()

