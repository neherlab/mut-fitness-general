"""File to load and read counts for specific mutation types"""

import pandas as pd


def load_synonymous_muts(
    df_path,
    include_noncoding_muts=False,
):
    # Read in curated dataset
    df = pd.read_csv(df_path)

    # Create masks for synonymous mutations, non-coding mutations, stop-codon tolerant ORFs, and ORF9b
    mask_synonymous = (df["synonymous"] == True).values
    mask_noncoding = (df["noncoding"] == True).values

    # Apply masks
    if include_noncoding_muts:
        mask_synonymous = mask_synonymous + mask_noncoding

    return df[mask_synonymous]


def load_nonsynonymous_muts(df_path):
    # Read in curated dataset
    df = pd.read_csv(df_path)

    # Create masks for synonymous mutations, non-coding mutations, stop-codon tolerant ORFs, and ORF9b
    mask_nonsynonymous = (df.mut_class == "nonsynonymous").values

    return df[mask_nonsynonymous]


def load_nonexcluded_muts(df_path):
    # Read in curated dataset
    df = pd.read_csv(df_path)

    return df
