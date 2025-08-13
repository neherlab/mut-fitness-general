"""Module for converting nucleotide fitness to amino acid ones"""

import pandas as pd
import numpy as np


# Getting only mutations at coding sites
def get_coding(df, gene_overlaps, explode_cols):
    overlaps_to_retain = [set(overlap) for overlap in gene_overlaps["retain"]]
    overlaps_to_exclude = [set(overlap) for overlap in gene_overlaps["exclude"]]

    def exclude_overlap(gene):
        if ";" not in gene:
            return False
        genes = set(gene.split(";"))
        if genes in overlaps_to_exclude:
            return True
        elif genes in overlaps_to_retain:
            return False
        else:
            raise ValueError(
                f"not specified how to handle overlap of {genes=} in {gene=}"
            )

    df_coding = (
        df.query("not noncoding")
        .assign(
            is_overlapping=lambda x: x["codon_site"]
            .str.split(";")
            .map(lambda s: len(set(s)) > 1),
            overlap_to_exclude=lambda x: (
                x["is_overlapping"] & x["gene"].map(exclude_overlap)
            ),
        )
        .query("not overlap_to_exclude")
    )

    df_coding["clade_founder_aa"] = df_coding["aa_mutation"].apply(
        lambda x: x[0]
        if len(x.split(";")) == 1
        else ";".join(map(lambda y: y[0], x.split(";")))
    )
    df_coding["mutant_aa"] = df_coding["aa_mutation"].apply(
        lambda x: x[-1]
        if len(x.split(";")) == 1
        else ";".join(map(lambda y: y[-1], x.split(";")))
    )

    for col in explode_cols:
        df_coding[col] = df_coding[col].str.split(";")

    df_coding_exp = df_coding.explode(explode_cols).query("gene != 'ORF1a'")

    return df_coding_exp


def aggregate_counts(df, explode_cols):
    df_agg = (
        df.groupby(["cluster", *explode_cols], as_index=False)
        .aggregate(
            expected_count=pd.NamedAgg("expected_count", "sum"),
            predicted_count=pd.NamedAgg("predicted_count", "sum"),
            actual_count=pd.NamedAgg("actual_count", "sum"),
            tau_squared=pd.NamedAgg("tau_squared", "sum"),
        )
        .rename(columns={"codon_site": "aa_site"})
        .assign(
            aa_site=lambda x: x["aa_site"].astype(int),
        )
    )

    assert (df_agg["clade_founder_aa"] == df_agg["aa_mutation"].str[0]).all()

    return df_agg


def map_orf1ab_to_nsps(orf1ab_to_nsps):
    orf1ab_to_nsps_df = pd.concat(
        [
            pd.DataFrame(
                [(i, i - start + 1) for i in range(start, end + 1)],
                columns=["ORF1ab_site", "nsp_site"],
            )
            .assign(nsp=nsp)
            .drop_duplicates()
            for nsp, (start, end) in orf1ab_to_nsps.items()
        ],
        ignore_index=True,
    )

    return orf1ab_to_nsps_df


def add_nsps(df, orf1ab_to_nsps_df):
    df_nsps = pd.concat(
        [
            df.assign(subset_of_ORF1ab=False),
            (
                df.query("gene == 'ORF1ab'")
                .merge(
                    orf1ab_to_nsps_df,
                    left_on="aa_site",
                    right_on="ORF1ab_site",
                    validate="many_to_one",
                )
                .drop(columns=["gene", "aa_mutation", "aa_site", "ORF1ab_site"])
                .rename(columns={"nsp": "gene", "nsp_site": "aa_site"})
                .assign(
                    aa_mutation=lambda x: (
                        x["clade_founder_aa"]
                        + x["aa_site"].astype(str)
                        + x["mutant_aa"]
                    ),
                    subset_of_ORF1ab=True,
                )
            ),
        ],
        ignore_index=True,
    )

    return df_nsps


def naive_fitness(aa_counts, fitness_pseudocount=0.5):
    aa_counts["naive_delta_fitness"] = np.log(
        (aa_counts["actual_count"] + fitness_pseudocount)
        / (aa_counts["expected_count"] + fitness_pseudocount)
    )


def aa_fitness(df, explode_cols):
    # Compute the weighted average for each group
    # Here we create new columns for the weights and weighted values
    df["weight"] = 1 / df["f_st_dev"] ** 2
    df["weighted_value"] = df["f_mean"] * df["weight"]

    # Sum the weights and weighted values for each group
    grouped_sums = df.groupby(["cluster", *explode_cols]).sum()

    # Compute the weighted average
    grouped_sums["delta_fitness"] = (
        grouped_sums["weighted_value"] / grouped_sums["weight"]
    )
    grouped_sums["uncertainty"] = np.sqrt(1 / grouped_sums["weight"])

    # Extract the final result
    aa_fit = grouped_sums[["delta_fitness", "uncertainty"]].reset_index()

    aa_fit = aa_fit.rename(columns={"codon_site": "aa_site"}).assign(
        aa_site=lambda x: x["aa_site"].astype(int)
    )

    return aa_fit


def merge_aa_df(aa_fit, aa_counts, explode_cols):
    # Assert counts and fitness dataframes have the same size
    assert aa_counts.shape[0] == aa_fit.shape[0]

    explode_cols[3] = "aa_site"

    aamut_fitness = pd.merge(
        aa_counts,
        aa_fit,
        on=["cluster", *explode_cols],
        how="inner",
    )

    return aamut_fitness
