#!/usr/bin/env python3
"""
Generate interactive plots of amino acid mutational fitness.

This script creates interactive Altair plots showing fitness effects
of amino acid mutations across different clusters of clades.
"""

import argparse
import ast
import json
import os
import pandas as pd
import altair as alt
import Bio.Seq


def clade_label(clade, clade_synonyms):
    """Get clade label with synonym if available."""
    if clade in clade_synonyms:
        return f"{clade} ({clade_synonyms[clade]})"
    else:
        return clade


def cluster_founder_map(cluster, cluster_founder):
    """Map cluster name to its founder clade."""
    if cluster in list(cluster_founder.keys()):
        return cluster_founder[cluster]
    else:
        print(f"Cluster {cluster} not defined\n")
        return None


def plot_fit_scatters(corr_df_tidy, min_predicted_count):
    """Plot set of correlation scatters."""
    subsets = corr_df_tidy["subset"].unique()
    genes = corr_df_tidy["gene"].unique()

    corr_ref = pd.merge(
        *[
            corr_df_tidy.pivot_table(
            index=["gene", "aa_mutation"],
            values=prop,
            columns="subset",
            )
            .reset_index()
            .rename(columns={subset: f"{prop} {subset}" for subset in subsets})
            for prop in ["delta_fitness", "predicted_count"]
        ]
    )

    corr_naive = pd.merge(
        *[
            corr_df_tidy.pivot_table(
            index=["gene", "aa_mutation"],
            values=prop,
            columns="subset",
            )
            .reset_index()
            .rename(columns={subset: f"{prop} {subset}" for subset in subsets})
            for prop in ["naive_delta_fitness", "expected_count"]
        ]
    )
    
    corr_df_wide = pd.merge(corr_ref, corr_naive)
    
    delta_fitness_min = corr_df_tidy["delta_fitness"].min()
    delta_fitness_max = corr_df_tidy["delta_fitness"].max()
    naive_fitness_min = corr_df_tidy["naive_delta_fitness"].min()
    naive_fitness_max = corr_df_tidy["naive_delta_fitness"].max()
    fitness_min = min(delta_fitness_min, naive_fitness_min)
    fitness_max = max(delta_fitness_max, naive_fitness_max)

    gene_selection = alt.selection_multi(
        fields=["gene"], bind="legend",
    )

    predicted_count_selection = alt.selection_single(
        bind=alt.binding_range(
            min=1,
            max=min(5 * min_predicted_count, corr_df_tidy["predicted_count"].quantile(0.9)),
            step=1,
            name="minimum predicted count",
        ),
        fields=["cutoff"],
        init={"cutoff": min_predicted_count},
    )

    expected_count_selection = alt.selection_single(
        bind=alt.binding_range(
            min=1,
            max=min(5 * min_predicted_count, corr_df_tidy["expected_count"].quantile(0.9)),
            step=1,
            name="minimum expected count",
        ),
        fields=["cutoff"],
        init={"cutoff": min_predicted_count},
    )
    
    highlight = alt.selection_single(
        on="mouseover",
        fields=["gene", "aa_mutation"],
        empty="none",
    )
    
    corr_charts = []
    base_chart = alt.Chart(corr_df_wide)
    for subset in subsets:
        base = (
            base_chart
            .encode(
                x=alt.X(
                    f"naive_delta_fitness {subset}",
                    title=f"{subset} naive fitness effect",
                    scale=alt.Scale(domain=(fitness_min,fitness_max), nice=False),
                ),
                y=alt.Y(
                    f"delta_fitness {subset}",
                    title=f"{subset} fitness effect",
                    scale=alt.Scale(domain=(fitness_min, fitness_max), nice=False),
                ),
                tooltip=[
                    "gene",
                    "aa_mutation",
                    alt.Tooltip(
                        f"delta_fitness {subset}", title=f"{subset} fitness effect",
                    ),
                    alt.Tooltip(
                        f"naive_delta_fitness {subset}", title=f"{subset} naive fitness effect",
                    ),
                    f"predicted_count {subset}",
                    f"expected_count {subset}",
                ],
            )
            .mark_circle(opacity=0.3)
            .properties(width=200, height=200)
            .transform_filter(gene_selection)
            .transform_filter(
                (alt.datum[f"predicted_count {subset}"] >= predicted_count_selection["cutoff"] - 1e-6)
                & (alt.datum[f"expected_count {subset}"] >= expected_count_selection["cutoff"] - 1e-6)
            )
        )
    
        scatter = (
            base
            .encode(
                color=alt.Color(
                    "gene",
                    scale=alt.Scale(
                        domain=genes,
                        range=["#5778a4"] * len(genes),
                    ),
                    legend=alt.Legend(
                        symbolOpacity=1,
                        orient="bottom",
                        title="click / shift-click to select specific genes to show",
                        titleLimit=500,
                        columns=6,
                    ),
                ),
                size=alt.condition(highlight, alt.value(85), alt.value(30)),
                opacity=alt.condition(highlight, alt.value(1), alt.value(0.3)),
                strokeWidth=alt.condition(highlight, alt.value(1.5), alt.value(0)),
            )
            .mark_circle(stroke="black")
        )

        line = alt.Chart(
            pd.DataFrame({
                "x": [fitness_min, fitness_max],
                "y": [fitness_min, fitness_max]
            })
        ).mark_line(color="orange", clip=True).encode(
            x="x:Q",
            y="y:Q",
        )
    
        params_r = (
            base
            .transform_regression(
                f"delta_fitness {subset}",
                f"naive_delta_fitness {subset}",
                params=True,
            )
            .transform_calculate(
                r=alt.expr.sqrt(alt.datum["rSquared"]),
                label='"r = " + format(datum.r, ".3f")',
            )
            .mark_text(align="left", color="orange", fontWeight="bold")
            .encode(
                x=alt.value(5),
                y=alt.value(8),
                text=alt.Text("label:N"),
            )
        )
        
        # show number of points
        params_n = (
            base
            .transform_filter(
                (~alt.expr.isNaN(alt.datum[f"delta_fitness {subset}"]))
                & (~alt.expr.isNaN(alt.datum[f"naive_delta_fitness {subset}"]))
            )
            .transform_calculate(dummy=alt.datum[f"delta_fitness {subset}"])
            .transform_aggregate(n="valid(dummy)")
            .transform_calculate(label='"n = " + datum.n')
            .mark_text(align="left", color="orange", fontWeight="bold")
            .encode(
                x=alt.value(5),
                y=alt.value(20),
                text=alt.Text("label:N"),
            )
        )
    
        chart = (
            (scatter + params_r + params_n)
            .add_selection(gene_selection)
            .add_selection(predicted_count_selection)
            .add_selection(expected_count_selection)
            .add_selection(highlight)
        )
    
        corr_charts.append(chart + line)
    
    ncols = 4
    rows = []
    for i in range(0, len(corr_charts), ncols):
        rows.append(alt.hconcat(*corr_charts[i: i + ncols]))
    corr_chart = alt.vconcat(*rows).configure_axis(grid=False)
    return corr_chart


def plot_aa_fitness(gene, fitness_df, clade_founder_df, min_predicted_count, heatmap_minimal_domain, cluster_founder, clade_synonyms):
    """Plot of amino-acid fitness values."""
    
    # biochemically ordered alphabet
    aas = tuple("RKHDEQNSTYWFAILMVGPC*")
    assert set(fitness_df["amino acid"]).issubset(aas)
    
    sites = fitness_df["site"].unique().tolist()
    
    predicted_count_selection = alt.selection_single(
        bind=alt.binding_range(
            min=1,
            max=min(5 * min_predicted_count, fitness_df["predicted_count"].quantile(0.9)),
            step=1,
            name="minimum predicted count",
        ),
        fields=["cutoff"],
        init={"cutoff": min_predicted_count},
    )
   
    site_zoom_brush = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(
            stroke="gold", strokeWidth=1.5, fill="yellow", fillOpacity=0.3,
        ),
    )
        
    base = (
        alt.Chart(fitness_df)
        .encode(x=alt.X("site:O", axis=alt.Axis(labelOverlap="parity")))
        .transform_filter(
            alt.datum[f"predicted_count"] >= predicted_count_selection["cutoff"] - 1e-6
        )
    )
    
    heatmap_y = alt.Y("amino acid", sort=aas, scale=alt.Scale(domain=aas))
    heatmap_base = (
        base
        .encode(y=heatmap_y)
        .properties(width=alt.Step(12), height=alt.Step(12))
    )
    
    # background fill for missing values in heatmap, imputing dummy stat
    # to get all cells
    heatmap_bg = (
        heatmap_base
        .transform_impute(
            impute="_stat_dummy",
            key="amino acid",
            keyvals=aas,
            groupby=["site"],
            value=None,
        )
        .mark_rect(color="gray", opacity=0.25)
    )

    # Select fitness for clades cluster
    cluster_selection = alt.selection_single(
        fields=["cluster"],
        bind=alt.binding_select(
            options=fitness_df["cluster"].unique(),
            name="Cluster of clades",
        ),
        init={"cluster": fitness_df["cluster"].unique().tolist()[-1]},
    )

    # place X values at "wildtype"
    wildtype_clade_selection = alt.selection_single(
        fields=["clade"],
        bind=alt.binding_select(
            options=clade_founder_df["clade"].unique(),
            name="X denotes wildtype in",
        ),
        init={"clade": clade_label(cluster_founder_map(fitness_df["cluster"].unique().tolist()[-1], cluster_founder), clade_synonyms)},
    )
    heatmap_wildtype = (
        alt.Chart(clade_founder_df.query("site in @sites"))
        .encode(
            x=alt.X("site:O"),
            y=heatmap_y,
        )
        .mark_text(text="x", color="black")
        .add_selection(wildtype_clade_selection)
        .transform_filter(wildtype_clade_selection)
        .transform_filter(site_zoom_brush)
    )
    
    # heatmap showing non-filtered amino acids
    heatmap_aas = (
        heatmap_base
        .encode(
            color=alt.Color(
                "fitness:Q",
                legend=alt.Legend(
                    orient="bottom",
                    titleOrient="left",
                    gradientLength=150,
                    gradientStrokeColor="black",
                    gradientStrokeWidth=0.5,
                ),
                scale=alt.Scale(
                    zero=True,
                    nice=False,
                    type="linear",
                    domainMid=0,
                    domain=alt.DomainUnionWith(heatmap_minimal_domain),
                ),
            ),
            stroke=alt.value("black"),
            tooltip=[
                alt.Tooltip(c, format=".3g")
                if fitness_df[c].dtype == float
                else c
                for c in fitness_df.columns
            ],
        )
        .mark_rect()
        .add_selection(cluster_selection)
        .transform_filter(cluster_selection)
        .transform_filter(site_zoom_brush)
    )

    heatmap = (
        (heatmap_bg + heatmap_aas + heatmap_wildtype)
        .add_selection(predicted_count_selection)
        .transform_filter(site_zoom_brush)
    )
    
    # make lineplot
    site_statistics = ["mean", "max", "min"]
    site_stat = alt.selection_single(
        bind=alt.binding_radio(
            options=site_statistics,
            name="site fitness statistic",
        ),
        fields=["site fitness statistic"],
        init={"site fitness statistic": site_statistics[0]},
    )
    
    lineplot = (
        base
        .transform_filter(alt.datum["amino acid"] != "*")
        .transform_filter(cluster_selection)
        .transform_aggregate(
            **{stat: f"{stat}(fitness)" for stat in site_statistics},
            groupby=["cluster", "site"],
        )
        .transform_fold(
            site_statistics,
            ["site fitness statistic", "site fitness"],
        )
        .add_selection(site_stat)
        .add_selection(site_zoom_brush)
        .transform_filter(site_stat)
        .encode(
            y=alt.Y("site fitness:Q", axis=alt.Axis(grid=False)),
            tooltip=[
                "site",
                alt.Tooltip("site fitness:Q", format=".3g"),
                "site fitness statistic:N",
            ],
        )
        .mark_area(color="black", opacity=0.7)
        .properties(
            height=75,
            width=min(750, 12 * fitness_df["site"].nunique()),
            title=alt.TitleParams(
                "use this site plot to zoom into regions on the heat map",
                anchor="start",
                fontWeight="normal",
                fontSize=11,
            ),
        )
    )
    
    show_stop = alt.selection_single(
        fields=["_dummy"],
        bind=alt.binding_radio(
            options=["yes", "no"],
            name="show stop in magenta on top site plot",
        ),
        init={"_dummy": "no"},
    )
    
    stopplot = (
        base
        .transform_filter(cluster_selection)
        .add_selection(show_stop)
        .transform_filter(alt.datum["amino acid"] == "*")
        .transform_calculate(_dummy="'yes'")
        .transform_filter(show_stop)
        .encode(
            y=alt.Y("fitness", title="site fitness"),
            color=alt.value("#CC79A7"),
            tooltip=["site", alt.Tooltip("fitness", format=".3g", title="stop fitness")],
        )
        .mark_line(point=True, strokeWidth=0.5, strokeDash=[2, 2])
    )
    
    return (
        (alt.layer(lineplot, stopplot) & heatmap)
        .properties(
            title=alt.TitleParams(
                f"estimated fitness of amino acids for {gene} protein",
                fontSize=15,
            ),
        )
        .resolve_scale(color="independent")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive plots of amino acid mutational fitness"
    )
    parser.add_argument(
        "--aamut-by-cluster",
        required=True,
        help="Input CSV file with amino acid mutations by cluster"
    )
    parser.add_argument(
        "--clade-founder-nts",
        required=True,
        help="Input CSV file with clade founder nucleotides"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for HTML plots"
    )
    parser.add_argument(
        "--min-predicted-count",
        type=int,
        help="Minimum predicted count for filtering (overrides config)"
    )
    parser.add_argument(
        "--cluster-corr-min-count",
        type=int,
        help="Minimum count for correlation calculation (overrides config)"
    )
    parser.add_argument(
        "--heatmap-minimal-domain",
        nargs=2,
        type=float,
        help="Minimal domain for heatmap [min max] (overrides config)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml file"
    )
    
    args = parser.parse_args()
    
    # Read config file
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    clade_cluster = config['clade_cluster']
    cluster_founder = config['cluster_founder']
    clade_synonyms = config.get('clade_synonyms', {})
    
    # Use command line args if provided, otherwise use config
    min_predicted_count = args.min_predicted_count if args.min_predicted_count is not None else config.get('min_predicted_count', 1)
    cluster_corr_min_count = args.cluster_corr_min_count if args.cluster_corr_min_count is not None else config.get('cluster_corr_min_count', 1)
    if args.heatmap_minimal_domain:
        heatmap_minimal_domain = args.heatmap_minimal_domain
    else:
        heatmap_minimal_domain = config.get('aa_fitness_heatmap_minimal_domain', [-6, 2])
    
    # Some settings
    _ = alt.data_transformers.disable_max_rows()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dataframe with clades founder amino acids
    print(f"Reading clade founder nucleotides from {args.clade_founder_nts}")
    clade_founder_nts = pd.read_csv(args.clade_founder_nts)
    
    # codon translation table
    codon_table = {
        f"{nt1}{nt2}{nt3}": str(Bio.Seq.Seq(f"{nt1}{nt2}{nt3}").translate())
        for nt1 in "ACGT" for nt2 in "ACGT" for nt3 in "ACGT"
    }
    
    # get clade founder amino-acids
    clade_founder_aas = (
        clade_founder_nts
        .query("gene != 'noncoding'")
        [["clade", "gene", "codon", "codon_site"]]
        .drop_duplicates()
        .assign(
            gene=lambda x: x["gene"].astype(str).str.split(";"),
            codon=lambda x: x["codon"].astype(str).str.split(";"),
            codon_site=lambda x: x["codon_site"].astype(str).str.split(";"),
        )
        .explode(["gene", "codon", "codon_site"])
        .assign(
            aa=lambda x: x["codon"].map(codon_table),
            codon_site=lambda x: x["codon_site"].astype(int),
        )
        .rename(columns={"codon_site": "site", "aa": "amino acid"})
        .drop(columns="codon")
    )
    
    # Read-in input dataframes
    print(f"Reading amino acid mutations from {args.aamut_by_cluster}")
    aamut = pd.read_csv(args.aamut_by_cluster)
    
    clust_fnd = list(cluster_founder.values())
    clust_founder_aas = clade_founder_aas.query("clade in @clust_fnd")
    
    # Adding clade columns to `aamut`
    aamut = aamut.assign(clade=lambda x: x['cluster'].map(lambda c: cluster_founder_map(c, cluster_founder)))
    
    # Retain only mutations from the cluster founder amino acids
    print(aamut['gene'].unique())
    aamut_cl_fnd = (
        aamut
        .rename(columns={'aa_site':'site', 'clade_founder_aa':'amino acid'})
        .merge(clust_founder_aas, on=['clade', 'gene', 'site', 'amino acid'], how='inner', validate='many_to_one')
        .rename(columns={'amino acid': 'ref_aa'})
        .drop(columns=['clade'])
    )
    
    # Adding Pango lineage to `clade` column in `clust_founder_aas`
    clust_founder_aas = clust_founder_aas.assign(clade=lambda x: x["clade"].map(lambda c: clade_label(c, clade_synonyms)))
    
    # Plotting
    # Scatter plot of fitness effects - Naive Vs novel fitness effects
    fit_corr_df = (
        aamut_cl_fnd
        .query("aa_mutation.str[0] != aa_mutation.str[-1]")
        [['cluster', 'gene', 'aa_mutation', 'predicted_count', 'delta_fitness', 'expected_count', 'naive_delta_fitness']]
        .assign(
            cluster_counts = lambda x: x.groupby(['cluster'])['predicted_count'].transform('sum'),
            clade = lambda x: x['cluster'].map(lambda c: clade_label(cluster_founder_map(c, cluster_founder), clade_synonyms)).str.replace(".", "_", regex=False),
            cluster = lambda x: x['cluster'].str.replace(".", "_", regex=False),
        )
        .query("cluster_counts >= @cluster_corr_min_count")
        .drop(columns="cluster_counts")
        .rename(columns={'cluster': 'subset'})
    )
    
    print("Generating fitness correlation chart...")
    fit_corr_chart = plot_fit_scatters(fit_corr_df, min_predicted_count)
    fit_corr_chart_file = os.path.join(args.output_dir, "fit_corr_chart.html")
    print(f"Saving to {fit_corr_chart_file}")
    fit_corr_chart.save(fit_corr_chart_file)
    
    # Heatmaps of mutational effects
    print("Generating heatmaps for each gene...")
    for gene, fitness_df in (
        aamut_cl_fnd
        .rename(columns={'delta_fitness': 'fitness', 'mutant_aa': 'amino acid'})
        [['cluster', 'gene', 'site', 'amino acid', 'fitness', 'predicted_count']]
        .groupby("gene")
    ):
        chart = plot_aa_fitness(
            gene, 
            fitness_df, 
            clust_founder_aas.query("gene == @gene"),
            min_predicted_count,
            heatmap_minimal_domain,
            cluster_founder,
            clade_synonyms
        )
        chartfile = os.path.join(args.output_dir, f"{gene.split()[0]}.html")
        print(f"\nSaving chart for {gene} to {chartfile}")
        chart.save(chartfile)
    
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

