"""
This pipeline fetches raw mutation counts
and produces refined fitness including estimates of uncertainty for each mutation
in different subsets of the total sequence availability.
"""

import yaml

configfile: "config.yaml"
with open(config["docs_plot_annotations"]) as f:
    docs_plot_annotations = yaml.safe_load(f)

rule all:
    """Target rule with desired output files."""
    input:
        'results/master_tables/master_table.csv',
        expand(
            "results/plots_for_docs/{plot}.html",
            plot=list(docs_plot_annotations["plots"]),
        ),

rule curated_counts:
    message:
        "Create training dataset to infer the General Linear Model for mutations"
    input:
        mut_counts='results/mut_counts_by_clade.csv',
        # clade_founder=rules.annotate_counts.output.founder_csv,
    output:
        outfile="results/curated/curated_mut_counts.csv",
    notebook:
        "notebook/curate_counts.py.ipynb"

rule master_table:
    message:
        "Create tables with predicted mutation rates for each mutation in each of its contexts"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        ms='results/master_tables/master_table.csv',
    notebook:
        "notebook/master_tables.py.ipynb"

rule predicted_counts:
    message:
        "Add predicted counts, based on the inferred mutation rate model, to the table with observed mutation counts."
    input:
        counts_df='results/mut_counts_by_clade.csv',
        curated_counts_df=rules.curated_counts.output.outfile,
    output:
        pred_count_csv="results/pred_mut_counts_by_clade.csv",
    notebook:
        "notebook/predicted_counts_by_clade.py.ipynb"

rule counts_cluster:
    message:
        """
        Create tables for each subset of sequences (clades, groups of clades, etc.) that contain the actual and prediced counts.
        These groups are defined in the config file as 'clade_clusters'.
        """
    params:
        cluster=lambda wc: wc.cluster,
        clades=lambda wc: config['clade_cluster'][wc.cluster],
    input:
        counts_df=rules.predicted_counts.output.pred_count_csv,
    output:
        cluster_counts=temp('results/ntmut_fitness/{cluster}_ntmut_counts.csv'),
    notebook:
        "notebook/ntmut_counts_cluster.py.ipynb"

rule ntmut_fitness:
    message:
        "Calculate estimates of the fitness effects of each nucleotide mutation."
    input:
        cluster_counts='results/ntmut_fitness/{cluster}_ntmut_counts.csv'
    output:
        ntfit_csv='results/ntmut_fitness/{cluster}_ntmut_fitness.csv',
    notebook:
        'notebook/ntmut_fitness.py.ipynb'

rule aamut_fitness:
    message:
        "Calculate estimates of the fitness effects of each amino acid substitution."
    params:
        # orf_to_nsps=config['orf1ab_to_nsps'],
        gene_ov=config['gene_overlaps'],
        genes=config['genes'],
        fit_pseudo=config['fitness_pseudocount'],
    input:
        ntfit_csv='results/ntmut_fitness/{cluster}_ntmut_fitness.csv',
    output:
        aafit_csv='results/aamut_fitness/{cluster}_aamut_fitness.csv',
    notebook:
        'notebook/aamut_fitness.py.ipynb'

rule concat_aamut:
    message:
        "Concatenating {{cluster}}_aamut_fitness.csv files",
    input:
        aafit_csv=expand('results/aamut_fitness/{cluster}_aamut_fitness.csv', cluster=config['clade_cluster'].keys()),
    output:
        aafit_concat='results/aamut_fitness/aamut_fitness_by_cluster.csv',
    shell:
        """
        {{ 
            head -n 1 {input.aafit_csv[0]};
            tail -n +2 -q {input.aafit_csv}
        }} > {output.aafit_concat}
        """

rule aamut_plots:
    message:
        "Generating interactive plots of a.a. mutational fitness",
    params:
        min_predicted_count = config['min_predicted_count'],
        clade_synonyms = config['clade_synonyms'],
        heatmap_minimal_domain = config['aa_fitness_heatmap_minimal_domain'],
        # orf1ab_to_nsps = config['orf1ab_to_nsps'],
        clade_cluster = config['clade_cluster'],
        cluster_founder = config['cluster_founder'],
        cluster_corr_min_count = config['cluster_corr_min_count'],
    input:
        clade_founder_nts="data/clade_founder.csv",
        aamut_by_cluster=rules.concat_aamut.output.aafit_concat,
    output:
        outdir=directory('results/aamut_fitness/plots')
    notebook:
        'notebook/aamut_plots.py.ipynb'

rule aggregate_plots_for_docs:
    """Aggregate plots to include in GitHub pages docs."""
    input:
        aa_fitness_plots_dir=rules.aamut_plots.output.outdir,
    output:
        expand(
            "results/plots_for_docs/{plot}.html",
            plot=docs_plot_annotations["plots"],
        ),
    params:
        plotsdir="results/plots_for_docs",
    shell:
        """
        mkdir -p {params.plotsdir}
        rm -f {params.plotsdir}/*
        cp {input.aa_fitness_plots_dir}/*.html {params.plotsdir}
#         """
    
# rule format_plot_for_docs:
#     message:
#         "Format a specific plot for the GitHub pages docs"
#     input:
#         plot=os.path.join(rules.aggregate_plots_for_docs.params.plotsdir, "{plot}.html"),
#         script="scripts/format_altair_html.py",
#     output:
#         plot="docs/{plot}.html",
#         markdown=temp("results/plots_for_docs/{plot}.md"),
#     params:
#         annotations=lambda wc: docs_plot_annotations["plots"][wc.plot],
#         url=config["docs_url"],
#         legend_suffix=docs_plot_annotations["legend_suffix"],
#     shell:
#         """
#         echo "## {params.annotations[title]}\n" > {output.markdown}
#         echo "{params.annotations[legend]}\n\n" >> {output.markdown}
#         echo "{params.legend_suffix}" >> {output.markdown}
#         python {input.script} \
#             --chart {input.plot} \
#             --markdown {output.markdown} \
#             --site {params.url} \
#             --title "{params.annotations[title]}" \
#             --description "{params.annotations[title]}" \
#             --output {output.plot}
#         """

# rule docs_index:
#     message:
#         "Write index for GitHub Pages doc"
#     output:
#         html="docs/index.html",
#     params:
#         plot_annotations=docs_plot_annotations,
#         current_mat=config["current_mat"],
#     script:
#         "scripts/docs_index.py"
