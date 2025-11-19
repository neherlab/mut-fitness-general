"""
This pipeline fetches raw mutation counts
and produces refined fitness including estimates of uncertainty for each mutation
in different subsets of the total sequence availability.
"""

import yaml
import os

configfile: "config.yaml"
with open(config["docs_plot_annotations"]) as f:
    docs_plot_annotations = yaml.safe_load(f)

OD = config["output_dir"]
INPUT_DIR = config["input_dir"]

rule all:
    """Target rule with desired output files."""
    input:
        os.path.join(OD, "master_tables", "master_table.csv"),
        expand(
            os.path.join(OD, "plots_for_docs", "{plot}.html"),
            plot=list(docs_plot_annotations["plots"]),
        ),

rule copy_input_files:
    input:
        mut_counts_src=os.path.join(INPUT_DIR, "mut_counts_by_clade.csv"),
        clade_founder_src=os.path.join(INPUT_DIR, "clade_founder.csv")
    output:
        mut_counts=os.path.join(OD, "mut_counts_by_clade.csv"),
        clade_founder=os.path.join(OD, "clade_founder.csv")
    run:
        os.makedirs(OD, exist_ok=True)
        shell("cp {input.mut_counts_src} {output.mut_counts}")
        shell("cp {input.clade_founder_src} {output.clade_founder}")


rule curated_counts:
    message:
        "Create training dataset to infer the General Linear Model for mutations"
    input:
        mut_counts=rules.copy_input_files.output.mut_counts,
        # clade_founder=rules.annotate_counts.output.founder_csv,
    output:
        outfile=os.path.join(OD, "curated", "curated_mut_counts.csv"),
    params:
        overwrite=lambda wc: "--overwrite" if config["overwrite"] else "",
    shell:
        "python scripts/curate_counts.py --mut-counts {input.mut_counts} --output {output.outfile} {params.overwrite}"

rule master_table:
    message:
        "Create tables with predicted mutation rates for each mutation in each of its contexts"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        ms=os.path.join(OD, "master_tables", "master_table.csv"),
    shell:
        "python scripts/master_tables.py --counts {input.counts} --output {output.ms}"

rule predicted_counts:
    message:
        "Add predicted counts, based on the inferred mutation rate model, to the table with observed mutation counts."
    input:
        counts_df=rules.copy_input_files.output.mut_counts,
        curated_counts_df=rules.curated_counts.output.outfile,
    output:
        pred_count_csv=os.path.join(OD, "pred_mut_counts_by_clade.csv"),
    shell:
        "python scripts/predicted_counts_by_clade.py --counts-df {input.counts_df} --curated-counts-df {input.curated_counts_df} --output {output.pred_count_csv}"

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
        cluster_counts=os.path.join(OD, "ntmut_fitness", "{cluster}_ntmut_counts.csv"),
    run:
        clades_str = ' '.join(params.clades)
        shell(
            "python scripts/ntmut_counts_cluster.py "
            "--counts-df {input.counts_df} "
            "--cluster {params.cluster} "
            "--clades " + clades_str + " "
            "--output {output.cluster_counts}"
        )

rule ntmut_fitness:
    message:
        "Calculate estimates of the fitness effects of each nucleotide mutation."
    input:
        cluster_counts=os.path.join(OD, "ntmut_fitness", "{cluster}_ntmut_counts.csv"),
    output:
        ntfit_csv=os.path.join(OD, "ntmut_fitness", "{cluster}_ntmut_fitness.csv"),
    shell:
        "python scripts/ntmut_fitness.py --cluster-counts {input.cluster_counts} --output {output.ntfit_csv}"

rule aamut_fitness:
    message:
        "Calculate estimates of the fitness effects of each amino acid substitution."
    params:
        gene_ov_retain=config['gene_overlaps']['retain'],
        gene_ov_exclude=config['gene_overlaps']['exclude'],
        genes=config['genes'],
        fit_pseudo=config['fitness_pseudocount'],
    input:
        ntfit_csv=os.path.join(OD, "ntmut_fitness", "{cluster}_ntmut_fitness.csv"),
    output:
        aafit_csv=os.path.join(OD, "aamut_fitness", "{cluster}_aamut_fitness.csv"),
    shell:
        """
        python scripts/aamut_fitness.py \
            --gene-overlaps-retain "{params.gene_ov_retain}" \
            --gene-overlaps-exclude "{params.gene_ov_exclude}" \
            --ntmut-fit {input.ntfit_csv} \
            --output {output.aafit_csv} \
            --genes {params.genes} \
            --fitness-pseudocount {params.fit_pseudo}
        """

rule concat_aamut:
    message:
        "Concatenating {{cluster}}_aamut_fitness.csv files",
    input:
        aafit_csv=expand(
            os.path.join(OD, "aamut_fitness", "{cluster}_aamut_fitness.csv"),
            cluster=config['clade_cluster'].keys()
        ),
    output:
        aafit_concat=os.path.join(OD, "aamut_fitness", "aamut_fitness_by_cluster.csv"),
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
    input:
        clade_founder_nts=rules.copy_input_files.output.clade_founder,
        aamut_by_cluster=rules.concat_aamut.output.aafit_concat,
        config_file="config.yaml",
    output:
        outdir=directory(os.path.join(OD, "aamut_fitness", "plots")),
    shell:
        "python scripts/aamut_plots.py "
        "--aamut-by-cluster {input.aamut_by_cluster} "
        "--clade-founder-nts {input.clade_founder_nts} "
        "--output-dir {output.outdir} "
        "--config {input.config_file}"

rule aggregate_plots_for_docs:
    """Aggregate plots to include in GitHub pages docs."""
    input:
        aa_fitness_plots_dir=rules.aamut_plots.output.outdir,
    output:
        expand(
            os.path.join(OD, "plots_for_docs", "{plot}.html"),
            plot=docs_plot_annotations["plots"],
        ),
    params:
        plotsdir=os.path.join(OD, "plots_for_docs"),
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

rule clobber:
    message:
        "Remove all output files"
    shell:
        "rm -rf {OD}"