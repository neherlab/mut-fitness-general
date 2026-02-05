"""
This pipeline fetches raw mutation counts
and produces refined fitness including estimates of uncertainty for each mutation
in different subsets of the total sequence availability.
"""

import yaml
import os
import pandas as pd

configfile: "config.yaml"


with open(config["docs_plot_annotations"]) as f:
    docs_plot_annotations = yaml.safe_load(f)

OD = config["output_dir"]
INPUT_DIR = config["input_dir"]
CLADE_FOUNDER_PATH = os.path.join(INPUT_DIR, "clade_founder.csv")

if config["genes"] == "all":
    GENES = list(
        pd.read_csv(CLADE_FOUNDER_PATH).gene.unique()
    )
    if "noncoding" in GENES:
        GENES.remove("noncoding")
else:
    GENES = config["genes"]
print("Genes to analyze:", GENES)

template = docs_plot_annotations["heatmap_template"]

# Auto-generate plot annotation entries for each gene
docs_plot_annotations["plots"] = {}

for g in GENES:
    docs_plot_annotations["plots"][g] = {
        **template,
        "title": template["title"].format(gene=g),
    }


rule all:
    """Target rule with desired output files."""
    input:
        os.path.join(OD, "master_tables", "master_table.csv"),
        expand(
            os.path.join(OD, "plots_for_docs", "{plot}.html"),
            plot=list(docs_plot_annotations["plots"]),
        ),
        os.path.join(OD, "exploratory_figures", "plot_summary.txt"),


rule copy_input_files:
    input:
        mut_counts_src=lambda wc: next(
            (os.path.join(INPUT_DIR, f) for f in sorted(os.listdir(INPUT_DIR), reverse=True)
             if "mut_counts_by_clade" in f and f.endswith(".csv")),
            os.path.join(INPUT_DIR, "mut_counts_by_clade.csv")
        ),
        clade_founder_src=lambda wc: next(
            (os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) 
             if "clade_founder" in f and f.endswith(".csv")),
            os.path.join(INPUT_DIR, "clade_founder.csv")
        ),
    output:
        mut_counts=os.path.join(OD, "mut_counts_by_clade.csv"),
        clade_founder=os.path.join(OD, "clade_founder.csv"),
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
        clades=lambda wc: config["clade_cluster"][wc.cluster],
    input:
        counts_df=rules.predicted_counts.output.pred_count_csv,
    output:
        cluster_counts=os.path.join(OD, "ntmut_fitness", "{cluster}_ntmut_counts.csv"),
    run:
        clades_str = " ".join(params.clades)
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
        gene_ov_retain=config["gene_overlaps"]["retain"],
        gene_ov_exclude=config["gene_overlaps"]["exclude"],
        genes=GENES,
        fit_pseudo=config["fitness_pseudocount"],
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
        "Concatenating {{cluster}}_aamut_fitness.csv files"
    input:
        aafit_csv=expand(
            os.path.join(OD, "aamut_fitness", "{cluster}_aamut_fitness.csv"),
            cluster=config["clade_cluster"].keys(),
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
        "Generating interactive plots of a.a. mutational fitness"
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


rule plot_local_context:
    message:
        "Generate local context effects plots"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        logfile=os.path.join(OD, "exploratory_figures", "local_context_scatterplots", "plot_local_context.log"),
    params:
        outdir=os.path.join(OD, "exploratory_figures", "local_context_scatterplots"),
        min_sites=config.get("plot_min_sites", 5),
    shell:
        """
        mkdir -p {params.outdir}
        python scripts/plot_local_context.py \
            --counts {input.counts} \
            --output-dir {params.outdir} \
            --min-sites {params.min_sites} \
            > {output.logfile} 2>&1
        """


rule plot_motif_correlations:
    message:
        "Generate motif correlation plots"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        logfile=os.path.join(OD, "exploratory_figures", "motif_correlations", "plot_motif_correlations.log"),
    params:
        outdir=os.path.join(OD, "exploratory_figures", "motif_correlations"),
        min_sites=config.get("plot_min_sites", 5),
    shell:
        """
        mkdir -p {params.outdir}
        python scripts/plot_motif_correlations.py \
            --counts {input.counts} \
            --output-dir {params.outdir} \
            --min-sites {params.min_sites} \
            > {output.logfile} 2>&1
        """


rule plot_genome_distribution:
    message:
        "Generate genome distribution plots"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        logfile=os.path.join(OD, "exploratory_figures", "genome_distribution", "plot_genome_distribution.log"),
    params:
        outdir=os.path.join(OD, "exploratory_figures", "genome_distribution"),
    shell:
        """
        mkdir -p {params.outdir}
        python scripts/plot_genome_distribution.py \
            --counts {input.counts} \
            --output-dir {params.outdir} \
            > {output.logfile} 2>&1
        """


rule plot_mut_distributions:
    message:
        "Generate mutation distribution plots"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        logfile=os.path.join(OD, "exploratory_figures", "mut_distributions", "plot_mut_distributions.log"),
    params:
        outdir=os.path.join(OD, "exploratory_figures", "mut_distributions"),
    shell:
        """
        mkdir -p {params.outdir}
        python scripts/plot_mut_distributions.py \
            --counts {input.counts} \
            --output-dir {params.outdir} \
            > {output.logfile} 2>&1
        """


rule plot_r2:
    message:
        "Generate R2 sequential model plots"
    input:
        counts=rules.curated_counts.output.outfile,
    output:
        r2_pdf=os.path.join(OD, "exploratory_figures", "R2_sequentially.pdf"),
        weights_pdf=os.path.join(OD, "exploratory_figures", "model_weights.pdf"),
    params:
        outdir=os.path.join(OD, "exploratory_figures"),
    shell:
        """
        python scripts/plot_r2.py \
            --path {OD} \
            --config config.yaml
        """


rule summarize_plots:
    message:
        "Summarize exploratory plots"
    input:
        local_context_log=rules.plot_local_context.output.logfile,
        motif_corr_log=rules.plot_motif_correlations.output.logfile,
        genome_dist_log=rules.plot_genome_distribution.output.logfile,
        mut_dist_log=rules.plot_mut_distributions.output.logfile,
        r2_pdf=rules.plot_r2.output.r2_pdf,
    output:
        summary=os.path.join(OD, "exploratory_figures", "plot_summary.txt"),
    params:
        outdir=os.path.join(OD, "exploratory_figures"),
    shell:
        """
        echo "Exploratory Plots Summary" > {output.summary}
        echo "=========================" >> {output.summary}
        echo "" >> {output.summary}
        echo "Local context plots:" >> {output.summary}
        grep -E "(Saved|Done)" {input.local_context_log} >> {output.summary} || true
        echo "" >> {output.summary}
        echo "Motif correlation plots:" >> {output.summary}
        grep -E "(Saved|Done)" {input.motif_corr_log} >> {output.summary} || true
        echo "" >> {output.summary}
        echo "Genome distribution plots:" >> {output.summary}
        grep -E "(Saved|Done)" {input.genome_dist_log} >> {output.summary} || true
        echo "" >> {output.summary}
        echo "Mutation distribution plots:" >> {output.summary}
        grep -E "(Saved|Done)" {input.mut_dist_log} >> {output.summary} || true
        echo "" >> {output.summary}
        echo "R2 plots:" >> {output.summary}
        if [ -f {input.r2_pdf} ]; then echo "Generated R2_sequentially.pdf and model_weights.pdf" >> {output.summary}; fi
        echo "" >> {output.summary}
        find {params.outdir} -name "*.png" 2>/dev/null | wc -l | xargs echo "Total plots generated:" >> {output.summary}
        cat {output.summary}
        """




rule clobber:
    message:
        "Remove all output files"
    shell:
        "rm -rf {OD}"
