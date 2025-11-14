# Fitness effects of nucleotide and amino acid mutations updated with novel estimates of neutral rates.

## Overview
This repository computes an updated estimate of the fitness effects of mutations of the given genome,
based on SARS-CoV-2 work presented in this [paper](https://doi.org/10.1101/2025.01.07.631013) by H.K. Haddox, 
G. Angehrn, L. Sesta, C. Jennings-Shaffer, S. Temple, J.G. Galloway, W.S. DeWitt, J.D. Bloom, [F.A. Matsen IV](https://matsen.fhcrc.org/), and [R.A. Neher](https://neherlab.org/).

It builds upon and expand a previous approach to estimate viral fitness that can be found at [jbloomlab/SARS2-mut-fitness](https://github.com/jbloomlab/SARS2-mut-fitness/tree/main).

The counts from the mutation-annotated tree from Nextstrain are used to accurately estimate the mutation rates according to:

* A site's local nucleotide context.
* (optional) The base pairing in RNA secondary structure.
* (optional) The region of the genome the site belongs to.

Fitness effects are subsequently estimated by comparing the actual observed counts to the one predicted from the
inferred rates, within a Bayesian probabilistic framework that also provides uncertainties.

## References
- Details about the computational framework can be found in the related [paper](https://doi.org/10.1101/2025.01.07.631013).
- The original approach for estimating mutational fitness is presented in [Bloom & Neher](https://academic.oup.com/ve/article/9/2/vead055/7265011).

## Computational pipeline
It is possible to reproduce the fitness estimates by running the computational analysis defined in this GitHub repository.

Firstly, a customized [conda](https://docs.conda.io/) environment needs to be built from the [environment.yml](environment.yml) file. To do so, you must install [conda](https://docs.conda.io/) and then run:

    conda env create -f environment.yml

This will create a [conda](https://docs.conda.io/) environment called `mut-fitness-env`, that you need to activate:

    conda activate mut-fitness-env

The pipeline is managed by [snakemake](https://snakemake.readthedocs.io/) through a [Snakefile](Snakefile), whose configuration is defined in [config.yaml](config.yaml). To run the pipeline use:

    snakemake -c <n_cpus>

where `n_cpus` is the number of CPU cores you want to use.

The pipeline mainly relies on Python Jupyter notebooks to run. These can be found in the [./notebook](notebook) folder.

### Input

The pipeline runs downstream from two files at given paths:

- Table of clade founders nucleotides, example for SARS-CoV-2: [~/results_gisaid_2024-04-24/clade_founder_nts/clade_founder_nts.csv](https://github.com/jbloomlab/SARS2-mut-fitness/blob/main/results_gisaid_2024-04-24/clade_founder_nts/clade_founder_nts.csv).
- Clade-wise table of counts, example for SARS-CoV-2: [~/results_gisaid_2024-04-24/expected_vs_actual_mut_counts/expected_vs_actual_mut_counts.csv](https://github.com/jbloomlab/SARS2-mut-fitness/blob/main/results_gisaid_2024-04-24/expected_vs_actual_mut_counts/expected_vs_actual_mut_counts.csv).

The related links are defined in the [config.yaml](config.yaml) file.

The file containing the nucleotide pairing predictions from [sec structure study] is located at [./data/rna_structure](data/rna_structure/).

### Configuration
Ahead of the computation of mutational fitness effects, predicted and actual mutation counts can be aggregated by defining clusters of clades. This is defined by a dictionary `clade_cluster` in the [config.yaml](config.yaml) file, which can be customized.

### Output
Files produced by the pipeline are saved into the [./results](results) folder. These are subsequently divided in the following subfolders:
- [curated](results/curated/): the training datasets for the *General Linear Model* (GLM) producing the predicted counts.
- [master_tables](results/master_tables/): the reference models inferred from the training datasets, i.e. a site's context and the associated mutation rate.
- [ntmut_fitness](results/ntmut_fitness/): files `{cluster}_ntmut_fitness.csv` for each cluster of clades with the nucleotide mutation fitness effects.
- [aamut_fitness](results/aamut_fitness/): files `{cluster}_aamut_fitness.csv` with fitness effects of the amino acid mutations for each cluster of clades.

#### Nucleotide fitness
The fitness effects of nucleotide mutations are reported in the `./results/_ntmut_fitness.csv` folder. In the dataframes therein, the output of the Bayesian probabilistic framework can be found, see section [Theoretical framework](#theoretical-framework) for additional details. Relevant entries are:
* `f_mean`: the average with respect to the posterior of the fitness effect. It provides the input for amino acid fitness effects.
* `f_st_dev`: the posterior standard deviation, representing the uncertainty on the nucleotide fitness effect. It is also input for the amino acid estimates.

#### Amino acid fitness
The results for amino acid fitness effects are found in the `./results/aamut_fitness/` folder. In the dataframes therein, relevant columns are:
* `delta_fitness`: the estimate of the fitness effect of an amino acid mutation. It is computed as the weighted average of the nucleotide effects in the corresponding codon that produce the nonsynonymous mutation. Given a set of weights $w_i$, the weighted average of $f$ reads $\overline f=\sum_iw_if_i/\sum_iw_i$. In our case, the weights are the inverse variances, i.e. $w_i=1/\sigma^2_i$.
* `uncertainty`: it is the uncertainty associated to the weighted average. It can be interpreted as the overall standard deviation of a probability distribution defined as the product of a set of univariate Gaussians, whose standard deviations are $\sigma_i$, i.e. $\sigma=\sqrt{1/\sum_i\left(1/\sigma^2_i\right)}$.

For additional details, you can take a look to the [Theoretical framework](#theoretical-framework) section.   

## Theoretical framework
A detailed description of the theoretical framework for the GLM and the Bayesian setting can be found in this [paper](https://doi.org/10.1101/2025.01.07.631013). Here we outline some fundamental elements.

### GLM for predicted counts
GLM's are inferred on two curated datasets containing counts for synonymous mutations. Genome sites are retained if:
- The wildtype nucleotide of the clade founder is equal to the [Wuhan-1](https://www.ncbi.nlm.nih.gov/nuccore/1798174254) reference.
- The site motif, i.e. 5'-3' nucleotides context does not change.
- The codon the site belongs to is conserved.
- Sites that are marked as `excluded=True` or `masked_in_usher=True` are excluded.

The output of the GLM is a predicted count for each possible nucleotide mutation and condition $`n^{x_i\rightarrow y_i}_{\mathrm{pred}}\left(\mathbf c_i \right) = n^{x_i\rightarrow y_i}_{\mathrm{pred}}\left(p_i, m_i, l_i\right)`$, where:

- $x_i$, $y_i$ are the wildtype and mutant nucleotide respectively.
- $p_i$ is the pairing state.
- $m_i$ is the site motif.
- $l_i$ indicates if the site is before/after a lightswitch position along the genome.

These predicted counts can be converted in terms of rates by imposing that:

$$ N_s = T\sum_{x, \mathbf c}\sum_{y\neq x} s^{x\rightarrow y}\left(\mathbf c\right)r^{x\rightarrow y}(\mathbf c), $$

where $r^{x\rightarrow y}(\mathbf c)$ are the rates, $s^{x\rightarrow y}(\mathbf c)$ are the number of sites in the genome for which $x\rightarrow y$ is synonymous in condition $\mathbf c$ and $N_s$ is the total number of synonymous mutations. From the previous equation one can show that $T = N_s / \sum_{x,\mathbf{c},y\neq x}s^{x\rightarrow y}(\mathbf{c})$ and finally $r^{x\rightarrow y}(\mathbf{c}) = n^{x\rightarrow y}_{\mathrm{pred}}(\mathbf{c}) / T$.

Once the rates are determined from the training datasets, it is possible to compute the predicted counts for every clade-specific subtree by re-scaling according to the proper evolutionary time $n^{a, x\rightarrow y}_{\mathrm{pred}}(\mathbf{c}) = T^a \times r^{{x\rightarrow y}}(\mathbf{c})$, $a$ being the clade label.

### Posterior estimate of mutation fitness

#### Nucleotide mutations
Fitness effects of nucleotide mutations are computed within a Bayesian
probabilistic framework. For a specific site and nucleotide mutation,
our objective is the posteior probability of the fitness effect $\Delta f$ given
the observed counts $n_{\mathrm{obs}}$ and integrated over the possible expected
counts $n_{\mathrm{exp}}$, whose mean is the model prediction $n_{\mathrm{pred}}$.

The posterior reads:

$$ P(\Delta f \vert n_{\mathrm{obs}}) \propto P(n_{\mathrm{obs}} \vert \Delta f) P(\Delta f), $$

with the likelihood being:

$$ P(n_{\mathrm{obs}} \vert \Delta f) =
\int_{0}^{+\infty}\mathrm{d}n_{\mathrm{exp}} P(n_{\mathrm{obs}} \vert \Delta f, n_{\mathrm{exp}}) 
P(n_{\mathrm{exp}}).
$$

Once the posterior is known, we can compute our estimate and uncertainty of the fitness
effect as:

$$
\begin{cases}
  \left\langle \Delta f \right\rangle = \int_{-\infty}^{+\infty}\mathrm{d}(\Delta f)P(\Delta f \vert n_{\mathrm{obs}})\Delta f  \\
  \sigma_{\Delta f} = \sqrt{\left\langle \Delta f^2 \right\rangle - \left\langle \Delta f \right\rangle^2}
\end{cases}
$$

#### Amino acid mutations
Fitness effects of amino acid mutations are computed as weighted averages
of all nucleotide mutations producing the same nonsynonymous mutation in a
given codon. For an amino acid mutation $a\rightarrow a'$ one gets:

$$
\begin{cases}
  \Delta f\left(a\rightarrow a'\right) = \frac{{\sum_i}{\sum_y}
  \frac{\left\langle \Delta f_i(x_i\rightarrow y_i) \right\rangle}{\sigma^2_{\Delta f_i(x_i\rightarrow y_i)}} \delta\left(a',g(\mathbf y)\right)}
  {{\sum_i}{\sum_y}
  \frac{1}{\sigma^2_{\Delta f_i(x_i\rightarrow y_i)}} \delta\left(a',g(\mathbf y)\right)} \\
  \sigma_{\Delta f(a\rightarrow a')} = \frac{1}
  {\sqrt{{\sum_i}{\sum_y}
  \frac{\delta\left(a',g(\mathbf y)\right)}{\sigma^2_{\Delta f_i(x_i\rightarrow y_i)}}}}
\end{cases}
$$

where $g(\mathbf y)$ maps the codon into the corresponding amino acid, and the sums run over
the codon sites $i$ and possible mutant nucleotides at these sites $y_i$.

