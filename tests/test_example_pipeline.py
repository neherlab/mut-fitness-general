"""End-to-end smoke test: runs the real Snakemake pipeline on the small
example dataset (example/hiv_pol_mini/) and checks it produces sane output.

This is the cheapest check that catches a broken pipeline stage: if any
rule crashes, `snakemake` returns non-zero and the test fails.
"""
import os
import subprocess

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results_example_hiv_pol")
CONFIG = os.path.join(REPO_ROOT, "configs", "example_hiv_pol.yaml")


@pytest.fixture(scope="module")
def pipeline_output_dir():
    result = subprocess.run(
        ["snakemake", "-c1", "--configfile", CONFIG, "--forceall"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"snakemake failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return OUTPUT_DIR


def test_master_table_has_predictions(pipeline_output_dir):
    df = pd.read_csv(os.path.join(pipeline_output_dir, "master_tables", "master_table.csv"))
    assert len(df) > 0
    assert {"mut_type", "motif", "predicted_count"}.issubset(df.columns)
    assert df["predicted_count"].notna().all()


def test_aamut_fitness_has_finite_estimates(pipeline_output_dir):
    df = pd.read_csv(
        os.path.join(pipeline_output_dir, "aamut_fitness", "aamut_fitness_by_cluster.csv")
    )
    assert len(df) > 0
    assert {"gene", "aa_mutation", "delta_fitness", "uncertainty"}.issubset(df.columns)
    assert (df["gene"] == "pol").all()
    assert df["delta_fitness"].notna().all()
    assert (df["uncertainty"] >= 0).all()


def test_docs_plot_generated(pipeline_output_dir):
    plot_path = os.path.join(pipeline_output_dir, "plots_for_docs", "pol.html")
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 0
