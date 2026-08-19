"""Shared helpers for the plot_coef_comparison*.py scripts.

These scripts are manual analysis tools (not part of the Snakemake DAG) used
to compare GLM coefficients across pathogens. This module holds the pieces
that are identical across all three scripts: dataset/color registry loading,
model loading, and CLI argument parsing.
"""
import os
import pickle
import argparse

import yaml

from modules.glm import GeneralLinearModel
from modules.load import load_synonymous_muts

MUT_TYPES = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

_DEFAULT_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "plot_datasets.yaml",
)


def load_dataset_registry(path=None):
    """Load the datasets/precomputed/colors registry from a YAML file.

    Returns (datasets, precomputed, colors) dicts.
    """
    if path is None:
        path = _DEFAULT_REGISTRY_PATH
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    return registry["datasets"], registry["precomputed"], registry["colors"]


def load_models(datasets, precomputed_models, transform=None):
    """Return dict mapping virus -> mut_type -> coefficients.

    If `transform` is given, it is applied to each mutation type's
    coefficient array after loading (both for CSV-trained and precomputed
    models).
    """
    coefs = {}

    # Train models from CSVs
    for name, path in datasets.items():
        df = load_synonymous_muts(path)
        model = GeneralLinearModel(included_factors=['local_context'])
        model.train(df_train=df)
        if transform is None:
            coefs[name] = model.W
        else:
            coefs[name] = {mt: transform(w) for mt, w in model.W.items()}

    # Load precomputed pickled models
    for name, path in precomputed_models.items():
        with open(path, 'rb') as f:
            W_dict = pickle.load(f)
        if transform is None:
            coefs[name] = W_dict
        else:
            coefs[name] = {mt: transform(w) for mt, w in W_dict.items()}

    return coefs


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save the PDF (optional)."
    )

    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Which datasets to include (default: all datasets in --datasets-file)."
    )

    parser.add_argument(
        "--only-precomputed",
        action="store_true",
        help="Skip training, only load precomputed models."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively."
    )

    parser.add_argument(
        "--datasets-file",
        type=str,
        default=_DEFAULT_REGISTRY_PATH,
        help="Path to the YAML file configuring dataset/precomputed paths "
             "and colors (default: scripts/plot_datasets.yaml)."
    )

    return parser


def select_datasets(args, datasets_default, precomputed_default):
    requested = args.datasets if args.datasets is not None else list(datasets_default.keys())
    selected_datasets = {k: datasets_default[k]
                         for k in requested if k in datasets_default}
    selected_precomputed = {k: v for k, v in precomputed_default.items()
                            if k in requested and os.path.exists(v)}
    return selected_datasets, selected_precomputed
