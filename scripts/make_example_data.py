"""Regenerate the tiny example dataset under example/hiv_pol_mini/.

Slices the first N nucleotide sites out of a real HIV-1 pol mutation-count
run (same data used for results_hiv_pol_141125), so the example stays a
realistic but small (< 200KB) input for the pipeline. Only needed if the
source run is available and you want to refresh the committed slice --
the checked-in CSVs under example/hiv_pol_mini/ are otherwise used as-is.
"""
import argparse
import os

import pandas as pd

DEFAULT_SOURCE = "/scicore/home/neher/kuznet0001/hiv_analysis/hiv-pol-nextstrain/results_051125"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example", "hiv_pol_mini")
MAX_SITE = 200


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE, help="Directory with the full-size mut_counts_by_clade.csv/clade_founder.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Where to write the sliced example CSVs")
    parser.add_argument("--max-site", type=int, default=MAX_SITE, help="Keep nucleotide sites 1..max-site")
    args = parser.parse_args()

    clade_founder = pd.read_csv(os.path.join(args.source_dir, "clade_founder.csv"))
    mut_counts = pd.read_csv(os.path.join(args.source_dir, "mut_counts_by_clade.csv"))

    clade_founder = clade_founder[clade_founder["site"] <= args.max_site]
    mut_counts = mut_counts[mut_counts["nt_site"] <= args.max_site]

    os.makedirs(args.output_dir, exist_ok=True)
    clade_founder.to_csv(os.path.join(args.output_dir, "clade_founder.csv"), index=False)
    mut_counts.to_csv(os.path.join(args.output_dir, "mut_counts_by_clade.csv"), index=False)
    print(f"Wrote {len(clade_founder)} clade_founder rows and {len(mut_counts)} mut_counts rows to {args.output_dir}")


if __name__ == "__main__":
    main()
