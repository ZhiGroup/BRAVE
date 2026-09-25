"""Per-comparison summary of the embedding concordance table.

Answers the question the three-pairing figure raises: does retraining agree with
the published encoder as well as two retrainings agree with each other?

Usage:
    $PY summarise_concordance.py --csv concordance_embeddings.csv
"""
from __future__ import print_function

import argparse

import pandas as pd

METRICS = ["cka", "mean_cca_r", "cca_r_top10", "dim_matched"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    cli = ap.parse_args()

    df = pd.read_csv(cli.csv)
    cols = [c for c in METRICS if c in df.columns]

    print("mean over the 16 regions")
    print(df.groupby("comparison")[cols].mean().to_string(float_format="%.4f"))
    print()
    print("min / max CKA per comparison")
    print(df.groupby("comparison")["cka"].agg(["min", "max"])
          .to_string(float_format="%.4f"))
    print()

    # Paired, region by region: is vs-published systematically different from
    # seed-vs-seed? A mean comparison alone would hide region-level structure.
    piv = df.pivot(index="region", columns="comparison", values="cka")
    if set(["seed1_vs_seed2", "seed1_vs_published",
            "seed2_vs_published"]).issubset(piv.columns):
        piv["pub_mean"] = piv[["seed1_vs_published",
                               "seed2_vs_published"]].mean(axis=1)
        piv["pub_minus_seedseed"] = piv["pub_mean"] - piv["seed1_vs_seed2"]
        n_up = int((piv["pub_minus_seedseed"] > 0).sum())
        print("per-region: mean of the two vs-published pairings "
              "minus seed1-vs-seed2")
        print(piv[["seed1_vs_seed2", "pub_mean", "pub_minus_seedseed"]]
              .sort_values("pub_minus_seedseed")
              .to_string(float_format="%.4f"))
        print()
        print("regions where vs-published exceeds seed-vs-seed: "
              "{0}/{1}".format(n_up, len(piv)))
        print("mean difference: {0:+.4f}".format(
            piv["pub_minus_seedseed"].mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
