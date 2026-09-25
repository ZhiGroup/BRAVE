"""Does spectral concentration explain the per-region spread in CKA?

The spectrum-matched surrogate in cka_significance.py shows that spectral shape
alone forces no agreement (surrogate CKA = permutation CKA = 0.0003). It does
not, however, test the degeneracy hypothesis, which is a different claim: when a
region's covariance is dominated by a few directions, CKA is driven almost
entirely by whether those top directions coincide across runs, so it swings hard
in both directions and is not comparable across regions with different spectra.

Thalamus is the case in question -- lowest CKA (0.30-0.36) alongside top-10
canonical correlations of 0.988, which is what a concentrated spectrum with a
rotated leading direction would look like.

Reported per region and seed:
  eff_rank_pr    participation ratio (sum l)^2 / sum l^2 -- effective number of
                 dimensions carrying variance
  eff_rank_ent   exp(Shannon entropy of the normalised spectrum)
  top1_frac      fraction of total variance in the leading direction

Usage:
  python effective_rank.py --seed1 <dir> --seed2 <dir> \
      --cka reproducibility/cka_significance.csv --out <csv>
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

EXCLUDE_LABELS = ("GM", "WM")


def load_seed(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, regions


def spectrum_stats(M):
    M = np.asarray(M, dtype=np.float64)
    M = M[np.isfinite(M).all(axis=1)]
    M = M - M.mean(axis=0)
    s = np.linalg.svd(M, compute_uv=False)
    lam = s ** 2                      # eigenvalues of the covariance
    lam = lam[lam > 0]
    p = lam / lam.sum()
    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    ent = float(np.exp(-(p * np.log(p)).sum()))
    return float(pr), ent, float(p[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--cka", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b1, regions = load_seed(args.seed1)
    b2, _ = load_seed(args.seed2)
    keep = [i for i, r in enumerate(regions) if r[2] not in EXCLUDE_LABELS]
    labels = [regions[i][2] for i in keep]

    rows = []
    for k, r in enumerate(keep):
        pr1, en1, t1 = spectrum_stats(b1[:, r, :])
        pr2, en2, t2 = spectrum_stats(b2[:, r, :])
        rows.append({
            "region": labels[k],
            "eff_rank_pr_s1": pr1, "eff_rank_pr_s2": pr2,
            "eff_rank_ent_s1": en1, "eff_rank_ent_s2": en2,
            "top1_frac_s1": t1, "top1_frac_s2": t2,
            "eff_rank_pr_mean": 0.5 * (pr1 + pr2),
            "top1_frac_mean": 0.5 * (t1 + t2),
        })

    df = pd.DataFrame(rows)
    cka = pd.read_csv(args.cka)[["region", "cka_observed", "cross_mean",
                                 "ratio_vs_cross"]]
    df = df.merge(cka, on="region", how="left")
    df.to_csv(args.out, index=False)

    print("{0:30s} {1:>9s} {2:>9s} {3:>9s} {4:>9s}".format(
        "region", "effRank", "top1%", "CKA", "ratio"))
    print("-" * 72)
    for _, r in df.sort_values("eff_rank_pr_mean").iterrows():
        print("{0:30s} {1:9.2f} {2:9.1%} {3:9.4f} {4:9.2f}".format(
            r["region"][:30], r["eff_rank_pr_mean"], r["top1_frac_mean"],
            r["cka_observed"], r["ratio_vs_cross"]))

    def corr(a, b):
        x, y = df[a].values, df[b].values
        pear = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        spear = float(np.corrcoef(rx, ry)[0, 1])
        return pear, spear

    print("\ncorrelation across the 16 regions (Pearson / Spearman):")
    for a, b in (("eff_rank_pr_mean", "cka_observed"),
                 ("top1_frac_mean", "cka_observed"),
                 ("eff_rank_pr_mean", "ratio_vs_cross"),
                 ("eff_rank_pr_mean", "cross_mean")):
        p, s = corr(a, b)
        print("  {0:20s} vs {1:16s} r = {2:+.3f}   rho = {3:+.3f}".format(
            a, b, p, s))

    print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
