"""How much CKA does a *trivial* perturbation cost? (the missing ceiling)

Every reference computed so far is a floor: permutation (0.0003), spectrum
surrogate (0.0003), cross-region (0.17-0.54). Floors establish that the observed
agreement is not nothing. They cannot say whether 0.87 is excellent or mediocre,
because nothing bounds the top of the scale.

The natural ceiling is a perturbation that is real but as small as possible.
Re-extracting from the same run one epoch earlier is exactly that: identical
seed, identical data order, identical code -- one optimiser epoch apart. Whatever
CKA that costs is roughly the floor on measurement instability, and the
seed-to-seed value should be read against it.

  ceiling   CKA(seed 1 epoch 6, seed 1 epoch 7)
  observed  CKA(seed 1 epoch 7, seed 2 epoch 7)
  null      mean CKA(seed 1 epoch 7 region i, seed 2 epoch 7 region j), i != j

All three are computed on the SAME subject subset, since the epoch-6 extraction
is limited to the first N subjects for I/O reasons. Comparing a subset-derived
ceiling to a full-cohort observed value would confound the two.

Usage:
  python ceiling_cka.py --ep6 <dir> --ep7 <dir> --seed2 <dir> --out <csv>
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
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def centre(M):
    M = np.asarray(M, dtype=np.float64)
    return M - M.mean(axis=0)


def cka(X, Y):
    X, Y = centre(X), centre(Y)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    num = float((X.T.dot(Y) ** 2).sum())
    den = (np.linalg.norm(X.T.dot(X), "fro")
           * np.linalg.norm(Y.T.dot(Y), "fro"))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ep6", required=True)
    ap.add_argument("--ep7", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b6, ids6, regions = load_seed(args.ep6)
    b7, ids7, _ = load_seed(args.ep7)
    b2, ids2, _ = load_seed(args.seed2)

    # The epoch-6 extraction is truncated, so everything is aligned to its IDs.
    pos7 = {s: i for i, s in enumerate(ids7)}
    pos2 = {s: i for i, s in enumerate(ids2)}
    common = [s for s in ids6 if s in pos7 and s in pos2]
    if len(common) < 500:
        sys.exit("[ERROR] only {0} shared subjects".format(len(common)))
    i6 = [ids6.index(s) for s in common]
    i7 = [pos7[s] for s in common]
    i2 = [pos2[s] for s in common]
    b6, b7, b2 = b6[i6], b7[i7], b2[i2]

    keep = [i for i, r in enumerate(regions) if r[2] not in EXCLUDE_LABELS]
    labels = [regions[i][2] for i in keep]
    b6, b7, b2 = b6[:, keep, :], b7[:, keep, :], b2[:, keep, :]
    n_reg = len(labels)
    print("subjects: {0:,}   regions: {1}\n".format(len(common), n_reg))

    rows = []
    for r in range(n_reg):
        ceil = cka(b6[:, r, :], b7[:, r, :])
        obs = cka(b7[:, r, :], b2[:, r, :])
        cross = [cka(b7[:, r, :], b2[:, j, :])
                 for j in range(n_reg) if j != r]
        rows.append({
            "region": labels[r],
            "ceiling_ep6_vs_ep7": ceil,
            "observed_seed1_vs_seed2": obs,
            "cross_region_null": float(np.mean(cross)),
            "obs_as_pct_of_ceiling": 100.0 * obs / ceil if ceil else np.nan,
        })
        print("{0:30s} ceiling {1:.4f} | observed {2:.4f} | {3:5.1f}% of "
              "ceiling".format(labels[r][:30], ceil, obs,
                               rows[-1]["obs_as_pct_of_ceiling"]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print("mean ceiling  (1 epoch apart, same seed) : {0:.4f}".format(
        df["ceiling_ep6_vs_ep7"].mean()))
    print("mean observed (seed 1 vs seed 2)         : {0:.4f}".format(
        df["observed_seed1_vs_seed2"].mean()))
    print("mean cross-region null                   : {0:.4f}".format(
        df["cross_region_null"].mean()))
    print("observed as a share of ceiling           : {0:.1f}%".format(
        100.0 * df["observed_seed1_vs_seed2"].mean()
        / df["ceiling_ep6_vs_ep7"].mean()))
    print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
