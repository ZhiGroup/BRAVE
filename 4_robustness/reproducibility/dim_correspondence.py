"""Do correspondingly indexed embedding dimensions really not correspond?

The draft states that matched-dimension correlation of 0.187-0.331 shows
dimensions do not correspond across runs, on the argument that a contrastive
objective is invariant to rotation of the embedding axes.

That argument is sound but the number needs a null. For a random rotation in
d = 128 dimensions the expected matched |correlation| is about sqrt(2/(pi*d)) =
0.07, which would put the observed values 3-5x above chance -- i.e. partial
correspondence rather than none. But that expectation assumes an isotropic
spectrum, and these embeddings have an effective rank of 3-11, so the analytic
value does not apply. The null has to come from the data.

Two nulls, both preserving each run's actual spectrum:

  off-diagonal   mean |corr| between dimension j of one run and dimension k of
                 the other, j != k. This is the correspondence one gets from
                 two representations of the same subjects with no alignment of
                 axes at all.
  rotated        seed 2's embedding multiplied by a random orthogonal matrix,
                 then matched-dimension correlation recomputed. Destroys axis
                 correspondence while preserving every pairwise relationship.

If matched is comparable to these, "do not correspond" stands. If matched is
clearly higher, the draft overstates and should say correspondence is weak.

Usage:
  python dim_correspondence.py --seed1 <dir> --seed2 <dir> --out <csv>
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

EXCLUDE_LABELS = ("GM", "WM")
N_ROT = 20


def load_seed(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def zscore(M):
    M = np.asarray(M, dtype=np.float64)
    ok = np.isfinite(M).all(axis=1)
    M = M[ok]
    return (M - M.mean(0)) / (M.std(0) + 1e-12), ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    b1, ids1, regions = load_seed(args.seed1)
    b2, ids2, _ = load_seed(args.seed2)
    if ids1 != ids2:
        common = [i for i in ids1 if i in set(ids2)]
        b1 = b1[[ids1.index(i) for i in common]]
        b2 = b2[[ids2.index(i) for i in common]]

    keep = [i for i, r in enumerate(regions) if r[2] not in EXCLUDE_LABELS]
    labels = [regions[i][2] for i in keep]

    rows = []
    print("{0:30s} {1:>9s} {2:>9s} {3:>9s} {4:>8s}".format(
        "region", "matched", "off-diag", "rotated", "ratio"))
    print("-" * 70)
    for k, r in enumerate(keep):
        X, okx = zscore(b1[:, r, :])
        Y, oky = zscore(b2[:, r, :])
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        C = X.T.dot(Y) / n
        d = C.shape[0]

        matched = float(np.mean(np.abs(np.diag(C))))
        off = float((np.abs(C).sum() - np.abs(np.diag(C)).sum())
                    / (d * d - d))

        rot = []
        for _ in range(N_ROT):
            Q, _r = np.linalg.qr(rng.standard_normal((d, d)))
            Yr = Y.dot(Q)
            Yr = (Yr - Yr.mean(0)) / (Yr.std(0) + 1e-12)
            Cr = X.T.dot(Yr) / n
            rot.append(float(np.mean(np.abs(np.diag(Cr)))))
        rot_m = float(np.mean(rot))

        rows.append({"region": labels[k], "matched": matched,
                     "off_diagonal": off, "rotated_null": rot_m,
                     "ratio_matched_over_rotated": matched / rot_m})
        print("{0:30s} {1:9.4f} {2:9.4f} {3:9.4f} {4:8.2f}".format(
            labels[k][:30], matched, off, rot_m, matched / rot_m))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("-" * 70)
    print("{0:30s} {1:9.4f} {2:9.4f} {3:9.4f} {4:8.2f}".format(
        "MEAN", df["matched"].mean(), df["off_diagonal"].mean(),
        df["rotated_null"].mean(), df["ratio_matched_over_rotated"].mean()))
    print("\nanalytic expectation under a random rotation, isotropic: "
          "{0:.4f}".format(np.sqrt(2.0 / (np.pi * 128))))
    print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
