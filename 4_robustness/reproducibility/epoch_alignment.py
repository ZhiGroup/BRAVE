"""Does "epoch 7" denote the same state of training in two different runs?

The seeds are matched on optimisation budget by construction -- identical
global_step (4,584) and identical learning rate to 17 digits -- and their three
loss terms agree to within 0.6-5% with no consistent direction. What that does
not establish is whether the two runs have reached a comparable *state*, and it
matters here for two reasons. Epoch 7 is not a convergence criterion: it was
chosen only to match the published checkpoint index, and training stops
mid-warm-up, where one epoch shifts CKA by 15-27%. If the exact stopping point
matters that much, part of the seed-to-seed gap could be a difference in training
stage rather than in solution.

The test is a cross-epoch alignment matrix. For epochs i in run 1 and j in run 2:

  diagonal (i = j)      runs compared at the same nominal epoch
  off-diagonal (i != j) runs compared one epoch out of step

If the diagonal exceeds the off-diagonal, the epoch index tracks something the
two runs share, and matching on it is doing real work. If the matrix is flat in
j, the index does not correspond to a common state and the comparison is less
controlled than the matched global_step suggests.

A stronger version of the same question: for each run-1 epoch, which run-2 epoch
maximises alignment? If that is consistently the same epoch, the runs progress in
step.

Usage:
  python epoch_alignment.py --a run1_ep6=<dir> run1_ep7=<dir> \
      --b run2_ep6=<dir> run2_ep7=<dir> --out <csv>
"""
from __future__ import print_function

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

EXCLUDE_LABELS = ("GM", "WM")


def load(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def cka(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = float((X.T.dot(Y) ** 2).sum())
    den = (np.linalg.norm(X.T.dot(X), "fro")
           * np.linalg.norm(Y.T.dot(Y), "fro"))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", nargs="+", required=True, help="name=dir for run 1")
    ap.add_argument("--b", nargs="+", required=True, help="name=dir for run 2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = dict(s.split("=", 1) for s in args.a)
    B = dict(s.split("=", 1) for s in args.b)

    # Align every extraction on the subject set of the smallest, since the
    # epoch-6 extractions are truncated for I/O reasons.
    loaded = {}
    for nm, p in list(A.items()) + list(B.items()):
        loaded[nm] = load(p)
    common = None
    for nm, (_, ids, _) in loaded.items():
        s = set(ids)
        common = s if common is None else (common & s)
    order = [i for i in loaded[list(A)[0]][1] if i in common]
    print("subjects common to all {0} extractions: {1:,}".format(
        len(loaded), len(order)), flush=True)

    regions = loaded[list(A)[0]][2]
    keep = [i for i, r in enumerate(regions) if r[2] not in EXCLUDE_LABELS]
    labels = [regions[i][2] for i in keep]

    mats = {}
    for nm, (bre, ids, _) in loaded.items():
        idx = [ids.index(i) for i in order]
        mats[nm] = bre[idx][:, keep, :]

    rows = []
    for an, bn in itertools.product(sorted(A), sorted(B)):
        for k in range(len(labels)):
            rows.append({"run1": an, "run2": bn, "region": labels[k],
                         "cka": cka(mats[an][:, k, :], mats[bn][:, k, :])})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    piv = df.groupby(["run1", "run2"])["cka"].mean().unstack()
    print("\nmean CKA over {0} regions\n".format(len(labels)))
    print(piv.to_string(float_format=lambda v: "{0:.4f}".format(v)))

    print("\nmatched vs mismatched epoch:")
    matched, mismatched = [], []
    for an in sorted(A):
        ea = an.split("_ep")[-1]
        for bn in sorted(B):
            eb = bn.split("_ep")[-1]
            (matched if ea == eb else mismatched).append(piv.loc[an, bn])
    print("  same epoch      {0:.4f}".format(np.mean(matched)))
    print("  one epoch apart {0:.4f}".format(np.mean(mismatched)))
    print("  difference      {0:+.4f}".format(
        np.mean(matched) - np.mean(mismatched)))

    print("\nbest-matching run-2 epoch for each run-1 epoch:")
    for an in sorted(A):
        best = piv.loc[an].idxmax()
        print("  {0:12s} -> {1:12s} ({2:.4f})".format(
            an, best, piv.loc[an, best]))

    print("\nper-region, does the same epoch win?")
    wins = 0
    for k, lab in enumerate(labels):
        sub = df[df["region"] == lab]
        p = sub.groupby(["run1", "run2"])["cka"].mean().unstack()
        for an in sorted(A):
            ea = an.split("_ep")[-1]
            if p.loc[an].idxmax().split("_ep")[-1] == ea:
                wins += 1
    print("  {0}/{1} region-by-epoch comparisons favour the matched epoch"
          .format(wins, len(labels) * len(A)))
    print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
