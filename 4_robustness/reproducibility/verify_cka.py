"""Check the feature-space CKA against the textbook Gram-matrix definition.

`concordance_embeddings.py` computes linear CKA as

    ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

on (n, d) matrices, which never forms an n x n Gram matrix. The definition in
Kornblith et al. (2019) is written over Gram matrices instead:

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

with K = XX^T, L = YY^T and HSIC computed on doubly-centred Grams. For a linear
kernel the two are algebraically identical; this verifies that numerically on a
subsample small enough to build the n x n matrices.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np


def cka_feature(X, Y):
    """The form used in production: O(n*d^2), no n x n matrix."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = float((X.T.dot(Y) ** 2).sum())
    den = (np.linalg.norm(X.T.dot(X), "fro")
           * np.linalg.norm(Y.T.dot(Y), "fro"))
    return num / den


def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H.dot(K).dot(H)


def cka_gram(X, Y):
    """The textbook form: builds n x n Grams and double-centres them."""
    K = _center_gram(X.dot(X.T))
    L = _center_gram(Y.dot(Y.T))
    hsic_kl = float((K * L).sum())
    hsic_kk = float((K * K).sum())
    hsic_ll = float((L * L).sum())
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--n", type=int, default=2000,
                    help="subsample size; n x n Gram must fit in memory")
    ap.add_argument("--regions", type=int, default=6)
    args = ap.parse_args()

    b1 = np.load(os.path.join(args.seed1, "bre_discovery.npy"))
    b2 = np.load(os.path.join(args.seed2, "bre_discovery.npy"))
    labels = [l.rstrip("\n").split("\t")[2]
              for l in open(os.path.join(args.seed1, "region_order.txt"))
              if l.strip()]

    rng = np.random.RandomState(0)
    idx = rng.choice(b1.shape[0], size=args.n, replace=False)

    print("subsample n = {0}, d = {1}".format(args.n, b1.shape[2]))
    print("Gram matrix would be {0:.2f} GB at full n = {1}\n".format(
        b1.shape[0] ** 2 * 8 / 1e9, b1.shape[0]))
    print("{0:26s} {1:>12s} {2:>12s} {3:>12s}".format(
        "region", "feature", "gram", "|difference|"))

    worst = 0.0
    for r in range(min(args.regions, b1.shape[1])):
        X = b1[idx, r, :].astype(np.float64)
        Y = b2[idx, r, :].astype(np.float64)
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        a = cka_feature(X, Y)
        b = cka_gram(X, Y)
        d = abs(a - b)
        worst = max(worst, d)
        print("{0:26s} {1:12.9f} {2:12.9f} {3:12.2e}".format(
            labels[r][:26], a, b, d))

    print("\nworst |difference| = {0:.2e}".format(worst))
    print("VERDICT: {0}".format(
        "identical to floating-point precision"
        if worst < 1e-10 else "DIFFER - investigate"))
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
