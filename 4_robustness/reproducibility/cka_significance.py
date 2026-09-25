"""Is the seed-to-seed embedding agreement actually high, or just above chance?

At n = 22,999 a permutation test is a weak bar: any consistent structure clears
it. Three references are computed so the observed CKA can be placed on a scale
rather than merely declared significant.

  permutation   subject labels of seed 2 shuffled. Breaks correspondence, keeps
                each run's spectrum. Estimates the chance floor -- and, since
                CKA is biased upward in finite samples, the null mean IS that
                bias.
  cross-region  CKA(seed1 region i, seed2 region j) for i != j. Two genuine BRE
                representations of different structures from independently
                trained models. This is the null that answers "would any two
                brain embeddings look like this?", and it is far stricter than
                permutation because it preserves realistic spectra AND realistic
                subject-level covariance.
  surrogate     random loadings with each region's singular values retained.
                Isolates how much agreement is forced by spectral shape alone --
                the thalamus check, since a degenerate spectrum depresses CKA
                without implying the runs disagree.

Bootstrap over subjects gives the precision of the observed value.

For CCA no simulation is needed for the floor: for two independent n x d
Gaussians the squared canonical correlations occupy a Jacobi/MANOVA bulk with
upper edge (sqrt(a(1-b)) + sqrt(b(1-a)))^2, a = b = d/n, so every canonical
correlation under independence is bounded. That bound is reported alongside.

Usage:
  python cka_significance.py --seed1 <dir> --seed2 <dir> --out <csv>
  python cka_significance.py ... --stages cross boot        # the fast pair
  python cka_significance.py ... --n-perm 1000 --n-boot 1000
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np


# Whole-tissue masks, not subcortical structures. concordance_embeddings.py
# drops them and the 16-region analysis excludes them, so they are excluded here
# too -- their CKA runs higher than any structure's and would inflate both the
# diagonal and the cross-region null.
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
    ok = np.isfinite(M).all(axis=1)
    M = M[ok]
    return M - M.mean(axis=0), ok


def cka_from_parts(X, Y, nx, ny):
    """Linear CKA given pre-centred X, Y and their pre-computed denominators."""
    num = float((X.T.dot(Y) ** 2).sum())
    return num / (nx * ny)


def self_norm(X):
    return np.linalg.norm(X.T.dot(X), "fro")


def canonical_correlations(X, Y):
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    try:
        s = np.linalg.svd(qx.T.dot(qy), compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as sp_svd
        s = sp_svd(qx.T.dot(qy), compute_uv=False, lapack_driver="gesvd")
    return np.clip(s, 0.0, 1.0)


def manova_upper_edge(n, d):
    """Largest canonical correlation attainable under independence.

    Jacobi/MANOVA bulk edge for two independent n x d Gaussian matrices.
    """
    a = b = float(d) / n
    lam = (np.sqrt(a * (1 - b)) + np.sqrt(b * (1 - a))) ** 2
    return float(np.sqrt(lam))


def spectrum_matched(X, rng):
    """Random subject loadings, singular values of X retained.

    Q from a QR of Gaussian noise supplies orthonormal loadings, so the
    surrogate has X's spectrum but no relationship to anything else.
    """
    n, d = X.shape
    s = np.linalg.svd(X, compute_uv=False)
    Q, _ = np.linalg.qr(rng.standard_normal((n, d)))
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * s).dot(V.T)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-out", default=None,
                    help="optional path for the full 16x16 cross-region matrix")
    ap.add_argument("--stages", nargs="+",
                    default=["cross", "boot", "perm", "surr"],
                    choices=["cross", "boot", "perm", "surr"])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-surr", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    b1, ids1, regions = load_seed(args.seed1)
    b2, ids2, _ = load_seed(args.seed2)
    if ids1 != ids2:
        print("subject orders differ; aligning on the intersection")
        common = [i for i in ids1 if i in set(ids2)]
        idx1 = [ids1.index(i) for i in common]
        idx2 = [ids2.index(i) for i in common]
        b1, b2 = b1[idx1], b2[idx2]

    keep = [i for i, r in enumerate(regions) if r[2] not in EXCLUDE_LABELS]
    if len(keep) != len(regions):
        print("excluding {0}: {1}".format(
            ", ".join(EXCLUDE_LABELS),
            len(regions) - len(keep)))
    b1 = b1[:, keep, :]
    b2 = b2[:, keep, :]
    labels = [regions[i][2] for i in keep]
    n_reg = len(labels)
    print("subjects: {0:,}   regions: {1}   dims: {2}".format(
        b1.shape[0], n_reg, b1.shape[2]), flush=True)

    # Centre every region once; the CKA denominator does not depend on the
    # pairing, so it is computed once per region too.
    X1, X2, N1, N2 = [], [], [], []
    for r in range(n_reg):
        a, _ = centre(b1[:, r, :])
        c, _ = centre(b2[:, r, :])
        X1.append(a)
        X2.append(c)
        N1.append(self_norm(a))
        N2.append(self_norm(c))

    n_sub, d = X1[0].shape
    edge = manova_upper_edge(n_sub, d)
    print("MANOVA upper edge (max canonical r under independence): "
          "{0:.4f}\n".format(edge), flush=True)

    observed = [cka_from_parts(X1[r], X2[r], N1[r], N2[r])
                for r in range(n_reg)]

    rows = [{"region": labels[r], "cka_observed": observed[r]}
            for r in range(n_reg)]

    # ---- cross-region null -------------------------------------------------
    cross = None
    if "cross" in args.stages:
        print("[cross] full {0}x{0} CKA matrix".format(n_reg), flush=True)
        cross = np.zeros((n_reg, n_reg))
        for i in range(n_reg):
            for j in range(n_reg):
                cross[i, j] = cka_from_parts(X1[i], X2[j], N1[i], N2[j])
        off = cross[~np.eye(n_reg, dtype=bool)]
        for r in range(n_reg):
            others = [cross[r, j] for j in range(n_reg) if j != r]
            rows[r]["cross_mean"] = float(np.mean(others))
            rows[r]["cross_sd"] = float(np.std(others, ddof=1))
            rows[r]["cross_max"] = float(np.max(others))
            rows[r]["ratio_vs_cross"] = observed[r] / float(np.mean(others))
        print("  diagonal mean {0:.4f} | off-diagonal mean {1:.4f} | "
              "ratio {2:.2f}x".format(np.mean(np.diag(cross)), off.mean(),
                                      np.mean(np.diag(cross)) / off.mean()),
              flush=True)
        if args.cross_out:
            import pandas as pd
            pd.DataFrame(cross, index=labels, columns=labels).to_csv(
                args.cross_out)
            print("  wrote {0}".format(args.cross_out), flush=True)

    # ---- bootstrap ---------------------------------------------------------
    if "boot" in args.stages:
        print("[boot] {0} resamples".format(args.n_boot), flush=True)
        for r in range(n_reg):
            vals = np.empty(args.n_boot)
            A, B = X1[r], X2[r]
            for b in range(args.n_boot):
                idx = rng.integers(0, n_sub, n_sub)
                Ab = A[idx]
                Bb = B[idx]
                Ab = Ab - Ab.mean(0)
                Bb = Bb - Bb.mean(0)
                vals[b] = cka_from_parts(Ab, Bb, self_norm(Ab), self_norm(Bb))
            rows[r]["boot_lo"] = float(np.percentile(vals, 2.5))
            rows[r]["boot_hi"] = float(np.percentile(vals, 97.5))
            print("  {0:28s} {1:.4f} [{2:.4f}, {3:.4f}]".format(
                labels[r][:28], observed[r], rows[r]["boot_lo"],
                rows[r]["boot_hi"]), flush=True)

    # ---- permutation -------------------------------------------------------
    if "perm" in args.stages:
        print("[perm] {0} subject permutations".format(args.n_perm), flush=True)
        for r in range(n_reg):
            A, B = X1[r], X2[r]
            nb = N2[r]           # permuting rows changes neither mean nor norm
            vals = np.empty(args.n_perm)
            for p in range(args.n_perm):
                vals[p] = cka_from_parts(A, B[rng.permutation(n_sub)],
                                         N1[r], nb)
            rows[r]["perm_mean"] = float(vals.mean())
            rows[r]["perm_sd"] = float(vals.std(ddof=1))
            rows[r]["perm_p"] = float((1 + int((vals >= observed[r]).sum()))
                                      / (args.n_perm + 1))
            rows[r]["perm_z"] = float((observed[r] - vals.mean())
                                      / (vals.std(ddof=1) + 1e-300))
            print("  {0:28s} obs {1:.4f} | null {2:.5f}+-{3:.5f} | p {4:.4g}"
                  .format(labels[r][:28], observed[r], vals.mean(),
                          vals.std(ddof=1), rows[r]["perm_p"]), flush=True)

    # ---- spectrum-matched surrogates --------------------------------------
    if "surr" in args.stages:
        print("[surr] {0} spectrum-matched surrogates".format(args.n_surr),
              flush=True)
        for r in range(n_reg):
            vals = np.empty(args.n_surr)
            for s in range(args.n_surr):
                A = spectrum_matched(X1[r], rng)
                B = spectrum_matched(X2[r], rng)
                A = A - A.mean(0)
                B = B - B.mean(0)
                vals[s] = cka_from_parts(A, B, self_norm(A), self_norm(B))
            rows[r]["surr_mean"] = float(vals.mean())
            rows[r]["surr_sd"] = float(vals.std(ddof=1))
            print("  {0:28s} surrogate {1:.5f}+-{2:.5f}".format(
                labels[r][:28], vals.mean(), vals.std(ddof=1)), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    if "perm" in args.stages:
        try:
            from statsmodels.stats.multitest import multipletests
            df["perm_q"] = multipletests(df["perm_p"], method="fdr_bh")[1]
        except ImportError:
            print("statsmodels unavailable; skipping FDR", flush=True)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("observed CKA        mean {0:.4f}  range {1:.4f}-{2:.4f}".format(
        np.mean(observed), np.min(observed), np.max(observed)))
    if "cross" in args.stages:
        print("cross-region null   mean {0:.4f}   -> observed is {1:.2f}x "
              "the cross-region floor".format(
                  df["cross_mean"].mean(),
                  np.mean(observed) / df["cross_mean"].mean()))
    if "perm" in args.stages:
        print("permutation null    mean {0:.5f}  (this is the finite-sample "
              "bias)".format(df["perm_mean"].mean()))
    if "surr" in args.stages:
        print("spectrum surrogate  mean {0:.5f}".format(df["surr_mean"].mean()))
    print("max canonical r under independence: {0:.4f}".format(edge))
    print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
