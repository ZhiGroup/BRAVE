"""Per-region agreement between BRE sets: seed 1, seed 2 and the published run.

Individual embedding dimensions are not expected to correspond across runs --
nothing anchors dimension k of one run to dimension k of another. The meaningful
question is whether the runs span the same 128-dimensional subspace, so the
headline metric is canonical correlation; per-dimension correlation is reported
alongside to show how misleading it is on its own.

Metrics per region:
  mean_cca_r   mean canonical correlation over all 128 components (1 = same
               subspace)
  cca_r_top10  mean over the 10 leading components
  procrustes   residual after optimal scaling/rotation, 0 = identical up to a
               similarity transform
  dim_matched  mean |r| between dimension k of A and dimension k of B
  dim_bestmatch mean over dimensions of the best |r| against any dimension of B
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.


# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import glob
import os
import pickle
import sys

import numpy as np

PUB_DICT = ("<EXTERNAL: PIXEL_EMBEDDING>/"
            "RBGWAS_RELATED_DATA/SUBJECT_DICTS/<EXTERNAL: encoder checkpoint>"
            "discovery_local")
EXCLUDE_LABELS = ("GM", "WM")


def _svdvals(M):
    """Singular values, falling back to the slower but sturdier LAPACK driver.

    The divide-and-conquer driver numpy uses by default can fail to converge on
    near-rank-deficient blocks, which happens for the smaller structures.
    """
    try:
        return np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as sp_svd
        return sp_svd(M, compute_uv=False, lapack_driver="gesvd")


def canonical_correlations(X, Y):
    """Canonical correlations via QR + SVD; no sklearn dependency."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    s = _svdvals(qx.T.dot(qy))
    return np.clip(s, 0.0, 1.0)


def linear_cka(X, Y):
    """Centered Kernel Alignment with a linear kernel (Kornblith et al. 2019).

        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Invariant to orthogonal transforms and isotropic scaling, but NOT to
    arbitrary invertible linear maps -- which is the point. CCA is invariant to
    those too, so it can report high agreement by stretching low-variance
    directions to match noise. CKA cannot, so it is the more conservative and
    generally more trustworthy single number for representation similarity.

    Computed through the feature-space (d x d) form, so cost is O(n d^2) rather
    than the O(n^2) of the Gram-matrix form -- important at n = 22,999.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 10:
        return float("nan")
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    xty = X.T.dot(Y)
    num = float((xty ** 2).sum())
    den = (np.linalg.norm(X.T.dot(X), "fro")
           * np.linalg.norm(Y.T.dot(Y), "fro"))
    if den == 0:
        return float("nan")
    return num / den


def procrustes_disparity(X, Y):
    """Residual after optimal translation, uniform scaling and rotation."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 10:
        return float("nan")
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    nx = np.linalg.norm(X)
    ny = np.linalg.norm(Y)
    if nx == 0 or ny == 0:
        return float("nan")
    X = X / nx
    Y = Y / ny
    s = _svdvals(X.T.dot(Y))
    return float(1.0 - (s.sum() ** 2))


def dim_correlations(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[ok], Y[ok]
    if X.shape[0] < 10:
        return float("nan"), float("nan")
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    C = Xz.T.dot(Yz) / X.shape[0]
    matched = float(np.mean(np.abs(np.diag(C))))
    best = float(np.mean(np.max(np.abs(C), axis=1)))
    return matched, best


def load_seed(path):
    bre = np.load(os.path.join(path, "bre_discovery.npy"))
    ids = [l.strip() for l in open(os.path.join(path, "subject_ids.txt"))
           if l.strip()]
    regions = [l.rstrip("\n").split("\t")
               for l in open(os.path.join(path, "region_order.txt"))
               if l.strip()]
    return bre, ids, regions


def load_published(region_id, label):
    pat = os.path.join(PUB_DICT, "{0}_*_mean.pkl".format(region_id))
    hits = [h for h in glob.glob(pat) if label.replace(" ", "") in
            os.path.basename(h).replace(" ", "")]
    if not hits:
        hits = glob.glob(pat)
    if not hits:
        return None
    with open(hits[0], "rb") as fh:
        raw = pickle.load(fh)
    out = {}
    for k, v in raw.items():
        vec = v[1] if isinstance(v, tuple) else v
        out[str(k).split("_")[0]] = np.asarray(vec).ravel()
    return out


def compare(name, A, B, rows):
    try:
        cc = canonical_correlations(A, B)
    except Exception as exc:
        print("    {0}: canonical correlation failed ({1})".format(
            name, str(exc)[:60]))
        cc = np.array([np.nan])
    matched, best = dim_correlations(A, B)
    nonfinite = int((~(np.isfinite(np.asarray(A, dtype=np.float64)).all(axis=1)
                       & np.isfinite(np.asarray(B, dtype=np.float64)).all(axis=1)
                       )).sum())
    if nonfinite:
        print("    {0}: dropped {1} subjects with non-finite embeddings"
              .format(name, nonfinite))
    rows.append({
        "comparison": name,
        "n_nonfinite_dropped": nonfinite,
        "cka": linear_cka(A, B),
        "mean_cca_r": float(cc.mean()),
        "cca_r_top10": float(cc[:10].mean()),
        "procrustes": procrustes_disparity(A, B),
        "dim_matched": matched,
        "dim_bestmatch": best,
    })
    return rows[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-published", action="store_true")
    args = ap.parse_args()

    b1, ids1, regions = load_seed(args.seed1)
    b2, ids2, _ = load_seed(args.seed2)
    if ids1 != ids2:
        print("subject orders differ; aligning on the intersection")
        common = [i for i in ids1 if i in set(ids2)]
        idx1 = [ids1.index(i) for i in common]
        idx2 = [ids2.index(i) for i in common]
        b1, b2, ids1 = b1[idx1], b2[idx2], common
    eids = [s.split("_")[0] for s in ids1]

    print("subjects: {0:,}   regions: {1}".format(len(ids1), len(regions)))

    all_rows = []
    for r_i, (dict_id, region_id, label) in enumerate(regions):
        if label in EXCLUDE_LABELS:
            continue
        A = b1[:, r_i, :].astype(np.float64)
        B = b2[:, r_i, :].astype(np.float64)
        print("\n=== {0}".format(label))
        row = compare("seed1_vs_seed2", A, B, [])
        row["region"] = label
        all_rows.append(row)
        print("  seed1 vs seed2   CKA={0:.4f}  mean_cca={1:.4f}  top10={2:.4f}  "
              "procrustes={3:.4f}  dim_matched={4:.3f}"
              .format(row["cka"], row["mean_cca_r"], row["cca_r_top10"],
                      row["procrustes"], row["dim_matched"]))

        if args.skip_published:
            continue
        pub = load_published(region_id, label)
        if not pub:
            print("  published: region not found")
            continue
        keep = [i for i, e in enumerate(eids) if e in pub]
        if len(keep) < 100:
            print("  published: only {0} subjects joined".format(len(keep)))
            continue
        P = np.vstack([pub[eids[i]] for i in keep]).astype(np.float64)
        for nm, M in (("seed1_vs_published", b1[keep][:, r_i, :]),
                      ("seed2_vs_published", b2[keep][:, r_i, :])):
            row = compare(nm, M.astype(np.float64), P, [])
            row["region"] = label
            all_rows.append(row)
            print("  {0:19s} CKA={1:.4f}  mean_cca={2:.4f}  top10={3:.4f}  "
                  "procrustes={4:.4f}  dim_matched={5:.3f}"
                  .format(nm, row["cka"], row["mean_cca_r"],
                          row["cca_r_top10"], row["procrustes"],
                          row["dim_matched"]))

    print("\n" + "=" * 74)
    print("SUMMARY (mean over regions)")
    print("=" * 74)
    for comp in ("seed1_vs_seed2", "seed1_vs_published", "seed2_vs_published"):
        sub = [r for r in all_rows if r["comparison"] == comp]
        if not sub:
            continue
        print("  {0:20s} CKA={1:.4f}  mean_cca={2:.4f}  top10={3:.4f}  "
              "procrustes={4:.4f}  dim_matched={5:.3f}  dim_best={6:.3f}  "
              "(n={7})".format(
                  comp,
                  np.mean([r["cka"] for r in sub]),
                  np.mean([r["mean_cca_r"] for r in sub]),
                  np.mean([r["cca_r_top10"] for r in sub]),
                  np.mean([r["procrustes"] for r in sub]),
                  np.mean([r["dim_matched"] for r in sub]),
                  np.mean([r["dim_bestmatch"] for r in sub]),
                  len(sub)))

    if args.out:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        cols = ["region", "comparison", "cka", "mean_cca_r", "cca_r_top10",
                "procrustes", "dim_matched", "dim_bestmatch",
                "n_nonfinite_dropped"]
        df[cols].to_csv(args.out, index=False)
        print("\nwrote {0}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
