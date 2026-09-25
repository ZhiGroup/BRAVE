"""Genome-wide JAGWAS joint test in the non-WB cohort (extension 1).

Purpose: produce lambda_GC -- which 276 targeted variants cannot estimate -- and the
per-region Manhattan panels.

Design. The dosage matrix is shared across regions, so ONE sequential pass over the
115 GB memmap serves all 16; only the residualised phenotypes and the correlation matrix
differ per region. Covariate residualisation uses a precomputed orthonormal basis Q from
a QR of the covariate matrix, so per block it is two matmuls rather than a fresh lstsq
(the benchmark showed lstsq-per-block was 6.4 s of an 11.7 s block).

Regions whose BRE has zero-vector subjects (no mask voxels -- the 5 Accumbens cells) use
their own subject mask, so residualisation is done once per DISTINCT mask per block, not
once per region.

Statistic, per variant per region, matching jagwas_joint_replication.py:

    z    = beta_j / se_j  over the 128 dimensions (Frisch-Waugh)
    chi2 = z' R_inv z      with eigenvalues clamped at max(eig) * 1e-5
    p    = upper tail, df = 128   (correct df, decision N7)

Outputs
-------
    chi2.npy        (16, 8931083) float32
    lambda_gc.csv   per-region lambda_GC and genome-wide significant counts

Usage
-----
    python genomewide_nonwb_jagwas.py --block 40000
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.stats import chi2 as chi2_dist

GATE_DIR = "<EXTERNAL: reproducibility>"
NONWB = "<EXTERNAL: nonwb_replication>"
GENO = "<EXTERNAL: geno_e_t1only>"
N_VAR, N_SAMPLES = 8931083, 6474
THRESH_GW = 3.125e-9

# the 16 GWAS regions, by BRE label; GM and WM are extracted but not analysed
SKIP_LABELS = {"gm", "wm"}


def norm_label(s):
    return "".join(str(s).split()).lower()


def clamped_inverse(cor):
    ev, evec = np.linalg.eigh(cor)
    tol = ev.max() * 1e-5
    n_clamped = int((ev < tol).sum())
    inv = np.where(ev < tol, 1.0 / tol, 1.0 / ev)
    return evec.dot(np.diag(inv)).dot(evec.T), n_clamped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, default=40000)
    ap.add_argument("--limit-variants", type=int, default=None,
                    help="stop early (smoke test)")
    ap.add_argument("--out-dir", default=os.path.join(GATE_DIR, "genomewide_nonwb"))
    ap.add_argument("--report-every", type=int, default=20)
    args = ap.parse_args()

    bre = np.load(os.path.join(GATE_DIR, "bre_nonwb", "bre_nonwb.npy"))
    sample_index = np.load(os.path.join(GATE_DIR, "bre_nonwb", "sample_index.npy"))
    labels = []
    with open(os.path.join(GATE_DIR, "bre_nonwb", "regions.txt")) as fh:
        for line in fh:
            labels.append(line.rstrip("\n").split("\t")[2])

    cov = np.load(os.path.join(NONWB, "covarQ_t1only.npy"))[sample_index, :]
    cov = np.column_stack([np.ones(cov.shape[0]), cov]).astype(np.float64)
    n_sub, n_cov = cov.shape
    n_dim = bre.shape[2]

    keep_regions = [i for i, l in enumerate(labels)
                    if norm_label(l) not in SKIP_LABELS]
    print("subjects   : {0:,}".format(n_sub))
    print("regions    : {0} analysed of {1}".format(len(keep_regions), len(labels)))
    print("dimensions : {0}".format(n_dim))

    # ---- per-region setup, grouped by distinct subject mask ----
    groups = {}          # mask_key -> {"mask", "Q", "df", "regions": [...]}
    meta = []
    for r in keep_regions:
        Y = bre[:, r, :].astype(np.float64)
        mask = np.abs(Y).sum(axis=1) > 0
        key = mask.tobytes()
        if key not in groups:
            C = cov[mask]
            Q, _ = np.linalg.qr(C)
            groups[key] = {"mask": mask, "Q": Q.astype(np.float32),
                           "df": int(mask.sum()) - n_cov - 1, "regions": []}
        g = groups[key]
        Ym = Y[mask]
        Yr = Ym - g["Q"].astype(np.float64).dot(g["Q"].astype(np.float64).T.dot(Ym))
        Rinv, n_clamped = clamped_inverse(np.corrcoef(Yr, rowvar=False))
        g["regions"].append({
            "row": r, "label": labels[r],
            "Yr": Yr.astype(np.float32),
            "Rinv": Rinv.astype(np.float32),
            "ynorm": (Yr ** 2).sum(axis=0).astype(np.float32),
        })
        meta.append({"row": r, "label": labels[r], "n": int(mask.sum()),
                     "n_clamped": n_clamped})
        print("  {0:<28s} n={1:,}  clamped {2}/{3}".format(
            labels[r], int(mask.sum()), n_clamped, n_dim))
    print("distinct subject masks: {0}".format(len(groups)))

    n_var = args.limit_variants or N_VAR
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    out = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "chi2.npy"), mode="w+",
        dtype=np.float32, shape=(len(labels), n_var))
    out[:] = np.nan

    geno = np.memmap(GENO, dtype="float16", mode="r", shape=(N_VAR, N_SAMPLES))
    n_blocks = int(np.ceil(float(n_var) / args.block))
    print("\nblocks: {0:,} of {1:,} variants\n".format(n_blocks, n_var))
    sys.stdout.flush()

    t0 = time.time()
    n_imputed = 0
    for b in range(n_blocks):
        lo = b * args.block
        hi = min(lo + args.block, n_var)
        G_all = np.asarray(geno[lo:hi, :], dtype=np.float32)[:, sample_index].T
        bad = ~np.isfinite(G_all)
        if bad.any():
            n_imputed += int(bad.sum())
            col_mean = np.nanmean(np.where(bad, np.nan, G_all), axis=0)
            G_all[bad] = np.take(col_mean, np.where(bad)[1])

        for g in groups.values():
            G = G_all[g["mask"], :]
            Q = g["Q"]
            Gr = G - Q.dot(Q.T.dot(G))
            gg = (Gr ** 2).sum(axis=0)
            gg[gg <= 0] = np.nan
            for reg in g["regions"]:
                B = reg["Yr"].T.dot(Gr) / gg
                rss = reg["ynorm"][:, None] - (B ** 2) * gg[None, :]
                np.maximum(rss, 1e-30, out=rss)
                se = np.sqrt(rss / g["df"] / gg[None, :])
                Z = B / se
                out[reg["row"], lo:hi] = (reg["Rinv"].dot(Z) * Z).sum(axis=0)

        if (b + 1) % args.report_every == 0 or b + 1 == n_blocks:
            el = time.time() - t0
            done = hi
            rate = done / el
            print("  {0:>10,}/{1:,}  {2:.1f} min elapsed  eta {3:.1f} min".format(
                done, n_var, el / 60, (n_var - done) / rate / 60 if rate else 0))
            sys.stdout.flush()

    out.flush()
    print("\nnon-finite dosages imputed: {0:,}".format(n_imputed))

    # ---- lambda_GC and significance counts ----
    med_null = float(chi2_dist.ppf(0.5, n_dim))
    rows = []
    print("\n{0:<28s} {1:>9s} {2:>12s} {3:>10s}".format(
        "region", "lambda_GC", "GW sig", "min p"))
    for m in meta:
        chi = out[m["row"], :]
        chi = chi[np.isfinite(chi)]
        lam = float(np.median(chi) / med_null)
        p = chi2_dist.sf(chi, n_dim)
        n_gw = int((p < THRESH_GW).sum())
        pmin = float(p.min()) if len(p) else float("nan")
        rows.append({"region": m["label"], "n": m["n"], "n_clamped": m["n_clamped"],
                     "lambda_gc": lam, "n_gw_sig": n_gw, "min_p": pmin,
                     "n_variants": int(len(chi))})
        print("{0:<28s} {1:9.4f} {2:12,} {3:10.2e}".format(
            m["label"], lam, n_gw, pmin))

    lams = np.array([r["lambda_gc"] for r in rows])
    print("\nlambda_GC across {0} regions: min {1:.4f}  median {2:.4f}  max {3:.4f}"
          .format(len(lams), lams.min(), np.median(lams), lams.max()))
    print("regions with lambda_GC > 1.05: {0}".format(int((lams > 1.05).sum())))

    with open(os.path.join(args.out_dir, "lambda_gc.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote {0}".format(args.out_dir))
    print("total {0:.1f} min".format((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
