"""Summarise the non-WB genome-wide pass under a MAF filter.

The unfiltered pass is dominated by ultra-rare variants: every top hit has MAF <= 0.0005,
i.e. a handful of carriers out of 12,940 alleles. The chi-square values are CORRECT --
recomputed in float64 they agree 48/48, and the 276 targeted variants agree at
correlation 1.00000000 -- but a 128-degree-of-freedom asymptotic null is not valid at
that minor allele count, so those p-values are meaningless rather than wrong.

This applies the MAF filter any GWAS would and recomputes lambda_GC and the significance
counts on the analysable variant set. No need to re-run the 40-minute pass; chi2.npy is
reused.

Usage
-----
    python summarise_genomewide_nonwb.py --maf 0.01
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import csv
import os

import numpy as np
from scipy.stats import chi2 as chi2_dist

GATE_DIR = "<EXTERNAL: reproducibility>"
NONWB = "<EXTERNAL: nonwb_replication>"
THRESH_GW = 3.125e-9
SKIP = {"gm", "wm"}


def norm_label(s):
    return "".join(str(s).split()).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maf", type=float, default=0.01)
    ap.add_argument("--n-dim", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(GATE_DIR, "genomewide_nonwb",
                                                  "lambda_gc_maf_filtered.csv"))
    args = ap.parse_args()

    maf = np.load(os.path.join(NONWB, "gwas_t1only", "nonwb_maf_all.npy"))
    keep = maf >= args.maf
    print("variants total    : {0:,}".format(len(maf)))
    print("MAF >= {0:<6}     : {1:,} ({2:.1f}%)".format(
        args.maf, int(keep.sum()), 100.0 * keep.mean()))

    labels = []
    with open(os.path.join(GATE_DIR, "bre_nonwb", "regions.txt")) as fh:
        for line in fh:
            labels.append(line.rstrip("\n").split("\t")[2])

    chi_arr = np.load(os.path.join(GATE_DIR, "genomewide_nonwb", "chi2.npy"),
                      mmap_mode="r")
    med_null = float(chi2_dist.ppf(0.5, args.n_dim))

    rows = []
    print("\n{0:<28s} {1:>9s} {2:>9s} {3:>9s} {4:>11s}".format(
        "region", "lam_raw", "lam_MAF", "GW sig", "min p"))
    for i, lab in enumerate(labels):
        if norm_label(lab) in SKIP:
            continue
        row = np.asarray(chi_arr[i, :])
        fin = np.isfinite(row)
        lam_raw = float(np.median(row[fin]) / med_null)
        sel = fin & keep
        c = row[sel]
        lam = float(np.median(c) / med_null)
        p = chi2_dist.sf(c, args.n_dim)
        n_gw = int((p < THRESH_GW).sum())
        rows.append({"region": lab, "lambda_raw": lam_raw, "lambda_maf": lam,
                     "n_tested": int(sel.sum()), "n_gw_sig": n_gw,
                     "min_p": float(p.min())})
        print("{0:<28s} {1:9.4f} {2:9.4f} {3:9,} {4:11.2e}".format(
            lab, lam_raw, lam, n_gw, float(p.min())))

    lam = np.array([r["lambda_maf"] for r in rows])
    tot = sum(r["n_gw_sig"] for r in rows)
    print("\nlambda_GC (MAF filtered): min {0:.4f}  median {1:.4f}  max {2:.4f}".format(
        lam.min(), np.median(lam), lam.max()))
    print("regions with lambda > 1.05 : {0}".format(int((lam > 1.05).sum())))
    print("total genome-wide significant across regions: {0:,}".format(tot))

    with open(args.out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote {0}".format(args.out))


if __name__ == "__main__":
    main()
