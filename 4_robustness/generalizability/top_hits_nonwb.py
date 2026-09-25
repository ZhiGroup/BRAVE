"""Locate the genome-wide hits from the non-WB pass, to tell signal from stratification.

The genome-wide run reported 29-233 significant variants per region with min p as low as
1e-299, in a cohort of 6,470. Discovery, at N = 22,878, topped out near 1e-246. Chi-square
scales with N, so a 3.5x smaller cohort should not produce stronger tails. Either these
are real, or they are artefacts -- most plausibly ancestry stratification, since the
cohort is deliberately heterogeneous (EUR 5,104 / SAS 676 / AFR 470 / EAS 224) and is
adjusted with only 10 PCs.

The diagnostic is WHERE the hits are. Real subcortical signal should land on the
discovery loci. Stratification lands on the classic ancestry-differentiated regions:
LCT (2q21), HLA/MHC (6p21-22), SLC24A5 (15q21), EDAR (2q13), and similar.

Usage
-----
    python top_hits_nonwb.py --top 5
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

# well-known ancestry-differentiated / long-range-LD regions (hg19), for labelling only
FLAGS = [
    ("LCT", 2, 134_000_000, 138_000_000),
    ("EDAR", 2, 108_000_000, 110_000_000),
    ("MHC", 6, 25_000_000, 34_000_000),
    ("SLC24A5", 15, 47_000_000, 49_000_000),
    ("SLC45A2", 5, 33_000_000, 34_500_000),
    ("chr8inv", 8, 7_000_000, 13_000_000),
    ("APOE", 19, 44_400_000, 46_500_000),
]


def flag(c, p):
    hits = [n for n, ch, lo, hi in FLAGS if ch == c and lo <= p <= hi]
    return ",".join(hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chi2", default=os.path.join(GATE_DIR, "genomewide_nonwb",
                                                   "chi2.npy"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--n-dim", type=int, default=128)
    args = ap.parse_args()

    import pandas as pd
    vo = pd.read_parquet(os.path.join(NONWB, "geno_variant_order_t1only.parquet"))
    chrom = vo["chrom"].astype(str).str.lstrip("0").values
    pos = vo["pos"].astype(np.int64).values
    rsid = vo["rsid"].astype(str).values

    labels = []
    with open(os.path.join(GATE_DIR, "bre_nonwb", "regions.txt")) as fh:
        for line in fh:
            labels.append(line.rstrip("\n").split("\t")[2])

    # our discovery lead positions, to ask whether hits land on them
    leads = set()
    with open(os.path.join(GATE_DIR, "all_region_leads.csv")) as fh:
        for r in csv.DictReader(fh):
            leads.add((str(r["chr"]).lstrip("0"), int(r["pos"])))

    chi = np.load(args.chi2, mmap_mode="r")
    print("chi2 array: {0}\n".format(chi.shape))

    all_rows = []
    for i, lab in enumerate(labels):
        row = np.asarray(chi[i, :])
        if not np.isfinite(row).any():
            continue
        p = chi2_dist.sf(row, args.n_dim)
        sig = np.where(p < THRESH_GW)[0]
        if len(sig) == 0:
            continue
        order = sig[np.argsort(p[sig])][:args.top]
        n_on_lead = sum(1 for j in sig if (chrom[j], int(pos[j])) in leads)
        flagged = {}
        for j in sig:
            f = flag(int(chrom[j]) if chrom[j].isdigit() else 0, int(pos[j]))
            if f:
                flagged[f] = flagged.get(f, 0) + 1
        print("{0:<28s} {1:4d} GW-sig | on a discovery lead: {2:3d} | "
              "in flagged regions: {3}".format(
                  lab, len(sig), n_on_lead,
                  ", ".join("{0}={1}".format(k, v) for k, v in
                            sorted(flagged.items())) or "none"))
        for j in order:
            print("    {0:<14s} chr{1}:{2:<10d} p={3:.2e}  {4}".format(
                rsid[j], chrom[j], int(pos[j]), p[j],
                flag(int(chrom[j]) if chrom[j].isdigit() else 0, int(pos[j]))))
            all_rows.append({"region": lab, "rsid": rsid[j], "chr": chrom[j],
                             "pos": int(pos[j]), "p": float(p[j]),
                             "chi2": float(row[j]),
                             "flag": flag(int(chrom[j]) if chrom[j].isdigit() else 0,
                                          int(pos[j])),
                             "on_discovery_lead": (chrom[j], int(pos[j])) in leads})

    if all_rows:
        out = os.path.join(GATE_DIR, "genomewide_nonwb", "top_hits.csv")
        with open(out, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print("\nwrote {0}".format(out))


if __name__ == "__main__":
    main()
