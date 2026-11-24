"""Label the rebuilt 1000G panel with dbSNP rsIDs.

The 20130502 v5b VCFs carry no rsIDs -- every ID field is '.', so
build_fuma_panel.py had to synthesise chr:pos:ref:alt identifiers. Those work
(locus positions reproduce exactly) but make the output unreadable and unusable
downstream, where lead SNPs must be named.

This rewrites each chromosome's .bim ID column from dbSNP b151 GRCh37, matching
on CHR:POS. Variants with no dbSNP entry keep their synthetic ID.

The same dbSNP map must then be passed to fuma_clump.py as --ref-map, so the
GWAS is relabelled from the identical source. Panel IDs and harmonised GWAS IDs
have to agree: plink joins the association file to the .bim by ID, and a
mismatch silently drops the variant.

Usage:
  python assign_panel_rsids.py --dry-run     # report coverage only
  python assign_panel_rsids.py
"""
from __future__ import print_function

import argparse
import os
import shutil
import sys

import pandas as pd

DBSNP_MAP = "<work_dir>/dbsnp/dbsnp_b151_ref_map.parquet"
PANEL_ROOT = "<path/to>/g1000_phase3"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-root", default=PANEL_ROOT)
    ap.add_argument("--dbsnp-map", default=DBSNP_MAP)
    ap.add_argument("--chroms", nargs="+", type=int, default=list(range(1, 23)))
    ap.add_argument("--from-gwas", default=None,
                    help="Take rsIDs from a fastGWA-format sumstats file "
                         "instead of dbSNP. Preferred: the dbSNP distribution "
                         "on disk is the common-only subset and covers under "
                         "half the panel, whereas the GWAS covers every variant "
                         "that could ever be reported as a lead -- and it is "
                         "the same naming FUMA's output uses.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.from_gwas:
        print("[gwas] loading {0}".format(args.from_gwas), flush=True)
        db = pd.read_csv(args.from_gwas, sep="\t", usecols=["CHR", "SNP", "POS"],
                         dtype={"CHR": str, "SNP": str})
        print("[gwas] {0:,} variants".format(len(db)), flush=True)
    else:
        print("[dbsnp] loading {0}".format(args.dbsnp_map), flush=True)
        db = pd.read_parquet(args.dbsnp_map, columns=["CHR", "POS", "SNP"])
        db["CHR"] = db["CHR"].astype(str)
        print("[dbsnp] {0:,} variants".format(len(db)), flush=True)

    tot_n = tot_hit = 0
    for chrom in args.chroms:
        bim = os.path.join(args.panel_root, "EUR.chr{0}.bim".format(chrom))
        if not os.path.exists(bim):
            print("[chr{0}] missing, skipping".format(chrom), flush=True)
            continue

        cols = ["CHR", "SNP", "CM", "POS", "A1", "A2"]
        df = pd.read_csv(bim, sep="\t", header=None, names=cols,
                         dtype={"CHR": str, "SNP": str, "A1": str, "A2": str})
        sub = db[db["CHR"] == str(chrom)][["POS", "SNP"]]
        sub = sub.drop_duplicates("POS").rename(columns={"SNP": "RS"})
        merged = df.merge(sub, on="POS", how="left")

        hit = merged["RS"].notna()
        # Only positions unique within the panel may take a dbSNP rsID; a split
        # multi-allelic site has several rows at one position and they must not
        # all collapse onto the same rsID, which plink would reject.
        dup_pos = merged["POS"].duplicated(keep=False)
        take = hit & ~dup_pos
        merged.loc[take, "SNP"] = merged.loc[take, "RS"]

        n, n_hit = len(merged), int(take.sum())
        tot_n += n
        tot_hit += n_hit
        print("[chr{0}] {1:,}/{2:,} ({3:.1%}) labelled with rsIDs".format(
            chrom, n_hit, n, n_hit / float(n)), flush=True)

        if args.dry_run:
            continue
        if int(merged.loc[take, "SNP"].duplicated().sum()):
            print("[chr{0}] WARNING duplicate rsIDs after merge, skipping"
                  .format(chrom), flush=True)
            continue
        if not os.path.exists(bim + ".synthetic"):
            shutil.copy(bim, bim + ".synthetic")
        merged[cols].to_csv(bim, sep="\t", header=False, index=False)

    print("\ntotal {0:,}/{1:,} ({2:.1%}) labelled".format(
        tot_hit, tot_n, tot_hit / float(tot_n) if tot_n else 0))
    if args.dry_run:
        print("dry run -- nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
