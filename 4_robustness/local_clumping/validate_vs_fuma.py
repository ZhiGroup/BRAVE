"""Reproduce one region's published FUMA genomic risk loci with fuma_clump.py.

The published loci were called by FUMA v1.5.2 against 1000 Genomes Phase 3 EUR.
This runs the local clumping tool on the same JAGWAS summary statistics with the
same reference panel and the same parameters, then compares the two locus lists.

Parameters are taken verbatim from the per-region params.config:
  leadP 3.125e-9, gwasP 0.05, r2 0.6, r2_2 0.1, mergeDist 250 kb, MAF 0.

MHC is excluded (exMHC=1). The published chr6 locus ends at 29,612,325, which
is 2.4 kb below the MHC start used here (29,614,758) -- it was truncated at the
boundary, confirming MHC SNPs were withheld from locus calling.

Usage:
  python validate_vs_fuma.py --region Right_Thalamus-Proper
  python validate_vs_fuma.py --region Right_Thalamus-Proper --compare-only
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys

import pandas as pd

BASE = "<EXTERNAL: JAGWAS output root containing FUMA/ and output_discovery/>"
FUMA_ROOT = os.path.join(BASE, "FUMA")
SUMSTATS_ROOT = os.path.join(BASE, "output_discovery")
# Rebuilt from the 1000G Phase 3 VCFs per FUMA's recipe (build_fuma_panel.py):
# 503 EUR samples, 24,152,754 variants across 22 per-chromosome bfiles. The
# earlier panel was the MAGMA g1000_eur build -- same samples, but only
# 8,550,156 variants, which was the sole remaining source of disagreement.
EUR_PANEL = "<path/to>/g1000_phase3/EUR.chr{}"
PLINK19 = "<path/to>/plink"
HERE = os.path.dirname(os.path.abspath(__file__))
CLUMP_SCRIPT = os.path.join(HERE, "fuma_clump.py")

# No ref-map harmonization. It existed to reconcile rsIDs between the GWAS and
# the old MAGMA panel, which named the same variants differently. The rebuilt
# panel is labelled directly from the GWAS by CHR:POS (assign_panel_rsids.py),
# so the two already share identifiers and a second remapping could only
# introduce disagreement.
EUR_REF_MAP = None


def find_sumstats(region):
    stem = os.path.join(SUMSTATS_ROOT, region,
                        "{0}_JAGWAS_results.txt".format(region))
    for cand in (stem + ".gz", stem):
        if os.path.exists(cand):
            return cand
    sys.exit("[ERROR] no JAGWAS sumstats for {0} under {1}".format(
        region, SUMSTATS_ROOT))


def run_clump(region, out_dir, chroms):
    cmd = [
        sys.executable, CLUMP_SCRIPT,
        "--gwas", find_sumstats(region),
        "--bfile-pattern", EUR_PANEL,
        "--plink", PLINK19,
        "--out", out_dir,
        "--lead-p", "3.125e-9",
        "--gwas-p", "0.05",
        "--r2", "0.6",
        "--r2-2", "0.1",
        "--merge-dist", "250000",
        "--maf", "0",
        "--no-mhc",
        # FUMA's LD tables are built with plink's default 1000 kb window
        # (--r2 --ld-window 99999 --ld-window-r2 0.05, no --ld-window-kb), so
        # it never links SNPs more than 1 Mb apart. This is now also
        # fuma_clump.py's default; passed explicitly so the run is
        # self-documenting rather than dependent on a default that once differed.
        "--clump-kb", "1000",
    ]
    if EUR_REF_MAP and os.path.exists(EUR_REF_MAP):
        cmd += ["--ref-map", EUR_REF_MAP]
    if chroms:
        cmd += ["--chroms"] + [str(c) for c in chroms]
    print("[RUN] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit("[ERROR] fuma_clump.py exited {0}".format(rc))


def load_loci(path, label):
    if not os.path.exists(path):
        sys.exit("[ERROR] {0} loci file not found: {1}".format(label, path))
    df = pd.read_csv(path, sep="\t", dtype={"chr": str, "rsID": str})
    df.columns = [c.strip() for c in df.columns]
    return df


def compare(fuma, local):
    f_leads = list(fuma["rsID"])
    l_leads = list(local["rsID"])
    f_set, l_set = set(f_leads), set(l_leads)

    # Position is the identity that matters. rsIDs can disagree between dbSNP
    # builds for the same variant (17:44310965 is rs12452789 to FUMA's panel and
    # rs147895577 to ours), so an rsID-only comparison understates agreement.
    f_pos = set(zip(fuma["chr"].astype(str), fuma["pos"].astype(int)))
    l_pos = set(zip(local["chr"].astype(str), local["pos"].astype(int)))

    print("\n" + "=" * 78)
    print("LEAD SNP COMPARISON")
    print("=" * 78)
    print("  FUMA loci          : {0}".format(len(f_leads)))
    print("  local loci         : {0}".format(len(l_leads)))
    print("  shared lead SNPs   : {0}".format(len(f_set & l_set)))
    print("  FUMA-only          : {0}".format(len(f_set - l_set)))
    print("  local-only         : {0}".format(len(l_set - f_set)))
    if f_set:
        print("  recall (FUMA leads recovered) : {0:.1%}".format(
            len(f_set & l_set) / float(len(f_set))))
    print("  --- by chr:pos (rsID-label independent) ---")
    print("  shared lead positions : {0}".format(len(f_pos & l_pos)))
    if f_pos:
        print("  positional recall     : {0:.1%}".format(
            len(f_pos & l_pos) / float(len(f_pos))))

    fi = fuma.set_index("rsID")
    li = local.set_index("rsID")

    print("\n" + "-" * 78)
    print("{0:14s} {1:>4s} {2:>11s} {3:>11s} {4:>11s} {5:>11s} {6:>9s}".format(
        "lead SNP", "chr", "FUMA start", "local start", "FUMA end",
        "local end", "bp delta"))
    print("-" * 78)
    for rs in f_leads:
        if rs not in li.index:
            print("{0:14s} {1:>4s} {2:>11} {3:>11s} {4:>11} {5:>11s} {6:>9s}"
                  .format(rs, str(fi.loc[rs, "chr"]), fi.loc[rs, "start"],
                          "--", fi.loc[rs, "end"], "--", "MISSING"))
            continue
        fs, fe = int(fi.loc[rs, "start"]), int(fi.loc[rs, "end"])
        ls, le = int(li.loc[rs, "start"]), int(li.loc[rs, "end"])
        delta = abs(fs - ls) + abs(fe - le)
        print("{0:14s} {1:>4s} {2:>11d} {3:>11d} {4:>11d} {5:>11d} {6:>9d}"
              .format(rs, str(fi.loc[rs, "chr"]), fs, ls, fe, le, delta))

    extra = [rs for rs in l_leads if rs not in f_set]
    if extra:
        print("\nlocal-only leads (not called by FUMA):")
        for rs in extra:
            print("  {0}  chr{1}:{2}  p={3}".format(
                rs, li.loc[rs, "chr"], li.loc[rs, "pos"], li.loc[rs, "p"]))

    missing = [rs for rs in f_leads if rs not in l_set]
    if missing:
        print("\nFUMA leads not recovered:")
        for rs in missing:
            print("  {0}  chr{1}:{2}  p={3}".format(
                rs, fi.loc[rs, "chr"], fi.loc[rs, "pos"], fi.loc[rs, "p"]))

    exact = (len(f_pos ^ l_pos) == 0)
    print("\nVERDICT: lead variants (by chr:pos) are {0}{1}".format(
        "IDENTICAL" if exact else "NOT identical",
        "" if len(f_set ^ l_set) == 0
        else "  [{0} differ by rsID label only]".format(
            len(f_pos & l_pos) - len(f_set & l_set))))
    return exact


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True)
    ap.add_argument("--out-root", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "validation"))
    ap.add_argument("--chroms", nargs="+", type=int, default=None,
                    help="restrict to these chromosomes (default: all 1-22)")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip the clumping run, just compare existing output")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.region)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if not args.compare_only:
        run_clump(args.region, out_dir, args.chroms)

    fuma = load_loci(os.path.join(FUMA_ROOT, args.region,
                                  "GenomicRiskLoci.txt"), "FUMA")
    local = load_loci(os.path.join(out_dir, "GenomicRiskLoci.txt"), "local")
    return 0 if compare(fuma, local) else 1


if __name__ == "__main__":
    sys.exit(main())
