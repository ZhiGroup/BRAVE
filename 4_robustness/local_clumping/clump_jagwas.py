"""Call genomic risk loci from a set of JAGWAS results with the FUMA-equivalent pipeline.

Generalises validate_vs_fuma.py, which is pinned to the published run's paths.
Point this at any directory laid out as <jagwas-dir>/<region>/<region>_JAGWAS_results.txt[.gz]
-- the replication seeds included -- and it applies exactly the configuration
validated against FUMA v1.5.2 in VALIDATION.md (99.8% lead-variant concordance
over the published 16 regions).

Regions run one at a time. The reference panel and summary statistics sit on a
shared NFS mount, and parallel plink jobs over it are not worth the risk.

Usage:
  python clump_jagwas.py --jagwas-dir .../seed_1/jagwas --out-root .../seed_1/loci
  python clump_jagwas.py --jagwas-dir ... --out-root ... --regions Left_Amygdala
"""
from __future__ import print_function

import argparse
import glob
import os
import subprocess
import sys
import traceback

import pandas as pd

import validate_vs_fuma as V


def find_sumstats(jagwas_dir, region):
    stem = os.path.join(jagwas_dir, region,
                        "{0}_JAGWAS_results.txt".format(region))
    for cand in (stem + ".gz", stem):
        if os.path.exists(cand):
            return cand
    hits = sorted(glob.glob(os.path.join(jagwas_dir, region, "*JAGWAS_results*")))
    return hits[0] if hits else None


def region_list(jagwas_dir):
    return sorted(d for d in os.listdir(jagwas_dir)
                  if os.path.isdir(os.path.join(jagwas_dir, d)))


def clump(gwas, out_dir):
    """Same flags as validate_vs_fuma.run_clump; see VALIDATION.md for why."""
    cmd = [
        sys.executable, V.CLUMP_SCRIPT,
        "--gwas", gwas,
        "--bfile-pattern", V.EUR_PANEL,
        "--plink", V.PLINK19,
        "--out", out_dir,
        "--lead-p", "3.125e-9",
        "--gwas-p", "0.05",
        "--r2", "0.6",
        "--r2-2", "0.1",
        "--merge-dist", "250000",
        "--maf", "0",
        "--no-mhc",
        "--clump-kb", "1000",
    ]
    if V.EUR_REF_MAP and os.path.exists(V.EUR_REF_MAP):
        cmd += ["--ref-map", V.EUR_REF_MAP]
    print("[RUN] " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jagwas-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run regions whose output already exists")
    args = ap.parse_args()

    regions = args.regions or region_list(args.jagwas_dir)
    print("[CLUMP] {0} regions from {1}".format(len(regions), args.jagwas_dir),
          flush=True)

    rows = []
    for i, region in enumerate(regions, 1):
        print("\n{0}\n[{1}/{2}] {3}\n{0}".format(
            "=" * 70, i, len(regions), region), flush=True)
        gwas = find_sumstats(args.jagwas_dir, region)
        if not gwas:
            print("[SKIP] no JAGWAS results", flush=True)
            continue

        out_dir = os.path.join(args.out_root, region)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        loci_file = os.path.join(out_dir, "GenomicRiskLoci.txt")

        try:
            if os.path.exists(loci_file) and not args.force:
                print("[SKIP] output already present", flush=True)
            else:
                rc = clump(gwas, out_dir)
                if rc != 0:
                    print("[FAIL] fuma_clump exited {0}".format(rc), flush=True)
                    continue
            if not os.path.exists(loci_file):
                print("[FAIL] no loci file produced", flush=True)
                continue
            loci = pd.read_csv(loci_file, sep="\t")
        except Exception:
            traceback.print_exc()
            continue

        rows.append({"region": region, "n_loci": len(loci),
                     "n_lead_snps": int(loci["nLeadSNPs"].sum()),
                     "n_ind_sig_snps": int(loci["nIndSigSNPs"].sum())})
        print("  {0} loci".format(len(loci)), flush=True)

    if not rows:
        sys.exit("[ERROR] no regions produced loci")

    df = pd.DataFrame(rows)
    summary = os.path.join(args.out_root, "loci_summary.csv")
    df.to_csv(summary, index=False)

    print("\n" + "=" * 62)
    print("{0:32s} {1:>7s} {2:>8s} {3:>9s}".format(
        "region", "loci", "leadSNP", "indSigSNP"))
    print("=" * 62)
    for _, r in df.iterrows():
        print("{0:32s} {1:7d} {2:8d} {3:9d}".format(
            r["region"], r["n_loci"], r["n_lead_snps"], r["n_ind_sig_snps"]))
    print("-" * 62)
    print("{0:32s} {1:7d} {2:8d} {3:9d}".format(
        "TOTAL", int(df["n_loci"].sum()), int(df["n_lead_snps"].sum()),
        int(df["n_ind_sig_snps"].sum())))
    print("\nwrote {0}".format(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
