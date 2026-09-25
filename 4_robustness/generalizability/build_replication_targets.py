"""Build the replication target table (step 1 of the non-WB replication).

Three things have to come together before any statistic can be computed:

  1. Which variants to test. The headline is 276 aggregate loci
     (cross_region/aggregate_loci.csv), each with a best_lead_snp, but that table does
     not record WHICH region produced the best p. That is recovered by matching the lead
     against each region's FUMA leadSNPs.txt.
  2. The effect alleles. FUMA's uniqID carries alleles but not which is the effect
     allele; the FastGWA-format minP files carry A1 as the effect allele, so alleles are
     taken from there.
  3. The discovery direction for design B. The per-region minP files also carry
     `most_sig_dim` and its signed BETA -- design B's entire input, at 261 MB gzipped per
     region instead of the 1.47 TB a per-dimension pass would cost.

Outputs
-------
    replication_targets.csv  one row per aggregate locus (the primary test set)
    all_region_leads.csv     every region-lead pair (the secondary, higher-power set)

Usage
-----
    python build_replication_targets.py
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import csv
import gzip
import os
import sys

BASE = ("<EXTERNAL: GWAS_DIR_pixpro>/"
        "fastgwa_pixpro/<EXTERNAL: encoder checkpoint>")
FUMA = os.path.join(BASE, "JAGWAS", "FUMA")
MINP = os.path.join(BASE, "discovery_<EXTERNAL: encoder checkpoint>_{0}__minP.csv.gz")
AGG = ("<EXTERNAL: jagwas_paper>/"
       "post_gwas_analysis/results/cross_region/aggregate_loci.csv")
GATE_DIR = "<EXTERNAL: reproducibility>"

# minP columns: CHR SNP POS A1 A2 N AF1 BETA SE P INFO most_sig_dim
M_CHR, M_SNP, M_POS, M_A1, M_A2, M_AF1, M_BETA, M_SE, M_P, M_DIM = (
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11)


def read_region_leads(region):
    """FUMA lead SNPs for one region."""
    path = os.path.join(FUMA, region, "leadSNPs.txt")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            out.append({"region": region, "rsid": p[col["rsID"]],
                        "chr": p[col["chr"]], "pos": p[col["pos"]],
                        "uniq_id": p[col["uniqID"]],
                        "p_jagwas": float(p[col["p"]]),
                        "genomic_locus": p[col["GenomicLocus"]]})
    return out


def attach_minp(region, leads):
    """One streaming pass over a region's minP file, for that region's leads only."""
    path = MINP.format(region)
    if not os.path.exists(path):
        print("  WARNING: no minP file for {0}".format(region))
        return 0
    want = {}
    for ld in leads:
        want.setdefault((ld["chr"], ld["pos"]), []).append(ld)
    hit = 0
    with gzip.open(path, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            key = (p[M_CHR], p[M_POS])
            if key in want:
                for ld in want[key]:
                    ld["a1"] = p[M_A1]
                    ld["a2"] = p[M_A2]
                    ld["af1"] = p[M_AF1]
                    ld["minp_dim"] = p[M_DIM]
                    ld["minp_beta"] = p[M_BETA]
                    ld["minp_se"] = p[M_SE]
                    ld["minp_p"] = p[M_P]
                    hit += 1
                del want[key]
                if not want:
                    break
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-targets",
                    default=os.path.join(GATE_DIR, "replication_targets.csv"))
    ap.add_argument("--out-leads",
                    default=os.path.join(GATE_DIR, "all_region_leads.csv"))
    args = ap.parse_args()

    regions = sorted(d for d in os.listdir(FUMA)
                     if os.path.isdir(os.path.join(FUMA, d)))
    print("regions with FUMA output: {0}".format(len(regions)))

    all_leads = []
    for region in regions:
        leads = read_region_leads(region)
        hit = attach_minp(region, leads)
        missing = [ld for ld in leads if "a1" not in ld]
        print("  {0:<28s} leads {1:3d}  minP matched {2:3d}{3}".format(
            region, len(leads), hit,
            "  MISSING {0}".format(len(missing)) if missing else ""))
        sys.stdout.flush()
        all_leads.extend(leads)

    print("\ntotal region-lead pairs: {0:,}".format(len(all_leads)))
    no_allele = [ld for ld in all_leads if "a1" not in ld]
    if no_allele:
        print("WARNING: {0} leads without alleles".format(len(no_allele)))

    # --- map the 276 aggregate loci onto their discovery region ---
    by_rsid = {}
    for ld in all_leads:
        by_rsid.setdefault(ld["rsid"], []).append(ld)

    targets, unmatched = [], []
    with open(AGG) as fh:
        for row in csv.DictReader(fh):
            lead = row["best_lead_snp"]
            best_p = float(row["best_p"])
            cands = by_rsid.get(lead, [])
            if not cands:
                unmatched.append((row["al_id"], lead))
                continue
            # the discovery region is the one whose p matches best_p; ties -> min p
            best = min(cands, key=lambda c: abs(c["p_jagwas"] - best_p))
            if abs(best["p_jagwas"] - best_p) > 1e-12 * max(1.0, best_p):
                best = min(cands, key=lambda c: c["p_jagwas"])
            t = dict(best)
            t["al_id"] = row["al_id"]
            t["n_regions"] = row["n_regions"]
            t["agg_regions"] = row["regions"]
            t["best_p"] = row["best_p"]
            targets.append(t)

    print("aggregate loci matched to a discovery region: {0}/276".format(len(targets)))
    if unmatched:
        print("UNMATCHED ({0}):".format(len(unmatched)))
        for al, s in unmatched[:10]:
            print("  {0} {1}".format(al, s))

    cols = ["al_id", "region", "rsid", "chr", "pos", "uniq_id", "a1", "a2", "af1",
            "p_jagwas", "best_p", "n_regions", "agg_regions",
            "minp_dim", "minp_beta", "minp_se", "minp_p"]
    with open(args.out_targets, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in targets:
            w.writerow(t)
    print("wrote {0}".format(args.out_targets))

    lcols = ["region", "rsid", "chr", "pos", "uniq_id", "a1", "a2", "af1",
             "p_jagwas", "genomic_locus", "minp_dim", "minp_beta", "minp_se", "minp_p"]
    with open(args.out_leads, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=lcols, extrasaction="ignore")
        w.writeheader()
        for ld in all_leads:
            w.writerow(ld)
    print("wrote {0}".format(args.out_leads))


if __name__ == "__main__":
    main()
