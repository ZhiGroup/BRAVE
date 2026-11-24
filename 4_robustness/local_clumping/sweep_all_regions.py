"""Reproduce the published FUMA genomic risk loci for all 16 BRE regions.

Runs validate_vs_fuma's clumping step region by region -- sequentially, never in
parallel, because the reference panel and summary statistics live on a shared
NFS mount -- and writes one summary row per region.

Agreement is scored on chr:pos rather than rsID. The two reference builds label
some variants differently (17:44310965 is rs12452789 to FUMA's panel and
rs147895577 to ours), which an rsID-only comparison would count as a mismatch.

Locus boundaries are expected to be equal or narrower than FUMA's, never wider:
FUMA's refSNPs=1 extends an LD block using reference-panel variants absent from
the GWAS, which this tool does not do. A wider local boundary would indicate a
real discrepancy and is counted separately.

Usage:
  python sweep_all_regions.py
  python sweep_all_regions.py --regions Left_Amygdala Left_Caudate
  python sweep_all_regions.py --compare-only
"""
from __future__ import print_function

import argparse
import os
import sys
import traceback

import pandas as pd

import validate_vs_fuma as V


def region_list():
    return sorted(d for d in os.listdir(V.FUMA_ROOT)
                  if os.path.isdir(os.path.join(V.FUMA_ROOT, d)))


def _poskey(df):
    """chr:pos as a plain string -- a tuple key would be read as a MultiIndex."""
    return df["chr"].astype(str) + ":" + df["pos"].astype(int).astype(str)


def summarize(region, fuma, local):
    fuma = fuma.assign(_k=_poskey(fuma))
    local = local.assign(_k=_poskey(local))
    f_pos, l_pos = set(fuma["_k"]), set(local["_k"])
    f_rs, l_rs = set(fuma["rsID"]), set(local["rsID"])

    fi = fuma.drop_duplicates("_k").set_index("_k")
    li = local.drop_duplicates("_k").set_index("_k")

    exact = wider = narrower = 0
    for key in f_pos & l_pos:
        fs, fe = int(fi.loc[key, "start"]), int(fi.loc[key, "end"])
        ls, le = int(li.loc[key, "start"]), int(li.loc[key, "end"])
        if (fs, fe) == (ls, le):
            exact += 1
        elif ls < fs or le > fe:
            wider += 1
        else:
            narrower += 1

    return {
        "region": region,
        "fuma_loci": len(f_pos),
        "local_loci": len(l_pos),
        "shared_by_pos": len(f_pos & l_pos),
        "positional_recall": (len(f_pos & l_pos) / float(len(f_pos))
                              if f_pos else float("nan")),
        "shared_by_rsid": len(f_rs & l_rs),
        "rsid_label_only_diff": len(f_pos & l_pos) - len(f_rs & l_rs),
        "fuma_only": len(f_pos - l_pos),
        "local_only": len(l_pos - f_pos),
        "bounds_exact": exact,
        "bounds_narrower": narrower,
        "bounds_wider": wider,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--out-root", default=os.path.join(V.HERE, "validation"))
    ap.add_argument("--summary", default=os.path.join(
        V.HERE, "validation", "fuma_reproduction_summary.csv"))
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    regions = args.regions or region_list()
    print("[SWEEP] {0} regions".format(len(regions)), flush=True)

    rows = []
    for i, region in enumerate(regions, 1):
        print("\n{0}\n[{1}/{2}] {3}\n{0}".format(
            "=" * 70, i, len(regions), region), flush=True)
        out_dir = os.path.join(args.out_root, region)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        loci_file = os.path.join(out_dir, "GenomicRiskLoci.txt")

        try:
            if not (args.compare_only or os.path.exists(loci_file)):
                V.run_clump(region, out_dir, None)
            elif os.path.exists(loci_file):
                print("[SKIP] output already present", flush=True)

            fuma = V.load_loci(os.path.join(V.FUMA_ROOT, region,
                                            "GenomicRiskLoci.txt"), "FUMA")
            local = V.load_loci(loci_file, "local")
            row = summarize(region, fuma, local)
        except SystemExit as exc:
            print("[FAIL] {0}: {1}".format(region, exc), flush=True)
            continue
        except Exception:
            traceback.print_exc()
            continue

        rows.append(row)
        print("  loci {0} vs {1} | positional recall {2:.1%} | bounds "
              "exact {3} narrower {4} wider {5}".format(
                  row["fuma_loci"], row["local_loci"],
                  row["positional_recall"], row["bounds_exact"],
                  row["bounds_narrower"], row["bounds_wider"]), flush=True)

    if not rows:
        sys.exit("[ERROR] no regions completed")

    df = pd.DataFrame(rows)
    df.to_csv(args.summary, index=False)

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print("{0:30s} {1:>5s} {2:>6s} {3:>7s} {4:>8s} {5:>7s} {6:>6s} {7:>6s}"
          .format("region", "FUMA", "local", "shared", "recall", "exact",
                  "narrow", "wider"))
    for _, r in df.iterrows():
        print("{0:30s} {1:5d} {2:6d} {3:7d} {4:7.1%} {5:7d} {6:6d} {7:6d}"
              .format(r["region"], r["fuma_loci"], r["local_loci"],
                      r["shared_by_pos"], r["positional_recall"],
                      r["bounds_exact"], r["bounds_narrower"],
                      r["bounds_wider"]))
    print("-" * 96)
    print("{0:30s} {1:5d} {2:6d} {3:7d} {4:7.1%} {5:7d} {6:6d} {7:6d}".format(
        "TOTAL", int(df["fuma_loci"].sum()), int(df["local_loci"].sum()),
        int(df["shared_by_pos"].sum()),
        df["shared_by_pos"].sum() / float(df["fuma_loci"].sum()),
        int(df["bounds_exact"].sum()), int(df["bounds_narrower"].sum()),
        int(df["bounds_wider"].sum())))
    print("\nrsID-label-only differences: {0}".format(
        int(df["rsid_label_only_diff"].sum())))
    print("wrote {0}".format(args.summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
