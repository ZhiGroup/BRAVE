"""Extract dosages at the replication target variants for the non-WB cohort (step 2).

Reads the 276 needed rows out of aim_1's dosage memmap. The memmap is (8,931,083
variants x 6,474 samples) float16 and is variant-major, so one variant is 12.9 kB of
CONTIGUOUS bytes: the whole job is ~3.6 MB of reads. This is the access pattern the
handoff's gotcha 1 prescribes -- read only the loci rows, never stream the 115 GB.

(An earlier attempt read the BGEN through bgen_reader instead, to avoid depending on a
memmap that does not survive the node-122 migration. That was the wrong trade: building
bgen_reader's metadata cost a 468 GB sparse file, 24 GB of real writes and ~64 minutes,
to serve 276 variants. The right answer is to read the memmap now and keep the small
extracted dosages, which survive on their own.)

Dosage convention, from build_geno_memmap_t1only.py:

    buf[n] = (v.probabilities[mask] * [0, 1, 2]).sum(1)

so the stored value is the dosage of allele2. aim_1 reports allele2 == discovery A1 with
zero flips; that is re-verified per variant here, not inherited, and any flip is applied.

Validation: the MAF implied by the extracted dosages is checked against aim_1's
precomputed nonwb_maf_all.npy at the same variant indices.

Outputs
-------
    dosages.npy    (n_targets, 6470) float32, dosage of the discovery A1 allele
    variants.csv   per target: allele check, MAF, and the MAF cross-check

Usage
-----
    python extract_dosages_nonwb.py
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
import pandas as pd

GATE_DIR = "<EXTERNAL: reproducibility>"
NONWB = "<EXTERNAL: nonwb_replication>"
GENO = "<EXTERNAL: geno_e_t1only>"
VARS = os.path.join(NONWB, "geno_variant_order_t1only.parquet")
MAF_ALL = os.path.join(NONWB, "gwas_t1only", "nonwb_maf_all.npy")
N_SAMPLES = 6474


def read_ids(path):
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip().isdigit()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=os.path.join(GATE_DIR,
                                                      "replication_targets.csv"))
    ap.add_argument("--geno", default=GENO)
    ap.add_argument("--variant-order", default=VARS)
    ap.add_argument("--sample-order",
                    default=os.path.join(NONWB, "sample_order_t1only.txt"))
    ap.add_argument("--sample-index",
                    default=os.path.join(GATE_DIR, "bre_nonwb", "sample_index.npy"))
    ap.add_argument("--out-dir", default=os.path.join(GATE_DIR, "dosages_nonwb"))
    args = ap.parse_args()

    targets = []
    with open(args.targets) as fh:
        for row in csv.DictReader(fh):
            targets.append(row)
    print("targets           : {0:,}".format(len(targets)))

    order = read_ids(args.sample_order)
    sample_index = np.load(args.sample_index)
    if len(order) != N_SAMPLES:
        sys.exit("sample_order has {0} rows, expected {1}".format(
            len(order), N_SAMPLES))
    print("memmap columns    : {0:,} (sample_order_t1only.txt)".format(len(order)))
    print("analysis cohort   : {0:,}".format(len(sample_index)))

    t0 = time.time()
    vo = pd.read_parquet(args.variant_order)
    print("variant order     : {0:,} rows, loaded in {1:.1f}s".format(
        len(vo), time.time() - t0))
    n_var = len(vo)

    chrom = vo["chrom"].astype(str).str.lstrip("0").values
    pos = vo["pos"].astype(np.int64).values
    a1_col = vo["allele1"].astype(str).str.upper().values
    a2_col = vo["allele2"].astype(str).str.upper().values

    lookup = {}
    for i in range(n_var):
        lookup.setdefault((chrom[i], pos[i]), []).append(i)

    rows = []
    for t in targets:
        key = (str(t["chr"]).lstrip("0"), int(t["pos"]))
        pick, flip, note = None, None, ""
        for i in lookup.get(key, []):
            ta1, ta2 = t["a1"].upper(), t["a2"].upper()
            if a2_col[i] == ta1 and a1_col[i] == ta2:
                pick, flip = i, False        # stored dosage is already A1 dosage
                break
            if a1_col[i] == ta1 and a2_col[i] == ta2:
                pick, flip = i, True
                break
        if pick is None:
            note = "no_allele_match" if key in lookup else "not_in_panel"
        rows.append({"al_id": t.get("al_id", ""), "region": t["region"],
                     "rsid": t["rsid"], "chr": t["chr"], "pos": t["pos"],
                     "a1": t["a1"], "a2": t["a2"], "row": pick, "flipped": flip,
                     "note": note, "disc_af1": t.get("af1", "")})

    found = [r for r in rows if r["row"] is not None]
    print("located in panel  : {0:,}/{1:,}".format(len(found), len(rows)))
    print("allele flips      : {0}".format(sum(1 for r in found if r["flipped"])))
    for r in rows:
        if r["note"]:
            print("  {0} {1} chr{2}:{3}  {4}".format(
                r["region"], r["rsid"], r["chr"], r["pos"], r["note"]))
    if not found:
        sys.exit("no target variants located")

    # ---- read only the needed rows, in ascending order ----
    geno = np.memmap(args.geno, dtype="float16", mode="r",
                     shape=(n_var, N_SAMPLES))
    out = np.full((len(rows), len(sample_index)), np.nan, dtype=np.float32)
    t1 = time.time()
    for r in sorted(found, key=lambda r: r["row"]):
        d = np.asarray(geno[r["row"], :], dtype=np.float32)[sample_index]
        if r["flipped"]:
            d = 2.0 - d
        out[rows.index(r), :] = d
        r["af_a1"] = float(np.nanmean(d) / 2.0)
        r["maf"] = float(min(r["af_a1"], 1.0 - r["af_a1"]))
    del geno
    print("read {0} rows in {1:.1f}s ({2:.1f} MB)".format(
        len(found), time.time() - t1, len(found) * N_SAMPLES * 2 / 1e6))

    # ---- validation: MAF against aim_1's precomputed array ----
    if os.path.exists(MAF_ALL):
        maf_ref = np.load(MAF_ALL)
        if len(maf_ref) == n_var:
            diffs = []
            for r in found:
                r["maf_ref"] = float(maf_ref[r["row"]])
                diffs.append(abs(r["maf"] - r["maf_ref"]))
            diffs = np.array(diffs)
            print("\nMAF cross-check vs nonwb_maf_all.npy:")
            print("  max abs diff {0:.6f}  mean {1:.6f}".format(
                diffs.max(), diffs.mean()))
            if diffs.max() > 0.01:
                worst = max(found, key=lambda r: abs(r["maf"] - r["maf_ref"]))
                print("  WORST: {0} ours {1:.5f} ref {2:.5f}".format(
                    worst["rsid"], worst["maf"], worst["maf_ref"]))
        else:
            print("\nMAF array length {0} != {1}, skipping cross-check".format(
                len(maf_ref), n_var))

    mafs = np.array([r["maf"] for r in found])
    print("\nMAF in the non-WB cohort: min {0:.5f} median {1:.4f}".format(
        mafs.min(), np.median(mafs)))
    for thr in (0.01, 0.005, 0.001):
        print("  MAF < {0}: {1}".format(thr, int((mafs < thr).sum())))

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    np.save(os.path.join(args.out_dir, "dosages.npy"), out)
    cols = ["al_id", "region", "rsid", "chr", "pos", "a1", "a2", "row", "flipped",
            "note", "disc_af1", "af_a1", "maf", "maf_ref"]
    with open(os.path.join(args.out_dir, "variants.csv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote {0}  dosages {1}".format(args.out_dir, out.shape))


if __name__ == "__main__":
    main()
