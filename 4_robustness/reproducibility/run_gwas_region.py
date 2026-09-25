"""Run one region of the seed GWAS cascade, deleting intermediates as it goes.

Per region: FastGWA over every dimension -> correlation matrix over the
surviving dimensions -> JAGWAS joint test -> thresholded min-P -> cleanup.

Storage is the reason this is region-at-a-time. The per-dimension FastGWA text
is ~720 MB x 128 = ~92 GB per region, ~1.5 TB per seed if kept. Nothing
downstream needs it once JAGWAS and min-P have consumed it, so it is deleted
before the next region starts. Retained per region: gzipped JAGWAS sumstats
(~135 MB), the correlation matrix, and a small min-P table.

Surviving-dimension contract, matching the original pipeline: dimensions that
fail to converge are dropped; the correlation script records which dimensions
entered the matrix, as the header of <region>_residuals_original.txt; exactly
those dimensions' FastGWA files are passed to JAGWAS, in matrix column order.
The counts must agree or JAGWAS's joint statistic is wrong, so it is asserted.
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import glob
import gzip
import os
import shutil
import subprocess
import sys
import time

GCTA = ("<EXTERNAL: gcta-1.94.1>")
BGEN = "<EXTERNAL: all_filtered.bgen>"
SAMPLE = "<EXTERNAL: MRI_samples_chr1.sample>"
GRM = "<EXTERNAL: ukb_grm>"
CCOVAR = "<EXTERNAL: T1_ccovar_>{cohort}_v2"
QCOVAR = "<EXTERNAL: T1_qcovar_>{cohort}_v2"

RSCRIPT = ("<EXTERNAL: bin>/"
           "Rscript")
JAGWAS = "<EXTERNAL: JAGWAS>"
SAMPLE_LIST = ("<EXTERNAL: PIXEL_EMBEDDING>/"
               "GWAS_DIR_pixpro/fastgwa_pixpro/<EXTERNAL: encoder checkpoint>"
               "<EXTERNAL: encoder checkpoint>/discovery_sample_list.txt")

TEMPLATE = "{cohort}_<EXTERNAL: encoder checkpoint>{region}_QT{dim}"


def run(cmd, log, label):
    """Run a command, appending stdout/stderr to log. Returns exit code."""
    with open(log, "a") as fh:
        fh.write("\n$ {0}\n".format(" ".join(cmd)))
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)


def fastgwa_all_dims(args, region, pheno_dir, out_dir, log):
    """One FastGWA run per dimension. Sequential: each writes ~720 MB."""
    os.makedirs(out_dir, exist_ok=True)
    done, failed = [], []
    t0 = time.time()
    for d in range(args.n_dims):
        name = TEMPLATE.format(cohort=args.cohort, region=region, dim=d)
        pheno = os.path.join(pheno_dir, name)
        if not os.path.exists(pheno):
            failed.append((d, "phenotype file missing"))
            continue
        out_prefix = os.path.join(out_dir, name + ".fastGWA")
        produced = out_prefix + ".fastGWA"
        if os.path.exists(produced) and not args.force:
            done.append(d)
            continue
        cmd = [GCTA, "--bgen", BGEN, "--sample", SAMPLE, "--grm-sparse", GRM,
               "--fastGWA-mlm", "--pheno", pheno,
               "--covar", CCOVAR.format(cohort=args.cohort),
               "--qcovar", QCOVAR.format(cohort=args.cohort),
               "--thread-num", str(args.threads), "--seed", "0",
               "--out", out_prefix]
        rc = run(cmd, log, "fastgwa d{0}".format(d))
        if rc == 0 and os.path.exists(produced):
            done.append(d)
        else:
            failed.append((d, "exit {0}".format(rc)))
        if (d + 1) % 16 == 0:
            el = (time.time() - t0) / 60.0
            print("    fastgwa {0}/{1}  {2:.1f} min elapsed".format(
                d + 1, args.n_dims, el))
            sys.stdout.flush()
    return done, failed


def surviving_dims(resid_file):
    """Dimensions that entered the correlation matrix, in column order."""
    with open(resid_file) as fh:
        header = fh.readline().split()
    return [h.strip().strip('"') for h in header]


def compact_minp(fastgwa_dir, region, args, out_csv):
    """Per-SNP minimum p across dimensions, thresholded to stay small."""
    import numpy as np
    best = {}
    files = sorted(glob.glob(os.path.join(
        fastgwa_dir,
        TEMPLATE.format(cohort=args.cohort, region=region, dim="*")
        + ".fastGWA.fastGWA")))
    for f in files:
        qt = f.split("_QT")[-1].split(".")[0]
        with open(f) as fh:
            fh.readline()
            for line in fh:
                p = line.rsplit("\t", 2)
                try:
                    pv = float(p[-2])
                except (ValueError, IndexError):
                    continue
                if pv >= args.minp_threshold:
                    continue
                snp = line.split("\t", 2)[1]
                cur = best.get(snp)
                if cur is None or pv < cur[0]:
                    best[snp] = (pv, qt, line.split("\t")[0],
                                 line.split("\t")[2])
    with open(out_csv, "w") as fh:
        fh.write("SNP,CHR,POS,min_p,best_dim\n")
        for snp, (pv, qt, chrom, pos) in sorted(best.items(),
                                                key=lambda kv: kv[1][0]):
            fh.write("{0},{1},{2},{3:.6g},QT{4}\n".format(snp, chrom, pos,
                                                          pv, qt))
    return len(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-dir", required=True,
                    help="per-seed root, e.g. reproducibility/seed_1")
    ap.add_argument("--region", required=True,
                    help="region token as used in the phenotype filenames")
    ap.add_argument("--cohort", default="discovery")
    ap.add_argument("--n-dims", type=int, default=128)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--minp-threshold", type=float, default=1e-5)
    ap.add_argument("--keep-fastgwa", action="store_true",
                    help="skip cleanup (debugging only; ~92 GB per region)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_dir = os.path.abspath(args.seed_dir)
    pheno_dir = os.path.join(seed_dir, "pheno")
    fastgwa_dir = os.path.join(seed_dir, "fastgwa", args.region)
    jagwas_dir = os.path.join(seed_dir, "jagwas")
    region_out = os.path.join(jagwas_dir, args.region)
    log = os.path.join(seed_dir, "logs",
                       "gwas_{0}.log".format(args.region))
    os.makedirs(os.path.dirname(log), exist_ok=True)
    os.makedirs(region_out, exist_ok=True)

    print("region      : {0}".format(args.region))
    print("seed dir    : {0}".format(seed_dir))
    print("phenotypes  : {0}".format(pheno_dir))
    print("fastgwa tmp : {0}".format(fastgwa_dir))
    print("retained    : {0}".format(region_out))
    if args.dry_run:
        name = TEMPLATE.format(cohort=args.cohort, region=args.region, dim=0)
        print("\nwould run {0} FastGWA jobs, first pheno: {1}".format(
            args.n_dims, os.path.join(pheno_dir, name)))
        print("would then: correlation -> JAGWAS -> min-P -> delete {0}"
              .format(fastgwa_dir))
        return 0

    # --- 1. FastGWA over dimensions ------------------------------------
    print("\n[1/4] FastGWA")
    t0 = time.time()
    done, failed = fastgwa_all_dims(args, args.region, pheno_dir,
                                    fastgwa_dir, log)
    print("    succeeded {0}/{1}  ({2:.1f} min)".format(
        len(done), args.n_dims, (time.time() - t0) / 60.0))
    for d, why in failed[:10]:
        print("    FAILED dim {0}: {1}".format(d, why))
    if not done:
        print("no dimensions succeeded; stopping")
        return 1

    # --- 2. correlation matrix -----------------------------------------
    print("\n[2/4] correlation matrix")
    t0 = time.time()
    rc = run([RSCRIPT, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "jagwas_correlation.R"),
              args.region, args.cohort, pheno_dir, fastgwa_dir, jagwas_dir,
              SAMPLE_LIST], log, "correlation")
    cor_file = os.path.join(region_out, args.region + "_residuals_cor.txt")
    resid_file = os.path.join(region_out,
                              args.region + "_residuals_original.txt")
    if rc != 0 or not os.path.exists(cor_file):
        print("    correlation step failed (rc={0}); see {1}".format(rc, log))
        return 1
    dims = surviving_dims(resid_file)
    with open(cor_file) as fh:
        matrix_dim = len(fh.readline().split())
    print("    surviving dims {0}, matrix {1}x{1}  ({2:.1f} min)".format(
        len(dims), matrix_dim, (time.time() - t0) / 60.0))
    if len(dims) != matrix_dim:
        print("    ABORT: residual header ({0}) != matrix dim ({1})".format(
            len(dims), matrix_dim))
        return 1

    # --- 3. JAGWAS ------------------------------------------------------
    print("\n[3/4] JAGWAS")
    files = []
    for qt in dims:
        f = os.path.join(fastgwa_dir, TEMPLATE.format(
            cohort=args.cohort, region=args.region,
            dim=qt.replace("QT", "")) + ".fastGWA.fastGWA")
        if not os.path.exists(f):
            print("    ABORT: missing FastGWA file for {0}".format(qt))
            return 1
        files.append(f)
    out_file = os.path.join(region_out, args.region + "_JAGWAS_results.txt")
    t0 = time.time()
    rc = run([JAGWAS, "--outputFilePath", out_file, "--cor_matrix", cor_file,
              "--nrow", "10000", "--MAF", "0.01", "--score_test", "0",
              "--beta_se", "1", "--logP", "0", "--delim", "\t",
              "--fileNames"] + files, log, "jagwas")
    if rc != 0 or not os.path.exists(out_file):
        print("    JAGWAS failed (rc={0})".format(rc))
        return 1
    with open(out_file, "rb") as fin, gzip.open(out_file + ".gz", "wb") as fo:
        shutil.copyfileobj(fin, fo)
    os.remove(out_file)
    print("    wrote {0}.gz  ({1:.1f} min)".format(
        os.path.basename(out_file), (time.time() - t0) / 60.0))

    # --- 4. min-P baseline + cleanup ------------------------------------
    print("\n[4/4] min-P and cleanup")
    minp_csv = os.path.join(region_out, args.region + "_minP.csv")
    n = compact_minp(fastgwa_dir, args.region, args, minp_csv)
    print("    min-P: {0:,} SNPs below {1:g}".format(n, args.minp_threshold))

    if args.keep_fastgwa:
        print("    keeping FastGWA output (--keep-fastgwa)")
    else:
        shutil.rmtree(fastgwa_dir, ignore_errors=True)
        for f in glob.glob(os.path.join(pheno_dir, TEMPLATE.format(
                cohort=args.cohort, region=args.region, dim="*"))):
            os.remove(f)
        if os.path.exists(resid_file):
            os.remove(resid_file)
        print("    deleted per-dim FastGWA, phenotypes and residuals")

    print("\nregion {0} complete".format(args.region))
    return 0


if __name__ == "__main__":
    sys.exit(main())
