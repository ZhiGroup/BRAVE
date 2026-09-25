"""Drive every region of one seed's GWAS cascade, with bounded concurrency.

Sequentially, a region costs ~5.6 h (5.0 h of FastGWA), so 16 regions is ~3.75
days per seed. The machine has 96 cores and the original invocation asks for 256
threads, so a single job oversubscribes rather than going faster; and each
FastGWA writes only ~5 MB/s, so disk is not the limit either. Running a few
regions side by side with a sane thread count therefore converts idle cores into
wall-clock.

Disk is the one thing to respect: each in-flight region holds ~92 GB of
per-dimension FastGWA output until its JAGWAS step finishes, so peak usage is
`max_parallel * 92 GB`. Phenotypes are generated per region just before it runs
and deleted with the rest.
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# the 16 GWAS regions, as they appear in the phenotype filenames
REGIONS = [
    "Left_Thalamus_Proper", "Left_Caudate", "Left_Putamen", "Left_Pallidum",
    "Brain_Stem_or_4th_Ventricle", "Left_Hippocampus", "Left_Amygdala",
    "Left_Accumbens-area", "Right_Thalamus-Proper", "Right_Caudate",
    "Right_Putamen", "Right_Pallidum", "Right_Hippocampus", "Right_Amygdala",
    "Right_Accumbens-area", "CSF",
]


def region_done(seed_dir, region):
    return os.path.exists(os.path.join(
        seed_dir, "jagwas", region, region + "_JAGWAS_results.txt.gz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-dir", required=True)
    ap.add_argument("--embeddings", default=None,
                    help="defaults to <seed-dir>/embeddings")
    ap.add_argument("--cohort", default="discovery")
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--threads", type=int, default=30,
                    help="FastGWA threads per region; max-parallel*threads "
                         "should stay at or below the core count")
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--skip-done", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_dir = os.path.abspath(args.seed_dir)
    emb = os.path.abspath(args.embeddings or os.path.join(seed_dir,
                                                          "embeddings"))
    pheno_dir = os.path.join(seed_dir, "pheno")
    log_dir = os.path.join(seed_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    todo = args.regions or REGIONS
    if args.skip_done:
        skipped = [r for r in todo if region_done(seed_dir, r)]
        todo = [r for r in todo if not region_done(seed_dir, r)]
        for r in skipped:
            print("already done, skipping: {0}".format(r))

    print("seed dir     : {0}".format(seed_dir))
    print("regions      : {0}".format(len(todo)))
    print("max parallel : {0}  ({1} threads each = {2} of {3} cores)".format(
        args.max_parallel, args.threads, args.max_parallel * args.threads,
        os.sysconf("SC_NPROCESSORS_ONLN")))
    print("peak disk    : ~{0} GB".format(args.max_parallel * 92))
    if args.dry_run:
        for r in todo:
            print("  would run {0}".format(r))
        return 0

    running = []          # (region, Popen, start)
    queue = list(todo)
    t_start = time.time()

    while queue or running:
        while queue and len(running) < args.max_parallel:
            region = queue.pop(0)
            # phenotypes for this region only, written just before it runs
            rc = subprocess.call(
                [sys.executable, "-u", os.path.join(HERE, "make_phenotypes.py"),
                 "--embeddings", emb, "--out", pheno_dir,
                 "--cohort", args.cohort, "--regions", region],
                stdout=open(os.path.join(log_dir,
                                         "pheno_{0}.log".format(region)), "w"),
                stderr=subprocess.STDOUT)
            if rc != 0:
                print("[{0}] phenotype generation FAILED (rc={1})".format(
                    region, rc))
                continue
            log = open(os.path.join(log_dir,
                                    "region_{0}.log".format(region)), "w")
            p = subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "run_gwas_region.py"),
                 "--seed-dir", seed_dir, "--region", region,
                 "--cohort", args.cohort, "--threads", str(args.threads)],
                stdout=log, stderr=subprocess.STDOUT)
            running.append((region, p, time.time()))
            print("[{0}] started ({1} running, {2} queued)".format(
                region, len(running), len(queue)))
            sys.stdout.flush()

        time.sleep(60)

        still = []
        for region, p, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((region, p, t0))
                continue
            mins = (time.time() - t0) / 60.0
            ok = rc == 0 and region_done(seed_dir, region)
            print("[{0}] {1} after {2:.1f} min".format(
                region, "COMPLETE" if ok else "FAILED rc={0}".format(rc), mins))
            sys.stdout.flush()
        running = still

    done = [r for r in todo if region_done(seed_dir, r)]
    print("\n{0}/{1} regions complete in {2:.1f} h".format(
        len(done), len(todo), (time.time() - t_start) / 3600.0))
    missing = [r for r in todo if r not in done]
    if missing:
        print("incomplete: {0}".format(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
