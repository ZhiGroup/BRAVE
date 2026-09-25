#!/usr/bin/env python3
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
"""
37_sbayesrc_pgs_sweep.py
Production SBayesRC PGS sweep for the JAGWAS BRE phenotypes. Extends the
Script 36 pilot to all 16 regions and configurable top-h²-ranked dims per
region.

Stages:
  Stage C (default, --dim-ranks "1"): top-h² dim per region (16 runs)
  Stage A (full,    --dim-ranks "1 2 3 4 5"): top-5 h²-ranked dims per
           region (80 runs total)

Pipeline (per (region, dim)):
  1. FastGWA per-dim sumstats → COJO/.ma format
  2. GCTB SBayesRC impute step (HM3 LD eigendecomp)
  3. GCTB SBayesRC main analysis (baseline-LF v2.2 annotations)
  4. PLINK2 --score on a cached replication-cohort PGEN
     (BGEN → PGEN cache built ONCE on first run, ~12 min, reused thereafter)
  5. R² of per-subject PGS vs the true replication BRE dim value

Outputs:
  results/sbayesrc/sweep/<region>_QT<dim>.*  — per-run GCTB + PLINK2 outputs
  results/sbayesrc/sweep/sweep_R2.csv        — region × dim × R² master table
  results/sbayesrc/sweep/_run.log            — driver log

Compute env: <EXTERNAL: python> (Python 3.7)
"""
import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

GCTB = "<EXTERNAL: gctb>"
LD_DIR = "<EXTERNAL: ukbEUR_HM3>"
ANNOT = "<EXTERNAL: annot_baseline2.2.txt>"
PLINK2 = "<EXTERNAL: plink2>"

FASTGWA_DIR = (
    "<EXTERNAL: PIXEL_EMBEDDING>/"
    "GWAS_DIR_pixpro/fastgwa_pixpro/<EXTERNAL: encoder checkpoint>"
)
DISC_OUT = f"{FASTGWA_DIR}/output"
# Canonical replication phenotype path per docs/data_paths.md §7a (not the
# stale `input/` folder next to discovery, which has a different, broken cohort).
REP_IN = f"{FASTGWA_DIR}/replication/input"
BGEN = "<EXTERNAL: all_filtered.bgen>"
SAMPLE = "<EXTERNAL: MRI_samples_chr1.sample>"

H2_CSV = (
    "<EXTERNAL: jagwas_paper>/"
    "paper_draft/figures/h2_table.csv"
)
N_DISCOVERY = 22878

OUT_DIR = Path(
    "<EXTERNAL: jagwas_paper>/"
    "post_gwas_analysis/results/sbayesrc/sweep"
)
PGEN_CACHE_DIR = OUT_DIR / "_pgen_cache"
PGEN_PREFIX = PGEN_CACHE_DIR / "rep_genotypes"

REGIONS = [
    "Brain_Stem_or_4th_Ventricle", "CSF",
    "Left_Accumbens-area", "Right_Accumbens-area",
    "Left_Amygdala", "Right_Amygdala",
    "Left_Caudate", "Right_Caudate",
    "Left_Hippocampus", "Right_Hippocampus",
    "Left_Pallidum", "Right_Pallidum",
    "Left_Putamen", "Right_Putamen",
    "Left_Thalamus_Proper", "Right_Thalamus-Proper",
]

def replication_qt_exists(region: str, dim: int) -> bool:
    return (Path(REP_IN) /
            f"replication_<EXTERNAL: encoder checkpoint>{region}_QT{dim}").exists()


def get_replication_qt_path(region: str, dim: int) -> Path:
    return Path(REP_IN) / f"replication_<EXTERNAL: encoder checkpoint>{region}_QT{dim}"


def fastgwa_to_ma(fastgwa_path: Path, ma_path: Path, n_default: int):
    df = pd.read_csv(fastgwa_path, sep="\t", low_memory=False)
    df = df.rename(columns={"AF1": "freq", "BETA": "b", "SE": "se", "P": "p"})
    df = df[["SNP", "A1", "A2", "freq", "b", "se", "p", "N"]]
    df = df.dropna(subset=["b", "se", "p"])
    df = df[(df["freq"] > 0) & (df["freq"] < 1)]
    df["N"] = df["N"].fillna(n_default).astype(int)
    df.to_csv(ma_path, sep=" ", index=False)
    return len(df)


def run(cmd, log_path=None):
    t0 = time.time()
    with open(log_path, "w") if log_path else open("/dev/null", "w") as fh:
        ret = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return ret.returncode, (time.time() - t0) / 60


def get_top_dims_per_region(h2_csv: Path, top_k: int):
    """For each region, return the top-k highest-h² dims. Canonical replication
    has all 128 dims for all 16 regions (per docs/data_paths.md §7a), so no
    replication-existence fallback is needed."""
    h2 = pd.read_csv(h2_csv)
    out = {}
    for reg in REGIONS:
        h2_reg = reg.replace("_Thalamus_Proper", "_Thalamus-Proper")
        sub = h2[h2["reg"] == h2_reg].sort_values("h2", ascending=False)
        out[reg] = [(int(r["qt"]), float(r["h2"])) for _, r in sub.head(top_k).iterrows()]
    return out


def build_pgen_cache(threads: int):
    PGEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pgen = Path(str(PGEN_PREFIX) + ".pgen")
    if pgen.exists() and pgen.stat().st_size > 1_000_000_000:
        log.info(f"  PGEN cache already exists: {pgen} ({pgen.stat().st_size // (1024**3)} GB)")
        return
    log.info(f"  Building PGEN cache from BGEN — one-time conversion, ~12 min ...")
    cmd = [PLINK2, "--bgen", BGEN, "ref-first", "--sample", SAMPLE,
           "--rm-dup", "force-first",
           "--make-pgen",
           "--out", str(PGEN_PREFIX),
           "--threads", str(threads)]
    rc, wall = run(cmd, log_path=str(PGEN_PREFIX) + ".plink2.log")
    if rc != 0:
        raise RuntimeError(f"PGEN build failed; see {PGEN_PREFIX}.plink2.log")
    log.info(f"  PGEN cache built: {wall:.1f} min")


def score_one(region: str, dim: int, h2_val: float, threads: int) -> dict:
    """Run full SBayesRC + PLINK2 score + R² for one (region, dim)."""
    prefix = OUT_DIR / f"{region}_QT{dim}"
    log.info(f"--- {region} QT{dim} (h²={h2_val:.4f}) ---")

    # Step 1: FastGWA → .ma
    fastgwa = (Path(DISC_OUT) /
               f"discovery_<EXTERNAL: encoder checkpoint>{region}_QT{dim}.fastGWA.fastGWA")
    if not fastgwa.exists():
        log.error(f"  FastGWA missing: {fastgwa}")
        return {"region": region, "dim": dim, "h2": h2_val, "status": "no_fastgwa"}
    ma_file = OUT_DIR / f"{region}_QT{dim}.ma"
    if not ma_file.exists() or ma_file.stat().st_size < 1_000_000:
        t0 = time.time()
        n = fastgwa_to_ma(fastgwa, ma_file, N_DISCOVERY)
        log.info(f"  .ma: {n:,} SNPs in {(time.time()-t0)/60:.1f} min")

    # Step 2: SBayesRC impute
    imputed_ma = Path(str(prefix) + ".imputed.ma")
    if not imputed_ma.exists():
        cmd = [GCTB, "--ldm-eigen", LD_DIR, "--gwas-summary", str(ma_file),
               "--impute-summary", "--out", str(prefix),
               "--thread", str(threads)]
        rc, wall = run(cmd, log_path=str(prefix) + ".impute.log")
        if rc != 0:
            log.error(f"  impute failed (see {prefix}.impute.log)")
            return {"region": region, "dim": dim, "h2": h2_val, "status": "impute_failed"}
        log.info(f"  impute: {wall:.1f} min")

    # Step 3: SBayesRC main
    snpres = Path(str(prefix) + ".snpRes")
    if not snpres.exists():
        cmd = [GCTB, "--ldm-eigen", LD_DIR, "--gwas-summary", str(imputed_ma),
               "--sbayes", "RC", "--annot", ANNOT, "--out", str(prefix),
               "--thread", str(threads)]
        rc, wall = run(cmd, log_path=str(prefix) + ".sbayesrc.log")
        if rc != 0:
            log.error(f"  sbayesrc failed (see {prefix}.sbayesrc.log)")
            return {"region": region, "dim": dim, "h2": h2_val, "status": "sbayesrc_failed"}
        log.info(f"  sbayesrc: {wall:.1f} min")

    # Step 4: PLINK2 score on cached PGEN
    sscore = Path(str(prefix) + ".sscore")
    if not sscore.exists():
        sn = pd.read_csv(snpres, sep=r"\s+", engine="python")
        score_file = OUT_DIR / f"{region}_QT{dim}.score"
        sn[["Name", "A1", "A1Effect"]].to_csv(score_file, sep="\t", index=False, header=False)
        cmd = [PLINK2, "--pfile", str(PGEN_PREFIX),
               "--score", str(score_file), "1", "2", "3",
               "no-mean-imputation",
               "--out", str(prefix),
               "--threads", str(threads)]
        rc, wall = run(cmd, log_path=str(prefix) + ".plink2.log")
        if rc != 0:
            log.error(f"  plink2 score failed (see {prefix}.plink2.log)")
            return {"region": region, "dim": dim, "h2": h2_val, "status": "plink2_failed"}
        log.info(f"  plink2 score: {wall:.1f} min")

    # Step 5: R² vs replication QT
    pgs = pd.read_csv(sscore, sep=r"\s+", engine="python")
    pgs_col = next((c for c in pgs.columns if c.endswith("_AVG")), pgs.columns[-1])
    pgs = pgs[["IID", pgs_col]].rename(columns={pgs_col: "PGS"})
    pgs["IID"] = pgs["IID"].astype(str)

    rep_qt = get_replication_qt_path(region, dim)
    qt = pd.read_csv(rep_qt, sep=r"\s+", engine="python",
                     header=0, names=["FID", "IID", "PHENO"], skiprows=1)
    qt["IID"] = qt["IID"].astype(str)
    qt["PHENO"] = pd.to_numeric(qt["PHENO"], errors="coerce")
    qt = qt.dropna(subset=["PHENO"])

    df = pgs.merge(qt[["IID", "PHENO"]], on="IID", how="inner").dropna()
    if len(df) < 100:
        return {"region": region, "dim": dim, "h2": h2_val, "status": "too_few_subjects"}
    r = float(np.corrcoef(df["PGS"].astype(float), df["PHENO"].astype(float))[0, 1])
    r2 = r * r
    p = float(2 * stats.norm.sf(abs(r) * np.sqrt(len(df) - 2)))
    log.info(f"  → N={len(df):,}  r={r:+.4f}  R²={r2:.5f}  p={p:.2e}")
    return {"region": region, "dim": dim, "h2": h2_val, "N_rep": len(df),
            "r": r, "R2": r2, "p": p, "status": "ok"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim-ranks", default="1",
                   help="space-separated h²-ranks to process per region. "
                        "'1' = top dim per region (Stage C); "
                        "'1 2 3 4 5' = top-5 dims per region (Stage A).")
    p.add_argument("--only-regions", nargs="+", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-pgen-cache", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in args.dim_ranks.split()]
    top_k = max(ranks)
    regions = args.only_regions if args.only_regions else REGIONS

    log.info("=" * 72)
    log.info("SBayesRC PGS sweep")
    log.info("=" * 72)
    log.info(f"Ranks to process per region: {ranks}")
    log.info(f"Regions: {len(regions)}")
    log.info(f"Threads: {args.threads}")
    log.info(f"Output:  {OUT_DIR}")

    # PGEN cache (one-time BGEN → PGEN conversion)
    if not args.no_pgen_cache:
        log.info("--- PGEN cache ---")
        build_pgen_cache(args.threads)

    # Identify top-k dims per region
    log.info("--- Identifying top-h² dims per region ---")
    top_dims = get_top_dims_per_region(Path(H2_CSV), top_k)
    for reg in regions:
        log.info(f"  {reg}: {top_dims[reg]}")

    # Build list of (region, dim, h², rank) to process
    jobs = []
    for reg in regions:
        for rank in ranks:
            if rank - 1 < len(top_dims[reg]):
                dim, h2 = top_dims[reg][rank - 1]
                jobs.append((reg, int(dim), float(h2), rank))
    log.info(f"\nTotal jobs to process: {len(jobs)}")

    # Existing results — append to / append to master table
    summary_csv = OUT_DIR / "sweep_R2.csv"
    existing = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()

    t_total = time.time()
    results = []
    for idx, (reg, dim, h2, rank) in enumerate(jobs):
        t0 = time.time()
        log.info(f"\n[{idx + 1}/{len(jobs)}] {reg} QT{dim} (rank {rank}, h²={h2:.4f})")
        result = score_one(reg, dim, h2, args.threads)
        result["rank"] = rank
        result["wall_min"] = (time.time() - t0) / 60
        results.append(result)
        # Write summary after each run (resumable)
        new_df = pd.DataFrame(results)
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.to_csv(summary_csv, index=False)
        log.info(f"  total elapsed: {(time.time() - t_total) / 60:.1f} min")

    log.info(f"\n=== SWEEP DONE in {(time.time() - t_total) / 60:.1f} min ===")
    df = pd.DataFrame(results)
    if "R2" in df.columns:
        ok = df[df["status"] == "ok"]
        if len(ok):
            log.info(f"Mean R² (ok runs): {ok['R2'].mean():.4f}")
            log.info(f"Range: {ok['R2'].min():.4f} – {ok['R2'].max():.4f}")
            log.info(f"Best:  {ok.loc[ok['R2'].idxmax(), 'region']} "
                     f"QT{int(ok.loc[ok['R2'].idxmax(), 'dim'])}: R²={ok['R2'].max():.4f}")


if __name__ == "__main__":
    main()
