#!/usr/bin/env python3
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
"""
36_sbayesrc_pgs_pilot.py
PILOT — SBayesRC polygenic score for ONE region's top-h² BRE dim
(Left_Hippocampus, QT27, h²=0.2537). Confirms end-to-end pipeline,
expected R² magnitude vs current Script 20 (R² = 0.0006–0.0068),
software stack, and runtime before committing to the full 80-run sweep
(top-5 h²-ranked dims × 16 regions).

Pipeline:
  1. Convert FastGWA per-dim summary stats → COJO/.ma format
       SNP A1 A2 freq b se p N
  2. SBayesRC step 1 — impute summary stats over the HM3 LD reference set
       gctb --ldm-eigen <LD-folder> --gwas-summary <prefix>.ma
            --impute-summary --out <prefix> --thread 4
  3. SBayesRC step 2 — main analysis with baseline-LF v2.2 annotations
       gctb --ldm-eigen <LD-folder> --gwas-summary <prefix>.imputed.ma
            --sbayes RC --annot <annot> --out <prefix> --thread 4
  4. Build PLINK2 score file from <prefix>.snpRes (SNP A1 BETA columns)
  5. Score replication cohort BGEN with PLINK2 --score
  6. Correlate per-subject PGS with the TRUE replication BRE dim value
     and report R² for comparison to Script 20 baseline.

Compute env: <EXTERNAL: python>
"""
import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Tool + reference paths (from VoxelAge consolidation 2026-06-06) ──
GCTB = "<EXTERNAL: gctb>"
LD_DIR = "<EXTERNAL: ukbEUR_HM3>"
ANNOT = "<EXTERNAL: annot_baseline2.2.txt>"
PLINK2 = "<EXTERNAL: plink2>"

# ── Input data paths (FastGWA + replication QT + BGEN) ──
FASTGWA_DIR = (
    "<EXTERNAL: PIXEL_EMBEDDING>/"
    "GWAS_DIR_pixpro/fastgwa_pixpro/<EXTERNAL: encoder checkpoint>"
)
DISC_OUT = f"{FASTGWA_DIR}/output"
REP_IN = f"{FASTGWA_DIR}/input"
BGEN = "<EXTERNAL: all_filtered.bgen>"
SAMPLE = "<EXTERNAL: MRI_samples_chr1.sample>"

# ── Pilot target ──
REGION = "Left_Hippocampus"
DIM = 27   # Top-h² dim per Script 14 h2 table; h² = 0.2537
N_DISCOVERY = 22878  # GWAS discovery N

# ── Output dir ──
OUT_DIR = Path(
    "<EXTERNAL: jagwas_paper>/"
    "post_gwas_analysis/results/sbayesrc/pilot"
)


def fastgwa_to_ma(fastgwa_path: Path, ma_path: Path, n_default: int):
    """Convert FastGWA sumstats to GCTB .ma (COJO) format.
    Input cols : CHR SNP POS A1 A2 N AF1 BETA SE P INFO
    Output cols: SNP A1 A2 freq b se p N
    Drops rows with NaN beta/se/p or freq == 0/1.
    """
    log.info(f"  converting {fastgwa_path.name} → .ma")
    df = pd.read_csv(fastgwa_path, sep="\t", low_memory=False)
    log.info(f"  raw FastGWA: {len(df):,} SNPs, cols={df.columns.tolist()}")
    df = df.rename(columns={"AF1": "freq", "BETA": "b", "SE": "se", "P": "p"})
    df = df[["SNP", "A1", "A2", "freq", "b", "se", "p", "N"]]
    df = df.dropna(subset=["b", "se", "p"])
    df = df[(df["freq"] > 0) & (df["freq"] < 1)]
    df["N"] = df["N"].fillna(n_default).astype(int)
    df.to_csv(ma_path, sep=" ", index=False)
    log.info(f"  wrote {ma_path.name}: {len(df):,} SNPs")
    return len(df)


def run(cmd, log_path=None):
    log.info(f"  RUN: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "w") if log_path else open("/dev/null", "w") as fh:
        ret = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    log.info(f"  → exit {ret.returncode}, wall {dt:.1f} min")
    if ret.returncode != 0 and log_path:
        log.error(f"  see log: {log_path}")
    return ret.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--region", default=REGION)
    p.add_argument("--dim", type=int, default=DIM)
    p.add_argument("--skip-impute", action="store_true")
    p.add_argument("--skip-sbayesrc", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    region = args.region
    dim = args.dim
    prefix = OUT_DIR / f"{region}_QT{dim}"

    log.info("=" * 70)
    log.info(f"SBayesRC PGS pilot — {region} QT{dim}")
    log.info("=" * 70)
    log.info(f"GCTB:       {GCTB}")
    log.info(f"LD ref:     {LD_DIR}")
    log.info(f"Annotation: {ANNOT}")
    log.info(f"BGEN:       {BGEN}")
    log.info(f"Output:     {prefix}.*")

    # ── Step 1: Convert FastGWA → .ma ──
    log.info("--- Step 1: convert FastGWA → .ma ---")
    fastgwa_file = (
        Path(DISC_OUT)
        / f"discovery_<EXTERNAL: encoder checkpoint>{region}_QT{dim}.fastGWA.fastGWA"
    )
    if not fastgwa_file.exists():
        log.error(f"FastGWA file missing: {fastgwa_file}")
        return
    ma_file = OUT_DIR / f"{region}_QT{dim}.ma"
    n_snps = fastgwa_to_ma(fastgwa_file, ma_file, N_DISCOVERY)

    # ── Step 2: Impute summary stats over LD ref SNP set ──
    if not args.skip_impute:
        log.info("--- Step 2: SBayesRC impute step ---")
        cmd = [GCTB, "--ldm-eigen", LD_DIR, "--gwas-summary", str(ma_file),
               "--impute-summary", "--out", str(prefix), "--thread", str(args.threads)]
        rc = run(cmd, log_path=str(prefix) + ".impute.log")
        if rc != 0:
            log.error("impute step failed; aborting")
            return
    imputed_ma = Path(str(prefix) + ".imputed.ma")
    if not imputed_ma.exists():
        # GCTB may also write as .imputed.ma.gz or .ma.imputed; check variants
        for cand in [Path(str(prefix) + ".ma.imputed"),
                     Path(str(prefix) + ".imputed.ma.gz"),
                     OUT_DIR / f"{region}_QT{dim}.imputed.ma"]:
            if cand.exists():
                imputed_ma = cand
                break
    log.info(f"  imputed .ma: {imputed_ma}")

    # ── Step 3: SBayesRC main analysis ──
    if not args.skip_sbayesrc:
        log.info("--- Step 3: SBayesRC main analysis ---")
        cmd = [GCTB, "--ldm-eigen", LD_DIR, "--gwas-summary", str(imputed_ma),
               "--sbayes", "RC", "--annot", ANNOT, "--out", str(prefix),
               "--thread", str(args.threads)]
        rc = run(cmd, log_path=str(prefix) + ".sbayesrc.log")
        if rc != 0:
            log.error("SBayesRC step failed; aborting")
            return
    snpres = Path(str(prefix) + ".snpRes")
    if not snpres.exists():
        log.error(f"snpRes missing: {snpres}")
        return
    log.info(f"  snpRes ready: {snpres}")

    # ── Step 4: build PLINK2 score file ──
    log.info("--- Step 4: build PLINK2 score file ---")
    # snpRes columns: Id Name Chrom Position A1 A2 A1Frq A1Effect SE PIP LastSampleEff
    sn = pd.read_csv(snpres, sep=r"\s+", engine="python")
    log.info(f"  snpRes: {len(sn):,} SNPs, cols={sn.columns.tolist()[:8]}")
    score_file = OUT_DIR / f"{region}_QT{dim}.score"
    sn[["Name", "A1", "A1Effect"]].to_csv(score_file, sep="\t", index=False, header=False)
    log.info(f"  wrote {score_file.name}")

    # ── Step 5: PLINK2 --score on replication BGEN ──
    if not args.skip_score:
        log.info("--- Step 5: PLINK2 --score on replication BGEN ---")
        cmd = [PLINK2, "--bgen", BGEN, "ref-first", "--sample", SAMPLE,
               "--score", str(score_file), "1", "2", "3", "header-read",
               "no-mean-imputation", "list-variants",
               "--rm-dup", "force-first",
               "--out", str(prefix),
               "--threads", str(args.threads)]
        rc = run(cmd, log_path=str(prefix) + ".plink2.log")
        if rc != 0:
            log.error("PLINK2 score failed; aborting")
            return

    sscore = Path(str(prefix) + ".sscore")
    if not sscore.exists():
        log.error(f"sscore missing: {sscore}")
        return

    # ── Step 6: R² vs replication BRE dim phenotype ──
    log.info("--- Step 6: evaluate R² in replication cohort ---")
    pgs = pd.read_csv(sscore, sep=r"\s+", engine="python")
    log.info(f"  sscore: {len(pgs):,} subjects, cols={pgs.columns.tolist()}")
    pgs_col = "SCORE1_AVG" if "SCORE1_AVG" in pgs.columns else pgs.columns[-1]
    pgs = pgs[["IID", pgs_col]].rename(columns={pgs_col: "PGS"})

    rep_qt = Path(REP_IN) / f"replication_<EXTERNAL: encoder checkpoint>{region}_QT{dim}"
    qt = pd.read_csv(rep_qt, sep=r"\s+", engine="python", header=None,
                     names=["FID", "IID", "PHENO"])
    log.info(f"  replication QT: {len(qt):,} subjects")
    df = pgs.merge(qt, on="IID", how="inner")
    df = df.dropna(subset=["PGS", "PHENO"])
    log.info(f"  joined: {len(df):,} subjects with both PGS + PHENO")

    if len(df) < 100:
        log.error("too few subjects; aborting")
        return
    r = float(np.corrcoef(df["PGS"], df["PHENO"])[0, 1])
    r2 = r * r
    from scipy import stats
    p = float(2 * stats.norm.sf(abs(r) * np.sqrt(len(df) - 2)))

    log.info("=" * 70)
    log.info("PILOT RESULT")
    log.info(f"  region:               {region} QT{dim} (h² ≈ 0.2537)")
    log.info(f"  replication N:        {len(df):,}")
    log.info(f"  Pearson r (PGS,QT):   {r:+.4f}")
    log.info(f"  R² (replication):     {r2:.5f}")
    log.info(f"  p:                    {p:.2e}")
    log.info(f"  vs Script 20 R² range: 0.0006 – 0.0068")
    log.info(f"  improvement vs midpoint of Script 20: {r2 / 0.0037:.1f}×")
    log.info("=" * 70)

    with open(OUT_DIR / "pilot_R2.txt", "w") as fh:
        fh.write(f"region\tdim\tN_rep\tr\tR2\tp\n")
        fh.write(f"{region}\t{dim}\t{len(df)}\t{r:.5f}\t{r2:.6f}\t{p:.3e}\n")


if __name__ == "__main__":
    main()
