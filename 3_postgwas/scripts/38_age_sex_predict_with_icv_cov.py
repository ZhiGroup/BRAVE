#!/usr/bin/env python3
"""
38_age_sex_predict_with_icv_cov.py
Per-region BRE → age and sex prediction with head-size (UK Biobank Data-Field
25000, volumetric scaling factor) included as a covariate. 10-fold CV linear
regression for age (MAE) and logistic regression for sex (accuracy).

This replaces the original notebook-derived predictions
(`discovery_local_scatter_accu_r2_with_hue_df.csv`) for Supplementary Fig S47,
because the original used only the 128 BRE dimensions without controlling for
overall brain size. Sex classification in particular is partly driven by
absolute brain volume (men have larger absolute volumes); the ICV-adjusted
estimate is the more conservative and reviewer-defensible value.

Inputs:
  - Per-dim FastGWA phenotype files (one file per region per dim):
      <cfg.gwas.pheno_dir>/input/discovery_<region>_QT<dim>
    Each file: tab-separated (fiid, iid, QT<dim>)
  - QCOVAR (AGE, 25000): <cfg.gwas.covar_dir>/T1_qcovar_discovery_v2
  - CCOVAR (SEX):        <cfg.gwas.covar_dir>/T1_ccovar_discovery_v2

Output:
  results/phesant/age_sex_with_icv_cov_per_region.csv
    columns: region, n_subjects, age_mae_yr, sex_accuracy

Compute env: Python 3.7
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

QT_BASE = Path(str(cfg.gwas.pheno_dir)) / "input"
QT_TEMPLATE = "discovery_{region}_QT{dim}"
QCOVAR = Path(str(cfg.gwas.covar_dir)) / "T1_qcovar_discovery_v2"
CCOVAR = Path(str(cfg.gwas.covar_dir)) / "T1_ccovar_discovery_v2"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "phesant"

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
N_DIMS = 128
N_FOLDS = 10
RANDOM_STATE = 0


def load_bre_matrix(region: str, qt_base: Path) -> pd.DataFrame:
    """Load all 128 QT files for a region; return DataFrame indexed by eid."""
    frames = []
    for d in range(N_DIMS):
        f = qt_base / QT_TEMPLATE.format(region=region, dim=d)
        if not f.exists():
            raise FileNotFoundError(f"Missing QT file: {f}")
        df = pd.read_csv(f, sep="\t", usecols=[1, 2])
        df.columns = ["eid", f"QT{d}"]
        df["eid"] = df["eid"].astype(int)
        frames.append(df.set_index("eid"))
    return pd.concat(frames, axis=1, join="inner")


def load_covariates() -> pd.DataFrame:
    """Return DataFrame with eid, age, sex, icv (UKB 25000)."""
    q = pd.read_csv(QCOVAR, sep=r"\s+", engine="python")
    c = pd.read_csv(CCOVAR, sep=r"\s+", engine="python")
    q["eid"] = q["IID"].astype(int)
    c["eid"] = c["IID"].astype(int)
    out = q[["eid", "AGE", "25000"]].rename(columns={"AGE": "age", "25000": "icv"})
    out = out.merge(c[["eid", "SEX"]].rename(columns={"SEX": "sex"}), on="eid")
    return out


def run_region(region: str, cov: pd.DataFrame, qt_base: Path):
    """Returns (n_subjects, age_mae, sex_acc, age_mae_no_icv, sex_acc_no_icv)."""
    t0 = time.time()
    bre = load_bre_matrix(region, qt_base)
    log.info(f"  {region}: BRE loaded {bre.shape}  ({time.time()-t0:.1f}s)")

    panel = bre.merge(cov.set_index("eid"), left_index=True, right_index=True, how="inner")
    n_before = len(panel)
    panel = panel.dropna(subset=["age", "sex", "icv"] + [f"QT{d}" for d in range(N_DIMS)])
    n_dropped = n_before - len(panel)
    if n_dropped > 0:
        log.info(f"  {region}: dropped {n_dropped} subjects with NaN BRE / covariate values")
    log.info(f"  {region}: analysis cohort N={len(panel):,}")

    bre_cols = [f"QT{d}" for d in range(N_DIMS)]
    X_bre = panel[bre_cols].values.astype(np.float64)
    X_icv = panel[["icv"]].values.astype(np.float64)
    X_full = np.hstack([X_bre, X_icv])
    y_age = panel["age"].values.astype(np.float64)
    y_sex = panel["sex"].values.astype(int)

    # 10-fold CV — same seed for both
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Standardise features for fast lbfgs convergence. The estimators are wrapped
    # in a Pipeline so the scaler is fit on each training fold (no leakage).
    def _lin_pipe():
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

    def _log_pipe():
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs"))])

    # Age: linear regression, MAE
    age_mae_full = -cross_val_score(_lin_pipe(), X_full, y_age,
                                    cv=kf, scoring="neg_mean_absolute_error").mean()
    age_mae_bre  = -cross_val_score(_lin_pipe(), X_bre,  y_age,
                                    cv=kf, scoring="neg_mean_absolute_error").mean()

    # Sex: logistic regression, accuracy
    sex_acc_full = cross_val_score(_log_pipe(), X_full, y_sex, cv=skf,
                                   scoring="accuracy").mean()
    sex_acc_bre  = cross_val_score(_log_pipe(), X_bre,  y_sex, cv=skf,
                                   scoring="accuracy").mean()

    log.info(f"  {region}: age MAE  full={age_mae_full:.4f}  BRE-only={age_mae_bre:.4f}")
    log.info(f"  {region}: sex acc  full={sex_acc_full:.4f}  BRE-only={sex_acc_bre:.4f}")
    return {
        "region": region,
        "n_subjects": len(panel),
        "age_mae_yr_with_icv":    age_mae_full,
        "sex_accuracy_with_icv":  sex_acc_full,
        "age_mae_yr_no_icv":      age_mae_bre,
        "sex_accuracy_no_icv":    sex_acc_bre,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qt-base", type=Path, default=QT_BASE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--only", nargs="+", default=None,
                   help="Subset of regions to run (folder names)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = args.only if args.only else REGIONS

    log.info("=== Script 38: BRE → age/sex with ICV (UKB 25000) covariate, 10-fold CV ===")
    log.info(f"Output: {args.out_dir / 'age_sex_with_icv_cov_per_region.csv'}")
    log.info("Loading covariates ...")
    cov = load_covariates()
    log.info(f"  cov rows: {len(cov):,}")

    rows = []
    t0 = time.time()
    for r in regions:
        try:
            rows.append(run_region(r, cov, args.qt_base))
        except Exception as e:
            log.error(f"  FAILED {r}: {e}")
    df = pd.DataFrame(rows)
    out_path = args.out_dir / "age_sex_with_icv_cov_per_region.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nWrote: {out_path}")
    log.info(f"Total wall: {(time.time()-t0)/60:.1f} min")
    log.info("\n=== Per-region results ===\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
