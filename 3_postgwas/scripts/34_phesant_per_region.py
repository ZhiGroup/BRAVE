#!/usr/bin/env python3
"""
34_phesant_per_region.py  (canonical single PHESANT, 2026-06-06)
PHESANT-style [Millard 2018] phenome-wide association scan for each of
the 16 JAGWAS regions. Predictor = per-region BRE composite scalar
(mean of z-scored top-5 h²-ranked BRE dimensions per region).

Covariates (field-standard PHESANT protocol + regional-volume control):
  - age (continuous)
  - age² (non-linear age term)
  - sex (binary)
  - scanner site (UKB Data-Field 54-2.0; one-hot encoded with reference)
  - intracranial volume (UKB Data-Field 25000-2.0, T1 volumetric scaling
    factor — head-size proxy; used in place of CAT12 TIV)
  - corresponding regional FreeSurfer subcortical volume (UKB Data-Fields
    25011-2.0 through 25025-2.0 for the 14 bilateral subcortical structures
    plus brainstem). CSF region uses the preceding covariate set only — no
    direct CSF Data-Field exists in the 25xxx block.

Continuous outcomes are rank-inverse-normal transformed (Blom) before
linear regression to match the field-standard protocol.

For each region × UK Biobank phenotype field:
  - continuous → IRN + OLS
  - ordinal    → OLS (no IRN)
  - binary     → logistic regression with Wald-sandwich SE
BH-FDR within each region's set of valid tests, q < 0.05.

Outputs: <OUT>/<region>/phesant_results.csv

Compute env: Python 3.7
"""
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

# Reuse helpers from the v2 module (a copy of the updated Script 34)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _phesant_per_region_v2 import (  # noqa: E402
    REGIONS, SITE_FIELD, ICV_FIELD, SKIP_FIELDS, UKB_MISSING,
    get_top_dims, build_composite, load_covariates, scan_region,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

H2_CSV_DEFAULT = Path("<EXTERNAL: LDSC h2_table.csv (per-region per-dim canonical h2 table)>")
QT_BASE_DEFAULT = Path(str(cfg.gwas.pheno_dir)) / "input"
QCOVAR_DEFAULT = Path(str(cfg.gwas.covar_dir)) / "T1_qcovar_discovery_v2"
CCOVAR_DEFAULT = Path(str(cfg.gwas.covar_dir)) / "T1_ccovar_discovery_v2"
UKB_CSV_DEFAULT = Path(str(cfg.ukb.master_csv))
OUT_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "phesant_icv_volume"

REGION_TO_VOLUME_FIELD = {
    "Brain_Stem_or_4th_Ventricle": "25011-2.0",
    "CSF": None,
    "Left_Thalamus_Proper": "25012-2.0",
    "Right_Thalamus-Proper": "25013-2.0",
    "Left_Caudate": "25014-2.0",
    "Right_Caudate": "25015-2.0",
    "Left_Putamen": "25016-2.0",
    "Right_Putamen": "25017-2.0",
    "Left_Pallidum": "25018-2.0",
    "Right_Pallidum": "25019-2.0",
    "Left_Hippocampus": "25020-2.0",
    "Right_Hippocampus": "25021-2.0",
    "Left_Amygdala": "25022-2.0",
    "Right_Amygdala": "25023-2.0",
    "Left_Accumbens-area": "25024-2.0",
    "Right_Accumbens-area": "25025-2.0",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h2-csv", type=Path, default=H2_CSV_DEFAULT)
    p.add_argument("--qt-base", type=Path, default=QT_BASE_DEFAULT)
    p.add_argument("--qcovar", type=Path, default=QCOVAR_DEFAULT)
    p.add_argument("--ccovar", type=Path, default=CCOVAR_DEFAULT)
    p.add_argument("--ukb-csv", type=Path, default=UKB_CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--max-fields", type=int, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = args.only if args.only else REGIONS

    log.info("=== Script 34b v2: PHESANT sensitivity (primary covariates + regional volume) ===")
    log.info(f"UKB CSV:     {args.ukb_csv}")
    log.info(f"Output dir:  {args.out_dir}")
    log.info(f"Regions:     {len(regions)}")
    log.info("Covariates:  age + age² + sex + scanner site (54-2.0) + ICV (25000-2.0) + regional volume")
    log.info("Continuous outcomes: rank-inverse-normal transformed (Blom)")

    log.info("Reading UKB column list ...")
    ukb_cols_all = pd.read_csv(args.ukb_csv, nrows=0).columns.tolist()
    pheno_cols = [c for c in ukb_cols_all
                  if c != "eid" and c.split("-")[0] not in SKIP_FIELDS]
    if args.max_fields:
        pheno_cols = pheno_cols[:args.max_fields]
    log.info(f"Phenotype columns to test: {len(pheno_cols):,}")

    needed_cov_cols = {SITE_FIELD, ICV_FIELD}
    for region in regions:
        regvol = REGION_TO_VOLUME_FIELD.get(region)
        if regvol is not None:
            needed_cov_cols.add(regvol)
    needed_cov_cols = {c for c in needed_cov_cols if c in ukb_cols_all}
    log.info(f"Covariate columns from UKB: {sorted(needed_cov_cols)}")

    load_cols = ["eid"] + pheno_cols + [c for c in needed_cov_cols if c not in pheno_cols]
    load_cols = list(dict.fromkeys(load_cols))

    log.info(f"Loading UKB phenome CSV ... ({args.ukb_csv.stat().st_size//(1024*1024)} MB)")
    t_load = time.time()
    ukb = pd.read_csv(args.ukb_csv, usecols=load_cols, low_memory=False)
    ukb["eid"] = ukb["eid"].astype(int)
    ukb = ukb.set_index("eid")
    log.info(f"  UKB loaded in {(time.time()-t_load)/60:.1f} min — shape {ukb.shape}")

    log.info("Loading age/sex covariates ...")
    cov_df = load_covariates(args.qcovar, args.ccovar)
    log.info(f"  age/sex covariates loaded: N={len(cov_df):,}")

    total_t0 = time.time()
    for region in regions:
        log.info(f"\n=== Region: {region} ===")
        try:
            dims = get_top_dims(args.h2_csv, region, top_k=args.top_k)
            log.info(f"  top-{args.top_k} h² dims: {dims}")
            composite_df = build_composite(args.qt_base, region, dims)
            regvol = REGION_TO_VOLUME_FIELD.get(region)
            scan_region(region, composite_df, cov_df, ukb, pheno_cols, args.out_dir,
                        extra_region_volume_field=regvol)
        except Exception as e:
            log.error(f"  FAILED {region}: {e}")
    log.info(f"\n=== DONE in {(time.time()-total_t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
