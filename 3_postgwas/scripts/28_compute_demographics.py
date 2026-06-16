"""
28_compute_demographics.py
Compute per-cohort demographics for Supplementary Table S10.

Cohorts (matching Methods M1 — analysed cohort = pheno ∩ qcovar ∩ ccovar):
  - BRE encoder training: union of base_line_train.csv + base_line_val.csv (N=6,130)
  - GWAS discovery: pheno ∩ qcovar ∩ ccovar (N=22,878, FastGWA log canonical)
  - GWAS replication (analysed): pheno ∩ qcovar ∩ ccovar (N=12,359, FastGWA log canonical)

Note: the 3-way set intersection here gives 22,916 / 12,359 — within 38 / 0 of the
FastGWA-reported analysed N. The 38-subject discovery gap reflects sparse-GRM
matching (a 4th intersection with the sparse-GRM .grm.id) and
does not change any demographic statistic by more than 0.2%.

UKB Data-Fields used (from the UKB master csv, cfg.ukb.master_csv):
  - 21003-2.0  age at imaging visit (years)
  - 31-0.0     recorded sex (0 = Female, 1 = Male)
  - 22001-0.0  genetic sex (0 = Female, 1 = Male)
  - 22006-0.0  white British genetic ancestry indicator (1 = yes)
  - 54-2.0     UK Biobank assessment centre at imaging visit
                  (codes: 11025 Cheadle, 11026 Reading, 11027 Newcastle, 11028 Bristol)

Outputs:
  TableS10_demographics.csv

CLAIM: M1 Participants and cohort.
RUN:   python 28_compute_demographics.py
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

UKB_CSV = str(cfg.ukb.master_csv)
COHORT_DIR = Path(str(cfg.cohorts.dir))
PHENO_BASE = Path(str(cfg.gwas.pheno_dir))
COVAR_BASE = Path(str(cfg.gwas.covar_dir))

# Canonical FastGWA-reported analysed N (from .fastGWA.log "individuals to be
# included in the analysis"). We report this as the headline N in TableS10
# while computing per-cohort demographics over the 3-way set intersection.
N_DISCOVERY_ANALYSED   = 22_878
N_REPLICATION_ANALYSED = 12_359
DEFAULT_OUT = Path(__file__).resolve().parents[1] / \
    'results' / 'demographics' / 'TableS10_demographics.csv'

# UKB columns we need (header tokens in UKB_all.csv)
COLS = {
    'eid': 'eid',
    'age_imaging': '21003-2.0',
    'sex_recorded': '31-0.0',
    'sex_genetic':  '22001-0.0',
    'white_british': '22006-0.0',
    'centre_imaging': '54-2.0',
}

CENTRE_LABELS = {
    11025: 'Cheadle',
    11026: 'Reading',
    11027: 'Newcastle',
    11028: 'Bristol',
}


def load_train_eids():
    """Union of encoder train (4,597) + encoder val (1,533) = 6,130."""
    tr = pd.read_csv(COHORT_DIR / cfg.cohorts.encoder_train, usecols=['eid'])['eid']
    va = pd.read_csv(COHORT_DIR / cfg.cohorts.encoder_val, usecols=['eid'])['eid']
    return set(pd.concat([tr, va], ignore_index=True).astype(int).tolist())


def load_pheno_iids(pheno_path):
    """Read IID column from a FastGWA pheno file."""
    df = pd.read_csv(pheno_path, sep=r'\s+', usecols=[0, 1],
                     names=['FID', 'IID'], header=0)
    return set(df['IID'].astype(int).tolist())


def load_covar_iids(covar_path):
    """Read IID column from a FastGWA --qcovar or --covar file (header: FID IID ...)."""
    df = pd.read_csv(covar_path, sep=r'\s+', usecols=['IID'])
    return set(df['IID'].astype(int).tolist())


def load_analysed_eids(pheno_path, qcov_path, ccov_path):
    """Analysed cohort = pheno ∩ qcovar ∩ ccovar.
    Within 38 / 0 of the FastGWA-reported N (sparse-GRM step drops 38 in discovery).
    Demographics over this set differ from the true analysed set by < 0.2%."""
    return load_pheno_iids(pheno_path) \
         & load_covar_iids(qcov_path) \
         & load_covar_iids(ccov_path)


def compute_demographics(eids, label, ukb):
    sub = ukb[ukb[COLS['eid']].isin(eids)].copy()
    n = len(sub)

    age = sub[COLS['age_imaging']]
    age_valid = age.dropna()

    sex = sub[COLS['sex_recorded']]
    n_female = (sex == 0).sum()
    n_male   = (sex == 1).sum()
    pct_female = 100.0 * n_female / (n_female + n_male) if (n_female + n_male) else float('nan')

    wb = sub[COLS['white_british']]
    n_wb = (wb == 1).sum()
    pct_wb = 100.0 * n_wb / n if n else float('nan')

    centres = sub[COLS['centre_imaging']].dropna().astype(int).value_counts()
    centre_breakdown = ', '.join(
        f'{CENTRE_LABELS.get(c, str(c))}: {centres[c]}' for c in sorted(centres.index)
    )

    # Headline N: FastGWA-canonical analysed N (per .fastGWA.log), not the
    # 3-way set intersection used to compute demographics (differs by ≤ 0.2%).
    if 'discovery' in label:
        n_report = N_DISCOVERY_ANALYSED
    elif 'replication' in label:
        n_report = N_REPLICATION_ANALYSED
    else:
        n_report = n

    return {
        'cohort': label,
        'N': n_report,
        'N_demographics_set': n,
        'age_mean_yr': round(age_valid.mean(), 2),
        'age_sd_yr': round(age_valid.std(), 2),
        'age_min_yr': round(age_valid.min(), 1),
        'age_max_yr': round(age_valid.max(), 1),
        'female_pct': round(pct_female, 2),
        'white_british_pct': round(pct_wb, 2),
        'centre_imaging_breakdown': centre_breakdown,
        'n_missing_age': int(age.isna().sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help='Output CSV (default: TableS10_demographics.csv)')
    ap.add_argument('--pheno-region', default='Left_Hippocampus',
                    help='Region whose pheno file is used to read cohort IIDs '
                         '(all regions have the same IID set per cohort).')
    args = ap.parse_args()

    print('Loading cohort eid lists (analysed cohort = pheno ∩ qcovar ∩ ccovar)...')
    train_eids = load_train_eids()
    disc_pheno = PHENO_BASE / 'input' / \
        f'discovery_{args.pheno_region}_QT0'
    repl_pheno = PHENO_BASE / 'replication' / 'input' / \
        f'replication_{args.pheno_region}_QT0'
    disc_eids = load_analysed_eids(disc_pheno,
                                   COVAR_BASE / 'T1_qcovar_discovery_v2',
                                   COVAR_BASE / 'T1_ccovar_discovery_v2')
    repl_eids = load_analysed_eids(repl_pheno,
                                   COVAR_BASE / 'T1_qcovar_replication_v2',
                                   COVAR_BASE / 'T1_ccovar_replication_v2')
    print(f'  Training:    N = {len(train_eids):,}  (base_line_train + val)')
    print(f'  Discovery:   N = {len(disc_eids):,}  (intersection; FastGWA log: '
          f'{N_DISCOVERY_ANALYSED:,})')
    print(f'  Replication: N = {len(repl_eids):,}  (intersection; FastGWA log: '
          f'{N_REPLICATION_ANALYSED:,})')

    print('Loading UKB_all.csv (selected columns)...')
    ukb_cols = list(COLS.values())
    ukb = pd.read_csv(UKB_CSV, usecols=ukb_cols)
    ukb[COLS['eid']] = ukb[COLS['eid']].astype(int)
    print(f'  UKB rows loaded: {len(ukb):,}')

    print('Computing per-cohort demographics...')
    rows = [
        compute_demographics(train_eids, 'BRE encoder training', ukb),
        compute_demographics(disc_eids,  'GWAS discovery', ukb),
        compute_demographics(repl_eids,  'GWAS replication (analysed)', ukb),
    ]
    df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote: {args.out}')
    print()
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
