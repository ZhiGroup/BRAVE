#!/usr/bin/env python3
"""
20_bre_pgs.py  —  BRE Polygenic Score validation

For each of 16 JAGWAS regions:
  1. Load IndSigSNPs from FUMA (all independent significant SNPs for that region).
  2. For each IndSigSNP, scan all 128 per-dim discovery FastGWA files and find the
     dimension with the most significant p-value (min-p selection). Use that dim's
     BETA as the PGS weight for the SNP.
  3. Build a PLINK2 score file (rsID, effect allele, BETA) and score replication
     subjects using the BGEN genotype data.
  4. Correlate per-subject PGS with each of the 128 replication BRE dim phenotypes
     → R² for each dim.
  5. Build a 16×16 cross-region R² matrix (rows = PGS region, cols = phenotype
     region) using the maximum R² over all 128 dims per pair.

Outputs:
  results/pgs/<region>_best_betas.csv       — best-dim BETA per IndSigSNP
  results/pgs/<region>_r2_dims.csv          — R² per dim (PGS vs BRE dim)
  results/pgs/cross_region_r2.csv           — 16×16 max-R² matrix
  results/pgs/pgs_summary.csv               — per-region n_snps / max_r2 / top_dim
  figures/pgs/cross_region_r2_heatmap.pdf   — publication heatmap figure

Note: grep scans 128 × 686 MB files per region. Expected runtime: ~15–30 min/region.
Use --regions to process a single region or subset.
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Region definitions ────────────────────────────────────────────────────────
REGIONS = [
    'Brain_Stem_or_4th_Ventricle',
    'Left_Accumbens-area',
    'Left_Amygdala',
    'Left_Caudate',
    'Left_Hippocampus',
    'Left_Pallidum',
    'Left_Putamen',
    'Left_Thalamus_Proper',
    'Right_Accumbens-area',
    'Right_Amygdala',
    'Right_Caudate',
    'Right_Hippocampus',
    'Right_Pallidum',
    'Right_Putamen',
    'Right_Thalamus-Proper',
    'CSF',
]

DISPLAY = {
    'Brain_Stem_or_4th_Ventricle': 'BrainStem',
    'Left_Accumbens-area':         'L.Accumbens',
    'Left_Amygdala':               'L.Amygdala',
    'Left_Caudate':                'L.Caudate',
    'Left_Hippocampus':            'L.Hippocampus',
    'Left_Pallidum':               'L.Pallidum',
    'Left_Putamen':                'L.Putamen',
    'Left_Thalamus_Proper':        'L.Thalamus',
    'Right_Accumbens-area':        'R.Accumbens',
    'Right_Amygdala':              'R.Amygdala',
    'Right_Caudate':               'R.Caudate',
    'Right_Hippocampus':           'R.Hippocampus',
    'Right_Pallidum':              'R.Pallidum',
    'Right_Putamen':               'R.Putamen',
    'Right_Thalamus-Proper':       'R.Thalamus',
    'CSF':                         'CSF',
}

N_DIMS = 128

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    BASE = str(Path(cfg.gwas.fastgwa_out).parent)

    p.add_argument('--fuma-dir', default=str(cfg.postgwas.fuma_dir),
                   help='Directory containing one folder per region with FUMA outputs.')
    p.add_argument('--fastgwa-disc-dir', default=f'{BASE}/output',
                   help='Discovery FastGWA output directory (one .fastGWA per dim/region).')
    p.add_argument('--fastgwa-rep-dir', default=f'{BASE}/replication/output',
                   help='Replication FastGWA output directory (used only to check file existence).')
    p.add_argument('--rep-pheno-dir', default=f'{BASE}/replication/input',
                   help='Replication phenotype files (one file per dim/region).')
    p.add_argument('--bgen', default=str(cfg.ukb.bgen),
                   help='Genome-wide BGEN file (with .bgi index alongside).')
    p.add_argument('--sample', default=str(cfg.ukb.bgen_sample),
                   help='Oxford .sample file for the BGEN.')
    p.add_argument('--plink2', default=str(cfg.tools.plink2),
                   help='Path to plink2 executable.')
    p.add_argument('--out-dir', default='results/pgs',
                   help='Output results directory (relative to CWD or absolute).')
    p.add_argument('--fig-dir', default='figures/pgs',
                   help='Output figures directory.')
    p.add_argument('--regions', nargs='+', default=None,
                   help='Process only these regions (default: all 16).')
    p.add_argument('--score-method', choices=['min_p', 'top_h2'], default='min_p',
                   help='How to select the effect dim per SNP: '
                        'min_p = dimension with smallest p in discovery FastGWA (default); '
                        'top_h2 = use only the top-h² dim for the region (requires --h2-csv).')
    p.add_argument('--h2-csv', default='results/h2/fastgwa_h2_top_dims.csv',
                   help='h² CSV from Script 14 (required when --score-method=top_h2).')
    p.add_argument('--max-r2-dim', action='store_true',
                   help='Report max R² across all 128 dims for cross-region matrix (default).')
    p.add_argument('--pgen-prefix', default=None,
                   help='Path prefix for pre-built PLINK2 binary files (.pgen/.pvar/.psam). '
                        'If provided and exists, skip BGEN scanning and use the pre-built '
                        'genotype subset directly. If not provided, the script creates '
                        'results/pgs/pgen/all_indsigsnps in the first run. '
                        'Building the pgen scans the full BGEN once; subsequent region '
                        'runs use the small pgen for fast scoring.')
    p.add_argument('--skip-plink2', action='store_true',
                   help='Skip PLINK2 scoring step (assume .sscore files already exist).')
    return p.parse_args()


# ── Step 1: Load IndSigSNPs ───────────────────────────────────────────────────
def load_indsigsnps(fuma_dir: Path, region: str) -> List[str]:
    """Return list of rsIDs from FUMA IndSigSNPs.txt for a region."""
    f = fuma_dir / region / 'IndSigSNPs.txt'
    if not f.exists():
        print(f'  WARNING: {f} not found, skipping region.')
        return []
    df = pd.read_csv(f, sep='\t', usecols=['rsID'])
    rsids = df['rsID'].dropna().tolist()
    print(f'  {region}: {len(rsids)} IndSigSNPs loaded.')
    return rsids


# ── Step 2: Grep best-dim BETA ────────────────────────────────────────────────
def find_best_dim_betas(
    region: str,
    rsids: List[str],
    fastgwa_disc_dir: Path,
    n_dims: int = N_DIMS,
) -> pd.DataFrame:
    """
    For each rsID, scan 128 per-dim discovery FastGWA files and find the dimension
    with the minimum p-value. Return a DataFrame with columns:
      rsID, best_dim, best_p, best_beta, best_a1

    Uses grep -F -f for fast batch extraction (one grep call per dim file).
    """
    # Track best result per SNP
    best = {rsid: {'best_p': 2.0, 'best_dim': -1,
                   'best_beta': np.nan, 'best_a1': ''}
            for rsid in rsids}
    rsid_set = set(rsids)

    # Write SNP list to a temp file for grep
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_snps.txt', delete=False)
    for rsid in rsids:
        tmp.write(rsid + '\n')
    tmp.flush()
    snp_list_path = tmp.name
    tmp.close()

    found_dims = []
    missing_dims = []

    for dim in range(n_dims):
        fname = fastgwa_disc_dir / (
            f'discovery_{region}_QT{dim}.fastGWA.fastGWA'
        )
        if not fname.exists():
            missing_dims.append(dim)
            continue

        # grep: -w = whole-word match (prevents substring false positives),
        #       -F = fixed strings (no regex overhead), -f = pattern file
        result = subprocess.run(
            ['grep', '-wF', '-f', snp_list_path, str(fname)],
            capture_output=True, text=True
        )
        if result.returncode not in (0, 1):
            # returncode 1 = no match (fine); anything else is an error
            print(f'    grep error dim={dim}: {result.stderr[:200]}')
            continue

        found_dims.append(dim)
        for line in result.stdout.splitlines():
            parts = line.split('\t')
            if len(parts) < 10:
                continue
            snp_id = parts[1]
            if snp_id not in rsid_set:
                continue
            try:
                p_val = float(parts[9])
                beta  = float(parts[7])
                a1    = parts[3]
            except (ValueError, IndexError):
                continue
            if p_val < best[snp_id]['best_p']:
                best[snp_id]['best_p']    = p_val
                best[snp_id]['best_dim']  = dim
                best[snp_id]['best_beta'] = beta
                best[snp_id]['best_a1']   = a1

    Path(snp_list_path).unlink()

    if missing_dims:
        print(f'    dims absent from discovery FastGWA: {len(missing_dims)} '
              f'(first few: {missing_dims[:5]})')

    # Build output DataFrame; drop SNPs never found in any dim
    rows = []
    not_found = 0
    for rsid, v in best.items():
        if v['best_dim'] < 0:
            not_found += 1
        else:
            rows.append({
                'rsID':      rsid,
                'best_dim':  v['best_dim'],
                'best_p':    v['best_p'],
                'best_beta': v['best_beta'],
                'best_a1':   v['best_a1'],
            })

    if not_found:
        print(f'    {not_found}/{len(rsids)} SNPs not found in any dim file '
              f'(may be indels or excluded variants).')
    print(f'    {len(rows)} SNPs with valid BETA selected from {len(found_dims)} dim files.')

    return pd.DataFrame(rows)


# ── Step 2b: Pre-build PLINK2 genotype subset ─────────────────────────────────
def make_genotype_subset(
    all_rsids:   List[str],
    rep_pheno_dir: Path,
    bgen:        Path,
    sample:      Path,
    plink2:      str,
    pgen_prefix: Path,
    regions:     List[str],
) -> bool:
    """
    One-time step: extract all IndSigSNPs and replication subjects from the BGEN
    and save as PLINK2 binary (.pgen/.pvar/.psam).

    This scans the full 81 GB BGEN once; subsequent per-region scoring uses
    the small .pgen for fast random-access scoring.

    Returns True on success.
    """
    pgen_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Already exists?
    if (Path(str(pgen_prefix) + '.pgen').exists() and
            Path(str(pgen_prefix) + '.pvar').exists() and
            Path(str(pgen_prefix) + '.psam').exists()):
        print(f'  Pre-built pgen found at {pgen_prefix}.*  — skipping BGEN scan.')
        return True

    # --- a. SNP extract file — must use BGEN SNPID format (chr:pos_A1_A2),
    #         NOT rsIDs, because PLINK2 --extract matches on the SNPID field only.
    #         Map rsID → SNPID via the .bgi SQLite index.
    snp_file = pgen_prefix.parent / 'all_indsigsnps.txt'
    unique_rsids = sorted(set(all_rsids))

    # Query .bgi: chromosome, position, allele1, allele2 for each rsID
    bgi_path = Path(str(bgen) + '.bgi')
    if not bgi_path.exists():
        print(f'  WARNING: .bgi not found at {bgi_path}; falling back to rsID extract.')
        bgi_snpids = {r: r for r in unique_rsids}  # identity fallback
    else:
        conn = sqlite3.connect(str(bgi_path))
        cur  = conn.cursor()
        placeholders = ','.join(['?'] * len(unique_rsids))
        cur.execute(
            f'SELECT rsid, chromosome, position, allele1, allele2 FROM Variant '
            f'WHERE rsid IN ({placeholders})',
            unique_rsids
        )
        bgi_snpids = {}
        for rsid, chrom, pos, a1, a2 in cur.fetchall():
            # BGEN SNPID format: "01:665266_T_C" (zero-padded chr for chr1-9)
            bgi_snpids[rsid] = f'{chrom}:{pos}_{a1}_{a2}'
        conn.close()

        n_found = len(bgi_snpids)
        n_miss  = len(unique_rsids) - n_found
        print(f'  BGI lookup: {n_found}/{len(unique_rsids)} rsIDs mapped to BGEN SNPIDs'
              f'{f"; {n_miss} missing (indels/excluded)" if n_miss else ""}.')

    with open(str(snp_file), 'w') as f:
        for rsid in unique_rsids:
            snpid = bgi_snpids.get(rsid, rsid)
            f.write(snpid + '\n')
    print(f'  Writing {len(unique_rsids)} BGEN SNPIDs to {snp_file}')

    # --- b. Subject keep file: union across all regions' replication subjects
    all_subjects = set()
    for region in regions:
        for dim in range(N_DIMS):
            pf = rep_pheno_dir / (
                f'replication_{region}_QT{dim}'
            )
            if pf.exists():
                tmp = pd.read_csv(str(pf), sep='\t', header=0, usecols=[0, 1])
                tmp.columns = ['FID', 'IID']
                for _, row in tmp.iterrows():
                    all_subjects.add((str(row['FID']), str(row['IID'])))
                break  # one dim file per region is enough

    keep_file = pgen_prefix.parent / 'replication_subjects.txt'
    with open(str(keep_file), 'w') as f:
        for fid, iid in sorted(all_subjects):
            f.write(f'{fid}\t{iid}\n')
    print(f'  Writing {len(all_subjects)} replication subjects to {keep_file}')

    # --- c. Run PLINK2 --make-pgen (scans full BGEN once)
    cmd = [
        plink2,
        '--bgen', str(bgen), 'ref-last',
        '--sample', str(sample),
        '--keep', str(keep_file),
        '--extract', str(snp_file),
        '--make-pgen',
        '--out', str(pgen_prefix),
    ]
    print(f'  Running PLINK2 --make-pgen (one-time BGEN scan, may take 15–30 min)...')
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'  PLINK2 --make-pgen error:\n{res.stderr[-800:]}')
        return False

    for line in res.stdout.splitlines():
        if any(kw in line for kw in ['variant', 'sample', 'Error', 'Warning', 'written']):
            print(f'    {line.strip()}')

    ok = (Path(str(pgen_prefix) + '.pgen').exists())
    if ok:
        print(f'  Genotype subset created: {pgen_prefix}.pgen')
    return ok


# ── Step 3: PLINK2 scoring ────────────────────────────────────────────────────
def run_plink2_score(
    beta_df:     pd.DataFrame,
    region:      str,
    rep_pheno_dir: Path,
    bgen:        Path,
    sample:      Path,
    plink2:      str,
    out_dir:     Path,
    pgen_prefix: Optional[Path] = None,
    skip:        bool = False,
) -> Optional[pd.DataFrame]:
    """
    Score replication subjects with PLINK2 using the best-dim BETAs.

    If pgen_prefix is provided (and .pgen/.pvar/.psam exist), use the pre-built
    genotype subset for fast scoring. Otherwise fall back to scanning the full BGEN.

    Replication subject IDs are taken from any available replication phenotype file
    for this region (fiid/iid columns).

    Returns DataFrame with columns: IID, PGS  (or None on failure).
    """
    out_prefix = out_dir / f'{region}_pgs'
    sscore_file = out_prefix.with_suffix('.sscore')

    if not (skip or sscore_file.exists()):
        # --- a. Write score file (no header): rsID  A1  BETA
        score_path = out_dir / f'{region}_score.txt'
        beta_df[['rsID', 'best_a1', 'best_beta']].to_csv(
            str(score_path), sep='\t', index=False, header=False
        )

        # --- b. Get replication IIDs from any phenotype file for this region
        rep_iid_df = None
        for dim in range(N_DIMS):
            pf = rep_pheno_dir / (
                f'replication_{region}_QT{dim}'
            )
            if pf.exists():
                tmp_df = pd.read_csv(pf, sep='\t', header=0, usecols=[0, 1])
                tmp_df.columns = ['FID', 'IID']
                rep_iid_df = tmp_df
                break
        if rep_iid_df is None:
            print(f'  ERROR: no replication phenotype file found for {region}.')
            return None

        keep_path = out_dir / f'{region}_keep.txt'
        rep_iid_df[['FID', 'IID']].to_csv(str(keep_path), sep='\t', index=False, header=False)

        # --- c. Run PLINK2 --score
        # Prefer pre-built pgen (fast); fall back to full BGEN scan
        pgen_ok = (pgen_prefix is not None and
                   Path(str(pgen_prefix) + '.pgen').exists())
        if pgen_ok:
            geno_args = [
                '--pfile', str(pgen_prefix),
            ]
            geno_label = f'pgen:{pgen_prefix.name}'
        else:
            geno_args = [
                '--bgen', str(bgen), 'ref-last',
                '--sample', str(sample),
            ]
            geno_label = 'bgen (full scan)'

        cmd = [
            plink2,
            *geno_args,
            '--keep', str(keep_path),
            '--score', str(score_path), '1', '2', '3',
            '--rm-dup', 'force-first',
            '--out', str(out_prefix),
        ]
        print(f'    Running PLINK2 --score [{geno_label}]...')
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f'    PLINK2 error:\n{res.stderr[-800:]}')
            return None
        if res.stdout:
            # Print a brief summary from PLINK2 log
            for line in res.stdout.splitlines():
                if any(kw in line for kw in ['variant', 'sample', 'Error', 'Warning']):
                    print(f'      {line.strip()}')

    if not sscore_file.exists():
        print(f'  ERROR: {sscore_file} not created by PLINK2.')
        return None

    # --- d. Load .sscore → IID + SCORE1_AVG
    scores = pd.read_csv(str(sscore_file), sep='\t')
    # PLINK2 column: #FID or FID, IID, ALLELE_CT, NAMED_ALLELE_DOSAGE_SUM, SCORE1_AVG
    scores.columns = [c.lstrip('#') for c in scores.columns]
    if 'SCORE1_AVG' not in scores.columns:
        print(f'  ERROR: SCORE1_AVG not in {sscore_file}. Columns: {scores.columns.tolist()}')
        return None

    out_df = scores[['IID', 'SCORE1_AVG']].copy()
    out_df.columns = ['IID', 'PGS']
    out_df['IID'] = out_df['IID'].astype(str)
    print(f'    PGS computed for {len(out_df)} replication subjects.')
    return out_df


# ── Step 4: Correlate PGS with replication BRE dims ──────────────────────────
def correlate_pgs_with_dims(
    pgs_df:       pd.DataFrame,
    region:       str,
    rep_pheno_dir: Path,
    n_dims:       int = N_DIMS,
) -> pd.DataFrame:
    """
    Correlate PGS scores with each of the 128 replication BRE dim phenotypes.
    Returns DataFrame: dim, r, r2, p, n
    """
    rows = []
    pgs = pgs_df.set_index('IID')['PGS']

    for dim in range(n_dims):
        pf = rep_pheno_dir / (
            f'replication_{region}_QT{dim}'
        )
        if not pf.exists():
            continue
        pheno = pd.read_csv(str(pf), sep='\t', header=0)
        pheno.columns = ['FID', 'IID', f'QT{dim}']
        pheno['IID'] = pheno['IID'].astype(str)
        pheno = pheno.set_index('IID')[f'QT{dim}']

        # Intersect on common IIDs
        common = pgs.index.intersection(pheno.index)
        if len(common) < 50:
            continue
        x = pgs.loc[common].values
        y = pheno.loc[common].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        x, y = x[mask], y[mask]

        r, p_val = pearsonr(x, y)
        rows.append({'dim': dim, 'r': r, 'r2': r ** 2, 'p': p_val, 'n': len(common)})

    df = pd.DataFrame(rows)
    if len(df):
        best_idx = df['r2'].idxmax()
        print(f'    Max R²={df.loc[best_idx, "r2"]:.4f} at dim {int(df.loc[best_idx, "dim"])} '
              f'(r={df.loc[best_idx, "r"]:.3f}, p={df.loc[best_idx, "p"]:.2e})')
    return df


# ── Cross-region: load all replication BRE phenotypes ────────────────────────
def load_all_rep_phenos(rep_pheno_dir: Path,
                        regions: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load replication BRE phenotypes for all regions.
    Returns {region: DataFrame with IID + QT0..QT127} or None if missing.
    """
    all_phenos = {}
    for region in regions:
        dfs = []
        for dim in range(N_DIMS):
            pf = rep_pheno_dir / (
                f'replication_{region}_QT{dim}'
            )
            if not pf.exists():
                continue
            tmp = pd.read_csv(str(pf), sep='\t', header=0)
            tmp.columns = ['FID', 'IID', f'QT{dim}']
            tmp['IID'] = tmp['IID'].astype(str)
            tmp = tmp[['IID', f'QT{dim}']]
            dfs.append(tmp)
        if not dfs:
            all_phenos[region] = None
            continue
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on='IID', how='inner')
        all_phenos[region] = merged.set_index('IID')
        print(f'  Loaded {len(dfs)} dims, {len(merged)} subjects for {region}.')
    return all_phenos


def compute_cross_region_r2(
    pgs_scores: Dict[str, pd.DataFrame],   # {region → DataFrame(IID, PGS)}
    all_phenos: Dict[str, Optional[pd.DataFrame]],  # {region → wide DataFrame}
    regions:    List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a 16×16 matrix of max R² over 128 dims.
    Row = PGS region, Col = phenotype region.
    Also returns a per-pair long-form DataFrame.
    """
    matrix = pd.DataFrame(np.nan, index=regions, columns=regions)
    records = []

    for pgs_region in regions:
        if pgs_region not in pgs_scores or pgs_scores[pgs_region] is None:
            continue
        pgs = pgs_scores[pgs_region].set_index('IID')['PGS']

        for pheno_region in regions:
            pheno_wide = all_phenos.get(pheno_region)
            if pheno_wide is None:
                continue

            common = pgs.index.intersection(pheno_wide.index)
            if len(common) < 50:
                continue
            x = pgs.loc[common].values

            best_r2 = -1.0
            best_dim = -1
            for col in pheno_wide.columns:
                y = pheno_wide.loc[common, col].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 50:
                    continue
                r, _ = pearsonr(x[mask], y[mask])
                if r ** 2 > best_r2:
                    best_r2 = r ** 2
                    best_dim = int(col.replace('QT', ''))

            matrix.loc[pgs_region, pheno_region] = best_r2
            records.append({
                'pgs_region':   pgs_region,
                'pheno_region': pheno_region,
                'max_r2':       best_r2,
                'best_dim':     best_dim,
                'n':            len(common),
            })
            same = '★' if pgs_region == pheno_region else ''
            print(f'    {DISPLAY[pgs_region]:>15} → {DISPLAY[pheno_region]:<15} '
                  f'maxR²={best_r2:.4f} dim={best_dim} {same}')

    return matrix, pd.DataFrame(records)


# ── Figure: 16×16 heatmap ─────────────────────────────────────────────────────
def plot_cross_region_heatmap(matrix: pd.DataFrame,
                              regions: List[str],
                              fig_dir: Path):
    apply_mpl_style()

    labels = [DISPLAY[r] for r in regions]
    data   = matrix.loc[regions, regions].values.astype(float)

    fig, ax = plt.subplots(figsize=(MM(180), MM(165)))

    cmap = plt.cm.YlOrRd
    im   = ax.imshow(data, cmap=cmap, vmin=0, aspect='equal')

    # Colour bar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Max $R^2$ (PGS × BRE dim)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Axis labels
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(regions)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Phenotype region (BRE dims)', fontsize=8)
    ax.set_ylabel('PGS region (JAGWAS IndSigSNPs)', fontsize=8)
    ax.set_title('BRE-PGS region specificity: max R² across 128 dims', fontsize=9)

    # Annotate cells
    for i in range(len(regions)):
        for j in range(len(regions)):
            val = data[i, j]
            if not np.isnan(val):
                weight = 'bold' if i == j else 'normal'
                color  = 'white' if val > 0.6 * np.nanmax(data) else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=5, color=color, fontweight=weight)

    # Highlight diagonal
    for k in range(len(regions)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor='navy', lw=1.2))

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, fig_dir / 'cross_region_r2_heatmap')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    fuma_dir        = Path(args.fuma_dir)
    disc_dir        = Path(args.fastgwa_disc_dir)
    rep_pheno_dir   = Path(args.rep_pheno_dir)
    bgen            = Path(args.bgen)
    sample          = Path(args.sample)
    plink2          = args.plink2
    out_dir         = Path(args.out_dir)
    fig_dir         = Path(args.fig_dir)
    regions         = args.regions if args.regions else REGIONS
    skip_plink2     = args.skip_plink2

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Optional: load top-h² dims for alternative scoring
    top_h2_dim = {}
    if args.score_method == 'top_h2':
        h2_path = Path(args.h2_csv)
        if not h2_path.exists():
            print(f'ERROR: --h2-csv {h2_path} not found. Use --score-method min_p instead.')
            sys.exit(1)
        h2_df = pd.read_csv(h2_path)
        # Keep top-1 dim per region (highest h2)
        for region, grp in h2_df.groupby('region'):
            top_h2_dim[region] = int(grp.sort_values('h2', ascending=False).iloc[0]['dim'])

    # ─── Pgen pre-build (optional, skip if --pgen-prefix not provided) ────────
    # NOTE: PLINK2 --extract with BGEN files matches against the pvar ID column,
    # which PLINK2 internally builds as chr:pos:REF:ALT — NOT matching the BGEN
    # SNPID field nor rsIDs. Rather than risk filtering failures, we skip pgen
    # pre-building by default and score each region directly from the BGEN
    # (which works correctly because --score also checks the RSID field).
    # If --pgen-prefix is provided and the files exist, we will use them.
    pgen_prefix = None
    if args.pgen_prefix:
        p_path = Path(args.pgen_prefix)
        if (Path(str(p_path) + '.pgen').exists() and
                Path(str(p_path) + '.pvar').exists() and
                Path(str(p_path) + '.psam').exists()):
            pgen_prefix = p_path
            print(f'  Using pre-built pgen: {p_path}')
        else:
            print(f'  --pgen-prefix specified but files not found; falling back to BGEN scoring.')

    if pgen_prefix is None and not skip_plink2:
        print('  Scoring directly from BGEN (per-region; ~15–20 min each).')
        print(f'  Total regions: {len(regions)}; estimated runtime: '
              f'{len(regions) * 20 // 60}h {len(regions) * 20 % 60}min')

    # ─── Per-region pipeline ─────────────────────────────────────────────────
    pgs_scores = {}   # {region: DataFrame(IID, PGS)}
    summary    = []

    for region in regions:
        print(f'\n══ {region} ══')

        # 1. Load IndSigSNPs
        rsids = load_indsigsnps(fuma_dir, region)
        if not rsids:
            continue

        # 2. Find best-dim BETA per SNP
        beta_csv = out_dir / f'{region}_best_betas.csv'
        if beta_csv.exists():
            print(f'  Loading cached best-betas from {beta_csv}')
            beta_df = pd.read_csv(beta_csv)
        else:
            if args.score_method == 'top_h2' and region in top_h2_dim:
                # Score using only the top h² dim
                t_dim = top_h2_dim[region]
                print(f'  top_h2 mode: using dim {t_dim} for all SNPs.')
                fname = disc_dir / (
                    f'discovery_{region}_QT{t_dim}.fastGWA.fastGWA'
                )
                if not fname.exists():
                    print(f'  WARNING: top-h2 dim file missing ({fname}), skipping.')
                    continue
                tmp_snp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                for r in rsids:
                    tmp_snp.write(r + '\n')
                tmp_snp.flush(); tmp_snp.close()
                res = subprocess.run(
                    ['grep', '-wF', '-f', tmp_snp.name, str(fname)],
                    capture_output=True, text=True
                )
                Path(tmp_snp.name).unlink()
                rows = []
                for line in res.stdout.splitlines():
                    parts = line.split('\t')
                    if len(parts) < 10:
                        continue
                    try:
                        rows.append({'rsID': parts[1], 'best_dim': t_dim,
                                     'best_p': float(parts[9]),
                                     'best_beta': float(parts[7]),
                                     'best_a1': parts[3]})
                    except ValueError:
                        continue
                beta_df = pd.DataFrame(rows)
            else:
                beta_df = find_best_dim_betas(region, rsids, disc_dir)

            beta_df.to_csv(str(beta_csv), index=False)
            print(f'  Saved best-betas → {beta_csv}')

        if beta_df.empty:
            print(f'  No valid betas for {region}, skipping PLINK2.')
            continue

        # 3. PLINK2 scoring
        pgs_df = run_plink2_score(
            beta_df, region, rep_pheno_dir, bgen, sample, plink2, out_dir,
            pgen_prefix=pgen_prefix,
            skip=skip_plink2
        )
        if pgs_df is None:
            continue
        pgs_scores[region] = pgs_df

        # 4. Correlate PGS with own-region BRE dims (within-region validation)
        print(f'  Correlating PGS with {region} replication BRE dims...')
        r2_df = correlate_pgs_with_dims(pgs_df, region, rep_pheno_dir)
        r2_csv = out_dir / f'{region}_r2_dims.csv'
        r2_df.to_csv(str(r2_csv), index=False)

        if len(r2_df):
            best_row = r2_df.loc[r2_df['r2'].idxmax()]
            summary.append({
                'region':      region,
                'display':     DISPLAY[region],
                'n_indsigsnps': len(rsids),
                'n_scored_snps': len(beta_df),
                'n_rep_subjects': len(pgs_df),
                'max_r2':      float(best_row['r2']),
                'best_dim':    int(best_row['dim']),
                'best_r':      float(best_row['r']),
                'best_p':      float(best_row['p']),
            })

    # Save per-region summary
    if summary:
        sum_df = pd.DataFrame(summary)
        sum_df.to_csv(str(out_dir / 'pgs_summary.csv'), index=False)
        print(f'\nPer-region summary → {out_dir}/pgs_summary.csv')
        print(sum_df[['display', 'n_indsigsnps', 'n_scored_snps', 'max_r2', 'best_dim']].to_string(index=False))

    # ─── Cross-region analysis ────────────────────────────────────────────────
    if len(pgs_scores) < 2:
        print('\nNot enough regions processed for cross-region analysis. Done.')
        return

    print('\n══ Cross-region R² matrix ══')
    print('Loading all replication BRE phenotypes (this may take a moment)...')
    all_phenos = load_all_rep_phenos(rep_pheno_dir, regions)

    matrix, long_df = compute_cross_region_r2(pgs_scores, all_phenos, regions)

    # Save
    matrix.index   = [DISPLAY[r] for r in matrix.index]
    matrix.columns = [DISPLAY[r] for r in matrix.columns]
    matrix.to_csv(str(out_dir / 'cross_region_r2.csv'))
    long_df.to_csv(str(out_dir / 'cross_region_r2_long.csv'), index=False)
    print(f'Cross-region matrix → {out_dir}/cross_region_r2.csv')

    # Diagonal vs off-diagonal summary
    proc_regions = [r for r in regions if r in pgs_scores]
    diag_vals  = [matrix.loc[DISPLAY[r], DISPLAY[r]] for r in proc_regions
                  if not np.isnan(matrix.loc[DISPLAY[r], DISPLAY[r]])]
    offdiag_vals = []
    for i, r1 in enumerate(proc_regions):
        for j, r2 in enumerate(proc_regions):
            if i != j:
                v = matrix.loc[DISPLAY[r1], DISPLAY[r2]]
                if not np.isnan(v):
                    offdiag_vals.append(v)

    if diag_vals and offdiag_vals:
        print(f'\nDiagonal (same region) mean R²:    {np.mean(diag_vals):.4f}')
        print(f'Off-diagonal (cross-region) mean R²: {np.mean(offdiag_vals):.4f}')
        print(f'Specificity ratio: {np.mean(diag_vals)/np.mean(offdiag_vals):.2f}×')

    # Figure
    # Rebuild matrix with original region labels for indexing
    mat_full = pd.DataFrame(np.nan, index=regions, columns=regions)
    for _, row in long_df.iterrows():
        mat_full.loc[row['pgs_region'], row['pheno_region']] = row['max_r2']
    plot_cross_region_heatmap(mat_full, regions, fig_dir)
    print(f'Heatmap → {fig_dir}/cross_region_r2_heatmap.pdf')


if __name__ == '__main__':
    main()
