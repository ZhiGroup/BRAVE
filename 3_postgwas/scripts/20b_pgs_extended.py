#!/usr/bin/env python3
"""
20b_pgs_extended.py  —  Extended BRE-PGS pilot (Pallidum / Putamen)

Tests whether including sub-threshold JAGWAS SNPs (beyond genome-wide
significant IndSigSNPs) improves diagonal enrichment in the cross-region R²
matrix for the anatomically-adjacent Pallidum and Putamen.

Pipeline (per region):
  1. Read JAGWAS discovery output → filter at --p-thresh (default 0.001)
  2. LD-clump filtered SNPs with PLINK2 (r²<0.1, 500 kb window)
  3. For each clumped SNP: find the FastGWA discovery dim with min p-value
     and use that dim's BETA as PGS weight  (same logic as Script 20)
  4. Score replication subjects with PLINK2 --score
  5. Correlate PGS with all 4 target regions' replication BRE dims
  6. Print 4×4 R² matrix + comparison with Script 20 (IndSigSNPs) results

Outputs:
  results/pgs_extended/<region>_clumped_snps.txt      — LD-clumped rsIDs
  results/pgs_extended/<region>_best_betas.csv        — min-p BETA per SNP
  results/pgs_extended/<region>_r2_dims.csv           — R² per dim
  results/pgs_extended/cross_region_r2_extended.csv   — 4×4 matrix
  figures/pgs_extended/cross_region_r2_heatmap.pdf    — comparison figure
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ─────────────────────────────────────────────────────────────────
N_DIMS = 128

PILOT_REGIONS = [
    'Left_Pallidum',
    'Left_Putamen',
    'Right_Pallidum',
    'Right_Putamen',
]

DISPLAY = {
    'Left_Pallidum':  'L.Pallidum',
    'Left_Putamen':   'L.Putamen',
    'Right_Pallidum': 'R.Pallidum',
    'Right_Putamen':  'R.Putamen',
}

# Script 20 (IndSigSNPs) diagonal and off-diagonal R² for comparison
# Row = PGS region, Col = phenotype region  (from pgs_run5 interim results)
SCRIPT20_R2 = {
    ('Left_Pallidum',  'Left_Pallidum'):  0.0052,
    ('Left_Pallidum',  'Left_Putamen'):   0.0054,
    ('Left_Pallidum',  'Right_Pallidum'): 0.0048,
    ('Left_Pallidum',  'Right_Putamen'):  0.0040,
    ('Left_Putamen',   'Left_Pallidum'):  0.0047,
    ('Left_Putamen',   'Left_Putamen'):   0.0042,
    ('Left_Putamen',   'Right_Pallidum'): 0.0035,
    ('Left_Putamen',   'Right_Putamen'):  0.0047,
    ('Right_Pallidum', 'Left_Pallidum'):  0.0068,
    ('Right_Pallidum', 'Left_Putamen'):   0.0038,
    ('Right_Pallidum', 'Right_Pallidum'): 0.0068,
    ('Right_Pallidum', 'Right_Putamen'):  0.0075,
    ('Right_Putamen',  'Left_Pallidum'):  0.0063,
    ('Right_Putamen',  'Left_Putamen'):   0.0045,
    ('Right_Putamen',  'Right_Pallidum'): 0.0044,
    ('Right_Putamen',  'Right_Putamen'):  0.0060,
}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    BASE = str(Path(cfg.gwas.fastgwa_out).parent)

    p.add_argument('--jagwas-dir',
                   default=str(cfg.gwas.jagwas_out),
                   help='JAGWAS discovery output directory (one subdir per region).')
    p.add_argument('--fastgwa-disc-dir', default=f'{BASE}/output',
                   help='Discovery FastGWA per-dim files.')
    p.add_argument('--rep-pheno-dir', default=f'{BASE}/replication/input',
                   help='Replication phenotype files.')
    p.add_argument('--bgen', default=str(cfg.ukb.bgen),
                   help='Genome-wide BGEN file.')
    p.add_argument('--sample', default=str(cfg.ukb.bgen_sample),
                   help='Oxford .sample file.')
    p.add_argument('--plink2', default=str(cfg.tools.plink2),
                   help='Path to plink2 executable.')
    p.add_argument('--p-thresh', type=float, default=0.001,
                   help='JAGWAS p-value threshold for candidate SNPs (default: 0.001).')
    p.add_argument('--clump-r2', type=float, default=0.1,
                   help='LD r² threshold for clumping (default: 0.1).')
    p.add_argument('--clump-kb', type=int, default=500,
                   help='Clumping window in kb (default: 500).')
    p.add_argument('--regions', nargs='+', default=PILOT_REGIONS,
                   help='Regions to process (default: Pallidum+Putamen x2).')
    p.add_argument('--out-dir', default='results/pgs_extended',
                   help='Output results directory.')
    p.add_argument('--fig-dir', default='figures/pgs_extended',
                   help='Output figures directory.')
    p.add_argument('--skip-clump', action='store_true',
                   help='Skip clumping if <region>_clumped_snps.txt already exists.')
    p.add_argument('--skip-plink2', action='store_true',
                   help='Skip scoring if .sscore already exists.')
    return p.parse_args()


# ── Step 1+2: Filter JAGWAS → LD clump ───────────────────────────────────────
def get_clumped_snps(
    region:      str,
    jagwas_dir:  Path,
    p_thresh:    float,
    clump_r2:    float,
    clump_kb:    int,
    bgen:        Path,
    sample:      Path,
    plink2:      str,
    out_dir:     Path,
    skip:        bool = False,
) -> List[str]:
    """
    Filter JAGWAS output at p_thresh, LD-clump with PLINK2, return rsID list.
    """
    clumped_file = out_dir / f'{region}_clumped_snps.txt'

    if skip and clumped_file.exists():
        snps = pd.read_csv(clumped_file, header=None, names=['rsID'])['rsID'].tolist()
        print(f'  Loaded {len(snps)} clumped SNPs from cache.')
        return snps

    # --- a. Read and filter JAGWAS output
    jagwas_file = jagwas_dir / region / f'{region}_JAGWAS_results.txt.gz'
    if not jagwas_file.exists():
        print(f'  ERROR: JAGWAS file not found: {jagwas_file}')
        return []

    print(f'  Reading JAGWAS output: {jagwas_file.name}')
    jagwas_df = pd.read_csv(str(jagwas_file), sep='\t', compression='gzip',
                            usecols=['SNP', 'P'])
    filtered = jagwas_df[jagwas_df['P'] < p_thresh].copy()
    print(f'  {len(filtered):,} SNPs with JAGWAS p < {p_thresh}')

    if len(filtered) == 0:
        return []

    # --- b. Write p-value file for PLINK2 --clump (needs ID and P columns)
    clump_input = out_dir / f'{region}_jagwas_pvals.txt'
    filtered.rename(columns={'SNP': 'ID'})[['ID', 'P']].to_csv(
        str(clump_input), sep='\t', index=False
    )

    # --- c. Run PLINK2 --clump (scans BGEN for LD)
    clump_prefix = out_dir / f'{region}_clump'
    cmd = [
        plink2,
        '--bgen', str(bgen), 'ref-last',
        '--sample', str(sample),
        '--rm-dup', 'force-first',
        '--clump', str(clump_input),
        '--clump-p1', str(p_thresh),
        '--clump-p2', str(p_thresh),
        '--clump-r2', str(clump_r2),
        '--clump-kb', str(clump_kb),
        '--out', str(clump_prefix),
    ]
    print(f'  Running PLINK2 --clump (r²<{clump_r2}, {clump_kb}kb, BGEN scan ~20 min)...')
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f'  PLINK2 --clump error:\n{res.stderr[-800:]}')
        return []

    # --- d. Read clumped output
    clumps_file = Path(str(clump_prefix) + '.clumps')
    if not clumps_file.exists():
        print(f'  ERROR: {clumps_file} not found after clumping.')
        print(f'  PLINK2 stdout: {res.stdout[-400:]}')
        return []

    clumps_df = pd.read_csv(str(clumps_file), sep='\t')
    # PLINK2 --clump output: #CHROM  POS  ID  P  TOTAL ...
    id_col = 'ID' if 'ID' in clumps_df.columns else clumps_df.columns[2]
    rsids = clumps_df[id_col].tolist()
    print(f'  After LD clumping: {len(rsids):,} independent SNPs '
          f'(from {len(filtered):,}; r²<{clump_r2}, {clump_kb}kb)')

    # Save to cache
    pd.Series(rsids).to_csv(str(clumped_file), index=False, header=False)
    return rsids


# ── Step 3: Find best-dim BETA per clumped SNP ───────────────────────────────
def find_best_dim_betas(
    region:          str,
    rsids:           List[str],
    fastgwa_disc_dir: Path,
    n_dims:          int = N_DIMS,
) -> pd.DataFrame:
    """
    For each rsID, scan 128 per-dim FastGWA files and find the dim with min p.
    Returns DataFrame: rsID, best_dim, best_p, best_beta, best_a1
    """
    best = {rsid: {'best_p': 2.0, 'best_dim': -1,
                   'best_beta': np.nan, 'best_a1': ''}
            for rsid in rsids}
    rsid_set = set(rsids)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='_snps.txt', delete=False)
    for rsid in rsids:
        tmp.write(rsid + '\n')
    tmp.flush()
    snp_list_path = tmp.name
    tmp.close()

    n_found_dims = 0
    for dim in range(n_dims):
        fname = fastgwa_disc_dir / (
            f'discovery_{region}_QT{dim}.fastGWA.fastGWA'
        )
        if not fname.exists():
            continue

        result = subprocess.run(
            ['grep', '-wF', '-f', snp_list_path, str(fname)],
            capture_output=True, text=True
        )
        if result.returncode not in (0, 1):
            continue

        n_found_dims += 1
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

    rows = []
    not_found = 0
    for rsid, v in best.items():
        if v['best_dim'] < 0:
            not_found += 1
        else:
            rows.append({'rsID': rsid, 'best_dim': v['best_dim'],
                         'best_p': v['best_p'], 'best_beta': v['best_beta'],
                         'best_a1': v['best_a1']})
    if not_found:
        print(f'    {not_found}/{len(rsids)} SNPs not found in any FastGWA dim file.')
    print(f'    {len(rows):,} SNPs with valid BETA from {n_found_dims} dim files.')
    return pd.DataFrame(rows)


# ── Step 4: PLINK2 scoring ────────────────────────────────────────────────────
def run_plink2_score(
    beta_df:       pd.DataFrame,
    region:        str,
    rep_pheno_dir: Path,
    bgen:          Path,
    sample:        Path,
    plink2:        str,
    out_dir:       Path,
    skip:          bool = False,
) -> Optional[pd.DataFrame]:
    """Score replication subjects; return DataFrame(IID, PGS) or None."""
    out_prefix  = out_dir / f'{region}_pgs'
    sscore_file = out_prefix.with_suffix('.sscore')

    if not (skip or sscore_file.exists()):
        score_path = out_dir / f'{region}_score.txt'
        beta_df[['rsID', 'best_a1', 'best_beta']].to_csv(
            str(score_path), sep='\t', index=False, header=False
        )

        # Get replication subject IDs
        keep_path = out_dir / f'{region}_keep.txt'
        for dim in range(N_DIMS):
            pf = rep_pheno_dir / (
                f'replication_{region}_QT{dim}'
            )
            if pf.exists():
                tmp_df = pd.read_csv(str(pf), sep='\t', header=0, usecols=[0, 1])
                tmp_df.columns = ['FID', 'IID']
                tmp_df[['FID', 'IID']].to_csv(str(keep_path), sep='\t',
                                               index=False, header=False)
                break

        cmd = [
            plink2,
            '--bgen', str(bgen), 'ref-last',
            '--sample', str(sample),
            '--keep', str(keep_path),
            '--score', str(score_path), '1', '2', '3',
            '--rm-dup', 'force-first',
            '--out', str(out_prefix),
        ]
        print(f'    Running PLINK2 --score (BGEN scan ~20 min)...')
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f'    PLINK2 error:\n{res.stderr[-800:]}')
            return None
        for line in res.stdout.splitlines():
            if any(kw in line for kw in ['variant', 'sample', '--rm-dup']):
                print(f'      {line.strip()}')

    if not sscore_file.exists():
        print(f'  ERROR: {sscore_file} not created.')
        return None

    scores = pd.read_csv(str(sscore_file), sep='\t')
    scores.columns = [c.lstrip('#') for c in scores.columns]
    out_df = scores[['IID', 'SCORE1_AVG']].copy()
    out_df.columns = ['IID', 'PGS']
    out_df['IID'] = out_df['IID'].astype(str)
    print(f'    PGS computed for {len(out_df):,} replication subjects.')
    return out_df


# ── Step 5: Cross-region R² ───────────────────────────────────────────────────
def compute_cross_r2(
    pgs_df:        pd.DataFrame,
    pgs_region:    str,
    target_regions: List[str],
    rep_pheno_dir: Path,
) -> Dict[str, float]:
    """
    Correlate one region's PGS with all target regions' BRE dims.
    Returns {pheno_region: max_r2}.
    """
    pgs = pgs_df.set_index('IID')['PGS']
    results = {}

    for pheno_region in target_regions:
        best_r2 = -1.0
        for dim in range(N_DIMS):
            pf = rep_pheno_dir / (
                f'replication_{pheno_region}_QT{dim}'
            )
            if not pf.exists():
                continue
            pheno = pd.read_csv(str(pf), sep='\t', header=0)
            pheno.columns = ['FID', 'IID', f'QT{dim}']
            pheno['IID'] = pheno['IID'].astype(str)
            pheno = pheno.set_index('IID')[f'QT{dim}']

            common = pgs.index.intersection(pheno.index)
            if len(common) < 50:
                continue
            x = pgs.loc[common].values
            y = pheno.loc[common].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 50:
                continue
            r, _ = pearsonr(x[mask], y[mask])
            if r ** 2 > best_r2:
                best_r2 = r ** 2

        results[pheno_region] = best_r2 if best_r2 >= 0 else np.nan
        diag = '★' if pgs_region == pheno_region else ''
        print(f'    {DISPLAY[pgs_region]:>12} → {DISPLAY[pheno_region]:<12} '
              f'maxR²={results[pheno_region]:.4f} {diag}')

    return results


# ── Figure: side-by-side 4×4 heatmaps ────────────────────────────────────────
def plot_comparison(
    matrix_orig: pd.DataFrame,
    matrix_ext:  pd.DataFrame,
    regions:     List[str],
    p_thresh:    float,
    fig_dir:     Path,
):
    apply_mpl_style()
    labels = [DISPLAY[r] for r in regions]
    data_orig = matrix_orig.loc[regions, regions].values.astype(float)
    data_ext  = matrix_ext.loc[regions, regions].values.astype(float)

    vmax = max(np.nanmax(data_orig), np.nanmax(data_ext))

    fig, axes = plt.subplots(1, 2, figsize=(MM(160), MM(80)))
    titles = [f'Script 20: IndSigSNPs\n(GWS, N=67–254 per region)',
              f'Script 20b: JAGWAS p<{p_thresh} + LD clump\n(N=? per region)']

    for ax, data, title in zip(axes, [data_orig, data_ext], titles):
        im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='equal')
        ax.set_xticks(range(len(regions)))
        ax.set_yticks(range(len(regions)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=7)
        for i in range(len(regions)):
            for j in range(len(regions)):
                val = data[i, j]
                if not np.isnan(val):
                    col = 'white' if val > 0.6 * vmax else 'black'
                    ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                            fontsize=5, color=col)
        for k in range(len(regions)):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                       fill=False, edgecolor='navy', lw=1.2))

    fig.colorbar(im, ax=axes, fraction=0.023, pad=0.04).set_label(
        'Max R² (PGS × BRE dim)', fontsize=7)
    fig.suptitle('BRE-PGS: IndSigSNPs vs JAGWAS sub-threshold (Pallidum + Putamen)',
                 fontsize=8)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, fig_dir / 'pgs_comparison_heatmap')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    jagwas_dir    = Path(args.jagwas_dir)
    disc_dir      = Path(args.fastgwa_disc_dir)
    rep_pheno_dir = Path(args.rep_pheno_dir)
    bgen          = Path(args.bgen)
    sample        = Path(args.sample)
    plink2        = args.plink2
    out_dir       = Path(args.out_dir)
    fig_dir       = Path(args.fig_dir)
    regions       = args.regions
    p_thresh      = args.p_thresh

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'Extended PGS pilot: {len(regions)} regions, JAGWAS p<{p_thresh}')
    print(f'Regions: {regions}')
    print(f'Pipeline: JAGWAS filter → LD clump (r²<{args.clump_r2}, '
          f'{args.clump_kb}kb) → min-p BETA → PLINK2 score\n')

    pgs_scores = {}

    for region in regions:
        print(f'\n══ {region} ══')

        # 1+2. Filter JAGWAS → LD clump
        clumped_file = out_dir / f'{region}_clumped_snps.txt'
        skip_clump = args.skip_clump and clumped_file.exists()
        rsids = get_clumped_snps(
            region, jagwas_dir, p_thresh, args.clump_r2, args.clump_kb,
            bgen, sample, plink2, out_dir, skip=skip_clump
        )
        if not rsids:
            continue

        # 3. Find best-dim BETA per clumped SNP
        beta_csv = out_dir / f'{region}_best_betas.csv'
        if beta_csv.exists():
            print(f'  Loading cached best-betas ({beta_csv.name})')
            beta_df = pd.read_csv(beta_csv)
        else:
            print(f'  Finding best-dim BETA for {len(rsids):,} clumped SNPs...')
            beta_df = find_best_dim_betas(region, rsids, disc_dir)
            beta_df.to_csv(str(beta_csv), index=False)
            print(f'  Saved → {beta_csv}')

        if beta_df.empty:
            continue

        # 4. PLINK2 scoring
        pgs_df = run_plink2_score(
            beta_df, region, rep_pheno_dir, bgen, sample, plink2, out_dir,
            skip=args.skip_plink2
        )
        if pgs_df is None:
            continue
        pgs_scores[region] = pgs_df

        # 5. Within-region R² (own region only, quick check)
        print(f'  Within-region correlation:')
        _ = compute_cross_r2(pgs_df, region, [region], rep_pheno_dir)

    # 6. Full 4×4 cross-region matrix
    if len(pgs_scores) < 2:
        print('\nNot enough regions completed. Done.')
        return

    print(f'\n══ 4×4 Cross-region R² matrix (extended PGS) ══')
    ext_matrix = pd.DataFrame(np.nan, index=regions, columns=regions)
    for pgs_region, pgs_df in pgs_scores.items():
        print(f'\n  PGS: {DISPLAY[pgs_region]}')
        row_r2 = compute_cross_r2(pgs_df, pgs_region, regions, rep_pheno_dir)
        for pheno_region, r2 in row_r2.items():
            ext_matrix.loc[pgs_region, pheno_region] = r2

    ext_matrix.to_csv(str(out_dir / 'cross_region_r2_extended.csv'))

    # Summary: diagonal vs off-diagonal
    diag_ext    = [ext_matrix.loc[r, r] for r in regions
                   if not np.isnan(ext_matrix.loc[r, r])]
    offdiag_ext = [ext_matrix.loc[r1, r2]
                   for r1 in regions for r2 in regions if r1 != r2
                   and not np.isnan(ext_matrix.loc[r1, r2])]

    # Original Script 20 matrix for these 4 regions
    orig_matrix = pd.DataFrame(np.nan, index=regions, columns=regions)
    for (pr, phr), r2 in SCRIPT20_R2.items():
        if pr in regions and phr in regions:
            orig_matrix.loc[pr, phr] = r2
    diag_orig    = [orig_matrix.loc[r, r] for r in regions
                    if not np.isnan(orig_matrix.loc[r, r])]
    offdiag_orig = [orig_matrix.loc[r1, r2]
                    for r1 in regions for r2 in regions if r1 != r2
                    and not np.isnan(orig_matrix.loc[r1, r2])]

    print(f'\n{"─"*55}')
    print(f'{"":30} {"Script20":>10} {"Extended":>10}')
    print(f'{"─"*55}')
    print(f'{"Mean diagonal R²":30} {np.mean(diag_orig):>10.4f} {np.mean(diag_ext):>10.4f}')
    print(f'{"Mean off-diagonal R²":30} {np.mean(offdiag_orig):>10.4f} {np.mean(offdiag_ext):>10.4f}')
    if np.mean(offdiag_orig) > 0:
        print(f'{"Specificity ratio (diag/offdiag)":30} '
              f'{np.mean(diag_orig)/np.mean(offdiag_orig):>10.2f}× '
              f'{np.mean(diag_ext)/np.mean(offdiag_ext):>10.2f}×')

    # Print cell-by-cell comparison
    print(f'\n{"Pair":<30} {"Script20":>10} {"Extended":>10} {"Delta":>10}')
    print('─' * 55)
    for r1 in regions:
        for r2 in regions:
            v0 = orig_matrix.loc[r1, r2]
            ve = ext_matrix.loc[r1, r2]
            tag = ' ★' if r1 == r2 else ''
            if not np.isnan(v0) and not np.isnan(ve):
                delta = ve - v0
                print(f'{DISPLAY[r1]+"/"+DISPLAY[r2]:<30} {v0:>10.4f} {ve:>10.4f} '
                      f'{delta:>+10.4f}{tag}')

    # Figure
    plot_comparison(orig_matrix, ext_matrix, regions, p_thresh, fig_dir)
    print(f'\nComparison heatmap → {fig_dir}/pgs_comparison_heatmap.pdf')

    # Save extended summary
    rows_out = []
    for r in regions:
        rows_out.append({
            'region': r, 'display': DISPLAY[r],
            'n_clumped_snps': len(pgs_scores.get(r, pd.DataFrame())),
            'diag_r2_orig': orig_matrix.loc[r, r],
            'diag_r2_ext':  ext_matrix.loc[r, r],
        })
    pd.DataFrame(rows_out).to_csv(
        str(out_dir / 'diagonal_comparison.csv'), index=False)
    print(f'Diagonal comparison → {out_dir}/diagonal_comparison.csv')


if __name__ == '__main__':
    main()
