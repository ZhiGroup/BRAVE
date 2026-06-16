#!/usr/bin/env python3
"""
fig_fastgwa_vs_jagwas_loci.py
------------------------------
Supplementary figure comparing per-region FUMA-defined loci counts for JAGWAS
vs. FastGWA univariate minP baseline across all 16 subcortical regions.

JAGWAS counts are derived from aggregate_loci.csv (per-region participation).
FastGWA counts are read from GenomicRiskLoci.txt in each FUMA folder.

Inputs:
  --al       results/cross_region/aggregate_loci.csv
  --fuma_dir <univariate FastGWA min-P FUMA output root>

Output:
  figures/supplementary/fastgwa_vs_jagwas_loci/fastgwa_vs_jagwas_loci.pdf/.png

Usage:
  python fig_fastgwa_vs_jagwas_loci.py
  python fig_fastgwa_vs_jagwas_loci.py --al ... --fuma_dir ... --out ...
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent

DEFAULT_AL       = ROOT_DIR / 'results' / 'cross_region' / 'aggregate_loci.csv'
DEFAULT_FUMA_DIR = Path('<EXTERNAL: univariate FastGWA min-P FUMA output root (one folder per region)>')
DEFAULT_OUT      = ROOT_DIR / 'figures' / 'supplementary' \
                   / 'fastgwa_vs_jagwas_loci' / 'fastgwa_vs_jagwas_loci'

sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

# Mapping: FastGWA FUMA folder suffix → canonical region label
FOLDER_TO_REGION = {
    'Brain_Stem_or_4th_Ventricle': 'Brain Stem / 4th V.',
    'CSF':                          'CSF',
    'Left_Accumbens-area':          'L. Accumbens',
    'Left_Amygdala':                'L. Amygdala',
    'Left_Caudate':                 'L. Caudate',
    'Left_Hippocampus':             'L. Hippocampus',
    'Left_Pallidum':                'L. Pallidum',
    'Left_Putamen':                 'L. Putamen',
    'Left_Thalamus_Proper':         'L. Thalamus',
    'Right_Accumbens-area':         'R. Accumbens',
    'Right_Amygdala':               'R. Amygdala',
    'Right_Caudate':                'R. Caudate',
    'Right_Hippocampus':            'R. Hippocampus',
    'Right_Pallidum':               'R. Pallidum',
    'Right_Putamen':                'R. Putamen',
    'Right_Thalamus':               'R. Thalamus',
}

PREFIX = 'discovery_'


def read_fastgwa_counts(fuma_dir: Path):
    """Return dict: canonical_region -> n_fuma_loci from GenomicRiskLoci.txt."""
    counts = {}
    for suffix, region in FOLDER_TO_REGION.items():
        folder = fuma_dir / (PREFIX + suffix)
        loci_file = folder / 'GenomicRiskLoci.txt'
        if not loci_file.exists():
            print(f"WARNING: missing {loci_file}", file=sys.stderr)
            counts[region] = 0
            continue
        df = pd.read_csv(loci_file, sep='\t')
        counts[region] = len(df)
    return counts


def read_jagwas_counts(al_csv: Path):
    """Return dict: canonical_region -> n_fuma_loci (per-region participation)."""
    df = pd.read_csv(al_csv)
    all_regs = []
    for r in df['regions']:
        all_regs.extend(r.split(';'))
    return dict(Counter(all_regs))


def make_figure(al_csv: Path, fuma_dir: Path, out_stem: Path):
    apply_mpl_style()

    jagwas = read_jagwas_counts(al_csv)
    fastgwa = read_fastgwa_counts(fuma_dir)

    regions = sorted(jagwas.keys(), key=lambda r: jagwas[r], reverse=True)

    j_vals = np.array([jagwas.get(r, 0) for r in regions], dtype=float)
    f_vals = np.array([fastgwa.get(r, 0) for r in regions], dtype=float)
    fold   = np.where(f_vals > 0, j_vals / f_vals, np.nan)

    # Hemisphere colour for JAGWAS bars
    def bar_color(region):
        if region.startswith('L.'):
            return '#4C72B0'   # blue
        elif region.startswith('R.'):
            return '#C44E52'   # red
        else:
            return '#8C8C8C'   # grey (midline)

    jagwas_colors = [bar_color(r) for r in regions]

    x = np.arange(len(regions))
    bw = 0.38

    fig, ax = plt.subplots(figsize=(MM(160), MM(80)))

    bars_j = ax.bar(x - bw/2, j_vals, width=bw,
                    color=jagwas_colors, edgecolor='white', linewidth=0.4,
                    label='JAGWAS')
    bars_f = ax.bar(x + bw/2, f_vals, width=bw,
                    color='#AAAAAA', edgecolor='white', linewidth=0.4,
                    label='FastGWA minP')

    # Annotate fold enrichment above JAGWAS bar
    for xi, jv, fv, fo in zip(x, j_vals, f_vals, fold):
        if not np.isnan(fo):
            ax.text(xi - bw/2, jv + 1.2, f'×{fo:.1f}',
                    ha='center', va='bottom', fontsize=FS_TICK - 1.5,
                    color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=FS_TICK - 0.5)
    ax.set_ylabel('FUMA-defined loci (n)', fontsize=FS_LABEL)
    ax.set_ylim(0, max(j_vals) * 1.22)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend: method + hemisphere coding
    p_jagwas_left  = mpatches.Patch(color='#4C72B0', label='JAGWAS — left hemisphere')
    p_jagwas_right = mpatches.Patch(color='#C44E52', label='JAGWAS — right hemisphere')
    p_jagwas_mid   = mpatches.Patch(color='#8C8C8C', label='JAGWAS — midline')
    p_fastgwa      = mpatches.Patch(color='#AAAAAA', label='FastGWA minP')
    ax.legend(handles=[p_jagwas_left, p_jagwas_right, p_jagwas_mid, p_fastgwa],
              fontsize=FS_TICK - 0.5, loc='upper right', frameon=False,
              ncol=2)

    # Total annotation
    total_j = int(j_vals.sum())
    total_f = int(f_vals.sum())
    ax.text(0.01, 0.97,
            f'JAGWAS total: {total_j} loci  |  FastGWA total: {total_f} loci',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FS_TICK - 0.5, color='#333333')

    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, str(out_stem), dpi=300)
    plt.close(fig)
    print(f"Saved: {out_stem}.pdf / .png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Per-region JAGWAS vs FastGWA FUMA loci comparison (Supp Fig S33).')
    parser.add_argument('--al',       default=str(DEFAULT_AL),
                        help='Path to aggregate_loci.csv')
    parser.add_argument('--fuma_dir', default=str(DEFAULT_FUMA_DIR),
                        help='FastGWA FUMA output root directory')
    parser.add_argument('--out',      default=str(DEFAULT_OUT),
                        help='Output path stem (no extension)')
    args = parser.parse_args()
    make_figure(Path(args.al), Path(args.fuma_dir), Path(args.out))
