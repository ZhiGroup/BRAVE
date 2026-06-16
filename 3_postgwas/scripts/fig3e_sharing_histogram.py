#!/usr/bin/env python3
"""
fig3e_sharing_histogram.py
--------------------------
Panel E for Figure 3: histogram of aggregate loci by number of regions in which
each locus was detected.

Input:
  results/cross_region/aggregate_loci.csv

Output:
  figures/main/Fig3_gwas_loci/Fig3E_sharing_histogram.png
  figures/main/Fig3_gwas_loci/Fig3E_sharing_histogram.pdf

Usage:
  python fig3e_sharing_histogram.py
  python fig3e_sharing_histogram.py \
      --input  results/cross_region/aggregate_loci.csv \
      --out    figures/main/Fig3_gwas_loci/Fig3E_sharing_histogram
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

DEFAULT_INPUT = ROOT_DIR / 'results' / 'cross_region' / 'aggregate_loci.csv'
DEFAULT_OUT   = ROOT_DIR / 'figures' / 'main' \
                / 'Fig3_gwas_loci' / 'Fig3E_sharing_histogram'


def make_panel(input_csv: Path, out_stem: Path):
    apply_mpl_style()

    df = pd.read_csv(input_csv)
    counts = df['n_regions'].value_counts().sort_index()
    x = counts.index.values          # 1 … 16
    y = counts.values

    # Colour: gradient blue (1 region) → red (16 regions)
    cmap   = cm.get_cmap('RdYlBu_r')
    norm   = mcolors.Normalize(vmin=1, vmax=16)
    colors = [cmap(norm(xi)) for xi in x]

    fig, ax = plt.subplots(figsize=(MM(68), MM(52)))

    bars = ax.bar(x, y, color=colors, edgecolor='white', linewidth=0.4, width=0.75)

    # Annotate bars with count (only for bars ≥ threshold height to avoid clutter)
    for xi, yi in zip(x, y):
        if yi >= 4:
            ax.text(xi, yi + 1.5, str(yi),
                    ha='center', va='bottom',
                    fontsize=FS_TICK - 1, color='#333333')

    ax.set_xlabel('Number of regions', fontsize=FS_LABEL)
    ax.set_ylabel('Aggregate loci (n)', fontsize=FS_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=FS_TICK - 1, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=FS_TICK)

    # Total annotation
    total = y.sum()
    ax.text(0.97, 0.97, f'Total: {total} loci',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FS_TICK, color='#333333')

    # Axis limits
    ax.set_xlim(0.35, 16.65)
    ax.set_ylim(0, max(y) * 1.18)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Colour bar legend
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04,
                        ticks=[1, 8, 16])
    cbar.set_label('Regions', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK)

    save_mpl(fig, str(out_stem), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Fig3E sharing histogram.')
    parser.add_argument('--input', default=str(DEFAULT_INPUT),
                        help='Path to aggregate_loci.csv')
    parser.add_argument('--out', default=str(DEFAULT_OUT),
                        help='Output path stem (no extension)')
    args = parser.parse_args()
    make_panel(Path(args.input), Path(args.out))
