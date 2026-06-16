#!/usr/bin/env python3
"""
fig_unique_loci_per_region.py
------------------------------
Region-specific (unique) loci per region: horizontal bar chart showing absolute
count (bar length) and % unique (annotation), sorted by count. Includes the
Jaccard value with the bilateral counterpart for each paired region, connecting
unique-locus scarcity in bilateral pairs to their high loci-sharing.

Inputs:
  results/cross_region/aggregate_loci.csv
  results/cross_region/jaccard_matrix.csv

Output:
  figures/supplementary/unique_loci_per_region/unique_loci_per_region.pdf/.png

Usage:
  python fig_unique_loci_per_region.py
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

DEFAULT_AL  = ROOT_DIR / 'results' / 'cross_region' / 'aggregate_loci.csv'
DEFAULT_JAC = ROOT_DIR / 'results' / 'cross_region' / 'jaccard_matrix.csv'
DEFAULT_OUT = ROOT_DIR / 'figures' / 'supplementary' \
              / 'unique_loci_per_region' / 'unique_loci_per_region'

# Bilateral counterpart lookup
BILATERAL = {
    'L. Accumbens':  'R. Accumbens',
    'R. Accumbens':  'L. Accumbens',
    'L. Amygdala':   'R. Amygdala',
    'R. Amygdala':   'L. Amygdala',
    'L. Caudate':    'R. Caudate',
    'R. Caudate':    'L. Caudate',
    'L. Hippocampus':'R. Hippocampus',
    'R. Hippocampus':'L. Hippocampus',
    'L. Pallidum':   'R. Pallidum',
    'R. Pallidum':   'L. Pallidum',
    'L. Putamen':    'R. Putamen',
    'R. Putamen':    'L. Putamen',
    'L. Thalamus':   'R. Thalamus',
    'R. Thalamus':   'L. Thalamus',
}

# Anatomical grouping → colour
GROUPS = {
    'Basal ganglia':   ['L. Putamen', 'R. Putamen', 'L. Caudate', 'R. Caudate',
                        'L. Accumbens', 'R. Accumbens', 'L. Pallidum', 'R. Pallidum'],
    'Medial temporal': ['L. Hippocampus', 'R. Hippocampus', 'L. Amygdala', 'R. Amygdala'],
    'Thalamus':        ['L. Thalamus', 'R. Thalamus'],
    'Brainstem / CSF': ['Brain Stem / 4th V.', 'CSF'],
}
GROUP_COLORS = {
    'Basal ganglia':   '#4C72B0',
    'Medial temporal': '#55A868',
    'Thalamus':        '#C44E52',
    'Brainstem / CSF': '#DD8452',
}

def region_group(region):
    for grp, members in GROUPS.items():
        if region in members:
            return grp
    return 'Other'


def make_figure(al_csv: Path, jac_csv: Path, out_stem: Path):
    apply_mpl_style()

    df  = pd.read_csv(al_csv)
    jac = pd.read_csv(jac_csv, index_col=0)

    # Total loci per region
    all_regions = []
    for r in df['regions']:
        all_regions.extend(r.split(';'))
    total_per = Counter(all_regions)

    # Unique loci per region (n_regions == 1)
    unique_rows = df[df['n_regions'] == 1]
    unique_per  = unique_rows['regions'].value_counts()

    # Build table for all 16 regions
    all_regs = list(jac.index)
    rows = []
    for reg in all_regs:
        n_total  = total_per.get(reg, 0)
        n_unique = int(unique_per.get(reg, 0))
        pct      = 100.0 * n_unique / n_total if n_total > 0 else 0.0
        bilat    = BILATERAL.get(reg, None)
        jac_val  = float(jac.loc[reg, bilat]) if bilat else None
        grp      = region_group(reg)
        rows.append(dict(region=reg, n_total=n_total, n_unique=n_unique,
                         pct_unique=pct, bilateral=bilat,
                         jaccard_bilateral=jac_val, group=grp))

    data = pd.DataFrame(rows).sort_values('n_unique', ascending=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(MM(120), MM(100)))

    y_pos = np.arange(len(data))
    bar_h = 0.65

    for i, (_, row) in enumerate(data.iterrows()):
        color = GROUP_COLORS[row['group']]
        ax.barh(i, row['n_unique'], height=bar_h,
                color=color, edgecolor='white', linewidth=0.4)

        # Annotate: count + % unique
        label = f"{int(row['n_unique'])}  ({row['pct_unique']:.0f}%)"
        ax.text(row['n_unique'] + 0.3, i, label,
                va='center', ha='left',
                fontsize=FS_TICK - 1, color='#333333')

        # Jaccard with bilateral counterpart (for paired regions only)
        jval = row['jaccard_bilateral']
        if jval is not None and not (isinstance(jval, float) and np.isnan(jval)):
            jlabel = f"J={jval:.2f}"
            ax.text(-0.5, i, jlabel,
                    va='center', ha='right',
                    fontsize=FS_TICK - 1.5, color='#666666', style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['region'], fontsize=FS_TICK)
    ax.set_xlabel('Region-specific loci (n)', fontsize=FS_LABEL)
    ax.set_xlim(-4, data['n_unique'].max() * 1.55)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Vertical reference line at 0
    ax.axvline(0, color='#999999', linewidth=0.5)

    # Legend for anatomical groups
    patches = [mpatches.Patch(color=c, label=g)
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, fontsize=FS_TICK - 0.5,
              loc='lower right', frameon=False)

    # Italic note about Jaccard labels
    ax.text(0.01, 0.01,
            'Italic: Jaccard similarity with bilateral counterpart',
            transform=ax.transAxes, fontsize=FS_TICK - 1.5,
            color='#666666', style='italic', va='bottom')

    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, str(out_stem), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--al',  default=str(DEFAULT_AL))
    parser.add_argument('--jac', default=str(DEFAULT_JAC))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()
    make_figure(Path(args.al), Path(args.jac), Path(args.out))
