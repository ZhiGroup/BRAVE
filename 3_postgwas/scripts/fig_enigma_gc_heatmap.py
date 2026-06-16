#!/usr/bin/env python3
"""
fig_enigma_gc_heatmap.py
------------------------
Regenerate the ENIGMA GC heatmap (Fig 4B) with a diagonal layout:
  - ICV removed (no matched BRE region)
  - ENIGMA columns reordered to match BRE row groups → staircase diagonal
  - Gold boxes on matched BRE ↔ ENIGMA pairs (same style as Shape GC)

Input:  results/gc/enigma_gc.csv
Output: figures/main/Fig4_novelty_gc/Fig4B_enigma_gc_heatmap.pdf/.png

Usage:
  python fig_enigma_gc_heatmap.py
  python fig_enigma_gc_heatmap.py --csv ... --out ...
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent

DEFAULT_CSV = ROOT_DIR / 'results' / 'gc' / 'enigma_gc.csv'
DEFAULT_OUT = ROOT_DIR / 'figures' / 'main' \
              / 'Fig4_novelty_gc' / 'Fig4B_enigma_gc_heatmap'

sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

# ── BRE row order (fixed) ──────────────────────────────────────────────────────
BRE_ROW_ORDER = [
    'Brain_Stem_or_4th_Ventricle', 'CSF',
    'Left_Accumbens-area',  'Right_Accumbens-area',
    'Left_Amygdala',        'Right_Amygdala',
    'Left_Caudate',         'Right_Caudate',
    'Left_Hippocampus',     'Right_Hippocampus',
    'Left_Pallidum',        'Right_Pallidum',
    'Left_Putamen',         'Right_Putamen',
    'Left_Thalamus_Proper', 'Right_Thalamus-Proper',
]
BRE_DISP = {
    'Brain_Stem_or_4th_Ventricle': 'BrainStem',
    'CSF':                          'CSF',
    'Left_Accumbens-area':          'L.Accumbens',
    'Right_Accumbens-area':         'R.Accumbens',
    'Left_Amygdala':                'L.Amygdala',
    'Right_Amygdala':               'R.Amygdala',
    'Left_Caudate':                 'L.Caudate',
    'Right_Caudate':                'R.Caudate',
    'Left_Hippocampus':             'L.Hippocampus',
    'Right_Hippocampus':            'R.Hippocampus',
    'Left_Pallidum':                'L.Pallidum',
    'Right_Pallidum':               'R.Pallidum',
    'Left_Putamen':                 'L.Putamen',
    'Right_Putamen':                'R.Putamen',
    'Left_Thalamus_Proper':         'L.Thalamus',
    'Right_Thalamus-Proper':        'R.Thalamus',
}

# ── ENIGMA column order: matches BRE row groups → diagonal layout ─────────────
# ICV omitted (no matched BRE region).
# ventralDC kept at end (off-diagonal context).
ENIGMA_COL_ORDER = [
    'Brainstem',    # matches BrainStem (row 0)
    'Accumbens',    # matches L/R.Accumbens (rows 2-3)
    'Amygdala',     # matches L/R.Amygdala (rows 4-5)
    'Caudate',      # matches L/R.Caudate (rows 6-7)
    'Hippocampus',  # matches L/R.Hippocampus (rows 8-9)
    'Pallidum',     # matches L/R.Pallidum (rows 10-11)
    'Putamen',      # matches L/R.Putamen (rows 12-13)
    'Thalamus',     # matches L/R.Thalamus (rows 14-15)
    'ventralDC',    # no matched BRE — context column
]
ENIGMA_DISP = {
    'Brainstem':   'Brainstem',
    'Accumbens':   'Accumbens',
    'Amygdala':    'Amygdala',
    'Caudate':     'Caudate',
    'Hippocampus': 'Hippocampus',
    'Pallidum':    'Pallidum',
    'Putamen':     'Putamen',
    'Thalamus':    'Thalamus',
    'ventralDC':   'VentralDC',
}

# BRE region → matched ENIGMA trait (for gold diagonal boxes)
BRE_TO_ENIGMA = {
    'Brain_Stem_or_4th_Ventricle': 'Brainstem',
    'Left_Accumbens-area':          'Accumbens',
    'Right_Accumbens-area':         'Accumbens',
    'Left_Amygdala':                'Amygdala',
    'Right_Amygdala':               'Amygdala',
    'Left_Caudate':                 'Caudate',
    'Right_Caudate':                'Caudate',
    'Left_Hippocampus':             'Hippocampus',
    'Right_Hippocampus':            'Hippocampus',
    'Left_Pallidum':                'Pallidum',
    'Right_Pallidum':               'Pallidum',
    'Left_Putamen':                 'Putamen',
    'Right_Putamen':                'Putamen',
    'Left_Thalamus_Proper':         'Thalamus',
    'Right_Thalamus-Proper':        'Thalamus',
    # CSF → no match
}


def make_figure(csv_path: Path, out_stem: Path):
    apply_mpl_style()

    df = pd.read_csv(csv_path)

    # Drop ICV
    df = df[df['enigma'] != 'ICV'].copy()

    # Re-run FDR on filtered set
    df = df.dropna(subset=['p'])
    _, fdr_q, _, _ = multipletests(df['p'].values, method='fdr_bh')
    df['fdr_q'] = fdr_q
    df['fdr_sig'] = fdr_q < 0.05

    # Aggregate: max |rg| per (bre_region, enigma) across dims.
    # We plot |rg| (max absolute correlation) — NOT signed best_rg — because BRE
    # dimensions are unlabelled embedding axes: the sign of any individual dim's
    # rg with an external trait is arbitrary (depends on the contrastive
    # encoder's weight signs, not biology). Plotting signed best_rg produced
    # apparent opposite-sign rg between L and R hemispheres for the same
    # matched ENIGMA region (e.g., L.Accumbens vs R.Accumbens against ENIGMA
    # Accumbens) — a sign-mapping artefact rather than lateralization. Updated
    # 2026-05-26 in response to PI review.
    agg = df.groupby(['bre_region', 'enigma']).apply(
        lambda g: pd.Series({
            'max_abs_rg': g['rg'].abs().max(),
            'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
        })
    ).reset_index()

    # Build matrices (16 rows × 9 cols)
    n_rows = len(BRE_ROW_ORDER)
    n_cols = len(ENIGMA_COL_ORDER)
    mat_rg = np.full((n_rows, n_cols), np.nan)
    mat_q  = np.full((n_rows, n_cols), np.nan)

    for _, row in agg.iterrows():
        if row['bre_region'] in BRE_ROW_ORDER and row['enigma'] in ENIGMA_COL_ORDER:
            i = BRE_ROW_ORDER.index(row['bre_region'])
            j = ENIGMA_COL_ORDER.index(row['enigma'])
            mat_rg[i, j] = row['max_abs_rg']
            mat_q[i, j]  = row['best_fdr_q']

    # Diagonal mask (matched BRE ↔ ENIGMA pairs)
    is_diag = np.zeros((n_rows, n_cols), dtype=bool)
    for i, br in enumerate(BRE_ROW_ORDER):
        matched = BRE_TO_ENIGMA.get(br)
        if matched and matched in ENIGMA_COL_ORDER:
            j = ENIGMA_COL_ORDER.index(matched)
            is_diag[i, j] = True

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Sequential colormap (0 → vmax) since we now plot |rg|, not signed rg.
    vmax = np.nanpercentile(mat_rg, 95)

    fig, ax = plt.subplots(figsize=(MM(130), MM(110)))
    im = ax.imshow(mat_rg, cmap='OrRd', vmin=0, vmax=vmax, aspect='auto')

    for i in range(n_rows):
        for j in range(n_cols):
            # FDR asterisk
            if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=FS_TICK - 1, color='black', fontweight='bold')
            # Gold box for matched diagonal
            if is_diag[i, j]:
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1.5, edgecolor='gold', facecolor='none', zorder=5)
                ax.add_patch(rect)

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([ENIGMA_DISP[e] for e in ENIGMA_COL_ORDER],
                       rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([BRE_DISP[r] for r in BRE_ROW_ORDER], fontsize=FS_TICK)

    # Separator line after ventralDC (off-diagonal column)
    ax.axvline(n_cols - 1 - 0.5, color='#888888', lw=0.8, ls='--')

    ax.set_title('Genetic correlation: BRE dims vs ENIGMA volume\n'
                 '(max |rg| across dims; * FDR q<0.05; □ matched region)',
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('ENIGMA brain region (volume)', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation |rg|', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, str(out_stem), dpi=300)
    plt.close(fig)
    print(f"Saved: {out_stem}.pdf / .png")

    # ── Print diagonal enrichment stats ───────────────────────────────────────
    diag_rg = [abs(mat_rg[i, j])
               for i in range(n_rows) for j in range(n_cols)
               if is_diag[i, j] and not np.isnan(mat_rg[i, j])]
    off_rg  = [abs(mat_rg[i, j])
               for i in range(n_rows) for j in range(n_cols)
               if not is_diag[i, j] and not np.isnan(mat_rg[i, j])]
    if diag_rg and off_rg:
        print(f"Diagonal mean |rg|   = {np.mean(diag_rg):.3f} (n={len(diag_rg)})")
        print(f"Off-diagonal mean |rg| = {np.mean(off_rg):.3f} (n={len(off_rg)})")
        print(f"Diagonal / off-diagonal ratio = {np.mean(diag_rg)/np.mean(off_rg):.2f}×")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regenerate ENIGMA GC heatmap with diagonal layout (ICV removed).')
    parser.add_argument('--csv', default=str(DEFAULT_CSV),
                        help='Path to enigma_gc.csv')
    parser.add_argument('--out', default=str(DEFAULT_OUT),
                        help='Output path stem (no extension)')
    args = parser.parse_args()
    make_figure(Path(args.csv), Path(args.out))
