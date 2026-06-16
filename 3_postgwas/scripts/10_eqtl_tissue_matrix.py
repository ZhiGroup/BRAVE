"""
10_eqtl_tissue_matrix.py
Cross-region eQTL tissue specificity matrix.

Reads eqtl_tissue_counts.csv (from Script 02) and builds a
16-region × N-tissue heatmap showing which brain eQTL tissues
are most active for each JAGWAS region.

Two heatmaps:
  (A) All 26 eQTL tissues × 16 regions (supplementary)
  (B) GTEx brain tissues only (13 tissues) × 16 regions (main figure)

Values: number of unique eQTL-mapped genes per region × tissue (eqtlMapFilt == 1).
Rows = regions (sorted anatomically). Columns = tissues (clustered by profile).

Outputs:
  results/cross_region/eqtl_tissue_matrix.csv   — pivoted matrix
  figures/cross_region/eqtl_tissue_matrix_all.pdf/.png
  figures/cross_region/eqtl_tissue_matrix_brain.pdf/.png

Supports paper Claim 3: shows whether brain-tissue eQTL specificity matches
  the expected region (e.g., Putamen → Brain_Putamen_basal_ganglia).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import scipy.cluster.hierarchy as sch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
RES_PER  = Path(__file__).parents[1] / "results" / "per_region"
RES_CR   = Path(__file__).parents[1] / "results" / "cross_region"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "cross_region"
RES_CR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── REGION ORDER ──────────────────────────────────────────────────────────────
BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def sort_key_disp(disp):
    """Sort key from display name."""
    for i, b in enumerate(BASE_ORDER):
        bshort = b.replace('-area', '').replace('_', ' ').replace('Thalamus Proper', 'Thalamus')
        if bshort in disp:
            return (i, 0 if disp.startswith('L.') else 1)
    for i, m in enumerate(MIDLINE):
        mshort = m.replace('_', ' ').replace('Brain Stem or 4th Ventricle', 'Brain Stem')
        if mshort in disp or 'BrStem' in disp or 'CSF' in disp:
            return (len(BASE_ORDER) + i, 0)
    return (99, 0)

# ── GTEX BRAIN TISSUES (key subset for main figure) ──────────────────────────
GTEX_BRAIN = {
    'Brain_Caudate_basal_ganglia':              'Caudate (BG)',
    'Brain_Putamen_basal_ganglia':              'Putamen (BG)',
    'Brain_Nucleus_accumbens_basal_ganglia':    'Accumbens (BG)',
    'Brain_Substantia_nigra':                   'Substantia Nigra',
    'Brain_Amygdala':                           'Amygdala',
    'Brain_Hippocampus':                        'Hippocampus',
    'Brain_Anterior_cingulate_cortex_BA24':     'ACC (BA24)',
    'Brain_Frontal_Cortex_BA9':                 'Frontal Ctx (BA9)',
    'Brain_Cortex':                             'Cortex',
    'Brain_Hypothalamus':                       'Hypothalamus',
    'Brain_Cerebellar_Hemisphere':              'Cerebellar Hem.',
    'Brain_Cerebellum':                         'Cerebellum',
    'Brain_Spinal_cord_cervical_c-1':           'Spinal Cord',
}

# ── LOAD EQTL TISSUE COUNTS ───────────────────────────────────────────────────
df = pd.read_csv(RES_PER / 'eqtl_tissue_counts.csv')
print(f"Loaded eQTL tissue counts: {len(df)} rows, "
      f"{df['display'].nunique()} regions, {df['tissue'].nunique()} tissues")

# Pivot: regions × tissues
pivot = df.pivot_table(index='display', columns='tissue',
                       values='n_genes', aggfunc='sum', fill_value=0)

# Sort regions anatomically
pivot = pivot.loc[sorted(pivot.index, key=sort_key_disp)]

# Save full matrix
pivot.to_csv(RES_CR / 'eqtl_tissue_matrix.csv')
print(f"Saved: {RES_CR}/eqtl_tissue_matrix.csv")

# ── HEATMAP FUNCTION ──────────────────────────────────────────────────────────
CMAP = LinearSegmentedColormap.from_list('eqtl', ['#FFFFFF', '#08306B'])

def tissue_heatmap(mat, row_labels, col_labels, title, out_stem,
                   figw=MM(180), figh_per_row=MM(10), min_figh=MM(100),
                   annotate=True, cluster_cols=True):

    n_rows, n_cols = mat.shape
    figh = max(min_figh, figh_per_row * n_rows)

    # Optional column clustering
    if cluster_cols and n_cols > 2:
        col_dist = 1 - np.corrcoef(mat.T)
        col_dist = np.clip(col_dist, 0, 2)
        np.fill_diagonal(col_dist, 0)
        from scipy.spatial.distance import squareform
        try:
            linkage = sch.linkage(squareform(col_dist), method='average')
            col_order = sch.dendrogram(linkage, no_plot=True)['leaves']
        except Exception:
            col_order = list(range(n_cols))
        mat        = mat[:, col_order]
        col_labels = [col_labels[i] for i in col_order]

    vmax = mat.max()
    fig, ax = plt.subplots(figsize=(figw, figh))
    im = ax.imshow(mat, cmap=CMAP, aspect='auto', vmin=0, vmax=vmax)

    if annotate and n_rows * n_cols <= 300:
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                col = 'white' if val > vmax * 0.6 else 'black'
                ax.text(j, i, str(int(val)), ha='center', va='center',
                        fontsize=5, color=col)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=FS_TICK - 0.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_LABEL, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('eQTL-mapped genes', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / out_stem)
    plt.close()

# ── (A) ALL TISSUES × ALL REGIONS ────────────────────────────────────────────
print("\nGenerating heatmap (A): all tissues ...")
mat_all    = pivot.values
row_labels = pivot.index.tolist()
col_labels = pivot.columns.tolist()

# Tidy column labels
def tidy(t):
    import re
    t = re.sub(r'^Brain_', '', t)
    return t.replace('_', ' ')

col_labels_tidy = [tidy(c) for c in col_labels]

tissue_heatmap(mat_all, row_labels, col_labels_tidy,
               title='eQTL-mapped genes per region × tissue (all tissues)',
               out_stem='eqtl_tissue_matrix_all',
               figh_per_row=MM(11), annotate=False, cluster_cols=True)
print("Saved: figures/cross_region/eqtl_tissue_matrix_all.pdf/.png")

# ── (B) GTEX BRAIN TISSUES ONLY ──────────────────────────────────────────────
print("\nGenerating heatmap (B): GTEx brain tissues only ...")
brain_cols = [c for c in pivot.columns if c in GTEX_BRAIN]
if not brain_cols:
    print("  WARNING: no GTEx brain tissue columns found in pivot. Available:", pivot.columns[:10].tolist())
else:
    pivot_brain    = pivot[brain_cols]
    mat_brain      = pivot_brain.values
    col_brain_disp = [GTEX_BRAIN[c] for c in brain_cols]

    tissue_heatmap(mat_brain, row_labels, col_brain_disp,
                   title='eQTL-mapped genes per region × GTEx brain tissue',
                   out_stem='eqtl_tissue_matrix_brain',
                   figw=MM(160), figh_per_row=MM(12), annotate=True,
                   cluster_cols=True)
    print("Saved: figures/cross_region/eqtl_tissue_matrix_brain.pdf/.png")

    # ── Print brain tissue matrix ─────────────────────────────────────────────
    print("\n=== GTEx brain tissue eQTL counts (genes) ===")
    disp_df = pivot_brain.copy()
    disp_df.columns = [GTEX_BRAIN[c] for c in brain_cols]
    print(disp_df.to_string())

    # Highlight: which tissue is #1 per region?
    print("\n=== Top GTEx brain tissue per region ===")
    for region in pivot_brain.index:
        top_col = pivot_brain.loc[region].idxmax()
        top_val = pivot_brain.loc[region].max()
        print(f"  {region:<30} → {GTEX_BRAIN[top_col]:<30} ({top_val} genes)")
