"""
04_celltype_enrichment.py
Per-region MAGMA cell-type enrichment bar charts.

Claim supported: BREs are structurally-encoded (not functionally-encoded).
Key finding: structural cell types (Fibroblast, Astrocyte, Vascular, Oligodendrocyte,
Microglia, Ependymal) are consistently enriched (BETA_STD > 0, FDR-sig) while
neuronal cell types (MSNs, pyramidal neurons, interneurons) are consistently depleted
(BETA_STD < 0) across ALL 16 BRE regions.

Input:  FUMA_CellType/{region}/magma_celltype_step1.txt
Output: results/celltype/per_region_celltype.csv
        figures/celltype/per_region_celltype_bars.pdf/.png
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FIG_DPI

apply_mpl_style()

# ── Paths ──────────────────────────────────────────────────────────────────────
CELLTYPE_DIR = Path(cfg.postgwas.fuma_dir).parent / "FUMA_CellType"
SCRIPT_DIR   = Path(__file__).parent
BASE_DIR     = SCRIPT_DIR.parent
RES_DIR      = BASE_DIR / 'results' / 'celltype'
FIG_DIR      = BASE_DIR / 'figures' / 'celltype'
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Region folders + display labels ───────────────────────────────────────────
REGIONS = {
    'Left_Accumbens-area':         'L. Accumbens',
    'Left_Amydala':                'L. Amygdala',
    'Left_Caudate':                'L. Caudate',
    'Left_Hippocampus':            'L. Hippocampus',
    'Left_Pallidium':              'L. Pallidum',
    'Left_Putamen':                'L. Putamen',
    'Left_Thalamus_Proper':        'L. Thalamus',
    'Right_Accumbens':             'R. Accumbens',
    'Right_Amygdala':              'R. Amygdala',
    'Right_Caudate':               'R. Caudate',
    'Right_Hippocampus':           'R. Hippocampus',
    'Right_Pallidum':              'R. Pallidum',
    'Right_Putamen':               'R. Putamen',
    'Right_Thalamus-Proper':       'R. Thalamus',
    'Brain_Stem_or_4th_Ventricle': 'Brain Stem',
    'CSF':                         'CSF',
}

# ── Cell-type broad categories ────────────────────────────────────────────────
# Each entry: (category_label, color, [keyword substrings that match cell_type])
# Order matters: first match wins.
STRUCTURAL_CATS = [
    ('Fibroblast/ECM',        '#2166AC', ['Fibroblast', 'fibroblast', 'leptomeningeal']),
    ('Astrocyte',             '#4DAC26', ['Astrocyte', 'Astrocytes', 'Astro_',
                                          'hippocampal_astrocyte', 'Bergmann']),
    ('Oligodendrocyte',       '#762A83', ['Oligodendrocyte', 'Oligos', 'Oligos_Pre',
                                          'Committed_oligodendrocyte', 'oligodendrocyte',
                                          'OPC_', 'Olig_']),
    ('Microglia',             '#E08214', ['Microglia', 'microglial', 'MG_', 'macrophage', 'Macro_']),
    ('Vascular/Endothelial',  '#1A9850', ['Vascular', 'Endothelial', 'endothelial', 'Endo_',
                                          'Mural', 'pericyte', 'smooth_muscle', 'vascular']),
    ('Ependymal',             '#80CDC1', ['Ependymal', 'ependymal', 'Ependyma_',
                                          'Choroid', 'choroid']),
]

NEURONAL_CATS = [
    ('D1/D2 MSN',      '#D6604D', ['D1_Matrix', 'D1_Striosome', 'D2_Matrix', 'D2_Striosome',
                                    'D1_D2_Hybrid', 'D1_', 'D2_']),
    ('MSN (general)',  '#F4A582', ['Medium_spiny_neuron', 'Eccentric_medium_spiny_neuron',
                                    'spiny_neuron']),
    ('Interneuron',    '#B2182B', ['Interneuron', 'interneuron', 'MGE_interneuron',
                                    'CGE_interneuron', 'LAMP5_LHX6', 'Chandelier',
                                    'inhibitory', 'Midbrain_derived', 'Inh_']),
    ('Pyramidal/IT',   '#67001F', ['hippocampal_pyramidal', 'pyramidal_neuron',
                                    'Deep_layer_intratelencephalic',
                                    'Upper_layer_intratelencephalic',
                                    'Deep_layer_corticothalamic', 'Deep_layer_near',
                                    'excitatory', 'Hippocampal_CA',
                                    'Amygdala_excitatory', 'Thalamic_excitatory', 'Ex_']),
    ('Granule/Other N',
                       '#C51B7D', ['hippocampal_granule', 'granule_cell',
                                    'Hippocampal_dentate_gyrus', 'Mammillary_body',
                                    'Upper_rhombic_lip', 'Lower_rhombic_lip',
                                    'Cerebellar', 'SOX6_', 'CALB1_', 'Splatter',
                                    'glutamatergic', 'GABAergic']),
]

ALL_CATS     = STRUCTURAL_CATS + NEURONAL_CATS
STRUCT_NAMES = {c[0] for c in STRUCTURAL_CATS}
NEURO_NAMES  = {c[0] for c in NEURONAL_CATS}
CAT_COLORS   = {c[0]: c[1] for c in ALL_CATS}


def classify_celltype(ct: str):
    """Return broad category label for a cell_type string, or None if unclassified."""
    for cat_label, _, kws in ALL_CATS:
        if any(kw in ct for kw in kws):
            return cat_label
    return None


# ── Load and classify step1 files ─────────────────────────────────────────────
records = []
for folder, label in REGIONS.items():
    fpath = CELLTYPE_DIR / folder / 'magma_celltype_step1.txt'
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['region_folder'] = folder
    df['region_label']  = label
    df['category']      = df['Cell_type'].apply(classify_celltype)
    records.append(df)

all_df = pd.concat(records, ignore_index=True)
print(f"Loaded {len(records)} regions, {len(all_df)} total rows")

# Unclassified summary
unc = all_df[all_df['category'].isna()]['Cell_type'].value_counts()
if len(unc):
    print(f"Unclassified cell types ({len(unc)} unique): {unc.head(10).to_dict()}")

# ── Aggregate: per region × category → mean BETA_STD (across all datasets) ───
agg = (all_df
       .dropna(subset=['category'])
       .groupby(['region_label', 'category'], sort=False)
       .agg(
           mean_beta_std=('BETA_STD', 'mean'),
           n_entries    =('BETA_STD', 'count'),
           n_fdr_sig    =('P.adj',    lambda x: (x < 0.05).sum()),
       )
       .reset_index())

# Flag structural vs neuronal
agg['group'] = np.where(agg['category'].isin(STRUCT_NAMES), 'structural', 'neuronal')

# Save full summary table
agg.to_csv(RES_DIR / 'per_region_celltype.csv', index=False)
print(f"Saved: {RES_DIR}/per_region_celltype.csv")

# ── Figure: 4×4 grid of per-region horizontal bar charts ──────────────────────
N_TOP    = 4   # top structural categories to show per region (all 6 shown; top by BETA_STD)
N_NEU    = 4   # neuronal categories to show (most depleted)
NCOLS    = 4
NROWS    = 4   # 4 × 4 = 16 panels

fig, axes = plt.subplots(
    NROWS, NCOLS,
    figsize=(MM(240), MM(220)),   # extra height to give legend room below
)

region_order = list(REGIONS.values())

for idx, label in enumerate(region_order):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]

    sub = agg[agg['region_label'] == label].copy()
    struct = (sub[sub['group'] == 'structural']
              .sort_values('mean_beta_std', ascending=False))
    neuro  = (sub[sub['group'] == 'neuronal']
              .sort_values('mean_beta_std', ascending=True))   # most negative first

    # Combine: structural (positive) on top, neuronal (negative) below
    plot_df = pd.concat([struct, neuro], ignore_index=True)
    plot_df = plot_df[::-1].reset_index(drop=True)   # flip so positive at top visually

    colors = [CAT_COLORS.get(c, '#888888') for c in plot_df['category']]
    y_pos  = np.arange(len(plot_df))

    bars = ax.barh(y_pos, plot_df['mean_beta_std'],
                   color=colors, height=0.65, linewidth=0.3,
                   edgecolor='white')
    ax.axvline(0, color='#333333', linewidth=0.6, linestyle='-')

    # Y-axis labels (short)
    short_labels = (plot_df['category']
                    .str.replace('Vascular/Endothelial', 'Vascular')
                    .str.replace('Oligodendrocyte',       'Oligo.')
                    .str.replace('Fibroblast/ECM',        'Fibroblast')
                    .str.replace('Granule/Other N',       'Granule'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=5.5)

    # Add FDR significance markers on structural bars
    for i, (_, row_data) in enumerate(plot_df[::-1].reset_index(drop=True).iterrows()):
        if row_data['group'] == 'structural' and row_data['n_fdr_sig'] > 0:
            x = row_data['mean_beta_std']
            ax.text(x + 0.002, i, '*', ha='left', va='center',
                    fontsize=5, color='#333333')

    ax.set_title(label, fontsize=7, fontweight='bold', pad=2)
    ax.set_xlabel('Mean BETA_STD', fontsize=6)
    ax.tick_params(axis='x', labelsize=5.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Light fill zones
    ax.axvspan(-0.3, 0, alpha=0.03, color='#D6604D', zorder=0)
    ax.axvspan( 0, 0.2,  alpha=0.03, color='#2166AC', zorder=0)

# Reserve bottom 10% for legend, 2% at top for suptitle
plt.tight_layout(rect=[0, 0.10, 1, 0.98], h_pad=1.5, w_pad=1.0)

# Legend — placed in the reserved bottom strip
legend_patches = [
    mpatches.Patch(color=CAT_COLORS[c[0]], label=c[0])
    for c in ALL_CATS
]
fig.legend(
    handles=legend_patches, loc='lower center',
    ncol=4, fontsize=5.5, frameon=False,
    bbox_to_anchor=(0.5, 0.01),   # 1% from bottom in figure coords
    title='Cell-type category', title_fontsize=6,
)

fig.suptitle(
    'MAGMA cell-type enrichment: structural enrichment vs neuronal depletion across 16 BRE regions',
    fontsize=8, fontweight='bold', y=0.995,
)

save_mpl(fig, FIG_DIR / 'per_region_celltype_bars')
plt.close(fig)
print("Done: 04_celltype_enrichment.py")
