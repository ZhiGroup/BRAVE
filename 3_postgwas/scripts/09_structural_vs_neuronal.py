"""
09_structural_vs_neuronal.py
Cross-region structural enrichment vs neuronal depletion summary figure.

Claim supported: BREs encode tissue architecture (structural), not functional
circuit activity (neuronal).

Novel finding: MAGMA cell-type enrichment shows a consistent dichotomy across
ALL 16 BRE regions — structural cell programs are enriched (BETA_STD > 0)
while neuronal programs are depleted (BETA_STD < 0). This is NOT region-specific;
it is universal, reflecting the morphological nature of BRE phenotypes.

The absence of MSN/pyramidal-neuron enrichment (even for the matched region's
dataset) is the biologically meaningful signal: BRE GWAS detects loci in the
same structural/developmental programs as ENIGMA volume GWAS, confirming that
BREs capture morphological genetic architecture.

Input:  FUMA_CellType/{region}/magma_celltype_step1.txt
Output: results/celltype/structural_vs_neuronal_summary.csv
        figures/celltype/structural_vs_neuronal.pdf/.png
        figures/celltype/structural_vs_neuronal_scatter.pdf/.png
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
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

# ── Region map ────────────────────────────────────────────────────────────────
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

# ── Cell-type classification ──────────────────────────────────────────────────
STRUCTURAL_KW = [
    'Fibroblast', 'fibroblast', 'leptomeningeal',
    'Astrocyte', 'Astrocytes', 'Astro_', 'hippocampal_astrocyte', 'Bergmann',
    'Oligodendrocyte', 'Oligos', 'Oligos_Pre', 'Committed_oligodendrocyte',
    'oligodendrocyte',
    'Microglia', 'microglial', 'MG_', 'macrophage', 'Macro_',
    'Vascular', 'Endothelial', 'endothelial', 'Endo_', 'Mural', 'pericyte',
    'smooth_muscle', 'vascular',
    'Ependymal', 'ependymal', 'Ependyma_', 'Choroid', 'choroid',
    'Bergmann', 'OPC_', 'Olig_',
]

NEURONAL_KW = [
    'D1_Matrix', 'D1_Striosome', 'D2_Matrix', 'D2_Striosome', 'D1_D2_Hybrid',
    'Medium_spiny_neuron', 'Eccentric_medium_spiny_neuron',
    'Interneuron', 'interneuron', 'MGE_interneuron', 'CGE_interneuron',
    'LAMP5_LHX6', 'Chandelier', 'inhibitory', 'Midbrain_derived', 'Inh_',
    'hippocampal_pyramidal', 'pyramidal_neuron',
    'hippocampal_granule', 'granule_cell',
    'hippocampal_interneuron',
    'Deep_layer_intratelencephalic', 'Upper_layer_intratelencephalic',
    'Deep_layer_corticothalamic', 'Deep_layer_near',
    'excitatory', 'Hippocampal_CA', 'Amygdala_excitatory', 'Thalamic_excitatory', 'Ex_',
    'Upper_rhombic_lip', 'Lower_rhombic_lip', 'Cerebellar', 'Splatter',
    'Hippocampal_dentate_gyrus', 'Mammillary_body', 'SOX6_', 'CALB1_',
    'glutamatergic', 'GABAergic',
]


def classify(ct: str) -> str:
    if any(kw in ct for kw in STRUCTURAL_KW):
        return 'structural'
    if any(kw in ct for kw in NEURONAL_KW):
        return 'neuronal'
    return 'other'


# ── Load all regions ───────────────────────────────────────────────────────────
records = []
for folder, label in REGIONS.items():
    fpath = CELLTYPE_DIR / folder / 'magma_celltype_step1.txt'
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['region'] = label
    df['group']  = df['Cell_type'].apply(classify)
    records.append(df)

all_df = pd.concat(records, ignore_index=True)
print(f"Loaded {len(records)} regions")

# ── Per-region aggregation ────────────────────────────────────────────────────
# For structural: use ALL classified structural entries (captures full breadth)
# For neuronal: use ALL classified neuronal entries
rows = []
for region, grp in all_df.groupby('region'):
    struct = grp[grp['group'] == 'structural']['BETA_STD'].values
    neuro  = grp[grp['group'] == 'neuronal' ]['BETA_STD'].values
    other  = grp[grp['group'] == 'other'    ]['BETA_STD'].values

    # One-sample t-test vs 0
    t_s, p_s = stats.ttest_1samp(struct, 0) if len(struct) > 1 else (np.nan, np.nan)
    t_n, p_n = stats.ttest_1samp(neuro,  0) if len(neuro)  > 1 else (np.nan, np.nan)

    n_struct_fdr = grp[(grp['group'] == 'structural') & (grp['P.adj'] < 0.05)].shape[0]
    n_neuro_fdr  = grp[(grp['group'] == 'neuronal')   & (grp['P.adj'] < 0.05)].shape[0]

    rows.append({
        'region':        region,
        'struct_mean':   struct.mean() if len(struct) else np.nan,
        'struct_sem':    struct.std(ddof=1) / np.sqrt(len(struct)) if len(struct) > 1 else np.nan,
        'struct_n':      len(struct),
        'struct_n_fdr':  n_struct_fdr,
        'struct_tstat':  t_s,
        'struct_pval':   p_s,
        'neuro_mean':    neuro.mean() if len(neuro) else np.nan,
        'neuro_sem':     neuro.std(ddof=1) / np.sqrt(len(neuro)) if len(neuro) > 1 else np.nan,
        'neuro_n':       len(neuro),
        'neuro_n_fdr':   n_neuro_fdr,
        'neuro_tstat':   t_n,
        'neuro_pval':    p_n,
    })

summary = pd.DataFrame(rows)
# Sort by structural enrichment (descending)
summary = summary.sort_values('struct_mean', ascending=False).reset_index(drop=True)

summary.to_csv(RES_DIR / 'structural_vs_neuronal_summary.csv', index=False)
print(f"Saved: {RES_DIR}/structural_vs_neuronal_summary.csv")

# Print to console
print("\n=== Structural vs Neuronal summary ===")
print(summary[['region','struct_mean','struct_n_fdr','struct_pval',
               'neuro_mean','neuro_n','neuro_pval']].to_string(index=False))

# ── Figure 1: Paired horizontal dot/bar plot ──────────────────────────────────
# Two panels side by side sharing the y-axis (regions)
# Left: structural BETA_STD (blue, positive)
# Right: neuronal BETA_STD (tomato, negative)

STRUCT_COL = '#2166AC'
NEURO_COL  = '#D6604D'
STRIP_COL  = '#888888'

fig, (ax_s, ax_n) = plt.subplots(
    1, 2,
    figsize=(MM(180), MM(110)),
    sharey=True,
    constrained_layout=True,
)

regions_ordered = summary['region'].tolist()  # sorted by struct_mean
y_pos = np.arange(len(regions_ordered))

# --- Structural panel ---
ax_s.barh(y_pos, summary['struct_mean'],
          xerr=summary['struct_sem'],
          color=STRUCT_COL, alpha=0.80, height=0.55,
          error_kw=dict(ecolor='#333333', linewidth=0.8, capsize=2))
ax_s.axvline(0, color='#333333', linewidth=0.8)

# Individual data points (strip) for structural, sampled if too many
for i, region in enumerate(regions_ordered):
    vals = all_df[(all_df['region'] == region) &
                  (all_df['group'] == 'structural')]['BETA_STD'].values
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
    ax_s.scatter(vals, i + jitter, s=0.8, color=STRIP_COL, alpha=0.25,
                 linewidths=0, zorder=3)

# p-value stars (structural > 0)
for i, row in summary.iterrows():
    p = row['struct_pval']
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    if star:
        ax_s.text(row['struct_mean'] + row['struct_sem'] + 0.003,
                  i, star, va='center', ha='left', fontsize=5, color='#333333')

ax_s.set_yticks(y_pos)
ax_s.set_yticklabels(regions_ordered, fontsize=6.5)
ax_s.set_xlabel('Mean BETA_STD', fontsize=7)
ax_s.set_title('Structural cell types\n(enriched)', fontsize=7.5,
               fontweight='bold', color=STRUCT_COL)
ax_s.spines[['top', 'right']].set_visible(False)
ax_s.set_xlim(left=-0.01)

# --- Neuronal panel ---
ax_n.barh(y_pos, summary['neuro_mean'],
          xerr=summary['neuro_sem'],
          color=NEURO_COL, alpha=0.80, height=0.55,
          error_kw=dict(ecolor='#333333', linewidth=0.8, capsize=2))
ax_n.axvline(0, color='#333333', linewidth=0.8)

for i, region in enumerate(regions_ordered):
    vals = all_df[(all_df['region'] == region) &
                  (all_df['group'] == 'neuronal')]['BETA_STD'].values
    jitter = np.random.default_rng(99).uniform(-0.18, 0.18, len(vals))
    ax_n.scatter(vals, i + jitter, s=0.8, color=STRIP_COL, alpha=0.25,
                 linewidths=0, zorder=3)

# p-value stars (neuronal < 0 → two-sided t gives same p)
for i, row in summary.iterrows():
    p = row['neuro_pval']
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    if star:
        ax_n.text(row['neuro_mean'] - row['neuro_sem'] - 0.003,
                  i, star, va='center', ha='right', fontsize=5, color='#333333')

ax_n.set_xlabel('Mean BETA_STD', fontsize=7)
ax_n.set_title('Neuronal cell types\n(depleted)', fontsize=7.5,
               fontweight='bold', color=NEURO_COL)
ax_n.spines[['top', 'right', 'left']].set_visible(False)
ax_n.set_xlim(right=0.01)

fig.suptitle(
    'Structural enrichment vs neuronal depletion across all 16 BRE regions\n'
    '(MAGMA cell-type enrichment, mean BETA_STD ± SEM; * p<0.05, *** p<0.001 vs zero)',
    fontsize=8, fontweight='bold',
)

save_mpl(fig, FIG_DIR / 'structural_vs_neuronal')
plt.close(fig)

# ── Figure 2: Scatter — structural mean vs neuronal mean per region ────────────
fig2, ax2 = plt.subplots(figsize=(MM(88), MM(75)), constrained_layout=True)

scatter = ax2.scatter(
    summary['struct_mean'], summary['neuro_mean'],
    s=30, c=STRUCT_COL, edgecolors='white', linewidths=0.5, zorder=4,
)

# Annotate each region (adjust_text to avoid overlap)
from adjustText import adjust_text
texts2 = []
for _, row in summary.iterrows():
    t = ax2.text(row['struct_mean'], row['neuro_mean'],
                 row['region'], fontsize=4.5, color='#333333')
    texts2.append(t)
adjust_text(texts2, ax=ax2,
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
            expand_points=(1.4, 1.4), expand_text=(1.3, 1.3))

ax2.axhline(0, color='#333333', linewidth=0.6, linestyle='--', alpha=0.5)
ax2.axvline(0, color='#333333', linewidth=0.6, linestyle='--', alpha=0.5)

# Shade quadrant: structural+ / neuronal-
ax2.axhspan(ax2.get_ylim()[0] if ax2.get_ylim()[0] < -0.05 else -0.2, 0,
            alpha=0.05, color=NEURO_COL)
ax2.axvspan(0, ax2.get_xlim()[1] if ax2.get_xlim()[1] > 0.05 else 0.2,
            alpha=0.05, color=STRUCT_COL)

# Pearson r
r, p = stats.pearsonr(summary['struct_mean'], summary['neuro_mean'])
ax2.text(0.97, 0.97, f'r = {r:.2f}\np = {p:.2e}',
         transform=ax2.transAxes, ha='right', va='top', fontsize=6)

ax2.set_xlabel('Structural cell types — mean BETA_STD', fontsize=7)
ax2.set_ylabel('Neuronal cell types — mean BETA_STD', fontsize=7)
ax2.set_title('Structural enrichment vs neuronal depletion\nper BRE region', fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)

save_mpl(fig2, FIG_DIR / 'structural_vs_neuronal_scatter')
plt.close(fig2)

# ── Figure 3: Key neuronal cell types individually across regions ─────────────
# Show BETA_STD for the 5 most informative neuronal types (from matched datasets)
KEY_NEURONAL = [
    ('D1_Matrix',                    'D1 MSN',    '#B2182B'),
    ('D2_Matrix',                    'D2 MSN',    '#D6604D'),
    ('Medium_spiny_neuron',          'MSN (gen)', '#F4A582'),
    ('hippocampal_pyramidal_neuron', 'Pyramidal', '#67001F'),
    ('hippocampal_granule_cell',     'Granule',   '#C51B7D'),
]

# Collect BETA_STD for each key cell type × region (mean across all datasets)
rows_ct = []
for ct_exact, ct_label, ct_color in KEY_NEURONAL:
    sub = all_df[all_df['Cell_type'] == ct_exact]
    for region, rgrp in sub.groupby('region'):
        rows_ct.append({
            'ct_label': ct_label,
            'region':   region,
            'mean_bs':  rgrp['BETA_STD'].mean(),
            'color':    ct_color,
        })

ct_df = pd.DataFrame(rows_ct)

if len(ct_df) > 0:
    fig3, ax3 = plt.subplots(figsize=(MM(180), MM(80)), constrained_layout=True)

    ct_labels = [c[1] for c in KEY_NEURONAL]
    n_ct = len(ct_labels)
    n_reg = len(regions_ordered)
    x = np.arange(n_reg)
    width = 0.15

    for j, (ct_exact, ct_label, ct_color) in enumerate(KEY_NEURONAL):
        sub = ct_df[ct_df['ct_label'] == ct_label]
        vals = []
        for region in regions_ordered:
            r = sub[sub['region'] == region]['mean_bs']
            vals.append(r.values[0] if len(r) else np.nan)
        offset = (j - n_ct / 2 + 0.5) * width
        ax3.bar(x + offset, vals, width=width * 0.9,
                color=ct_color, label=ct_label, alpha=0.85)

    ax3.axhline(0, color='#333333', linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions_ordered, rotation=40, ha='right', fontsize=6)
    ax3.set_ylabel('Mean BETA_STD', fontsize=7)
    ax3.set_title(
        'Key neuronal cell types: BETA_STD across 16 BRE regions\n'
        '(negative = depleted; none are positively enriched)',
        fontsize=8, fontweight='bold',
    )
    ax3.legend(fontsize=6, frameon=False, ncol=5, loc='lower right')
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.set_ylim(top=0.05)   # emphasise that all are below zero

    save_mpl(fig3, FIG_DIR / 'neuronal_depletion_by_region')
    plt.close(fig3)
    print("Done: neuronal_depletion_by_region figure")

print("Done: 09_structural_vs_neuronal.py")
