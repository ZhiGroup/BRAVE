"""
17_matched_region_celltype.py
Matched-region structural cell-type enrichment test.

Question: For each JAGWAS BRE region, do structural cell types from the
ANATOMICALLY MATCHED scRNA-seq dataset show higher MAGMA enrichment (BETA_STD)
than structural cell types from non-matched brain-region datasets?

Logic:
  - FUMA tests ALL 42 brain-region scRNA-seq datasets for EVERY JAGWAS run.
  - Each dataset is from a SPECIFIC brain region (e.g. 52_Siletti_CerebralNuclei.Pu
    = Putamen, 62_Siletti_Hippocampus = Hippocampus).
  - For Putamen JAGWAS: compare structural BETA_STD of Putamen datasets vs
    all other datasets.
  - If matched > unmatched → region-specific structural signal.

Matched dataset assignments (anatomical prior):
  Accumbens  → Siletti.NAC
  Amygdala   → Siletti.BLN.BL/BM/La + Siletti.CEN (central nucleus)
  Caudate    → Siletti.CaB + Phan CaudateNucleus
  Hippocampus→ Siletti.Hippocampus (3 sub-regions) + Xu Hippocampus
  Pallidum   → Siletti.GP.Gpe/Gpi/CMN.CoA
  Putamen    → Siletti.Pu + Phan Putamen
  Thalamus   → Siletti.Thalamus.* (9 sub-nuclei)
  Brain Stem → Siletti.Myelencephalon + Siletti.Pons + Siletti.Midbrain.SN
  CSF        → no matched datasets (skip)

Output:
  results/celltype/matched_region_celltype.csv
  figures/celltype/matched_region_structural.pdf/.png
  figures/celltype/matched_vs_unmatched_delta.pdf/.png
"""

import sys
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
from fig_style import apply_mpl_style, save_mpl, MM

apply_mpl_style()

# ── Paths ──────────────────────────────────────────────────────────────────────
CELLTYPE_DIR = Path(cfg.postgwas.fuma_dir).parent / "FUMA_CellType"
BASE_DIR     = Path(__file__).parent.parent
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

# ── Matched datasets per anatomical group ─────────────────────────────────────
# Key: display label (or short name) → list of matched Dataset strings
MATCHED_DATASETS = {
    # Accumbens (NAC = Nucleus Accumbens)
    'accumbens': ['56_Siletti_CerebralNuclei.NAC_Human_2022_level2'],

    # Amygdala (BLN = Basolateral Amygdala nuclei; CEN = Central nucleus)
    'amygdala':  ['50_Siletti_CerebralNuclei.BLN.BL_Human_2022_level2',
                  '55_Siletti_CerebralNuclei.BLN.BM_Human_2022_level2',
                  '57_Siletti_CerebralNuclei.BLN.La_Human_2022_level2',
                  '45_Siletti_CerebralNuclei.CEN_Human_2022_level2'],

    # Caudate (CaB = Caudate/Basal ganglia)
    'caudate':   ['54_Siletti_CerebralNuclei.CaB_Human_2022_level2',
                  '427_Phan2024_Human_2024_CaudateNucleus_level2'],

    # Hippocampus
    'hippocampus': ['60_Siletti_Hippocampus.HiH.HiT.Sub_Human_2022_level2',
                    '62_Siletti_Hippocampus.HiH.CA1-3_Human_2022_level2',
                    '64_Siletti_Hippocampus.HiH.DG-CA4_Human_2022_level2',
                    '549_Xu_Human_2023_Hippocampus_level1'],

    # Pallidum (GP = Globus Pallidus: GPe external, GPi internal)
    'pallidum':  ['44_Siletti_CerebralNuclei.GP.Gpe_Human_2022_level2',
                  '53_Siletti_CerebralNuclei.GP.Gpi_Human_2022_level2',
                  '58_Siletti_CerebralNuclei.GP.CMN.CoA_Human_2022_level2'],

    # Putamen
    'putamen':   ['52_Siletti_CerebralNuclei.Pu_Human_2022_level2',
                  '428_Phan2024_Human_2024_Putamen_level2'],

    # Thalamus (multiple nuclei from Siletti)
    'thalamus':  ['98_Siletti_Thalamus.LNC.Pul_Human_2022_level2',
                  '99_Siletti_Thalamus.ANC_Human_2022_level2',
                  '100_Siletti_Thalamus.LNC.VLN_Human_2022_level2',
                  '102_Siletti_Thalamus.ILN.PILN.CM.Pf_Human_2022_level2',
                  '103_Siletti_Thalamus.ETH_Human_2022_level2',
                  '104_Siletti_Thalamus.MNC.MD_Human_2022_level2',
                  '105_Siletti_Thalamus.STH_Human_2022_level2',
                  '108_Siletti_Thalamus.MNC.MD.Re_Human_2022_level2',
                  '110_Siletti_Thalamus.LNC.VA_Human_2022_level2'],

    # Brain Stem (medulla + pons + midbrain substantia nigra)
    'brain_stem': ['87_Siletti_Myelencephalon.MoRF-MoEN_Human_2022_level2',
                   '90_Siletti_Pons.PnRF_Human_2022_level2',
                   '93_Siletti_Pons.XPnTg.DTg_Human_2022_level2',
                   '78_Siletti_Midbrain.SN_Human_2022_level2',
                   '84_Siletti_Midbrain.PAG-DR_Human_2022_level2'],
}

# Map each display region label → matched key
REGION_TO_MATCH_KEY = {
    'L. Accumbens':  'accumbens',
    'R. Accumbens':  'accumbens',
    'L. Amygdala':   'amygdala',
    'R. Amygdala':   'amygdala',
    'L. Caudate':    'caudate',
    'R. Caudate':    'caudate',
    'L. Hippocampus':'hippocampus',
    'R. Hippocampus':'hippocampus',
    'L. Pallidum':   'pallidum',
    'R. Pallidum':   'pallidum',
    'L. Putamen':    'putamen',
    'R. Putamen':    'putamen',
    'L. Thalamus':   'thalamus',
    'R. Thalamus':   'thalamus',
    'Brain Stem':    'brain_stem',
    'CSF':           None,   # no matched datasets
}

# ── Structural cell-type keywords ─────────────────────────────────────────────
STRUCTURAL_KW = [
    'Fibroblast', 'fibroblast', 'leptomeningeal',
    'Astrocyte', 'Astrocytes', 'Astro_', 'hippocampal_astrocyte', 'Bergmann',
    'Oligodendrocyte', 'Oligos', 'Committed_oligodendrocyte', 'oligodendrocyte',
    'OPC_', 'Olig_',
    'Microglia', 'microglial', 'MG_', 'macrophage', 'Macro_',
    'Vascular', 'Endothelial', 'endothelial', 'Endo_', 'Mural',
    'pericyte', 'smooth_muscle', 'vascular',
    'Ependymal', 'ependymal', 'Ependyma_', 'Choroid', 'choroid',
]


def is_structural(ct: str) -> bool:
    return any(kw in ct for kw in STRUCTURAL_KW)


# ── Load all regions ───────────────────────────────────────────────────────────
all_records = []
for folder, label in REGIONS.items():
    fpath = CELLTYPE_DIR / folder / 'magma_celltype_step1.txt'
    if not fpath.exists():
        print(f"[WARN] Missing: {fpath}")
        continue
    df = pd.read_csv(fpath, sep='\t')
    df['region']         = label
    df['is_structural']  = df['Cell_type'].apply(is_structural)
    match_key            = REGION_TO_MATCH_KEY.get(label)
    if match_key:
        matched_set = set(MATCHED_DATASETS[match_key])
        df['match_type'] = df['Dataset'].apply(
            lambda d: 'matched' if d in matched_set else 'unmatched'
        )
    else:
        df['match_type'] = 'unmatched'
    all_records.append(df)

all_df = pd.concat(all_records, ignore_index=True)
print(f"Loaded {len(all_df)} total rows from {len(all_records)} regions")

# Structural only
struct_df = all_df[all_df['is_structural']].copy()
print(f"Structural cell-type rows: {len(struct_df)}")

# ── Per-region: matched vs unmatched structural BETA_STD ─────────────────────
rows = []
for region, grp in struct_df.groupby('region'):
    match_key = REGION_TO_MATCH_KEY.get(region)
    matched   = grp[grp['match_type'] == 'matched']['BETA_STD'].values
    unmatched = grp[grp['match_type'] == 'unmatched']['BETA_STD'].values

    if match_key is None or len(matched) < 2:
        # CSF or no matched datasets: skip stats
        rows.append({
            'region': region,
            'match_key': match_key,
            'n_matched': len(matched),
            'n_unmatched': len(unmatched),
            'matched_mean': matched.mean() if len(matched) else np.nan,
            'unmatched_mean': unmatched.mean() if len(unmatched) else np.nan,
            'delta': np.nan,
            'pval_mwu': np.nan,
            'stat_mwu': np.nan,
        })
        continue

    # One-sided Mann-Whitney U: H1 matched > unmatched
    stat, pval = stats.mannwhitneyu(matched, unmatched, alternative='greater')

    rows.append({
        'region': region,
        'match_key': match_key,
        'n_matched': len(matched),
        'n_unmatched': len(unmatched),
        'matched_mean': matched.mean(),
        'unmatched_mean': unmatched.mean(),
        'delta': matched.mean() - unmatched.mean(),
        'pval_mwu': pval,
        'stat_mwu': stat,
    })

    print(f"{region:20s}  matched={matched.mean():.4f} (n={len(matched)})"
          f"  unmatched={unmatched.mean():.4f} (n={len(unmatched)})"
          f"  delta={matched.mean()-unmatched.mean():+.4f}  p={pval:.4f}")

summary = pd.DataFrame(rows)
summary = summary[summary['match_key'].notna()].reset_index(drop=True)  # drop CSF
summary = summary.sort_values('delta', ascending=False).reset_index(drop=True)
summary.to_csv(RES_DIR / 'matched_region_celltype.csv', index=False)
print(f"\nSaved: {RES_DIR}/matched_region_celltype.csv")

# ── FDR correction across regions ─────────────────────────────────────────────
from statsmodels.stats.multitest import multipletests
valid = summary['pval_mwu'].notna()
if valid.sum() > 0:
    _, fdr_q, _, _ = multipletests(summary.loc[valid, 'pval_mwu'], method='fdr_bh')
    summary.loc[valid, 'fdr_q'] = fdr_q
else:
    summary['fdr_q'] = np.nan

print("\n=== Matched-region structural enrichment test (Mann-Whitney U, one-sided) ===")
print(summary[['region','matched_mean','unmatched_mean','delta','pval_mwu','fdr_q']].to_string(index=False))

n_sig_nom = (summary['pval_mwu'] < 0.05).sum()
n_sig_fdr = (summary['fdr_q'] < 0.05).sum() if 'fdr_q' in summary else 0
print(f"\nNominal p<0.05: {n_sig_nom}/{len(summary)} regions")
print(f"FDR q<0.05:     {n_sig_fdr}/{len(summary)} regions")

# ── Figure 1: Delta (matched - unmatched) per region ─────────────────────────
fig, ax = plt.subplots(figsize=(MM(140), MM(110)), constrained_layout=True)

colors = []
for _, row in summary.iterrows():
    p = row['pval_mwu'] if not np.isnan(row['pval_mwu']) else 1.0
    q = row.get('fdr_q', 1.0)
    if not isinstance(q, float) or np.isnan(q):
        q = 1.0
    if q < 0.05:
        colors.append('#1A9850')   # FDR sig: green
    elif p < 0.05:
        colors.append('#4DAC26')   # nominal: light green
    else:
        colors.append('#AAAAAA')   # ns: grey

y_pos = np.arange(len(summary))
ax.barh(y_pos, summary['delta'], color=colors, height=0.65, edgecolor='white', linewidth=0.3)
ax.axvline(0, color='#333333', linewidth=0.8, linestyle='-')

# Significance markers
for i, (_, row) in enumerate(summary.iterrows()):
    p = row['pval_mwu']
    q = row.get('fdr_q', np.nan)
    if isinstance(q, float) and not np.isnan(q) and q < 0.05:
        marker = '**'
    elif not np.isnan(p) and p < 0.05:
        marker = '*'
    else:
        marker = ''
    if marker:
        x = row['delta']
        ax.text(x + 0.001, i, marker, ha='left', va='center',
                fontsize=5.5, color='#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(summary['region'], fontsize=7)
ax.set_xlabel('Δ BETA_STD (matched − unmatched structural)', fontsize=8)
ax.set_title('Hypothesis test: do anatomically matched scRNA-seq datasets\nshow higher structural enrichment than non-matched?',
             fontsize=9, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

legend_patches = [
    mpatches.Patch(color='#1A9850', label='FDR q<0.05'),
    mpatches.Patch(color='#4DAC26', label='p<0.05 (nominal)'),
    mpatches.Patch(color='#AAAAAA', label='n.s.'),
]
ax.legend(handles=legend_patches, fontsize=6.5, frameon=False,
          bbox_to_anchor=(0.98, 1.01), loc='upper right')

save_mpl(fig, FIG_DIR / 'matched_region_structural_delta')
plt.close(fig)
print("Saved: matched_region_structural_delta")

# ── Figure 2: Matched vs Unmatched side-by-side per region ────────────────────
fig2, ax2 = plt.subplots(figsize=(MM(160), MM(115)), constrained_layout=True)

y_pos  = np.arange(len(summary))
height = 0.35

ax2.barh(y_pos + height / 2, summary['matched_mean'],
         height=height, color='#2166AC', alpha=0.85, label='Matched region', edgecolor='white')
ax2.barh(y_pos - height / 2, summary['unmatched_mean'],
         height=height, color='#B2182B', alpha=0.60, label='Other brain regions', edgecolor='white')

ax2.axvline(0, color='#333333', linewidth=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(summary['region'], fontsize=7)
ax2.set_xlabel('Mean structural BETA_STD', fontsize=8)
ax2.set_title('Structural cell-type enrichment:\nmatched vs non-matched scRNA-seq datasets',
              fontsize=9, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.98, 1.01), loc='upper right')

# Significance connectors
for i, (_, row) in enumerate(summary.iterrows()):
    p = row['pval_mwu']
    q = row.get('fdr_q', np.nan)
    if isinstance(q, float) and not np.isnan(q) and q < 0.05:
        marker = '**'
    elif not np.isnan(p) and p < 0.05:
        marker = '*'
    else:
        marker = ''
    if marker:
        xmax = max(row['matched_mean'], row['unmatched_mean']) + 0.003
        ax2.text(xmax, i, marker, ha='left', va='center',
                 fontsize=5.5, color='#333333')

save_mpl(fig2, FIG_DIR / 'matched_region_structural_barh')
plt.close(fig2)
print("Saved: matched_region_structural_barh")

# ── Figure 3: Scatter — matched_mean vs unmatched_mean per region ─────────────
fig3, ax3 = plt.subplots(figsize=(MM(90), MM(85)), constrained_layout=True)

sc = ax3.scatter(summary['unmatched_mean'], summary['matched_mean'],
                 s=28, c='#2166AC', edgecolors='white', linewidths=0.5, zorder=4)

# Diagonal y=x line
xlim = (summary['unmatched_mean'].min() * 0.9, summary['unmatched_mean'].max() * 1.05)
ylim = (summary['matched_mean'].min() * 0.9, summary['matched_mean'].max() * 1.05)
lo = min(xlim[0], ylim[0])
hi = max(xlim[1], ylim[1])
ax3.plot([lo, hi], [lo, hi], color='#333333', linewidth=0.8, linestyle='--',
         alpha=0.6, label='y = x')
ax3.fill_between([lo, hi], [lo, hi], hi, color='#2166AC', alpha=0.04)

from adjustText import adjust_text
texts = []
for _, row in summary.iterrows():
    t = ax3.text(row['unmatched_mean'], row['matched_mean'],
                 row['region'], fontsize=4.5, color='#333333')
    texts.append(t)
adjust_text(texts, ax=ax3,
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4),
            expand_points=(1.4, 1.4), expand_text=(1.3, 1.3))

ax3.set_xlabel('Non-matched structural BETA_STD', fontsize=7)
ax3.set_ylabel('Matched-region structural BETA_STD', fontsize=7)
ax3.set_title('Points above dashed line = matched > non-matched', fontsize=7.5, fontweight='bold')
ax3.spines[['top', 'right']].set_visible(False)

# Count above diagonal
n_above = (summary['matched_mean'] > summary['unmatched_mean']).sum()
ax3.text(0.05, 0.97,
         f'{n_above}/{len(summary)} regions:\nmatched > non-matched',
         transform=ax3.transAxes, fontsize=6, va='top', color='#2166AC')

save_mpl(fig3, FIG_DIR / 'matched_region_structural_scatter')
plt.close(fig3)
print("Saved: matched_region_structural_scatter")

print("\nDone: 17_matched_region_celltype.py")
