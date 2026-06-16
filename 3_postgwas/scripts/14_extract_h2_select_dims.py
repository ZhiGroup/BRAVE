"""
14_extract_h2_select_dims.py
Extract per-dimension SNP heritability (h²) from FastGWA log files.

FastGWA computes h² for each dimension during REML. Log file template:
  discovery_<region>_QT<dim>.fastGWA.log
  Contains: "Heritability = <h2> (Pval = <pval>)"

Outputs:
  results/h2/fastgwa_h2_all_dims.csv        — all 128 × 16 h² estimates
  results/h2/fastgwa_h2_top_dims.csv        — top-K dims per region (by h², sig p)
  figures/h2/h2_heatmap.pdf/.png            — 16 regions × 128 dims h² heatmap
  figures/h2/h2_top_per_region.pdf/.png     — top dims highlighted per region

Selected top dims → input for LDSC genetic correlation with shape GWAS (Script 15).
"""

import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
LOG_DIR  = Path(cfg.postgwas.bre_munged_dir)
PREFIX   = "discovery"

RES_DIR  = Path(__file__).parents[1] / "results" / "h2"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "h2"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_DIMS   = 128   # BRE dimensions (QT0 … QT127)
TOP_K    = 10    # top dimensions to select per region for LDSC

# ── REGION LIST (JAGWAS folder names) ─────────────────────────────────────────
REGIONS = [
    'Brain_Stem_or_4th_Ventricle',
    'CSF',
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
]

# Display names for plotting
DISP = {
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
    'Right_Thalamus-Proper':        'R. Thalamus',
}

H2_PATTERN = re.compile(
    r'Heritability\s*=\s*([\d.eE+\-]+)\s*\(Pval\s*=\s*([\d.eE+\-]+)\)')

# ── EXTRACT H² FROM LOG FILES ─────────────────────────────────────────────────
print(f"Extracting h² from log files ({len(REGIONS)} regions × {N_DIMS} dims) ...")
rows = []
missing = 0

for region in REGIONS:
    disp = DISP[region]
    for dim in range(N_DIMS):
        log_path = LOG_DIR / f"{PREFIX}_{region}_QT{dim}.fastGWA.log"
        h2, pval = np.nan, np.nan
        if log_path.exists():
            txt = log_path.read_text()
            m = H2_PATTERN.search(txt)
            if m:
                h2   = float(m.group(1))
                pval = float(m.group(2))
        else:
            missing += 1
        rows.append(dict(region=region, display=disp, dim=dim, h2=h2, pval=pval))

df = pd.DataFrame(rows)
print(f"  Extracted {df['h2'].notna().sum()} valid h² values, {missing} missing log files")

# Clip negative h² to 0 (REML can occasionally return slightly negative values)
df['h2_clipped'] = df['h2'].clip(lower=0)

df.to_csv(RES_DIR / 'fastgwa_h2_all_dims.csv', index=False)
print(f"Saved: {RES_DIR}/fastgwa_h2_all_dims.csv")

# ── SUMMARY STATS ─────────────────────────────────────────────────────────────
print("\n=== Mean h² per region (across 128 dims) ===")
summary = df.groupby(['region', 'display'])['h2_clipped'].agg(
    mean_h2='mean', max_h2='max', n_sig=lambda x: (df.loc[x.index, 'pval'] < 0.05).sum()
).reset_index()
summary = summary.sort_values('mean_h2', ascending=False)
for _, row in summary.iterrows():
    print(f"  {row['display']:<25} mean h²={row['mean_h2']:.4f}  "
          f"max h²={row['max_h2']:.4f}  n_sig(p<0.05)={int(row['n_sig'])}")

# ── SELECT TOP-K DIMS PER REGION ──────────────────────────────────────────────
print(f"\nSelecting top-{TOP_K} dims per region (by h², pval < 0.05 preferred) ...")
top_rows = []

for region in REGIONS:
    sub = df[df['region'] == region].copy()
    sub = sub.dropna(subset=['h2'])
    # Sort: significant first (pval<0.05), then by h2 descending
    sub['sig'] = (sub['pval'] < 0.05).astype(int)
    sub = sub.sort_values(['sig', 'h2_clipped'], ascending=[False, False])
    top = sub.head(TOP_K)
    top_rows.append(top)
    print(f"  {DISP[region]:<25} top dim=QT{top['dim'].iloc[0]} "
          f"h²={top['h2_clipped'].iloc[0]:.4f} (p={top['pval'].iloc[0]:.4f})")

top_df = pd.concat(top_rows, ignore_index=True)
top_df.to_csv(RES_DIR / 'fastgwa_h2_top_dims.csv', index=False)
print(f"\nSaved: {RES_DIR}/fastgwa_h2_top_dims.csv")

# ── FIGURE 1: H² HEATMAP (16 regions × 128 dims) ─────────────────────────────
print("\nGenerating h² heatmap ...")

pivot = df.pivot(index='display', columns='dim', values='h2_clipped')
# Sort rows by mean h2
region_order = summary.sort_values('mean_h2', ascending=False)['display'].tolist()
pivot = pivot.reindex(region_order)

mat = pivot.values
vmax = np.nanpercentile(mat, 95)   # clip at 95th percentile to avoid outlier dominance

CMAP = LinearSegmentedColormap.from_list('h2', ['#FFFFFF', '#2166AC'])
fig, ax = plt.subplots(figsize=(MM(180), MM(95)))
im = ax.imshow(mat, cmap=CMAP, aspect='auto', vmin=0, vmax=vmax)

ax.set_yticks(range(len(region_order)))
ax.set_yticklabels(region_order, fontsize=FS_TICK)
ax.set_xlabel('BRE Dimension (QT)', fontsize=FS_LABEL)
ax.set_title('Per-dimension SNP heritability h² across 16 regions',
             fontsize=FS_LABEL, fontweight='bold')

# Mark top dim per region with a red dot
for i, region_disp in enumerate(region_order):
    region_folder = {v: k for k, v in DISP.items()}[region_disp]
    sub = df[(df['region'] == region_folder) & df['h2'].notna()]
    if len(sub):
        best_dim = sub.loc[sub['h2_clipped'].idxmax(), 'dim']
        ax.plot(best_dim, i, 'r.', markersize=4, zorder=5)

# X-axis ticks every 16 dims
ax.set_xticks(range(0, N_DIMS, 16))
ax.set_xticklabels([f'QT{i}' for i in range(0, N_DIMS, 16)], fontsize=FS_TICK)

cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
cbar.set_label('h² (SNP heritability)', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker='.', color='r', linestyle='None',
                          markersize=6, label='Top dim per region')],
          fontsize=FS_TICK, frameon=False, loc='upper right')

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'h2_heatmap')
plt.close()
print("Saved: figures/h2/h2_heatmap.pdf/.png")

# ── FIGURE 2: TOP h² PER REGION BARPLOT ───────────────────────────────────────
print("Generating top-h² per region barplot ...")

# Best dim per region
best_per_region = df.loc[df.groupby('region')['h2_clipped'].idxmax()].copy()
best_per_region = best_per_region.set_index('display').reindex(region_order)

fig, ax = plt.subplots(figsize=(MM(120), MM(95)))
y = np.arange(len(region_order))
colors = ['#CB181D' if p < 0.05 else '#4292C6'
          for p in best_per_region['pval'].values]

bars = ax.barh(y, best_per_region['h2_clipped'].values,
               color=colors, height=0.7, edgecolor='none')

for i, (_, row) in enumerate(best_per_region.iterrows()):
    ax.text(row['h2_clipped'] + 0.005, i,
            f"QT{int(row['dim'])} (p={row['pval']:.3f})",
            va='center', fontsize=FS_TICK - 1, color='#333333')

ax.set_yticks(y)
ax.set_yticklabels(region_order, fontsize=FS_TICK)
ax.set_xlabel('h² of best dimension', fontsize=FS_LABEL)
ax.set_title('Highest h² dimension per region\n(red = p<0.05, blue = p≥0.05)',
             fontsize=FS_LABEL, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
save_mpl(fig, FIG_DIR / 'h2_top_per_region')
plt.close()
print("Saved: figures/h2/h2_top_per_region.pdf/.png")

# ── PRINT TOP DIMS FOR LDSC (MATCHED TO SHAPE REGIONS) ────────────────────────
print("""
=== Top dims for LDSC genetic correlation with shape GWAS ===
(Shape GWAS regions: accu, amyg, caud, hipp, pall, puta, thal)
""")
shape_matched = {
    'Left_Accumbens-area':  'accu',
    'Right_Accumbens-area': 'accu',
    'Left_Amygdala':        'amyg',
    'Right_Amygdala':       'amyg',
    'Left_Caudate':         'caud',
    'Right_Caudate':        'caud',
    'Left_Hippocampus':     'hipp',
    'Right_Hippocampus':    'hipp',
    'Left_Pallidum':        'pall',
    'Right_Pallidum':       'pall',
    'Left_Putamen':         'puta',
    'Right_Putamen':        'puta',
    'Left_Thalamus_Proper': 'thal',
    'Right_Thalamus-Proper':'thal',
}
print(f"{'Region':<25} {'Shape':<6} {'Top dim':<8} {'h²':<8} {'pval'}")
print("-" * 65)
for region, shape_region in shape_matched.items():
    row = best_per_region.loc[DISP[region]] if DISP[region] in best_per_region.index else None
    if row is not None:
        print(f"  {DISP[region]:<23} {shape_region:<6} "
              f"QT{int(row['dim']):<6} {row['h2_clipped']:.4f}   {row['pval']:.4f}")

print(f"\nTop-{TOP_K} dims per region saved to: {RES_DIR}/fastgwa_h2_top_dims.csv")
print("Use these dims as input to Script 15 (LDSC genetic correlation with shape).")
