"""
11_ci_mapping_analysis.py
Chromatin interaction (CI) mapping analysis — two panels:

(A) CI tissue heatmap (16 regions × CI tissue sources)
    ciMapts column in genes.txt: colon-separated tissue/loop labels.
    Brain: Adult_Cortex, Fetal_Cortex, Dorsolateral_Prefrontal_Cortex, Hippocampus
    Cardiac: Left_Ventricle, Right_Ventricle
    Loop type: Promoter_anchored_loops, EP_links_oneway
    Shows which regulatory contexts link our loci to genes — brain CI confirms
    brain-specific regulatory loops; cardiac CI mirrors cardiovascular pleiotropy (Script 05).

(B) Gene mapping method proportions (16 regions × mapping method)
    For each region: fraction of mapped genes from positional only, eQTL only,
    CI only, and pairwise/triple overlaps. Stacked barplot.
    Shows the regulatory architecture of our loci — whether they act primarily
    through coding proximity or distal regulatory loops.

Outputs:
  results/cross_region/ci_tissue_counts.csv
  results/cross_region/gene_mapping_proportions.csv
  figures/cross_region/ci_tissue_heatmap.pdf/.png
  figures/cross_region/gene_mapping_proportions.pdf/.png

Supports paper Claim 3: loci act through brain-specific regulatory loops
  (CI tissue) and predominantly through distal regulatory mechanisms (CI > positional).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
RES_DIR  = Path(__file__).parents[1] / "results" / "cross_region"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "cross_region"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── REGION ORDER ──────────────────────────────────────────────────────────────
BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def parse_region(folder):
    n = folder.strip(); nl = n.lower()
    if nl.startswith('left_'):  side, base = 'Left', n[5:]
    elif nl.startswith('right_'): side, base = 'Right', n[6:]
    else: side, base = 'Midline', n
    base = base.replace('Thalamus-Proper', 'Thalamus_Proper')
    return side, base

def display_name(side, base):
    short = base.replace('_', ' ').replace('-area', '').replace('Thalamus Proper', 'Thalamus')
    prefix = {'Left': 'L.', 'Right': 'R.', 'Midline': ''}[side]
    name = f"{prefix} {short}".strip()
    name = name.replace('Brain Stem or 4th Ventricle', 'Brain Stem / 4th V.')
    return name

def sort_key(side, base):
    if base in BASE_ORDER:
        return (BASE_ORDER.index(base), {'Left': 0, 'Right': 1}.get(side, 2))
    elif base in MIDLINE:
        return (len(BASE_ORDER) + MIDLINE.index(base), 0)
    return (99, 0)

# ── CI TISSUE DISPLAY NAMES & GROUPING ───────────────────────────────────────
CI_DISPLAY = {
    'Adult_Cortex':                     'Adult Cortex',
    'Fetal_Cortex':                     'Fetal Cortex',
    'Dorsolateral_Prefrontal_Cortex':   'DLPFC',
    'Hippocampus':                      'Hippocampus',
    'Left_Ventricle':                   'Left Ventricle',
    'Right_Ventricle':                  'Right Ventricle',
    'Promoter_anchored_loops':          'Promoter loops',
    'EP_links_oneway':                  'E–P links',
}
BRAIN_CI    = {'Adult_Cortex', 'Fetal_Cortex', 'Dorsolateral_Prefrontal_Cortex', 'Hippocampus'}
CARDIAC_CI  = {'Left_Ventricle', 'Right_Ventricle'}
LOOP_CI     = {'Promoter_anchored_loops', 'EP_links_oneway'}

# Ordered columns for heatmap (brain first, then cardiac, then loop type)
CI_ORDER = ['Adult_Cortex', 'Fetal_Cortex', 'Dorsolateral_Prefrontal_Cortex',
            'Hippocampus', 'Left_Ventricle', 'Right_Ventricle',
            'Promoter_anchored_loops', 'EP_links_oneway']

# ── LOAD GENES.TXT PER REGION ─────────────────────────────────────────────────
print("Loading genes.txt for all regions ...")

region_dirs = sorted(FUMA_DIR.iterdir(),
                     key=lambda d: sort_key(*parse_region(d.name)) if d.is_dir() else (99,0))

ci_rows     = []   # for heatmap A
prop_rows   = []   # for stacked barplot B
region_order = []

for region_dir in region_dirs:
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    gene_file = region_dir / 'genes.txt'
    if not gene_file.exists():
        continue

    side, base = parse_region(folder)
    disp = display_name(side, base)
    region_order.append(disp)

    df = pd.read_csv(gene_file, sep='\t', low_memory=False)
    df['posMapSNPs']  = pd.to_numeric(df['posMapSNPs'],  errors='coerce').fillna(0)
    df['eqtlMapSNPs'] = pd.to_numeric(df['eqtlMapSNPs'], errors='coerce').fillna(0)
    is_pos  = df['posMapSNPs']  > 0
    is_eqtl = df['eqtlMapSNPs'] > 0
    is_ci   = df['ciMap'].astype(str).str.strip().str.lower() != 'no' \
              if 'ciMap' in df.columns else pd.Series([False]*len(df))

    n_total = len(df)

    # ── (A) CI tissue counts ──────────────────────────────────────────────────
    ci_df = df[is_ci & df['ciMapts'].notna() & (df['ciMapts'].astype(str) != 'NA')].copy()
    tissue_counts = {t: 0 for t in CI_ORDER}
    for _, row in ci_df.iterrows():
        tissues = str(row['ciMapts']).split(':')
        for t in tissues:
            t = t.strip()
            if t in tissue_counts:
                tissue_counts[t] += 1  # count gene-tissue pairs
    tissue_counts['folder']  = folder
    tissue_counts['display'] = disp
    ci_rows.append(tissue_counts)

    # ── (B) Mapping method proportions ───────────────────────────────────────
    # Categories: pos_only, eqtl_only, ci_only, pos+eqtl, pos+ci, eqtl+ci, all3, none
    pos_only    = ( is_pos & ~is_eqtl & ~is_ci).sum()
    eqtl_only   = (~is_pos &  is_eqtl & ~is_ci).sum()
    ci_only     = (~is_pos & ~is_eqtl &  is_ci).sum()
    pos_eqtl    = ( is_pos &  is_eqtl & ~is_ci).sum()
    pos_ci      = ( is_pos & ~is_eqtl &  is_ci).sum()
    eqtl_ci     = (~is_pos &  is_eqtl &  is_ci).sum()
    all3        = ( is_pos &  is_eqtl &  is_ci).sum()
    none_mapped = (~is_pos & ~is_eqtl & ~is_ci).sum()

    prop_rows.append(dict(
        display=disp, n_total=n_total,
        pos_only=pos_only, eqtl_only=eqtl_only, ci_only=ci_only,
        pos_eqtl=pos_eqtl, pos_ci=pos_ci, eqtl_ci=eqtl_ci,
        all3=all3, none=none_mapped,
    ))
    print(f"  {folder}: CI tissues={len(ci_df)}, pos={is_pos.sum()}, "
          f"eqtl={is_eqtl.sum()}, ci={is_ci.sum()}, total={n_total}")

# ── SAVE CSVs ─────────────────────────────────────────────────────────────────
ci_df_out   = pd.DataFrame(ci_rows)
prop_df_out = pd.DataFrame(prop_rows)

ci_df_out.to_csv(RES_DIR / 'ci_tissue_counts.csv', index=False)
prop_df_out.to_csv(RES_DIR / 'gene_mapping_proportions.csv', index=False)
print(f"\nSaved: {RES_DIR}/ci_tissue_counts.csv")
print(f"Saved: {RES_DIR}/gene_mapping_proportions.csv")

# ── (A) CI TISSUE HEATMAP ─────────────────────────────────────────────────────
print("\nGenerating CI tissue heatmap ...")

ci_pivot = ci_df_out.set_index('display')[CI_ORDER]
ci_pivot = ci_pivot.loc[region_order]   # anatomical order
mat = ci_pivot.values.astype(float)

col_labels = [CI_DISPLAY[c] for c in CI_ORDER]

# Colour breaks: brain = Blues, cardiac = Reds, loops = Greens
# Use a single sequential colourmap; mark group boundaries with lines
CMAP = LinearSegmentedColormap.from_list('ci', ['#FFFFFF', '#08519C'])
vmax = mat.max()

fig, ax = plt.subplots(figsize=(MM(145), MM(110)))
im = ax.imshow(mat, cmap=CMAP, aspect='auto', vmin=0, vmax=vmax)

# Annotate
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = int(mat[i, j])
        col = 'white' if val > vmax * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=6, color=col)

ax.set_xticks(range(len(CI_ORDER)))
ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=FS_TICK)
ax.set_yticks(range(len(region_order)))
ax.set_yticklabels(region_order, fontsize=FS_TICK)
ax.set_title('Chromatin interaction tissue source — genes per region',
             fontsize=FS_LABEL, fontweight='bold')

# Group boundary lines (after brain col 3, after cardiac col 5)
for x in [3.5, 5.5]:
    ax.axvline(x, color='#666666', lw=1.2, ls='--')

# Group labels on top
ax.annotate('Brain', xy=(1.5, -0.7), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=FS_TICK, color='#08519C', fontweight='bold')
ax.annotate('Cardiac', xy=(4.5, -0.7), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=FS_TICK, color='#CB181D', fontweight='bold')
ax.annotate('Loop type', xy=(7, -0.7), xycoords=('data', 'axes fraction'),
            ha='center', fontsize=FS_TICK, color='#238B45', fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Gene–tissue CI pairs', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'ci_tissue_heatmap')
plt.close()
print("Saved: figures/cross_region/ci_tissue_heatmap.pdf/.png")

# ── (B) GENE MAPPING PROPORTIONS STACKED BARPLOT ─────────────────────────────
print("Generating gene mapping proportions barplot ...")

METHODS = ['pos_only', 'eqtl_only', 'ci_only',
           'pos_eqtl', 'pos_ci', 'eqtl_ci', 'all3', 'none']
METHOD_LABELS = ['Positional only', 'eQTL only', 'CI only',
                 'Pos + eQTL', 'Pos + CI', 'eQTL + CI', 'All three', 'None']
METHOD_COLORS = ['#4292C6', '#74C476', '#FD8D3C',
                 '#2171B5', '#238B45', '#A63603', '#6A0572', '#CCCCCC']

prop_df_out = prop_df_out.set_index('display').loc[region_order]
totals = prop_df_out['n_total'].values

# Convert to fractions
frac = prop_df_out[METHODS].div(prop_df_out['n_total'], axis=0)

fig, ax = plt.subplots(figsize=(MM(160), MM(100)))
y = np.arange(len(region_order))
lefts = np.zeros(len(region_order))

bars_list = []
for method, label, color in zip(METHODS, METHOD_LABELS, METHOD_COLORS):
    vals = frac[method].values
    bars = ax.barh(y, vals, left=lefts, color=color, height=0.7,
                   edgecolor='none', label=label)
    bars_list.append(bars)
    lefts += vals

ax.set_yticks(y)
ax.set_yticklabels(region_order, fontsize=FS_TICK)
ax.set_xlabel('Fraction of mapped genes', fontsize=FS_LABEL)
ax.set_title('Gene mapping method composition per region', fontsize=FS_LABEL,
             fontweight='bold')
ax.set_xlim(0, 1)
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()

# Annotate n_total on right
for i, (region, row) in enumerate(prop_df_out.iterrows()):
    ax.text(1.01, i, f"n={int(row['n_total'])}", va='center', ha='left',
            fontsize=FS_TICK - 1)

ax.legend(loc='lower right', frameon=False, fontsize=FS_TICK - 0.5,
          ncol=2, bbox_to_anchor=(1.25, 0))

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'gene_mapping_proportions')
plt.close()
print("Saved: figures/cross_region/gene_mapping_proportions.pdf/.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n=== CI tissue totals across all regions ===")
ci_totals = ci_df_out[CI_ORDER].sum()
for t, n in ci_totals.items():
    print(f"  {CI_DISPLAY[t]:<30} {int(n):5d}")

print("\n=== Average gene mapping proportions across regions ===")
avg_frac = frac.mean()
for m, l in zip(METHODS, METHOD_LABELS):
    print(f"  {l:<25} {avg_frac[m]*100:.1f}%")
