"""
16_magma_gobp_enrichment.py
MAGMA Gene Ontology Biological Process (GOBP) enrichment across all 16 regions.

Inputs: magma.gsa.out per FUMA region (GOBP_* gene-set rows)
Analysis:
  - Filter to GOBP_* sets (~7,744 terms per region)
  - Per-region BH FDR correction
  - Classify terms: Structural/ECM/Glial | Neuronal/CNS | Morphogenesis | Other
  - Cross-regional shared terms (FDR-sig in >= MIN_REGIONS regions)

Figures:
  figures/geneset/gobp_heatmap.pdf/.png         — top shared terms × 16 regions heatmap
  figures/geneset/gobp_per_region_panel.pdf/.png — 4×4 panel, top terms per region

Claim 1: Structural/ECM enrichment → BREs capture structural, not functional, variation
Claim 3: Region-specific term patterns → biological specificity of individual regions
"""

import sys
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── Paths ──────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
RES_DIR  = Path(__file__).parents[1] / "results"  / "geneset"
FIG_DIR  = Path(__file__).parents[1] / "figures"  / "geneset"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Regions ────────────────────────────────────────────────────────────────────
REGIONS = {
    'Brain_Stem_or_4th_Ventricle': 'Brain Stem',
    'CSF':                          'CSF',
    'Left_Accumbens-area':          'L. Accumbens',
    'Right_Accumbens-area':         'R. Accumbens',
    'Left_Amygdala':                'L. Amygdala',
    'Right_Amygdala':               'R. Amygdala',
    'Left_Caudate':                 'L. Caudate',
    'Right_Caudate':                'R. Caudate',
    'Left_Hippocampus':             'L. Hippocampus',
    'Right_Hippocampus':            'R. Hippocampus',
    'Left_Pallidum':                'L. Pallidum',
    'Right_Pallidum':               'R. Pallidum',
    'Left_Putamen':                 'L. Putamen',
    'Right_Putamen':                'R. Putamen',
    'Left_Thalamus_Proper':         'L. Thalamus',
    'Right_Thalamus-Proper':        'R. Thalamus',
}

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_REGIONS  = 3     # min regions FDR-sig to include in cross-region heatmap
TOP_HEATMAP  = 30    # top terms for heatmap (sorted by breadth then mean -log10p)
TOP_PER_REG  = 8     # top terms shown per region in per-region panel
FDR_THRESH   = 0.05
MAX_LABEL_LEN = 55   # max chars for y-axis labels (word-truncate, no ellipsis)

# ── Term categorisation ────────────────────────────────────────────────────────
CAT_STRUCTURAL = 'Structural/ECM/Glial'
CAT_NEURONAL   = 'Neuronal/CNS dev.'
CAT_MORPHO     = 'Morphogenesis/Dev.'
CAT_OTHER      = 'Other'

CAT_COLORS = {
    CAT_STRUCTURAL: '#E6550D',   # orange-red
    CAT_NEURONAL:   '#3182BD',   # blue
    CAT_MORPHO:     '#756BB1',   # purple
    CAT_OTHER:      '#969696',   # grey
}

# Keywords are matched against normalised full name (UPPER, spaces→underscores).
# Matching is substring, so use stems where needed.
# Priority: STRUCTURAL > NEURONAL > MORPHOGENESIS > OTHER
STRUCT_KW = [
    # Mesenchyme / connective tissue
    'MESENCHYM',           # MESENCHYME, MESENCHYMAL, MESENCHYMAL_CELL
    'CHONDROCYTE', 'CHONDROGENESIS', 'CHONDRAL', 'CARTILAGE',
    'FIBROBLAST', 'COLLAGEN', 'EXTRACELLULAR_MATRIX',
    'CONNECTIVE_TISSUE',
    # Vascular / endothelial
    'VASCUL',              # VASCULAR, VASCULATURE, VASCULOGENESIS
    'ANGIOGENESIS', 'ENDOTHELI',
    'ARTERY', 'ARTERIOLE', 'BLOOD_VESSEL', 'VENOUS',
    # Bone / skeletal
    'OSSIF', 'OSTEOB', 'OSTEOCYTE', 'OSTEOCLAST',
    'SKELETAL', 'BONE_', '_BONE',
    # Glial / myelin
    'GLIAL', 'OLIGODENDROCYTE', 'ASTROCYTE', 'MYELINATION',
    # Tooth / odontogenesis
    'ODONTOGEN',
    # Smooth muscle
    'SMOOTH_MUSCLE_CELL',
]

NEURO_KW = [
    'NEURON', 'NEUROGENESIS', 'NEURAL_CREST',
    'CENTRAL_NERVOUS_SYSTEM', 'NERVOUS_SYSTEM_DEVELOPMENT',
    'SYNAP', 'AXON', 'DENDRIT', 'NEURONAL',
    'GENERATION_OF_NEURONS', 'NEUROTRANSMIT', 'NEUROPIL',
    'NEURAL_TUBE',
]

MORPHO_KW = [
    # Morphogenesis (general)
    'MORPHOGENESIS',
    # Differentiation / development
    'CELL_DIFFERENTIATION', 'CELL_DEVELOPMENT',
    'ORGAN_DEVELOPMENT', 'ORGANOGENESIS',
    # Migration / projection / motility
    'CELL_MIGRATION', 'CELL_PROJECTION', 'AMEBOIDAL',
    # Cytoskeletal / component organisation
    'CYTOSKELETON', 'CYTOSKELETAL', 'CELLULAR_COMPONENT_ORGANIZATION',
    # Embryonic
    'EMBRYONIC', 'EMBRYO_DEVELOPMENT',
    # Anatomical structure
    'ANATOMICAL',
    # Stem cell
    'STEM_CELL',
    # Appendage / limb
    'APPENDAGE', 'LIMB_',
    # Tube
    'TUBE_DEVELOPMENT', 'TUBE_',
    # Valve / heart development
    'VALVE', 'SEPTUM',
    # Urogenital / pulmonary organ development
    'URETER', 'UROGENITAL', 'PULMONARY',
]


def categorise(full_name, variable):
    # type: (str, str) -> str
    """Classify a GO term using FULL_NAME (preferred) or VARIABLE (fallback).
    Normalise: UPPER, spaces and hyphens → underscores."""
    fn = str(full_name).strip()
    if fn and fn.lower() not in ('nan', ''):
        t = fn.upper().replace(' ', '_').replace('-', '_')
    else:
        # Fallback: strip GOBP_ prefix and trailing '...' from truncated VARIABLE
        v = re.sub(r'^GOBP_', '', str(variable))
        v = re.sub(r'\.\.\.$', '', v)
        t = v.upper()
    if any(k in t for k in STRUCT_KW):
        return CAT_STRUCTURAL
    elif any(k in t for k in NEURO_KW):
        return CAT_NEURONAL
    elif any(k in t for k in MORPHO_KW):
        return CAT_MORPHO
    else:
        return CAT_OTHER


def make_label(full_name, variable, max_len=MAX_LABEL_LEN):
    # type: (str, str, int) -> str
    """Return a clean display label from the GOBP identifier.
    FUMA magma.gsa.out: VARIABLE is truncated (GOBP_..28chars...N),
    FULL_NAME (col 8) is the full untruncated identifier.
    Use FULL_NAME when available; fall back to stripped VARIABLE.
    Word-truncates at max_len chars — no trailing ellipsis."""
    fn = str(full_name).strip()
    if fn and fn.lower() not in ('nan', ''):
        base = fn   # FULL_NAME: full untruncated GOBP_... identifier
    else:
        base = re.sub(r'\.\.\.\d*$', '', str(variable))  # strip ...N artefact
    v = re.sub(r'^GOBP_', '', base)
    label = v.replace('_', ' ').lower().capitalize()
    if len(label) > max_len:
        label = label[:max_len].rsplit(' ', 1)[0]
    return label


# ── STEP 1: Parse all regions ──────────────────────────────────────────────────
def parse_gsa(path):
    # type: (Path) -> pd.DataFrame
    """Parse magma.gsa.out line-by-line.
    Splits each line into max 8 tokens (col 7+ joined as FULL_NAME).
    This avoids misalignment when FULL_NAME contains spaces."""
    rows = []
    with open(str(path), 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7 or parts[0] == 'VARIABLE':
                continue
            var = parts[0]
            if not var.startswith('GOBP_'):
                continue
            try:
                p_val = float(parts[6])
            except (ValueError, IndexError):
                continue
            full_name = ' '.join(parts[7:]) if len(parts) > 7 else ''
            rows.append({'VARIABLE': var, 'P': p_val, 'FULL_NAME': full_name})
    return pd.DataFrame(rows)


print("Parsing magma.gsa.out for all 16 regions ...")
all_rows = []

for folder, display in REGIONS.items():
    gsa = FUMA_DIR / folder / 'magma.gsa.out'
    if not gsa.exists():
        print("  MISSING: {}".format(folder))
        continue

    gobp = parse_gsa(gsa)
    if gobp.empty:
        continue
    gobp['P'] = pd.to_numeric(gobp['P'], errors='coerce')
    gobp = gobp.dropna(subset=['P'])

    # Per-region FDR
    _, fdr_q, _, _ = multipletests(gobp['P'].values, method='fdr_bh')
    gobp['fdr_q']   = fdr_q
    gobp['fdr_sig'] = fdr_q < FDR_THRESH
    gobp['region']  = display
    gobp['log10p']  = -np.log10(gobp['P'].clip(lower=1e-300))
    gobp['category'] = gobp.apply(
        lambda r: categorise(r['FULL_NAME'], r['VARIABLE']), axis=1)
    gobp['label'] = gobp.apply(
        lambda r: make_label(r['FULL_NAME'], r['VARIABLE']), axis=1)

    all_rows.append(gobp[['VARIABLE','FULL_NAME','P','fdr_q','fdr_sig',
                           'log10p','region','category','label']].copy())
    n_sig = gobp['fdr_sig'].sum()
    print("  {:<18} {:>5} GOBP terms, {:>4} FDR-sig".format(
        display, len(gobp), n_sig))

long_df = pd.concat(all_rows, ignore_index=True)
long_df.to_csv(RES_DIR / 'gobp_all_regions.csv', index=False)
print("\nSaved: {}  ({} rows)".format(RES_DIR / 'gobp_all_regions.csv', len(long_df)))


# ── STEP 2: Cross-regional summary ────────────────────────────────────────────
print("\nIdentifying cross-regional enriched terms ...")

cross = long_df.groupby('VARIABLE').apply(lambda g: pd.Series({
    'label':        g['label'].iloc[0],
    'full_name':    g['FULL_NAME'].iloc[0],
    'category':     g['category'].iloc[0],
    'n_sig':        g['fdr_sig'].sum(),
    'n_regions':    len(g),
    'mean_log10p':  g['log10p'].mean(),
    'max_log10p':   g['log10p'].max(),
    'min_fdr_q':    g['fdr_q'].min(),
})).reset_index()

cross = cross.sort_values(['n_sig', 'mean_log10p'], ascending=False)
cross_broad = cross[cross['n_sig'] >= MIN_REGIONS].reset_index(drop=True)
cross_broad.to_csv(RES_DIR / 'gobp_cross_region_top.csv', index=False)

print("Terms FDR-sig in >= {} regions: {}".format(MIN_REGIONS, len(cross_broad)))
print("\nTop 20 by breadth:")
print(cross_broad[['label','category','n_sig','mean_log10p']].head(20).to_string(index=False))


# ── STEP 3: Figure A — Cross-region heatmap ───────────────────────────────────
print("\nGenerating cross-region heatmap ...")

top_terms = cross_broad.head(TOP_HEATMAP)['VARIABLE'].tolist()
region_order = list(REGIONS.values())

# Build matrices
mat_log10p = np.full((len(top_terms), len(region_order)), 0.0)
mat_fdr    = np.full((len(top_terms), len(region_order)), 1.0)

for i, term in enumerate(top_terms):
    sub = long_df[long_df['VARIABLE'] == term]
    for j, reg in enumerate(region_order):
        row = sub[sub['region'] == reg]
        if len(row):
            mat_log10p[i, j] = row['log10p'].values[0]
            mat_fdr[i, j]    = row['fdr_q'].values[0]

# Labels and metadata
term_labels   = [long_df[long_df['VARIABLE'] == t]['label'].iloc[0] for t in top_terms]
term_cats     = [long_df[long_df['VARIABLE'] == t]['category'].iloc[0] for t in top_terms]
term_n_sig    = [cross_broad[cross_broad['VARIABLE'] == t]['n_sig'].values[0]
                 for t in top_terms]

fig_h = MM(165)
fig_w = MM(220)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

vmax = np.percentile(mat_log10p[mat_log10p > 0], 95) if mat_log10p.max() > 0 else 10
im = ax.imshow(mat_log10p, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')

# FDR-sig asterisks
for i in range(mat_fdr.shape[0]):
    for j in range(mat_fdr.shape[1]):
        if mat_fdr[i, j] < FDR_THRESH:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=6, color='black', fontweight='bold')

# y-axis: term labels coloured by category, with n_sig appended
yticklabels = ['{} ({}/16)'.format(lbl, int(n))
               for lbl, n in zip(term_labels, term_n_sig)]
ax.set_yticks(range(len(top_terms)))
ax.set_yticklabels(yticklabels, fontsize=6)
for tick, cat in zip(ax.get_yticklabels(), term_cats):
    tick.set_color(CAT_COLORS[cat])

# x-axis
ax.set_xticks(range(len(region_order)))
ax.set_xticklabels(region_order, rotation=45, ha='right', fontsize=FS_TICK)

ax.set_title(
    'MAGMA GO Biological Process enrichment across 16 JAGWAS regions\n'
    '(terms FDR-sig in \u2265{} regions; * FDR q<0.05 per cell)'.format(MIN_REGIONS),
    fontsize=FS_TITLE, fontweight='bold')
ax.set_xlabel('BRE region', fontsize=FS_LABEL)
ax.set_ylabel('GO Biological Process term', fontsize=FS_LABEL)

# Manual layout — no tight_layout (avoids twinx/colorbar conflicts)
# left/bottom/right/top in figure fraction; right margin for colorbar
fig.subplots_adjust(left=0.31, right=0.84, top=0.92, bottom=0.13)

# Colorbar: manual axes placed in right margin
cbar_ax = fig.add_axes([0.863, 0.15, 0.013, 0.70])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('\u2212log\u2081\u2080(p)', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

# Category legend below figure
legend_patches = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
fig.legend(handles=legend_patches, loc='lower center',
           bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=FS_TICK, frameon=False)

save_mpl(fig, FIG_DIR / 'gobp_heatmap')
plt.close(fig)


# ── STEP 4: Figure B — 4×4 per-region panel ───────────────────────────────────
print("Generating per-region panel ...")

NCOLS, NROWS = 4, 4
fig2, axes = plt.subplots(NROWS, NCOLS,
                           figsize=(MM(240), MM(250)))

for idx, (folder, display) in enumerate(REGIONS.items()):
    row_idx, col_idx = divmod(idx, NCOLS)
    ax = axes[row_idx][col_idx]

    sub = long_df[long_df['region'] == display].copy()
    # Show only FDR-significant terms, sorted by −log10p
    sub = sub[sub['fdr_sig']].sort_values('log10p', ascending=False).head(TOP_PER_REG)
    sub = sub.iloc[::-1]   # flip so highest is on top

    # Short labels for per-region panel (max 32 chars)
    short_labels = [make_label(r['FULL_NAME'], r['VARIABLE'], max_len=32)
                    for _, r in sub.iterrows()]

    colors  = [CAT_COLORS[c] for c in sub['category']]
    y_pos   = list(range(len(sub)))
    ax.barh(y_pos, sub['log10p'].tolist(),
            color=colors, height=0.7, edgecolor='none')

    # FDR threshold dashed line (lowest p among shown, still FDR-sig)
    if len(sub):
        ax.axvline(-np.log10(sub['P'].max()), color='#333333',
                   linewidth=0.6, linestyle='--', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=4.5)
    ax.set_xlabel('\u2212log\u2081\u2080(p)', fontsize=5)
    ax.set_title(display, fontsize=6.5, fontweight='bold', pad=3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=5)

# Overall title
fig2.suptitle(
    'Top FDR-significant GO Biological Process terms per subcortical region (MAGMA)\n'
    'dashed = lowest shown FDR threshold  |  colour = term category',
    fontsize=FS_TITLE, fontweight='bold', y=0.98)

# Category legend below all subplots
legend_patches2 = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_COLORS]
fig2.legend(handles=legend_patches2, loc='lower center',
            ncol=4, fontsize=6, frameon=False,
            bbox_to_anchor=(0.5, 0.005))

# Explicit spacing: top for suptitle, bottom for legend, wider wspace for labels
fig2.subplots_adjust(left=0.07, right=0.97, top=0.91,
                     bottom=0.06, hspace=0.65, wspace=0.60)

save_mpl(fig2, FIG_DIR / 'gobp_per_region_panel')
plt.close(fig2)


# ── STEP 5: Summary stats ──────────────────────────────────────────────────────
print("\n=== Category breakdown of cross-regional terms ===")
cat_counts = cross_broad['category'].value_counts()
for cat, n in cat_counts.items():
    pct = 100 * n / len(cross_broad)
    print("  {:<30}: {:>4}  ({:.1f}%)".format(cat, n, pct))

print("\n=== Top 5 terms per category (cross-regional, sorted by breadth) ===")
for cat in [CAT_STRUCTURAL, CAT_NEURONAL, CAT_MORPHO, CAT_OTHER]:
    sub = cross_broad[cross_broad['category'] == cat].head(5)
    print("\n  {}:".format(cat))
    for _, r in sub.iterrows():
        print("    {:<50}  n_sig={}/16  mean\u2212log10p={:.2f}".format(
            r['label'], int(r['n_sig']), r['mean_log10p']))

print("\nDone: 16_magma_gobp_enrichment.py")
