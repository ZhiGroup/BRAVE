"""
02_eqtl_tissue_breakdown.py
Per-region eQTL tissue breakdown — shows which tissues our JAGWAS hits act through.

Inputs (per region):  eqtl.txt  (FUMA eQTL mapping output)
Outputs:
  results/per_region/eqtl_tissue_counts.csv      — genes per tissue per region (long format)
  figures/per_region/eqtl_tissue/<region>.pdf/.png  — per-region top-tissue barplot (supp.)
  figures/per_region/eqtl_tissue_panel.pdf/.png     — 4×4 multi-panel summary figure

Key filter: eqtlMapFilt == 1  (gene mapping passed FDR threshold in FUMA)
Metric:     number of unique eQTL-mapped genes per tissue per region

Supports paper Claim 3: loci colocalize with brain-tissue eQTLs, enriched in the
  expected brain region's tissue (e.g., Putamen → Brain_Putamen_basal_ganglia).
"""

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FONT_FAMILY, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR   = Path(cfg.postgwas.fuma_dir)
OUT_DIR    = Path(__file__).parents[1] / "results" / "per_region"
FIG_DIR    = Path(__file__).parents[1] / "figures" / "per_region" / "eqtl_tissue"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20   # top tissues to show per region

# ── TISSUE DISPLAY NAMES ──────────────────────────────────────────────────────
# Tidy up GTEx and CMC/BrainSeq tissue codes for display
GTEx_RENAME = {
    'Brain_Caudate_basal_ganglia':              'Caudate (BG)',
    'Brain_Putamen_basal_ganglia':              'Putamen (BG)',
    'Brain_Nucleus_accumbens_basal_ganglia':    'Accumbens (BG)',
    'Brain_Cerebellum':                         'Cerebellum',
    'Brain_Cerebellar_Hemisphere':              'Cerebellar Hem.',
    'Brain_Cortex':                             'Cortex',
    'Brain_Frontal_Cortex_BA9':                 'Frontal Ctx (BA9)',
    'Brain_Anterior_cingulate_cortex_BA24':     'ACC (BA24)',
    'Brain_Hippocampus':                        'Hippocampus',
    'Brain_Amygdala':                           'Amygdala',
    'Brain_Hypothalamus':                       'Hypothalamus',
    'Brain_Substantia_nigra':                   'Substantia Nigra',
    'Brain_Spinal_cord_cervical_c-1':           'Spinal Cord (c-1)',
    'Brain_Corpus_callosum':                    'Corpus Callosum',
}
CMC_RENAME = {
    'HIPP': 'Hippocampus (CMC)',
    'CRBL': 'Cerebellum (CMC)',
    'TCTX': 'Temporal Ctx (CMC)',
    'MEDU': 'Medulla (CMC)',
    'THAL': 'Thalamus (CMC)',
    'FCTX': 'Frontal Ctx (CMC)',
    'OCTX': 'Occipital Ctx (CMC)',
    'PUTM': 'Putamen (CMC)',
    'WHMT': 'White Matter (CMC)',
    'aveALL': 'Average All (CMC)',
    'PsychENCODE_eQTLs': 'PsychENCODE',
    'BrainSeq_ge_brain':  'BrainSeq (brain)',
}

def tidy_tissue(t: str) -> str:
    if t in GTEx_RENAME:
        return GTEx_RENAME[t]
    if t in CMC_RENAME:
        return CMC_RENAME[t]
    # Generic tidy: strip leading 'Brain_', replace underscores
    t2 = re.sub(r'^Brain_', '', t).replace('_', ' ')
    return t2

# Brain tissues (for colour coding)
BRAIN_KEYWORDS = ['brain', 'hipp', 'crbl', 'tctx', 'medu', 'thal', 'fctx', 'octx',
                  'putm', 'whmt', 'psychencode', 'brainseq', 'cerebell', 'cortex',
                  'cingulate', 'amygdala', 'substantia', 'hypothalamus', 'caudate',
                  'putamen', 'accumbens', 'spinal']

def is_brain(tissue: str) -> bool:
    tl = tissue.lower()
    return any(k in tl for k in BRAIN_KEYWORDS)

# ── REGION PARSING ────────────────────────────────────────────────────────────
BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def parse_region(folder):
    n = folder.strip(); nl = n.lower()
    if nl.startswith('left_'):
        side, base = 'Left', n[5:]
    elif nl.startswith('right_'):
        side, base = 'Right', n[6:]
    else:
        side, base = 'Midline', n
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

# ── LOAD AND COUNT ────────────────────────────────────────────────────────────
print("Loading eQTL data ...")
all_counts = []

region_dirs = sorted(FUMA_DIR.iterdir(), key=lambda d: sort_key(*parse_region(d.name)))

for region_dir in region_dirs:
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    eqtl_file = region_dir / 'eqtl.txt'
    if not eqtl_file.exists():
        print(f"  SKIP {folder}: eqtl.txt missing")
        continue

    side, base = parse_region(folder)
    disp = display_name(side, base)

    # Read only needed columns — eqtl.txt can be very large
    usecols = ['tissue', 'symbol', 'eqtlMapFilt']
    df = pd.read_csv(eqtl_file, sep='\t', usecols=usecols, low_memory=False)

    # Filter to FUMA-mapped eQTLs only
    df_filt = df[df['eqtlMapFilt'] == 1].copy()

    # Count unique genes per tissue
    counts = (df_filt.groupby('tissue')['symbol']
              .nunique()
              .reset_index(name='n_genes')
              .sort_values('n_genes', ascending=False))
    counts['tissue_display'] = counts['tissue'].apply(tidy_tissue)
    counts['is_brain']       = counts['tissue'].apply(is_brain)
    counts['folder']         = folder
    counts['side']           = side
    counts['base']           = base
    counts['display']        = disp

    all_counts.append(counts)
    print(f"  {folder}: {df_filt['symbol'].nunique()} eQTL-mapped genes, "
          f"{counts['tissue'].nunique()} tissues")

# ── SAVE LONG-FORMAT CSV ──────────────────────────────────────────────────────
master = pd.concat(all_counts, ignore_index=True)
master.to_csv(OUT_DIR / 'eqtl_tissue_counts.csv', index=False)
print(f"\nSaved: {OUT_DIR}/eqtl_tissue_counts.csv")

# ── PER-REGION BARPLOTS ───────────────────────────────────────────────────────
BRAIN_COLOR  = '#4878CF'   # blue for brain tissues
OTHER_COLOR  = '#AAAAAA'   # grey for non-brain tissues

def plot_region(df_region, disp, out_stem):
    top = df_region.head(TOP_N)
    colors = [BRAIN_COLOR if b else OTHER_COLOR for b in top['is_brain']]

    fig, ax = plt.subplots(figsize=(MM(110), MM(90)))
    y = np.arange(len(top))
    bars = ax.barh(y, top['n_genes'], color=colors, height=0.7, edgecolor='none')

    # Value labels
    xmax = top['n_genes'].max()
    for bar, val in zip(bars, top['n_genes']):
        ax.text(bar.get_width() + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', ha='left', fontsize=FS_TICK - 1)

    ax.set_yticks(y)
    ax.set_yticklabels(top['tissue_display'], fontsize=FS_TICK)
    ax.set_xlabel('Number of eQTL-mapped genes', fontsize=FS_LABEL)
    ax.set_title(disp, fontsize=FS_LABEL, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, xmax * 1.15)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=BRAIN_COLOR, label='Brain tissue'),
        mpatches.Patch(color=OTHER_COLOR, label='Non-brain tissue'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', frameon=False, fontsize=FS_TICK)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / out_stem)
    plt.close()

print("\nGenerating per-region barplots ...")
for df_r in all_counts:
    folder = df_r['folder'].iloc[0]
    disp   = df_r['display'].iloc[0]
    out_stem = folder.replace('-', '_').replace('/', '_')
    plot_region(df_r, disp, out_stem)
    print(f"  Saved: {out_stem}")

# ── 4×4 MULTI-PANEL SUMMARY ───────────────────────────────────────────────────
# Show top 10 tissues for each region in a compact panel
print("\nGenerating multi-panel figure ...")

ordered_folders = [d.name for d in region_dirs
                   if d.is_dir() and (d / 'eqtl.txt').exists()]

n_panels = len(ordered_folders)
ncols = 4
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(MM(180), MM(55 * nrows)),
                         constrained_layout=True)
axes_flat = axes.flatten()

TOP_PANEL = 10

for ax, folder in zip(axes_flat, ordered_folders):
    side, base = parse_region(folder)
    disp = display_name(side, base)

    df_r = master[master['folder'] == folder].sort_values('n_genes', ascending=False)
    top  = df_r.head(TOP_PANEL)
    colors = [BRAIN_COLOR if b else OTHER_COLOR for b in top['is_brain']]

    y = np.arange(len(top))
    ax.barh(y, top['n_genes'], color=colors, height=0.7, edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(top['tissue_display'], fontsize=5.5)
    ax.set_title(disp, fontsize=FS_TICK, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=5.5)

# Hide unused axes
for ax in axes_flat[n_panels:]:
    ax.set_visible(False)

# Shared legend
brain_patch = mpatches.Patch(color=BRAIN_COLOR, label='Brain tissue')
other_patch = mpatches.Patch(color=OTHER_COLOR, label='Non-brain tissue')
fig.legend(handles=[brain_patch, other_patch],
           loc='lower right', ncol=2, frameon=False, fontsize=FS_TICK)

save_mpl(fig, Path(__file__).parents[1] / 'figures' / 'per_region' / 'eqtl_tissue_panel')
plt.close()
print(f"Saved: figures/per_region/eqtl_tissue_panel.pdf/.png")

# ── SUMMARY: top tissue per region ───────────────────────────────────────────
print("\n=== Top eQTL tissue per region ===")
top1 = (master.sort_values('n_genes', ascending=False)
        .groupby('folder').first()
        .reset_index()[['folder', 'tissue_display', 'n_genes']])
print(top1.to_string(index=False))
