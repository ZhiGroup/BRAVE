"""
03_magma_tissue_enrichment.py
MAGMA GTEx v8 tissue enrichment — shows which tissues are enriched for our JAGWAS loci genes.

Inputs (per region):
  magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out         (54 GTEx tissues)
  magma_exp_gtex_v8_ts_general_avg_log2TPM.gsa.out  (30 GTEx general categories)

Outputs:
  results/per_region/magma_tissue_pvals.csv           — all tissues × all regions (long)
  figures/per_region/magma_tissue/<region>.pdf/.png   — per-region lollipop (supp.)
  figures/per_region/magma_brain_panel.pdf/.png       — brain-only 4×4 multi-panel (main)

Significance: Bonferroni α = 0.05 / n_tissues per file.

Supports paper Claim 3: JAGWAS loci genes are enriched in the expected brain tissues,
  demonstrating region-specific biological signal.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
OUT_DIR  = Path(__file__).parents[1] / "results" / "per_region"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "per_region" / "magma_tissue"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── TISSUE COLOURS ─────────────────────────────────────────────────────────────
BRAIN_COLOR  = '#4878CF'   # blue
OTHER_COLOR  = '#CCCCCC'   # light grey
SIG_COLOR    = '#D62728'   # red for significant hits

BRAIN_TISSUES = {
    'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
    'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex',
    'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus',
    'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
    'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
}

TISSUE_DISPLAY = {
    'Brain_Amygdala':                           'Amygdala',
    'Brain_Anterior_cingulate_cortex_BA24':     'ACC (BA24)',
    'Brain_Caudate_basal_ganglia':              'Caudate (BG)',
    'Brain_Cerebellar_Hemisphere':              'Cerebellar Hem.',
    'Brain_Cerebellum':                         'Cerebellum',
    'Brain_Cortex':                             'Cortex',
    'Brain_Frontal_Cortex_BA9':                 'Frontal Ctx (BA9)',
    'Brain_Hippocampus':                        'Hippocampus',
    'Brain_Hypothalamus':                       'Hypothalamus',
    'Brain_Nucleus_accumbens_basal_ganglia':    'Accumbens (BG)',
    'Brain_Putamen_basal_ganglia':              'Putamen (BG)',
    'Brain_Spinal_cord_cervical_c-1':           'Spinal Cord (c-1)',
    'Brain_Substantia_nigra':                   'Substantia Nigra',
}

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

# ── READ MAGMA FILE ────────────────────────────────────────────────────────────
def read_magma(path):
    """Read MAGMA .gsa.out file, skipping # comment lines."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if header is None:
                header = parts
                continue
            if len(parts) >= len(header):
                rows.append(parts[:len(header)])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=header)
    # Last column may be FULL_NAME or repeated VARIABLE
    for col in ['BETA', 'BETA_STD', 'SE', 'P']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ── LOAD ALL REGIONS ──────────────────────────────────────────────────────────
print("Loading MAGMA tissue enrichment data ...")
all_rows = []

region_dirs = sorted(FUMA_DIR.iterdir(),
                     key=lambda d: sort_key(*parse_region(d.name)) if d.is_dir() else (99,0))

for region_dir in region_dirs:
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    side, base = parse_region(folder)
    disp = display_name(side, base)

    magma_file = region_dir / 'magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out'
    if not magma_file.exists():
        print(f"  SKIP {folder}: MAGMA tissue file missing")
        continue

    df = read_magma(magma_file)
    if df is None or df.empty:
        print(f"  SKIP {folder}: empty MAGMA file")
        continue

    n_tests   = len(df)
    bonf_thr  = 0.05 / n_tests
    df['sig'] = df['P'] < bonf_thr
    df['neg_log10_p'] = -np.log10(df['P'].clip(lower=1e-15))
    df['is_brain']    = df['VARIABLE'].isin(BRAIN_TISSUES)
    df['folder']      = folder
    df['side']        = side
    df['base']        = base
    df['display']     = disp
    df['bonf_thr']    = bonf_thr
    df['n_tests']     = n_tests

    all_rows.append(df)
    n_sig = df['sig'].sum()
    n_brain_sig = df[df['is_brain'] & df['sig']].shape[0]
    print(f"  {folder}: {n_tests} tissues, {n_sig} sig (Bonf), "
          f"{n_brain_sig} brain sig | "
          f"top: {df.sort_values('P').iloc[0]['VARIABLE']} p={df['P'].min():.3g}")

# ── SAVE LONG-FORMAT CSV ──────────────────────────────────────────────────────
master = pd.concat(all_rows, ignore_index=True)
master.to_csv(OUT_DIR / 'magma_tissue_pvals.csv', index=False)
print(f"\nSaved: {OUT_DIR}/magma_tissue_pvals.csv")

# ── LOLLIPOP PLOT FUNCTION ────────────────────────────────────────────────────
def lollipop(df_region, disp, bonf_thr, out_stem, brain_only=False):
    if brain_only:
        df_plot = df_region[df_region['is_brain']].copy()
        df_plot['label'] = df_plot['VARIABLE'].map(
            lambda x: TISSUE_DISPLAY.get(x, x.replace('Brain_', '').replace('_', ' ')))
    else:
        df_plot = df_region.copy()
        df_plot['label'] = df_plot['VARIABLE'].str.replace('_', ' ')

    df_plot = df_plot.sort_values('neg_log10_p', ascending=True).reset_index(drop=True)
    y = np.arange(len(df_plot))

    # Colour: red if significant, brain blue if brain, grey otherwise
    def pt_color(row):
        if row['sig']:
            return SIG_COLOR
        return BRAIN_COLOR if row['is_brain'] else OTHER_COLOR

    colors = [pt_color(row) for _, row in df_plot.iterrows()]
    bonf_line = -np.log10(bonf_thr)

    h = max(MM(70), MM(5) * len(df_plot))
    fig, ax = plt.subplots(figsize=(MM(110), h))

    # Stems
    for i, (_, row) in enumerate(df_plot.iterrows()):
        col = SIG_COLOR if row['sig'] else (BRAIN_COLOR if row['is_brain'] else OTHER_COLOR)
        ax.plot([0, row['neg_log10_p']], [i, i], color=col, lw=0.8, alpha=0.7)
        ax.scatter(row['neg_log10_p'], i, color=col, s=20, zorder=3)

    # Significance line
    ax.axvline(bonf_line, color='#888888', lw=0.8, ls='--', label=f'Bonf. α (p={bonf_thr:.2g})')

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot['label'], fontsize=FS_TICK - 0.5)
    ax.set_xlabel('−log₁₀(P)', fontsize=FS_LABEL)
    ax.set_title(disp + (' — brain tissues' if brain_only else ''), fontsize=FS_LABEL,
                 fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=FS_TICK, frameon=False, loc='lower right')

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / out_stem)
    plt.close()

# ── PER-REGION PLOTS ──────────────────────────────────────────────────────────
print("\nGenerating per-region lollipop plots ...")
for df_r in all_rows:
    folder = df_r['folder'].iloc[0]
    disp   = df_r['display'].iloc[0]
    bonf   = df_r['bonf_thr'].iloc[0]
    stem   = folder.replace('-', '_').replace('/', '_')

    # Full (supplementary)
    lollipop(df_r, disp, bonf, stem + '_all', brain_only=False)
    # Brain-only (main figure input)
    lollipop(df_r, disp, bonf, stem + '_brain', brain_only=True)
    print(f"  {folder}: done")

# ── 4×4 BRAIN-ONLY MULTI-PANEL ────────────────────────────────────────────────
print("\nGenerating brain-only multi-panel figure ...")
ordered = [df for df in all_rows]   # already sorted
n_panels = len(ordered)
ncols = 4
nrows = int(np.ceil(n_panels / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(MM(180), MM(52 * nrows)),
                         constrained_layout=True)
axes_flat = axes.flatten()

for ax, df_r in zip(axes_flat, ordered):
    disp   = df_r['display'].iloc[0]
    bonf   = df_r['bonf_thr'].iloc[0]
    bonf_line = -np.log10(bonf)

    df_b = df_r[df_r['is_brain']].copy()
    df_b['label'] = df_b['VARIABLE'].map(
        lambda x: TISSUE_DISPLAY.get(x, x.replace('Brain_', '').replace('_', ' ')))
    df_b = df_b.sort_values('neg_log10_p', ascending=True).reset_index(drop=True)
    y = np.arange(len(df_b))

    for i, (_, row) in enumerate(df_b.iterrows()):
        col = SIG_COLOR if row['sig'] else BRAIN_COLOR
        ax.plot([0, row['neg_log10_p']], [i, i], color=col, lw=0.8, alpha=0.8)
        ax.scatter(row['neg_log10_p'], i, color=col, s=12, zorder=3)

    ax.axvline(bonf_line, color='#888888', lw=0.7, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(df_b['label'], fontsize=5.5)
    ax.set_title(disp, fontsize=FS_TICK, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=5.5)

for ax in axes_flat[n_panels:]:
    ax.set_visible(False)

# Shared legend
sig_pt  = mlines.Line2D([], [], color=SIG_COLOR,  marker='o', ls='-', ms=4, label='Significant (Bonf.)')
ns_pt   = mlines.Line2D([], [], color=BRAIN_COLOR, marker='o', ls='-', ms=4, label='Not significant')
thr_ln  = mlines.Line2D([], [], color='#888888',   ls='--',    lw=0.8,       label='Bonf. threshold')
fig.legend(handles=[sig_pt, ns_pt, thr_ln], loc='lower right',
           ncol=3, frameon=False, fontsize=FS_TICK)

save_mpl(fig, Path(__file__).parents[1] / 'figures' / 'per_region' / 'magma_brain_panel')
plt.close()
print("Saved: figures/per_region/magma_brain_panel.pdf/.png")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print("\n=== MAGMA significant tissues per region (Bonferroni) ===")
sig_df = master[master['sig'] & master['is_brain']][
    ['display', 'VARIABLE', 'P', 'BETA']
].sort_values(['display', 'P'])

if sig_df.empty:
    print("  No brain tissues reach Bonferroni significance across any region.")
else:
    print(sig_df.to_string(index=False))

print("\n=== Top brain tissue per region (by P) ===")
top = (master[master['is_brain']]
       .sort_values('P')
       .groupby('folder')
       .first()
       .reset_index()[['folder', 'VARIABLE', 'P', 'BETA', 'sig']])
print(top.to_string(index=False))
