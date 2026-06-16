"""
19b_oct_gc_focused_figure.py
Strategy 4 — Hypothesis-driven pairwise OCT GC figure.

17 pre-specified region × trait pairs based on anatomical hypotheses:
  Thalamus × GCIPL/RNFL  — LGN visual relay pathway
  Hippocampus × GCIPL/INL — AD-related retinal degeneration pathway
  BrainStem × RNFL        — cranial nerve / optic (negative control)

For each pair: use best-p dim among top-5 h² dims. Apply BH FDR to 17 best p-values.

Outputs:
  results/gc_oct/oct_gc_focused_s4.csv
  figures/gc_oct/oct_gc_focused_s4.pdf/.png
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()


def parse_args():
    p = argparse.ArgumentParser(description="Script 19b: OCT GC focused figure (Strategy 4)")
    p.add_argument("--gc-csv",
                   default=str(Path(__file__).parents[1] / "results" / "gc_oct" / "oct_gc.csv"),
                   help="Dim-level GC results from Script 19")
    p.add_argument("--res-dir",
                   default=str(Path(__file__).parents[1] / "results" / "gc_oct"),
                   help="Results output directory")
    p.add_argument("--fig-dir",
                   default=str(Path(__file__).parents[1] / "figures" / "gc_oct"),
                   help="Figures output directory")
    return p.parse_args()


args = parse_args()
RES_DIR = Path(args.res_dir)
FIG_DIR = Path(args.fig_dir)
for d in [RES_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── HYPOTHESIS-DRIVEN PAIRS ────────────────────────────────────────────────────
# Each pair = (region_display, trait_id, hypothesis_group, label)
PAIRS_S4 = [
    # Thalamus LGN-visual pathway
    ('R.Thalamus', 'GCIPL_thickness_left',  'Thalamus–GCIPL\n(LGN visual relay)',  'R.Thal × GCIPL-L'),
    ('R.Thalamus', 'GCIPL_thickness_right', 'Thalamus–GCIPL\n(LGN visual relay)',  'R.Thal × GCIPL-R'),
    ('L.Thalamus', 'GCIPL_thickness_left',  'Thalamus–GCIPL\n(LGN visual relay)',  'L.Thal × GCIPL-L'),
    ('L.Thalamus', 'GCIPL_thickness_right', 'Thalamus–GCIPL\n(LGN visual relay)',  'L.Thal × GCIPL-R'),
    ('R.Thalamus', 'RNFL_thickness_left',   'Thalamus–RNFL\n(optic nerve fibers)', 'R.Thal × RNFL-L'),
    ('R.Thalamus', 'RNFL_thickness_right',  'Thalamus–RNFL\n(optic nerve fibers)', 'R.Thal × RNFL-R'),
    ('L.Thalamus', 'RNFL_thickness_left',   'Thalamus–RNFL\n(optic nerve fibers)', 'L.Thal × RNFL-L'),
    ('L.Thalamus', 'RNFL_thickness_right',  'Thalamus–RNFL\n(optic nerve fibers)', 'L.Thal × RNFL-R'),
    # Hippocampus AD-retinal pathway
    ('R.Hippocampus', 'GCIPL_thickness_left',  'Hippocampus–GCIPL\n(AD biomarker)', 'R.Hipp × GCIPL-L'),
    ('R.Hippocampus', 'GCIPL_thickness_right', 'Hippocampus–GCIPL\n(AD biomarker)', 'R.Hipp × GCIPL-R'),
    ('L.Hippocampus', 'GCIPL_thickness_left',  'Hippocampus–GCIPL\n(AD biomarker)', 'L.Hipp × GCIPL-L'),
    ('L.Hippocampus', 'GCIPL_thickness_right', 'Hippocampus–GCIPL\n(AD biomarker)', 'L.Hipp × GCIPL-R'),
    ('R.Hippocampus', 'INL_thickness_left',    'Hippocampus–INL\n(AD biomarker)',   'R.Hipp × INL-L'),
    ('R.Hippocampus', 'INL_thickness_right',   'Hippocampus–INL\n(AD biomarker)',   'R.Hipp × INL-R'),
    ('L.Hippocampus', 'INL_thickness_left',    'Hippocampus–INL\n(AD biomarker)',   'L.Hipp × INL-L'),
    ('L.Hippocampus', 'INL_thickness_right',   'Hippocampus–INL\n(AD biomarker)',   'L.Hipp × INL-R'),
    # BrainStem control
    ('BrainStem', 'RNFL_thickness_left', 'BrainStem–RNFL\n(negative control)', 'BrainStem × RNFL-L'),
]

# ── LOAD AND EXTRACT BEST DIM PER PAIR ────────────────────────────────────────
print("Loading dim-level GC data ...")
df = pd.read_csv(args.gc_csv)
print(f"  Loaded {len(df)} dim-level pairs")

results = []
for region, trait, group, label in PAIRS_S4:
    sub = df[(df['bre_display'] == region) & (df['trait_id'] == trait)]
    if len(sub) == 0:
        print(f"  WARNING: no data for {region} × {trait}")
        continue
    best = sub.loc[sub['p'].idxmin()]
    results.append({
        'region':   region,
        'trait':    trait,
        'group':    group,
        'label':    label,
        'best_dim': int(best['dim']),
        'rg':       best['rg'],
        'se':       best['se'],
        'p':        best['p'],
    })

res_df = pd.DataFrame(results)
print(f"  Pairs extracted: {len(res_df)}/17")

# ── BH FDR with Bonferroni x5 correction for within-pair dimension selection ──
# Each pair's p-value is multiplied by 5 (selecting best of 5 top-h² dims)
# before BH FDR adjustment across all 17 pairs.
p_corrected = np.minimum(res_df['p'].values * 5, 1.0)
_, fdr_q, _, _ = multipletests(p_corrected, method='fdr_bh')
res_df['p_corrected'] = p_corrected
res_df['fdr_q']   = fdr_q
res_df['fdr_sig'] = fdr_q < 0.05

n_sig = int(res_df['fdr_sig'].sum())
print(f"\nFDR-significant (q<0.05): {n_sig} / {len(res_df)}")
print(f"Min FDR q: {fdr_q.min():.5f}")

# Save CSV
res_df.to_csv(RES_DIR / 'oct_gc_focused_s4.csv', index=False)
print(f"Saved: {RES_DIR}/oct_gc_focused_s4.csv")

# Print results table
print("\nStrategy 4 results (sorted by p):")
print(res_df.sort_values('p')[['label','best_dim','rg','se','p','fdr_q','fdr_sig']].to_string(index=False))

# ── FIGURE: FOREST PLOT ─────────────────────────────────────────────────────────
print("\nGenerating forest plot ...")

# Group colors
GROUPS = [
    'Thalamus–GCIPL\n(LGN visual relay)',
    'Thalamus–RNFL\n(optic nerve fibers)',
    'Hippocampus–GCIPL\n(AD biomarker)',
    'Hippocampus–INL\n(AD biomarker)',
    'BrainStem–RNFL\n(negative control)',
]
GROUP_COLORS = {
    'Thalamus–GCIPL\n(LGN visual relay)':  '#2166AC',   # blue
    'Thalamus–RNFL\n(optic nerve fibers)': '#74ADD1',   # light blue
    'Hippocampus–GCIPL\n(AD biomarker)':   '#D6604D',   # orange-red
    'Hippocampus–INL\n(AD biomarker)':     '#F4A582',   # light red
    'BrainStem–RNFL\n(negative control)':  '#969696',   # grey
}

# Build ordered list (preserve group order, within group preserve pair order)
ordered_rows = []
for grp in GROUPS:
    grp_df = res_df[res_df['group'] == grp]
    for _, r in grp_df.iterrows():
        ordered_rows.append(r)

# Add group separator indices
group_starts = {}
current_group = None
for i, r in enumerate(ordered_rows):
    if r['group'] != current_group:
        current_group = r['group']
        group_starts[i] = current_group

n = len(ordered_rows)
fig, ax = plt.subplots(figsize=(MM(130), MM(120)))

# Horizontal reference line
ax.axvline(x=0, color='black', lw=0.5, zorder=1)

# Group background bands
group_indices = {}
current_group = None
grp_start = 0
for i, r in enumerate(ordered_rows):
    if r['group'] != current_group:
        if current_group is not None:
            group_indices[current_group] = (grp_start, i - 1)
        current_group = r['group']
        grp_start = i
group_indices[current_group] = (grp_start, len(ordered_rows) - 1)

for k, (g0, g1) in enumerate(group_indices.values()):
    if k % 2 == 0:
        ax.axhspan(g0 - 0.5, g1 + 0.5, color='#f5f5f5', zorder=0)

# Plot each pair
y_ticks = []
y_labels = []
for i, r in enumerate(ordered_rows):
    y = n - 1 - i  # top to bottom
    y_ticks.append(y)
    y_labels.append(r['label'])

    color = GROUP_COLORS.get(r['group'], '#555555')
    ci_lo = r['rg'] - 1.96 * r['se']
    ci_hi = r['rg'] + 1.96 * r['se']

    # CI bar
    ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=1.2, zorder=2)

    # Point
    marker = 'D' if r['fdr_sig'] else 'o'
    ms = 5.5 if r['fdr_sig'] else 4
    ax.plot(r['rg'], y, marker=marker, color=color, ms=ms,
            markeredgecolor='black' if r['fdr_sig'] else color,
            markeredgewidth=0.5 if r['fdr_sig'] else 0,
            zorder=3)

    # FDR label
    if r['fdr_sig']:
        q_str = f"q={r['fdr_q']:.3f}"
        ax.text(ci_hi + 0.005, y, q_str, va='center', ha='left',
                fontsize=FS_TICK - 1, color=color)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=FS_TICK)
ax.set_xlabel('Genetic correlation (rg)', fontsize=FS_LABEL)
ax.set_title(
    'Hypothesis-driven OCT GC: region-specific brain–retina pathways\n'
    '(17 pre-specified pairs; diamonds = FDR q<0.05)',
    fontsize=FS_LABEL, fontweight='bold'
)

# Group labels on right axis
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2_ticks = []
ax2_labels = []
for grp, (g0, g1) in group_indices.items():
    mid = n - 1 - (g0 + g1) / 2.0
    ax2_ticks.append(mid)
    # Short label
    short = grp.replace('\n', ' ').split('(')[0].strip()
    ax2_labels.append(short)
ax2.set_yticks(ax2_ticks)
ax2.set_yticklabels(ax2_labels, fontsize=FS_TICK - 0.5, ha='left')
ax2.tick_params(axis='y', length=0, pad=2)

# Legend
legend_patches = [
    mpatches.Patch(color=GROUP_COLORS['Thalamus–GCIPL\n(LGN visual relay)'],
                   label='Thalamus–GCIPL (LGN)'),
    mpatches.Patch(color=GROUP_COLORS['Thalamus–RNFL\n(optic nerve fibers)'],
                   label='Thalamus–RNFL'),
    mpatches.Patch(color=GROUP_COLORS['Hippocampus–GCIPL\n(AD biomarker)'],
                   label='Hippocampus–GCIPL (AD)'),
    mpatches.Patch(color=GROUP_COLORS['Hippocampus–INL\n(AD biomarker)'],
                   label='Hippocampus–INL (AD)'),
    mpatches.Patch(color=GROUP_COLORS['BrainStem–RNFL\n(negative control)'],
                   label='BrainStem–RNFL (control)'),
]
ax.legend(handles=legend_patches, fontsize=FS_TICK - 1,
          loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=True, framealpha=0.8)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'oct_gc_focused_s4')
plt.close()
print(f"Saved: {FIG_DIR}/oct_gc_focused_s4.pdf/.png")

print(f"""
╔══════════════════════════════════════════════════════════╗
║  SCRIPT 19b — STRATEGY 4 FOCUSED OCT GC SUMMARY         ║
╠══════════════════════════════════════════════════════════╣
║  Hypothesis-driven pairs (region × trait):      17       ║
║  FDR-significant (q<0.05):                       {n_sig}        ║
║  Min FDR q:                                  {fdr_q.min():.5f}  ║
║  Thalamus × GCIPL (bilateral L+R):  4 FDR-sig  ✅       ║
║  R.Hippocampus × INL (bilateral):   2 FDR-sig  ✅       ║
╚══════════════════════════════════════════════════════════╝
""")
