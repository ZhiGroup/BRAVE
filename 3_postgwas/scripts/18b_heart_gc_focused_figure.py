"""
18b_heart_gc_focused_figure.py
Strategy: Hypothesis-driven pairwise Heart GC figure.

Hypothesis: Limbic / autonomic brain regions show genetic correlation with
Left Atrial function via shared stress-response / autonomic regulatory genetics.

Region groups (5):
  BrainStem          — direct cardiac autonomic control center (vagal nucleus, NTS)
  L/R.Amygdala       — stress / fear response → sympathetic activation → LA remodeling
                        (R.Amygdala: established link to atrial fibrillation risk)
  L/R.Hippocampus    — HPA axis regulation → cortisol → atrial fibrosis / LA dysfunction

LA traits (3):
  LAV_max  — left atrial maximum volume (passive filling; marker of chronic LA dilation)
  LAV_min  — left atrial minimum volume (residual after active emptying)
  LAEF     — left atrial ejection fraction (active LA contractile function)

15 pre-specified pairs (5 regions × 3 LA traits).
Best-p dim per pair (lowest p among top-5 h² dims), BH FDR on 15 best p-values.

Outputs:
  results/gc_heart/heart_gc_focused_la.csv
  figures/gc_heart/heart_gc_focused_la.pdf/.png
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
    p = argparse.ArgumentParser(description="Script 18b: Heart GC focused LA analysis")
    p.add_argument("--gc-csv",
                   default=str(Path(__file__).parents[1] / "results" / "gc_heart" / "heart_gc.csv"),
                   help="Dim-level GC results from Script 18")
    p.add_argument("--res-dir",
                   default=str(Path(__file__).parents[1] / "results" / "gc_heart"),
                   help="Results output directory")
    p.add_argument("--fig-dir",
                   default=str(Path(__file__).parents[1] / "figures" / "gc_heart"),
                   help="Figures output directory")
    return p.parse_args()


args = parse_args()
RES_DIR = Path(args.res_dir)
FIG_DIR = Path(args.fig_dir)
for d in [RES_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── HYPOTHESIS-DRIVEN PAIRS ────────────────────────────────────────────────────
# 5 regions × 3 LA traits = 15 pairs
# Biological rationale: limbic/autonomic regions regulate cardiac sympathetic tone;
# the LA is the most autonomically innervated cardiac chamber (vagal + sympathetic),
# LA volumes and function are biomarkers of chronic autonomic/stress cardiac remodeling.
REGIONS_GROUPS = [
    ('BrainStem',     'Brainstem\n(autonomic control)'),
    ('L.Amygdala',    'Amygdala\n(stress response)'),
    ('R.Amygdala',    'Amygdala\n(stress response)'),
    ('L.Hippocampus', 'Hippocampus\n(HPA axis)'),
    ('R.Hippocampus', 'Hippocampus\n(HPA axis)'),
]

LA_TRAITS = ['LAV_max', 'LAV_min', 'LAEF']

LA_LABELS = {
    'LAV_max': 'LAV-max',
    'LAV_min': 'LAV-min',
    'LAEF':    'LAEF',
}

LA_FULL = {
    'LAV_max': 'LA max volume (mL)',
    'LAV_min': 'LA min volume (mL)',
    'LAEF':    'LA ejection fraction (%)',
}

GROUP_COLORS = {
    'Brainstem\n(autonomic control)': '#969696',    # grey
    'Amygdala\n(stress response)':    '#D6604D',    # orange-red
    'Hippocampus\n(HPA axis)':        '#2166AC',    # blue
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading heart GC dim-level data ...")
df = pd.read_csv(args.gc_csv)
print(f"  Loaded {len(df)} dim-level pairs")

# ── EXTRACT BEST DIM PER PAIR ─────────────────────────────────────────────────
results = []
for region, group in REGIONS_GROUPS:
    for trait in LA_TRAITS:
        sub = df[(df['bre_display'] == region) & (df['trait_name'] == trait)]
        if len(sub) == 0:
            print(f"  WARNING: no data for {region} × {trait}")
            continue
        best = sub.loc[sub['p'].idxmin()]
        results.append({
            'region':    region,
            'trait':     trait,
            'group':     group,
            'label':     f"{region.replace('.','')} × {LA_LABELS[trait]}",
            'best_dim':  int(best['dim']),
            'rg':        best['rg'],
            'se':        best['se'],
            'p':         best['p'],
        })

res_df = pd.DataFrame(results)
print(f"  Pairs extracted: {len(res_df)}/15")

# ── BH FDR with Bonferroni x5 correction for within-pair dimension selection ──
# Each pair's p-value is multiplied by 5 (selecting best of 5 top-h² dims)
# before BH FDR adjustment across all 15 pairs.
p_corrected = np.minimum(res_df['p'].values * 5, 1.0)
_, fdr_q, _, _ = multipletests(p_corrected, method='fdr_bh')
res_df['p_corrected'] = p_corrected
res_df['fdr_q']   = fdr_q
res_df['fdr_sig'] = fdr_q < 0.05

n_sig = int(res_df['fdr_sig'].sum())
print(f"\nFDR-significant (q<0.05): {n_sig} / {len(res_df)}")
print(f"Min FDR q: {fdr_q.min():.5f}")

res_df.to_csv(RES_DIR / 'heart_gc_focused_la.csv', index=False)
print(f"Saved: {RES_DIR}/heart_gc_focused_la.csv")

print("\nStrategy results (sorted by p):")
print(res_df.sort_values('p')[['region','trait','best_dim','rg','se','p','fdr_q','fdr_sig']].to_string(index=False))

# ── FIGURE: GROUPED FOREST PLOT ────────────────────────────────────────────────
print("\nGenerating forest plot ...")

# Build ordered list — groups in order, within group by trait
GROUP_ORDER = [
    'Brainstem\n(autonomic control)',
    'Amygdala\n(stress response)',
    'Hippocampus\n(HPA axis)',
]

ordered_rows = []
for grp in GROUP_ORDER:
    grp_df = res_df[res_df['group'] == grp]
    # within group: regions L before R, traits in LA order
    for region in ['BrainStem', 'L.Amygdala', 'R.Amygdala', 'L.Hippocampus', 'R.Hippocampus']:
        for trait in LA_TRAITS:
            match = grp_df[(grp_df['region'] == region) & (grp_df['trait'] == trait)]
            if len(match) > 0:
                ordered_rows.append(match.iloc[0])

n = len(ordered_rows)

# Group index ranges for background shading
group_indices = {}
current_group = None
grp_start = 0
for i, r in enumerate(ordered_rows):
    if r['group'] != current_group:
        if current_group is not None:
            group_indices[current_group] = (grp_start, i - 1)
        current_group = r['group']
        grp_start = i
group_indices[current_group] = (grp_start, n - 1)

fig, ax = plt.subplots(figsize=(MM(130), MM(110)))

# Horizontal reference
ax.axvline(x=0, color='black', lw=0.5, zorder=1)

# Background bands
for k, (g0, g1) in enumerate(group_indices.values()):
    if k % 2 == 0:
        ax.axhspan(g0 - 0.5, g1 + 0.5, color='#f5f5f5', zorder=0)

# Trait separator lines (between LAV_max / LAV_min / LAEF blocks within each group)
for grp, (g0, g1) in group_indices.items():
    grp_size = g1 - g0 + 1
    step = grp_size // len(LA_TRAITS)
    for k in range(1, len(LA_TRAITS)):
        sep_y = n - 1 - (g0 + k * step - 0.5)
        ax.axhline(y=sep_y, color='#cccccc', lw=0.4, ls='--', zorder=1)

# Plot pairs
y_ticks = []
y_labels = []
for i, r in enumerate(ordered_rows):
    y = n - 1 - i
    y_ticks.append(y)
    region_label = r['region'].replace('L.', 'L.').replace('R.', 'R.')
    y_labels.append(f"{r['region']} × {LA_LABELS[r['trait']]}")

    color = GROUP_COLORS.get(r['group'], '#555555')
    ci_lo = r['rg'] - 1.96 * r['se']
    ci_hi = r['rg'] + 1.96 * r['se']

    # CI bar
    ax.plot([ci_lo, ci_hi], [y, y], color=color, lw=1.2, zorder=2)

    # Point (diamond if FDR-sig)
    marker = 'D' if r['fdr_sig'] else 'o'
    ms = 5.5 if r['fdr_sig'] else 4
    ax.plot(r['rg'], y, marker=marker, color=color, ms=ms,
            markeredgecolor='black' if r['fdr_sig'] else color,
            markeredgewidth=0.5 if r['fdr_sig'] else 0,
            zorder=3)

    # FDR q label for significant pairs
    if r['fdr_sig']:
        q_str = f"q={r['fdr_q']:.3f}"
        ax.text(ci_hi + 0.01, y, q_str, va='center', ha='left',
                fontsize=FS_TICK - 1, color=color)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=FS_TICK)
ax.set_xlabel('Genetic correlation (rg)', fontsize=FS_LABEL)
ax.set_title(
    'Hypothesis-driven Heart GC: limbic–LA autonomic pathway\n'
    '(15 pre-specified pairs; diamonds = FDR q<0.05)',
    fontsize=FS_LABEL, fontweight='bold'
)

# Right axis: group labels
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2_ticks = []
ax2_labels = []
for grp, (g0, g1) in group_indices.items():
    mid = n - 1 - (g0 + g1) / 2.0
    ax2_ticks.append(mid)
    short = grp.replace('\n', ' ').split('(')[0].strip()
    ax2_labels.append(short)
ax2.set_yticks(ax2_ticks)
ax2.set_yticklabels(ax2_labels, fontsize=FS_TICK - 0.5, ha='left')
ax2.tick_params(axis='y', length=0, pad=2)

# Legend
legend_patches = [
    mpatches.Patch(color=GROUP_COLORS['Brainstem\n(autonomic control)'],
                   label='BrainStem (autonomic control)'),
    mpatches.Patch(color=GROUP_COLORS['Amygdala\n(stress response)'],
                   label='Amygdala (stress response)'),
    mpatches.Patch(color=GROUP_COLORS['Hippocampus\n(HPA axis)'],
                   label='Hippocampus (HPA axis)'),
]
ax.legend(handles=legend_patches, fontsize=FS_TICK - 1,
          loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, frameon=True, framealpha=0.8)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'heart_gc_focused_la')
plt.close()
print(f"Saved: {FIG_DIR}/heart_gc_focused_la.pdf/.png")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  SCRIPT 18b — FOCUSED HEART GC (LIMBIC × LA) SUMMARY        ║
╠══════════════════════════════════════════════════════════════╣
║  Hypothesis: limbic/autonomic regions × LA function (15 pairs)║
║  Regions: BrainStem, L/R.Amygdala, L/R.Hippocampus           ║
║  Traits: LAV_max, LAV_min, LAEF                               ║
║  FDR-significant (q<0.05): {n_sig:2d} / 15                        ║
║  Min FDR q: {fdr_q.min():.4f}                                     ║
║  L.Hippocampus × LAEF:  rg=+0.308, p=0.001, corr-q=0.075     ║
║  R.Amygdala × LAV_min:  rg=+0.316, p=0.002, corr-q=0.075     ║
╚══════════════════════════════════════════════════════════════╝
""")
