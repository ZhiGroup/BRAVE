#!/usr/bin/env python3
"""
43_univariate_baseline_novelty.py
Aggregate the FastGWA-minP univariate baseline loci across 16 regions and run
the same 3-way novelty pipeline (vs ENIGMA 2024 + Shape 2022) that was applied
to the 276 JAGWAS aggregate loci. Goal: a like-for-like substrate-vs-statistical-
test comparison.

  - Source loci:        fastgwa_minp/<region>/GenomicRiskLoci.txt (16 regions)
  - Aggregation:        greedy interval merge (same algo as Script 06)
  - ENIGMA reference:   Garcia-Marin 2024 FUMA outputs (same as Script 12)
  - Shape reference:    Zhao 2022 TableS5 'structure' sheet (same as Script 12b)
  - Novelty rules:      identical to Scripts 12 + 12b

Outputs
  results/cross_region/fastgwa_minp_aggregate_loci.csv          — 60-ish AL- loci
  results/cross_region/fastgwa_minp_novelty_classification.csv  — per-locus calls
  results/cross_region/fastgwa_minp_novelty_summary.txt         — counts + %
  figures/cross_region/jagwas_vs_univariate_novelty_compare.{pdf,png}
                                                                — 4-bar comparison

Run:  python 43_univariate_baseline_novelty.py
"""
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parents[1] / "config"))
from paths import cfg  # noqa: E402

sys.path.insert(0, str(SCRIPTS))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL  # noqa: E402

PROJECT_BASE = SCRIPTS.parent
FUMA_UNI = Path("<EXTERNAL: univariate FastGWA min-P FUMA output root (one folder per region)>")
FUMA_PREFIX = "discovery_"

ENIGMA_DIR = Path(str(cfg.postgwas.enigma_dir))
ACCUMBENS_DIR = PROJECT_BASE / "external_clumping" / "accumbens"
SHAPE_SUPP = Path(str(cfg.postgwas.shape_gwas))

RES_DIR = PROJECT_BASE / "results" / "cross_region"
FIG_DIR = PROJECT_BASE / "figures" / "cross_region"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# JAGWAS reference numbers (from results/cross_region/aggregate_loci.csv + scripts 12/12b)
N_JAGWAS = 276
JAGWAS_NOVEL_VS_ENIGMA = 196      # 71.0%
JAGWAS_NOVEL_3WAY      = 176      # 63.8%

ENIGMA_REGIONS = ['Accumbens', 'Amygdala', 'Brainstem', 'Caudate', 'Hippocampus',
                  'ICV', 'Pallidium', 'Putamen', 'Thalamus', 'Ventral']
SHAPE_WINDOW = 500_000   # ±500 kb around each Zhao lead SNP (Script 12b convention)


def greedy_merge(df_in: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
    """Merge overlapping intervals on the same chromosome. Matches Script 06."""
    df = df_in.copy()
    df['chr'] = df['chr'].astype(str)
    df = df.sort_values(['chr', 'start']).reset_index(drop=True)

    out, cur = [], None
    for _, row in df.iterrows():
        if cur is None or row['chr'] != cur['chr'] or row['start'] > cur['end']:
            if cur is not None:
                out.append(cur)
            cur = {
                'chr': row['chr'], 'start': int(row['start']), 'end': int(row['end']),
                'n_raw': 1,
                'regions': [row.get(label_col, '')] if label_col else [],
                'best_p': row.get('p', np.nan),
                'best_snp': row.get('rsID', ''),
            }
        else:
            cur['end'] = max(cur['end'], int(row['end']))
            cur['n_raw'] += 1
            if label_col:
                cur['regions'].append(row.get(label_col, ''))
            p = row.get('p', np.nan)
            if pd.notna(p) and (pd.isna(cur['best_p']) or p < cur['best_p']):
                cur['best_p'] = p
                cur['best_snp'] = row.get('rsID', '')
    if cur is not None:
        out.append(cur)
    res = pd.DataFrame(out)
    if label_col:
        res['regions'] = res['regions'].apply(lambda lst: ';'.join(sorted(set([r for r in lst if r]))))
        res['n_regions'] = res['regions'].apply(lambda s: len(s.split(';')) if s else 0)
    return res


def overlaps_any(chrom, start, end, ref_df) -> bool:
    same = ref_df[ref_df['chr'].astype(str) == str(chrom)]
    if same.empty:
        return False
    return ((same['start'] <= end) & (same['end'] >= start)).any()


# ── STEP 1: aggregate 16 fastgwa-minP regions ───────────────────────────────────
print("Loading FastGWA-minP per-region FUMA loci ...")
raw_rows = []
for d in sorted(FUMA_UNI.iterdir()):
    if not d.is_dir() or not d.name.startswith(FUMA_PREFIX):
        continue
    region = d.name.replace(FUMA_PREFIX, '')
    grl = d / 'GenomicRiskLoci.txt'
    if not grl.exists():
        print(f"  WARN missing: {grl}")
        continue
    df = pd.read_csv(grl, sep='\t', low_memory=False)
    df = df[['chr', 'start', 'end', 'rsID', 'p']].copy()
    df['region'] = region
    raw_rows.append(df)
    print(f"  {region:30s}  {len(df):>3} raw loci")

raw = pd.concat(raw_rows, ignore_index=True)
for c in ('chr', 'start', 'end'):
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
raw = raw.dropna(subset=['chr', 'start', 'end'])
raw[['chr', 'start', 'end']] = raw[['chr', 'start', 'end']].astype(int)
print(f"\nTotal raw univariate loci across 16 regions: {len(raw)}")

uni_agg = greedy_merge(raw, label_col='region')
uni_agg.insert(0, 'al_id', [f"UL-{i+1:04d}" for i in range(len(uni_agg))])
uni_agg['width_kb'] = ((uni_agg['end'] - uni_agg['start']) / 1000).round(1)
print(f"Univariate aggregate loci (UL-): {len(uni_agg)}")
uni_agg.to_csv(RES_DIR / 'fastgwa_minp_aggregate_loci.csv', index=False)


# ── STEP 2: build ENIGMA + Shape reference loci sets (same as Scripts 12, 12b) ─
print("\nLoading ENIGMA 2024 reference loci ...")
enigma_rows = []
for region in ENIGMA_REGIONS:
    region_dir = ACCUMBENS_DIR if region == 'Accumbens' else ENIGMA_DIR / region
    grl = region_dir / 'GenomicRiskLoci.txt'
    if not grl.exists():
        print(f"  WARN missing ENIGMA {region}")
        continue
    df = pd.read_csv(grl, sep='\t')
    df['enigma_region'] = region
    enigma_rows.append(df[['chr', 'start', 'end', 'rsID', 'p', 'enigma_region']])
enigma_raw = pd.concat(enigma_rows, ignore_index=True)
for c in ('chr', 'start', 'end'):
    enigma_raw[c] = pd.to_numeric(enigma_raw[c], errors='coerce')
enigma_raw = enigma_raw.dropna(subset=['chr', 'start', 'end'])
enigma_raw[['chr', 'start', 'end']] = enigma_raw[['chr', 'start', 'end']].astype(int)
enigma_agg = greedy_merge(enigma_raw, label_col='enigma_region')
print(f"  ENIGMA aggregate loci (all 10 regions): {len(enigma_agg)}")

print("\nLoading Zhao 2022 Shape GWAS loci ...")
shape_df = pd.read_excel(SHAPE_SUPP, sheet_name='structure', header=1)
shape_df['chr'] = pd.to_numeric(shape_df['chr'], errors='coerce')
shape_df['pos'] = pd.to_numeric(shape_df['pos'], errors='coerce')
shape_df = shape_df.dropna(subset=['chr', 'pos']).copy()
shape_df['chr'] = shape_df['chr'].astype(int).astype(str)
shape_df['pos'] = shape_df['pos'].astype(int)
shape_df['start'] = shape_df['pos'] - SHAPE_WINDOW
shape_df['end']   = shape_df['pos'] + SHAPE_WINDOW
print(f"  Shape lead SNPs: {len(shape_df)} (±{SHAPE_WINDOW/1000:.0f} kb each)")

# ── STEP 3: classify each UL- locus ─────────────────────────────────────────────
uni_agg['enigma_overlap'] = uni_agg.apply(
    lambda r: overlaps_any(r['chr'], r['start'], r['end'], enigma_agg), axis=1)
uni_agg['shape_overlap'] = uni_agg.apply(
    lambda r: overlaps_any(r['chr'], r['start'], r['end'], shape_df), axis=1)
uni_agg['novel_vs_enigma'] = ~uni_agg['enigma_overlap']
uni_agg['novel_vs_shape']  = ~uni_agg['shape_overlap']
uni_agg['novel_3way']      = ~(uni_agg['enigma_overlap'] | uni_agg['shape_overlap'])

# 3-way category column
def cat(row):
    if row['enigma_overlap'] and row['shape_overlap']:
        return 'ENIGMA+Shape'
    if row['enigma_overlap']:
        return 'ENIGMA-only'
    if row['shape_overlap']:
        return 'Shape-only'
    return 'Novel'
uni_agg['three_way_cat'] = uni_agg.apply(cat, axis=1)

uni_agg.to_csv(RES_DIR / 'fastgwa_minp_novelty_classification.csv', index=False)

# ── STEP 4: summarise ───────────────────────────────────────────────────────────
n_uni = len(uni_agg)
counts = uni_agg['three_way_cat'].value_counts().to_dict()
n_novel_enigma = int(uni_agg['novel_vs_enigma'].sum())
n_novel_3way   = int(uni_agg['novel_3way'].sum())

summary = f"""=== FastGWA-minP univariate baseline novelty summary ===
Total univariate aggregate loci (UL-):  {n_uni}

vs ENIGMA 2024 (Garcia-Marin et al.):
  Novel vs ENIGMA:                       {n_novel_enigma:>3}  ({100*n_novel_enigma/n_uni:.1f}%)
  ENIGMA-known:                          {n_uni - n_novel_enigma:>3}  ({100*(n_uni-n_novel_enigma)/n_uni:.1f}%)

vs ENIGMA + Shape GWAS (3-way):
  Novel beyond both ENIGMA AND Shape:    {n_novel_3way:>3}  ({100*n_novel_3way/n_uni:.1f}%)
  ENIGMA-only:                           {counts.get('ENIGMA-only', 0):>3}
  Shape-only:                            {counts.get('Shape-only', 0):>3}
  ENIGMA + Shape:                        {counts.get('ENIGMA+Shape', 0):>3}

Comparison with JAGWAS on the same BREs / same cohort:
                          JAGWAS-276            Univariate-{n_uni}
  Novel vs ENIGMA:        {JAGWAS_NOVEL_VS_ENIGMA}/{N_JAGWAS} = {100*JAGWAS_NOVEL_VS_ENIGMA/N_JAGWAS:.1f}%   {n_novel_enigma}/{n_uni} = {100*n_novel_enigma/n_uni:.1f}%
  Novel 3-way:            {JAGWAS_NOVEL_3WAY}/{N_JAGWAS} = {100*JAGWAS_NOVEL_3WAY/N_JAGWAS:.1f}%   {n_novel_3way}/{n_uni} = {100*n_novel_3way/n_uni:.1f}%
"""
print('\n' + summary)
(RES_DIR / 'fastgwa_minp_novelty_summary.txt').write_text(summary)
print(f"Saved: {RES_DIR}/fastgwa_minp_novelty_summary.txt")

# ── STEP 5: side-by-side comparison figure ─────────────────────────────────────
apply_mpl_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(MM(180), MM(85)))

# Panel left: vs ENIGMA only
COL_NOVEL = '#E6550D'
COL_KNOWN = '#3182BD'
labels_a = [f'JAGWAS\n(n = {N_JAGWAS})', f'Univariate\n(n = {n_uni})']
novel_a  = [JAGWAS_NOVEL_VS_ENIGMA,           n_novel_enigma]
known_a  = [N_JAGWAS - JAGWAS_NOVEL_VS_ENIGMA, n_uni - n_novel_enigma]
totals_a = [N_JAGWAS, n_uni]
fract_n  = [novel_a[i]/totals_a[i] for i in range(2)]
fract_k  = [known_a[i]/totals_a[i] for i in range(2)]
xpos = np.arange(2)
ax1.bar(xpos, fract_n, color=COL_NOVEL, label='Novel vs ENIGMA')
ax1.bar(xpos, fract_k, bottom=fract_n, color=COL_KNOWN, label='ENIGMA-known')
for i, (n, k, t) in enumerate(zip(novel_a, known_a, totals_a)):
    ax1.text(i, fract_n[i]/2, f"{n}\n({100*n/t:.1f}%)",
             ha='center', va='center', color='white', fontsize=FS_TICK, fontweight='bold')
    ax1.text(i, fract_n[i] + fract_k[i]/2, f"{k}\n({100*k/t:.1f}%)",
             ha='center', va='center', color='white', fontsize=FS_TICK, fontweight='bold')
ax1.set_xticks(xpos); ax1.set_xticklabels(labels_a, fontsize=FS_TICK)
ax1.set_ylabel('Fraction of aggregate loci', fontsize=FS_LABEL)
ax1.set_ylim(0, 1.05)
ax1.set_title('Novelty vs ENIGMA 2024 volume GWAS', fontsize=FS_LABEL, loc='left', fontweight='bold')
ax1.legend(fontsize=FS_TICK - 0.5, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, handlelength=1.4, columnspacing=1.0)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

# Panel right: 3-way classification stacked bars
CAT_ORDER  = ['Novel', 'Shape-only', 'ENIGMA-only', 'ENIGMA+Shape']
CAT_COLORS = {'Novel': '#E6550D', 'Shape-only': '#9370DB',
              'ENIGMA-only': '#3182BD', 'ENIGMA+Shape': '#2CA02C'}

# JAGWAS 276 from Script 12b results (hard-coded canonical numbers)
JAGWAS_3WAY = {'Novel': 176, 'ENIGMA-only': 52, 'Shape-only': 20, 'ENIGMA+Shape': 28}
UNI_3WAY = {c: int(uni_agg['three_way_cat'].eq(c).sum()) for c in CAT_ORDER}

# Stacked bar: each method = one bar
labels_b = [f'JAGWAS\n(n = {N_JAGWAS})', f'Univariate\n(n = {n_uni})']
for ax_idx, (lbl, dct, total) in enumerate([
        (labels_b[0], JAGWAS_3WAY, N_JAGWAS),
        (labels_b[1], UNI_3WAY,    n_uni)]):
    bottom = 0
    for c in CAT_ORDER:
        f = dct[c] / total if total else 0
        ax2.bar(ax_idx, f, bottom=bottom, color=CAT_COLORS[c],
                edgecolor='white', linewidth=0.6,
                label=c if ax_idx == 0 else None)
        if f > 0.03:
            ax2.text(ax_idx, bottom + f/2, f"{dct[c]}\n({100*dct[c]/total:.0f}%)",
                     ha='center', va='center', color='white',
                     fontsize=FS_TICK - 0.5, fontweight='bold')
        bottom += f
ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels_b, fontsize=FS_TICK)
ax2.set_ylabel('Fraction of aggregate loci', fontsize=FS_LABEL)
ax2.set_ylim(0, 1.05)
ax2.set_title('3-way novelty (ENIGMA + Shape)', fontsize=FS_LABEL, loc='left', fontweight='bold')
handles = [mpatches.Patch(color=CAT_COLORS[c], label=c) for c in CAT_ORDER]
ax2.legend(handles=handles, fontsize=FS_TICK - 0.5, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=4, handlelength=1.4, columnspacing=1.0)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)

fig.suptitle('Same BREs, same cohort: novelty under JAGWAS multivariate vs FastGWA univariate-minP',
             fontsize=FS_LABEL + 1, fontweight='bold', y=1.02)
plt.tight_layout(rect=(0, 0.05, 1, 0.98))
out_stem = FIG_DIR / 'jagwas_vs_univariate_novelty_compare'
save_mpl(fig, out_stem)
plt.close(fig)
print(f"\nSaved figure: {out_stem}.pdf / .png")
