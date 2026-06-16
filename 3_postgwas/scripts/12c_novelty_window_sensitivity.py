"""
12c_novelty_window_sensitivity.py
Sensitivity analysis: 3-way loci novelty under harmonized window definitions.

Reviewer concern (Minor 10): Script 12b uses LD-expanded FUMA windows for ENIGMA
but ±500 kb around lead SNPs for Shape GWAS — asymmetry could artificially inflate
novelty relative to shape GWAS.

This script re-runs the 3-way classification under three harmonized conditions:
  (A) Original (asymmetric): ENIGMA = FUMA LD-expanded; Shape = ±500 kb   [reference]
  (B) Harmonized ±500 kb:    ENIGMA = ±500 kb from lead SNP; Shape = ±500 kb
  (C) Harmonized ±1 Mb:      ENIGMA = ±1 Mb  from lead SNP; Shape = ±1 Mb

ENIGMA lead SNP positions: extracted from FUMA leadSNPs.txt per region.
If leadSNPs.txt is absent, falls back to midpoint of GenomicRiskLoci windows.

Outputs:
  results/cross_region/novelty_window_sensitivity.csv   — counts per condition
  figures/cross_region/novelty_window_sensitivity.pdf/.png
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Novelty 3-way window sensitivity')
parser.add_argument('--enigma-dir',
    default=str(cfg.postgwas.enigma_dir),
    help='Root directory containing one subfolder per ENIGMA region')
parser.add_argument('--shape-xlsx',
    default=str(cfg.postgwas.shape_gwas),
    help='Path to Zhao 2022 TableS5_Locus.xlsx')
parser.add_argument('--results-dir',
    default=None,
    help='Override results directory (default: ../results/cross_region)')
args = parser.parse_args()

ENIGMA_DIR = Path(args.enigma_dir)
# Project-local Accumbens FUMA loci (ENIGMA 2024 did not publish FUMA for
# Accumbens; derived locally with in-house clumping tool, UKB 35k LD panel).
# See docs/data_paths.md §10b.
ACCUMBENS_DIR = (Path(__file__).parents[1]
                 / 'external_clumping' / 'accumbens')
SHAPE_SUPP = Path(args.shape_xlsx)
RES_DIR    = Path(args.results_dir) if args.results_dir else \
             Path(__file__).parents[1] / 'results' / 'cross_region'
FIG_DIR    = Path(__file__).parents[1] / 'figures' / 'cross_region'
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENIGMA_REGIONS = [
    'Accumbens', 'Amygdala', 'Brainstem', 'Caudate', 'Hippocampus',
    'ICV', 'Pallidium', 'Putamen', 'Thalamus', 'Ventral',
]

# ── Colours ────────────────────────────────────────────────────────────────────
COL_NOVEL       = '#E6550D'
COL_ENIGMA_ONLY = '#3182BD'
COL_SHAPE_ONLY  = '#9370DB'
COL_BOTH        = '#2CA02C'
CAT_ORDER       = ['Novel', 'Shape-only', 'ENIGMA-only', 'ENIGMA + Shape']

# ── Helper: greedy interval merge ─────────────────────────────────────────────
def greedy_merge(df_in):
    df = df_in.copy()
    df['chr'] = df['chr'].astype(str)
    df = df.sort_values(['chr', 'start']).reset_index(drop=True)
    merged = []
    cur_chr, cur_start, cur_end = None, None, None
    for _, row in df.iterrows():
        if row['chr'] != cur_chr or row['start'] > cur_end:
            if cur_chr is not None:
                merged.append({'chr': cur_chr, 'start': cur_start, 'end': cur_end})
            cur_chr, cur_start, cur_end = row['chr'], row['start'], row['end']
        else:
            cur_end = max(cur_end, row['end'])
    if cur_chr is not None:
        merged.append({'chr': cur_chr, 'start': cur_start, 'end': cur_end})
    return pd.DataFrame(merged)


def overlaps(chr_, s, e, ref_df):
    sc = ref_df[ref_df['chr'] == str(chr_)]
    if len(sc) == 0:
        return False
    return bool(((sc['start'] <= e) & (sc['end'] >= s)).any())


# ── STEP 1: Load ENIGMA lead SNP positions ────────────────────────────────────
print("Loading ENIGMA lead SNP positions ...")
enigma_snp_rows = []
for region in ENIGMA_REGIONS:
    region_dir = ACCUMBENS_DIR if region == 'Accumbens' else ENIGMA_DIR / region
    lead_f = region_dir / 'leadSNPs.txt'
    grl_f  = region_dir / 'GenomicRiskLoci.txt'

    if lead_f.exists():
        df = pd.read_csv(lead_f, sep='\t')
        # FUMA leadSNPs.txt columns: GenomicLocus, chr, pos, rsID, p, ...
        if 'pos' in df.columns:
            df['snp_pos'] = df['pos']
        elif 'bp' in df.columns:
            df['snp_pos'] = df['bp']
        else:
            # fallback: midpoint from GenomicRiskLoci.txt
            grl = pd.read_csv(grl_f, sep='\t')
            grl['snp_pos'] = ((grl['start'] + grl['end']) // 2).astype(int)
            df = grl[['chr', 'snp_pos']].copy()
        df = df[['chr', 'snp_pos']].copy()
        src = 'leadSNPs.txt'
    elif grl_f.exists():
        grl = pd.read_csv(grl_f, sep='\t')
        grl['snp_pos'] = ((grl['start'] + grl['end']) // 2).astype(int)
        df = grl[['chr', 'snp_pos']].copy()
        src = 'GRL midpoint'
    else:
        print(f"  WARNING: {region} — no FUMA files found, skipping")
        continue

    df['enigma_region'] = region
    enigma_snp_rows.append(df)
    print(f"  {region}: {len(df)} lead SNPs ({src})")

enigma_snps = pd.concat(enigma_snp_rows, ignore_index=True)
enigma_snps['chr'] = pd.to_numeric(enigma_snps['chr'], errors='coerce')
enigma_snps = enigma_snps.dropna(subset=['chr', 'snp_pos'])
enigma_snps['chr'] = enigma_snps['chr'].astype(int).astype(str)
enigma_snps['snp_pos'] = enigma_snps['snp_pos'].astype(int)
print(f"\nTotal ENIGMA lead SNPs: {len(enigma_snps)}")

# ── STEP 2: Load Shape GWAS lead SNP positions (same as 12b) ──────────────────
print("\nLoading Shape GWAS lead SNPs ...")
shape_df = pd.read_excel(SHAPE_SUPP, sheet_name='structure', header=1)
shape_df['chr'] = pd.to_numeric(shape_df['chr'], errors='coerce')
shape_df['pos'] = pd.to_numeric(shape_df['pos'], errors='coerce')
shape_df = shape_df.dropna(subset=['chr', 'pos']).copy()
shape_df['chr'] = shape_df['chr'].astype(int).astype(str)
shape_df['snp_pos'] = shape_df['pos'].astype(int)
shape_snps = shape_df[['chr', 'snp_pos']].drop_duplicates().reset_index(drop=True)
print(f"Shape GWAS unique lead SNP positions: {len(shape_snps)}")

# ── STEP 3: Load original ENIGMA LD-expanded windows (condition A reference) ──
print("\nLoading original ENIGMA LD-expanded windows (condition A) ...")
enigma_agg_orig = pd.read_csv(RES_DIR / 'enigma_aggregate_loci.csv')
enigma_agg_orig['chr'] = enigma_agg_orig['chr'].astype(str)

# ── STEP 4: Load JAGWAS loci with ENIGMA classification from Script 12 ────────
jag = pd.read_csv(RES_DIR / 'jagwas_novelty_classification.csv')
jag['chr'] = jag['chr'].astype(str)

# ── STEP 5: Build locus sets for each window condition ────────────────────────

def build_loci(snp_df, window_bp):
    """Expand lead SNPs to ±window_bp intervals and merge."""
    df = snp_df.copy()
    df['start'] = (df['snp_pos'] - window_bp).clip(lower=0)
    df['end']   =  df['snp_pos'] + window_bp
    return greedy_merge(df[['chr', 'start', 'end']])


def classify_jagwas(jag_df, enigma_agg, shape_agg):
    """Return counts dict for Novel/ENIGMA-only/Shape-only/ENIGMA+Shape."""
    e_flag = jag_df.apply(lambda r: overlaps(r['chr'], r['start'], r['end'], enigma_agg), axis=1)
    s_flag = jag_df.apply(lambda r: overlaps(r['chr'], r['start'], r['end'], shape_agg), axis=1)

    cats = []
    for e, s in zip(e_flag, s_flag):
        if e and s:
            cats.append('ENIGMA + Shape')
        elif e:
            cats.append('ENIGMA-only')
        elif s:
            cats.append('Shape-only')
        else:
            cats.append('Novel')

    ser = pd.Series(cats).value_counts().reindex(CAT_ORDER, fill_value=0)
    return ser


# Shape agg at ±500 kb (same for all conditions because shape always uses ±500 kb)
shape_500  = build_loci(shape_snps, 500_000)
shape_1000 = build_loci(shape_snps, 1_000_000)

# ENIGMA agg at ±500 kb and ±1 Mb
enigma_500  = build_loci(enigma_snps, 500_000)
enigma_1000 = build_loci(enigma_snps, 1_000_000)

print(f"\nEnigma merged loci:  original={len(enigma_agg_orig)} | ±500kb={len(enigma_500)} | ±1Mb={len(enigma_1000)}")
print(f"Shape  merged loci:  ±500kb={len(shape_500)} | ±1Mb={len(shape_1000)}")

# Condition A: original (ENIGMA LD-expanded, Shape ±500 kb) — replicates Script 12b
counts_A = classify_jagwas(jag, enigma_agg_orig, shape_500)

# Condition B: harmonized ±500 kb for both
counts_B = classify_jagwas(jag, enigma_500, shape_500)

# Condition C: harmonized ±1 Mb for both
counts_C = classify_jagwas(jag, enigma_1000, shape_1000)

total = len(jag)
conditions = [
    ('A: Original (ENIGMA=LD-expanded, Shape=±500kb)', counts_A),
    ('B: Harmonized ±500kb (both)',                    counts_B),
    ('C: Harmonized ±1Mb  (both)',                     counts_C),
]

print(f"\n{'Category':20s}", end='')
for label, _ in conditions:
    short = label.split(':')[0]
    print(f"  {short:>10s}", end='')
print()

for cat in CAT_ORDER:
    print(f"{cat:20s}", end='')
    for _, counts in conditions:
        n = counts[cat]
        print(f"  {n:4d} ({100*n/total:.0f}%)", end='')
    print()

# Save summary CSV
rows = []
for cond_label, counts in conditions:
    for cat in CAT_ORDER:
        rows.append({'condition': cond_label, 'category': cat,
                     'n': int(counts[cat]), 'pct': round(100 * counts[cat] / total, 1)})
sens_df = pd.DataFrame(rows)
out_csv = RES_DIR / 'novelty_window_sensitivity.csv'
sens_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── STEP 6: Figure — grouped bar by condition ──────────────────────────────────
cond_labels = ['A\n(original)', 'B\n(±500 kb)', 'C\n(±1 Mb)']
cat_colors  = {c: col for c, col in zip(CAT_ORDER,
               [COL_NOVEL, COL_SHAPE_ONLY, COL_ENIGMA_ONLY, COL_BOTH])}

fig, ax = plt.subplots(figsize=(MM(140), MM(90)))
n_conds = len(conditions)
n_cats  = len(CAT_ORDER)
x = np.arange(n_conds)
width = 0.18
offsets = np.linspace(-(n_cats - 1) * width / 2, (n_cats - 1) * width / 2, n_cats)

for i, (cat, offset) in enumerate(zip(CAT_ORDER, offsets)):
    vals = [counts[cat] for _, counts in conditions]
    bars = ax.bar(x + offset, vals, width=width, label=cat, color=cat_colors[cat],
                  edgecolor='white', linewidth=0.6)
    for bar, v in zip(bars, vals):
        if v > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(v), ha='center', va='bottom', fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels(cond_labels, fontsize=FS_TICK)
ax.set_ylabel('Number of JAGWAS AL- loci', fontsize=FS_LABEL)
ax.set_title('Novelty classification under harmonized window definitions\n'
             f'(N={total} JAGWAS loci)',
             fontsize=FS_TITLE, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, total * 0.80)

legend_patches = [mpatches.Patch(color=cat_colors[c], label=c) for c in CAT_ORDER]
ax.legend(handles=legend_patches, fontsize=FS_TICK, frameon=False,
          loc='upper right', ncol=2)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'novelty_window_sensitivity')
plt.close(fig)
print(f"Saved figure: {FIG_DIR}/novelty_window_sensitivity.pdf/.png")

# ── Print key novelty numbers for manuscript ───────────────────────────────────
print("\n=== KEY NUMBERS FOR MANUSCRIPT ===")
for label, counts in conditions:
    n_nov = counts['Novel']
    print(f"{label.split('(')[0].strip()}: Novel = {n_nov}/{total} ({100*n_nov/total:.1f}%)")

print("\nDone: 12c_novelty_window_sensitivity.py")
