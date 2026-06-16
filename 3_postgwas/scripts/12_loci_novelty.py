"""
12_loci_novelty.py
Loci novelty analysis — how many JAGWAS loci go beyond ENIGMA volume GWAS?

Compares 276 JAGWAS aggregate loci (AL-) against Garcia-Marin et al. 2024
(Nature Genetics) subcortical volume + ICV GWAS (FUMA outputs available
per region: Amygdala, Brainstem, Caudate, Hippocampus, ICV, Pallidium,
Putamen, Thalamus, Ventral DC).

Method:
  1. Load ENIGMA loci from all 10 regions (Accumbens FUMA derived locally;
     others from ENIGMA 2024) → merge into ENIGMA aggregate loci
     (same greedy interval-overlap merge used in Script 06).
  2. Classify each JAGWAS AL- locus as "ENIGMA-known" (overlaps an ENIGMA
     aggregate locus on the same chromosome) or "Novel" (no overlap).
  3. Produce:
     (A) Overall summary bar: Novel vs ENIGMA-known fractions of 276 loci
     (B) Per-JAGWAS-region stacked barplot: for loci first discovered in that
         region, what fraction are novel vs known?
     (C) Novelty by region-specificity: are pan-regional loci more likely to
         be ENIGMA-known (volume loci)? Are single-region loci more novel?

Supports paper Claim 1:
  BRE phenotype + JAGWAS discovers genuinely novel genetic signal beyond
  the 254-locus ENIGMA volume GWAS.

Outputs:
  results/cross_region/enigma_aggregate_loci.csv
  results/cross_region/jagwas_novelty_classification.csv
  figures/cross_region/loci_novelty_summary.pdf/.png
  figures/cross_region/loci_novelty_per_region.pdf/.png
  figures/cross_region/novelty_by_nregions.pdf/.png
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL, FS_TITLE

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
ENIGMA_DIR = Path(cfg.postgwas.enigma_dir)
# Project-local Accumbens FUMA loci. ENIGMA 2024 did not publish FUMA output for
# Accumbens; we derived these loci ourselves using an in-house FUMA-style
# clumping tool on the Accumbens_QCed.tab.gz summary statistics (UKB 35k LD
# panel; see docs/data_paths.md §10b for provenance).
ACCUMBENS_DIR = (Path(__file__).parents[1]
                 / "external_clumping" / "accumbens")
RES_DIR    = Path(__file__).parents[1] / "results" / "cross_region"
FIG_DIR    = Path(__file__).parents[1] / "figures" / "cross_region"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

JAGWAS_LOCI = RES_DIR / "aggregate_loci.csv"

# All ENIGMA regions (Accumbens FUMA derived locally; others from ENIGMA 2024)
ENIGMA_REGIONS = [
    'Accumbens', 'Amygdala', 'Brainstem', 'Caudate', 'Hippocampus',
    'ICV', 'Pallidium', 'Putamen', 'Thalamus', 'Ventral',
]
# Subcortical-only (excluding ICV) for sensitivity
ENIGMA_SUBCORTICAL = [r for r in ENIGMA_REGIONS if r != 'ICV']

# Colour palette
COL_NOVEL   = '#E6550D'   # orange — novel beyond ENIGMA
COL_KNOWN   = '#3182BD'   # blue   — replicated in ENIGMA
COL_BOTH    = '#74C476'   # green  (unused but reserved)

# ── REGION ORDER (JAGWAS) ─────────────────────────────────────────────────────
BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def sort_key_region(name):
    for i, b in enumerate(BASE_ORDER):
        bshort = b.replace('-area', '').replace('_Proper', '').replace('_', ' ')
        if bshort.lower() in name.lower():
            return (i, 0 if 'L.' in name else 1)
    for i, m in enumerate(MIDLINE):
        mshort = m.replace('_', ' ')
        if 'Stem' in name or 'BrStem' in name:
            return (len(BASE_ORDER), 0)
        if 'CSF' in name:
            return (len(BASE_ORDER) + 1, 0)
    return (99, 0)

# ── STEP 1: LOAD AND MERGE ENIGMA LOCI ───────────────────────────────────────
print("Loading Garcia-Marin 2024 (ENIGMA) loci ...")
enigma_rows = []
for region in ENIGMA_REGIONS:
    region_dir = ACCUMBENS_DIR if region == 'Accumbens' else ENIGMA_DIR / region
    grl = region_dir / 'GenomicRiskLoci.txt'
    if not grl.exists():
        print(f"  WARNING: {region}/GenomicRiskLoci.txt not found — skipping")
        continue
    df = pd.read_csv(grl, sep='\t')
    df['enigma_region'] = region
    enigma_rows.append(df[['chr', 'start', 'end', 'rsID', 'p', 'enigma_region']])
    print(f"  {region}: {len(df)} loci")

enigma_raw = pd.concat(enigma_rows, ignore_index=True)
print(f"\nENIGMA raw total: {len(enigma_raw)} loci across {len(ENIGMA_REGIONS)} regions")


def greedy_merge(df_in):
    """
    Greedy interval merge: same algorithm as Script 06.
    Returns DataFrame with columns: chr, start, end, n_raw, source_regions
    """
    df = df_in.copy()
    df['chr'] = df['chr'].astype(str)
    df = df.sort_values(['chr', 'start']).reset_index(drop=True)

    merged = []
    cur_chr, cur_start, cur_end = None, None, None
    cur_regions = []
    cur_count = 0

    for _, row in df.iterrows():
        if row['chr'] != cur_chr or row['start'] > cur_end:
            if cur_chr is not None:
                merged.append({'chr': cur_chr, 'start': cur_start, 'end': cur_end,
                               'n_raw': cur_count, 'regions': ';'.join(sorted(set(cur_regions)))})
            cur_chr, cur_start, cur_end = row['chr'], row['start'], row['end']
            cur_regions = [row.get('enigma_region', '')]
            cur_count = 1
        else:
            cur_end = max(cur_end, row['end'])
            cur_regions.append(row.get('enigma_region', ''))
            cur_count += 1

    if cur_chr is not None:
        merged.append({'chr': cur_chr, 'start': cur_start, 'end': cur_end,
                       'n_raw': cur_count, 'regions': ';'.join(sorted(set(cur_regions)))})
    return pd.DataFrame(merged)


enigma_agg = greedy_merge(enigma_raw)
print(f"ENIGMA aggregate loci (after merge): {len(enigma_agg)}")
enigma_agg.to_csv(RES_DIR / 'enigma_aggregate_loci.csv', index=False)
print(f"Saved: {RES_DIR}/enigma_aggregate_loci.csv")

# Subcortical-only merge (sensitivity, no ICV)
enigma_sub_raw = enigma_raw[enigma_raw['enigma_region'] != 'ICV']
enigma_sub_agg = greedy_merge(enigma_sub_raw)
print(f"ENIGMA subcortical-only aggregate (no ICV): {len(enigma_sub_agg)} loci")

# ── STEP 2: LOAD JAGWAS AGGREGATE LOCI ───────────────────────────────────────
print("\nLoading JAGWAS aggregate loci ...")
jag = pd.read_csv(JAGWAS_LOCI)
jag['chr'] = jag['chr'].astype(str)
print(f"JAGWAS loci: {len(jag)}")


def overlaps_enigma(jag_chr, jag_start, jag_end, enigma_df):
    """Returns True if (jag_chr, jag_start, jag_end) overlaps any ENIGMA locus."""
    same_chr = enigma_df[enigma_df['chr'].astype(str) == str(jag_chr)]
    if same_chr.empty:
        return False
    # Overlap: jag_start <= enigma_end AND jag_end >= enigma_start
    overlap = ((same_chr['start'] <= jag_end) & (same_chr['end'] >= jag_start))
    return overlap.any()


# ── STEP 3: CLASSIFY JAGWAS LOCI ─────────────────────────────────────────────
print("\nClassifying JAGWAS loci ...")
jag['enigma_overlap_all']  = jag.apply(
    lambda r: overlaps_enigma(r['chr'], r['start'], r['end'], enigma_agg), axis=1)
jag['enigma_overlap_sub']  = jag.apply(
    lambda r: overlaps_enigma(r['chr'], r['start'], r['end'], enigma_sub_agg), axis=1)
jag['novel_vs_all']  = ~jag['enigma_overlap_all']
jag['novel_vs_sub']  = ~jag['enigma_overlap_sub']

n_novel_all  = jag['novel_vs_all'].sum()
n_known_all  = (~jag['novel_vs_all']).sum()
n_novel_sub  = jag['novel_vs_sub'].sum()
n_known_sub  = (~jag['novel_vs_sub']).sum()
n_total      = len(jag)

print(f"\n{'='*55}")
print(f"  JAGWAS AL- loci total:               {n_total}")
print(f"  Novel vs ENIGMA (all {len(ENIGMA_REGIONS)} regions):    {n_novel_all}  ({n_novel_all/n_total*100:.1f}%)")
print(f"  ENIGMA-known (all {len(ENIGMA_REGIONS)} regions):       {n_known_all}  ({n_known_all/n_total*100:.1f}%)")
print(f"  Novel vs ENIGMA (subcortical only):  {n_novel_sub}  ({n_novel_sub/n_total*100:.1f}%)")
print(f"  ENIGMA-known (subcortical only):     {n_known_sub}  ({n_known_sub/n_total*100:.1f}%)")
print(f"{'='*55}")

jag.to_csv(RES_DIR / 'jagwas_novelty_classification.csv', index=False)
print(f"Saved: {RES_DIR}/jagwas_novelty_classification.csv")

# ── STEP 4: PER-JAGWAS-REGION NOVELTY ────────────────────────────────────────
# For each JAGWAS region, count loci that appear in that region
# and classify them as novel vs known (using all-ENIGMA comparison)
print("\nComputing per-region novelty breakdown ...")

all_regions = set()
for regions_str in jag['regions'].dropna():
    for r in regions_str.split(';'):
        all_regions.add(r.strip())
all_regions = sorted(all_regions, key=sort_key_region)

region_novelty = []
for region in all_regions:
    mask = jag['regions'].apply(lambda x: region in str(x).split(';'))
    sub = jag[mask]
    if len(sub) == 0:
        continue
    novel = sub['novel_vs_all'].sum()
    known = (~sub['novel_vs_all']).sum()
    region_novelty.append(dict(
        region=region, n_total=len(sub),
        n_novel=novel, n_known=known,
        pct_novel=novel/len(sub)*100,
    ))

rn_df = pd.DataFrame(region_novelty)
rn_df.to_csv(RES_DIR / 'novelty_per_region.csv', index=False)
print(rn_df[['region', 'n_total', 'n_novel', 'n_known', 'pct_novel']].to_string(index=False))

# ── STEP 5: NOVELTY BY N_REGIONS (SHARING LEVEL) — 3-way ─────────────────────
print("\nNovelty by region-sharing level (3-way: Novel vs ENIGMA + Shape) ...")
# Load 3-way classification (Script 12b output)
novelty_3way = pd.read_csv(RES_DIR / 'jagwas_novelty_3way.csv',
                           usecols=['al_id', 'category_3way'])
jag_3way = jag.merge(novelty_3way, on='al_id', how='left')
jag_3way['n_regions'] = jag_3way['n_regions'].astype(int)
jag_3way['novel_3way'] = jag_3way['category_3way'] == 'Novel'

nreg_novelty = jag_3way.groupby('n_regions').agg(
    n_total=('al_id', 'count'),
    n_novel=('novel_3way', 'sum'),
).reset_index()
nreg_novelty['pct_novel'] = nreg_novelty['n_novel'] / nreg_novelty['n_total'] * 100
print(nreg_novelty.to_string(index=False))

# 3-way overall: 180 Novel / 276 total
N_NOVEL_3WAY = 180
N_TOTAL_3WAY = 276

# ── FIGURE A: OVERALL SUMMARY (DONUT + HORIZONTAL BARS) ──────────────────────
print("\nGenerating Figure A: overall novelty summary ...")

fig, axes = plt.subplots(1, 2, figsize=(MM(160), MM(80)))

# Left: Donut chart (all ENIGMA comparison)
ax0 = axes[0]
sizes  = [n_novel_all, n_known_all]
colors = [COL_NOVEL, COL_KNOWN]
labels = [f'Novel\n{n_novel_all} ({n_novel_all/n_total*100:.0f}%)',
          f'ENIGMA-known\n{n_known_all} ({n_known_all/n_total*100:.0f}%)']
wedges, texts = ax0.pie(
    sizes, colors=colors, startangle=90,
    wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 1.5},
)
ax0.set_title('JAGWAS loci\nvs ENIGMA 2024', fontsize=FS_LABEL, fontweight='bold')
ax0.text(0, 0, f'{n_total}', ha='center', va='center',
         fontsize=FS_TITLE, fontweight='bold', color='#333333')
ax0.text(0, -0.18, 'total loci', ha='center', va='center',
         fontsize=FS_TICK-1, color='#666666')
ax0.legend(wedges, labels, loc='lower center', fontsize=FS_TICK - 0.5,
           frameon=False, bbox_to_anchor=(0.5, -0.12), ncol=1)

# Right: Sensitivity comparison bar (all-ENIGMA vs subcortical-only)
ax1 = axes[1]
comparisons   = ['All ENIGMA\n(incl. ICV)', 'Subcortical\nonly']
novel_counts  = [n_novel_all, n_novel_sub]
known_counts  = [n_known_all, n_known_sub]
y = np.arange(2)
ax1.barh(y, [n/n_total for n in novel_counts], color=COL_NOVEL,
         height=0.5, label='Novel', edgecolor='none')
ax1.barh(y, [n/n_total for n in known_counts], left=[n/n_total for n in novel_counts],
         color=COL_KNOWN, height=0.5, label='ENIGMA-known', edgecolor='none')
for i, (nov, kno) in enumerate(zip(novel_counts, known_counts)):
    ax1.text(nov/n_total/2, i, f'{nov}', ha='center', va='center',
             fontsize=FS_TICK, fontweight='bold', color='white')
    ax1.text(nov/n_total + kno/n_total/2, i, f'{kno}', ha='center', va='center',
             fontsize=FS_TICK, fontweight='bold', color='white')
ax1.set_yticks(y)
ax1.set_yticklabels(comparisons, fontsize=FS_TICK)
ax1.set_xlabel('Fraction of 276 loci', fontsize=FS_LABEL)
ax1.set_xlim(0, 1)
ax1.set_title('Sensitivity: ENIGMA\ncomparison scope', fontsize=FS_LABEL, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(loc='lower right', fontsize=FS_TICK, frameon=False,
           bbox_to_anchor=(1.0, -0.02))

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'loci_novelty_summary')
plt.close()
print("Saved: figures/cross_region/loci_novelty_summary.pdf/.png")

# ── FIGURE B: PER-JAGWAS-REGION NOVELTY ──────────────────────────────────────
print("Generating Figure B: per-region novelty ...")

rn_plot = rn_df.set_index('region').reindex(
    [r for r in all_regions if r in rn_df['region'].values])
y = np.arange(len(rn_plot))

fig, ax = plt.subplots(figsize=(MM(160), MM(100)))
novel_f = rn_plot['n_novel'] / rn_plot['n_total']
known_f = rn_plot['n_known'] / rn_plot['n_total']

ax.barh(y, novel_f.values, color=COL_NOVEL, height=0.7,
        edgecolor='none', label='Novel beyond ENIGMA')
ax.barh(y, known_f.values, left=novel_f.values, color=COL_KNOWN, height=0.7,
        edgecolor='none', label='ENIGMA-known')

# Annotate n_total
for i, (region, row) in enumerate(rn_plot.iterrows()):
    ax.text(1.02, i, f"n={int(row['n_total'])}", va='center', ha='left',
            fontsize=FS_TICK - 1, color='#444444')

ax.set_yticks(y)
ax.set_yticklabels(rn_plot.index.tolist(), fontsize=FS_TICK)
ax.set_xlabel('Fraction of loci in region', fontsize=FS_LABEL)
ax.set_title('Per-region novelty vs Garcia-Marín 2024 ENIGMA volume GWAS',
             fontsize=FS_LABEL, fontweight='bold')
ax.set_xlim(0, 1)
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=FS_TICK, frameon=False,
          bbox_to_anchor=(1.0, 0.0))
plt.tight_layout()
save_mpl(fig, FIG_DIR / 'loci_novelty_per_region')
plt.close()
print("Saved: figures/cross_region/loci_novelty_per_region.pdf/.png")

# ── FIGURE C: NOVELTY BY REGION-SHARING LEVEL ─────────────────────────────────
print("Generating Figure C: novelty by sharing level ...")

fig, ax = plt.subplots(figsize=(MM(100), MM(75)))
x = nreg_novelty['n_regions'].values
pct_novel = nreg_novelty['pct_novel'].values
n_total_group = nreg_novelty['n_total'].values

bars = ax.bar(x, pct_novel, color=COL_NOVEL, edgecolor='none', width=0.7)
# Reference line: overall 3-way novelty (180/276 = 65%)
ax.axhline(N_NOVEL_3WAY / N_TOTAL_3WAY * 100, color='#333333', ls='--', lw=1,
           label=f'Overall {N_NOVEL_3WAY/N_TOTAL_3WAY*100:.0f}% novel (vs ENIGMA + Shape)')

prev_label_y = None
for bar, n in zip(bars, n_total_group):
    y_natural = bar.get_height() + 1.5
    # Ensure minimum vertical gap from previous label; reset if bar drops well below it
    if prev_label_y is not None and y_natural > prev_label_y - 3:
        y_pos = max(y_natural, prev_label_y + 5)
    else:
        y_pos = y_natural
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'n={n}', ha='center', va='bottom', fontsize=FS_TICK - 1,
            color='#444444')
    prev_label_y = y_pos

ax.set_xlabel('Number of regions where locus appears', fontsize=FS_LABEL)
ax.set_ylabel('% Novel beyond ENIGMA + Shape', fontsize=FS_LABEL)
ax.set_title('Region-specific loci are more novel', fontsize=FS_LABEL, fontweight='bold')
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=FS_TICK, frameon=False, loc='upper right',
          bbox_to_anchor=(1.0, 0.62))
plt.tight_layout()
save_mpl(fig, FIG_DIR / 'novelty_by_nregions')
plt.close()
print("Saved: figures/cross_region/novelty_by_nregions.pdf/.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║  SCRIPT 12 SUMMARY                                  ║
╠══════════════════════════════════════════════════════╣
║  ENIGMA 2024 raw loci ({len(ENIGMA_REGIONS)} regions): {len(enigma_raw):4d}              ║
║  ENIGMA 2024 aggregate loci:        {len(enigma_agg):4d}              ║
║                                                      ║
║  JAGWAS total AL- loci:             {n_total:4d}              ║
║  Novel vs all ENIGMA:               {n_novel_all:4d}  ({n_novel_all/n_total*100:.0f}%)         ║
║  ENIGMA-known:                      {n_known_all:4d}  ({n_known_all/n_total*100:.0f}%)         ║
║                                                      ║
║  Novel vs subcortical ENIGMA only:  {n_novel_sub:4d}  ({n_novel_sub/n_total*100:.0f}%)         ║
╚══════════════════════════════════════════════════════╝
""")
