"""
12b_loci_novelty_3way.py
3-way loci novelty: JAGWAS AL- loci vs ENIGMA volume GWAS vs Shape GWAS.

Claim supported: Claim 1 — BREs are richer phenotypes that discover more loci
than either volumetric or shape GWAS.

JAGWAS loci (276 AL-): already classified vs ENIGMA in Script 12.
Shape GWAS loci: from Garcia-Marin 2024 shape paper TableS5_Locus.xlsx
  - Sheet 'structure': 363 structure-level shape features
  - Lead SNP chr + pos only → ±500 kb proxy window (Zhao 2022 published only
    lead SNPs without LD-expanded intervals; FUMA's own default merge distance
    is 250 kb, but we use a wider proxy here because we lack LD info)

3-way categories per JAGWAS AL- locus:
  Novel           — no overlap with ENIGMA AND no overlap with Shape
  ENIGMA-only     — overlaps ENIGMA, not Shape
  Shape-only      — overlaps Shape, not ENIGMA
  ENIGMA + Shape  — overlaps both

Outputs:
  results/cross_region/jagwas_novelty_3way.csv
  figures/cross_region/loci_novelty_3way_bar.pdf/.png
  figures/cross_region/loci_novelty_3way_venn.pdf/.png  (matplotlib_venn or fallback)
"""

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

# ── Paths ──────────────────────────────────────────────────────────────────────
SHAPE_SUPP = Path(cfg.postgwas.shape_gwas)

RES_DIR  = Path(__file__).parents[1] / 'results'  / 'cross_region'
FIG_DIR  = Path(__file__).parents[1] / 'figures'  / 'cross_region'
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# JAGWAS novelty file from Script 12 (has enigma_overlap_all)
JAGWAS_NOVELTY = RES_DIR / 'jagwas_novelty_classification.csv'

# Window around each Shape lead SNP (bp)
WINDOW_BP = 500_000

# ── Colours ───────────────────────────────────────────────────────────────────
COL_NOVEL        = '#E6550D'   # orange  — novel to all
COL_ENIGMA_ONLY  = '#3182BD'   # blue    — ENIGMA volume only
COL_SHAPE_ONLY   = '#9370DB'   # purple  — Shape only
COL_BOTH         = '#2CA02C'   # green   — ENIGMA + Shape

# ── STEP 1: Load Shape GWAS loci ──────────────────────────────────────────────
print("Loading Shape GWAS loci from TableS5_Locus.xlsx (structure sheet) ...")
# Row 0 = long title; row 1 = true column headers (ID, uniqID, rsID, chr, pos, ...)
shape_df = pd.read_excel(SHAPE_SUPP, sheet_name='structure', header=1)

# Keep only chr / pos, drop rows with missing values
shape_df['chr'] = pd.to_numeric(shape_df['chr'], errors='coerce')
shape_df['pos'] = pd.to_numeric(shape_df['pos'], errors='coerce')
shape_df = shape_df.dropna(subset=['chr', 'pos']).copy()
shape_df['chr'] = shape_df['chr'].astype(int).astype(str)
shape_df['pos'] = shape_df['pos'].astype(int)

# Create interval
shape_df['start'] = (shape_df['pos'] - WINDOW_BP).clip(lower=0)
shape_df['end']   =  shape_df['pos'] + WINDOW_BP

# Deduplicate: a SNP may appear multiple times (different phenotypes but same locus)
shape_unique = shape_df[['chr', 'start', 'end', 'pos']].drop_duplicates()

# Merge overlapping shape intervals (same greedy algorithm as Scripts 06/12)
shape_unique = shape_unique.sort_values(['chr', 'start']).reset_index(drop=True)

def greedy_merge(df_in):
    """Greedy interval merge — same as Scripts 06 and 12."""
    df = df_in.copy()
    df['chr'] = df['chr'].astype(str)
    df = df.sort_values(['chr', 'start']).reset_index(drop=True)
    merged = []
    cur_chr, cur_start, cur_end = None, None, None
    cur_count = 0
    for _, row in df.iterrows():
        if row['chr'] != cur_chr or row['start'] > cur_end:
            if cur_chr is not None:
                merged.append({'chr': cur_chr, 'start': cur_start,
                               'end': cur_end, 'n_raw': cur_count})
            cur_chr, cur_start, cur_end = row['chr'], row['start'], row['end']
            cur_count = 1
        else:
            cur_end = max(cur_end, row['end'])
            cur_count += 1
    if cur_chr is not None:
        merged.append({'chr': cur_chr, 'start': cur_start,
                       'end': cur_end, 'n_raw': cur_count})
    return pd.DataFrame(merged)

shape_agg = greedy_merge(shape_unique)
print(f"Shape GWAS: {len(shape_df)} lead-SNP entries → "
      f"{len(shape_unique)} unique intervals → "
      f"{len(shape_agg)} merged loci (±{WINDOW_BP//1000} kb windows)")


# ── STEP 2: Load JAGWAS loci (with ENIGMA classification from Script 12) ──────
print("\nLoading JAGWAS novelty classification ...")
jag = pd.read_csv(JAGWAS_NOVELTY)
jag['chr'] = jag['chr'].astype(str)   # ensure str for overlap matching
print(f"JAGWAS AL- loci: {len(jag)}")
print(f"  ENIGMA-novel (novel_vs_all=True): {jag['novel_vs_all'].sum()}")


# ── STEP 3: Classify JAGWAS loci vs Shape ─────────────────────────────────────
def overlaps_shape(jag_chr, jag_start, jag_end, shape_df):
    """Returns True if JAGWAS locus overlaps any merged Shape locus."""
    same_chr = shape_df[shape_df['chr'] == str(jag_chr)]
    if len(same_chr) == 0:
        return False
    overlap = ((same_chr['start'] <= jag_end) & (same_chr['end'] >= jag_start))
    return overlap.any()

jag['shape_overlap'] = jag.apply(
    lambda r: overlaps_shape(r['chr'], r['start'], r['end'], shape_agg), axis=1)

# 3-way category
def three_way(row):
    e = not row['novel_vs_all']     # True = ENIGMA-known
    s = row['shape_overlap']        # True = Shape-known
    if e and s:
        return 'ENIGMA + Shape'
    elif e and not s:
        return 'ENIGMA-only'
    elif not e and s:
        return 'Shape-only'
    else:
        return 'Novel'

jag['category_3way'] = jag.apply(three_way, axis=1)

# Summary
counts = jag['category_3way'].value_counts()
cat_order = ['Novel', 'Shape-only', 'ENIGMA-only', 'ENIGMA + Shape']
counts = counts.reindex(cat_order, fill_value=0)
total  = len(jag)
print(f"\n3-way loci classification (N={total}):")
for cat, n in counts.items():
    print(f"  {cat:20s}: {n:4d}  ({100*n/total:.1f}%)")

# Shape overlap counts
n_shape_known = jag['shape_overlap'].sum()
n_shape_novel = (~jag['shape_overlap']).sum()
print(f"\nShape-known: {n_shape_known} ({100*n_shape_known/total:.1f}%)")
print(f"Shape-novel: {n_shape_novel} ({100*n_shape_novel/total:.1f}%)")

# Save
jag.to_csv(RES_DIR / 'jagwas_novelty_3way.csv', index=False)
print(f"\nSaved: {RES_DIR}/jagwas_novelty_3way.csv")


# ── STEP 4: Figure A — Stacked horizontal bar (proportion) ────────────────────
cat_colors = {
    'Novel':          COL_NOVEL,
    'Shape-only':     COL_SHAPE_ONLY,
    'ENIGMA-only':    COL_ENIGMA_ONLY,
    'ENIGMA + Shape': COL_BOTH,
}

fig1, ax1 = plt.subplots(figsize=(MM(140), MM(65)))

left = 0.0
bar_height = 0.5
for cat in cat_order:
    n  = counts[cat]
    pct = n / total
    ax1.barh(0, pct, left=left, height=bar_height,
             color=cat_colors[cat], edgecolor='white', linewidth=0.8)
    if pct > 0.03:   # only label if wide enough
        ax1.text(left + pct / 2, 0, f'{n}\n({100*pct:.0f}%)',
                 ha='center', va='center', fontsize=6.5, fontweight='bold',
                 color='white')
    left += pct

ax1.set_xlim(0, 1)
ax1.set_yticks([])
ax1.set_xlabel('Fraction of 276 JAGWAS AL- loci', fontsize=FS_LABEL)
ax1.set_title(
    'JAGWAS loci novelty: 3-way comparison vs ENIGMA volume + Shape GWAS',
    fontsize=FS_TITLE, fontweight='bold',
)
ax1.spines[['top', 'right', 'left']].set_visible(False)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

legend_patches = [mpatches.Patch(color=cat_colors[c], label=c) for c in cat_order]
# Place legend inside the reserved bottom strip — well below the bar
fig1.legend(handles=legend_patches, loc='lower center',
            bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=FS_TICK, frameon=False)

plt.tight_layout(rect=[0, 0.18, 1, 1])
save_mpl(fig1, FIG_DIR / 'loci_novelty_3way_bar')
plt.close(fig1)


# ── STEP 5: Figure B — Grouped vertical bar (absolute counts) ─────────────────
fig2, ax2 = plt.subplots(figsize=(MM(110), MM(80)))

x = np.arange(len(cat_order))
bars = ax2.bar(x, [counts[c] for c in cat_order],
               color=[cat_colors[c] for c in cat_order],
               width=0.6, edgecolor='white', linewidth=0.8)

for bar, cat in zip(bars, cat_order):
    n   = counts[cat]
    pct = 100 * n / total
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 2,
             f'{n}\n({pct:.0f}%)',
             ha='center', va='bottom', fontsize=7, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(cat_order, fontsize=7)
ax2.set_ylabel('Number of AL- loci', fontsize=FS_LABEL)
ax2.set_title(
    f'JAGWAS loci (N={total}): overlap with\nENIGMA volume + Shape GWAS',
    fontsize=FS_TITLE, fontweight='bold',
)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_ylim(0, max(counts) * 1.20)

plt.tight_layout()
save_mpl(fig2, FIG_DIR / 'loci_novelty_3way_counts')
plt.close(fig2)


# ── STEP 6: Figure C — Proper 3-set Venn diagram ─────────────────────────────
# All JAGWAS-containing regions use JAGWAS perspective (consistent with bar chart).
# Non-JAGWAS regions use ENIGMA/Shape perspective.
#
# Abc  = 180  JAGWAS only                   (JAGWAS perspective)
# aBc  = ?    ENIGMA only, not JAGWAS/Shape  (ENIGMA perspective)
# ABc  = 48   JAGWAS+ENIGMA, not Shape       (JAGWAS perspective = ENIGMA perspective)
# abC  = 6    Shape only                     (Shape perspective)
# AbC  = 21   JAGWAS+Shape, not ENIGMA       (JAGWAS perspective → matches bar chart)
# aBC  = ?    ENIGMA+Shape, not JAGWAS       (ENIGMA perspective)
# ABC  = 27   all three                      (JAGWAS perspective → matches bar chart)
# → JAGWAS total: 180+48+21+27 = 276 ✓

# Load ENIGMA aggregate loci (saved by Script 12)
enigma_agg = pd.read_csv(RES_DIR / 'enigma_aggregate_loci.csv')
enigma_agg['chr'] = enigma_agg['chr'].astype(str)

def overlaps_ref(chr_, s, e, ref_df):
    sc = ref_df[ref_df['chr'] == str(chr_)]
    if len(sc) == 0:
        return False
    return ((sc['start'] <= e) & (sc['end'] >= s)).any()

# Tag ENIGMA loci vs JAGWAS and Shape
enigma_agg['in_jagwas'] = enigma_agg.apply(
    lambda r: overlaps_ref(r['chr'], r['start'], r['end'], jag), axis=1)
enigma_agg['in_shape']  = enigma_agg.apply(
    lambda r: overlaps_ref(r['chr'], r['start'], r['end'], shape_agg), axis=1)

# Tag Shape loci for abC (Shape only, not in JAGWAS or ENIGMA)
shape_agg['in_jagwas'] = shape_agg.apply(
    lambda r: overlaps_ref(r['chr'], r['start'], r['end'], jag), axis=1)
shape_agg['in_enigma'] = shape_agg.apply(
    lambda r: overlaps_ref(r['chr'], r['start'], r['end'], enigma_agg), axis=1)

# JAGWAS perspective (consistent with bar chart)
venn_Abc = counts['Novel']       # 180
venn_ABc = counts['ENIGMA-only'] # 48
venn_AbC = counts['Shape-only']  # 21  ← matches bar chart
venn_ABC = counts['ENIGMA + Shape']  # 27  ← matches bar chart

# ENIGMA perspective (loci not overlapping JAGWAS)
venn_aBc = int((~enigma_agg['in_jagwas'] & ~enigma_agg['in_shape']).sum())  # ENIGMA only
venn_aBC = int((~enigma_agg['in_jagwas'] &  enigma_agg['in_shape']).sum())  # ENIGMA+Shape not J

# Shape perspective (Shape loci not overlapping JAGWAS or ENIGMA)
venn_abC = int((~shape_agg['in_jagwas'] & ~shape_agg['in_enigma']).sum())  # Shape only

print(f"\n=== Full 7-region Venn counts (owner-consistent) ===")
for lbl, val in [('Abc JAGWAS only', venn_Abc), ('aBc ENIGMA only', venn_aBc),
                 ('ABc JAGWAS+ENIGMA', venn_ABc), ('abC Shape only', venn_abC),
                 ('AbC JAGWAS+Shape', venn_AbC), ('aBC ENIGMA+Shape', venn_aBC),
                 ('ABC all three', venn_ABC)]:
    print(f"  {lbl:25s}: {val}")
n_j = venn_Abc + venn_ABc + venn_AbC + venn_ABC
n_e = venn_aBc + venn_ABc + venn_aBC + venn_ABC
n_s = venn_abC + venn_AbC + venn_aBC + venn_ABC
print(f"  JAGWAS total: {n_j}  (expected 276)")
print(f"  ENIGMA total: {n_e}  (expected ~328)")
print(f"  Shape total:  {n_s}  (expected 57)")

try:
    from matplotlib_venn import venn3, venn3_circles

    fig3, ax3 = plt.subplots(figsize=(MM(120), MM(115)))
    v = venn3(
        subsets=(venn_Abc, venn_aBc, venn_ABc,
                 venn_abC, venn_AbC, venn_aBC, venn_ABC),
        set_labels=(f'JAGWAS\n(276)', f'ENIGMA\nVolume\n({n_e})',
                    f'Shape\n({n_s})'),
        set_colors=(COL_NOVEL, COL_ENIGMA_ONLY, COL_SHAPE_ONLY),
        alpha=0.55,
        ax=ax3,
    )
    # Bold the JAGWAS-only label (most important number)
    for label_id, txt in [('100', str(venn_Abc)), ('010', str(venn_aBc)),
                          ('110', str(venn_ABc)), ('001', str(venn_abC)),
                          ('101', str(venn_AbC)), ('011', str(venn_aBC)),
                          ('111', str(venn_ABC))]:
        lbl = v.get_label_by_id(label_id)
        if lbl:
            lbl.set_fontsize(8)
    # Make set labels slightly larger
    for lbl in v.set_labels:
        if lbl:
            lbl.set_fontsize(8)
            lbl.set_fontweight('bold')

    ax3.set_title('Loci overlap: JAGWAS / ENIGMA volume / Shape GWAS\n'
                  f'(180/276 JAGWAS loci are novel to both)',
                  fontsize=FS_TITLE, fontweight='bold')
    plt.tight_layout()
    save_mpl(fig3, FIG_DIR / 'loci_novelty_3way_venn')
    plt.close(fig3)
    print("Done: Venn figure (matplotlib_venn, proper 3-set)")

except ImportError:
    print("matplotlib_venn not available — skipping Venn figure")

print("Done: 12b_loci_novelty_3way.py")
