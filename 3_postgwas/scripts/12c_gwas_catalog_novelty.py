"""
12c_gwas_catalog_novelty.py
GWAS Catalog classification of the 180 novel JAGWAS loci (novel to ENIGMA volume
AND Shape GWAS from 12b).

Strategy: interval-based matching — for each AL- locus (chr, start, end) search
cross_trait_all_categories.csv by SNP position (chr, bp). This catches tagged SNPs
that carry catalog hits even when the lead SNP is new.

4-way classification (JAGWAS perspective):
  Truly novel            — no GWAS Catalog hit anywhere in the locus interval
  Non-brain-imaging only — catalog hits exist but NO 'Brain/Imaging' Organ_Category
  Brain/Imaging GWAS only— has 'Brain/Imaging' hits, NO other organ categories
  Brain/Imaging + other  — has 'Brain/Imaging' AND at least one other organ category

Outputs:
  results/cross_region/novel_loci_gwas_catalog_classification.csv
  figures/cross_region/gwas_catalog_novelty_donut.pdf/.png   (180 novel loci, 4-way)
  figures/cross_region/gwas_catalog_novelty_bar276.pdf/.png  (276 loci, Novel sub-divided)
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
RES_CROSS   = Path(__file__).parents[1] / 'results'  / 'cross_region'
RES_CTRAIT  = Path(__file__).parents[1] / 'results'  / 'cross_trait'
FIG_DIR     = Path(__file__).parents[1] / 'figures'  / 'cross_region'
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOVELTY_3WAY = RES_CROSS  / 'jagwas_novelty_3way.csv'
CATALOG_CSV  = RES_CTRAIT / 'cross_trait_all_categories.csv'

# ── Colours ───────────────────────────────────────────────────────────────────
# 4 sub-categories for Novel loci (warm → cool = more novelty → less novelty)
COL_TRULY_NOVEL   = '#D62728'   # dark red   — no catalog precedent
COL_NONBRAIN      = '#FF7F0E'   # orange     — non-brain-imaging catalog only
COL_BRAIN_ONLY    = '#AEC7E8'   # light blue — brain/imaging, no other
COL_BRAIN_OTHER   = '#6BAED6'   # medium blue— brain/imaging + other traits
# 3 non-Novel categories (same as 12b)
COL_SHAPE_ONLY    = '#9370DB'   # purple
COL_ENIGMA_ONLY   = '#3182BD'   # blue
COL_BOTH          = '#2CA02C'   # green

# ── STEP 1: Load data ──────────────────────────────────────────────────────────
print("Loading data ...")
jag3 = pd.read_csv(NOVELTY_3WAY)
jag3['chr'] = jag3['chr'].astype(str)
novel = jag3[jag3['category_3way'] == 'Novel'].copy().reset_index(drop=True)
print(f"Novel loci (not ENIGMA, not Shape): {len(novel)}")

ct = pd.read_csv(CATALOG_CSV)
ct['chr'] = ct['chr'].astype(str)   # match jag3 chr type
ct_uniq = ct[['chr', 'bp', 'snp', 'Trait', 'Organ_Category', 'PMID']].drop_duplicates()
print(f"GWAS Catalog entries: {len(ct_uniq)} unique (chr, bp, snp, Trait, Organ_Category)")

# ── STEP 2: Interval-based matching ───────────────────────────────────────────
print("\nMatching novel loci to GWAS Catalog by interval (chr:start-end) ...")

rows = []
for _, locus in novel.iterrows():
    hits = ct_uniq[
        (ct_uniq['chr'] == locus['chr']) &
        (ct_uniq['bp']  >= locus['start']) &
        (ct_uniq['bp']  <= locus['end'])
    ]

    n_snps   = hits['snp'].nunique()
    n_traits = hits['Trait'].nunique()
    n_pmids  = hits['PMID'].nunique()
    organs   = sorted(hits['Organ_Category'].dropna().unique())
    traits_ex = '; '.join(sorted(hits['Trait'].dropna().unique())[:5])

    # Classify
    if len(hits) == 0:
        cat = 'Truly novel'
    elif 'Brain/Imaging' not in organs:
        cat = 'Non-brain-imaging only'
    elif len(organs) == 1:
        cat = 'Brain/Imaging GWAS only'
    else:
        cat = 'Brain/Imaging + other traits'

    rows.append({
        'al_id':            locus['al_id'],
        'chr':              locus['chr'],
        'start':            locus['start'],
        'end':              locus['end'],
        'best_lead_snp':    locus['best_lead_snp'],
        'best_p':           locus['best_p'],
        'n_regions':        locus['n_regions'],
        'regions':          locus['regions'],
        'catalog_class':    cat,
        'n_catalog_snps':   n_snps,
        'n_catalog_traits': n_traits,
        'n_catalog_pmids':  n_pmids,
        'organ_categories': '|'.join(organs) if organs else '',
        'example_traits':   traits_ex,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(RES_CROSS / 'novel_loci_gwas_catalog_classification.csv', index=False)
print(f"Saved: {RES_CROSS}/novel_loci_gwas_catalog_classification.csv")

# ── STEP 3: Summary ───────────────────────────────────────────────────────────
cat_order_novel = [
    'Truly novel',
    'Non-brain-imaging only',
    'Brain/Imaging GWAS only',
    'Brain/Imaging + other traits',
]
novel_counts = out_df['catalog_class'].value_counts().reindex(cat_order_novel, fill_value=0)
total_novel  = len(out_df)         # 180
total_all    = len(jag3)           # 276

print(f"\n=== GWAS Catalog classification of {total_novel} novel loci ===")
for cat, n in novel_counts.items():
    print(f"  {cat:35s}: {n:4d} / {total_novel}  "
          f"({100*n/total_novel:.1f}% of novel, {100*n/total_all:.1f}% of all 276)")

n_no_brain = novel_counts['Truly novel'] + novel_counts['Non-brain-imaging only']
print(f"\n  Combined 'no brain imaging GWAS precedent' : "
      f"{n_no_brain} / {total_novel}  "
      f"({100*n_no_brain/total_novel:.1f}% of novel, "
      f"{100*n_no_brain/total_all:.1f}% of all 276)")

print("\n  Top 5 truly novel loci (by best_p):")
truly = out_df[out_df['catalog_class'] == 'Truly novel'].sort_values('best_p')
for _, r in truly.head(5).iterrows():
    print(f"    {r['al_id']}  chr{r['chr']} {r['best_lead_snp']}  p={float(r['best_p']):.2e}"
          f"  regions={r['regions']}")

# ── STEP 4: Figure A — Donut chart (180 novel loci, 4-way) ────────────────────
cat_colors_novel = {
    'Truly novel':              COL_TRULY_NOVEL,
    'Non-brain-imaging only':   COL_NONBRAIN,
    'Brain/Imaging GWAS only':  COL_BRAIN_ONLY,
    'Brain/Imaging + other traits': COL_BRAIN_OTHER,
}

fig1, ax1 = plt.subplots(figsize=(MM(100), MM(90)))

sizes  = [novel_counts[c] for c in cat_order_novel]
colors = [cat_colors_novel[c] for c in cat_order_novel]

wedge_props = dict(width=0.45, edgecolor='white', linewidth=1.0)
wedges, texts = ax1.pie(
    sizes, colors=colors,
    startangle=90,
    wedgeprops=wedge_props,
)

# Annotate each wedge with count + percent
for wedge, n in zip(wedges, sizes):
    angle  = (wedge.theta1 + wedge.theta2) / 2
    r_mid  = 1 - wedge_props['width'] / 2   # midpoint radius of ring
    x = r_mid * np.cos(np.deg2rad(angle))
    y = r_mid * np.sin(np.deg2rad(angle))
    pct = 100 * n / total_novel
    if pct > 4:   # only label if slice is wide enough
        ax1.text(x, y, f'{n}\n({pct:.0f}%)',
                 ha='center', va='center', fontsize=7, fontweight='bold', color='white')

# Centre annotation
ax1.text(0, 0, f'N={total_novel}\nnovel loci',
         ha='center', va='center', fontsize=8, fontweight='bold', color='#333333')

# Legend
legend_patches = [mpatches.Patch(color=cat_colors_novel[c], label=c)
                  for c in cat_order_novel]
fig1.legend(handles=legend_patches, loc='lower center',
            bbox_to_anchor=(0.5, 0.0), ncol=1,
            fontsize=FS_TICK, frameon=False)

ax1.set_title(f'GWAS Catalog classification of {total_novel} novel JAGWAS loci\n'
              f'(novel to ENIGMA volume + Shape GWAS)',
              fontsize=FS_TITLE, fontweight='bold')

plt.tight_layout(rect=[0, 0.22, 1, 1])
save_mpl(fig1, FIG_DIR / 'gwas_catalog_novelty_donut')
plt.close(fig1)

# ── STEP 5: Figure B — Stacked bar (276 loci, Novel sub-divided) ──────────────
# Order from left: 4 novel sub-cats, then Shape-only, ENIGMA-only, ENIGMA+Shape
non_novel_counts = jag3[jag3['category_3way'] != 'Novel']['category_3way'].value_counts()

cat_order_all = cat_order_novel + ['Shape-only', 'ENIGMA-only', 'ENIGMA + Shape']
counts_all    = (list(novel_counts[cat_order_novel]) +
                 [int(non_novel_counts.get('Shape-only', 0)),
                  int(non_novel_counts.get('ENIGMA-only', 0)),
                  int(non_novel_counts.get('ENIGMA + Shape', 0))])
colors_all    = [cat_colors_novel[c] for c in cat_order_novel] + \
                [COL_SHAPE_ONLY, COL_ENIGMA_ONLY, COL_BOTH]

labels_all = [
    'Truly novel\n(no catalog)',
    'Non-brain\nimaging only',
    'Brain/Imaging\nonly',
    'Brain/Imaging\n+ other',
    'Shape-only',
    'ENIGMA-only',
    'ENIGMA\n+ Shape',
]

fig2, ax2 = plt.subplots(figsize=(MM(155), MM(70)))

left = 0.0
bar_h = 0.5
for n, col in zip(counts_all, colors_all):
    pct = n / total_all
    ax2.barh(0, pct, left=left, height=bar_h,
             color=col, edgecolor='white', linewidth=0.8)
    if pct > 0.025:
        ax2.text(left + pct / 2, 0, f'{n}\n({100*pct:.0f}%)',
                 ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    left += pct

ax2.set_xlim(0, 1)
ax2.set_yticks([])
ax2.set_xlabel('Fraction of 276 JAGWAS AL- loci', fontsize=FS_LABEL)
ax2.set_title('JAGWAS loci (N=276): novelty vs ENIGMA/Shape + GWAS Catalog classification',
              fontsize=FS_TITLE, fontweight='bold')
ax2.spines[['top', 'right', 'left']].set_visible(False)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

legend_patches2 = [mpatches.Patch(color=col, label=lbl.replace('\n', ' '))
                   for col, lbl in zip(colors_all, labels_all)]
fig2.legend(handles=legend_patches2, loc='lower center',
            bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=FS_TICK, frameon=False)

plt.tight_layout(rect=[0, 0.22, 1, 1])
save_mpl(fig2, FIG_DIR / 'gwas_catalog_novelty_bar276')
plt.close(fig2)

print("\nDone: 12c_gwas_catalog_novelty.py")
