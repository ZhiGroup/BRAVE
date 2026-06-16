"""
01_per_region_summary.py
Per-region loci and gene mapping summary — Table 1 of the JAGWAS paper.

Inputs (per region):
  GenomicRiskLoci.txt  — one row per merged locus
  leadSNPs.txt         — one row per lead SNP
  IndSigSNPs.txt       — one row per independent significant SNP
  genes.txt            — one row per mapped gene (positional/eQTL/CI)

Outputs:
  results/per_region/loci_summary.csv     — loci counts per region
  results/per_region/gene_summary.csv     — gene mapping counts per region
  results/per_region/table1_combined.csv  — merged Table 1
  figures/per_region/loci_counts.pdf/.png — horizontal barplot, loci per region
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
from fig_style import apply_mpl_style, save_mpl, MM

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
OUT_DIR  = Path(__file__).parents[1] / "results" / "per_region"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "per_region"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── REGION ORDER (bilateral pairs + midline) ──────────────────────────────────
BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def parse_region(folder: str):
    n = folder.strip()
    nl = n.lower()
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
    # special cases
    name = name.replace('Brain Stem or 4th Ventricle', 'Brain Stem / 4th V.')
    return name

# ── COLLECT PER-REGION STATS ─────────────────────────────────────────────────
print("Processing FUMA regions ...")
loci_rows, gene_rows = [], []

for region_dir in sorted(FUMA_DIR.iterdir()):
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    side, base = parse_region(folder)

    def read_tsv(fname):
        p = region_dir / fname
        return pd.read_csv(p, sep='\t', low_memory=False) if p.exists() else None

    loci_df  = read_tsv('GenomicRiskLoci.txt')
    lead_df  = read_tsv('leadSNPs.txt')
    ind_df   = read_tsv('IndSigSNPs.txt')
    genes_df = read_tsv('genes.txt')

    if loci_df is None:
        print(f"  SKIP {folder}: GenomicRiskLoci.txt missing")
        continue

    # ── loci stats ──
    n_loci       = len(loci_df)
    n_lead       = len(lead_df)  if lead_df  is not None else np.nan
    n_ind        = len(ind_df)   if ind_df   is not None else np.nan
    top_p        = loci_df['p'].min() if 'p' in loci_df.columns else np.nan
    median_nsnps = loci_df['nSNPs'].median() if 'nSNPs' in loci_df.columns else np.nan

    loci_rows.append(dict(
        folder=folder, side=side, base=base,
        display=display_name(side, base),
        n_loci=n_loci, n_lead_SNPs=n_lead, n_IndSigSNPs=n_ind,
        top_p=top_p, median_locus_nSNPs=median_nsnps,
    ))
    print(f"  {folder}: {n_loci} loci, {n_lead} lead SNPs, {n_ind} IndSigSNPs")

    # ── gene stats ──
    if genes_df is not None:
        # Positional: posMapSNPs > 0
        n_pos  = (pd.to_numeric(genes_df['posMapSNPs'],  errors='coerce').fillna(0) > 0).sum()
        # eQTL: eqtlMapSNPs > 0
        n_eqtl = (pd.to_numeric(genes_df['eqtlMapSNPs'], errors='coerce').fillna(0) > 0).sum()
        # CI: ciMap != 'No'
        n_ci   = (genes_df['ciMap'].astype(str).str.strip().str.lower() != 'no').sum() \
                 if 'ciMap' in genes_df.columns else np.nan
        n_total = len(genes_df)

        gene_rows.append(dict(
            folder=folder, side=side, base=base,
            display=display_name(side, base),
            n_genes_total=n_total,
            n_genes_positional=int(n_pos),
            n_genes_eqtl=int(n_eqtl),
            n_genes_ci=int(n_ci) if not np.isnan(n_ci) else np.nan,
        ))
    else:
        print(f"    genes.txt missing for {folder}")

# ── BUILD DATAFRAMES ─────────────────────────────────────────────────────────
loci_df_out  = pd.DataFrame(loci_rows)
gene_df_out  = pd.DataFrame(gene_rows)

# Sort: bilateral pairs in BASE_ORDER, then midline
def sort_key(row):
    base = row['base']
    side = row['side']
    if base in BASE_ORDER:
        bi = BASE_ORDER.index(base)
        si = {'Left': 0, 'Right': 1}.get(side, 2)
        return (bi, si)
    elif base in MIDLINE:
        return (len(BASE_ORDER) + MIDLINE.index(base), 0)
    return (99, 0)

loci_df_out['_sort'] = loci_df_out.apply(sort_key, axis=1)
gene_df_out['_sort'] = gene_df_out.apply(sort_key, axis=1)
loci_df_out = loci_df_out.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
gene_df_out = gene_df_out.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

# Table 1: merge loci + gene counts
table1 = pd.merge(
    loci_df_out[['display', 'side', 'base', 'n_loci', 'n_lead_SNPs',
                 'n_IndSigSNPs', 'top_p', 'median_locus_nSNPs']],
    gene_df_out[['display', 'n_genes_total', 'n_genes_positional',
                 'n_genes_eqtl', 'n_genes_ci']],
    on='display', how='left'
)

loci_df_out.to_csv(OUT_DIR / 'loci_summary.csv', index=False)
gene_df_out.to_csv(OUT_DIR / 'gene_summary.csv', index=False)
table1.to_csv(OUT_DIR / 'table1_combined.csv', index=False)
print(f"\nSaved: {OUT_DIR}/loci_summary.csv")
print(f"Saved: {OUT_DIR}/gene_summary.csv")
print(f"Saved: {OUT_DIR}/table1_combined.csv")

# ── PRINT TABLE 1 ─────────────────────────────────────────────────────────────
print("\n=== Table 1 preview ===")
preview = table1[['display', 'n_loci', 'n_lead_SNPs', 'n_IndSigSNPs',
                  'n_genes_total', 'n_genes_positional', 'n_genes_eqtl', 'n_genes_ci']].copy()
preview.columns = ['Region', 'Loci', 'Lead SNPs', 'IndSigSNPs',
                   'Genes (all)', 'Positional', 'eQTL', 'CI']
print(preview.to_string(index=False))

# ── FIGURE: loci counts per region (horizontal barplot) ──────────────────────
# Colours: Left = steelblue, Right = coral, Midline = grey
SIDE_COLOR = {'Left': '#4878CF', 'Right': '#E8735A', 'Midline': '#888888'}

regions = loci_df_out['display'].tolist()
counts  = loci_df_out['n_loci'].tolist()
sides   = loci_df_out['side'].tolist()
colors  = [SIDE_COLOR[s] for s in sides]

fig, ax = plt.subplots(figsize=(MM(120), MM(118)))
y = np.arange(len(regions))
bars = ax.barh(y, counts, color=colors, height=0.65, edgecolor='none')

# Value labels
for bar, val in zip(bars, counts):
    ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va='center', ha='left', fontsize=7)

ax.set_yticks(y)
ax.set_yticklabels(regions)
ax.set_xlabel('Number of genomic risk loci')
ax.set_title('JAGWAS loci per subcortical region')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0, max(counts) * 1.15)
ax.invert_yaxis()

legend_patches = [
    mpatches.Patch(color=SIDE_COLOR['Left'],    label='Left hemisphere'),
    mpatches.Patch(color=SIDE_COLOR['Right'],   label='Right hemisphere'),
    mpatches.Patch(color=SIDE_COLOR['Midline'], label='Midline'),
]
fig.legend(handles=legend_patches, loc='lower center',
           bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.07, 1, 1])
save_mpl(fig, FIG_DIR / 'loci_counts')
plt.close()

print(f"\nFigure saved: {FIG_DIR}/loci_counts.pdf/.png")
print("\nTotal loci across all regions:", int(loci_df_out['n_loci'].sum()))
print("Total unique genes across all regions:", int(gene_df_out['n_genes_total'].sum()))
