"""
08_upset_plots.py
UpSet plots showing region-specificity and overlap of:
  (A) Aggregate loci (AL-)         — 276 non-redundant loci × 16 regions
  (B) Independent significant SNPs — raw IndSigSNPs across regions
  (C) Genes (positional)           — positionally mapped genes
  (D) Genes (eQTL-mapped)          — eQTL-mapped genes
  (E) Genes (CI-mapped)            — chromatin-interaction-mapped genes

Each UpSet: sets = 16 regions, elements = loci/SNPs/genes.
Show top 30 intersections by size.

Outputs:
  figures/cross_region/upset_loci.pdf/.png
  figures/cross_region/upset_snps.pdf/.png
  figures/cross_region/upset_genes_pos.pdf/.png
  figures/cross_region/upset_genes_eqtl.pdf/.png
  figures/cross_region/upset_genes_ci.pdf/.png

Supports paper Claim 2: demonstrates both shared and region-specific signal
  at multiple levels (loci, SNPs, genes).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_memberships
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
RES_DIR  = Path(__file__).parents[1] / "results" / "cross_region"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "cross_region"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_INTERSECTIONS = 30   # show top N intersections

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

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data for UpSet plots ...")

# (A) Aggregate loci — already computed, load binary matrix
matrix_df = pd.read_csv(RES_DIR / 'loci_region_matrix.csv', index_col='al_id')
regions   = matrix_df.columns.tolist()   # display names in sorted order

# (B–E) Per-region SNPs and genes
snp_sets  = {r: set() for r in regions}
pos_sets  = {r: set() for r in regions}
eqtl_sets = {r: set() for r in regions}
ci_sets   = {r: set() for r in regions}

region_dirs = sorted(FUMA_DIR.iterdir(),
                     key=lambda d: sort_key(*parse_region(d.name)) if d.is_dir() else (99,0))

for region_dir in region_dirs:
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    side, base = parse_region(folder)
    disp = display_name(side, base)
    if disp not in regions:
        continue

    # IndSigSNPs
    snp_file = region_dir / 'IndSigSNPs.txt'
    if snp_file.exists():
        df = pd.read_csv(snp_file, sep='\t', usecols=['rsID'], low_memory=False)
        snp_sets[disp] = set(df['rsID'].dropna().astype(str))

    # Genes
    gene_file = region_dir / 'genes.txt'
    if gene_file.exists():
        df = pd.read_csv(gene_file, sep='\t',
                         usecols=['symbol', 'posMapSNPs', 'eqtlMapSNPs', 'ciMap'],
                         low_memory=False)
        df['posMapSNPs']  = pd.to_numeric(df['posMapSNPs'],  errors='coerce').fillna(0)
        df['eqtlMapSNPs'] = pd.to_numeric(df['eqtlMapSNPs'], errors='coerce').fillna(0)
        symbols = df['symbol'].dropna().astype(str)

        pos_sets[disp]  = set(symbols[df['posMapSNPs']  > 0])
        eqtl_sets[disp] = set(symbols[df['eqtlMapSNPs'] > 0])
        if 'ciMap' in df.columns:
            ci_sets[disp] = set(symbols[
                df['ciMap'].astype(str).str.strip().str.lower() != 'no'])

print(f"  Loaded SNP and gene sets for {len(regions)} regions")

# ── UPSET PLOT FUNCTION ───────────────────────────────────────────────────────
def make_upset(membership_data, regions_list, title, out_stem,
               figsize=(MM(180), MM(120)), element_label='count'):
    """
    membership_data: dict {region_name: set_of_elements}
    """
    # Build membership list: for each element, which regions contain it?
    all_elements = set()
    for s in membership_data.values():
        all_elements |= s

    memberships = []
    for elem in all_elements:
        mem = tuple(r for r in regions_list if elem in membership_data[r])
        if mem:
            memberships.append(mem)

    if not memberships:
        print(f"  SKIP {out_stem}: no elements")
        return

    data = from_memberships(memberships)

    # Sort by size descending, keep top N
    data_sorted = data.sort_values(ascending=False)
    if len(data_sorted) > TOP_INTERSECTIONS:
        data_sorted = data_sorted.iloc[:TOP_INTERSECTIONS]

    fig = plt.figure(figsize=figsize)
    upset = UpSet(data_sorted,
                  subset_size='count',
                  show_counts=True,
                  sort_by='cardinality',
                  totals_plot_elements=4,
                  min_subset_size=1)
    upset.style_subsets(present=None, facecolor='#4878CF', edgecolor='none')
    upset.plot(fig)

    # Title
    fig.suptitle(title, fontsize=FS_LABEL, y=1.01, fontweight='bold')

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / out_stem, tight=False)
    plt.close()

    n_unique = sum(1 for m in set(map(frozenset, memberships)) if len(m) == 1)
    n_shared = sum(1 for m in set(map(frozenset, memberships)) if len(m) > 1)
    print(f"  {out_stem}: {len(all_elements)} total, "
          f"{n_unique} region-unique sets, {n_shared} shared sets")

# ── (A) UPSET: AGGREGATE LOCI ─────────────────────────────────────────────────
print("\n(A) UpSet — Aggregate loci ...")
loci_membership = {}
for r in regions:
    loci_membership[r] = set(matrix_df.index[matrix_df[r] == 1].tolist())

make_upset(loci_membership, regions,
           title=f'Region overlap — Aggregate loci (n={len(matrix_df)})',
           out_stem='upset_loci')

# ── (B) UPSET: INDEPENDENT SIGNIFICANT SNPS ──────────────────────────────────
print("\n(B) UpSet — Independent significant SNPs ...")
total_snps = len(set().union(*snp_sets.values()))
make_upset(snp_sets, regions,
           title=f'Region overlap — Independent significant SNPs (n={total_snps})',
           out_stem='upset_snps',
           figsize=(MM(180), MM(130)))

# ── (C) UPSET: POSITIONAL GENES ──────────────────────────────────────────────
print("\n(C) UpSet — Positional genes ...")
total_pos = len(set().union(*pos_sets.values()))
make_upset(pos_sets, regions,
           title=f'Region overlap — Positionally mapped genes (n={total_pos})',
           out_stem='upset_genes_pos',
           figsize=(MM(180), MM(130)))

# ── (D) UPSET: eQTL GENES ────────────────────────────────────────────────────
print("\n(D) UpSet — eQTL genes ...")
total_eqtl = len(set().union(*eqtl_sets.values()))
make_upset(eqtl_sets, regions,
           title=f'Region overlap — eQTL-mapped genes (n={total_eqtl})',
           out_stem='upset_genes_eqtl',
           figsize=(MM(180), MM(130)))

# ── (E) UPSET: CI GENES ───────────────────────────────────────────────────────
print("\n(E) UpSet — CI-mapped genes ...")
total_ci = len(set().union(*ci_sets.values()))
make_upset(ci_sets, regions,
           title=f'Region overlap — CI-mapped genes (n={total_ci})',
           out_stem='upset_genes_ci',
           figsize=(MM(180), MM(130)))

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n=== UpSet summary ===")
print(f"  Aggregate loci:     {len(set().union(*loci_membership.values()))} total")
print(f"  IndSigSNPs:         {total_snps} total")
print(f"  Positional genes:   {total_pos} unique")
print(f"  eQTL genes:         {total_eqtl} unique")
print(f"  CI genes:           {total_ci} unique")
