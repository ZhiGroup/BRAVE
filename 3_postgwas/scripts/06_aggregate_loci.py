"""
06_aggregate_loci.py
Aggregate loci across all 16 regions into a non-redundant set (AL- table).

Two loci from different regions are merged if their genomic windows overlap
(chr matches AND intervals [start, end] intersect with any buffer).

Outputs:
  results/cross_region/aggregate_loci.csv     — one row per AL- locus
  results/cross_region/loci_region_matrix.csv — binary matrix: AL- loci × 16 regions
  results/cross_region/al_summary.txt         — key numbers for paper

Supports paper Claim 2:
  JAGWAS unique loci count >> FastGWA 60 loci → demonstrates multivariate gain.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR = Path(cfg.postgwas.fuma_dir)
OUT_DIR  = Path(__file__).parents[1] / "results" / "cross_region"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── REGION ORDER ──────────────────────────────────────────────────────────────
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

# ── LOAD ALL GENOMIC RISK LOCI ────────────────────────────────────────────────
print("Loading GenomicRiskLoci.txt for all regions ...")
all_loci = []

region_dirs = sorted(FUMA_DIR.iterdir(),
                     key=lambda d: sort_key(*parse_region(d.name)) if d.is_dir() else (99,0))

region_order = []   # display names in sorted order

for region_dir in region_dirs:
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    loci_file = region_dir / 'GenomicRiskLoci.txt'
    if not loci_file.exists():
        continue

    side, base = parse_region(folder)
    disp = display_name(side, base)
    region_order.append(disp)

    df = pd.read_csv(loci_file, sep='\t', low_memory=False)
    df['folder']  = folder
    df['display'] = disp
    all_loci.append(df)
    print(f"  {folder}: {len(df)} loci")

loci_df = pd.concat(all_loci, ignore_index=True)
print(f"\nTotal loci (raw sum): {len(loci_df)}")

# Ensure numeric types
for col in ['chr', 'start', 'end']:
    loci_df[col] = pd.to_numeric(loci_df[col], errors='coerce')
loci_df = loci_df.dropna(subset=['chr', 'start', 'end'])
loci_df[['chr', 'start', 'end']] = loci_df[['chr', 'start', 'end']].astype(int)

# ── MERGE OVERLAPPING LOCI (greedy interval merge) ────────────────────────────
# Sort by chr, start; then merge intervals that overlap across any region
print("\nMerging overlapping loci across regions ...")

loci_sorted = loci_df.sort_values(['chr', 'start']).reset_index(drop=True)

# Build merged intervals
merged = []          # list of dicts: chr, start, end, constituent rows indices
current = None

for idx, row in loci_sorted.iterrows():
    if current is None:
        current = {'chr': row['chr'], 'start': row['start'], 'end': row['end'],
                   'rows': [idx]}
    elif row['chr'] == current['chr'] and row['start'] <= current['end']:
        # Overlap — extend current interval
        current['end']  = max(current['end'], row['end'])
        current['rows'].append(idx)
    else:
        merged.append(current)
        current = {'chr': row['chr'], 'start': row['start'], 'end': row['end'],
                   'rows': [idx]}

if current is not None:
    merged.append(current)

print(f"Aggregate loci (AL-) after merging: {len(merged)}")

# ── BUILD AL- TABLE ───────────────────────────────────────────────────────────
al_rows = []
locus_region_matrix = {disp: [] for disp in region_order}

for i, m in enumerate(merged):
    al_id = f"AL-{i+1:04d}"
    rows_df = loci_sorted.loc[m['rows']]

    regions_hit   = rows_df['display'].unique().tolist()
    n_regions     = len(regions_hit)
    lead_snps     = rows_df['LeadSNPs'].dropna().str.split(';').explode().unique().tolist()
    best_p        = rows_df['p'].min() if 'p' in rows_df.columns else np.nan
    best_snp_row  = rows_df.loc[rows_df['p'].idxmin()] if 'p' in rows_df.columns else rows_df.iloc[0]
    best_snp      = best_snp_row.get('rsID', best_snp_row.get('LeadSNPs', 'NA'))
    n_raw_loci    = len(rows_df)

    al_rows.append(dict(
        al_id=al_id, chr=m['chr'], start=m['start'], end=m['end'],
        width_kb=round((m['end'] - m['start']) / 1000, 1),
        n_regions=n_regions, regions=';'.join(sorted(regions_hit)),
        n_raw_loci=n_raw_loci,
        best_p=best_p, best_lead_snp=str(best_snp),
        n_lead_snps=len(lead_snps),
    ))

    for disp in region_order:
        locus_region_matrix[disp].append(1 if disp in regions_hit else 0)

al_df = pd.DataFrame(al_rows)

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
al_df.to_csv(OUT_DIR / 'aggregate_loci.csv', index=False)
print(f"Saved: {OUT_DIR}/aggregate_loci.csv")

# Binary region matrix
matrix_df = pd.DataFrame(locus_region_matrix,
                          index=[r['al_id'] for r in al_rows])
matrix_df.index.name = 'al_id'
matrix_df.to_csv(OUT_DIR / 'loci_region_matrix.csv')
print(f"Saved: {OUT_DIR}/loci_region_matrix.csv")

# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────
n_total_al  = len(al_df)
n_shared    = (al_df['n_regions'] > 1).sum()
n_unique    = (al_df['n_regions'] == 1).sum()
n_all16     = (al_df['n_regions'] == 16).sum()
n_fastgwa   = 60   # FastGWA minP baseline

summary = f"""=== Aggregate Loci (AL-) Summary ===
Raw loci (sum across 16 regions):     {len(loci_df):>6}
Unique aggregate loci (AL-):          {n_total_al:>6}
  Region-specific (1 region only):    {n_unique:>6}  ({100*n_unique/n_total_al:.1f}%)
  Shared (≥2 regions):                {n_shared:>6}  ({100*n_shared/n_total_al:.1f}%)
  Shared across all 16 regions:       {n_all16:>6}

FastGWA baseline (univariate minP):   {n_fastgwa:>6}
JAGWAS improvement:                   {n_total_al:>6}  (×{n_total_al/n_fastgwa:.1f} over FastGWA)

Locus size distribution (kb):
  Median: {al_df['width_kb'].median():.0f} kb
  Max:    {al_df['width_kb'].max():.0f} kb

Regions with most unique loci:
{al_df[al_df['n_regions']==1]['regions'].value_counts().head(8).to_string()}
"""

print(summary)
with open(OUT_DIR / 'al_summary.txt', 'w') as f:
    f.write(summary)
print(f"Saved: {OUT_DIR}/al_summary.txt")

# ── SHARED-LOCUS BREAKDOWN ────────────────────────────────────────────────────
print("\n=== Loci sharing by number of regions ===")
share_counts = al_df['n_regions'].value_counts().sort_index()
for n_reg, count in share_counts.items():
    print(f"  {n_reg:2d} region(s): {count:4d} loci  ({100*count/n_total_al:.1f}%)")
