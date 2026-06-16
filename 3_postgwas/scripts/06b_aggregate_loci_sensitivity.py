"""
06b_aggregate_loci_sensitivity.py
Sensitivity analysis for locus-merging criterion (Major Issue 7).

Tests three merging strategies and reports region-specificity proportions:
  - Overlap-only   (original Script 06: buffer = 0 bp)
  - ±250 kb buffer (intermediate)
  - ±500 kb buffer (reviewer-requested)

Outputs:
  results/cross_region/loci_sensitivity.csv
  results/cross_region/loci_sensitivity_summary.txt
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    p = argparse.ArgumentParser(description="Script 06b: locus merging sensitivity")
    p.add_argument("--fuma-dir",
                   default=str(cfg.postgwas.fuma_dir))
    p.add_argument("--out-dir",
                   default=str(Path(__file__).parents[1] / "results" / "cross_region"))
    return p.parse_args()


args = parse_args()
FUMA_DIR = Path(args.fuma_dir)
OUT_DIR  = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_ORDER = ['Accumbens-area', 'Amygdala', 'Caudate', 'Hippocampus',
              'Pallidum', 'Putamen', 'Thalamus_Proper']
MIDLINE    = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def parse_region(folder):
    n = folder.strip(); nl = n.lower()
    if nl.startswith('left_'):  return 'Left',    n[5:]
    if nl.startswith('right_'): return 'Right',   n[6:]
    return 'Midline', n

def display_name(side, base):
    short = base.replace('_', ' ').replace('-area', '').replace('Thalamus Proper', 'Thalamus')
    prefix = {'Left': 'L.', 'Right': 'R.', 'Midline': ''}[side]
    name = f"{prefix} {short}".strip()
    return name.replace('Brain Stem or 4th Ventricle', 'Brain Stem / 4th V.')

def sort_key(side, base):
    base = base.replace('Thalamus-Proper', 'Thalamus_Proper')
    if base in BASE_ORDER:
        return (BASE_ORDER.index(base), {'Left': 0, 'Right': 1}.get(side, 2))
    if base in MIDLINE:
        return (len(BASE_ORDER) + MIDLINE.index(base), 0)
    return (99, 0)

# ── LOAD ALL FUMA LOCI ────────────────────────────────────────────────────────
print("Loading GenomicRiskLoci.txt for all regions ...")
all_loci = []
region_dirs = sorted(FUMA_DIR.iterdir(),
                     key=lambda d: sort_key(*parse_region(d.name)) if d.is_dir() else (99, 0))

for region_dir in region_dirs:
    if not region_dir.is_dir(): continue
    loci_file = region_dir / 'GenomicRiskLoci.txt'
    if not loci_file.exists(): continue
    side, base = parse_region(region_dir.name)
    base = base.replace('Thalamus-Proper', 'Thalamus_Proper')
    disp = display_name(side, base)
    df = pd.read_csv(loci_file, sep='\t', low_memory=False)
    df['display'] = disp
    all_loci.append(df)
    print(f"  {region_dir.name}: {len(df)} loci")

loci_df = pd.concat(all_loci, ignore_index=True)
for col in ['chr', 'start', 'end']:
    loci_df[col] = pd.to_numeric(loci_df[col], errors='coerce')
loci_df = loci_df.dropna(subset=['chr', 'start', 'end'])
loci_df[['chr', 'start', 'end']] = loci_df[['chr', 'start', 'end']].astype(int)
print(f"\nTotal raw loci: {len(loci_df)}")

# ── MERGING FUNCTION ──────────────────────────────────────────────────────────
def merge_loci(loci_df, buffer_bp=0):
    """Merge overlapping loci with optional buffer (bp added to each end)."""
    loci_sorted = loci_df.sort_values(['chr', 'start']).reset_index(drop=True)
    merged = []
    current = None
    for _, row in loci_sorted.iterrows():
        if current is None:
            current = {'chr': row['chr'], 'start': row['start'], 'end': row['end'],
                       'displays': [row['display']]}
        elif row['chr'] == current['chr'] and row['start'] <= current['end'] + buffer_bp:
            current['end'] = max(current['end'], row['end'])
            current['displays'].append(row['display'])
        else:
            merged.append(current)
            current = {'chr': row['chr'], 'start': row['start'], 'end': row['end'],
                       'displays': [row['display']]}
    if current is not None:
        merged.append(current)

    rows = []
    for m in merged:
        regions_hit = list(set(m['displays']))
        rows.append({
            'chr': m['chr'], 'start': m['start'], 'end': m['end'],
            'n_regions': len(regions_hit),
        })
    return pd.DataFrame(rows)

# ── RUN THREE BUFFERS ─────────────────────────────────────────────────────────
BUFFERS = [
    (0,       'Overlap-only (original)'),
    (250_000, '±250 kb buffer'),
    (500_000, '±500 kb buffer (reviewer)'),
]

records = []
print("\n=== Merging sensitivity ===")
for buf, label in BUFFERS:
    al = merge_loci(loci_df, buffer_bp=buf)
    n_total    = len(al)
    n_unique   = (al['n_regions'] == 1).sum()
    n_shared   = (al['n_regions'] > 1).sum()
    n_all16    = (al['n_regions'] == 16).sum()
    pct_unique = 100 * n_unique / n_total
    pct_shared = 100 * n_shared / n_total
    print(f"\n{label}  (buffer = {buf:,} bp)")
    print(f"  Aggregate loci:        {n_total}")
    print(f"  Region-specific (1):   {n_unique}  ({pct_unique:.1f}%)")
    print(f"  Shared (≥2):           {n_shared}  ({pct_shared:.1f}%)")
    print(f"  Pan-regional (16):     {n_all16}")
    records.append({
        'buffer_bp': buf,
        'label': label,
        'n_total': n_total,
        'n_unique': n_unique,
        'n_shared': n_shared,
        'n_all16': n_all16,
        'pct_unique': round(pct_unique, 1),
        'pct_shared': round(pct_shared, 1),
    })

sens_df = pd.DataFrame(records)
sens_df.to_csv(OUT_DIR / 'loci_sensitivity.csv', index=False)
print(f"\nSaved: {OUT_DIR}/loci_sensitivity.csv")

# ── SUMMARY TEXT ──────────────────────────────────────────────────────────────
summary_lines = ["=== Locus-merging sensitivity analysis ===\n"]
for _, r in sens_df.iterrows():
    summary_lines.append(
        f"{r['label']}:\n"
        f"  Aggregate loci = {r['n_total']}; "
        f"region-specific = {r['n_unique']} ({r['pct_unique']}%); "
        f"shared = {r['n_shared']} ({r['pct_shared']}%); "
        f"pan-regional = {r['n_all16']}\n"
    )
summary_text = "\n".join(summary_lines)
print(summary_text)
with open(OUT_DIR / 'loci_sensitivity_summary.txt', 'w') as f:
    f.write(summary_text)
print(f"Saved: {OUT_DIR}/loci_sensitivity_summary.txt")
