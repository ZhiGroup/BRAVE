"""
19_oct_gc.py
Genetic correlation between JAGWAS BRE dimensions and 46 retinal OCT traits.

Strategy (mirrors Script 18 Heart GC):
  - For each of 16 BRE regions, select top-5 dims by h²
    (from results/h2/fastgwa_h2_top_dims.csv)
  - Munge 46 OCT fastGWA sumstats (if not already done)
  - Run LDSC rg between those dims and each OCT trait
  - FDR: BH across all dim-level pairs (16 × 5 × 46 = 3,680)
  - Heatmap: 16 regions × 46 traits

OCT GWAS: UK Biobank, N~80K
  Metadata: ID_OCT.xlsx (File_name column matches folder names)
  Layout:   OCT sumstats root / {File_name} /
            eye_oct_80k_march10_2022_pheno{N}.fastGWA
  Format:   CHR SNP POS A1 A2 N AF1 BETA SE P  (N varies per row)

Region-specificity hypothesis:
  - Thalamus: LGN relay → RNFL/GCIPL (visual pathway)
  - Hippocampus: AD-related → GCIPL (AD retinal thinning)
  - Pan-brain: general vascular → all retinal layers

Outputs:
  results/gc_oct/oct_gc.csv              — all parsed rg pairs (dim level)
  results/gc_oct/oct_gc_region.csv       — aggregated per region × trait
  figures/gc_oct/oct_gc_heatmap.pdf/.png — 16 regions × 46 traits

Supports Claim 3: region-specific biological validity via visual pathway genetics.
"""

import re
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL
from statsmodels.stats.multitest import multipletests

apply_mpl_style()

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Script 19: OCT GC analysis")
    p.add_argument("--oct-dir",     default=str(cfg.postgwas.oct_sumstats_dir),
                   help="Root directory containing OCT trait folders")
    p.add_argument("--oct-meta",    default=str(Path(cfg.postgwas.oct_sumstats_dir) / "ID_OCT.xlsx"),
                   help="Path to ID_OCT.xlsx metadata file")
    p.add_argument("--base-dir",
                   default=str(Path(cfg.postgwas.bre_munged_dir).parent),
                   help="Base GWAS directory (contains munged/ and BRE outputs)")
    p.add_argument("--munged-oct-dir", default=None,
                   help="Where to write munged OCT files (default: base-dir/munged_oct)")
    p.add_argument("--gc-oct-dir",     default=None,
                   help="Where to write GC log files (default: base-dir/gc_oct)")
    p.add_argument("--res-dir",
                   default=str(Path(__file__).parents[1] / "results" / "gc_oct"),
                   help="Results output directory")
    p.add_argument("--fig-dir",
                   default=str(Path(__file__).parents[1] / "figures" / "gc_oct"),
                   help="Figures output directory")
    p.add_argument("--h2-csv",
                   default=str(Path(__file__).parents[1] / "results" / "h2" / "fastgwa_h2_top_dims.csv"),
                   help="CSV with top h² dims per region")
    p.add_argument("--top-k",       type=int, default=5,
                   help="Number of top h² dims per region (default: 5)")
    p.add_argument("--pool-munge",  type=int, default=20,
                   help="Pool size for munging (default: 20)")
    p.add_argument("--pool-rg",     type=int, default=40,
                   help="Pool size for LDSC rg (default: 40)")
    p.add_argument("--ldsc-py",
                   default="<EXTERNAL: python interpreter for the LDSC conda environment>")
    p.add_argument("--ldsc-script",
                   default=str(Path(cfg.tools.ldsc_dir) / "ldsc.py"))
    p.add_argument("--munge-script",
                   default=str(Path(cfg.tools.ldsc_dir) / "munge_sumstats.py"))
    p.add_argument("--ld-chr",
                   default="<EXTERNAL: 1000 Genomes LD score reference panel directory>")
    return p.parse_args()


args = parse_args()

BASE_DIR         = Path(args.base_dir)
MUNGED_DIR       = BASE_DIR / "munged"        # existing BRE munged files
BRE_PREFIX       = "discovery"
MUNGED_OCT_DIR   = Path(args.munged_oct_dir) if args.munged_oct_dir else BASE_DIR / "munged_oct"
GC_OCT_DIR       = Path(args.gc_oct_dir)     if args.gc_oct_dir     else BASE_DIR / "gc_oct"
OCT_DIR          = Path(args.oct_dir)
RES_DIR          = Path(args.res_dir)
FIG_DIR          = Path(args.fig_dir)
H2_CSV           = Path(args.h2_csv)
OCT_META         = Path(args.oct_meta)
TOP_K            = args.top_k
LDSC_PY          = args.ldsc_py
LDSC_SCRIPT      = args.ldsc_script
MUNGE_SCRIPT     = args.munge_script
LD_CHR           = args.ld_chr

for d in [MUNGED_OCT_DIR, GC_OCT_DIR, RES_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── REGION DEFINITIONS ────────────────────────────────────────────────────────
REGIONS_16 = [
    'Brain_Stem_or_4th_Ventricle', 'CSF',
    'Left_Accumbens-area',  'Right_Accumbens-area',
    'Left_Amygdala',        'Right_Amygdala',
    'Left_Caudate',         'Right_Caudate',
    'Left_Hippocampus',     'Right_Hippocampus',
    'Left_Pallidum',        'Right_Pallidum',
    'Left_Putamen',         'Right_Putamen',
    'Left_Thalamus_Proper', 'Right_Thalamus-Proper',
]
DISP = {
    'Brain_Stem_or_4th_Ventricle': 'BrainStem',
    'CSF':                          'CSF',
    'Left_Accumbens-area':          'L.Accumbens',
    'Right_Accumbens-area':         'R.Accumbens',
    'Left_Amygdala':                'L.Amygdala',
    'Right_Amygdala':               'R.Amygdala',
    'Left_Caudate':                 'L.Caudate',
    'Right_Caudate':                'R.Caudate',
    'Left_Hippocampus':             'L.Hippocampus',
    'Right_Hippocampus':            'R.Hippocampus',
    'Left_Pallidum':                'L.Pallidum',
    'Right_Pallidum':               'R.Pallidum',
    'Left_Putamen':                 'L.Putamen',
    'Right_Putamen':                'R.Putamen',
    'Left_Thalamus_Proper':         'L.Thalamus',
    'Right_Thalamus-Proper':        'R.Thalamus',
}

# ── LOAD OCT METADATA ─────────────────────────────────────────────────────────
print("Loading OCT metadata and h² top dims ...")

# xlsx has a descriptive first row; actual column headers are in row 1
raw = pd.read_excel(OCT_META, header=None)
# Find header row: the row containing 'File_name'
header_row = None
for i, row in raw.iterrows():
    if 'File_name' in row.values:
        header_row = i
        break
if header_row is None:
    print("ERROR: could not find 'File_name' column in OCT metadata")
    sys.exit(1)

oct_df = pd.read_excel(OCT_META, header=header_row)
oct_df = oct_df.dropna(subset=['File_name']).reset_index(drop=True)

print(f"  OCT traits loaded: {len(oct_df)}")
print(f"  Columns: {oct_df.columns.tolist()}")
print(oct_df[['ID', 'File_name']].head(5).to_string())

# Find the sumstat file inside each folder
def find_oct_sumstat(folder_name):
    """Return Path to the .fastGWA file inside the given folder, or None."""
    folder = OCT_DIR / folder_name
    if not folder.exists():
        return None
    fastgwa_files = list(folder.glob("*.fastGWA"))
    if not fastgwa_files:
        return None
    return fastgwa_files[0]   # only one file per folder

oct_traits = []   # list of (trait_id, description, folder_name, sumstat_path)
missing_folders = []
for _, row in oct_df.iterrows():
    trait_id    = str(row['ID']).strip()
    description = str(row.get('Description', trait_id)).strip()
    folder_name = str(row['File_name']).strip()
    p = find_oct_sumstat(folder_name)
    if p is not None:
        oct_traits.append((trait_id, description, folder_name, p))
    else:
        missing_folders.append(folder_name)

print(f"  Found sumstats: {len(oct_traits)}/{len(oct_df)}")
if missing_folders:
    print(f"  Missing folders: {missing_folders}")

# ── LOAD H² TOP DIMS ──────────────────────────────────────────────────────────
h2 = pd.read_csv(H2_CSV)
bre_dims = {}
for region, grp in h2.groupby('region'):
    bre_dims[region] = grp.head(TOP_K)['dim'].tolist()
print(f"  BRE regions with top-{TOP_K} dims: {len(bre_dims)}")

# ── STEP 1: MUNGE OCT SUMSTATS ────────────────────────────────────────────────
print(f"\nMunging OCT sumstats to {MUNGED_OCT_DIR} ...")

CMD_MUNGE = (
    f"{LDSC_PY} {MUNGE_SCRIPT} "
    "--sumstats {inp} "
    "--out {out} "
    "--N-col N "
    "--a1 A1 --a2 A2 --p P --frq AF1"
)


def munge_oct(args_tuple):
    """Munge one OCT sumstat. Skip if .sumstats.gz already exists."""
    trait_id, inp_path = args_tuple
    out_stem = MUNGED_OCT_DIR / f"oct_{trait_id}"
    out_gz   = Path(str(out_stem) + ".sumstats.gz")
    if out_gz.exists():
        return True
    cmd = CMD_MUNGE.format(inp=inp_path, out=out_stem)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


munge_jobs = [
    (tid, str(p))
    for tid, desc, folder, p in oct_traits
    if not (MUNGED_OCT_DIR / f"oct_{tid}.sumstats.gz").exists()
]
print(f"  Already munged: {len(oct_traits) - len(munge_jobs)}")
print(f"  To munge:       {len(munge_jobs)}")

if munge_jobs:
    print(f"  Munging {len(munge_jobs)} files (Pool {args.pool_munge}) ...")
    with Pool(args.pool_munge) as pool:
        munge_res = pool.map(munge_oct, munge_jobs)
    print(f"  Munged: {sum(munge_res)}/{len(munge_jobs)} succeeded")

# Collect munged paths
munged_oct = {}   # trait_id -> Path
for tid, desc, folder, p in oct_traits:
    gz = MUNGED_OCT_DIR / f"oct_{tid}.sumstats.gz"
    if gz.exists():
        munged_oct[tid] = gz

print(f"  Available munged: {len(munged_oct)}/{len(oct_traits)}")

# ── STEP 2: RUN LDSC rg ────────────────────────────────────────────────────────
print(f"\nRunning LDSC rg: top-{TOP_K} BRE dims × {len(munged_oct)} OCT traits ...")

CMD_RG = (
    f"{LDSC_PY} {LDSC_SCRIPT} "
    "--rg {p1},{p2} "
    f"--ref-ld-chr {LD_CHR} --w-ld-chr {LD_CHR} "
    "--out {out}"
)

# trait_id → description lookup
trait_desc = {tid: desc for tid, desc, _, _ in oct_traits}


def run_oct_rg(args_tuple):
    """Run LDSC rg for one BRE dim × one OCT trait. Skip if log exists."""
    bre_region, dim, trait_id, oct_gz = args_tuple
    out = str(GC_OCT_DIR / f"{bre_region}_QT{dim}_oct_{trait_id}")
    if Path(out + ".log").exists():
        return True
    p1 = str(MUNGED_DIR / f"{BRE_PREFIX}_{bre_region}_QT{dim}.fastGWA.fastGWA.sumstats.gz")
    p2 = str(oct_gz)
    if not Path(p1).exists():
        print(f"  MISSING BRE munged: {p1}")
        return False
    cmd = CMD_RG.format(p1=p1, p2=p2, out=out)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


jobs = []
for region, dims in bre_dims.items():
    for dim in dims:
        for tid, gz in munged_oct.items():
            log_path = GC_OCT_DIR / f"{region}_QT{dim}_oct_{tid}.log"
            if not log_path.exists():
                jobs.append((region, dim, tid, gz))

total_expected = sum(len(d) for d in bre_dims.values()) * len(munged_oct)
print(f"  Total pairs expected: {total_expected}")
print(f"  Already done:         {total_expected - len(jobs)}")
print(f"  To run:               {len(jobs)}")

if jobs:
    print(f"  Running {len(jobs)} LDSC rg jobs (Pool {args.pool_rg}) ...")
    with Pool(args.pool_rg) as pool:
        rg_results = pool.map(run_oct_rg, jobs)
    print(f"  Done: {sum(rg_results)}/{len(jobs)} succeeded")

# ── STEP 3: PARSE LOGS ────────────────────────────────────────────────────────
print("\nParsing OCT GC logs ...")


def parse_ldsc_rg_log(log_path):
    """Parse LDSC rg log; return dict or None."""
    try:
        txt = Path(log_path).read_text()
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        header_idx = None
        for i, l in enumerate(lines):
            if l.startswith('p1') and 'rg' in l:
                header_idx = i
                break
        if header_idx is None or header_idx + 1 >= len(lines):
            return None
        header = lines[header_idx].split()
        values = lines[header_idx + 1].split()
        if len(values) < len(header):
            return None
        return dict(zip(header, values))
    except Exception:
        return None


rows = []
for region, dims in bre_dims.items():
    for dim in dims:
        for tid in munged_oct.keys():
            log_path = GC_OCT_DIR / f"{region}_QT{dim}_oct_{tid}.log"
            if not log_path.exists():
                continue
            rec = parse_ldsc_rg_log(log_path)
            if rec is None:
                continue
            try:
                rows.append(dict(
                    bre_region   = region,
                    bre_display  = DISP.get(region, region),
                    dim          = int(dim),
                    trait_id     = tid,
                    description  = trait_desc.get(tid, tid),
                    rg           = float(rec['rg']),
                    se           = float(rec['se']),
                    z            = float(rec['z']),
                    p            = float(rec['p']),
                ))
            except (ValueError, KeyError):
                continue

oct_gc = pd.DataFrame(rows)
print(f"  Parsed {len(oct_gc)} valid pairs")

if len(oct_gc) == 0:
    print("ERROR: no pairs parsed. Check log paths.")
    sys.exit(1)

# ── STEP 4: FDR CORRECTION ─────────────────────────────────────────────────────
print("Applying BH FDR correction ...")
oct_gc = oct_gc.dropna(subset=['p'])
_, fdr_q, _, _ = multipletests(oct_gc['p'].values, method='fdr_bh')
oct_gc['fdr_q']   = fdr_q
oct_gc['fdr_sig'] = fdr_q < 0.05

n_sig = int(oct_gc['fdr_sig'].sum())
print(f"  FDR-significant dim-level pairs (q<0.05): {n_sig} / {len(oct_gc)}")

oct_gc.to_csv(RES_DIR / 'oct_gc.csv', index=False)
print(f"  Saved: {RES_DIR}/oct_gc.csv")

# ── STEP 5: AGGREGATE TO REGION × TRAIT ────────────────────────────────────────
print("Aggregating to region × trait (max |rg| across top dims) ...")

agg = oct_gc.groupby(['bre_region', 'bre_display', 'trait_id', 'description']).apply(
    lambda g: pd.Series({
        'max_abs_rg': g['rg'].abs().max(),
        'best_rg':    g.loc[g['rg'].abs().idxmax(), 'rg'],
        'best_p':     g.loc[g['rg'].abs().idxmax(), 'p'],
        'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
        'n_fdr_sig':  int(g['fdr_sig'].sum()),
        'best_dim':   int(g.loc[g['rg'].abs().idxmax(), 'dim']),
    })
).reset_index()

agg.to_csv(RES_DIR / 'oct_gc_region.csv', index=False)
print(f"  Saved: {RES_DIR}/oct_gc_region.csv")

n_sig_region = int((agg['best_fdr_q'] < 0.05).sum())
print(f"  FDR-sig region × trait pairs: {n_sig_region} / {len(agg)}")

print("\nTop region × OCT trait pairs (by max |rg|):")
top_agg = agg.nlargest(20, 'max_abs_rg')[
    ['bre_display', 'trait_id', 'best_rg', 'best_p', 'best_fdr_q', 'best_dim']
].reset_index(drop=True)
print(top_agg.to_string(index=False))

# ── STEP 6: HEATMAP ────────────────────────────────────────────────────────────
print("\nGenerating OCT GC heatmap ...")

from matplotlib.colors import TwoSlopeNorm

# Trait order: preserve xlsx order
trait_order = [tid for tid, _, _, _ in oct_traits if tid in agg['trait_id'].values]
trait_labels = [trait_desc.get(tid, tid) for tid in trait_order]

# Shorten labels for display
def shorten(label):
    return label.replace('_thickness', '').replace('_left', '_L').replace('_right', '_R')

trait_labels_short = [shorten(t) for t in trait_order]

region_order_filt = [r for r in REGIONS_16 if r in agg['bre_region'].values]
disp_order = [DISP[r] for r in region_order_filt]

mat_rg = np.full((len(region_order_filt), len(trait_order)), np.nan)
mat_q  = np.full((len(region_order_filt), len(trait_order)), np.nan)

for _, row in agg.iterrows():
    if row['bre_region'] in region_order_filt and row['trait_id'] in trait_order:
        i = region_order_filt.index(row['bre_region'])
        j = trait_order.index(row['trait_id'])
        mat_rg[i, j] = row['best_rg']
        mat_q[i, j]  = row['best_fdr_q']

valid = mat_rg[~np.isnan(mat_rg)]
vmax = max(np.percentile(np.abs(valid), 95), 0.05) if len(valid) > 0 else 0.3
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(MM(180), MM(110)))
im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')

for i in range(mat_rg.shape[0]):
    for j in range(mat_rg.shape[1]):
        if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=FS_TICK, color='black', fontweight='bold')

ax.set_xticks(range(len(trait_order)))
ax.set_xticklabels(trait_labels_short, rotation=45, ha='right', fontsize=FS_TICK - 1)
ax.set_yticks(range(len(disp_order)))
ax.set_yticklabels(disp_order, fontsize=FS_TICK)
ax.set_title(
    f'Genetic correlation: BRE dims vs retinal OCT traits (top-{TOP_K} h² dims)\n* FDR q<0.05',
    fontsize=FS_LABEL, fontweight='bold'
)
ax.set_xlabel('Retinal OCT trait', fontsize=FS_LABEL)
ax.set_ylabel('BRE region', fontsize=FS_LABEL)

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'oct_gc_heatmap')
plt.close()
print("Saved: figures/gc_oct/oct_gc_heatmap.pdf/.png")

# ── STEP 7: TOP-1 SENSITIVITY ─────────────────────────────────────────────────
print("\n--- Sensitivity: top-1 h² dim per region ---")
top1_dims = {}
for region, grp in h2.groupby('region'):
    top1_dims[region] = [grp.iloc[0]['dim']]

oct_gc_top1 = oct_gc[
    oct_gc.apply(lambda r: r['dim'] in top1_dims.get(r['bre_region'], []), axis=1)
].copy()

if len(oct_gc_top1) > 0:
    _, fdr_q1, _, _ = multipletests(oct_gc_top1['p'].values, method='fdr_bh')
    oct_gc_top1 = oct_gc_top1.copy()
    oct_gc_top1['fdr_q_top1'] = fdr_q1
    n_sig_top1 = int((fdr_q1 < 0.05).sum())
    print(f"  Top-1 FDR-sig dim-level pairs: {n_sig_top1} / {len(oct_gc_top1)}")
    oct_gc_top1.to_csv(RES_DIR / 'oct_gc_top1dim.csv', index=False)
else:
    n_sig_top1 = 0

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║  SCRIPT 19 — OCT GC SUMMARY                         ║
╠══════════════════════════════════════════════════════╣
║  BRE regions:                {len(bre_dims):4d}                     ║
║  OCT traits (munged):        {len(munged_oct):4d}                     ║
║  Top dims per region:        {TOP_K:4d}                     ║
║  Total dim-level pairs:      {len(oct_gc):4d}                     ║
║  FDR-sig dim-level pairs:    {n_sig:4d}                     ║
║  Region × trait pairs:       {len(agg):4d}                     ║
║  FDR-sig region×trait:       {n_sig_region:4d}                     ║
║  Top-1 sensitivity FDR-sig:  {n_sig_top1:4d}                     ║
╚══════════════════════════════════════════════════════╝
""")
