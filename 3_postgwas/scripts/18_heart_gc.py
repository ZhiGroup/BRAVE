"""
18_heart_gc.py
Genetic correlation between JAGWAS BRE dimensions and 82 cardiac MRI traits (Bai et al.).

Strategy (mirrors Script 16 Disease GC):
  - For each of 16 BRE regions, select top-5 dims by h²
    (from results/h2/fastgwa_h2_top_dims.csv)
  - Munge 82 cardiac fastGWA sumstats (if not already done)
  - Run LDSC rg between those dims and each cardiac trait
  - FDR: BH across all dim-level pairs (16 × 5 × 82 = 6,560)
  - Heatmap 1: global/summary traits (28 representative traits)
  - Heatmap 2: full 82-trait heatmap

Cardiac GWAS: Bai et al. UK Biobank, N~28-30K
  Layout: heart sumstats root /
          ukbiobank_heart_pheno{ID}_may2022/
          ukb_phase1to3_heart_may_2022_pheno{ID}.fastGWA
  Format: CHR SNP POS A1 A2 N AF1 BETA SE P  (N varies per row)

Outputs:
  results/gc_heart/heart_gc.csv          — all parsed rg pairs (dim level)
  results/gc_heart/heart_gc_region.csv   — aggregated per region × trait
  figures/gc_heart/heart_gc_heatmap_global.pdf/.png   — 28 global traits
  figures/gc_heart/heart_gc_heatmap_all.pdf/.png      — all 82 traits

Supports Claim 3: region-specific biological validity via cardiac genetic architecture.
"""

import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL
from statsmodels.stats.multitest import multipletests

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
MUNGED_DIR = Path(cfg.postgwas.bre_munged_dir)   # BRE munged files (existing)
BASE_DIR   = MUNGED_DIR.parent
BRE_PREFIX = "discovery"

GC_HEART_DIR     = BASE_DIR / "gc_heart"
MUNGED_HEART_DIR = BASE_DIR / "munged_heart"
GC_HEART_DIR.mkdir(parents=True, exist_ok=True)
MUNGED_HEART_DIR.mkdir(parents=True, exist_ok=True)

HEART_DIR = Path(cfg.postgwas.heart_sumstats_dir)
BAI82_CSV = Path("<EXTERNAL: Bai et al. cardiac-trait reference csv>")

LDSC_PY      = "<EXTERNAL: python interpreter for the LDSC conda environment>"
LDSC_SCRIPT  = str(Path(cfg.tools.ldsc_dir) / "ldsc.py")
MUNGE_SCRIPT = str(Path(cfg.tools.ldsc_dir) / "munge_sumstats.py")
LD_CHR       = "<EXTERNAL: 1000 Genomes LD score reference panel directory>"

RES_DIR = Path(__file__).parents[1] / "results" / "gc_heart"
FIG_DIR = Path(__file__).parents[1] / "figures" / "gc_heart"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

H2_CSV  = Path(__file__).parents[1] / "results" / "h2" / "fastgwa_h2_top_dims.csv"

TOP_K = 5   # top dims per region by h²

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

# 28 global/summary traits (no regional AHA segments):
# IDs 1-10 (whole-chamber), 27 (WT_global), 28-35 (LA/RA), 36-41 (AAo/DAo),
# 48 (Ell_global), 65 (Ecc_global), 82 (Err_global)
GLOBAL_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              27, 28, 29, 30, 31, 32, 33, 34, 35,
              36, 37, 38, 39, 40, 41,
              48, 65, 82}

# ── LOAD INPUTS ───────────────────────────────────────────────────────────────
print("Loading h² top dims and Bai82 cardiac trait metadata ...")

h2 = pd.read_csv(H2_CSV)
bai82 = pd.read_csv(BAI82_CSV)

# Build trait info dict keyed by ID
bai82_meta = {}
for _, row in bai82.iterrows():
    bai82_meta[int(row['ID'])] = {
        'name':      row['Name'],
        'full_name': row['Full_Name'],
        'category':  row['Category'],
        'short':     row['Short'],
        'h2':        float(row['Repo']),
    }

# top-5 dims per region
bre_dims = {}
for region, grp in h2.groupby('region'):
    bre_dims[region] = grp.head(TOP_K)['dim'].tolist()

print(f"  BRE regions with top-{TOP_K} dims: {len(bre_dims)}")
print(f"  Cardiac traits: {len(bai82_meta)}")

# ── STEP 1: CHECK HEART SUMSTAT FILES ─────────────────────────────────────────
print("\nChecking cardiac sumstat files ...")
trait_paths = {}   # id -> Path
missing = []
for pheno_id in bai82_meta.keys():
    p = (HEART_DIR / f"ukbiobank_heart_pheno{pheno_id}_may2022"
         / f"ukb_phase1to3_heart_may_2022_pheno{pheno_id}.fastGWA")
    if p.exists():
        trait_paths[pheno_id] = p
    else:
        missing.append(pheno_id)

print(f"  Found: {len(trait_paths)}/82 sumstat files")
if missing:
    print(f"  Missing IDs: {missing}")

# ── STEP 2: MUNGE HEART SUMSTATS ──────────────────────────────────────────────
print(f"\nMunging cardiac sumstats to {MUNGED_HEART_DIR} ...")

CMD_MUNGE = (
    f"{LDSC_PY} {MUNGE_SCRIPT} "
    "--sumstats {inp} "
    "--out {out} "
    "--N-col N "
    "--a1 A1 --a2 A2 --p P --frq AF1"
)


def munge_heart(args):
    """Munge one cardiac sumstat. Skip if .sumstats.gz already exists."""
    pheno_id, inp_path = args
    out_stem = MUNGED_HEART_DIR / f"heart_pheno{pheno_id}"
    out_gz   = Path(str(out_stem) + ".sumstats.gz")
    if out_gz.exists():
        return True
    cmd = CMD_MUNGE.format(inp=inp_path, out=out_stem)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


munge_jobs = [(pid, p) for pid, p in trait_paths.items()
              if not (MUNGED_HEART_DIR / f"heart_pheno{pid}.sumstats.gz").exists()]
print(f"  Already munged: {len(trait_paths) - len(munge_jobs)}")
print(f"  To munge:       {len(munge_jobs)}")

if munge_jobs:
    print(f"  Munging {len(munge_jobs)} files (Pool 20) ...")
    with Pool(20) as pool:
        munge_res = pool.map(munge_heart, munge_jobs)
    print(f"  Munged: {sum(munge_res)}/{len(munge_jobs)} succeeded")

# Collect successfully munged files
munged_paths = {}
for pid in trait_paths.keys():
    gz = MUNGED_HEART_DIR / f"heart_pheno{pid}.sumstats.gz"
    if gz.exists():
        munged_paths[pid] = gz

print(f"  Available munged: {len(munged_paths)}/82")

# ── STEP 3: RUN LDSC rg ────────────────────────────────────────────────────────
print(f"\nRunning LDSC rg: top-{TOP_K} BRE dims × {len(munged_paths)} cardiac traits ...")

CMD_RG = (
    f"{LDSC_PY} {LDSC_SCRIPT} "
    "--rg {p1},{p2} "
    f"--ref-ld-chr {LD_CHR} --w-ld-chr {LD_CHR} "
    "--out {out}"
)


def run_heart_rg(args):
    """Run LDSC rg for one BRE dim × one cardiac trait. Skip if log exists."""
    bre_region, dim, pheno_id, heart_gz = args
    out = str(GC_HEART_DIR / f"{bre_region}_QT{dim}_heart{pheno_id}")
    if Path(out + ".log").exists():
        return True
    p1 = str(MUNGED_DIR / f"{BRE_PREFIX}_{bre_region}_QT{dim}.fastGWA.fastGWA.sumstats.gz")
    p2 = str(heart_gz)
    if not Path(p1).exists():
        print(f"  MISSING BRE munged: {p1}")
        return False
    if not Path(p2).exists():
        print(f"  MISSING heart munged: {p2}")
        return False
    cmd = CMD_RG.format(p1=p1, p2=p2, out=out)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


jobs = []
for region, dims in bre_dims.items():
    for dim in dims:
        for pid, gz in munged_paths.items():
            log_path = GC_HEART_DIR / f"{region}_QT{dim}_heart{pid}.log"
            if not log_path.exists():
                jobs.append((region, dim, pid, gz))

total_expected = sum(len(d) for d in bre_dims.values()) * len(munged_paths)
print(f"  Total pairs expected: {total_expected}")
print(f"  Already done:         {total_expected - len(jobs)}")
print(f"  To run:               {len(jobs)}")

if jobs:
    print(f"  Running {len(jobs)} LDSC rg jobs (Pool 40) ...")
    with Pool(40) as pool:
        rg_results = pool.map(run_heart_rg, jobs)
    print(f"  Done: {sum(rg_results)}/{len(jobs)} succeeded")

# ── STEP 4: PARSE LOGS ────────────────────────────────────────────────────────
print("\nParsing heart GC logs ...")


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
        for pid in munged_paths.keys():
            log_path = GC_HEART_DIR / f"{region}_QT{dim}_heart{pid}.log"
            if not log_path.exists():
                continue
            rec = parse_ldsc_rg_log(log_path)
            if rec is None:
                continue
            meta = bai82_meta.get(pid, {})
            try:
                rows.append(dict(
                    bre_region   = region,
                    bre_display  = DISP.get(region, region),
                    dim          = int(dim),
                    pheno_id     = pid,
                    trait_name   = meta.get('name', f'pheno{pid}'),
                    trait_full   = meta.get('full_name', ''),
                    category     = meta.get('category', ''),
                    short        = meta.get('short', ''),
                    rg           = float(rec['rg']),
                    se           = float(rec['se']),
                    z            = float(rec['z']),
                    p            = float(rec['p']),
                    is_global    = pid in GLOBAL_IDS,
                ))
            except (ValueError, KeyError):
                continue

heart_gc = pd.DataFrame(rows)
print(f"  Parsed {len(heart_gc)} valid pairs")

if len(heart_gc) == 0:
    print("ERROR: no pairs parsed. Check log paths.")
    sys.exit(1)

# ── STEP 5: FDR CORRECTION ─────────────────────────────────────────────────────
print("Applying BH FDR correction ...")
heart_gc = heart_gc.dropna(subset=['p'])
_, fdr_q, _, _ = multipletests(heart_gc['p'].values, method='fdr_bh')
heart_gc['fdr_q']   = fdr_q
heart_gc['fdr_sig'] = fdr_q < 0.05

n_sig = int(heart_gc['fdr_sig'].sum())
print(f"  FDR-significant dim-level pairs (q<0.05): {n_sig} / {len(heart_gc)}")

heart_gc.to_csv(RES_DIR / 'heart_gc.csv', index=False)
print(f"  Saved: {RES_DIR}/heart_gc.csv")

# ── STEP 6: AGGREGATE TO REGION × TRAIT ────────────────────────────────────────
print("Aggregating to region × trait (max |rg| across top dims) ...")

agg = heart_gc.groupby(['bre_region', 'bre_display', 'pheno_id',
                         'trait_name', 'category', 'short', 'is_global']).apply(
    lambda g: pd.Series({
        'max_abs_rg': g['rg'].abs().max(),
        'best_rg':    g.loc[g['rg'].abs().idxmax(), 'rg'],
        'best_p':     g.loc[g['rg'].abs().idxmax(), 'p'],
        'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
        'n_fdr_sig':  int(g['fdr_sig'].sum()),
        'best_dim':   int(g.loc[g['rg'].abs().idxmax(), 'dim']),
    })
).reset_index()

agg.to_csv(RES_DIR / 'heart_gc_region.csv', index=False)
print(f"  Saved: {RES_DIR}/heart_gc_region.csv")

n_sig_region = int((agg['best_fdr_q'] < 0.05).sum())
print(f"  FDR-sig region × trait pairs: {n_sig_region} / {len(agg)}")

print("\nTop region × cardiac trait pairs (by max |rg|):")
top_agg = agg.nlargest(20, 'max_abs_rg')[
    ['bre_display', 'trait_name', 'category', 'best_rg', 'best_p', 'best_fdr_q', 'best_dim']
].reset_index(drop=True)
print(top_agg.to_string(index=False))

# ── STEP 7: HEATMAPS ───────────────────────────────────────────────────────────
from matplotlib.colors import TwoSlopeNorm


def make_heatmap(agg_sub, title, out_stem, region_order, trait_ids, trait_labels,
                 figw_mm, figh_mm):
    """Build and save an rg heatmap for a subset of traits."""
    disp_order = [DISP[r] for r in region_order if r in agg_sub['bre_region'].values]
    # Filter region_order to those present
    region_order_filt = [r for r in region_order if r in agg_sub['bre_region'].values]
    disp_order = [DISP[r] for r in region_order_filt]

    mat_rg = np.full((len(region_order_filt), len(trait_ids)), np.nan)
    mat_q  = np.full((len(region_order_filt), len(trait_ids)), np.nan)

    for _, row in agg_sub.iterrows():
        if row['bre_region'] in region_order_filt and row['pheno_id'] in trait_ids:
            i = region_order_filt.index(row['bre_region'])
            j = trait_ids.index(row['pheno_id'])
            mat_rg[i, j] = row['best_rg']
            mat_q[i, j]  = row['best_fdr_q']

    valid = mat_rg[~np.isnan(mat_rg)]
    if len(valid) == 0:
        print(f"  No data for heatmap: {out_stem}")
        return
    vmax = max(np.percentile(np.abs(valid), 95), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(MM(figw_mm), MM(figh_mm)))
    im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')

    for i in range(mat_rg.shape[0]):
        for j in range(mat_rg.shape[1]):
            if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=FS_TICK, color='black', fontweight='bold')

    ax.set_xticks(range(len(trait_labels)))
    ax.set_xticklabels(trait_labels, rotation=45, ha='right', fontsize=FS_TICK - 1)
    ax.set_yticks(range(len(disp_order)))
    ax.set_yticklabels(disp_order, fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('Cardiac trait (Bai et al.)', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / out_stem)
    plt.close()
    print(f"  Saved: figures/gc_heart/{out_stem}.pdf/.png")


# --- Heatmap 1: Global/summary traits (28 traits) ---
print("\nGenerating global-traits heatmap (28 traits) ...")
agg_global = agg[agg['is_global']].copy()
global_ids_present = sorted([pid for pid in GLOBAL_IDS if pid in agg_global['pheno_id'].values])
global_labels = [bai82_meta[pid]['name'] for pid in global_ids_present]

make_heatmap(
    agg_sub=agg_global,
    title=f'Genetic correlation: BRE dims vs cardiac traits (global, top-{TOP_K} h² dims)\n* FDR q<0.05',
    out_stem='heart_gc_heatmap_global',
    region_order=REGIONS_16,
    trait_ids=global_ids_present,
    trait_labels=global_labels,
    figw_mm=160, figh_mm=110,
)

# --- Heatmap 2: All 82 traits, grouped by category ---
print("Generating full 82-trait heatmap ...")
# Order traits by category then by ID
cat_order = ['left ventricle', 'right ventricle', 'left atrium', 'right atrium',
             'ascending aorta', 'descending aorta']

def cat_sort_key(pid):
    meta = bai82_meta.get(pid, {})
    cat = meta.get('category', 'z')
    idx = cat_order.index(cat) if cat in cat_order else len(cat_order)
    return (idx, pid)

all_ids_present = sorted(agg['pheno_id'].unique(), key=cat_sort_key)
all_labels = [bai82_meta[pid]['name'] for pid in all_ids_present]

make_heatmap(
    agg_sub=agg,
    title=f'Genetic correlation: BRE dims vs all 82 cardiac traits (top-{TOP_K} h² dims)\n* FDR q<0.05',
    out_stem='heart_gc_heatmap_all',
    region_order=REGIONS_16,
    trait_ids=all_ids_present,
    trait_labels=all_labels,
    figw_mm=240, figh_mm=110,
)

# ── STEP 8: TOP-1 SENSITIVITY (fallback) ─────────────────────────────────────
print("\n--- Sensitivity: top-1 h² dim per region ---")
top1_dims = {}
for region, grp in h2.groupby('region'):
    top1_dims[region] = [grp.iloc[0]['dim']]

heart_gc_top1 = heart_gc[
    heart_gc.apply(lambda r: r['dim'] in top1_dims.get(r['bre_region'], []), axis=1)
].copy()

if len(heart_gc_top1) > 0:
    _, fdr_q1, _, _ = multipletests(heart_gc_top1['p'].values, method='fdr_bh')
    heart_gc_top1 = heart_gc_top1.copy()
    heart_gc_top1['fdr_q_top1'] = fdr_q1
    n_sig_top1 = int((fdr_q1 < 0.05).sum())
    print(f"  Top-1 FDR-sig dim-level pairs: {n_sig_top1} / {len(heart_gc_top1)}")
    heart_gc_top1.to_csv(RES_DIR / 'heart_gc_top1dim.csv', index=False)
    print(f"  Saved: {RES_DIR}/heart_gc_top1dim.csv")
else:
    print("  No top-1 pairs found.")
    n_sig_top1 = 0

# ── SUMMARY ──────────────────────────────────────────────────────────────────
total_pairs = sum(len(d) for d in bre_dims.values()) * len(munged_paths)
print(f"""
╔══════════════════════════════════════════════════════╗
║  SCRIPT 18 — HEART GC SUMMARY                       ║
╠══════════════════════════════════════════════════════╣
║  BRE regions:                {len(bre_dims):4d}                     ║
║  Cardiac traits (munged):    {len(munged_paths):4d}                     ║
║  Top dims per region:        {TOP_K:4d}                     ║
║  Total dim-level pairs:      {len(heart_gc):4d}                     ║
║  FDR-sig dim-level pairs:    {n_sig:4d}                     ║
║  Region × trait pairs:       {len(agg):4d}                     ║
║  FDR-sig region×trait:       {n_sig_region:4d}                     ║
║  Top-1 sensitivity FDR-sig:  {n_sig_top1:4d}                     ║
╚══════════════════════════════════════════════════════╝
""")
