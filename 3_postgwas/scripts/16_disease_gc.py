"""
16_disease_gc.py
Genetic correlation between JAGWAS BRE dimensions and 10 brain disease GWAS.

Strategy (mirrors Script 15 Shape GC):
  - For each of the 16 BRE regions, select the top-5 dims by h²
    (from results/h2/fastgwa_h2_top_dims.csv)
  - Run LDSC rg between those dims and each of 10 pre-munged disease sumstats
  - Disease sumstats: already in .sumstats.gz format (no munging needed)
  - FDR: BH across all individual dim-level pairs (16 × 5 × 10 = 800)
  - Heatmap: 16 BRE regions × 10 diseases, max |rg| per pair, * = FDR q<0.05

Disease GWAS (all pre-munged .sumstats.gz):
  ADHD, ALS, ASD, ALZ, MDD, Neuroticism, BIP, SCZ, SD, IS

Outputs:
  results/gc/disease_gc.csv          — all parsed rg pairs (dim level)
  results/gc/disease_gc_region.csv   — aggregated per region × disease
  figures/gc/disease_gc_heatmap.pdf/.png

Supports Claim 3: biologically valid — BRE loci overlap with psychiatric /
neurological disease genetic architecture.
"""

import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL
from statsmodels.stats.multitest import multipletests

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
MUNGED_DIR = Path(cfg.postgwas.bre_munged_dir)
BASE_DIR   = MUNGED_DIR.parent
GC_DIR     = BASE_DIR / "gc_engima"      # re-use same dir (logs follow same naming)
BRE_PREFIX = "discovery"

LDSC_PY      = "<EXTERNAL: python interpreter for the LDSC conda environment>"
LDSC_SCRIPT  = str(Path(cfg.tools.ldsc_dir) / "ldsc.py")
LD_CHR       = "<EXTERNAL: 1000 Genomes LD score reference panel directory>"

RES_DIR = Path(__file__).parents[1] / "results" / "gc"
FIG_DIR = Path(__file__).parents[1] / "figures" / "gc"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

H2_CSV      = Path(__file__).parents[1] / "results" / "h2" / "fastgwa_h2_top_dims.csv"
DISEASE_CSV = Path("<EXTERNAL: disease sumstats manifest csv>")

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

# Disease display names and column order for heatmap
DISEASE_ORDER = ['SCZ', 'BIP', 'MDD', 'ADHD', 'ASD',
                 'Neuroticism', 'ALZ', 'IS', 'ALS', 'SD']
DISEASE_DISP  = {
    'SCZ':         'Schizophrenia',
    'BIP':         'Bipolar',
    'MDD':         'Depression',
    'ADHD':        'ADHD',
    'ASD':         'Autism',
    'Neuroticism': 'Neuroticism',
    'ALZ':         'Alzheimer\'s',
    'IS':          'Isch. Stroke',
    'ALS':         'ALS',
    'SD':          'Sleep Disorder',
}

# ── LOAD INPUTS ───────────────────────────────────────────────────────────────
print("Loading h² top dims and disease paths ...")

h2 = pd.read_csv(H2_CSV)
disease_df = pd.read_csv(DISEASE_CSV)
disease_paths = dict(zip(disease_df['Trait'], disease_df['sumstat_path']))
DISEASES = list(disease_paths.keys())
print(f"  Diseases: {DISEASES}")

# top-5 dims per region
bre_dims = {}
for region, grp in h2.groupby('region'):
    bre_dims[region] = grp.head(TOP_K)['dim'].tolist()

print(f"  BRE regions with top-{TOP_K} dims: {len(bre_dims)}")
for region, dims in bre_dims.items():
    print(f"    {DISP.get(region, region):<16} dims: {dims}")

# ── STEP 1: RUN LDSC rg FOR MISSING PAIRS ─────────────────────────────────────
print(f"\nChecking / running LDSC rg for top-{TOP_K} dims × {len(DISEASES)} diseases ...")

CMD_RG = (
    f"{LDSC_PY} {LDSC_SCRIPT} "
    "--rg {p1},{p2} "
    f"--ref-ld-chr {LD_CHR} --w-ld-chr {LD_CHR} "
    "--out {out}"
)

LOG_PAT = re.compile(r'^(.+)_QT(\d+)_(.+)\.log$')


def run_disease_rg(args):
    """Run LDSC rg for one BRE dim × one disease. Skip if log already exists."""
    bre_region, dim, disease, disease_sumstats = args
    out = str(GC_DIR / f"{bre_region}_QT{dim}_{disease}")
    if Path(out + ".log").exists():
        return True
    p1 = str(MUNGED_DIR / f"{BRE_PREFIX}_{bre_region}_QT{dim}.fastGWA.fastGWA.sumstats.gz")
    p2 = disease_sumstats
    if not Path(p1).exists():
        print(f"  MISSING BRE munged: {p1}")
        return False
    if not Path(p2).exists():
        print(f"  MISSING disease sumstats: {p2}")
        return False
    cmd = CMD_RG.format(p1=p1, p2=p2, out=out)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


# Build job list
jobs = []
for region, dims in bre_dims.items():
    for dim in dims:
        for disease in DISEASES:
            log_path = GC_DIR / f"{region}_QT{dim}_{disease}.log"
            if not log_path.exists():
                jobs.append((region, dim, disease, disease_paths[disease]))

print(f"  Total pairs needed: {sum(len(d) for d in bre_dims.values()) * len(DISEASES)}")
print(f"  Already done:       {sum(len(d) for d in bre_dims.values()) * len(DISEASES) - len(jobs)}")
print(f"  To run:             {len(jobs)}")

if jobs:
    print(f"  Running {len(jobs)} LDSC rg jobs (Pool 40) ...")
    with Pool(40) as pool:
        results = pool.map(run_disease_rg, jobs)
    ok = sum(results)
    print(f"  Done: {ok}/{len(jobs)} succeeded")

# ── STEP 2: PARSE ALL RELEVANT LOGS ───────────────────────────────────────────
print("\nParsing disease GC logs ...")


def parse_ldsc_rg_log(log_path):
    """Parse LDSC rg log file; return dict or None."""
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
        for disease in DISEASES:
            log_path = GC_DIR / f"{region}_QT{dim}_{disease}.log"
            if not log_path.exists():
                continue
            rec = parse_ldsc_rg_log(log_path)
            if rec is None:
                continue
            try:
                rows.append(dict(
                    bre_region  = region,
                    bre_display = DISP.get(region, region),
                    dim         = int(dim),
                    disease     = disease,
                    rg          = float(rec['rg']),
                    se          = float(rec['se']),
                    z           = float(rec['z']),
                    p           = float(rec['p']),
                ))
            except (ValueError, KeyError):
                continue

disease_gc = pd.DataFrame(rows)
print(f"  Parsed {len(disease_gc)} valid pairs")

if len(disease_gc) == 0:
    print("ERROR: no pairs parsed. Check log paths.")
    sys.exit(1)

# ── STEP 3: FDR CORRECTION ─────────────────────────────────────────────────────
print("Applying BH FDR correction ...")
disease_gc = disease_gc.dropna(subset=['p'])
_, fdr_q, _, _ = multipletests(disease_gc['p'].values, method='fdr_bh')
disease_gc['fdr_q']   = fdr_q
disease_gc['fdr_sig'] = fdr_q < 0.05

n_sig = disease_gc['fdr_sig'].sum()
print(f"  FDR-significant pairs (q<0.05): {n_sig} / {len(disease_gc)}")

disease_gc.to_csv(RES_DIR / 'disease_gc.csv', index=False)
print(f"  Saved: {RES_DIR}/disease_gc.csv")

# ── STEP 4: AGGREGATE TO REGION × DISEASE ─────────────────────────────────────
print("Aggregating to region × disease (max |rg| across top dims) ...")

agg = disease_gc.groupby(['bre_region', 'bre_display', 'disease']).apply(
    lambda g: pd.Series({
        'max_abs_rg': g['rg'].abs().max(),
        'best_rg':    g.loc[g['rg'].abs().idxmax(), 'rg'],
        'best_p':     g.loc[g['rg'].abs().idxmax(), 'p'],
        'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
        'n_fdr_sig':  g['fdr_sig'].sum(),
        'best_dim':   int(g.loc[g['rg'].abs().idxmax(), 'dim']),
    })
).reset_index()

agg.to_csv(RES_DIR / 'disease_gc_region.csv', index=False)
print(f"  Saved: {RES_DIR}/disease_gc_region.csv")

print("\nTop region × disease pairs (by max |rg|):")
top_agg = agg.nlargest(20, 'max_abs_rg')[
    ['bre_display', 'disease', 'best_rg', 'best_p', 'best_fdr_q', 'best_dim']
].reset_index(drop=True)
print(top_agg.to_string(index=False))

n_sig_pairs = (agg['best_fdr_q'] < 0.05).sum()
print(f"\nFDR-significant region × disease pairs: {n_sig_pairs} / {len(agg)}")

# ── STEP 5: HEATMAP ───────────────────────────────────────────────────────────
print("\nGenerating disease GC heatmap ...")

# Build matrix (16 regions × 10 diseases)
region_order = [r for r in REGIONS_16 if r in agg['bre_region'].values]
disp_order   = [DISP[r] for r in region_order]
disease_cols = [d for d in DISEASE_ORDER if d in DISEASES]
disease_lbls = [DISEASE_DISP[d] for d in disease_cols]

mat_rg = np.full((len(region_order), len(disease_cols)), np.nan)
mat_q  = np.full((len(region_order), len(disease_cols)), np.nan)

for _, row in agg.iterrows():
    if row['bre_region'] in region_order and row['disease'] in disease_cols:
        i = region_order.index(row['bre_region'])
        j = disease_cols.index(row['disease'])
        mat_rg[i, j] = row['best_rg']
        mat_q[i, j]  = row['best_fdr_q']

# Symmetric colour scale centred at 0
vmax = np.nanpercentile(np.abs(mat_rg[~np.isnan(mat_rg)]), 95) if not np.all(np.isnan(mat_rg)) else 0.3
vmax = max(vmax, 0.05)

from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, ax = plt.subplots(figsize=(MM(140), MM(110)))
im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')

# Mark FDR-significant cells
for i in range(mat_rg.shape[0]):
    for j in range(mat_rg.shape[1]):
        if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=FS_TICK, color='black', fontweight='bold')

ax.set_xticks(range(len(disease_cols)))
ax.set_xticklabels(disease_lbls, rotation=40, ha='right', fontsize=FS_TICK)
ax.set_yticks(range(len(region_order)))
ax.set_yticklabels(disp_order, fontsize=FS_TICK)
ax.set_title(
    f'Genetic correlation: BRE dims vs brain diseases\n'
    f'(top-{TOP_K} h² dims per region; max |rg|; * FDR q<0.05)',
    fontsize=FS_LABEL, fontweight='bold'
)
ax.set_xlabel('Brain disease / trait', fontsize=FS_LABEL)
ax.set_ylabel('BRE region', fontsize=FS_LABEL)

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
cbar.ax.tick_params(labelsize=FS_TICK - 1)

plt.tight_layout()
save_mpl(fig, FIG_DIR / 'disease_gc_heatmap')
plt.close()
print("Saved: figures/gc/disease_gc_heatmap.pdf/.png")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════╗
║  SCRIPT 16 — DISEASE GC SUMMARY                     ║
╠══════════════════════════════════════════════════════╣
║  BRE regions:                {len(region_order):4d}                     ║
║  Diseases:                   {len(disease_cols):4d}                     ║
║  Top dims per region:        {TOP_K:4d}                     ║
║  Total dim-level pairs:      {len(disease_gc):4d}                     ║
║  FDR-sig dim-level pairs:    {n_sig:4d}                     ║
║  Region × disease pairs:     {len(agg):4d}                     ║
║  FDR-sig region×disease:     {n_sig_pairs:4d}                     ║
╚══════════════════════════════════════════════════════╝
""")
