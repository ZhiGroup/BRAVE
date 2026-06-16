"""
15_ldsc_genetic_correlation.py
LDSC genetic correlation analysis — two parts.

PART 1 (ENIGMA GC — already run):
  Parse existing LDSC rg log files from gc_engima/ directory.
  Log file naming: {BRE_region}_QT{dim}_{trait}.log
  Traits: ENIGMA brain volume regions (Accumbens, Amygdala, Brainstem, Caudate,
          Hippocampus, ICV, Pallidum, Putamen, Thalamus, ventralDC)
  Outputs:
    results/gc/enigma_gc.csv          — all parsed rg pairs
    figures/gc/enigma_gc_heatmap.pdf  — 16 BRE regions × 10 ENIGMA regions (max |rg|)

PART 2 (Shape GC — new):
  Munge shape GWAS per-PC zip files and run LDSC rg vs top-h² BRE dims
  per matched region.
  Shape zip files: per-region subcortical-shape phenotype zip archives.
  BRE dims selected: top-5 by h² per matched region (from fastgwa_h2_top_dims.csv)
  Outputs:
    results/gc/shape_gc.csv           — all parsed rg pairs
    figures/gc/shape_gc_heatmap.pdf   — matched BRE regions × shape PCs (max |rg|)

Both parts support Claim 1: BREs subsume volume (ENIGMA rg) and shape information.
"""

import re
import sys
import os
import zipfile
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from multiprocessing import Pool
from glob import glob

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── PATHS ─────────────────────────────────────────────────────────────────────
MUNGED_DIR  = Path(cfg.postgwas.bre_munged_dir)
BASE_DIR    = MUNGED_DIR.parent
GC_ENGIMA   = BASE_DIR / "gc_engima"

SHAPE_ZIP_DIR   = Path("<EXTERNAL: subcortical-shape phenotype zip directory>")
GC_SHAPE_DIR    = BASE_DIR / "gc_shape"
MUNGED_SHAPE_DIR = BASE_DIR / "munged_shape"
GC_SHAPE_DIR.mkdir(exist_ok=True)
MUNGED_SHAPE_DIR.mkdir(exist_ok=True)

RES_DIR  = Path(__file__).parents[1] / "results" / "gc"
FIG_DIR  = Path(__file__).parents[1] / "figures" / "gc"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

H2_TOP_CSV = Path(__file__).parents[1] / "results" / "h2" / "fastgwa_h2_top_dims.csv"

# LDSC infrastructure
LDSC_PY    = "<EXTERNAL: python interpreter for the LDSC conda environment>"
MUNGE_SCRIPT = str(Path(cfg.tools.ldsc_dir) / "munge_sumstats.py")
LDSC_SCRIPT  = str(Path(cfg.tools.ldsc_dir) / "ldsc.py")
LD_CHR       = "<EXTERNAL: 1000 Genomes LD score reference panel directory>"

BRE_PREFIX  = "discovery"

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

ENIGMA_REGIONS = [
    'Accumbens', 'Amygdala', 'Brainstem', 'Caudate',
    'Hippocampus', 'ICV', 'Pallidum', 'Putamen', 'Thalamus', 'ventralDC',
]
ENIGMA_DISP = {
    'Accumbens': 'Accumbens', 'Amygdala': 'Amygdala',
    'Brainstem': 'Brainstem', 'Caudate': 'Caudate',
    'Hippocampus': 'Hippocampus', 'ICV': 'ICV',
    'Pallidum': 'Pallidum', 'Putamen': 'Putamen',
    'Thalamus': 'Thalamus', 'ventralDC': 'VentralDC',
}

# BRE region → matched shape region (used for diagonal highlighting only)
BRE_TO_SHAPE = {
    'Left_Accumbens-area':  'accu',
    'Right_Accumbens-area': 'accu',
    'Left_Amygdala':        'amyg',
    'Right_Amygdala':       'amyg',
    'Left_Caudate':         'caud',
    'Right_Caudate':        'caud',
    'Left_Hippocampus':     'hipp',
    'Right_Hippocampus':    'hipp',
    'Left_Pallidum':        'pall',
    'Right_Pallidum':       'pall',
    'Left_Putamen':         'puta',
    'Right_Putamen':        'puta',
    'Left_Thalamus_Proper': 'thal',
    'Right_Thalamus-Proper':'thal',
}

# All 7 shape regions — used for full off-diagonal GC computation
SHAPE_REGION_ORDER = ['accu', 'amyg', 'caud', 'hipp', 'pall', 'puta', 'thal']

TOP_K_DIMS = 5   # top BRE dims per region for shape GC


# ══════════════════════════════════════════════════════════════════════════════
# SHARED PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_ldsc_rg_log(log_path):
    """Parse LDSC rg log file; return dict or None on failure."""
    try:
        txt = Path(log_path).read_text()
        # The summary table is the last TSV line in the log
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        # Find the summary header line
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
        rec = dict(zip(header, values))
        return rec
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: ENIGMA GC — parse existing logs
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: Parsing ENIGMA GC log files")
print("=" * 70)

log_files = list(GC_ENGIMA.glob("*.log"))
print(f"  Found {len(log_files)} log files in {GC_ENGIMA}")

rows = []
# Log filename: {BRE_region}_QT{dim}_{trait}.log
LOG_PAT = re.compile(r'^(.+)_QT(\d+)_(.+)\.log$')

for lf in log_files:
    m = LOG_PAT.match(lf.name)
    if not m:
        continue
    bre_region, dim_str, trait = m.group(1), m.group(2), m.group(3)
    if trait not in ENIGMA_REGIONS:
        continue   # skip psychiatric / non-ENIGMA traits
    rec = parse_ldsc_rg_log(lf)
    if rec is None:
        continue
    try:
        rows.append(dict(
            bre_region  = bre_region,
            bre_display = DISP.get(bre_region, bre_region),
            dim         = int(dim_str),
            enigma      = trait,
            rg          = float(rec['rg']),
            se          = float(rec['se']),
            z           = float(rec['z']),
            p           = float(rec['p']),
        ))
    except (ValueError, KeyError):
        continue

enigma_df = pd.DataFrame(rows)
print(f"  Parsed {len(enigma_df)} valid ENIGMA GC pairs")

if len(enigma_df):
    # FDR correction (statsmodels BH)
    from statsmodels.stats.multitest import multipletests
    enigma_df = enigma_df.dropna(subset=['p'])
    _, fdr_q, _, _ = multipletests(enigma_df['p'].values, method='fdr_bh')
    enigma_df['fdr_q'] = fdr_q
    enigma_df['fdr_sig'] = fdr_q < 0.05

    enigma_df.to_csv(RES_DIR / 'enigma_gc.csv', index=False)
    print(f"  Saved: {RES_DIR}/enigma_gc.csv")
    print(f"  FDR-significant pairs (q<0.05): {enigma_df['fdr_sig'].sum()}")

    # ── FIGURE: 16 BRE regions × 10 ENIGMA regions heatmap (max |rg|) ──────
    print("\n  Generating ENIGMA GC heatmap ...")

    # Aggregate: max |rg| per (bre_region, enigma) pair across dims
    agg = enigma_df.groupby(['bre_region', 'bre_display', 'enigma']).apply(
        lambda g: pd.Series({
            'max_abs_rg': g['rg'].abs().max(),
            'best_rg':    g.loc[g['rg'].abs().idxmax(), 'rg'],
            'best_p':     g.loc[g['rg'].abs().idxmax(), 'p'],
            'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
            'n_sig':      g['fdr_sig'].sum(),
        })
    ).reset_index()

    # Build matrix (16 × 10)
    region_order = REGIONS_16
    disp_order   = [DISP[r] for r in region_order]
    enigma_order = ENIGMA_REGIONS
    enigma_disp  = [ENIGMA_DISP[e] for e in enigma_order]

    mat_rg  = np.full((len(region_order), len(enigma_order)), np.nan)
    mat_q   = np.full((len(region_order), len(enigma_order)), np.nan)

    for _, row in agg.iterrows():
        if row['bre_region'] in region_order and row['enigma'] in enigma_order:
            i = region_order.index(row['bre_region'])
            j = enigma_order.index(row['enigma'])
            # Plot max |rg| (matches the caption).
            # Rationale: BRE dimensions are unlabelled embedding axes, so the sign
            # of the rg of any individual dim with an external trait is arbitrary
            # (depends on contrastive-encoder weight signs, not biology). Only the
            # |rg| magnitude carries interpretable meaning. Plotting signed best_rg
            # creates apparent "opposite-sign rg" between L and R hemispheres for
            # the same anatomical pair (e.g., L.Accumbens vs R.Accumbens against
            # ENIGMA Accumbens), which is a sign-mapping artefact rather than a
            # biological lateralization. Updated 2026-05-26 in response to PI comment.
            mat_rg[i, j] = row['max_abs_rg']
            mat_q[i, j]  = row['best_fdr_q']

    vmax = np.nanpercentile(mat_rg, 95)

    fig, ax = plt.subplots(figsize=(MM(130), MM(110)))
    im = ax.imshow(mat_rg, cmap='OrRd', vmin=0, vmax=vmax, aspect='auto')

    # Mark FDR-significant cells with asterisk
    for i in range(mat_rg.shape[0]):
        for j in range(mat_rg.shape[1]):
            if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=FS_TICK - 1, color='black', fontweight='bold')

    ax.set_xticks(range(len(enigma_order)))
    ax.set_xticklabels(enigma_disp, rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(disp_order, fontsize=FS_TICK)
    ax.set_title('Genetic correlation: BRE dims vs ENIGMA volume\n(max |rg| across dims; * FDR q<0.05)',
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('ENIGMA brain region (volume)', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation |rg|', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / 'enigma_gc_heatmap')
    plt.close()
    print("  Saved: figures/gc/enigma_gc_heatmap.pdf/.png")

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print("\n  Top ENIGMA GC pairs (by |rg|):")
    top_agg = agg.nlargest(15, 'max_abs_rg')[
        ['bre_display', 'enigma', 'best_rg', 'best_p', 'best_fdr_q']].reset_index(drop=True)
    print(top_agg.to_string(index=False))
else:
    print("  WARNING: No ENIGMA GC pairs parsed. Check GC_ENGIMA path.")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Shape GC — munge + run + parse
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: Shape GC — munge, run, parse")
print("=" * 70)

# ── 2a. Discover shape zip files per region ───────────────────────────────────
SHAPE_PAT = re.compile(
    r'ukbiobank_shape_pheno(\w+)_pc_jan_2022_pheno(\d+)_march2022\.zip')

shape_files = {}   # shape_region → list of (pheno_num, zip_path)
for z in sorted(SHAPE_ZIP_DIR.glob("ukbiobank_shape_pheno*_march2022.zip")):
    m = SHAPE_PAT.match(z.name)
    if m:
        region, pheno_num = m.group(1), int(m.group(2))
        shape_files.setdefault(region, []).append((pheno_num, z))

# Sort by pheno number (lowest = leading PCs)
for r in shape_files:
    shape_files[r].sort(key=lambda x: x[0])

print("  Shape GWAS regions and PC counts:")
for r, fs in sorted(shape_files.items()):
    print(f"    {r:<8} {len(fs)} PCs  (pheno nums: "
          f"{','.join(str(f[0]) for f in fs[:5])}{',...' if len(fs) > 5 else ''})")

# ── 2b. Select top BRE dims for shape-matched regions ────────────────────────
top_df = pd.read_csv(H2_TOP_CSV)
# Keep only matched regions and top TOP_K_DIMS
shape_matched_regions = list(BRE_TO_SHAPE.keys())
top_shape = top_df[top_df['region'].isin(shape_matched_regions)].copy()

# For each region, take top TOP_K_DIMS dims
bre_dims_for_shape = {}
for region in shape_matched_regions:
    sub = top_shape[top_shape['region'] == region].head(TOP_K_DIMS)
    if len(sub):
        bre_dims_for_shape[region] = sub['dim'].tolist()

print(f"\n  Top-{TOP_K_DIMS} BRE dims per matched region (for shape GC):")
for region, dims in bre_dims_for_shape.items():
    print(f"    {DISP[region]:<15} QT dims: {dims}")

# ── 2c. Munge shape GWAS zip files ────────────────────────────────────────────
print(f"\n  Munging shape GWAS files → {MUNGED_SHAPE_DIR}")

CMD_MUNGE_SHAPE = (
    f"{LDSC_PY} {MUNGE_SCRIPT} "
    "--sumstats {infile} --N-col N --out {outbase} "
    "--a1 A1 --a2 A2 --p P --frq AF1"
)

def munge_shape_zip(args):
    """Extract zip → tmp file → munge → cleanup."""
    region, pheno_num, zip_path = args
    outbase = str(MUNGED_SHAPE_DIR / f"shape_{region}_pheno{pheno_num}")
    out_sumstats = outbase + ".sumstats.gz"
    if Path(out_sumstats).exists():
        return True   # already done
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix=f"shape_{region}_{pheno_num}_")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            zf.extractall(tmp_dir)
            infile = os.path.join(tmp_dir, names[0])
        cmd = CMD_MUNGE_SHAPE.format(infile=infile, outbase=outbase)
        ret = os.system(cmd + " > /dev/null 2>&1")
        return ret == 0
    except Exception as e:
        print(f"    ERROR munging {zip_path.name}: {e}")
        return False
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

# Build list of jobs — munge all 7 shape regions (needed for full matrix)
shape_munge_jobs = []
for sregion, files in shape_files.items():
    if sregion not in SHAPE_REGION_ORDER:
        continue
    for pheno_num, zip_path in files:
        shape_munge_jobs.append((sregion, pheno_num, zip_path))

print(f"  Shape munge jobs: {len(shape_munge_jobs)}")
already_done = sum(
    1 for (r, n, _) in shape_munge_jobs
    if (MUNGED_SHAPE_DIR / f"shape_{r}_pheno{n}.sumstats.gz").exists()
)
print(f"  Already munged: {already_done}")

if already_done < len(shape_munge_jobs):
    todo_jobs = [(r, n, z) for (r, n, z) in shape_munge_jobs
                 if not (MUNGED_SHAPE_DIR / f"shape_{r}_pheno{n}.sumstats.gz").exists()]
    print(f"  Running munge for {len(todo_jobs)} remaining files ...")
    with Pool(30) as pool:
        results = pool.map(munge_shape_zip, todo_jobs)
    ok = sum(results)
    print(f"  Munge done: {ok}/{len(todo_jobs)} succeeded")

# ── 2d. Run LDSC rg (BRE top dims × shape PCs) ────────────────────────────────
print(f"\n  Running LDSC rg for shape GC → {GC_SHAPE_DIR}")

CMD_RG = (
    f"{LDSC_PY} {LDSC_SCRIPT} "
    "--rg {p1},{p2} "
    f"--ref-ld-chr {LD_CHR} --w-ld-chr {LD_CHR} "
    "--out {out}"
)


def run_shape_rg(args):
    """Run LDSC rg for one BRE dim × one shape PC."""
    bre_region, dim, sregion, pheno_num = args
    out = str(GC_SHAPE_DIR / f"{bre_region}_QT{dim}_{sregion}_pheno{pheno_num}")
    if Path(out + ".log").exists():
        return True
    p1 = str(MUNGED_DIR / f"{BRE_PREFIX}_{bre_region}_QT{dim}.fastGWA.fastGWA.sumstats.gz")
    p2 = str(MUNGED_SHAPE_DIR / f"shape_{sregion}_pheno{pheno_num}.sumstats.gz")
    if not Path(p1).exists() or not Path(p2).exists():
        return False
    cmd = CMD_RG.format(p1=p1, p2=p2, out=out)
    ret = os.system(cmd + " > /dev/null 2>&1")
    return ret == 0


# Build rg jobs — full 14 BRE × 7 shape matrix (not just matched diagonal)
rg_jobs = []
for bre_region, dims in bre_dims_for_shape.items():
    for sregion in SHAPE_REGION_ORDER:   # all 7 shape regions
        if sregion not in shape_files:
            continue
        for dim in dims:
            for pheno_num, _ in shape_files[sregion]:
                rg_jobs.append((bre_region, dim, sregion, pheno_num))

print(f"  Shape rg jobs: {len(rg_jobs)}")
already_rg = sum(
    1 for (br, d, sr, n) in rg_jobs
    if (GC_SHAPE_DIR / f"{br}_QT{d}_{sr}_pheno{n}.log").exists()
)
print(f"  Already run: {already_rg}")

if already_rg < len(rg_jobs):
    todo_rg = [(br, d, sr, n) for (br, d, sr, n) in rg_jobs
               if not (GC_SHAPE_DIR / f"{br}_QT{d}_{sr}_pheno{n}.log").exists()]
    print(f"  Running LDSC rg for {len(todo_rg)} remaining pairs ...")
    with Pool(50) as pool:
        results = pool.map(run_shape_rg, todo_rg)
    ok = sum(results)
    print(f"  LDSC rg done: {ok}/{len(todo_rg)} succeeded")

# ── 2e. Parse shape GC results ────────────────────────────────────────────────
print("\n  Parsing shape GC results ...")

shape_rows = []
LOG_SHAPE_PAT = re.compile(r'^(.+)_QT(\d+)_(\w+)_pheno(\d+)$')

for log_path in GC_SHAPE_DIR.glob("*.log"):
    m = LOG_SHAPE_PAT.match(log_path.stem)
    if not m:
        continue
    bre_region, dim_str, sregion, pheno_str = (
        m.group(1), m.group(2), m.group(3), m.group(4))
    rec = parse_ldsc_rg_log(log_path)
    if rec is None:
        continue
    try:
        shape_rows.append(dict(
            bre_region   = bre_region,
            bre_display  = DISP.get(bre_region, bre_region),
            dim          = int(dim_str),
            shape_region = sregion,
            pheno        = int(pheno_str),
            rg           = float(rec['rg']),
            se           = float(rec['se']),
            z            = float(rec['z']),
            p            = float(rec['p']),
        ))
    except (ValueError, KeyError):
        continue

shape_gc_df = pd.DataFrame(shape_rows)
print(f"  Parsed {len(shape_gc_df)} valid shape GC pairs")

if len(shape_gc_df):
    from statsmodels.stats.multitest import multipletests
    shape_gc_df = shape_gc_df.dropna(subset=['p'])
    _, fdr_q, _, _ = multipletests(shape_gc_df['p'].values, method='fdr_bh')
    shape_gc_df['fdr_q'] = fdr_q
    shape_gc_df['fdr_sig'] = fdr_q < 0.05

    shape_gc_df.to_csv(RES_DIR / 'shape_gc.csv', index=False)
    print(f"  Saved: {RES_DIR}/shape_gc.csv")
    print(f"  FDR-significant pairs (q<0.05): {shape_gc_df['fdr_sig'].sum()}")

    # ── FIGURE: BRE region × shape region heatmap (max |rg| across dims+PCs) ─
    print("\n  Generating shape GC heatmap ...")

    shape_region_order = SHAPE_REGION_ORDER
    bre_for_plot = [r for r in REGIONS_16 if r in BRE_TO_SHAPE]
    bre_disp_plot = [DISP[r] for r in bre_for_plot]

    agg_shape = shape_gc_df.groupby(['bre_region', 'shape_region']).apply(
        lambda g: pd.Series({
            'max_abs_rg': g['rg'].abs().max(),
            'best_rg':    g.loc[g['rg'].abs().idxmax(), 'rg'],
            'best_p':     g.loc[g['rg'].abs().idxmax(), 'p'],
            'best_fdr_q': g.loc[g['rg'].abs().idxmax(), 'fdr_q'],
            'n_fdr_sig':  g['fdr_sig'].sum(),
        })
    ).reset_index()

    mat_s  = np.full((len(bre_for_plot), len(shape_region_order)), np.nan)
    mat_sq = np.full((len(bre_for_plot), len(shape_region_order)), np.nan)

    for _, row in agg_shape.iterrows():
        if row['bre_region'] in bre_for_plot and row['shape_region'] in shape_region_order:
            i = bre_for_plot.index(row['bre_region'])
            j = shape_region_order.index(row['shape_region'])
            mat_s[i, j]  = row['best_rg']
            mat_sq[i, j] = row['best_fdr_q']

    # Highlight matched BRE × shape diagonal pairs
    is_diag = np.zeros_like(mat_s, dtype=bool)
    for i, br in enumerate(bre_for_plot):
        expected_shape = BRE_TO_SHAPE.get(br)
        if expected_shape and expected_shape in shape_region_order:
            j = shape_region_order.index(expected_shape)
            is_diag[i, j] = True

    vmax_s = np.nanpercentile(np.abs(mat_s), 95)
    norm_s = TwoSlopeNorm(vmin=-vmax_s, vcenter=0, vmax=vmax_s)

    fig, ax = plt.subplots(figsize=(MM(110), MM(100)))
    im = ax.imshow(mat_s, cmap='RdBu_r', norm=norm_s, aspect='auto')

    for i in range(mat_s.shape[0]):
        for j in range(mat_s.shape[1]):
            # FDR significance marker
            if not np.isnan(mat_sq[i, j]) and mat_sq[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=FS_TICK - 1, color='black', fontweight='bold')
            # Diagonal box (matched region)
            if is_diag[i, j]:
                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1.5, edgecolor='gold', facecolor='none', zorder=5)
                ax.add_patch(rect)

    ax.set_xticks(range(len(shape_region_order)))
    ax.set_xticklabels(shape_region_order, rotation=30, ha='right', fontsize=FS_TICK)
    ax.set_yticks(range(len(bre_for_plot)))
    ax.set_yticklabels(bre_disp_plot, fontsize=FS_TICK)
    ax.set_title('Genetic correlation: BRE dims vs shape GWAS PCs\n'
                 '(max |rg| across top-5 dims and all shape PCs; * FDR q<0.05; □ matched region)',
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('Shape GWAS region', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / 'shape_gc_heatmap')
    plt.close()
    print("  Saved: figures/gc/shape_gc_heatmap.pdf/.png")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n  Diagonal (matched region) GC stats:")
    diag_rows = agg_shape[agg_shape.apply(
        lambda r: BRE_TO_SHAPE.get(r['bre_region'], '') == r['shape_region'], axis=1)]
    if len(diag_rows):
        print(diag_rows[['bre_region', 'shape_region', 'best_rg',
                          'best_fdr_q', 'n_fdr_sig']].to_string(index=False))

    off_diag = agg_shape[~agg_shape.apply(
        lambda r: BRE_TO_SHAPE.get(r['bre_region'], '') == r['shape_region'], axis=1)]
    if len(diag_rows) and len(off_diag):
        diag_mean = diag_rows['max_abs_rg'].mean()
        off_mean  = off_diag['max_abs_rg'].mean()
        print(f"\n  Diagonal mean |rg|   = {diag_mean:.4f}")
        print(f"  Off-diagonal mean |rg| = {off_mean:.4f}")
        print(f"  Diagonal / off-diag ratio = {diag_mean / off_mean:.2f}x")

else:
    print("  WARNING: No shape GC results yet — run the munge + rg steps first.")
    print("  Shape GC jobs queued, please wait for LDSC to complete.")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FOR PAPER (CLAIM 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CLAIM 1 SUMMARY")
print("=" * 70)

if len(enigma_df):
    n_enigma_sig = enigma_df['fdr_sig'].sum()
    n_enigma_total = len(enigma_df)
    print(f"  ENIGMA GC: {n_enigma_sig}/{n_enigma_total} pairs FDR-significant (q<0.05)")

if 'shape_gc_df' in dir() and len(shape_gc_df):
    n_shape_sig = shape_gc_df['fdr_sig'].sum()
    n_shape_total = len(shape_gc_df)
    print(f"  Shape GC:  {n_shape_sig}/{n_shape_total} pairs FDR-significant (q<0.05)")

print("\nDone.")
