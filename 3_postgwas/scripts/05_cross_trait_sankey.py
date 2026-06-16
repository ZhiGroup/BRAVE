"""
05_cross_trait_sankey.py
Cross-trait pleiotropy analysis using FUMA GWAS catalog output.

Approach:
  - For each region: load gwascatalog.txt + snps.txt
  - Filter catalog hits at p < 9e-6 (Zhao/Zhu lab threshold)
  - Merge with snps.txt on rsID to get LD info (r2 >= 0.6 with our lead/IndSigSNPs)
  - Categorise traits into organ systems
  - Visualise as mirrored tripartite Sankey:
      Left regions -> Organ categories -> Right/Midline regions

Two output versions:
  (1) Including Brain/Imaging traits (shows all pleiotropy)
  (2) Excluding Brain/Imaging (non-brain pleiotropy; brain traits dominate otherwise)

Adapted from an internal cross-trait association notebook.
Updated for new FUMA path and corrected folder naming (no _JAGWAS suffix).
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import plotly_layout, save_plotly, _fmt_region_label, NODE_COLOR, FS_LABEL

# ── PATHS ─────────────────────────────────────────────────────────────────────
FUMA_DIR  = Path(cfg.postgwas.fuma_dir)
OUT_DIR   = Path(__file__).parents[1] / "results" / "cross_trait"
FIG_DIR   = Path(__file__).parents[1] / "figures" / "cross_trait"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── SETTINGS ──────────────────────────────────────────────────────────────────
CATALOG_P_THRESH = 9e-6   # Zhao/Zhu lab standard
LD_R2_THRESH     = 0.6    # minimum r2 for LD-proxy inclusion

# ── TRAIT CATEGORISATION ──────────────────────────────────────────────────────
CATEGORY_MAP = {
    'Brain/Imaging':      ['volume', 'cortical', 'thickness', 'surface area',
                           'white matter', 'diffusion', 'brain morphology',
                           'fractional anisotropy', 'brain'],
    'Neurological':       ['alzheimer', 'parkinson', 'dementia', 'stroke', 'epilepsy'],
    'Psychiatric':        ['schizophrenia', 'bipolar', 'depress', 'anxiety', 'adhd'],
    'Eye/Vision':         ['glaucoma', 'retinal', 'macular', 'oct', 'refractive', 'vision'],
    'Cardiovascular':     ['heart', 'atrial', 'aorta', 'coronary', 'hypertension',
                           'blood pressure'],
    'Renal/Kidney':       ['egfr', 'kidney', 'creatinine', 'renal', 'nephro'],
    'Hepatic/Liver':      ['liver', 'cirrhosis', 'hepatitis', 'steatosis', 'bilirubin'],
    'Pulmonary/Lung':     ['lung', 'copd', 'respiratory', 'asthma', 'spirometry'],
    'Metabolic':          ['diabetes', 'glucose', 'cholesterol', 'triglyceride',
                           'bmi', 'weight'],
    'Cognitive/Education':['intelligence', 'cognitive', 'education', 'math', 'reasoning'],
}

CATEGORY_COLORS = {
    'Brain/Imaging':       (31,  119, 180),
    'Neurological':        (228,  26,  28),
    'Psychiatric':         (55,  126, 184),
    'Eye/Vision':          (77,  175,  74),
    'Cognitive/Education': (152,  78, 163),
    'Cardiovascular':      (255, 127,   0),
    'Renal/Kidney':        (230, 220,  50),
    'Metabolic':           (166,  86,  40),
    'Hepatic/Liver':       (247, 129, 191),
    'Pulmonary/Lung':      (153, 153, 153),
}

def categorise(trait: str) -> str:
    t = str(trait).lower()
    # Brain/Imaging checked first so morphological traits don't fall into Neurological
    for cat, kws in CATEGORY_MAP.items():
        if any(k in t for k in kws):
            return cat
    return 'Other'

# ── REGION NAMING ─────────────────────────────────────────────────────────────
# Normalise folder names → (side, base_region)
# Handles Right_Thalamus-Proper vs Left_Thalamus_Proper inconsistency
def parse_region(folder: str):
    name = folder.strip()
    name_low = name.lower()
    if name_low.startswith('left_'):
        side = 'Left'
        base = name[5:]   # strip 'Left_'
    elif name_low.startswith('right_'):
        side = 'Right'
        base = name[6:]   # strip 'Right_'
    else:
        side = 'Midline'
        base = name
    # normalise thalamus naming inconsistency
    base = base.replace('Thalamus-Proper', 'Thalamus_Proper')
    return side, base

# ── LOAD AND MERGE ────────────────────────────────────────────────────────────
print("Loading FUMA data ...")
all_rows = []

for region_dir in sorted(FUMA_DIR.iterdir()):
    if not region_dir.is_dir():
        continue
    folder = region_dir.name
    cat_file = region_dir / "gwascatalog.txt"
    snp_file = region_dir / "snps.txt"

    if not cat_file.exists() or not snp_file.exists():
        print(f"  SKIP {folder}: missing files")
        continue

    gwas_cat = pd.read_csv(cat_file, sep='\t', low_memory=False)
    snps_df  = pd.read_csv(snp_file,  sep='\t', low_memory=False)

    # Filter catalog to threshold and merge with snps for LD info
    gwas_filt = gwas_cat[gwas_cat['P'] < CATALOG_P_THRESH].copy()
    merged = pd.merge(
        gwas_filt,
        snps_df[['rsID', 'IndSigSNP', 'GenomicLocus', 'r2']],
        left_on='snp', right_on='rsID', how='inner'
    )
    # Apply LD filter
    merged = merged[merged['r2'] >= LD_R2_THRESH]

    if merged.empty:
        print(f"  {folder}: no hits after filtering")
        continue

    side, base = parse_region(folder)
    merged['folder']       = folder
    merged['Our_Region']   = f"{side}_{base}" if side != 'Midline' else base
    merged['Side']         = side
    merged['Base_Region']  = base
    merged['Organ_Category'] = merged['Trait'].apply(categorise)

    all_rows.append(merged)
    print(f"  {folder}: {len(merged)} catalog hits after LD filter")

master_df = pd.concat(all_rows, ignore_index=True)
master_df.to_csv(OUT_DIR / "cross_trait_all_categories.csv", index=False)
print(f"\nTotal rows: {len(master_df)}")
print(f"Saved: {OUT_DIR / 'cross_trait_all_categories.csv'}")

# ── SANKEY BUILDER ────────────────────────────────────────────────────────────
BASE_ORDER = ['Pallidum', 'Putamen', 'Hippocampus', 'Caudate',
              'Thalamus_Proper', 'Accumbens-area', 'Amygdala']
MIDLINE_ORDER = ['Brain_Stem_or_4th_Ventricle', 'CSF']

def build_sankey(plot_df: pd.DataFrame, title: str, out_stem: str):
    """Build and save a mirrored tripartite Sankey diagram."""
    if plot_df.empty:
        print(f"No data for: {title}")
        return

    cat_list = sorted(plot_df['Organ_Category'].unique())

    # ── node positions ──
    node_labels, x_pos, y_pos = [], [], []
    y_steps   = np.linspace(0.02, 0.98, len(BASE_ORDER) + len(MIDLINE_ORDER))
    y_map     = {b: y_steps[i] for i, b in enumerate(BASE_ORDER)}
    mid_y_map = {m: y_steps[len(BASE_ORDER) + i] for i, m in enumerate(MIDLINE_ORDER)}

    # Left column
    for base in BASE_ORDER:
        node_labels.append(f"Left_{base}"); x_pos.append(0.08); y_pos.append(y_map[base])

    # Middle — organ categories
    cat_y = np.linspace(0.02, 0.98, len(cat_list))
    for i, cat in enumerate(cat_list):
        node_labels.append(cat); x_pos.append(0.5); y_pos.append(cat_y[i])

    # Right column + midline
    for base in BASE_ORDER:
        node_labels.append(f"Right_{base}"); x_pos.append(0.88); y_pos.append(y_map[base])
    for mid in MIDLINE_ORDER:
        node_labels.append(mid); x_pos.append(0.88); y_pos.append(mid_y_map[mid])

    node_idx = {lbl: i for i, lbl in enumerate(node_labels)}

    # ── link colours ──
    max_v = plot_df.groupby(['Our_Region', 'Organ_Category']).size().max()

    def link_color(cat, val):
        rgb   = CATEGORY_COLORS.get(cat, (128, 128, 128))
        alpha = 0.2 + (val / max_v) * 0.5
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f})"

    sources, targets, values, colors = [], [], [], []

    # Left → Middle
    left_links = (plot_df[plot_df['Side'] == 'Left']
                  .groupby(['Our_Region', 'Organ_Category']).size()
                  .reset_index(name='v'))
    for _, row in left_links.iterrows():
        if row['Our_Region'] in node_idx and row['Organ_Category'] in node_idx:
            sources.append(node_idx[row['Our_Region']])
            targets.append(node_idx[row['Organ_Category']])
            values.append(row['v'])
            colors.append(link_color(row['Organ_Category'], row['v']))

    # Middle → Right/Midline
    rm_links = (plot_df[plot_df['Side'].isin(['Right', 'Midline'])]
                .groupby(['Our_Region', 'Organ_Category']).size()
                .reset_index(name='v'))
    for _, row in rm_links.iterrows():
        if row['Organ_Category'] in node_idx and row['Our_Region'] in node_idx:
            sources.append(node_idx[row['Organ_Category']])
            targets.append(node_idx[row['Our_Region']])
            values.append(row['v'])
            colors.append(link_color(row['Organ_Category'], row['v']))

    clean_labels = [_fmt_region_label(l) for l in node_labels]

    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=60, thickness=12,
            label=clean_labels, x=x_pos, y=y_pos,
            line=dict(color='#444444', width=0.5),
            color=NODE_COLOR,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=colors,
            line=dict(color='rgba(0,0,0,0.05)', width=0.3),
        ),
    )])
    fig.update_layout(**plotly_layout(
        title=title,
        width=1200, height=950,
        # t=130 (was 75) reserves top whitespace so the title sits well below the
        # source PNG's top edge; this keeps the title visible after Fig 5
        # composite-assembly tight-bbox cropping (PI feedback 2026-06-02 —
        # "figure 5D is cutoff at the top").
        margin=dict(l=60, r=200, t=130, b=55),
    ))
    save_plotly(fig, FIG_DIR / out_stem, width=1200, height=950)

# ── VERSION 1: all categories (including Brain/Imaging) ─────────────────────
plot_all = master_df[master_df['Organ_Category'] != 'Other'].copy()
plot_all.to_csv(OUT_DIR / "cross_trait_all_categories_filtered.csv", index=False)
build_sankey(
    plot_all,
    title="Multi-Organ Genetic Pleiotropy: Subcortical JAGWAS (all trait categories)",
    out_stem="sankey_all_categories",
)

# ── VERSION 2: exclude Brain/Imaging (non-brain pleiotropy focus) ─────────────
plot_nobrain = master_df[
    (master_df['Organ_Category'] != 'Other') &
    (master_df['Organ_Category'] != 'Brain/Imaging')
].copy()
plot_nobrain.to_csv(OUT_DIR / "cross_trait_no_brain_imaging.csv", index=False)
build_sankey(
    plot_nobrain,
    title="Multi-Organ Genetic Pleiotropy: Subcortical JAGWAS (non-brain trait categories)",
    out_stem="sankey_no_brain_imaging",
)

# ── VERSION 3: Brain/Imaging only (positive-control panel for Fig 5D) ────────
plot_brain = master_df[master_df['Organ_Category'] == 'Brain/Imaging'].copy()
plot_brain.to_csv(OUT_DIR / "cross_trait_brain_only.csv", index=False)
build_sankey(
    plot_brain,
    title="Cross-trait GWAS Catalog Sankey: brain / neuroimaging traits (positive control)",
    out_stem="sankey_brain_only",
)

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
summary = (master_df[master_df['Organ_Category'] != 'Other']
           .groupby(['Our_Region', 'Organ_Category'])
           .agg(n_hits=('snp', 'count'), n_traits=('Trait', 'nunique'))
           .reset_index()
           .sort_values(['Our_Region', 'n_hits'], ascending=[True, False]))
summary.to_csv(OUT_DIR / "cross_trait_summary_by_region_category.csv", index=False)
print(f"\nSummary table saved: {OUT_DIR / 'cross_trait_summary_by_region_category.csv'}")
print("\nHits per category (across all regions):")
print(summary.groupby('Organ_Category')['n_hits'].sum().sort_values(ascending=False).to_string())
