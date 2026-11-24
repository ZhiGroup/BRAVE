#!/usr/bin/env python3
"""
50_replot_gc_heatmaps.py
Re-draw the three untargeted genetic-correlation heatmaps (Supplementary Figs. S43, S55,
S56) from the region-level result tables that Scripts 16, 18 and 19 already saved, with
the ranking named in the title ("top-5 fastGWA-REML h²-ranked dimensions").

Why a separate script: Scripts 16, 18 and 19 run LDSC before they plot, so re-running them
to change a title would recompute every genetic correlation. This script reads only the
saved tables. Its plotting code is copied from those scripts (16 L277-327, 18 L352-443,
19 L394-447, as of 2026-11-22) with only the title text changed.

Inputs (read only):
  results/gc/disease_gc_region.csv       Script 16 (region x disease, max |rg| over 5 dims)
  results/gc_heart/heart_gc_region.csv   Script 18 (region x cardiac trait)
  results/gc_oct/oct_gc_region.csv       Script 19 (region x OCT trait)
  --disease-csv  disease list (decides which disease columns exist), as Script 16
  --bai82-csv    cardiac trait names and categories, as Script 18
  --oct-meta     OCT trait order (xlsx row order), as Script 19

Outputs: <out-dir>/disease_gc_heatmap.{pdf,png}, heart_gc_heatmap_all.{pdf,png},
         oct_gc_heatmap.{pdf,png}

Usage:
  $PY 50_replot_gc_heatmaps.py --out-dir <dir> [--which disease heart oct]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL  # noqa: E402

TOP_K = 5
RANK_LABEL = "top-{0} fastGWA-REML h²-ranked dimensions".format(TOP_K)

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
    'Brain_Stem_or_4th_Ventricle': 'BrainStem', 'CSF': 'CSF',
    'Left_Accumbens-area': 'L.Accumbens', 'Right_Accumbens-area': 'R.Accumbens',
    'Left_Amygdala': 'L.Amygdala', 'Right_Amygdala': 'R.Amygdala',
    'Left_Caudate': 'L.Caudate', 'Right_Caudate': 'R.Caudate',
    'Left_Hippocampus': 'L.Hippocampus', 'Right_Hippocampus': 'R.Hippocampus',
    'Left_Pallidum': 'L.Pallidum', 'Right_Pallidum': 'R.Pallidum',
    'Left_Putamen': 'L.Putamen', 'Right_Putamen': 'R.Putamen',
    'Left_Thalamus_Proper': 'L.Thalamus', 'Right_Thalamus-Proper': 'R.Thalamus',
}
# Script 16
DISEASE_ORDER = ['SCZ', 'BIP', 'MDD', 'ADHD', 'ASD',
                 'Neuroticism', 'ALZ', 'IS', 'ALS', 'SD']
DISEASE_DISP = {
    'SCZ': 'Schizophrenia', 'BIP': 'Bipolar', 'MDD': 'Depression', 'ADHD': 'ADHD',
    'ASD': 'Autism', 'Neuroticism': 'Neuroticism', 'ALZ': 'Alzheimer\'s',
    'IS': 'Isch. Stroke', 'ALS': 'ALS', 'SD': 'Sleep Disorder',
}
# Script 18
CARDIAC_CAT_ORDER = ['left ventricle', 'right ventricle', 'left atrium', 'right atrium',
                     'ascending aorta', 'descending aorta']


def star_cells(ax, mat_q):
    for i in range(mat_q.shape[0]):
        for j in range(mat_q.shape[1]):
            if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=FS_TICK, color='black', fontweight='bold')


def plot_disease(res_dir, disease_csv, out_dir):
    agg = pd.read_csv(res_dir / "gc" / "disease_gc_region.csv")
    diseases = list(pd.read_csv(disease_csv)['Trait'])
    region_order = [r for r in REGIONS_16 if r in agg['bre_region'].values]
    disp_order = [DISP[r] for r in region_order]
    disease_cols = [d for d in DISEASE_ORDER if d in diseases]
    disease_lbls = [DISEASE_DISP[d] for d in disease_cols]

    mat_rg = np.full((len(region_order), len(disease_cols)), np.nan)
    mat_q = np.full((len(region_order), len(disease_cols)), np.nan)
    for _, row in agg.iterrows():
        if row['bre_region'] in region_order and row['disease'] in disease_cols:
            i = region_order.index(row['bre_region'])
            j = disease_cols.index(row['disease'])
            mat_rg[i, j] = row['best_rg']
            mat_q[i, j] = row['best_fdr_q']

    vmax = np.nanpercentile(np.abs(mat_rg[~np.isnan(mat_rg)]), 95) \
        if not np.all(np.isnan(mat_rg)) else 0.3
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(MM(140), MM(110)))
    im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')
    star_cells(ax, mat_q)
    ax.set_xticks(range(len(disease_cols)))
    ax.set_xticklabels(disease_lbls, rotation=40, ha='right', fontsize=FS_TICK)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(disp_order, fontsize=FS_TICK)
    ax.set_title('Genetic correlation: BRE dims vs brain diseases\n'
                 '({0} per region; max |rg|; * FDR q < 0.05)'.format(RANK_LABEL),
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('Brain disease / trait', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    plt.tight_layout()
    save_mpl(fig, out_dir / 'disease_gc_heatmap')
    plt.close(fig)
    print("disease: {0} x {1} cells, {2} non-empty, stars {3}".format(
        mat_rg.shape[0], mat_rg.shape[1], int(np.isfinite(mat_rg).sum()),
        int((mat_q < 0.05).sum())))


def plot_heart(res_dir, bai82_csv, out_dir):
    agg = pd.read_csv(res_dir / "gc_heart" / "heart_gc_region.csv")
    bai82 = pd.read_csv(bai82_csv)
    meta = {int(r['ID']): {'name': r['Name'], 'category': r['Category']}
            for _, r in bai82.iterrows()}

    def cat_sort_key(pid):
        cat = meta.get(pid, {}).get('category', 'z')
        idx = CARDIAC_CAT_ORDER.index(cat) if cat in CARDIAC_CAT_ORDER else len(CARDIAC_CAT_ORDER)
        return (idx, pid)

    trait_ids = sorted(agg['pheno_id'].unique(), key=cat_sort_key)
    trait_labels = [meta[pid]['name'] for pid in trait_ids]
    region_order = [r for r in REGIONS_16 if r in agg['bre_region'].values]
    disp_order = [DISP[r] for r in region_order]

    mat_rg = np.full((len(region_order), len(trait_ids)), np.nan)
    mat_q = np.full((len(region_order), len(trait_ids)), np.nan)
    for _, row in agg.iterrows():
        if row['bre_region'] in region_order and row['pheno_id'] in trait_ids:
            i = region_order.index(row['bre_region'])
            j = trait_ids.index(row['pheno_id'])
            mat_rg[i, j] = row['best_rg']
            mat_q[i, j] = row['best_fdr_q']

    valid = mat_rg[~np.isnan(mat_rg)]
    vmax = max(np.percentile(np.abs(valid), 95), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(MM(240), MM(110)))
    im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')
    star_cells(ax, mat_q)
    ax.set_xticks(range(len(trait_labels)))
    ax.set_xticklabels(trait_labels, rotation=45, ha='right', fontsize=FS_TICK - 1)
    ax.set_yticks(range(len(disp_order)))
    ax.set_yticklabels(disp_order, fontsize=FS_TICK)
    ax.set_title('Genetic correlation: BRE dims vs all 82 cardiac traits ({0})\n'
                 '* FDR q < 0.05'.format(RANK_LABEL),
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('Cardiac trait (Bai et al.)', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    plt.tight_layout()
    save_mpl(fig, out_dir / 'heart_gc_heatmap_all')
    plt.close(fig)
    print("heart: {0} x {1} cells, {2} non-empty, stars {3}".format(
        mat_rg.shape[0], mat_rg.shape[1], int(np.isfinite(mat_rg).sum()),
        int((mat_q < 0.05).sum())))


def plot_oct(res_dir, oct_meta, out_dir):
    agg = pd.read_csv(res_dir / "gc_oct" / "oct_gc_region.csv")
    raw = pd.read_excel(oct_meta, header=None)
    header_row = next(i for i, r in raw.iterrows() if 'File_name' in r.values)
    oct_df = pd.read_excel(oct_meta, header=header_row).dropna(subset=['File_name'])
    xlsx_ids = [str(x).strip() for x in oct_df['ID']]
    trait_order = [t for t in xlsx_ids if t in agg['trait_id'].values]

    def shorten(label):
        return label.replace('_thickness', '').replace('_left', '_L').replace('_right', '_R')

    trait_labels_short = [shorten(t) for t in trait_order]
    region_order = [r for r in REGIONS_16 if r in agg['bre_region'].values]
    disp_order = [DISP[r] for r in region_order]

    mat_rg = np.full((len(region_order), len(trait_order)), np.nan)
    mat_q = np.full((len(region_order), len(trait_order)), np.nan)
    for _, row in agg.iterrows():
        if row['bre_region'] in region_order and row['trait_id'] in trait_order:
            i = region_order.index(row['bre_region'])
            j = trait_order.index(row['trait_id'])
            mat_rg[i, j] = row['best_rg']
            mat_q[i, j] = row['best_fdr_q']

    valid = mat_rg[~np.isnan(mat_rg)]
    vmax = max(np.percentile(np.abs(valid), 95), 0.05) if len(valid) > 0 else 0.3
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(MM(180), MM(110)))
    im = ax.imshow(mat_rg, cmap='RdBu_r', norm=norm, aspect='auto')
    star_cells(ax, mat_q)
    ax.set_xticks(range(len(trait_order)))
    ax.set_xticklabels(trait_labels_short, rotation=45, ha='right', fontsize=FS_TICK - 1)
    ax.set_yticks(range(len(disp_order)))
    ax.set_yticklabels(disp_order, fontsize=FS_TICK)
    ax.set_title('Genetic correlation: BRE dims vs retinal OCT traits ({0})\n'
                 '* FDR q < 0.05'.format(RANK_LABEL),
                 fontsize=FS_LABEL, fontweight='bold')
    ax.set_xlabel('Retinal OCT trait', fontsize=FS_LABEL)
    ax.set_ylabel('BRE region', fontsize=FS_LABEL)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Genetic correlation (rg)', fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    plt.tight_layout()
    save_mpl(fig, out_dir / 'oct_gc_heatmap')
    plt.close(fig)
    print("oct: {0} x {1} cells, {2} non-empty, stars {3}".format(
        mat_rg.shape[0], mat_rg.shape[1], int(np.isfinite(mat_rg).sum()),
        int((mat_q < 0.05).sum())))


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res-dir", default=str(here.parent / "results"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--which", nargs="+", default=["disease", "heart", "oct"],
                   choices=["disease", "heart", "oct"])
    p.add_argument("--disease-csv",
                   default="<EXTERNAL: disease sumstats manifest csv>")
    p.add_argument("--bai82-csv",
                   default="<EXTERNAL: Bai et al. cardiac-trait reference csv>")
    p.add_argument("--oct-meta",
                   default=str(Path(cfg.postgwas.oct_sumstats_dir) / "ID_OCT.xlsx"))
    a = p.parse_args()

    res_dir = Path(a.res_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_mpl_style()
    if "disease" in a.which:
        plot_disease(res_dir, a.disease_csv, out_dir)
    if "heart" in a.which:
        plot_heart(res_dir, a.bai82_csv, out_dir)
    if "oct" in a.which:
        plot_oct(res_dir, a.oct_meta, out_dir)
    print("wrote to {0}".format(out_dir))


if __name__ == "__main__":
    main()
