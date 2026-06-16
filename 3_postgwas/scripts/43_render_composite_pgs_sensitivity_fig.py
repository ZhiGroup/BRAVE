#!/usr/bin/env python3
"""
43_render_composite_pgs_sensitivity_fig.py
Render Supplementary Figure S52: per-region PGS aggregation sensitivity.

Compares four PGS variants per region (output of Script 42):
  (1) top-h² rank-1 (current main-paper headline)
  (2) composite mean (5 dims, z-scored, equal weight)
  (3) composite h²-weighted (5 dims, z-scored, h²-weighted)
  (4) best individual per-dim (post-hoc selection across top-5 ranks)

Outputs:
  figures/supplementary/S52_composite_pgs_sensitivity/S52_composite_pgs_sensitivity.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM

PROJECT_BASE = SCRIPT_DIR.parents[1]
DATA = PROJECT_BASE / "results" / "sbayesrc" / "composite" / "headline_summary.csv"
OUT_DIR = PROJECT_BASE / "figures" / "supplementary" / "S52_composite_pgs_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_DISPLAY = {
    "Brain_Stem_or_4th_Ventricle": "Brainstem",
    "CSF": "CSF",
    "Left_Accumbens-area": "L.Accumbens",
    "Right_Accumbens-area": "R.Accumbens",
    "Left_Amygdala": "L.Amygdala",
    "Right_Amygdala": "R.Amygdala",
    "Left_Caudate": "L.Caudate",
    "Right_Caudate": "R.Caudate",
    "Left_Hippocampus": "L.Hippocampus",
    "Right_Hippocampus": "R.Hippocampus",
    "Left_Pallidum": "L.Pallidum",
    "Right_Pallidum": "R.Pallidum",
    "Left_Putamen": "L.Putamen",
    "Right_Putamen": "R.Putamen",
    "Left_Thalamus_Proper": "L.Thalamus",
    "Right_Thalamus-Proper": "R.Thalamus",
}

VARIANTS = [
    ("top1_R2",            "Top-h² rank-1 (main paper)",   "#1f77b4"),
    ("composite_mean_R2",  "Composite (unweighted mean)",  "#ff7f0e"),
    ("composite_h2wt_R2",  "Composite (h²-weighted mean)", "#2ca02c"),
    ("best_per_dim_R2",    "Best individual dim (post-hoc)", "#888888"),
]


def main():
    df = pd.read_csv(DATA)
    df["region_disp"] = df["region"].map(REGION_DISPLAY)
    # Order by top1 R² descending so the figure reads cleanly
    df = df.sort_values("top1_R2", ascending=True).reset_index(drop=True)

    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(MM(180), MM(155)))
    n_regions = len(df)
    n_variants = len(VARIANTS)
    bar_h = 0.18
    y = np.arange(n_regions)

    for k, (col, label, color) in enumerate(VARIANTS):
        offset = (k - (n_variants - 1) / 2) * bar_h
        ax.barh(y + offset, df[col].values, height=bar_h, color=color, label=label, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df["region_disp"].values, fontsize=7.5)
    ax.set_xlabel("R² in independent replication cohort", fontsize=8)
    ax.set_title("PGS aggregation sensitivity — composite mean R² across 16 regions:\n"
                 f"top-h² rank-1 = {df['top1_R2'].mean():.4f}; composite mean = {df['composite_mean_R2'].mean():.4f}; "
                 f"composite h²-wt = {df['composite_h2wt_R2'].mean():.4f}; best per-dim (post-hoc) = {df['best_per_dim_R2'].mean():.4f}",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, max(0.06, df[[c for c, _, _ in VARIANTS]].values.max() * 1.1))

    fig.tight_layout()
    save_mpl(fig, OUT_DIR / "S52_composite_pgs_sensitivity")
    plt.close(fig)
    print(f"Saved: {OUT_DIR}/S52_composite_pgs_sensitivity.{{pdf,png}}")


if __name__ == "__main__":
    main()
