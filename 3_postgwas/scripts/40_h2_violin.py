#!/usr/bin/env python3
"""
40_h2_violin.py
Per-region LDSC h² violin for Fig 3 panel E (post D/E swap; was panel D).

Addresses PI comment B2 — "L.Accumbens steals the thunder" — by:
1. Sorting regions by **mean h²** (ascending bottom→top) so L.Accumbens sits
   in the middle of the figure rather than near the start.
2. Using **constant violin width** so regions with fewer LDSC-qualified dims
   (L.Accumbens: 43 dims; CSF: 37; Brainstem: 33; L.Amygdala: 59) don't
   produce sharper, taller-looking peaks than regions with 128 dims.
3. Colouring violins by hemisphere (left = blue, right = red, midline = grey).
4. Annotating the mean h² above each violin.

Input:  LDSC h2_table.csv (per-region per-dim LDSC h²)
Output: figures/h2/h2_violin.{pdf,png}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

POST_BASE = Path(__file__).resolve().parents[1]
H2_CSV_DEFAULT = Path("<EXTERNAL: LDSC h2_table.csv (per-region per-dim canonical h2 table)>")
OUT_DIR_DEFAULT = POST_BASE / "figures" / "h2"

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


REGION_SHORT = {
    "Brain_Stem_or_4th_Ventricle": "Brainstem", "CSF": "CSF",
    "Left_Accumbens-area": "L.Acc",  "Right_Accumbens-area": "R.Acc",
    "Left_Amygdala": "L.Amyg",       "Right_Amygdala": "R.Amyg",
    "Left_Caudate": "L.Cau",         "Right_Caudate": "R.Cau",
    "Left_Hippocampus": "L.Hipp",    "Right_Hippocampus": "R.Hipp",
    "Left_Pallidum": "L.Pal",        "Right_Pallidum": "R.Pal",
    "Left_Putamen": "L.Put",         "Right_Putamen": "R.Put",
    "Left_Thalamus-Proper": "L.Thal", "Right_Thalamus-Proper": "R.Thal",
}


def _hemi_colour(reg: str) -> str:
    if reg in {"Brain_Stem_or_4th_Ventricle", "CSF"}:
        return "#7F7F7F"
    if reg.startswith("Left_"):
        return "#2166AC"
    return "#B2182B"


def render(h2_csv: Path, out_dir: Path):
    apply_mpl_style()
    df = pd.read_csv(h2_csv)
    means = df.groupby("reg")["h2"].mean().sort_values()
    region_order = means.index.tolist()
    data = [df.loc[df["reg"] == r, "h2"].values for r in region_order]

    fig, ax = plt.subplots(figsize=(MM(180), MM(85)))
    positions = np.arange(len(region_order))

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,                   # constant — independent of n_dims
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    # Custom face / edge colours per hemisphere
    for body, reg in zip(parts["bodies"], region_order):
        body.set_facecolor(_hemi_colour(reg))
        body.set_alpha(0.55)
        body.set_edgecolor("#222")
        body.set_linewidth(0.5)

    # Mean h² markers + value labels
    mean_vals = [d.mean() for d in data]
    ax.scatter(positions, mean_vals, s=18, color="black",
               edgecolor="white", linewidth=0.6, zorder=4)
    for i, (pos, m) in enumerate(zip(positions, mean_vals)):
        ax.text(pos, max(data[i]) + 0.012, f"{m:.3f}",
                ha="center", va="bottom", fontsize=6, color="#222")

    # n_dims annotation below each violin (smaller, italic)
    for i, reg in enumerate(region_order):
        ax.text(positions[i], -0.012, f"n={len(data[i])}",
                ha="center", va="top", fontsize=5.5,
                color="#555", style="italic")

    ax.set_xticks(positions)
    ax.set_xticklabels([REGION_SHORT[r] for r in region_order],
                       rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("LDSC SNP heritability (h²)", fontsize=8)
    ax.set_title("Per-region BRE-dimension h² (regions sorted ascending by mean)",
                 loc="left", fontweight="bold", fontsize=9, pad=4)
    ax.set_ylim(-0.02, 0.35)
    ax.set_xlim(-0.7, len(region_order) - 0.3)
    ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        mpatches.Patch(facecolor="#2166AC", alpha=0.55, label="Left hemisphere"),
        mpatches.Patch(facecolor="#B2182B", alpha=0.55, label="Right hemisphere"),
        mpatches.Patch(facecolor="#7F7F7F", alpha=0.55, label="Midline (Brainstem, CSF)"),
    ]
    ax.legend(handles=legend, fontsize=6.5, frameon=False, loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / "h2_violin"
    save_mpl(fig, out_stem)
    plt.close(fig)
    print(f"Wrote: {out_stem}.pdf / .png")
    print(f"  Order: {' < '.join(REGION_SHORT[r] for r in region_order)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h2-csv", type=Path, default=H2_CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = p.parse_args()
    render(args.h2_csv, args.out_dir)


if __name__ == "__main__":
    main()
