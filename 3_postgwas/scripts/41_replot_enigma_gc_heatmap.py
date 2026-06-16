#!/usr/bin/env python3
"""
41_replot_enigma_gc_heatmap.py
Replot the ENIGMA rg heatmap (Fig 4D) using max |rg| + sequential colormap.

Reads cached pairs from results/gc/enigma_gc.csv (produced by Script 15 PART 1)
and rewrites figures/gc/enigma_gc_heatmap.{pdf,png}.

Why: BRE dimensions are unlabelled embedding axes — the sign of any individual
dim's rg with an external trait is arbitrary (depends on contrastive-encoder
weight signs, not biology). The previous signed best_rg display caused
apparent opposite-sign rg between L and R hemispheres for the same anatomical
pair (e.g., L.Accumbens vs R.Accumbens against ENIGMA Accumbens) — a
sign-mapping artefact, not lateralization. The caption already documented
"max |rg|"; this replotter aligns the figure with the caption. Updated
2026-05-26 in response to PI comment.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL  # noqa: E402

PROJECT_BASE = SCRIPTS.parent
GC_CSV = PROJECT_BASE / "results" / "gc" / "enigma_gc.csv"
FIG_DIR = PROJECT_BASE / "figures" / "gc"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Region order — match Script 15 / Fig 4D layout (16 BRE rows × 9 ENIGMA cols
# matched-region view; ICV removed because it has no BRE counterpart).
REGIONS_16 = [
    "Left_Accumbens-area", "Right_Accumbens-area",
    "Left_Amygdala",       "Right_Amygdala",
    "Brain_Stem_or_4th_Ventricle", "CSF",
    "Left_Caudate",        "Right_Caudate",
    "Left_Hippocampus",    "Right_Hippocampus",
    "Left_Pallidum",       "Right_Pallidum",
    "Left_Putamen",        "Right_Putamen",
    "Left_Thalamus-Proper","Right_Thalamus-Proper",
]
DISP = {
    "Left_Accumbens-area": "L.Accumbens", "Right_Accumbens-area": "R.Accumbens",
    "Left_Amygdala": "L.Amygdala",         "Right_Amygdala": "R.Amygdala",
    "Brain_Stem_or_4th_Ventricle": "Brainstem", "CSF": "CSF",
    "Left_Caudate": "L.Caudate",           "Right_Caudate": "R.Caudate",
    "Left_Hippocampus": "L.Hippocampus",   "Right_Hippocampus": "R.Hippocampus",
    "Left_Pallidum": "L.Pallidum",         "Right_Pallidum": "R.Pallidum",
    "Left_Putamen": "L.Putamen",           "Right_Putamen": "R.Putamen",
    "Left_Thalamus-Proper": "L.Thalamus",  "Right_Thalamus-Proper": "R.Thalamus",
}
# ENIGMA columns (matched-region view; ICV excluded — see caption).
ENIGMA_REGIONS = ["Accumbens", "Amygdala", "Brainstem", "Caudate", "Hippocampus",
                  "Pallidum", "Putamen", "Thalamus", "ventralDC"]
ENIGMA_DISP = {r: r for r in ENIGMA_REGIONS}


def main() -> None:
    apply_mpl_style()
    df = pd.read_csv(GC_CSV)

    # Aggregate per (BRE region × ENIGMA): max |rg| over BRE dims; best (lowest) FDR q.
    df["abs_rg"] = df["rg"].abs()
    agg = (
        df.groupby(["bre_region", "enigma"])
          .agg(max_abs_rg=("abs_rg", "max"),
               best_fdr_q=("fdr_q", "min"))
          .reset_index()
    )

    mat_rg = np.full((len(REGIONS_16), len(ENIGMA_REGIONS)), np.nan)
    mat_q  = np.full((len(REGIONS_16), len(ENIGMA_REGIONS)), np.nan)
    for _, row in agg.iterrows():
        if row["bre_region"] in REGIONS_16 and row["enigma"] in ENIGMA_REGIONS:
            i = REGIONS_16.index(row["bre_region"])
            j = ENIGMA_REGIONS.index(row["enigma"])
            mat_rg[i, j] = row["max_abs_rg"]
            mat_q[i, j]  = row["best_fdr_q"]

    vmax = np.nanpercentile(mat_rg, 95)

    fig, ax = plt.subplots(figsize=(MM(130), MM(110)))
    im = ax.imshow(mat_rg, cmap="OrRd", vmin=0, vmax=vmax, aspect="auto")

    for i in range(mat_rg.shape[0]):
        for j in range(mat_rg.shape[1]):
            if not np.isnan(mat_q[i, j]) and mat_q[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=FS_TICK - 1, color="black", fontweight="bold")

    # Gold box marking matched anatomical pair on the diagonal where one exists.
    matched_pairs = {
        "Left_Accumbens-area": "Accumbens", "Right_Accumbens-area": "Accumbens",
        "Left_Amygdala": "Amygdala",        "Right_Amygdala": "Amygdala",
        "Brain_Stem_or_4th_Ventricle": "Brainstem",
        "Left_Caudate": "Caudate",          "Right_Caudate": "Caudate",
        "Left_Hippocampus": "Hippocampus",  "Right_Hippocampus": "Hippocampus",
        "Left_Pallidum": "Pallidum",        "Right_Pallidum": "Pallidum",
        "Left_Putamen": "Putamen",          "Right_Putamen": "Putamen",
        "Left_Thalamus-Proper": "Thalamus", "Right_Thalamus-Proper": "Thalamus",
    }
    for r, e in matched_pairs.items():
        if r in REGIONS_16 and e in ENIGMA_REGIONS:
            i = REGIONS_16.index(r); j = ENIGMA_REGIONS.index(e)
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="#D4AF37", linewidth=1.2))

    ax.set_xticks(range(len(ENIGMA_REGIONS)))
    ax.set_xticklabels([ENIGMA_DISP[e] for e in ENIGMA_REGIONS],
                       rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_yticks(range(len(REGIONS_16)))
    ax.set_yticklabels([DISP[r] for r in REGIONS_16], fontsize=FS_TICK)
    ax.set_title("Genetic correlation: BRE dims vs ENIGMA volume\n"
                 "(max |rg| across dims; * FDR q<0.05; gold = matched region)",
                 fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("ENIGMA brain region (volume)", fontsize=FS_LABEL)
    ax.set_ylabel("BRE region", fontsize=FS_LABEL)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Genetic correlation |rg|", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    plt.tight_layout()
    save_mpl(fig, FIG_DIR / "enigma_gc_heatmap")
    plt.close(fig)
    print(f"Wrote: {FIG_DIR / 'enigma_gc_heatmap'}.pdf / .png")

    # Sanity: matched-pair |rg| values, sorted.
    print("\nMatched-region |rg| (highest first):")
    rows = []
    for r, e in matched_pairs.items():
        if r in REGIONS_16 and e in ENIGMA_REGIONS:
            i = REGIONS_16.index(r); j = ENIGMA_REGIONS.index(e)
            rows.append((DISP[r], e, mat_rg[i, j], mat_q[i, j]))
    for nm, en, v, q in sorted(rows, key=lambda x: -x[2] if not np.isnan(x[2]) else 0):
        print(f"  {nm:>14s}  {en:<12s}  |rg|={v:5.3f}   q={q:.3g}")


if __name__ == "__main__":
    main()
