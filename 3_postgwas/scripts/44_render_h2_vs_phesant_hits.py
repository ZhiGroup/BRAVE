#!/usr/bin/env python3
"""
44_render_h2_vs_phesant_hits.py
Render Supplementary Figure S53: per-region mean SNP heritability vs non-brain
PHESANT hit count scatter (anchor for the Results §1 Spearman ρ = 0.19 claim).

Inputs:
  - LDSC h2_table.csv                        (per-region per-dim LDSC h²)
  - results/phesant/aggregate_per_region.csv (per-region non-brain hit count)

Output:
  figures/supplementary/S53_h2_vs_phesant_hits/S53_h2_vs_phesant_hits.{pdf,png}
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM

PROJECT_BASE = SCRIPT_DIR.parents[1]
H2_CSV = Path("<EXTERNAL: LDSC h2_table.csv (per-region per-dim canonical h2 table)>")
PHESANT_CSV = PROJECT_BASE / "results" / "phesant" / "aggregate_per_region.csv"
OUT_DIR = PROJECT_BASE / "figures" / "supplementary" / "S53_h2_vs_phesant_hits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_DISPLAY = {
    "Brain_Stem_or_4th_Ventricle": "Brainstem",
    "CSF": "CSF",
    "Left_Accumbens-area": "L.Acc",
    "Right_Accumbens-area": "R.Acc",
    "Left_Amygdala": "L.Amy",
    "Right_Amygdala": "R.Amy",
    "Left_Caudate": "L.Cau",
    "Right_Caudate": "R.Cau",
    "Left_Hippocampus": "L.Hip",
    "Right_Hippocampus": "R.Hip",
    "Left_Pallidum": "L.Pal",
    "Right_Pallidum": "R.Pal",
    "Left_Putamen": "L.Put",
    "Right_Putamen": "R.Put",
    # PHESANT uses underscore for Left, hyphen for Right (see aggregate_per_region.csv); cover both.
    "Left_Thalamus_Proper": "L.Tha",
    "Left_Thalamus-Proper": "L.Tha",
    "Right_Thalamus-Proper": "R.Tha",
    "Right_Thalamus_Proper": "R.Tha",
}

HEMI_COLOUR = {
    "L": "#1f77b4",
    "R": "#d62728",
    "M": "#888888",
}


def hemi_of(region):
    if region.startswith("Left_"): return "L"
    if region.startswith("Right_"): return "R"
    return "M"


def main():
    # Per-region mean h² over all dims
    h2 = pd.read_csv(H2_CSV)
    mean_h2 = h2.groupby("reg")["h2"].mean()

    # Per-region non-brain hit count
    ph = pd.read_csv(PHESANT_CSV)
    nonbrain = ph.set_index("region")["n_sig_nonbrain"]

    # Build the 16-region table by aligning the two SSoTs
    # PHESANT uses "Left_Thalamus_Proper", h² uses "Left_Thalamus-Proper" — bridge
    region_map = {
        "Left_Thalamus_Proper": "Left_Thalamus-Proper",
        "Right_Thalamus-Proper": "Right_Thalamus-Proper",
    }
    rows = []
    for r in ph["region"]:
        r_h2 = region_map.get(r, r)
        if r_h2 in mean_h2.index:
            rows.append({"region": r, "mean_h2": mean_h2[r_h2],
                         "nonbrain_hits": int(nonbrain[r]),
                         "hemi": hemi_of(r),
                         "label": REGION_DISPLAY.get(r, r)})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} regions")

    # Spearman ρ — over ALL 16 regions AND over the 15 regions excluding R.Thalamus
    rho_all, p_all = spearmanr(df["mean_h2"], df["nonbrain_hits"])
    excl = df[df["region"] != "Right_Thalamus-Proper"]
    rho_excl, p_excl = spearmanr(excl["mean_h2"], excl["nonbrain_hits"])
    print(f"All 16 regions: ρ = {rho_all:.3f}, p = {p_all:.3f}")
    print(f"15 regions (excl. R.Thalamus): ρ = {rho_excl:.3f}, p = {p_excl:.3f}")

    # ── Render ────────────────────────────────────────────────────────────────
    apply_mpl_style()
    # Slightly wider figure so labels have room to push outward
    fig, ax = plt.subplots(figsize=(MM(130), MM(95)))
    from adjustText import adjust_text

    texts = []
    for _, r in df.iterrows():
        is_rtha = (r["region"] in ("Right_Thalamus-Proper", "Right_Thalamus_Proper"))
        ax.scatter(r["mean_h2"], r["nonbrain_hits"],
                   s=70, c=HEMI_COLOUR[r["hemi"]],
                   edgecolor="black" if is_rtha else "white",
                   linewidth=1.8 if is_rtha else 0.6,
                   alpha=0.92, zorder=3)
        texts.append(ax.text(r["mean_h2"], r["nonbrain_hits"], r["label"],
                             fontsize=7.5,
                             fontweight="bold" if is_rtha else "normal",
                             zorder=4))

    # Auto-adjust label positions to avoid overlap; leader lines for displaced labels
    adjust_text(
        texts,
        ax=ax,
        expand_points=(1.4, 1.6),
        expand_text=(1.1, 1.3),
        force_text=(0.6, 0.8),
        force_points=(0.4, 0.5),
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.6),
    )

    ax.set_xlabel("Per-region mean SNP heritability (LDSC h², averaged across 128 BRE dimensions)",
                 fontsize=8)
    ax.set_ylabel("Non-brain PHESANT hit count (q < 0.05)", fontsize=8)
    ax.set_title(f"Per-region h² vs non-brain PHESANT hit count\n"
                 f"Spearman ρ = {rho_excl:.2f}, p = {p_excl:.2f} (excl. R.Thalamus, n = 15)",
                 fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # Legend for hemisphere colour
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HEMI_COLOUR["L"], markersize=8, label="Left"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HEMI_COLOUR["R"], markersize=8, label="Right"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HEMI_COLOUR["M"], markersize=8, label="Midline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, markeredgewidth=1.6, label="R.Thalamus (outlier)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    save_mpl(fig, OUT_DIR / "S53_h2_vs_phesant_hits")
    plt.close(fig)

    # Save the 16-row CSV so the audit chain is complete
    out_csv = OUT_DIR / "S53_h2_vs_phesant_hits.csv"
    df[["region", "label", "hemi", "mean_h2", "nonbrain_hits"]].to_csv(out_csv, index=False)
    print(f"\nSaved figure + CSV to {OUT_DIR}")
    print(f"  S53_h2_vs_phesant_hits.{{pdf,png}}")
    print(f"  S53_h2_vs_phesant_hits.csv (16 rows)")


if __name__ == "__main__":
    main()
