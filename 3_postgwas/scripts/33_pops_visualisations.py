#!/usr/bin/env python3
"""
33_pops_visualisations.py
Two PoPS supplementary figures, in the style of Weeks 2023 Fig. 5 and Fig. 4b.

(F1) PoPS-effector summary (Weeks Fig 5 analogue):
     For a curated subset of loci (Tier 1, ranked by PoPS_Score), show the top-1 PoPS
     gene per locus × region, with shaded boxes indicating which other locus-based
     FUMA methods (closest gene, positional, eQTL, chromatin interaction) also
     nominate that gene. Demonstrates Weeks' "PoPS + local" combined nomination.

(F2) Cross-method agreement matrix (Weeks Fig 4b analogue):
     5×5 matrix of FUMA locus-based methods + PoPS. Cell colour = proportion of loci
     where both methods nominate the *same* top-1 gene, among loci where both
     methods nominate any gene. Validates that PoPS adds independent information
     beyond the individual FUMA mapping methods.

Inputs:
  results/pops/high_confidence_genes.csv (Script 30 aggregation, refactored to
                                          Weeks-style tiering 2026-05-26)

Outputs:
  figures/pops/pops_tier1_effector_table.{pdf,png}
  figures/pops/pops_method_agreement.{pdf,png}

Compute env: Python 3.7
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

POST_BASE = Path(__file__).resolve().parents[1]
HC_CSV = POST_BASE / "results/pops/high_confidence_genes.csv"
FIG_DIR = POST_BASE / "figures/pops"

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


METHOD_LABELS = {
    "is_closest_gene": "Closest",
    "in_positional":   "Positional",
    "in_eqtl":         "eQTL",
    "in_ciMap":        "ciMap",
}
METHOD_ORDER = ["is_closest_gene", "in_positional", "in_eqtl", "in_ciMap"]


def _region_short(folder: str) -> str:
    m = {
        "Brain_Stem_or_4th_Ventricle": "BrStem", "CSF": "CSF",
        "Left_Accumbens-area": "L.Acc", "Right_Accumbens-area": "R.Acc",
        "Left_Amygdala": "L.Amyg", "Right_Amygdala": "R.Amyg",
        "Left_Caudate": "L.Cau", "Right_Caudate": "R.Cau",
        "Left_Hippocampus": "L.Hipp", "Right_Hippocampus": "R.Hipp",
        "Left_Pallidum": "L.Pal", "Right_Pallidum": "R.Pal",
        "Left_Putamen": "L.Put", "Right_Putamen": "R.Put",
        "Left_Thalamus_Proper": "L.Thal", "Right_Thalamus-Proper": "R.Thal",
    }
    return m.get(folder, folder)


# ── F1: Effector summary table (top-1 PoPS per locus × region, with method shading) ──
def make_effector_table(hc: pd.DataFrame, fig_dir: Path, top_n: int = 25):
    """For the top-N (locus, region) entries by Tier-1 PoPS_Score, draw a table
    with columns: locus, region, top-1 PoPS gene, PoPS score, then a 4-column shaded
    grid (Closest / Positional / eQTL / ciMap) showing agreement.
    """
    # Restrict to Tier 1 (top-1 PoPS ∩ closest gene) for the headline table; if too few,
    # add Tier 2 to fill out to top_n
    t1 = hc[hc["tier"] == 1].sort_values("pops_score", ascending=False)
    if len(t1) < top_n:
        t2 = hc[hc["tier"] == 2].sort_values("pops_score", ascending=False)
        rows_df = pd.concat([t1, t2.head(top_n - len(t1))], ignore_index=True)
    else:
        rows_df = t1.head(top_n).reset_index(drop=True)

    n = len(rows_df)
    fig_h = max(80, 5 * (n + 4))  # mm
    fig, ax = plt.subplots(figsize=(MM(180), MM(fig_h)))
    ax.axis("off")

    col_x = {
        "locus":   0.02,
        "region":  0.10,
        "gene":    0.22,
        "pops":    0.34,
        "Closest":     0.46,
        "Positional":  0.58,
        "eQTL":        0.70,
        "ciMap":       0.82,
        "tier":    0.94,
    }
    row_top = 0.97
    row_dy = 0.92 / (n + 2)

    # Header row
    headers = list(col_x.keys())
    for h in headers:
        ax.text(col_x[h], row_top, h, fontsize=7, fontweight="bold",
                ha="left", va="bottom", transform=ax.transAxes)
    ax.plot([0.02, 0.98], [row_top - 0.005, row_top - 0.005],
            color="black", linewidth=0.6, transform=ax.transAxes, clip_on=False)

    method_color = {"Closest": "#B2182B", "Positional": "#2166AC",
                    "eQTL": "#67A9CF", "ciMap": "#F4A582"}
    for i, row in rows_df.iterrows():
        y = row_top - 0.02 - (i + 1) * row_dy
        ax.text(col_x["locus"], y, row["al_id"], fontsize=6.5,
                ha="left", va="center", transform=ax.transAxes)
        ax.text(col_x["region"], y, _region_short(row["region"]),
                fontsize=6.5, ha="left", va="center", transform=ax.transAxes)
        ax.text(col_x["gene"], y, str(row["symbol"]), fontsize=6.5,
                ha="left", va="center", fontstyle="italic", transform=ax.transAxes)
        ax.text(col_x["pops"], y, f"{row['pops_score']:.2f}", fontsize=6.5,
                ha="left", va="center", transform=ax.transAxes)
        for m_key, m_label in zip(METHOD_ORDER, ["Closest", "Positional", "eQTL", "ciMap"]):
            x = col_x[m_label]
            agreed = int(row[m_key]) == 1
            color = method_color[m_label] if agreed else "white"
            rect = mpatches.Rectangle(
                (x + 0.005, y - row_dy * 0.35), 0.06, row_dy * 0.7,
                facecolor=color, edgecolor="#404040", linewidth=0.4,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
        tier_int = int(row["tier"])
        tier_str = {1: "1 ◆◆◆", 2: "2 ◆◆", 3: "3 ◆"}.get(tier_int, str(tier_int))
        ax.text(col_x["tier"], y, tier_str, fontsize=6.5,
                ha="left", va="center", transform=ax.transAxes)

    ax.set_title(
        "PoPS + local effector-gene nominations (top-1 PoPS gene per locus × region; "
        "shaded boxes indicate agreement with FUMA locus-based methods). "
        "Tiers per Weeks 2023.",
        loc="left", fontweight="bold", fontsize=8.5, pad=12,
    )
    out = fig_dir / "pops_tier1_effector_table"
    save_mpl(fig, out)
    print(f"Wrote: {out}.pdf / .png  ({n} rows)")
    plt.close(fig)


# ── F2: Method agreement matrix (Weeks Fig 4b analogue) ──
def make_method_agreement_matrix(hc: pd.DataFrame, fig_dir: Path):
    """5×5 matrix: PoPS + 4 FUMA locus-based methods. Cell = proportion of loci
    where both methods nominate the same top-1 gene.

    For each locus × region:
      - top-1 PoPS gene = pops_rank == 1
      - closest gene    = is_closest_gene == 1 (FUMA leadSNP-nearest)
      - positional      = in_positional == 1 (FUMA posMapSNPs > 0)
      - eqtl            = in_eqtl == 1 (FUMA eqtlMapSNPs > 0)
      - ciMap           = in_ciMap == 1 (FUMA ciMap == 'Yes')

    "Agreement" for a pair (M1, M2) = at this locus×region, both methods nominate
    the *same single* gene (or any single overlapping gene if a method nominates more
    than one).
    """
    methods = ["PoPS", "Closest", "Positional", "eQTL", "ciMap"]

    def _gene_set_for_method(sub: pd.DataFrame, method: str):
        if method == "PoPS":
            return set(sub.loc[sub["pops_rank"] == 1, "ensg"])
        col = {"Closest": "is_closest_gene", "Positional": "in_positional",
               "eQTL": "in_eqtl", "ciMap": "in_ciMap"}[method]
        return set(sub.loc[sub[col] == 1, "ensg"])

    pair_groups = hc.groupby(["al_id", "region"])
    # Build per-locus-region method → gene-set mapping once
    locus_methods = {}
    for key, sub in pair_groups:
        locus_methods[key] = {m: _gene_set_for_method(sub, m) for m in methods}

    M = np.full((len(methods), len(methods)), np.nan)
    Nmat = np.zeros((len(methods), len(methods)), dtype=int)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            agree = 0; both = 0
            for key, mset in locus_methods.items():
                if mset[m1] and mset[m2]:
                    both += 1
                    if mset[m1] & mset[m2]:
                        agree += 1
            M[i, j] = (agree / both) if both else np.nan
            Nmat[i, j] = both

    fig, ax = plt.subplots(figsize=(MM(100), MM(95)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlBu_r")
    ax.set_xticks(range(len(methods))); ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}\n(n={Nmat[i,j]})", ha="center", va="center",
                        fontsize=6.5, color="white" if (v > 0.5 or v < 0.2) else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prop. loci-regions both methods nominate same gene", fontsize=7)
    ax.set_title("Cross-method top-1 gene agreement\n(PoPS vs FUMA locus-based methods, all 16 regions pooled)",
                 loc="left", fontweight="bold", fontsize=8.5, pad=10)
    out = fig_dir / "pops_method_agreement"
    save_mpl(fig, out)
    print(f"Wrote: {out}.pdf / .png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hc-csv", type=Path, default=HC_CSV)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--top-n-effectors", type=int, default=25)
    args = p.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    apply_mpl_style()
    hc = pd.read_csv(args.hc_csv)
    print(f"Loaded: {len(hc)} rows; tiers = {hc['tier'].value_counts().sort_index().to_dict()}")

    make_effector_table(hc, args.fig_dir, top_n=args.top_n_effectors)
    make_method_agreement_matrix(hc, args.fig_dir)


if __name__ == "__main__":
    main()
