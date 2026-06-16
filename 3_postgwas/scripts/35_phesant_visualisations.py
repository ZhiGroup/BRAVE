#!/usr/bin/env python3
"""
35_phesant_visualisations.py
Two PHESANT panels to replace current Fig 2 b/c, plus a supplementary
"old age/sex predictability" panel (S47) that displaces the original Fig 2b.

Panels produced:
  (b) Per-region brain-IDP (UKB 25xxx) FDR-sig hit count — positive control
       showing the BRE composite recovers FreeSurfer + dMRI + fMRI phenome
       comprehensively across all 16 regions.

  (c) Region × phenotype-category FDR-sig count heatmap — downstream
       phenome-wide relevance, split by UKB category (body composition,
       cardio/PWV, cognitive/mental health, anthropometric, body/impedance,
       ICD-10, lifestyle, etc.). Demonstrates region-specific phenotypic
       association patterns mirroring §3 region-specificity.

Outputs:
  figures/phesant/fig2b_brain_idp_bar.{pdf,png}
  figures/phesant/fig2c_nonbrain_category_heatmap.{pdf,png}
  results/phesant/aggregate_per_region.csv
  results/phesant/aggregate_per_category.csv
  results/phesant/aggregate_top_hits.csv (top 5 non-brain hits per region)

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
PHE_DIR = POST_BASE / "results" / "phesant"
FIG_DIR = POST_BASE / "figures" / "phesant"

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


# UKB Data-Field coarse category map (numeric ranges).
# 25xxx is brain-imaging IDP (positive control, separated out).
CATEGORY_RANGES = [
    (25000, 25999, "Brain imaging (IDP)"),
    (23000, 23999, "Body composition"),
    (12500, 12999, "Cardiovascular (PWV)"),
    (20000, 21999, "Cognitive / mental health"),
    (22000, 22999, "Body / impedance / activity"),
    (40000, 41999, "ICD-10 / hospital"),
    (30000, 30999, "Blood assay"),
    (6000, 6999, "Lifestyle / activity"),
    (1000, 1999, "Lifestyle / activity"),
    (2000, 2999, "Health history"),
    (3000, 3999, "Medical history"),
    (4000, 4999, "Reproduction / hormonal"),
    (12000, 12499, "Other physical"),
    (10, 99, "Anthropometric / vital"),
    (100, 999, "Demographics / vital"),
    (26000, 26999, "Eye / OCT"),
    (27000, 27999, "Heart MRI"),
]


def _category(field_code: str) -> str:
    try:
        n = int(str(field_code).split("-")[0])
    except (ValueError, AttributeError):
        return "Other"
    for lo, hi, name in CATEGORY_RANGES:
        if lo <= n <= hi:
            return name
    return "Other"


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


REGION_ORDER = [
    "Brain_Stem_or_4th_Ventricle", "CSF",
    "Left_Thalamus_Proper", "Right_Thalamus-Proper",
    "Left_Hippocampus", "Right_Hippocampus",
    "Left_Amygdala", "Right_Amygdala",
    "Left_Caudate", "Right_Caudate",
    "Left_Putamen", "Right_Putamen",
    "Left_Pallidum", "Right_Pallidum",
    "Left_Accumbens-area", "Right_Accumbens-area",
]


def load_results(phe_dir: Path):
    rows = []
    for d in sorted(phe_dir.iterdir()):
        if not d.is_dir():
            continue
        f = d / "phesant_results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["region"] = d.name
        df["category"] = df["field"].apply(_category)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_panel_b_brain_idp_bar(df_all: pd.DataFrame, ax):
    """Panel B: per-region brain-IDP FDR-sig count bar."""
    sig = df_all[df_all["q_fdr"] < 0.05].copy()
    is_idp = sig["category"] == "Brain imaging (IDP)"
    counts = (sig[is_idp].groupby("region").size()
              .reindex(REGION_ORDER, fill_value=0))
    counts.index = [_region_short(r) for r in counts.index]

    colors = []
    for r in REGION_ORDER:
        if r.startswith("Left_"):
            colors.append("#2166AC")
        elif r.startswith("Right_"):
            colors.append("#B2182B")
        else:
            colors.append("#7F7F7F")
    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("FDR-sig brain-imaging fields (q < 0.05)", fontsize=7.5)
    ax.set_title("Brain-imaging phenome (UKB 25xxx)",
                 loc="left", fontweight="bold", fontsize=9, pad=4)
    # Add count labels above bars
    for i, (bar, v) in enumerate(zip(bars, counts.values)):
        ax.text(i, v + max(counts.values) * 0.01, str(int(v)),
                ha="center", va="bottom", fontsize=5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts.values) * 1.12)
    legend_handles = [
        mpatches.Patch(color="#2166AC", label="Left hem."),
        mpatches.Patch(color="#B2182B", label="Right hem."),
        mpatches.Patch(color="#7F7F7F", label="Midline"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, frameon=False, loc="upper right")
    return counts


def make_panel_c_nonbrain_heatmap(df_all: pd.DataFrame, ax, top_n_categories=8):
    """Panel C: region × non-brain phenotype-category FDR-sig count heatmap."""
    sig = df_all[df_all["q_fdr"] < 0.05].copy()
    nb = sig[sig["category"] != "Brain imaging (IDP)"]
    mat = (nb.groupby(["region", "category"]).size()
           .unstack(fill_value=0))
    # Reorder rows to manuscript order
    mat = mat.reindex(REGION_ORDER, fill_value=0)
    # Pick top-N categories by total
    totals = mat.sum(axis=0).sort_values(ascending=False)
    keep = totals.head(top_n_categories).index.tolist()
    mat = mat[keep]

    # Plot heatmap
    im = ax.imshow(mat.values, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(REGION_ORDER)))
    ax.set_yticklabels([_region_short(r) for r in REGION_ORDER], fontsize=6.5)
    ax.set_title("Non-brain phenome (region × category)",
                 loc="left", fontweight="bold", fontsize=9, pad=4)

    # Annotate counts
    vmax = mat.values.max()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat.values[i, j])
            if v == 0:
                continue
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=5.5, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("# FDR-sig (q < 0.05)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)
    return mat


def make_supp_panel_old_b(out_dir: Path):
    """Supplementary version of OLD Fig 2b (age/sex predictability).
    Reads from results/per_region/regression_metrics.csv if available, otherwise
    skip with a warning (old plotting code lives elsewhere)."""
    # Placeholder: link to existing figure if it already exists
    existing = PROJECT_BASE / "post_gwas_analysis" / "figures" / "per_region" / "regression_metrics_summary.png"
    if existing.exists():
        print(f"  Old Fig 2b source already at {existing}; will be referenced as Supp Fig S47")
    else:
        print(f"  Old Fig 2b source not found at {existing} — needs manual relocation")


def write_aggregates(df_all: pd.DataFrame, out_dir: Path):
    """Write 3 aggregate CSVs for supp table + Methods support."""
    sig = df_all[df_all["q_fdr"] < 0.05].copy()
    # Per-region counts (brain-IDP and non-brain)
    is_idp = sig["category"] == "Brain imaging (IDP)"
    per_region = pd.DataFrame({
        "region": REGION_ORDER,
        "n_total_tests": [(df_all["region"] == r).sum() for r in REGION_ORDER],
        "n_sig_total":   [(sig["region"] == r).sum() for r in REGION_ORDER],
        "n_sig_idp":     [((sig["region"] == r) & is_idp).sum() for r in REGION_ORDER],
        "n_sig_nonbrain": [((sig["region"] == r) & ~is_idp).sum() for r in REGION_ORDER],
    })
    per_region.to_csv(out_dir / "aggregate_per_region.csv", index=False)

    # Per-region per-category breakdown
    cat_mat = (sig.groupby(["region", "category"]).size().unstack(fill_value=0))
    cat_mat = cat_mat.reindex(REGION_ORDER, fill_value=0)
    cat_mat.to_csv(out_dir / "aggregate_per_category.csv")

    # Top 5 non-brain hits per region
    nb_top = []
    for r in REGION_ORDER:
        sub = df_all[(df_all["region"] == r)
                     & (df_all["category"] != "Brain imaging (IDP)")
                     & (df_all["q_fdr"] < 0.05)].sort_values("p").head(5)
        sub["rank_in_region"] = range(1, len(sub) + 1)
        nb_top.append(sub)
    top_df = pd.concat(nb_top, ignore_index=True)
    cols = ["region", "rank_in_region", "field", "category", "type", "N",
            "r", "beta", "se", "p", "q_fdr", "OR"]
    top_df[cols].to_csv(out_dir / "aggregate_top_nonbrain_hits.csv", index=False)

    print(f"  per-region summary  -> aggregate_per_region.csv")
    print(f"  per-category matrix -> aggregate_per_category.csv")
    print(f"  top non-brain hits  -> aggregate_top_nonbrain_hits.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phe-dir", type=Path, default=PHE_DIR)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    p.add_argument("--top-n-categories", type=int, default=8)
    args = p.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    apply_mpl_style()
    df_all = load_results(args.phe_dir)
    print(f"Loaded {len(df_all):,} total tests from {df_all['region'].nunique()} regions")

    # Panel B: brain-IDP bar
    fig, ax = plt.subplots(figsize=(MM(95), MM(70)))
    make_panel_b_brain_idp_bar(df_all, ax)
    out_b = args.fig_dir / "fig2b_brain_idp_bar"
    save_mpl(fig, out_b)
    print(f"Wrote: {out_b}.pdf / .png")
    plt.close(fig)

    # Panel C: non-brain category heatmap
    fig, ax = plt.subplots(figsize=(MM(95), MM(95)))
    make_panel_c_nonbrain_heatmap(df_all, ax, top_n_categories=args.top_n_categories)
    out_c = args.fig_dir / "fig2c_nonbrain_category_heatmap"
    save_mpl(fig, out_c)
    print(f"Wrote: {out_c}.pdf / .png")
    plt.close(fig)

    # Aggregate CSVs
    write_aggregates(df_all, args.phe_dir)


if __name__ == "__main__":
    main()
