#!/usr/bin/env python3
"""
36_age_sex_scatter_supp.py
Render the per-region age MAE × sex accuracy scatter (former Fig 2b) as
Supplementary Fig. S47 from the existing 10-fold cross-validation CSV
produced by the embedding-visualization notebook.

This script does NOT recompute the regression — it consumes the existing CSV
`discovery_local_scatter_accu_r2_with_hue_df.csv` and renders a labelled scatter
in NComms style (Arial, 7-pt tick, 8-pt label, 9-pt title).

Output: figures/supplementary/S47_age_sex_predictability/S47_age_sex_predictability.{pdf,png}
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

POST_BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = (
    POST_BASE / "results" / "phesant"
    / "age_sex_with_icv_cov_per_region.csv"
)
OUT_DIR_DEFAULT = (POST_BASE / "figures"
                   / "supplementary" / "S47_age_sex_predictability")

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


SHORT_NAME_MAP = {
    "BrainStem": "Brainstem", "BrainStem_or_4thV": "Brainstem",
    "Ventricle": "Brainstem",
    "Brain_Stem_or_4th_Ventricle": "Brainstem",
    "CSF": "CSF",
    "LAccumbens": "L.Accumbens", "RAccumbens": "R.Accumbens",
    "Left_Accumbens-area": "L.Accumbens", "Right_Accumbens-area": "R.Accumbens",
    "LAmygdala": "L.Amygdala", "RAmygdala": "R.Amygdala",
    "Left_Amygdala": "L.Amygdala", "Right_Amygdala": "R.Amygdala",
    "LCaudate": "L.Caudate", "RCaudate": "R.Caudate",
    "Left_Caudate": "L.Caudate", "Right_Caudate": "R.Caudate",
    "LHippocampus": "L.Hippocampus", "RHippocampus": "R.Hippocampus",
    "Left_Hippocampus": "L.Hippocampus", "Right_Hippocampus": "R.Hippocampus",
    "LPallidum": "L.Pallidum", "RPallidum": "R.Pallidum",
    "Left_Pallidum": "L.Pallidum", "Right_Pallidum": "R.Pallidum",
    "LPutamen": "L.Putamen", "RPutamen": "R.Putamen",
    "Left_Putamen": "L.Putamen", "Right_Putamen": "R.Putamen",
    "LThalamus": "L.Thalamus", "RThalamus": "R.Thalamus",
    "Left_Thalamus_Proper": "L.Thalamus", "Right_Thalamus-Proper": "R.Thalamus",
}


def _accuracy_to_float(x):
    """Convert '[0.7677...]'-style strings to float."""
    if isinstance(x, str):
        return float(x.strip("[]"))
    return float(x)


def render(csv: Path, out_dir: Path):
    df = pd.read_csv(csv)
    # Detect schema: new ICV-adjusted CSV has columns
    #   region, age_mae_yr_with_icv, sex_accuracy_with_icv, ...
    # Old notebook CSV has:
    #   regions, mae_values, accuracies, region_volume
    if "age_mae_yr_with_icv" in df.columns:
        # New ICV-adjusted format (Script 38)
        df["accuracy"] = df["sex_accuracy_with_icv"].astype(float)
        df["MAE"] = df["age_mae_yr_with_icv"].astype(float)
        df["short"] = df["region"].map(SHORT_NAME_MAP).fillna(df["region"])
    else:
        # Legacy notebook format
        df["accuracy"] = df["accuracies"].apply(_accuracy_to_float)
        df["MAE"] = pd.to_numeric(df["mae_values"], errors="coerce")
        df["short"] = df["regions"].map(SHORT_NAME_MAP).fillna(df["regions"])

    apply_mpl_style()

    # Fixed anatomical region order — same y-axis labels in both panels so a reader
    # can scan one region horizontally from MAE → sex accuracy without re-locating.
    # Matches REGION_ORDER used in Script 35 (Fig 2b/c) for cross-figure consistency.
    REGION_DISPLAY_ORDER = [
        "Brainstem", "CSF",
        "L.Thalamus", "R.Thalamus",
        "L.Hippocampus", "R.Hippocampus",
        "L.Amygdala", "R.Amygdala",
        "L.Caudate", "R.Caudate",
        "L.Putamen", "R.Putamen",
        "L.Pallidum", "R.Pallidum",
        "L.Accumbens", "R.Accumbens",
    ]
    # Index by short name for direct lookup
    df = df.set_index("short")
    # Reindex into fixed display order (rows for any missing region come in as NaN)
    df = df.reindex(REGION_DISPLAY_ORDER)
    df = df.dropna(subset=["MAE", "accuracy"])

    def _color(short: str) -> str:
        if short in {"CSF", "Brainstem"}:
            return "#7F7F7F"
        return "#2166AC" if short.startswith("L.") else "#B2182B"

    colors = [_color(s) for s in df.index]
    y_pos = np.arange(len(df))

    # Two horizontal bar charts; same region order on y-axis so labels align.
    # Wider figure to avoid value-label overlap on tight x-scale.
    fig, (ax_mae, ax_sex) = plt.subplots(
        1, 2, figsize=(MM(190), MM(130)),
        gridspec_kw={"wspace": 0.05},
        sharey=True,
    )

    # Best-of values for annotation
    best_mae_short = df["MAE"].idxmin()
    best_mae_val = df["MAE"].min()
    best_acc_short = df["accuracy"].idxmax()
    best_acc_val = df["accuracy"].max()

    # ── Panel a — age MAE bars (lower = better) ──
    bars_mae = ax_mae.barh(y_pos, df["MAE"], color=colors,
                           edgecolor="white", linewidth=0.5)
    # Highlight best
    for i, short in enumerate(df.index):
        if short == best_mae_short:
            bars_mae[i].set_edgecolor("black")
            bars_mae[i].set_linewidth(1.4)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels(df.index, fontsize=7.5)
    ax_mae.invert_yaxis()
    mae_min, mae_max = df["MAE"].min(), df["MAE"].max()
    ax_mae.set_xlim(mae_min - 0.01, mae_max + 0.05)
    ax_mae.set_xlabel("Age prediction MAE (years)", fontsize=8)
    ax_mae.set_title("(a) Age prediction — lower MAE is better",
                     loc="left", fontweight="bold", fontsize=9, pad=6)
    for i, v in enumerate(df["MAE"]):
        ax_mae.text(v + 0.002, i, f"{v:.3f}",
                    va="center", fontsize=6.5, color="#222")
    # Place "← best" annotation with a clear horizontal gap past the value text.
    # Value text "X.XXX" at 6.5pt occupies ~0.020 units on this x-scale; place
    # arrow at best_val + 0.050 so it sits well beyond the value text.
    ax_mae.text(best_mae_val + 0.050, df.index.get_loc(best_mae_short),
                f"← best",
                va="center", fontsize=6.5, color="black", fontweight="bold")
    ax_mae.spines["top"].set_visible(False)
    ax_mae.spines["right"].set_visible(False)

    # ── Panel b — sex accuracy bars (higher = better) ──
    bars_acc = ax_sex.barh(y_pos, df["accuracy"], color=colors,
                           edgecolor="white", linewidth=0.5)
    for i, short in enumerate(df.index):
        if short == best_acc_short:
            bars_acc[i].set_edgecolor("black")
            bars_acc[i].set_linewidth(1.4)
            bars_acc[i].set_facecolor("#000000")  # CSF outlier in black
    ax_sex.set_xlim(0.70, 0.93)
    ax_sex.set_xlabel("Sex classification accuracy", fontsize=8)
    ax_sex.set_title("(b) Sex prediction — higher accuracy is better",
                     loc="left", fontweight="bold", fontsize=9, pad=6)
    for i, v in enumerate(df["accuracy"]):
        ax_sex.text(v + 0.003, i, f"{v:.3f}",
                    va="center", fontsize=6.5, color="#222")
    # Place "best" label just after the value text of the best bar
    ax_sex.text(best_acc_val + 0.022, df.index.get_loc(best_acc_short),
                f"← best",
                va="center", fontsize=6.5, color="black", fontweight="bold")
    ax_sex.spines["top"].set_visible(False)
    ax_sex.spines["right"].set_visible(False)
    ax_sex.tick_params(axis="y", left=False, labelleft=False)

    legend_handles = [
        mpatches.Patch(color="#2166AC", label="Left hemisphere"),
        mpatches.Patch(color="#B2182B", label="Right hemisphere"),
        mpatches.Patch(color="#7F7F7F", label="Midline (Brainstem, CSF)"),
        mpatches.Patch(facecolor="#000000", edgecolor="black",
                       label="Bold border + black fill = best on this metric"),
    ]
    fig.legend(handles=legend_handles, fontsize=6.5, frameon=False,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))

    plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / "S47_age_sex_predictability"
    save_mpl(fig, out_stem)
    plt.close(fig)
    print(f"Wrote: {out_stem}.pdf / .png  (N={len(df)} regions)")
    print(f"  Best MAE: {best_mae_short} = {best_mae_val:.3f} yr")
    print(f"  Best sex: {best_acc_short} = {best_acc_val:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = p.parse_args()
    render(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
