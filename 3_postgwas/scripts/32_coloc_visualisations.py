#!/usr/bin/env python3
"""
32_coloc_visualisations.py
Three-panel supplementary figure summarising focused coloc results from Script 31.

Panel (a): stacked bar per pair — locus counts by coloc_class
            (strong shared, suggestive shared, distinct variants, indeterminate, no signal)
Panel (b): strip plot per pair — PP.H4 distribution with thresholds at 0.5 and 0.8
Panel (c): mirror-Manhattan zoom of the best-coloc locus
            (whichever locus×pair has the highest PP.H4 across all 369 tests)

Output: figures/coloc/coloc_focused_summary.{pdf,png} (Panels a–b)
        figures/coloc/coloc_focused_mirror_<pair>_<al_id>.{pdf,png} (Panel c)

Compute env: Python 3.7
Style: fig_style.apply_mpl_style (Arial 7-pt tick / 8-pt label / 9-pt title)
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

# Project paths
POST_BASE = Path(__file__).resolve().parents[1]
COLOC_RESULTS = POST_BASE / "results/coloc/coloc_focused_per_pair.csv"
FIG_DIR = POST_BASE / "figures/coloc"
BRE_SUMSTATS_BASE = Path(str(cfg.gwas.fastgwa_out))
OCT_BASE = Path(str(cfg.postgwas.oct_sumstats_dir))
HEART_BASE = Path(str(cfg.postgwas.heart_sumstats_dir))

# Ensure fig_style import works
sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


# ── Classification thresholds (match Script 31) ──────────────────────────────
def classify_pp(pp_h4, pp_h3, n_snps):
    if pd.isna(pp_h4):
        return "insufficient SNPs"
    if pp_h4 > 0.8:
        return "strong shared causal"
    if pp_h4 >= 0.5:
        return "suggestive shared"
    if pp_h3 > 0.5:
        return "distinct variants"
    return "no clear signal"


CLASS_ORDER = ["strong shared causal", "suggestive shared",
               "distinct variants", "no clear signal", "insufficient SNPs"]
CLASS_COLORS = {
    "strong shared causal": "#B2182B",
    "suggestive shared":    "#EF8A62",
    "distinct variants":    "#2166AC",
    "no clear signal":      "#BBBBBB",
    "insufficient SNPs":    "#EEEEEE",
}


def _pair_short(pair_label: str) -> str:
    """RThalamus_GCIPL_thickness_left → R.Thal × GCIPL-L."""
    region, _, trait = pair_label.partition("_")
    region_map = {
        "RThalamus": "R.Thal", "LThalamus": "L.Thal",
        "RHippocampus": "R.Hipp", "LHippocampus": "L.Hipp",
        "RAmygdala": "R.Amyg",
    }
    region_short = region_map.get(region, region)
    trait_map = {
        "GCIPL_thickness_left": "GCIPL-L", "GCIPL_thickness_right": "GCIPL-R",
        "INL_thickness_left": "INL-L", "INL_thickness_right": "INL-R",
        "LAEF": "LAEF", "LAV_min": "LAV-min",
    }
    trait_short = trait_map.get(trait, trait)
    return f"{region_short} × {trait_short}"


def _ext_sumstats_path_from_row(modality: str, trait: str) -> Path:
    """Resolve an external-trait fastGWA file given the trait label."""
    if modality == "oct":
        meta = pd.read_excel(OCT_BASE / "ID_OCT.xlsx", header=1).dropna(subset=["File_name"])
        row = meta[meta["ID"].astype(str).str.strip() == trait]
        if row.empty:
            return None
        folder = OCT_BASE / str(row.iloc[0]["File_name"]).strip()
        files = sorted(folder.glob("*.fastGWA"))
        return files[0] if files else None
    elif modality == "heart":
        heart_map = {"LAEF": "31", "LAV_min": "29"}
        pid = heart_map.get(trait)
        if pid is None:
            return None
        return HEART_BASE / f"ukbiobank_heart_pheno{pid}_may2022" / f"ukb_phase1to3_heart_may_2022_pheno{pid}.fastGWA"
    return None


def _load_locus_snps(fastgwa_path: Path, chr_: int, start: int, end: int) -> pd.DataFrame:
    """Read a fastGWA file and return SNPs in [chr:start-end]."""
    cols = ["CHR", "SNP", "POS", "P"]
    out = []
    for chunk in pd.read_csv(fastgwa_path, sep="\t", usecols=cols,
                             chunksize=200_000,
                             dtype={"CHR": str, "SNP": str, "POS": "Int32", "P": "float64"}):
        mask = (chunk["CHR"] == str(chr_)) & (chunk["POS"] >= start) & (chunk["POS"] <= end)
        if mask.any():
            out.append(chunk[mask])
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)


# ── Panel A: stacked bar of locus counts per pair ────────────────────────────
def make_panel_a(df: pd.DataFrame, ax):
    pair_class = df.groupby(["pair", "coloc_class"]).size().unstack(fill_value=0)
    pair_class = pair_class.reindex(columns=CLASS_ORDER, fill_value=0)
    pair_class.index = [_pair_short(p) for p in pair_class.index]
    pair_class = pair_class.sort_index()
    bottom = np.zeros(len(pair_class))
    for c in CLASS_ORDER:
        ax.bar(pair_class.index, pair_class[c], bottom=bottom,
               color=CLASS_COLORS[c], edgecolor="white", linewidth=0.4, label=c)
        bottom += pair_class[c].values
    ax.set_ylabel("# loci tested")
    ax.set_title("(a) Per-pair locus classification", loc="left", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=3, frameon=False, fontsize=6)


# ── Panel B: strip plot of PP.H4 per pair with threshold lines ───────────────
def make_panel_b(df: pd.DataFrame, ax):
    plot_df = df.dropna(subset=["PP_H4"]).copy()
    pair_order = sorted(plot_df["pair"].unique(), key=_pair_short)
    pair_short = [_pair_short(p) for p in pair_order]
    rng = np.random.RandomState(0)
    for i, p in enumerate(pair_order):
        vals = plot_df.loc[plot_df["pair"] == p, "PP_H4"].values
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=4.5, color="#404040", alpha=0.55, edgecolor="none")
    ax.axhline(0.8, color="#B2182B", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(0.5, color="#EF8A62", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(len(pair_order) - 0.5, 0.82, "PP.H4 = 0.8 (strong)",
            color="#B2182B", fontsize=6, ha="right", va="bottom")
    ax.text(len(pair_order) - 0.5, 0.52, "PP.H4 = 0.5 (suggestive)",
            color="#EF8A62", fontsize=6, ha="right", va="bottom")
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_short, rotation=35, ha="right")
    ax.set_ylabel("PP.H4 (shared causal variant)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) PP.H4 distribution per pair", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel C: mirror-Manhattan zoom of best-coloc locus ───────────────────────
def make_panel_c(df: pd.DataFrame, fig_dir: Path):
    """Locate the row with highest PP.H4 + ≥50 SNPs, then mirror plot both signals."""
    best = df.dropna(subset=["PP_H4"]).query("N_SNPS >= 50").sort_values("PP_H4", ascending=False).iloc[0]
    pair = best["pair"]
    region = best["region"]
    modality = best["modality"]
    trait = best["trait"]
    top_dim = int(best["top_dim"])
    al_id = best["al_id"]
    chr_, start, end = int(best["chr"]), int(best["start"]), int(best["end"])

    bre_path = (BRE_SUMSTATS_BASE /
                f"discovery_{region}_QT{top_dim}.fastGWA.fastGWA")
    ext_path = _ext_sumstats_path_from_row(modality, trait)
    if bre_path is None or not bre_path.exists() or ext_path is None or not ext_path.exists():
        print(f"  Mirror plot skipped: paths missing\n    bre={bre_path}\n    ext={ext_path}")
        return None

    print(f"  Loading mirror data for {pair} {al_id} chr{chr_}:{start:,}-{end:,}")
    bre = _load_locus_snps(bre_path, chr_, start, end)
    ext = _load_locus_snps(ext_path, chr_, start, end)
    print(f"    BRE SNPs: {len(bre):,}, ext SNPs: {len(ext):,}")

    fig, axs = plt.subplots(2, 1, figsize=(MM(180), MM(95)),
                            sharex=True, gridspec_kw={"hspace": 0.05})
    pair_short = _pair_short(pair)
    region_short, _, trait_short = pair_short.partition(" × ")
    bre_neglog = -np.log10(bre["P"].clip(lower=1e-300))
    ext_neglog = -np.log10(ext["P"].clip(lower=1e-300))
    axs[0].scatter(bre["POS"] / 1e6, bre_neglog, s=4, color="#2166AC", alpha=0.6, edgecolor="none")
    axs[0].set_ylabel(f"{region_short} dim {top_dim}\n−log₁₀(P)")
    axs[0].axhline(-np.log10(5e-8), linestyle="--", color="grey", linewidth=0.6)
    axs[0].set_title(
        f"(c) Best-PP.H4 locus: {pair_short}, {al_id}, chr{chr_}:{start//1_000_000:.1f}–{end//1_000_000:.1f} Mb "
        f"(PP.H4 = {best['PP_H4']:.3f}, PP.H3 = {best['PP_H3']:.3f}, N_SNPs = {int(best['N_SNPS'])})",
        loc="left", fontweight="bold", fontsize=8,
    )
    axs[1].scatter(ext["POS"] / 1e6, -ext_neglog, s=4, color="#B2182B", alpha=0.6, edgecolor="none")
    axs[1].set_ylabel(f"{trait_short}\n−log₁₀(P)")
    axs[1].axhline(np.log10(5e-8), linestyle="--", color="grey", linewidth=0.6)
    axs[1].set_xlabel(f"chr{chr_} position (Mb)")
    yticks = axs[1].get_yticks()
    axs[1].set_yticklabels([f"{abs(int(t))}" for t in yticks])
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = fig_dir / f"coloc_focused_mirror_{pair}_{al_id}"
    save_mpl(fig, out)
    print(f"    Wrote: {out}.pdf / .png")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coloc-csv", type=Path, default=COLOC_RESULTS)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = p.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    apply_mpl_style()
    df = pd.read_csv(args.coloc_csv)
    df["coloc_class"] = df.apply(
        lambda r: classify_pp(r["PP_H4"], r["PP_H3"], r["N_SNPS"]), axis=1)

    # Panels (a) + (b) on one figure
    fig, axes = plt.subplots(1, 2, figsize=(MM(180), MM(105)),
                             gridspec_kw={"wspace": 0.4})
    make_panel_a(df, axes[0])
    make_panel_b(df, axes[1])
    out = args.fig_dir / "coloc_focused_summary"
    save_mpl(fig, out)
    print(f"Wrote: {out}.pdf / .png")
    plt.close(fig)

    # Panel (c) on its own figure (different size needs)
    make_panel_c(df, args.fig_dir)


if __name__ == "__main__":
    main()
