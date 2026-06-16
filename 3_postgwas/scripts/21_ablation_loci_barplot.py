"""
21_ablation_loci_barplot.py
----------------------------
Supplementary Fig. S5: locus yield by embedding training strategy.

Three bars:
  Untrained model    →   0 aggregate loci
  UDIP-Voxel         →  33 aggregate loci
  Contrastive RASVE  → 276 aggregate loci (JAGWAS)

Usage
-----
python 21_ablation_loci_barplot.py \
    --out-dir ../figures/ablation
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_LABEL, FS_TICK, FS_TITLE


# ── Data (fixed — from ablation analyses, all using FastGWA minP) ────────────
# Untrained: 0 loci  (randomly initialized weights, no training)
# UDIP-Voxel: 33 loci (reconstruction autoencoder, no contrastive objective)
# Contrastive RASVE: 60 loci (our model, FastGWA univariate minP — fair comparison)
# Note: JAGWAS on RASVE embeddings yields 276 loci (shown separately in §2)
MODELS = ["Untrained\nmodel", "UDIP-Voxel\n(reconstruction)", "Contrastive\nBREs\n(FastGWA minP)"]
LOCI   = [0, 33, 60]
COLORS = ["#BBBBBB", "#88AABB", "#2166AC"]   # grey → mid-blue → dark-blue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation loci bar chart (Supp Fig S5)")
    p.add_argument("--out-dir", default="../figures/ablation",
                   help="Output directory for PDF + PNG (default: ../figures/ablation)")
    return p.parse_args()


def make_figure(out_dir: Path) -> None:
    apply_mpl_style()

    fig, ax = plt.subplots(figsize=(MM(88), MM(70)))

    bars = ax.bar(MODELS, LOCI, color=COLORS, width=0.55,
                  edgecolor="white", linewidth=0.5)

    # Value labels above bars
    for bar, val in zip(bars, LOCI):
        y_pos = val + 4 if val > 0 else 4
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                str(val), ha="center", va="bottom",
                fontsize=FS_LABEL, fontweight="bold", color="#222222")

    # ×-fold annotation between bar 2 and bar 3
    x2 = bars[1].get_x() + bars[1].get_width() / 2
    x3 = bars[2].get_x() + bars[2].get_width() / 2
    y_ann = 52
    ax.annotate("", xy=(x3, y_ann), xytext=(x2, y_ann),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    ax.text((x2 + x3) / 2, y_ann + 1, "×1.8",
            ha="center", va="bottom", fontsize=FS_TICK, color="#444444")

    # Note about JAGWAS
    ax.text(bars[2].get_x() + bars[2].get_width() / 2, 63,
            "JAGWAS: 276 ↑", ha="center", va="bottom",
            fontsize=FS_TICK - 1, color="#2166AC", fontstyle="italic")

    # Axes
    ax.set_ylim(0, 80)
    ax.set_ylabel("Aggregate loci", fontsize=FS_LABEL)
    ax.set_title("Locus yield by embedding training strategy",
                 fontsize=FS_TITLE, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)  # no x-axis ticks
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    # Legend note
    ax.text(0.98, 0.04,
            "All bars: same 128-dim BREs, FastGWA univariate minP",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_TICK - 1, color="#666666", style="italic")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_mpl(fig, out_dir / "ablation_loci_barplot")
    plt.close(fig)
    print(f"Saved: {out_dir}/ablation_loci_barplot.pdf / .png")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    make_figure(out_dir)


if __name__ == "__main__":
    main()
