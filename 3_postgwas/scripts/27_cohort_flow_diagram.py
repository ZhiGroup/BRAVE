"""
27_cohort_flow_diagram.py
Supplementary Fig. S39 — UK Biobank participant flow for brain region embedding
(BRE) training and GWAS analysis.

Static flow diagram. All numbers known a priori from JAGWAS_PAPER_FACTS.md and
methods_section1.md; no data computation required.

Inputs:
  None (all counts hard-coded from canonical FACTS).

Outputs:
  figures/supplementary/S39_cohort_flow/S39_cohort_flow.{pdf,png}

CLAIM: M1 Participants and cohort (supplementary backing).
RUN:   python 27_cohort_flow_diagram.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM

apply_mpl_style()


# ── Canonical cohort numbers (from JAGWAS_PAPER_FACTS.md + methods_section1.md) ──
N_TRAIN       = 6_130       # BRE encoder contrastive training cohort
N_DISCOVERY   = 22_878      # Discovery analysed (FastGWA log canonical)
N_REPLICATION = 12_359      # Replication analysed (FastGWA log canonical)
N_GWAS        = N_DISCOVERY + N_REPLICATION  # 35,237
N_KINSHIP_EX  = 457         # Replication participants excluded for kinship > 0.0442
N_REPL_PREEX  = N_REPLICATION + N_KINSHIP_EX  # 12,816

assert N_DISCOVERY + N_REPLICATION == N_GWAS, (
    f"discovery + replication = {N_DISCOVERY + N_REPLICATION} != GWAS total {N_GWAS}")


# ── Visual constants ──────────────────────────────────────────────────────────
FC_PRIMARY  = "#E8EEF7"   # soft slate-blue
FC_TRAINING = "#FDEEDC"   # soft peach
FC_EXCLUDED = "#F2F2F2"   # grey
EC          = "#2B3A55"
ARROW_C     = "#2B3A55"
TEXT_C      = "#1A1A1A"


def draw_box(ax, x, y, w, h, title, n_label, body, facecolor=FC_PRIMARY):
    """Three-line box: bold title at top, big N in middle, single italic body line at bottom."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=EC,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    cx = x + w / 2
    # Title (top, bold)
    ax.text(cx, y + h * 0.78, title,
            ha="center", va="center", fontsize=8, fontweight="bold", color=TEXT_C)
    # N (middle, big)
    ax.text(cx, y + h * 0.46, n_label,
            ha="center", va="center", fontsize=9, fontweight="bold", color=TEXT_C)
    # Body (bottom, small italic)
    ax.text(cx, y + h * 0.18, body,
            ha="center", va="center", fontsize=6.5, style="italic", color=TEXT_C)


def draw_arrow(ax, x0, y0, x1, y1, label=None, label_pos=None):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.8,
        color=ARROW_C,
    )
    ax.add_patch(arr)
    if label is not None:
        if label_pos is None:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.012
        else:
            mx, my = label_pos
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.5,
                color=TEXT_C)


def build_figure():
    # 180 mm × 95 mm — double column, modest height
    fig, ax = plt.subplots(figsize=(MM(180), MM(95)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Top: title bar (text only, no box) ────────────────────────────────────
    ax.text(0.50, 0.96,
            "Eligible UK Biobank participants",
            ha="center", va="center", fontsize=9, fontweight="bold", color=TEXT_C)
    ax.text(0.50, 0.905,
            "T1-weighted MRI (Data-Field 20252) · white British ancestry "
            "(Data-Fields 21000, 22006) · concordant genetic and recorded sex · "
            "consent valid as of 21 Aug 2023",
            ha="center", va="center", fontsize=6.5, style="italic", color="#444444",
            wrap=True)

    # ── Tier 2: BRE training (left) and GWAS cohort (right) ───────────────────
    # Training box (left)
    draw_box(
        ax, 0.04, 0.50, 0.30, 0.22,
        title="BRE encoder training",
        n_label=f"N = {N_TRAIN:,}",
        body="Contrastive voxel-level\nrepresentation learning",
        facecolor=FC_TRAINING,
    )
    # GWAS cohort box (right)
    draw_box(
        ax, 0.62, 0.50, 0.34, 0.22,
        title="Independent GWAS cohort",
        n_label=f"N = {N_GWAS:,}",
        body="Non-overlapping with training set",
        facecolor=FC_PRIMARY,
    )

    # Arrows from title bar → tier 2
    draw_arrow(ax, 0.45, 0.86, 0.19, 0.73,
               label=f"N = {N_TRAIN:,}", label_pos=(0.27, 0.81))
    draw_arrow(ax, 0.55, 0.86, 0.79, 0.73,
               label=f"N = {N_GWAS:,}", label_pos=(0.71, 0.81))

    # ── Tier 3: discovery + replication split (right column) ──────────────────
    # Discovery
    draw_box(
        ax, 0.40, 0.18, 0.24, 0.22,
        title="Discovery",
        n_label=f"N = {N_DISCOVERY:,}",
        body="Random fold partition",
        facecolor=FC_PRIMARY,
    )
    # Replication (analysed)
    draw_box(
        ax, 0.72, 0.18, 0.24, 0.22,
        title="Replication (analysed)",
        n_label=f"N = {N_REPLICATION:,}",
        body="Random fold partition",
        facecolor=FC_PRIMARY,
    )

    # Arrows: GWAS cohort → discovery + replication
    draw_arrow(ax, 0.74, 0.50, 0.52, 0.40,
               label=f"N = {N_DISCOVERY:,}", label_pos=(0.59, 0.46))
    draw_arrow(ax, 0.82, 0.50, 0.84, 0.40,
               label=f"N = {N_REPL_PREEX:,}", label_pos=(0.86, 0.46))

    # ── Tier 4: kinship-excluded sidecar (below replication) ──────────────────
    draw_box(
        ax, 0.72, 0.02, 0.24, 0.10,
        title="Excluded for kinship",
        n_label=f"N = {N_KINSHIP_EX:,}",
        body="kinship > 0.0442 to discovery",
        facecolor=FC_EXCLUDED,
    )
    draw_arrow(ax, 0.84, 0.18, 0.84, 0.12,
               label=f"− {N_KINSHIP_EX:,}", label_pos=(0.88, 0.15))

    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] /
                            "figures" / "supplementary" /
                            "S39_cohort_flow" / "S39_cohort_flow",
                    help="Output path stem (no extension)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    save_mpl(fig, args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()
