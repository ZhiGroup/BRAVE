#!/usr/bin/env python3
"""
42_fig4a_replication_bars.py
PI-requested redesign of Fig 4A: drop the donut, use two horizontal stacked bars
(ENIGMA-known vs Novel) split across five replication strength categories.

Why: PI flagged the donut as space-inefficient and inconsistent with the rest of
the figure's stacked-bar style. The information PI wants preserved is the
replication breakdown side-by-side for ENIGMA-known vs novel loci, with the
same five categories used in the previous donut.

Input:  results/replication/aggregate_replication.csv (276 rows, rep_category +
        novel_vs_all columns)
Output: figures/main/Fig4_novelty_gc/Fig4A_replication_bars.{pdf,png}
        + figures/replication/aggregate_replication_summary.{pdf,png}

Run:    python 42_fig4a_replication_bars.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL  # noqa: E402

PROJECT_BASE = SCRIPTS.parent
DATA_CSV = PROJECT_BASE / "results" / "replication" / "aggregate_replication.csv"

# Two output destinations under the repo figures tree:
# (a) the composite-feeding path used by the Fig 4 assembler.
# (b) the source-of-truth path under figures/replication/.
PAPER_OUT = PROJECT_BASE / "figures" / "supplementary" \
            / "S23_replication" / "aggregate_replication_summary"
LOCAL_OUT = PROJECT_BASE / "figures" / "replication" / "aggregate_replication_summary"

CATEGORY_ORDER = ["GW", "Bonferroni", "Nominal", "Not replicated", "Not found"]
CATEGORY_COLOURS = {
    "GW":             "#B2182B",  # deep red
    "Bonferroni":     "#E27D5F",  # red-orange
    "Nominal":        "#F4B26B",  # orange
    "Not replicated": "#9CA3A8",  # mid grey
    "Not found":      "#D8DCDE",  # light grey
}


def main() -> None:
    apply_mpl_style()
    df = pd.read_csv(DATA_CSV)

    # Split by novelty vs ENIGMA: novel_vs_all == True means novel beyond ENIGMA.
    groups = [
        ("ENIGMA-known", df[~df["novel_vs_all"]]),
        ("Novel", df[df["novel_vs_all"]]),
    ]

    # Build the per-group composition matrix (rows: 2 groups; cols: 5 categories).
    counts = np.zeros((len(groups), len(CATEGORY_ORDER)), dtype=int)
    totals = np.zeros(len(groups), dtype=int)
    for i, (_, gdf) in enumerate(groups):
        totals[i] = len(gdf)
        for j, c in enumerate(CATEGORY_ORDER):
            counts[i, j] = int((gdf["rep_category"] == c).sum())
    fractions = counts / totals[:, None]

    # Horizontal stacked bars.
    fig, ax = plt.subplots(figsize=(MM(180), MM(60)))
    y_positions = np.arange(len(groups))[::-1]  # ENIGMA-known on top
    left = np.zeros(len(groups))
    for j, c in enumerate(CATEGORY_ORDER):
        widths = fractions[:, j]
        ax.barh(y_positions, widths, left=left,
                color=CATEGORY_COLOURS[c], edgecolor="white", linewidth=0.6,
                label=c, height=0.55)
        # Annotate counts at stack midpoints (skip if width < 0.04 to avoid clutter)
        for i, w in enumerate(widths):
            if w < 0.035:
                continue
            text_colour = "white" if c in ("GW", "Bonferroni", "Not replicated") else "#222"
            ax.text(left[i] + w / 2.0, y_positions[i],
                    f"{counts[i, j]}\n({w*100:.0f}%)",
                    ha="center", va="center",
                    fontsize=FS_TICK - 0.5, color=text_colour, linespacing=0.9)
        left += widths

    # Row labels (left) with totals
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{g[0]}\n(n = {totals[i]})" for i, g in enumerate(groups)],
                       fontsize=FS_TICK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=FS_TICK)
    ax.set_xlabel("Fraction of loci by replication category", fontsize=FS_LABEL)
    ax.set_title("Replication of 276 JAGWAS aggregate loci in the independent cohort (N = 12,359)",
                 loc="left", fontweight="bold", fontsize=FS_LABEL, pad=4)
    ax.invert_yaxis()  # ENIGMA-known on top reading-order

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", left=False)

    # Legend with thresholds.
    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS["GW"],
                      label="GW (p < 5 × 10⁻⁸)"),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS["Bonferroni"],
                      label="Bonferroni (p < 1.8 × 10⁻⁴)"),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS["Nominal"],
                      label="Nominal (p < 0.05)"),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS["Not replicated"],
                      label="Not replicated"),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOURS["Not found"],
                      label="Not found in replication"),
    ]
    ax.legend(handles=leg_handles, ncol=5, fontsize=FS_TICK - 0.5,
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.55))

    plt.tight_layout()

    for out_stem in (PAPER_OUT, LOCAL_OUT):
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        save_mpl(fig, out_stem)
        print(f"Wrote: {out_stem}.pdf / .png")
    plt.close(fig)

    print("\nReplication composition:")
    for i, (gname, _) in enumerate(groups):
        parts = ", ".join(f"{CATEGORY_ORDER[j]}={counts[i, j]} ({fractions[i, j]*100:.1f}%)"
                          for j in range(len(CATEGORY_ORDER)))
        print(f"  {gname:>13s} (n={totals[i]:3d}): {parts}")


if __name__ == "__main__":
    main()
