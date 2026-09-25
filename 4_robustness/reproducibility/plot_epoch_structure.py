"""Figure: the epoch index does not identify a common state, and neither run has converged.

Two panels supporting the report's final conclusion.

  a  Alignment between the two runs at every combination of training epoch. If
     the epoch index tracked a shared state, the diagonal would dominate. It does
     not: run 1's epoch 5 aligns better with run 2's epoch 7 than with run 2's
     own epoch 5.

  b  How much the representation moves between consecutive epochs, within each
     run separately. A converging model changes less and less per epoch, so this
     should climb toward 1. It falls.

Usage:
  python plot_epoch_structure.py --cross <csv> --within1 <csv> --within2 <csv> \
      --out <stem>
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "<EXTERNAL: aim_3>/"
                   "jagwas_paper/post_gwas_analysis/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from fig_style import apply_mpl_style, save_mpl, MM, FS_LABEL, FS_TITLE  # noqa

C1 = "#2c5f8a"
C2 = "#d98c00"


def matrix(csv):
    d = pd.read_csv(csv)
    return d.groupby(["run1", "run2"])["cka"].mean().unstack()


def consecutive(csv):
    """CKA between adjacent epochs within one run, in epoch order.

    Rows and columns carry different prefixes in the within-run files (the same
    run is passed as both arguments), so columns are resolved by epoch number
    rather than by name.
    """
    m = matrix(csv)
    rows = sorted(m.index, key=lambda s: int(s.split("_ep")[-1]))
    cols = {int(c.split("_ep")[-1]): c for c in m.columns}
    out = []
    for a, b in zip(rows, rows[1:]):
        ea, eb = int(a.split("_ep")[-1]), int(b.split("_ep")[-1])
        out.append((ea, eb, float(m.loc[a, cols[eb]])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--within1", required=True)
    ap.add_argument("--within2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    apply_mpl_style()
    cross = matrix(args.cross)
    c1 = consecutive(args.within1)
    c2 = consecutive(args.within2)

    fig = plt.figure(figsize=(MM(180), MM(72)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.42,
                          left=0.10, right=0.965, top=0.86, bottom=0.19)

    # ---- panel a ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    M = cross.values
    im = ax.imshow(M, cmap="Blues", vmin=0.45, vmax=0.82)
    labs = [s.replace("s1_ep", "").replace("s2_ep", "") for s in cross.index]
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_yticklabels(labs)
    ax.set_xlabel("run 2, training epoch", fontsize=FS_LABEL)
    ax.set_ylabel("run 1, training epoch", fontsize=FS_LABEL)
    ax.set_title("a", loc="left", fontsize=FS_TITLE, fontweight="bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "{0:.3f}".format(M[i, j]), ha="center", va="center",
                    fontsize=6.5,
                    color="white" if M[i, j] > 0.70 else "#222222")
    # Ring the diagonal: if the epoch index meant anything, these would be the
    # row maxima. Two of three are not.
    for i in range(M.shape[0]):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=C2, lw=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.ax.set_title("CKA", fontsize=5.5, pad=3)
    cb.ax.tick_params(labelsize=5)
    ax.text(0.0, -0.80, "gold = matched epoch", fontsize=5.8, color=C2)

    # ---- panel b ----------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Run 2's labels go below its line; placing both above put them under run 1.
    for cc, col, nm, dy in ((c1, C1, "run 1", 9), (c2, C2, "run 2", -15)):
        x = [a for a, b, v in cc]
        y = [v for a, b, v in cc]
        ax2.plot(x, y, "-o", color=col, ms=5, lw=1.4, label=nm)
        for xi, yi in zip(x, y):
            ax2.annotate("{0:.3f}".format(yi), (xi, yi),
                         textcoords="offset points", xytext=(0, dy),
                         ha="center", fontsize=6, color=col)
    ax2.axhline(1.0, color="#999999", lw=0.7, ls="--")
    ax2.text(5.02, 0.985, "a converged model would sit here",
             fontsize=5.8, color="#666666", va="top")
    # Arrow points down, matching the direction of the lines: falling CKA means
    # the representation moves further per epoch.
    ax2.annotate("", xy=(5.42, 0.655), xytext=(5.42, 0.725),
                 arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.0))
    ax2.text(5.48, 0.690, "moving further\neach epoch", fontsize=5.8,
             color="#c44e52", va="center")
    ax2.set_xticks([5, 6])
    ax2.set_xticklabels(["5 → 6", "6 → 7"])
    ax2.set_xlim(4.85, 6.55)
    ax2.set_ylim(0.60, 1.04)
    ax2.set_xlabel("consecutive training epochs", fontsize=FS_LABEL)
    ax2.set_ylabel("CKA within the same run", fontsize=FS_LABEL)
    ax2.set_title("b", loc="left", fontsize=FS_TITLE, fontweight="bold")
    ax2.legend(loc="lower left", frameon=False, fontsize=6)

    save_mpl(fig, args.out)
    print("wrote {0}.pdf / .png".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
