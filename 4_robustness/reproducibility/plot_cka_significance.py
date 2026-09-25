"""Figure: is the seed-to-seed CKA high, and against what?

Three panels:

  a  Per-region observed CKA with bootstrap CI, each against its OWN
     cross-region null. The region-specific null is the point -- a single pooled
     null would misrank the thalami, whose representations are unlike any other
     region's and so have a much lower floor. All three pairings are shown:
     the two retrained encoders against each other, and each against the
     published encoder. They sit on top of one another, which is the result:
     retraining reproduces the published encoder about as well as it reproduces
     another retraining.
  b  Effective rank against CKA, which isolates the thalamus. Spectral
     concentration explains the two thalami and nothing else; the rank
     correlation across all 16 regions is near zero.
  c  Seed-to-seed CKA against the CKA cost of moving one training epoch within
     a single run. Points above the identity line are regions where two
     differently seeded encoders agree MORE closely than one encoder agrees
     with itself an epoch earlier, which is why that reference is not a ceiling.

Reads the CSVs written by cka_significance.py, effective_rank.py,
ceiling_cka.py and concordance_embeddings.py.

Usage:
  python plot_cka_significance.py --cka <csv> --rank <csv> \
      --ceiling <csv> --concordance <csv> --out <stem>
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

PRETTY = {
    "Left_Thalamus_Proper": "L. Thalamus", "Right_Thalamus-Proper": "R. Thalamus",
    "Left_Caudate": "L. Caudate", "Right_Caudate": "R. Caudate",
    "Left_Putamen": "L. Putamen", "Right_Putamen": "R. Putamen",
    "Left_Pallidum": "L. Pallidum", "Right_Pallidum": "R. Pallidum",
    "Left_Hippocampus": "L. Hippocampus", "Right_Hippocampus": "R. Hippocampus",
    "Left_Amygdala": "L. Amygdala", "Right_Amygdala": "R. Amygdala",
    "Left_Accumbens-area": "L. Accumbens",
    "Right_Accumbens-area": "R. Accumbens",
    "Brain_Stem _or_4th_Ventricle": "Brainstem", "CSF": "CSF",
}

C_OBS = "#2c5f8a"
C_NULL = "#c44e52"
C_THAL = "#d98c00"
C_PUB = "#7b5aa6"

# Vertical offset, in row units, applied to the two vs-published series so they
# do not sit under the seed-vs-seed marker.
DODGE = 0.26

# (column in concordance_embeddings.csv, marker, legend label)
PUB_SERIES = [
    ("seed1_vs_published", "^", "seed 1 vs published"),
    ("seed2_vs_published", "v", "seed 2 vs published"),
]


def label_side(ax, x_value):
    """Annotate away from the nearer axis edge.

    Fixed labels running off the right spine in panel c, where the thalamus
    points sit at the top of the x-range.
    """
    lo, hi = ax.get_xlim()
    if x_value > (lo + hi) / 2.0:
        return "right", -6
    return "left", 6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cka", required=True)
    ap.add_argument("--rank", required=True)
    ap.add_argument("--ceiling", default=None,
                    help="ceiling_cka.csv; adds the reference-perturbation panel")
    ap.add_argument("--concordance", default=None,
                    help="concordance_embeddings.csv; adds the two "
                         "vs-published pairings to panel a")
    ap.add_argument("--out", required=True, help="output path stem")
    args = ap.parse_args()

    apply_mpl_style()
    df = pd.read_csv(args.cka).sort_values("cka_observed").reset_index(drop=True)
    rk = pd.read_csv(args.rank)
    ceil = pd.read_csv(args.ceiling) if args.ceiling else None
    con = pd.read_csv(args.concordance) if args.concordance else None

    # Two rows, with panel a spanning both on the left. Three panels in a single
    # row left panel a about 79 mm wide, which is not enough: with three series
    # per region the markers collided both with each other and with the null
    # bars. Stacking b over c gives panel a ~115 mm of width and the full figure
    # height, so its 16 rows sit ~7 mm apart.
    #
    # save_mpl crops with bbox_inches='tight', which trims outside the legend but
    # not the gap between it and the x-labels, so the bottom margin stays tight.
    if ceil is not None:
        fig = plt.figure(figsize=(MM(180), MM(132)))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0],
                              height_ratios=[1.0, 1.0], wspace=0.32,
                              hspace=0.40, left=0.145, right=0.985,
                              top=0.955, bottom=0.115)
        slot_a, slot_b, slot_c = gs[:, 0], gs[0, 1], gs[1, 1]
    else:
        fig = plt.figure(figsize=(MM(180), MM(92)))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.40,
                              left=0.115, right=0.985, top=0.91, bottom=0.185)
        slot_a, slot_b, slot_c = gs[0, 0], gs[0, 1], None

    # ---- panel a ----------------------------------------------------------
    ax = fig.add_subplot(slot_a)
    y = np.arange(len(df))
    names = [PRETTY.get(r, r) for r in df["region"]]
    is_thal = np.array(["Thalamus" in n for n in names])

    lo = df["cka_observed"] - df["boot_lo"]
    hi = df["boot_hi"] - df["cka_observed"]

    # Region-specific null first, so the observed points read as sitting above it.
    for i in y:
        ax.plot([df["cross_mean"][i] - df["cross_sd"][i],
                 df["cross_mean"][i] + df["cross_sd"][i]], [i, i],
                color=C_NULL, lw=2.4, alpha=0.30, solid_capstyle="butt",
                zorder=1)
    ax.scatter(df["cross_mean"], y, s=13, marker="|", color=C_NULL, zorder=2)
    # The null maximum matters: two regions sit below their own null's maximum,
    # so "exceeds the null" is true of the mean but not of the distribution.
    ax.scatter(df["cross_max"], y, s=9, marker=".", color=C_NULL, alpha=0.85,
               zorder=2)
    ax.errorbar(df["cka_observed"], y, xerr=[lo, hi], fmt="none",
                ecolor="#444444", elinewidth=0.8, capsize=1.6, zorder=3)
    # At n = 22,999 the bootstrap CI is +-0.004, narrower than the marker, so
    # the caption says so rather than implying an invisible interval is drawn.
    ax.scatter(df["cka_observed"], y, s=17, color=C_OBS, zorder=4)

    # The two pairings against the published encoder, dodged off the row centre.
    # Several regions agree to within 0.01 across all three pairings, so drawn on
    # a common baseline the markers sat on top of one another and the agreement
    # -- which is the result -- was unreadable. The dodge is cosmetic: all three
    # series share the row's x-axis.
    if con is not None:
        piv = con.pivot(index="region", columns="comparison", values="cka")
        piv = piv.reindex(df["region"].values)
        for (col, marker, _lab), dy in zip(PUB_SERIES, (DODGE, -DODGE)):
            if col not in piv.columns:
                continue
            ax.scatter(piv[col].values, y + dy, s=19, marker=marker,
                       facecolors="none", edgecolors=C_PUB, linewidths=0.85,
                       zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    # The thalamus highlight moves to the tick labels: with three series in the
    # panel, recolouring one series' markers would read as a fourth series.
    for tick, thal in zip(ax.get_yticklabels(), is_thal):
        if thal:
            tick.set_color(C_THAL)
    ax.set_ylim(-0.8, len(df) - 0.2)
    # Headroom past 1.0 so the per-region ratios sit outside the data range;
    # at xlim 1.0 they printed on top of the highest observed points.
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("CKA between encoders", fontsize=FS_LABEL)
    ax.set_title("a", loc="left", fontsize=FS_TITLE, fontweight="bold")

    for i in y:
        ax.text(1.15, i, "{0:.2f}x".format(df["ratio_vs_cross"][i]),
                fontsize=5.6, va="center", ha="right", color="#555555")

    # ---- panel b ----------------------------------------------------------
    ax2 = fig.add_subplot(slot_b)
    m = rk.copy()
    m["pretty"] = [PRETTY.get(r, r) for r in m["region"]]
    thal = np.array(["Thalamus" in p for p in m["pretty"]])
    ax2.scatter(m.loc[~thal, "eff_rank_pr_mean"], m.loc[~thal, "cka_observed"],
                s=17, color=C_OBS, zorder=3)
    ax2.scatter(m.loc[thal, "eff_rank_pr_mean"], m.loc[thal, "cka_observed"],
                s=22, color=C_THAL, zorder=4)
    for _, r in m[thal].iterrows():
        ha, dx = label_side(ax2, r["eff_rank_pr_mean"])
        ax2.annotate(r["pretty"], (r["eff_rank_pr_mean"], r["cka_observed"]),
                     textcoords="offset points", xytext=(dx, -1), ha=ha,
                     fontsize=6, color=C_THAL)

    x = m["eff_rank_pr_mean"].values
    yy = m["cka_observed"].values
    rho = float(np.corrcoef(pd.Series(x).rank(), pd.Series(yy).rank())[0, 1])
    ax2.set_xlabel("effective rank (participation ratio)", fontsize=FS_LABEL)
    ax2.set_ylabel("CKA, seed 1 vs seed 2", fontsize=FS_LABEL)
    ax2.set_title("b", loc="left", fontsize=FS_TITLE, fontweight="bold")
    ax2.set_ylim(0.2, 1.0)
    ax2.text(0.96, 0.06, "Spearman $\\rho$ = {0:+.2f}".format(rho),
             transform=ax2.transAxes, ha="right", fontsize=6, color="#555555")

    # ---- panel c ----------------------------------------------------------
    if ceil is not None:
        ax3 = fig.add_subplot(slot_c)
        c = ceil.copy()
        c["pretty"] = [PRETTY.get(r, r) for r in c["region"]]
        th = np.array(["Thalamus" in p for p in c["pretty"]])
        lim = [0.25, 0.95]
        # Identity line: points above it are regions where two seeds agree more
        # closely than one run does with itself an epoch earlier, which is why
        # this reference is not a strict ceiling.
        ax3.plot(lim, lim, color="#999999", lw=0.7, ls="--", zorder=1)
        ax3.scatter(c.loc[~th, "ceiling_ep6_vs_ep7"],
                    c.loc[~th, "observed_seed1_vs_seed2"],
                    s=17, color=C_OBS, zorder=3)
        ax3.scatter(c.loc[th, "ceiling_ep6_vs_ep7"],
                    c.loc[th, "observed_seed1_vs_seed2"],
                    s=22, color=C_THAL, zorder=4)
        ax3.set_xlim(lim)
        ax3.set_ylim(lim)
        for _, r in c[th].iterrows():
            ha, dx = label_side(ax3, r["ceiling_ep6_vs_ep7"])
            ax3.annotate(r["pretty"], (r["ceiling_ep6_vs_ep7"],
                                       r["observed_seed1_vs_seed2"]),
                         textcoords="offset points", xytext=(dx, -2), ha=ha,
                         fontsize=6, color=C_THAL)
        n_above = int((c["observed_seed1_vs_seed2"]
                       > c["ceiling_ep6_vs_ep7"]).sum())
        ax3.set_xlabel("CKA, one epoch apart", fontsize=FS_LABEL)
        ax3.set_ylabel("CKA, seed 1 vs seed 2", fontsize=FS_LABEL)
        ax3.set_title("c", loc="left", fontsize=FS_TITLE, fontweight="bold")
        ax3.text(0.04, 0.96, "{0}/16 regions above\nthe identity line".format(
            n_above), transform=ax3.transAxes, va="top", fontsize=6,
            color="#555555")

    # ---- legend, below the axes -------------------------------------------
    # Explicit handles: a scatter given a colour list hands the legend whichever
    # colour happens to come last. Placed in-axes the legend had to go
    # upper-left, the only corner free of both the nulls and the points, which
    # forced labels short enough to clear the per-region ratio annotations at
    # the right edge. Outside there is room to say what each mark is.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="|", ls="none", color=C_NULL, markersize=5,
               label="cross-region null (mean $\\pm$ SD)"),
        Line2D([], [], marker=".", ls="none", color=C_NULL, markersize=4,
               label="null maximum"),
        Line2D([], [], marker="o", ls="none", color=C_OBS, markersize=3.6,
               label="seed 1 vs seed 2"),
    ]
    if con is not None:
        for col, marker, lab in PUB_SERIES:
            handles.append(
                Line2D([], [], marker=marker, ls="none", mfc="none",
                       mec=C_PUB, mew=0.85, markersize=3.8, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=6, handletextpad=0.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.008))

    save_mpl(fig, args.out)
    print("wrote {0}.pdf / .png".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
