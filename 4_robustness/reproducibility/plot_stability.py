"""Supplementary figure: stability of association statistics and of loci.

Five panels.

  a-c  Do the runs agree on the association statistics? Hexbins of -log10 P for
       all three pairings: the two retrained encoders against each other, and
       each against the published encoder. Showing only the retrained pair
       answers "do two reruns agree", which is not the question a reviewer asks
       -- they ask whether retraining reproduces the PUBLISHED result.
  d    How many of the published loci survive retraining? The 276 loci by tier.
       The point of the panel is the absent fourth category: no locus is null in
       both runs.
  e    Are the non-recovered loci absent, or merely sub-threshold? The minimum P
       across the 16 regional tests attained by non-recovered loci in the run
       that missed them, against the same quantity for randomly drawn variants.
       Because the lookup takes a minimum over 16 correlated tests, an
       unadjusted P < 0.05 is nearly uninformative -- roughly 40% of random
       variants clear it -- so the matched random distribution is drawn
       alongside rather than a nominal threshold. This panel carries the
       argument.

Correlations are computed by streaming over every variant rather than from the
plotted sample, which is deliberately enriched for significant variants and
would give a biased r. The seed-1-vs-seed-2 value is printed next to the
independently computed one in selection_diagnostics.csv as a consistency check.

Usage:
  python plot_stability.py --seed1 <jagwas-dir> --seed2 <jagwas-dir> \
      --published <jagwas-dir> --consensus <csv> --seldiag <csv> --out <stem>
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

LEAD_P = 3.125e-9
KEEP_BELOW = 1e-6          # keep every variant this significant in either run
BACKGROUND = 30_000        # random variants per region for the hexbin backdrop
N_NULL = 50_000            # random variants for the panel-e null

C_MAIN = "#2c5f8a"
C_NULL = "#c44e52"
C_TIER = ["#2c5f8a", "#7ba7c7", "#d98c00"]

# (key_a, key_b, panel letter, axis label a, axis label b)
PAIRINGS = [
    ("seed1", "seed2", "a", "seed 1", "seed 2"),
    ("seed1", "published", "b", "seed 1", "published"),
    ("seed2", "published", "c", "seed 2", "published"),
]


def nlp(p):
    return -np.log10(np.clip(p, 1e-300, 1.0))


def norm_region(name):
    """Directory names differ between runs: the published tree writes
    'Brain_Stem_or_4th_Ventricle', the seed trees carry a stray space."""
    return name.replace(" ", "").replace("-", "_").lower()


def region_maps(dirs):
    """normalised region name -> {run: results path}, for regions in every run."""
    per_run = {}
    for run, root in dirs.items():
        found = {}
        for d in sorted(os.listdir(root)):
            f = os.path.join(root, d, "{0}_JAGWAS_results.txt.gz".format(d))
            if os.path.exists(f):
                found[norm_region(d)] = f
        per_run[run] = found
    common = set.intersection(*[set(v) for v in per_run.values()])
    return sorted(common), per_run


def read_region(path):
    """P plus locus keys, so alignment across runs can be verified not assumed."""
    try:
        df = pd.read_csv(path, sep="\t", usecols=["CHR", "POS", "P"])
        key = (df["CHR"].values.astype(np.int64) * 1_000_000_000
               + df["POS"].values.astype(np.int64))
    except ValueError:
        df = pd.read_csv(path, sep="\t", usecols=["P"])
        key = None
    p = pd.to_numeric(df["P"], errors="coerce").values
    return p, key


class Accum(object):
    """Streaming Pearson accumulator over -log10 P."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        self.n += len(x)
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()

    def r(self):
        if self.n < 3:
            return np.nan
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx ** 2 / self.n
        vy = self.syy - self.sy ** 2 / self.n
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def align_by_key(key, runs):
    """Positions of the CHR:POS values shared by every run.

    The published run and the retrained runs do NOT carry the same variant
    order, so the earlier positional truncation was silently comparing
    different variants. Duplicated keys (multi-allelics) are reduced to their
    first occurrence before intersecting.
    """
    uniq = {}
    for r in runs:
        u, first = np.unique(key[r], return_index=True)
        uniq[r] = (u, first)
    common = uniq[runs[0]][0]
    for r in runs[1:]:
        common = np.intersect1d(common, uniq[r][0], assume_unique=True)
    if len(common) == 0:
        return None, None
    out = {}
    for r in runs:
        u, first = uniq[r]
        out[r] = first[np.searchsorted(u, common)]
    return out, common


def collect(dirs, rng):
    """One pass over every run: hexbin samples, streaming r, min-over-16 nulls."""
    regions, per_run = region_maps(dirs)
    runs = sorted(dirs)
    print("regions common to all runs: {0}".format(len(regions)))

    samples = {(a, b): ([], []) for a, b, _, _, _ in PAIRINGS}
    r_all = {(a, b): Accum() for a, b, _, _, _ in PAIRINGS}
    r_sig = {(a, b): Accum() for a, b, _, _, _ in PAIRINGS}
    mins = {run: None for run in runs}
    null_keys = None

    for region in regions:
        p, key = {}, {}
        for run in runs:
            p[run], key[run] = read_region(per_run[run][region])

        if any(key[r] is None for r in runs):
            lens = set(len(p[r]) for r in runs)
            if len(lens) != 1:
                print("  [SKIP] {0}: no CHR/POS columns and lengths differ"
                      .format(region))
                continue
            ckeys = None
        else:
            pos, ckeys = align_by_key(key, runs)
            if pos is None:
                print("  [SKIP] {0}: no shared variants".format(region))
                continue
            for run in runs:
                p[run] = p[run][pos[run]]

        n = len(p[runs[0]])
        ok_all = np.ones(n, dtype=bool)
        for run in runs:
            ok_all &= np.isfinite(p[run]) & (p[run] > 0)

        for a, b, _lt, _la, _lb in PAIRINGS:
            xa, yb = nlp(p[a][ok_all]), nlp(p[b][ok_all])
            r_all[(a, b)].add(xa, yb)
            s = (p[a][ok_all] < LEAD_P) | (p[b][ok_all] < LEAD_P)
            if s.sum() > 2:
                r_sig[(a, b)].add(xa[s], yb[s])

            sig = ok_all & ((p[a] < KEEP_BELOW) | (p[b] < KEEP_BELOW))
            rest = np.where(ok_all & ~sig)[0]
            take = rng.choice(rest, size=min(BACKGROUND, len(rest)),
                              replace=False)
            sel = np.concatenate([np.where(sig)[0], take])
            samples[(a, b)][0].append(nlp(p[a][sel]))
            samples[(a, b)][1].append(nlp(p[b][sel]))

        # The min-over-16 null must follow the SAME variants through every
        # region, so it is keyed on CHR:POS rather than on row position -- the
        # shared variant set differs slightly from region to region. np.fmin
        # ignores the NaNs left where a variant is absent from a region.
        if ckeys is not None:
            if null_keys is None:
                null_keys = np.sort(rng.choice(ckeys,
                                               size=min(N_NULL, len(ckeys)),
                                               replace=False))
            at = np.searchsorted(ckeys, null_keys)
            at_c = np.clip(at, 0, len(ckeys) - 1)
            hit = ckeys[at_c] == null_keys
            for run in runs:
                vals = np.full(len(null_keys), np.nan)
                vals[hit] = p[run][at_c[hit]]
                mins[run] = (vals if mins[run] is None
                             else np.fmin(mins[run], vals))
        print("  {0:30s} shared variants {1:,}".format(region[:30], n),
              flush=True)

    null_p = np.concatenate([mins[r][np.isfinite(mins[r])]
                             for r in ("seed1", "seed2") if mins[r] is not None])
    samples = {k: (np.concatenate(v[0]), np.concatenate(v[1]))
               for k, v in samples.items()}
    return samples, r_all, r_sig, null_p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--seed2", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--seldiag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    dirs = {"seed1": args.seed1, "seed2": args.seed2,
            "published": args.published}
    print("collecting ...", flush=True)
    samples, r_all, r_sig, null_p = collect(dirs, rng)

    sd = pd.read_csv(args.seldiag)
    print("\nstreamed r (all variants) vs selection_diagnostics.csv:")
    print("  seed1-seed2 here {0:.4f}   seldiag {1:.4f}".format(
        r_all[("seed1", "seed2")].r(), sd["r_joint_genomewide"].mean()))
    for a, b, _lt, _la, _lb in PAIRINGS:
        print("  {0:>9s} vs {1:<10s} all {2:.4f}   genome-wide sig {3:.4f}"
              .format(a, b, r_all[(a, b)].r(), r_sig[(a, b)].r()))

    cons = pd.read_csv(args.consensus)

    apply_mpl_style()
    # Two rows: three hexbins over the two locus panels. One row of five would
    # leave each hexbin ~30 mm wide, too small to read a density.
    fig = plt.figure(figsize=(MM(180), MM(128)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], wspace=0.52,
                          hspace=0.46, left=0.075, right=0.955, top=0.94,
                          bottom=0.085)

    # ---- panels a-c: the three pairings ------------------------------------
    lim = max(v.max() for pair in samples.values() for v in pair) * 1.02
    hb = None
    for i, (a, b, letter, la, lb) in enumerate(PAIRINGS):
        ax = fig.add_subplot(gs[0, i])
        x, y = samples[(a, b)]
        hb = ax.hexbin(x, y, gridsize=52, bins="log", mincnt=1, cmap="Blues",
                       linewidths=0, extent=(0, lim, 0, lim))
        ax.plot([0, lim], [0, lim], color="#999999", lw=0.7, ls="--", zorder=3)
        ax.axvline(-np.log10(LEAD_P), color=C_NULL, lw=0.6, ls=":", zorder=2)
        ax.axhline(-np.log10(LEAD_P), color=C_NULL, lw=0.6, ls=":", zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        # Concatenated, not formatted: str.format reads the {10} in \log_{10}
        # as positional field 10 and raises IndexError.
        ax.set_xlabel("$-\\log_{10}P$, " + la, fontsize=FS_LABEL)
        ax.set_ylabel("$-\\log_{10}P$, " + lb, fontsize=FS_LABEL)
        ax.set_title(letter, loc="left", fontsize=FS_TITLE, fontweight="bold")
        ax.text(0.04, 0.96, "all  $r$ = {0:.2f}\nGWS  $r$ = {1:.2f}".format(
            r_all[(a, b)].r(), r_sig[(a, b)].r()), transform=ax.transAxes,
            va="top", fontsize=6, color="#333333")
    cb = fig.colorbar(hb, ax=fig.axes[:3], fraction=0.020, pad=0.012)
    cb.ax.set_title("variants", fontsize=5.5, pad=3)
    cb.ax.tick_params(labelsize=5)

    # ---- panel d: locus tiers ----------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    order = ["consensus", "partial", "published_subthreshold"]
    names = ["both seeds", "one seed", "neither,\nsub-threshold"]
    counts = [int((cons["tier"] == t).sum()) for t in order]
    n_null_tier = int((cons["tier"] == "published_only").sum())
    bottom = 0
    for c, nm, col in zip(counts, names, C_TIER):
        ax2.bar(0, c, bottom=bottom, color=col, width=0.62)
        ax2.text(0, bottom + c / 2.0, "{0}\n{1}".format(nm, c), ha="center",
                 va="center", fontsize=6,
                 color="white" if col != C_TIER[2] else "#222222")
        bottom += c
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(0, len(cons) * 1.14)
    ax2.set_xticks([])
    ax2.set_ylabel("published loci", fontsize=FS_LABEL)
    ax2.set_title("d", loc="left", fontsize=FS_TITLE, fontweight="bold")
    ax2.text(0, len(cons) * 1.06, "null in both: {0}".format(n_null_tier),
             ha="center", fontsize=6, color=C_NULL, fontweight="bold")

    # ---- panel e: min-P distributions --------------------------------------
    # Plotted as -log10 so the axis runs the same direction as panels a-c:
    # further right means more significant everywhere in the figure.
    ax3 = fig.add_subplot(gs[1, 1:])
    miss = pd.concat([cons["p_seed1"], cons["p_seed2"]]).dropna().values
    miss = miss[miss > 0]
    bins = np.linspace(0, 14, 57)
    ax3.hist(nlp(null_p), bins=bins, density=True, color=C_NULL, alpha=0.45,
             label="random variants")
    ax3.hist(nlp(miss), bins=bins, density=True, color=C_MAIN, alpha=0.75,
             label="non-recovered loci")
    ax3.axvline(-np.log10(LEAD_P), color="#444444", lw=0.8, ls="--")
    ax3.text(-np.log10(LEAD_P) + 0.18, ax3.get_ylim()[1] * 0.02,
             "calling threshold", rotation=90, ha="left", va="bottom",
             fontsize=5.4, color="#444444")
    ax3.set_xlabel("$-\\log_{10}$ min $P$ across 16 regions", fontsize=FS_LABEL)
    ax3.set_ylabel("density", fontsize=FS_LABEL)
    ax3.set_title("e", loc="left", fontsize=FS_TITLE, fontweight="bold")
    # Upper right: with the axis flipped the random-variant peak now sits at the
    # left, so the legend would land on top of it there.
    ax3.legend(loc="upper right", frameon=False, fontsize=5.8,
               handletextpad=0.5, borderaxespad=0.25, labelspacing=0.3)
    ax3.text(0.985, 0.62, "median $P$\n{0:.1e} vs {1:.1e}".format(
        np.median(miss), np.median(null_p)), transform=ax3.transAxes,
        ha="right", va="top", fontsize=5.6, color="#333333")

    save_mpl(fig, args.out)
    print("wrote {0}.pdf / .png".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
