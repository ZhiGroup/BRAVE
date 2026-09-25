"""Training and held-out loss per epoch, for every run we have.

All runs write the same Lightning progress-bar format to stdout, so one parser
covers the published run and every retrained one. For each epoch the LAST
progress line is taken, which is the end-of-epoch running value; where a run
restarted, later lines overwrite earlier ones for the same epoch, which is the
wanted behaviour.

The point of the figure is the epoch-7 marker: the published model is the
epoch-7 checkpoint, and this shows where that sits on the curve and how closely
the retrained runs track it up to that point.

Usage
-----
    $PY plot_loss_curves.py --run published=/path/log.txt \\
        --run seed_1=/path/seed1_train.log ... --out <stem>
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "<EXTERNAL: aim_3>/"
                   "jagwas_paper/post_gwas_analysis/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from fig_style import apply_mpl_style, save_mpl, MM, FS_LABEL, FS_TITLE  # noqa

EPOCH_RE = re.compile(r"Epoch (\d+):")
FIELDS = ["train_local_loss", "train_global_loss", "train_instance_loss",
          "val_local_loss_epoch", "val_global_loss_epoch",
          "val_instance_loss_epoch"]
FIELD_RE = {f: re.compile(r"\b" + f + r"=([0-9.eE+-]+)") for f in FIELDS}

PAPER_EPOCH = 7


def parse_log(path):
    """epoch -> {field: value}, taking the last line seen for each epoch."""
    per_epoch = {}
    with open(path, "r", errors="replace") as fh:
        blob = fh.read()
    for line in blob.replace("\r", "\n").split("\n"):
        m = EPOCH_RE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        row = per_epoch.setdefault(ep, {})
        for f, rx in FIELD_RE.items():
            hit = rx.search(line)
            if hit:
                try:
                    row[f] = float(hit.group(1))
                except ValueError:
                    pass
    rows = []
    for ep in sorted(per_epoch):
        r = dict(per_epoch[ep])
        r["epoch"] = ep
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="name=path/to/log, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv", default=None)
    cli = ap.parse_args()

    runs = []
    for spec in cli.run:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print("[skip] {0}: {1} not found".format(name, path))
            continue
        df = parse_log(path)
        if df.empty:
            print("[skip] {0}: no epoch lines parsed".format(name))
            continue
        df["run"] = name
        runs.append(df)
        have_val = "val_local_loss_epoch" in df.columns
        print("{0:14s} epochs {1:3d}..{2:3d}   val logged: {3}".format(
            name, int(df["epoch"].min()), int(df["epoch"].max()),
            "yes" if have_val else "no"))
    if not runs:
        return 1
    allr = pd.concat(runs, ignore_index=True)
    if cli.out_csv:
        allr.to_csv(cli.out_csv, index=False)
        print("\nwrote {0}".format(cli.out_csv))

    print("\nvalues at epoch {0}:".format(PAPER_EPOCH))
    at7 = allr[allr["epoch"] == PAPER_EPOCH]
    cols = [c for c in ["train_local_loss", "train_global_loss",
                        "train_instance_loss", "val_local_loss_epoch"]
            if c in at7.columns]
    print(at7[["run"] + cols].to_string(index=False, float_format="%.3f"))

    apply_mpl_style()
    fig = plt.figure(figsize=(MM(180), MM(70)))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.075, right=0.985,
                          top=0.90, bottom=0.28)

    panels = [("train_local_loss", "training loss (voxel term)", "a"),
              ("val_local_loss_epoch", "held-out loss (voxel term)", "b"),
              ("train_instance_loss", "training loss (instance term)", "c")]

    names = sorted(allr["run"].unique())
    # published in a heavier line: it is the reference the others are read against
    colours = {}
    palette = ["#d98c00", "#2c5f8a", "#7ba7c7", "#7b5aa6", "#4c9a70", "#c44e52"]
    for i, nm in enumerate(names):
        colours[nm] = "#222222" if nm == "published" else palette[i % len(palette)]

    for col, ylab, letter in panels:
        ax = fig.add_subplot(gs[0, len(fig.axes)])
        for nm in names:
            d = allr[(allr["run"] == nm)].sort_values("epoch")
            if col not in d.columns:
                continue
            d = d.dropna(subset=[col])
            if d.empty:
                continue
            ax.plot(d["epoch"], d[col], lw=1.4 if nm == "published" else 1.0,
                    color=colours[nm],
                    alpha=1.0 if nm == "published" else 0.85)
        ax.axvline(PAPER_EPOCH, color="#c44e52", lw=0.7, ls=":", zorder=1)
        ax.set_xlabel("epoch", fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.set_title(letter, loc="left", fontsize=FS_TITLE, fontweight="bold")
    fig.axes[0].text(PAPER_EPOCH + 0.8, fig.axes[0].get_ylim()[1],
                     "published\ncheckpoint", fontsize=5.6, color="#c44e52",
                     va="top")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[nm],
                      lw=1.4 if nm == "published" else 1.0, label=nm)
               for nm in names]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, fontsize=6, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.01))
    save_mpl(fig, cli.out)
    print("wrote {0}.pdf / .png".format(cli.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
