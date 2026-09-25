"""Extract the held-out validation loss curve from a run's TensorBoard log.

The encoder pipeline already carries a validation cohort: `preprocessing.py`
builds `dataset_val` from `ae_val` (1,530 participants, disjoint from the 4,586
training images), `models.Model.validation_step` evaluates the same three
contrastive terms on it, and Lightning runs it once per epoch. The losses are
logged as `val_local_loss_epoch`, `val_global_loss_epoch` and
`val_instance_loss_epoch` but were never read out.

This dumps them next to the corresponding training losses so the two can be
compared epoch by epoch -- the standard self-supervised check that the pretext
task is still improving on data the encoder has not seen.

Usage
-----
    $PY val_curve.py --run seed_1 seed_2 --out val_curve.csv
"""

# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.

import argparse
import glob
import os

import pandas as pd

DEFAULT_ROOT = ("<EXTERNAL: aim_3>/"
                "reproducibility")
TAGS = ["val_local_loss_epoch", "val_global_loss_epoch",
        "val_instance_loss_epoch", "train_local_loss_epoch",
        "train_global_loss_epoch", "train_instance_loss_epoch"]


def event_files(root, run):
    """Every TB event file under <root>/<run>/log, newest last."""
    pattern = os.path.join(root, run, "log", "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def read_run(root, run):
    from tensorboard.backend.event_processing import event_accumulator

    rows = {}
    files = event_files(root, run)
    if not files:
        print("  [{0}] no event files found".format(run))
        return pd.DataFrame()

    for path in files:
        acc = event_accumulator.EventAccumulator(
            path, size_guidance={event_accumulator.SCALARS: 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for tag in TAGS:
            if tag not in available:
                continue
            for ev in acc.Scalars(tag):
                # epoch-level scalars are logged once per epoch, so the running
                # step index is remapped to epoch order after collection
                rows.setdefault(ev.step, {})[tag] = ev.value

    if not rows:
        print("  [{0}] event files carry none of the requested tags; "
              "available: {1}".format(run, sorted(available)[:12]))
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "global_step"
    df = df.reset_index()
    df.insert(0, "epoch", range(len(df)))
    df.insert(0, "run", run)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    frames = []
    for run in cli.run:
        print("[read] {0}".format(run))
        df = read_run(cli.root, run)
        if not df.empty:
            frames.append(df)
            cols = [c for c in TAGS if c in df.columns]
            print(df[["epoch"] + cols].to_string(index=False,
                                                 float_format="%.4f"))
            print()

    if not frames:
        return 1
    out = pd.concat(frames, ignore_index=True)
    path = cli.out or os.path.join(cli.root, "val_curve.csv")
    out.to_csv(path, index=False)
    print("[write] {0}  ({1} rows)".format(path, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
