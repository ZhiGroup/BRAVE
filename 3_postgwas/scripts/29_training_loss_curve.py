"""
29_training_loss_curve.py
Supplementary Fig. S46 (file stem S40_training_loss) -- per-epoch training loss of
the published BRAVE encoder run, with the epoch-7 checkpoint marked.

Source: the published run's own training log, <embedding.log_dir>/train_voxelpixel.txt
(Lightning progress-bar output, 2025-01-18 -> 2025-01-23, epochs 0-39). For each
epoch the last progress line is taken, i.e. the end-of-epoch running value; where
the run restarted (epoch 0 was started three times) later lines overwrite earlier
ones. This is the same parsing as 4_robustness/reproducibility/plot_loss_curves.py
(Fig. S50), and the script refuses to plot unless its values equal the 'published'
rows of the loss_curves.csv that script writes (pass it with --check).

History: until 2026-11-21 this script parsed iteration_epoch_*_tier_*.pt checkpoint
file names. Those belong to a separate, earlier training run (2024-12-31 ->
2025-01-02, batch size 3, 16 epochs), not to the published encoder, whose epoch-7
checkpoint reproduces the stored BREs exactly.

Panels (training images only; the held-out curve is in Fig. S50b):
  a  fine-scale voxel-level InfoNCE   (logged as train_local_loss)
  b  coarse-scale voxel-level InfoNCE (logged as train_global_loss)
  c  instance-level InfoNCE           (logged as train_instance_loss)
The training objective is the unweighted sum of the three.

Outputs:
  3_postgwas/figures/training_loss/S40_training_loss.{pdf,png} (override with --out)

ENV:   torch_re_torch_7 (matplotlib + pandas + fig_style)
RUN:   python 29_training_loss_curve.py
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg  # noqa: E402

apply_mpl_style()

HERE = Path(__file__).resolve()
DEFAULT_LOG = Path(str(cfg.embedding.log_dir)) / 'train_voxelpixel.txt'
DEFAULT_CHECK = Path('<EXTERNAL: loss_curves.csv written by 4_robustness/reproducibility/plot_loss_curves.py>')
DEFAULT_OUT = HERE.parents[1] / 'figures' / 'training_loss' / 'S40_training_loss'

PRODUCTION_EPOCH = 7      # the checkpoint used for all BRE extraction
N_EPOCHS = 40             # the published run completed epochs 0-39

FIELDS = ('train_local_loss', 'train_global_loss', 'train_instance_loss')
LABELS = ('Fine-scale voxel InfoNCE', 'Coarse-scale voxel InfoNCE',
          'Instance-level InfoNCE')
PALETTE = ('#C44E52', '#2B3A55', '#55A868')

EPOCH_RE = re.compile(r'Epoch (\d+):')
FIELD_RE = {f: re.compile(r'\b' + f + r'=([0-9.eE+-]+)') for f in FIELDS}


def parse_log(path):
    """epoch -> {field: value}, keeping the last progress line seen per epoch."""
    per_epoch = {}
    with open(path, 'r', errors='replace') as fh:
        blob = fh.read()
    for line in blob.replace('\r', '\n').split('\n'):
        m = EPOCH_RE.search(line)
        if not m:
            continue
        row = per_epoch.setdefault(int(m.group(1)), {})
        for f, rx in FIELD_RE.items():
            hit = rx.search(line)
            if hit:
                try:
                    row[f] = float(hit.group(1))
                except ValueError:
                    pass
    df = pd.DataFrame([dict(per_epoch[e], epoch=e) for e in sorted(per_epoch)])
    return df


def check_against(df, check_csv):
    """Refuse to continue unless the parse equals the Fig. S50 data exactly."""
    ref = pd.read_csv(check_csv)
    ref = ref[ref['run'] == 'published'].sort_values('epoch').reset_index(drop=True)
    if list(df['epoch']) != list(range(N_EPOCHS)):
        print('ERROR: expected epochs 0-{0}, parsed {1}'.format(
            N_EPOCHS - 1, list(df['epoch'])))
        return False
    if list(ref['epoch']) != list(df['epoch']):
        print('ERROR: epoch sets differ from {0}'.format(check_csv))
        return False
    ok = True
    for f in FIELDS:
        a = df[f].values
        b = ref[f].values
        if not np.array_equal(a, b):
            bad = np.where(a != b)[0]
            print('ERROR: {0} differs from {1} at epochs {2}'.format(
                f, check_csv.name, list(df['epoch'].values[bad])))
            ok = False
    return ok


def build_figure(df):
    epochs = df['epoch'].values
    fig, axes = plt.subplots(1, 3, figsize=(MM(180), MM(55)), sharex=True)
    for k, (ax, f, label, color) in enumerate(zip(axes, FIELDS, LABELS, PALETTE)):
        ax.plot(epochs, df[f].values, marker='o', markersize=2.2,
                linewidth=1.0, color=color)
        ax.axvline(PRODUCTION_EPOCH, color='#888888', linestyle='--',
                   linewidth=0.6, zorder=0)
        ax.set_xlabel('Training epoch', fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_xticks(np.arange(0, N_EPOCHS + 1, 10))
        ax.set_xlim(-1, N_EPOCHS)
        ax.set_title('abc'[k], loc='left', fontsize=9, fontweight='bold')
    axes[0].text(PRODUCTION_EPOCH + 0.8, 0.96, 'epoch 7\ncheckpoint',
                 transform=axes[0].get_xaxis_transform(), fontsize=6.5,
                 color='#444444', va='top', ha='left')
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--log', type=Path, default=DEFAULT_LOG,
                    help='published-run training log (Lightning progress output)')
    ap.add_argument('--check', type=Path, default=DEFAULT_CHECK,
                    help='loss_curves.csv whose published rows must match')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help='output path stem (no extension)')
    args = ap.parse_args()

    df = parse_log(args.log)
    missing = [f for f in FIELDS if f not in df.columns or df[f].isna().any()]
    if df.empty or missing:
        print('ERROR: could not parse {0} from {1}'.format(missing or FIELDS,
                                                          args.log))
        sys.exit(1)
    if not check_against(df, args.check):
        sys.exit(1)
    print('Parsed {0} epochs from {1}'.format(len(df), args.log))
    print('Cross-check against {0}: identical for all {1} fields'.format(
        args.check, len(FIELDS)))
    for f in FIELDS:
        v = df.set_index('epoch')[f]
        print('  {0:20s} epoch 0 {1:.3f} | epoch 7 {2:.3f} | epoch 39 {3:.3f} | '
              'range {4:.3f}-{5:.3f}'.format(f, v[0], v[PRODUCTION_EPOCH],
                                             v[N_EPOCHS - 1], v.min(), v.max()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(df)
    save_mpl(fig, args.out)
    plt.close(fig)
    print('wrote {0}.pdf / .png'.format(args.out))


if __name__ == '__main__':
    main()
