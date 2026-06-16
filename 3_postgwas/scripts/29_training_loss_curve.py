"""
29_training_loss_curve.py
Supplementary Fig. S40 — Training loss vs step for the BRE encoder
(production encoder run; the epoch-7 checkpoint is used downstream).

Loss values are parsed from per-epoch checkpoint filenames at
<encoder weights dir>/iteration_epoch_{N}_tier_{M}_..._loss_{a}_{b}_{c}.pt
where (a, b, c) = (total, voxel-InfoNCE, instance-InfoNCE) and M is the
cumulative iteration/step counter printed at the console during training.

Outputs:
  figures/training_loss/S40_training_loss.{pdf,png}

CLAIM: M3 BRE encoder training (supplementary backing for the "loss plateaued
by epoch 7" claim).
RUN:   python 29_training_loss_curve.py
"""
import argparse
import re
import sys
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM

apply_mpl_style()

CKPT_DIR = Path(str(cfg.embedding.weights_dir))
PRODUCTION_EPOCH = 7  # the BRE-extraction checkpoint
DEFAULT_OUT = Path(__file__).resolve().parents[1] / \
    'figures' / 'training_loss' / 'S40_training_loss'

# filename pattern: iteration_epoch_<N>_tier_<M>_model_4_exper_3_lr_<x>_loss_<total>_<voxel>_<inst>.pt
PATTERN = re.compile(
    r'iteration_epoch_(\d+)_tier_(\d+)_model_4_exper_3_lr_\d+_'
    r'loss_([\d.]+)_([\d.]+)_([\d.]+)\.pt$'
)


def collect_loss_trace():
    rows = []
    for f in sorted(CKPT_DIR.glob('iteration_epoch_*_tier_*_loss_*.pt')):
        m = PATTERN.match(f.name)
        if not m:
            continue
        epoch, tier, total, voxel, inst = m.groups()
        rows.append({
            'epoch': int(epoch),
            'step': int(tier),
            'total_loss': float(total),
            'voxel_loss': float(voxel),
            'instance_loss': float(inst),
        })
    # Sort by epoch
    rows.sort(key=lambda r: r['epoch'])
    return rows


def build_figure(rows):
    epochs = np.array([r['epoch'] for r in rows])
    steps  = np.array([r['step'] for r in rows])
    total  = np.array([r['total_loss'] for r in rows])
    voxel  = np.array([r['voxel_loss'] for r in rows])
    inst   = np.array([r['instance_loss'] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(MM(180), MM(55)), sharex=True)
    palette = ('#2B3A55', '#C44E52', '#55A868')

    for ax, y, label, color in zip(
        axes,
        (total, voxel, inst),
        ('Total loss', 'Voxel-level InfoNCE', 'Instance-level InfoNCE'),
        palette,
    ):
        ax.plot(epochs, y, marker='o', markersize=3, linewidth=1.1, color=color)
        ax.axvline(PRODUCTION_EPOCH, color='#888888', linestyle='--', linewidth=0.6, zorder=0)
        ax.set_xlabel('Training epoch', fontsize=7)
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(axis='both', labelsize=6.5)
        ax.set_xticks(np.arange(0, max(epochs)+1, 5))

    axes[0].annotate(
        f'epoch {PRODUCTION_EPOCH}\n(production)',
        xy=(PRODUCTION_EPOCH, total[epochs == PRODUCTION_EPOCH][0]),
        xytext=(PRODUCTION_EPOCH + 1.5, total.max() - 0.2),
        fontsize=6, color='#444444',
        arrowprops=dict(arrowstyle='->', color='#888888', linewidth=0.6),
    )

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help='Output path stem (no extension)')
    args = ap.parse_args()

    rows = collect_loss_trace()
    if not rows:
        print('ERROR: no checkpoint filenames matched the pattern.')
        sys.exit(1)
    print(f'Parsed {len(rows)} epoch checkpoints:')
    print(f'  epoch range: {rows[0]["epoch"]} → {rows[-1]["epoch"]}')
    print(f'  step range:  {rows[0]["step"]} → {rows[-1]["step"]}')
    print(f'  total loss range: {min(r["total_loss"] for r in rows):.3f} – {max(r["total_loss"] for r in rows):.3f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(rows)
    save_mpl(fig, args.out)
    plt.close(fig)


if __name__ == '__main__':
    main()
