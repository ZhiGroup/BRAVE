#!/usr/bin/env python
"""
Fig1_framework.py
-----------------
Figure 1: RASVE framework overview — methods-only schematic.

Panels:
  A  Cohort design  (UK Biobank → discovery + replication → 16 regions)
  B  RASVE contrastive voxel training
  C  Two phenotype paths  (BRE → JAGWAS  vs  scalar IDP → FastGWA)
  D  Downstream analysis workflow  (no results, pipeline only)

Output:
  figures/main/Fig1_framework/Fig1_framework.pdf
  figures/main/Fig1_framework/Fig1_framework.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, MM, FIG_DPI

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR   = SCRIPT_DIR.parent / 'figures' / 'main' / 'Fig1_framework'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BRE    = '#2166AC'   # blue  — BRE / JAGWAS path
C_SCALAR = '#B2182B'   # red   — scalar IDP / FastGWA path
C_TRAIN  = '#4D9221'   # green — RASVE training
C_DOWN   = '#762A83'   # purple — downstream analyses
C_BOX    = '#F7F7F7'   # light grey box fill
C_ARROW  = '#333333'   # arrow / border colour
C_MRI    = '#E0E0E0'   # neutral grey for shared MRI nodes


# ── Drawing helpers ───────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, text,
         fc=C_BOX, ec=C_ARROW, lw=0.8,
         fs=7, bold=False, tc='black',
         pad=0.03, radius=0.05):
    """Rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transData, clip_on=False, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            color=tc, zorder=3,
            multialignment='center')


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=0.9, style='->'):
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=8),
                zorder=2)


def label(ax, txt, fs=9, bold=True):
    """Panel label (A, B, C, D) in top-left corner."""
    ax.text(-0.04, 1.06, txt, transform=ax.transAxes,
            fontsize=fs, fontweight='bold' if bold else 'normal',
            va='top', ha='left', color='black')


def clean_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


# ── Panel A — Cohort design ───────────────────────────────────────────────────

def draw_panel_a(ax):
    clean_ax(ax)
    label(ax, 'A')

    # UK Biobank MRI box
    rbox(ax, 0.01, 0.2, 0.18, 0.6,
         'UK Biobank\nT1-weighted MRI',
         fc='#E8F4FD', ec=C_MRI, lw=1.0, fs=7)

    # Split arrow → two cohorts
    ax.annotate('', xy=(0.28, 0.72), xytext=(0.20, 0.72),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.9,
                                mutation_scale=8))
    ax.annotate('', xy=(0.28, 0.32), xytext=(0.20, 0.32),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.9,
                                mutation_scale=8))
    # vertical connector
    ax.plot([0.20, 0.20], [0.32, 0.72], color=C_ARROW, lw=0.9, zorder=2)

    # Discovery cohort
    rbox(ax, 0.28, 0.58, 0.22, 0.28,
         'Discovery\nN = 22,878',
         fc='#EFF3FF', ec=C_BRE, lw=1.0, fs=7, bold=False)

    # Replication cohort
    rbox(ax, 0.28, 0.18, 0.22, 0.28,
         'Replication\nN = 12,359',
         fc='#FFF5EB', ec='#FC8D59', lw=1.0, fs=7, bold=False)

    # Arrows → atlas
    arrow(ax, 0.51, 0.72, 0.60, 0.72)
    arrow(ax, 0.51, 0.32, 0.60, 0.32)
    ax.plot([0.60, 0.60], [0.32, 0.72], color=C_ARROW, lw=0.9, zorder=2)
    arrow(ax, 0.60, 0.52, 0.63, 0.52)

    # 16 brain regions box
    rbox(ax, 0.63, 0.15, 0.35, 0.70,
         '16 brain regions\n─────────────\n14 bilateral\nsubcortical structures\n+  brainstem\n+  CSF',
         fc='#F0F0F0', ec='#555555', lw=1.0, fs=6.5, bold=False)

    ax.set_title('Cohort and brain atlas', fontsize=8, pad=4)


# ── Panel B — RASVE contrastive training ─────────────────────────────────────

def draw_panel_b(ax):
    clean_ax(ax)
    label(ax, 'B')

    # Training pipeline: left to right
    boxes = [
        (0.02, 0.60, 0.14, 0.28, 'T1-MRI\nscan',           '#E8F4FD', C_MRI),
        (0.22, 0.60, 0.18, 0.28, '3D overlapping\npatches\n(96×96×96)',  '#E8F4FD', C_TRAIN),
        (0.47, 0.60, 0.20, 0.28, 'FPN encoder\n(3D ResNet-18)',          '#EDF8E9', C_TRAIN),
        (0.74, 0.60, 0.22, 0.28, '128-dim\nvoxel embeddings',           '#EDF8E9', C_TRAIN),
    ]
    for (x, y, w, h, txt, fc, ec) in boxes:
        rbox(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.0, fs=6.5)

    # Arrows between boxes
    arrow(ax, 0.17, 0.74, 0.22, 0.74, color=C_ARROW)
    arrow(ax, 0.41, 0.74, 0.47, 0.74, color=C_ARROW)
    arrow(ax, 0.68, 0.74, 0.74, 0.74, color=C_ARROW)

    # Contrastive loss annotation below
    rbox(ax, 0.10, 0.08, 0.38, 0.38,
         'Positive pairs\n(overlapping patches,\nsame scan)',
         fc='#F7FCF5', ec=C_TRAIN, lw=0.8, fs=6.5)

    rbox(ax, 0.54, 0.08, 0.38, 0.38,
         'Negative pairs\n(non-overlapping\nregions)',
         fc='#FFF5F0', ec='#FC8D59', lw=0.8, fs=6.5)

    # InfoNCE label
    ax.text(0.50, 0.51, 'InfoNCE contrastive loss', ha='center', va='center',
            fontsize=6.5, style='italic', color='#555555')
    ax.annotate('', xy=(0.29, 0.46), xytext=(0.42, 0.51),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7,
                                mutation_scale=7))
    ax.annotate('', xy=(0.73, 0.46), xytext=(0.60, 0.51),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.7,
                                mutation_scale=7))

    ax.set_title('Contrastive voxel training (RASVE)', fontsize=8, pad=4)


# ── Panel C — Two phenotype paths ────────────────────────────────────────────

def draw_panel_c(ax):
    clean_ax(ax)
    label(ax, 'C')

    # Shared MRI node
    rbox(ax, 0.01, 0.35, 0.13, 0.30,
         'T1-MRI', fc='#E8F4FD', ec=C_MRI, lw=1.0, fs=7)

    # ── BRE path (top, blue) ──────────────────────────────────────────────────
    y_bre = 0.64
    h = 0.22
    rbox(ax, 0.20, y_bre, 0.18, h,
         '4D tensor\n(R×X×Y×Z×128)',
         fc='#EFF3FF', ec=C_BRE, lw=1.0, fs=6.5)
    rbox(ax, 0.44, y_bre, 0.16, h,
         'Mean-pool\nwithin mask',
         fc='#EFF3FF', ec=C_BRE, lw=1.0, fs=6.5)
    rbox(ax, 0.66, y_bre, 0.14, h,
         'BRE\n(R × 128)',
         fc='#C6DBEF', ec=C_BRE, lw=1.2, fs=6.5, bold=True, tc=C_BRE)
    rbox(ax, 0.86, y_bre, 0.13, h,
         'JAGWAS\nχ²(128)',
         fc='#084594', ec=C_BRE, lw=1.2, fs=6.5, bold=True, tc='white')

    # BRE path arrows
    arrow(ax, 0.15, 0.56, 0.20, y_bre + h/2, color=C_BRE)
    arrow(ax, 0.39, y_bre + h/2, 0.44, y_bre + h/2, color=C_BRE)
    arrow(ax, 0.61, y_bre + h/2, 0.66, y_bre + h/2, color=C_BRE)
    arrow(ax, 0.81, y_bre + h/2, 0.86, y_bre + h/2, color=C_BRE)

    # BRE path label
    ax.text(0.57, 0.92, 'Brain-region embedding (BRE) path',
            ha='center', va='center', fontsize=7,
            color=C_BRE, fontweight='bold')

    # ── Scalar path (bottom, red) ─────────────────────────────────────────────
    y_sc = 0.14
    rbox(ax, 0.20, y_sc, 0.18, h,
         'Atlas\nparcellation',
         fc='#FFF5EB', ec=C_SCALAR, lw=1.0, fs=6.5)
    rbox(ax, 0.44, y_sc, 0.16, h,
         'Scalar IDP\n(R × 1)',
         fc='#FCBBA1', ec=C_SCALAR, lw=1.2, fs=6.5, bold=True, tc=C_SCALAR)
    rbox(ax, 0.66, y_sc, 0.14, h,
         'FastGWA\nunivariate',
         fc='#67000D', ec=C_SCALAR, lw=1.2, fs=6.5, bold=True, tc='white')

    # Scalar path arrows
    arrow(ax, 0.15, 0.44, 0.20, y_sc + h/2, color=C_SCALAR)
    arrow(ax, 0.39, y_sc + h/2, 0.44, y_sc + h/2, color=C_SCALAR)
    arrow(ax, 0.61, y_sc + h/2, 0.66, y_sc + h/2, color=C_SCALAR)

    # Scalar path label
    ax.text(0.50, 0.08, 'Scalar IDP path',
            ha='center', va='center', fontsize=7,
            color=C_SCALAR, fontweight='bold')

    # Vertical connector from MRI node
    ax.plot([0.15, 0.15], [0.44, 0.56], color=C_ARROW, lw=0.9, zorder=2)

    ax.set_title('Two phenotype derivation paths', fontsize=8, pad=4)


# ── Panel D — Downstream analysis workflow ────────────────────────────────────

def draw_panel_d(ax):
    clean_ax(ax)
    label(ax, 'D')

    steps = [
        ('① Replication\n(N = 12,359)',      '#EFF3FF', C_BRE),
        ('② Novelty vs\nprior GWAS',          '#EFF3FF', C_BRE),
        ('③ Cell-type\nenrichment\n(scRNA-seq)', '#F7FCF5', C_TRAIN),
        ('④ GO Biological\nProcess\nenrichment', '#F7FCF5', C_TRAIN),
        ('⑤ Cross-modal\ngenetic\ncorrelation', '#F2F0F7', C_DOWN),
    ]

    n = len(steps)
    box_w = 0.16
    gap   = (1.0 - n * box_w) / (n + 1)
    y0, bh = 0.12, 0.76

    for idx, (txt, fc, ec) in enumerate(steps):
        x0 = gap + idx * (box_w + gap)
        rbox(ax, x0, y0, box_w, bh, txt,
             fc=fc, ec=ec, lw=1.0, fs=6.5)
        if idx < n - 1:
            ax.annotate('',
                        xy=(x0 + box_w + gap, y0 + bh / 2),
                        xytext=(x0 + box_w, y0 + bh / 2),
                        arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                        lw=0.9, mutation_scale=8))

    # "Loci" input on the left
    ax.text(0.01, 0.50, 'Identified\nloci', ha='center', va='center',
            fontsize=6.5, color='#444444')
    ax.annotate('', xy=(gap, y0 + bh / 2), xytext=(0.07, y0 + bh / 2),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.9,
                                mutation_scale=8))

    ax.set_title('Post-GWAS analysis workflow', fontsize=8, pad=4)


# ── Assemble figure ───────────────────────────────────────────────────────────

def build(out_stem):
    apply_mpl_style()

    fig = plt.figure(figsize=(MM(180), MM(200)))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 1.6, 0.9],
        hspace=0.50, wspace=0.32,
        left=0.05, right=0.97,
        top=0.96,  bottom=0.04,
    )

    ax_a = fig.add_subplot(gs[0, :])   # full-width top
    ax_b = fig.add_subplot(gs[1, 0])   # left middle
    ax_c = fig.add_subplot(gs[1, 1])   # right middle
    ax_d = fig.add_subplot(gs[2, :])   # full-width bottom

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)

    for fmt in ('pdf', 'png'):
        path = OUT_DIR / f'{out_stem}.{fmt}'
        fig.savefig(str(path), dpi=FIG_DPI, bbox_inches='tight',
                    facecolor='white')
        print(f'Saved: {path}')

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Generate Fig1 framework schematic')
    p.add_argument('--out', default='Fig1_framework',
                   help='Output file stem (default: Fig1_framework)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.out)
