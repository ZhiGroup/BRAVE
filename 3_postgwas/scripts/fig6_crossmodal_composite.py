"""
fig6_crossmodal_composite.py
Build the 3-panel Fig. 6 composite for the JAGWAS manuscript.

Panel layout:
  Top row:
    A (left)  — OCT GC focused forest plot (17 pre-specified pairs)
                (figures/gc_oct/oct_gc_focused_s4.png)
    B (right) — Heart GC focused LA forest plot (15 pre-specified pairs)
                (figures/gc_heart/heart_gc_focused_la.png)
  Bottom row (full width):
    C         — Cross-trait Sankey (non-brain trait categories)
                (figures/cross_trait/sankey_all_categories.png)

Output:
  figures/main/Fig6_cross_modal/Fig6_cross_modal.pdf
  figures/main/Fig6_cross_modal/Fig6_cross_modal.png
  figures/main/Fig6_cross_modal.pdf
  figures/main/Fig6_cross_modal.png

Usage:
  python fig6_crossmodal_composite.py
"""

import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, MM, FIG_DPI

apply_mpl_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
FIG_SRC    = BASE_DIR / 'figures'

PANEL_A = FIG_SRC / 'gc_oct'      / 'oct_gc_focused_s4.png'
PANEL_B = FIG_SRC / 'gc_heart'    / 'heart_gc_focused_la.png'
PANEL_C = FIG_SRC / 'cross_trait' / 'sankey_all_categories.png'

OUT_DIR_PAPER = BASE_DIR / 'figures' / 'main' / 'Fig6_cross_modal'
OUT_DIR_FIG   = BASE_DIR / 'figures' / 'main'
OUT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing panel source: {path}")
    return mpimg.imread(str(path))


def main():
    parser = argparse.ArgumentParser(description='Build Fig. 6 composite')
    parser.add_argument('--dpi', type=int, default=FIG_DPI,
                        help='Output DPI (default: %(default)s)')
    args = parser.parse_args()

    img_a = load(PANEL_A)
    img_b = load(PANEL_B)
    img_c = load(PANEL_C)

    # Double-column width (180 mm); top row ~60%, bottom Sankey ~40% of height
    fig = plt.figure(figsize=(MM(180), MM(210)), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])   # full-width bottom

    panels = [(ax_a, img_a, 'A'), (ax_b, img_b, 'B'), (ax_c, img_c, 'C')]
    for ax, img, label in panels:
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.02, 1.02, label,
                transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                va='bottom', ha='right')

    # Save to paper_draft and post_gwas_analysis/figures/main
    for stem, out_dir in [('Fig6_cross_modal', OUT_DIR_PAPER),
                           ('Fig6_cross_modal', OUT_DIR_FIG)]:
        fig.savefig(str(out_dir / f'{stem}.pdf'), dpi=args.dpi, bbox_inches='tight')
        fig.savefig(str(out_dir / f'{stem}.png'), dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {out_dir / stem}.pdf")

    plt.close(fig)


if __name__ == '__main__':
    main()
