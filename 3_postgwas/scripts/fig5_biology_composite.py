"""
fig5_biology_composite.py
Build the 4-panel Fig. 5 composite for the JAGWAS manuscript.

Panel layout (2×2):
  A (top-left)    — matched-region structural delta barh
                    (post_gwas_analysis/figures/celltype/matched_region_structural_delta.png)
  B (top-right)   — structural vs neuronal scatter (r=−0.998)
                    (post_gwas_analysis/figures/celltype/structural_vs_neuronal_scatter.png)
  C (bottom-left) — GOBP cross-region heatmap (379 terms × 16 regions)
                    (post_gwas_analysis/figures/geneset/gobp_heatmap.png)
  D (bottom-right)— cross-trait GWAS Catalog Sankey, BRAIN / NEUROIMAGING traits ONLY
                    (positive-control panel; non-brain Sankey lives in Fig. 6C)
                    (post_gwas_analysis/figures/cross_trait/sankey_brain_only.png)

Output:
  figures/main/Fig5_biology/Fig5_biology.pdf
  figures/main/Fig5_biology/Fig5_biology.png

Usage:
  python fig5_biology_composite.py
"""

import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, MM, FIG_DPI

apply_mpl_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
FIG_SRC    = BASE_DIR / 'figures'

PANEL_A = FIG_SRC / 'celltype'    / 'matched_region_structural_delta.png'
PANEL_B = FIG_SRC / 'celltype'    / 'structural_vs_neuronal_scatter.png'
PANEL_C = FIG_SRC / 'geneset'     / 'gobp_heatmap.png'
PANEL_D = FIG_SRC / 'cross_trait' / 'sankey_brain_only.png'

OUT_DIR = BASE_DIR / 'figures' / 'main' / 'Fig5_biology'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing panel source: {path}")
    return mpimg.imread(str(path))


def main():
    parser = argparse.ArgumentParser(description='Build Fig. 5 composite')
    parser.add_argument('--dpi', type=int, default=FIG_DPI,
                        help='Output DPI (default: %(default)s)')
    args = parser.parse_args()

    img_a = load(PANEL_A)
    img_b = load(PANEL_B)
    img_c = load(PANEL_C)
    img_d = load(PANEL_D)

    # Double-column width (180 mm), height scaled for 2-row layout
    fig, axes = plt.subplots(
        2, 2,
        figsize=(MM(180), MM(160)),
        constrained_layout=True,
    )

    panel_labels = ['A', 'B', 'C', 'D']
    images       = [img_a, img_b, img_c, img_d]

    for ax, img, label in zip(axes.flat, images, panel_labels):
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.02, 1.02, label,
                transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                va='bottom', ha='right')

    # Save
    out_pdf = OUT_DIR / 'Fig5_biology.pdf'
    out_png = OUT_DIR / 'Fig5_biology.png'
    fig.savefig(str(out_pdf), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(str(out_png), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == '__main__':
    main()
