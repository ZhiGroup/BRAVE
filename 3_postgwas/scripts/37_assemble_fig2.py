#!/usr/bin/env python3
"""
37_assemble_fig2.py
Assemble the new 4-panel Figure 2 composite for the JAGWAS manuscript.

New Fig 2 layout (replaces old 5-panel a/b/c/d/e):
  (a) UMAP of 128-dim BREs across 16 regions      — block_fig2/a_umap.{pdf,png}
  (b) PHESANT brain-imaging (UKB 25xxx) bar       — phesant/fig2b_brain_idp_bar
  (c) PHESANT non-brain category heatmap          — phesant/fig2c_nonbrain_category_heatmap
  (d) BRE → 16 subcortical-volume R² heatmap      — block_fig2/heatmap_idp_prediction.pdf

Output: figures/main/Fig2_embedding/Fig2_embedding.{pdf,png}

Renders each panel from its source PDF (panels a + d, rasterised) and PDF
(panels b + c, vector). Panel sizes follow NComms style.
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

POST_BASE = Path(__file__).resolve().parents[1]
BLOCK_DIR = POST_BASE / "figures" / "block_fig2"
PHE_FIG  = POST_BASE / "figures" / "phesant"
OUT_DIR  = POST_BASE / "figures" / "main" / "Fig2_embedding"

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300):
    """Rasterise the first page of a PDF to PNG using PyMuPDF (fitz)."""
    if png_path.exists() and png_path.stat().st_mtime > pdf_path.stat().st_mtime:
        return png_path
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(png_path))
    doc.close()
    return png_path


def assemble(out_dir: Path):
    apply_mpl_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve panel sources
    a_png = BLOCK_DIR / "a_umap.png"
    b_png = PHE_FIG / "fig2b_brain_idp_bar.png"
    c_png = PHE_FIG / "fig2c_nonbrain_category_heatmap.png"
    d_pdf = BLOCK_DIR / "heatmap_idp_prediction.pdf"
    d_png = BLOCK_DIR / "heatmap_idp_prediction.png"

    if not d_png.exists():
        print(f"Rasterising {d_pdf.name} → {d_png.name} ...")
        _pdf_to_png(d_pdf, d_png, dpi=300)

    for p in [a_png, b_png, c_png, d_png]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Two-row layout: top row = (a UMAP, d IDP heatmap)
    #                 bottom row = (b PHESANT IDP bar, c PHESANT non-brain heatmap)
    # NComms 2-column page ~ 180 mm wide; tall layout for vertically dense panels.
    fig = plt.figure(figsize=(MM(190), MM(200)))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10,
                          left=0.04, right=0.98, top=0.97, bottom=0.04,
                          width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.05])

    # Reading-order (Z-pattern) panel layout:
    #   row 1, L→R: a (UMAP)          b (PHESANT brain-IDP)
    #   row 2, L→R: c (PHESANT NB)    d (BRE → volume R²)
    ax_a = fig.add_subplot(gs[0, 0])  # top-left
    ax_b = fig.add_subplot(gs[0, 1])  # top-right
    ax_c = fig.add_subplot(gs[1, 0])  # bottom-left
    ax_d = fig.add_subplot(gs[1, 1])  # bottom-right

    for ax, src, label in [
        (ax_a, a_png, "a"),
        (ax_b, b_png, "b"),
        (ax_c, c_png, "c"),
        (ax_d, d_png, "d"),
    ]:
        img = mpimg.imread(str(src))
        ax.imshow(img)
        ax.axis("off")
        # NComms panel label, top-left in axis coords
        ax.text(-0.04, 1.02, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", ha="left", va="bottom")

    out_stem = out_dir / "Fig2_embedding"
    save_mpl(fig, out_stem)
    plt.close(fig)
    print(f"Wrote: {out_stem}.pdf / .png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    assemble(args.out_dir)


if __name__ == "__main__":
    main()
