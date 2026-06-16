"""
fig_style.py
Centralised publication-style formatting for all post-GWAS figures.

Target journal: Nature Communications
  - Font:   Arial
  - Label size: 7 pt (tick labels, annotations), 8 pt (axis labels), 9 pt (panel titles)
  - Figure size: single column = 88 mm, double column = 180 mm
  - Resolution: ≥300 DPI (colour figures); vector PDF preferred
  - Background: white
  - Colour models: RGB (online submission)

Usage
-----
Matplotlib:
    from fig_style import apply_mpl_style, MM, FIG_DPI
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(MM(88), MM(60)))
    fig.savefig("out.pdf", dpi=FIG_DPI, bbox_inches='tight')

Plotly:
    from fig_style import plotly_layout, save_plotly
    fig.update_layout(**plotly_layout(title="My figure"))
    save_plotly(fig, FIG_DIR / "out")           # writes .pdf + .png
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Physical dimensions ───────────────────────────────────────────────────────
def MM(mm: float) -> float:
    """Convert millimetres to inches (for matplotlib figsize)."""
    return mm / 25.4

# Standard Nature Comms column widths (mm)
COL1 = 88   # single column
COL2 = 180  # double column

# ── Resolution ────────────────────────────────────────────────────────────────
FIG_DPI   = 300    # raster export DPI (matplotlib savefig)
PLT_SCALE = 3      # kaleido scale factor for Plotly (base ×3 → ≥300 DPI at COL2)

# ── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY = 'Arial'
FS_TICK     = 7    # tick labels, legend text, annotations
FS_LABEL    = 8    # axis labels, node labels in Sankey
FS_TITLE    = 9    # panel/figure titles

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COLOR    = 'white'
LINE_COLOR  = '#333333'   # axes, node outlines
GRID_COLOR  = '#DDDDDD'
NODE_COLOR  = 'rgba(190,190,190,1.0)'   # Sankey nodes (Plotly rgba string)

# ── Matplotlib global style ───────────────────────────────────────────────────
MPL_RC = {
    # Font
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'DejaVu Sans'],
    'font.size':            FS_TICK,
    'axes.titlesize':       FS_TITLE,
    'axes.labelsize':       FS_LABEL,
    'xtick.labelsize':      FS_TICK,
    'ytick.labelsize':      FS_TICK,
    'legend.fontsize':      FS_TICK,
    # Lines
    'axes.linewidth':       0.6,
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.minor.width':    0.4,
    'ytick.minor.width':    0.4,
    'xtick.major.size':     3.0,
    'ytick.major.size':     3.0,
    'lines.linewidth':      1.0,
    # Background
    'axes.facecolor':       BG_COLOR,
    'figure.facecolor':     BG_COLOR,
    'savefig.facecolor':    BG_COLOR,
    # Grid
    'axes.grid':            False,
    # Legend
    'legend.frameon':       False,
    # PDF font embedding (required for journal submission)
    'pdf.fonttype':         42,   # TrueType — avoids Type 3 fonts rejected by journals
    'ps.fonttype':          42,
}

def apply_mpl_style():
    """Apply Nature Communications rcParams globally."""
    mpl.rcParams.update(MPL_RC)


def save_mpl(fig: "plt.Figure", path_stem, tight=True, dpi=FIG_DPI):
    """Save a matplotlib figure to <path_stem>.pdf and <path_stem>.png.

    Parameters
    ----------
    path_stem : Path or str  (no extension)
    """
    from pathlib import Path
    stem = Path(path_stem)
    kw = dict(dpi=dpi, bbox_inches='tight' if tight else None)
    fig.savefig(str(stem.with_suffix('.pdf')), **kw)
    fig.savefig(str(stem.with_suffix('.png')), **kw)
    print(f"Saved: {stem}.pdf / .png")


# ── Plotly style helpers ──────────────────────────────────────────────────────
def _fmt_region_label(lbl: str) -> str:
    """Clean up FUMA folder names → readable Sankey labels."""
    lbl = lbl.replace('_', ' ')
    lbl = lbl.replace('Left ', 'L. ').replace('Right ', 'R. ')
    lbl = lbl.replace('Brain Stem or 4th Ventricle', 'Brain Stem / 4th V.')
    lbl = lbl.replace('Accumbens area', 'Accumbens')
    return lbl


def plotly_layout(title: str = '',
                  width: int = COL2 * 4,    # ~720 px base → ×3 scale = 2160 px
                  height: int = 950,
                  margin=None) -> dict:
    """Return a dict of layout kwargs for Nature Comms Plotly figures.

    Pass the result directly to fig.update_layout(**plotly_layout(...)).
    """
    if margin is None:
        margin = dict(l=60, r=200, t=75, b=55)
    return dict(
        title_text=title,
        title_font=dict(family=FONT_FAMILY, size=FS_TITLE, color='black'),
        font=dict(family=FONT_FAMILY, size=FS_LABEL, color='black'),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        width=width,
        height=height,
        margin=margin,
    )


def save_plotly(fig, path_stem, width=None, height=None,
                scale: int = PLT_SCALE):
    """Save Plotly figure to <path_stem>.pdf and <path_stem>.png via kaleido.

    Parameters
    ----------
    path_stem : Path or str  (no extension)
    width, height : override fig layout dimensions if provided
    """
    from pathlib import Path
    stem = Path(path_stem)
    kw = dict(engine='kaleido', scale=scale)
    if width is not None:
        kw['width'] = width
    if height is not None:
        kw['height'] = height
    fig.write_image(str(stem.with_suffix('.pdf')), **kw)
    fig.write_image(str(stem.with_suffix('.png')), **kw)
    print(f"Saved: {stem}.pdf / .png")
