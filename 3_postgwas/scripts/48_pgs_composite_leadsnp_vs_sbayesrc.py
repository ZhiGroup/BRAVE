#!/usr/bin/env python3
"""
48_pgs_composite_leadsnp_vs_sbayesrc.py
Composite 2-panel figure comparing the whole-genome SBayesRC PGS cross-region
R² matrix with the lead-SNP polygenic score baseline across the 16 subcortical
regions. Anchors the Results §3 claim "R² = 0.06-0.68% ... 8-fold lower
maximum and ~10-fold lower mean than the SBayesRC analysis" (Supplementary
Fig. S29).

Inputs:
  results/sbayesrc/cross_region/sbayesrc_pgs_cross_region_R2.csv
  results/pgs/cross_region_r2.csv

Output:
  figures/pgs_composite/pgs_composite_leadsnp_vs_sbayesrc.pdf/.png

Panels a (SBayesRC) and b (lead-SNP baseline) share a colorbar range so the
magnitude gap between the two PGS schemes reads visually.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa: E402

apply_mpl_style()

CANONICAL_ORDER = [
    "Brainstem", "CSF",
    "L.Thalamus", "R.Thalamus",
    "L.Caudate", "R.Caudate",
    "L.Putamen", "R.Putamen",
    "L.Pallidum", "R.Pallidum",
    "L.Hippocampus", "R.Hippocampus",
    "L.Amygdala", "R.Amygdala",
    "L.Accumbens", "R.Accumbens",
]

RENAME = {"BrainStem": "Brainstem"}


def load_matrix(csv_path: Path) -> pd.DataFrame:
    m = pd.read_csv(csv_path, index_col=0)
    m = m.rename(index=RENAME, columns=RENAME)
    missing = [r for r in CANONICAL_ORDER if r not in m.index]
    if missing:
        raise ValueError(f"{csv_path} missing rows: {missing}")
    return m.reindex(index=CANONICAL_ORDER, columns=CANONICAL_ORDER)


def draw_heatmap(ax, mat: pd.DataFrame, title: str, norm, cmap: str = "viridis"):
    im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(len(CANONICAL_ORDER)))
    ax.set_yticks(range(len(CANONICAL_ORDER)))
    ax.set_xticklabels(CANONICAL_ORDER, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(CANONICAL_ORDER, fontsize=6)
    ax.set_xlabel("Target region (top-h² BRE dim)", fontsize=8)
    ax.set_ylabel("PGS source region", fontsize=8)
    ax.set_title(title, fontsize=9)
    diag_mean = np.diag(mat.values).mean()
    diag_max = np.diag(mat.values).max()
    ax.text(0.02, 0.98, f"diag mean R²={diag_mean:.4f}\ndiag max R²={diag_max:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
    for k in range(len(CANONICAL_ORDER)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="navy", lw=0.8))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=str(SCRIPT_DIR.parents[1]))
    p.add_argument("--sbayes-csv", default=None)
    p.add_argument("--leadsnp-csv", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--vmax", type=float, default=0.06,
                   help="Colorbar maximum (default 0.06 to cover SBayesRC diag max ~0.056)")
    a = p.parse_args()

    root = Path(a.project_root)
    sbayes_csv = Path(a.sbayes_csv) if a.sbayes_csv else \
        root / "post_gwas_analysis" / "results" / "sbayesrc" / "cross_region" / \
        "sbayesrc_pgs_cross_region_R2.csv"
    leadsnp_csv = Path(a.leadsnp_csv) if a.leadsnp_csv else \
        root / "post_gwas_analysis" / "results" / "pgs" / "cross_region_r2.csv"
    out_dir = Path(a.out_dir) if a.out_dir else \
        root / "post_gwas_analysis" / "figures" / "pgs_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    sbayes = load_matrix(sbayes_csv)
    leadsnp = load_matrix(leadsnp_csv)

    norm = Normalize(vmin=0.0, vmax=a.vmax)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(MM(180), MM(90)))
    im_a = draw_heatmap(ax_a, sbayes,
                        "a  SBayesRC whole-genome PGS", norm)
    _ = draw_heatmap(ax_b, leadsnp,
                     "b  Lead-SNP polygenic score baseline", norm)

    cbar = fig.colorbar(im_a, ax=[ax_a, ax_b], shrink=0.75,
                        fraction=0.03, pad=0.02)
    cbar.set_label("Replication R²", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    save_mpl(fig, out_dir / "pgs_composite_leadsnp_vs_sbayesrc")
    plt.close(fig)

    sd = np.diag(sbayes.values)
    bd = np.diag(leadsnp.values)
    print("SBayesRC diag: mean={:.4f}, max={:.4f}, min={:.4f}".format(
        sd.mean(), sd.max(), sd.min()))
    print("Lead-SNP diag: mean={:.4f}, max={:.4f}, min={:.4f}".format(
        bd.mean(), bd.max(), bd.min()))
    print("Ratios (SBayesRC/lead-SNP): max={:.2f}x, mean={:.2f}x".format(
        sd.max()/bd.max(), sd.mean()/bd.mean()))
    print(f"Saved: {out_dir}/pgs_composite_leadsnp_vs_sbayesrc.pdf/.png")


if __name__ == "__main__":
    main()
