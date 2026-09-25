#!/usr/bin/env python3
# NOTE: paths to data, tools and model checkpoints are NOT hard-coded.
# Set them in config/paths.yaml, or fill in each inline <EXTERNAL: ...>
# marker below with a location on your own system. See docs/DATA.md.
"""
38_sbayesrc_pgs_cross_region.py
Cross-region SBayesRC PGS R² matrix — polygenic-level region specificity.

For each of the 16 SBayesRC PGS (one per region, built from each region's
top-h²-ranked BRE dimension), compute the replication R² against EACH of
the 16 regions' top-h² dimension targets. This produces a 16 × 16 matrix
of R² values.

Expected pattern (if BREs are region-specific at the polygenic level):
  - Diagonal (PGS for region X × target X): high R² (~0.028 mean from Stage C)
  - Bilateral homologue (PGS X × target X-bilateral): moderate (anatomically
    related)
  - Distinct region (PGS X × target Y): low R²

Outputs:
  results/sbayesrc/cross_region/sbayesrc_pgs_cross_region_R2.csv  (16 × 17 matrix)
  results/sbayesrc/cross_region/sbayesrc_pgs_cross_region_summary.csv  (diag/off-diag)
  figures/sbayesrc/sbayesrc_pgs_cross_region_heatmap.{pdf,png}

Compute env: <EXTERNAL: python>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fig_style import apply_mpl_style, save_mpl, MM  # noqa

PROJECT_BASE = SCRIPT_DIR.parents[1]
SWEEP_DIR = PROJECT_BASE / "post_gwas_analysis" / "results" / "sbayesrc" / "sweep"
# Canonical replication QT path per docs/data_paths.md §7a (L53)
REP_IN = Path(
    "<EXTERNAL: GWAS_DIR_pixpro>/"
    "fastgwa_pixpro/<EXTERNAL: encoder checkpoint>"
)
OUT_DIR = PROJECT_BASE / "post_gwas_analysis" / "results" / "sbayesrc" / "cross_region"
FIG_DIR = PROJECT_BASE / "post_gwas_analysis" / "figures" / "sbayesrc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Display order: roughly anatomical, bilateral pairs adjacent
REGIONS_ORDER = [
    ("Brain_Stem_or_4th_Ventricle", "Brainstem"),
    ("CSF", "CSF"),
    ("Left_Thalamus_Proper", "L.Thalamus"),
    ("Right_Thalamus-Proper", "R.Thalamus"),
    ("Left_Caudate", "L.Caudate"),
    ("Right_Caudate", "R.Caudate"),
    ("Left_Putamen", "L.Putamen"),
    ("Right_Putamen", "R.Putamen"),
    ("Left_Pallidum", "L.Pallidum"),
    ("Right_Pallidum", "R.Pallidum"),
    ("Left_Hippocampus", "L.Hippocampus"),
    ("Right_Hippocampus", "R.Hippocampus"),
    ("Left_Amygdala", "L.Amygdala"),
    ("Right_Amygdala", "R.Amygdala"),
    ("Left_Accumbens-area", "L.Accumbens"),
    ("Right_Accumbens-area", "R.Accumbens"),
]

# Top-h²-ranked dim per region (from Stage C); confirmed against h2_table.csv
TOP_DIMS = {
    "Brain_Stem_or_4th_Ventricle": 127,
    "CSF": 28,
    "Left_Thalamus_Proper": 83,
    "Right_Thalamus-Proper": 40,
    "Left_Caudate": 52,
    "Right_Caudate": 81,
    "Left_Putamen": 18,
    "Right_Putamen": 118,
    "Left_Pallidum": 11,
    "Right_Pallidum": 20,
    "Left_Hippocampus": 27,
    "Right_Hippocampus": 35,
    "Left_Amygdala": 68,
    "Right_Amygdala": 54,
    "Left_Accumbens-area": 100,
    "Right_Accumbens-area": 31,
}


def load_pgs(region):
    """Load PGS for a given region (built from its own top-h² dim FastGWA)."""
    sscore = SWEEP_DIR / f"{region}_QT{TOP_DIMS[region]}.sscore"
    pgs = pd.read_csv(sscore, sep=r"\s+", engine="python")
    pgs_col = next((c for c in pgs.columns if c.endswith("_AVG")), pgs.columns[-1])
    pgs = pgs[["IID", pgs_col]].rename(columns={pgs_col: "PGS"})
    pgs["IID"] = pgs["IID"].astype(str)
    return pgs


def load_target(region):
    """Load replication phenotype for region's top-h² dim (canonical path)."""
    qt_path = REP_IN / f"replication_<EXTERNAL: encoder checkpoint>{region}_QT{TOP_DIMS[region]}"
    qt = pd.read_csv(qt_path, sep=r"\s+", engine="python",
                     header=0, names=["FID", "IID", "PHENO"], skiprows=1)
    qt["IID"] = qt["IID"].astype(str)
    qt["PHENO"] = pd.to_numeric(qt["PHENO"], errors="coerce")
    return qt.dropna(subset=["PHENO"])[["IID", "PHENO"]]


def main():
    print("Loading 16 PGS + 16 replication targets ...")
    pgs_dict = {r: load_pgs(r) for r, _ in REGIONS_ORDER}
    tgt_dict = {r: load_target(r) for r, _ in REGIONS_ORDER}

    print("Computing 16 × 16 R² matrix ...")
    n = len(REGIONS_ORDER)
    R2_mat = np.zeros((n, n))
    r_mat = np.zeros((n, n))
    N_mat = np.zeros((n, n), dtype=int)
    for i, (pgs_reg, _) in enumerate(REGIONS_ORDER):
        for j, (tgt_reg, _) in enumerate(REGIONS_ORDER):
            df = pgs_dict[pgs_reg].merge(tgt_dict[tgt_reg], on="IID", how="inner").dropna()
            if len(df) < 100:
                R2_mat[i, j] = np.nan
                continue
            r = float(np.corrcoef(df["PGS"].astype(float),
                                  df["PHENO"].astype(float))[0, 1])
            r_mat[i, j] = r
            R2_mat[i, j] = r * r
            N_mat[i, j] = len(df)

    labels = [lab for _, lab in REGIONS_ORDER]
    R2_df = pd.DataFrame(R2_mat, index=labels, columns=labels)
    r_df = pd.DataFrame(r_mat, index=labels, columns=labels)

    R2_df.to_csv(OUT_DIR / "sbayesrc_pgs_cross_region_R2.csv")
    r_df.to_csv(OUT_DIR / "sbayesrc_pgs_cross_region_r.csv")
    print(f"\nWrote R² matrix to {OUT_DIR}/sbayesrc_pgs_cross_region_R2.csv")

    # Summary stats
    diag = np.diagonal(R2_mat)
    off_diag = R2_mat[~np.eye(n, dtype=bool)]
    summary = {
        "diag_mean_R2": float(np.mean(diag)),
        "diag_min_R2": float(np.min(diag)),
        "diag_max_R2": float(np.max(diag)),
        "off_diag_mean_R2": float(np.mean(off_diag)),
        "off_diag_min_R2": float(np.min(off_diag)),
        "off_diag_max_R2": float(np.max(off_diag)),
        "diag_to_offdiag_ratio": float(np.mean(diag) / np.mean(off_diag)),
        "n_diag_gt_max_offdiag_row": sum(
            R2_mat[i, i] >= max(R2_mat[i, j] for j in range(n) if j != i)
            for i in range(n)
        ),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / "sbayesrc_pgs_cross_region_summary.csv", index=False)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Heatmap
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(MM(170), MM(155)))
    vmax = float(np.nanpercentile(R2_mat, 99))
    im = ax.imshow(R2_mat, cmap="viridis", aspect="equal", vmin=0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Target region (top-h² BRE dim)")
    ax.set_ylabel("PGS source region")
    ax.set_title(f"SBayesRC PGS × target replication R²\n"
                 f"(diagonal mean R² = {summary['diag_mean_R2']:.4f}, "
                 f"off-diagonal mean R² = {summary['off_diag_mean_R2']:.4f}, "
                 f"ratio = {summary['diag_to_offdiag_ratio']:.2f}×)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("R²")
    # Cell values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{R2_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=5.5,
                    color="white" if R2_mat[i, j] < vmax * 0.6 else "black")
    fig.tight_layout()
    save_mpl(fig, FIG_DIR / "sbayesrc_pgs_cross_region_heatmap")
    plt.close(fig)
    print(f"\nHeatmap saved to {FIG_DIR / 'sbayesrc_pgs_cross_region_heatmap'}.pdf/.png")


if __name__ == "__main__":
    main()
