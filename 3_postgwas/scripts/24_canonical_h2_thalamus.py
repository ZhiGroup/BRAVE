"""
24_canonical_h2_thalamus.py

Estimate the *canonical heritability* of the Left Thalamus BRE embedding space.

Theory
------
For p-dimensional phenotype vector Y, the genetic covariance matrix G_g has
eigenvalues λ₁ ≥ … ≥ λₚ (canonical heritabilities).
λ_max = max linear combination of dims with highest h².
For DIPs, individual h² is low but λ_max >> mean(h²) because genetic effects
are correlated across dims.

Method
------
1. Read per-dim h² for Left_Thalamus_Proper from h2_table.csv.
2. Select top-K dims by h² (default K=20 → 190 LDSC rg pairs).
3. Run pairwise LDSC rg in parallel using pre-munged sumstats.
4. Build K×K genetic covariance matrix G_g[i,j] = rg_ij * sqrt(h²_i * h²_j).
5. Eigendecompose → λ_max, cumulative spectrum.
6. Plot: eigenvalue spectrum + G_g heatmap + comparison bar.

Outputs
-------
  results/multivariate_h2/thalamus_canonical_h2.csv
  figures/multivariate_h2/thalamus_h2_spectrum.pdf/png
  figures/multivariate_h2/thalamus_Gg_heatmap.pdf/png
"""

import argparse
import os
import subprocess
import tempfile
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations
from multiprocessing import Pool
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import apply_mpl_style, save_mpl, MM, FS_TICK, FS_LABEL

apply_mpl_style()

# ── Paths ──────────────────────────────────────────────────────────────────
MUNGED_DIR   = Path(str(cfg.postgwas.bre_munged_dir))
_SCRIPT_DIR  = Path(__file__).resolve().parent          # 3_postgwas/scripts/
_POST_DIR    = _SCRIPT_DIR.parent                       # 3_postgwas/
H2_CSV       = Path("<EXTERNAL: LDSC h2_table.csv (per-region per-dim canonical h2 table)>")
LDSC_PY      = "python"                                 # interpreter for the LDSC env
LDSC_SCRIPT  = str(Path(str(cfg.tools.ldsc_dir)) / "ldsc.py")
LD_CHR       = "<EXTERNAL: 1000 Genomes LD-score reference panel (--ref-ld-chr / --w-ld-chr)>"
BRE_PREFIX   = "discovery"
REGION_TAG   = "Left_Thalamus_Proper"

RES_DIR = _POST_DIR / "results" / "multivariate_h2"
FIG_DIR = _POST_DIR / "figures" / "multivariate_h2"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Canonical h² for Left Thalamus BRE")
    p.add_argument("--top_k",    type=int, default=20,
                   help="Number of top-h² dims to use (default 20)")
    p.add_argument("--ncpu",     type=int, default=8,
                   help="Parallel LDSC processes (default 8)")
    p.add_argument("--rg_out",   type=str, default=None,
                   help="If provided, skip rg computation and load this CSV")
    return p.parse_args()


def munged_path(dim):
    """Return path to munged sumstats for Left_Thalamus_Proper QT{dim}."""
    fname = f"{BRE_PREFIX}_{REGION_TAG}_QT{dim}.fastGWA.fastGWA.sumstats.gz"
    return MUNGED_DIR / fname


def run_ldsc_rg(args_tuple):
    """Run LDSC rg for one pair (dim_i, dim_j). Returns (i, j, rg, rg_se)."""
    dim_i, dim_j, tmp_root = args_tuple
    f1 = str(munged_path(dim_i))
    f2 = str(munged_path(dim_j))
    out_prefix = os.path.join(tmp_root, f"rg_{dim_i}_{dim_j}")
    cmd = [
        LDSC_PY, LDSC_SCRIPT,
        "--rg", f"{f1},{f2}",
        "--ref-ld-chr", LD_CHR,
        "--w-ld-chr",   LD_CHR,
        "--out", out_prefix,
        "--no-intercept",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        log_text = result.stdout + result.stderr
        # Parse rg and se from log
        match = re.search(
            r"Genetic Correlation:\s+([-\d.]+)\s+\(([-\d.]+)\)", log_text
        )
        if match:
            rg   = float(match.group(1))
            rgse = float(match.group(2))
            return (dim_i, dim_j, rg, rgse, None)
        else:
            return (dim_i, dim_j, np.nan, np.nan, "parse_fail")
    except Exception as e:
        return (dim_i, dim_j, np.nan, np.nan, str(e))


def build_Gg(dims, h2_dict, rg_df):
    """Construct genetic covariance matrix from rg pairs and h² values."""
    k = len(dims)
    G = np.zeros((k, k))
    for ii, di in enumerate(dims):
        G[ii, ii] = h2_dict[di]
    for _, row in rg_df.iterrows():
        di, dj = int(row['dim_i']), int(row['dim_j'])
        if di in dims and dj in dims:
            ii = dims.index(di)
            jj = dims.index(dj)
            rg = np.clip(row['rg'], -1.0, 1.0)
            cov = rg * np.sqrt(h2_dict[di] * h2_dict[dj])
            G[ii, jj] = cov
            G[jj, ii] = cov
    return G


def nearest_psd(A):
    """Project to nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def plot_results(dims, h2_vals, Gg, eigvals, out_prefix):
    fig = plt.figure(figsize=(180 * MM, 55 * MM))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # ── Panel A: per-dim h² and canonical eigenvalues ──────────────────────
    ax1 = fig.add_subplot(gs[0])
    cum_eigvals = np.cumsum(sorted(eigvals, reverse=True))
    ax1.bar(range(1, len(eigvals) + 1), sorted(eigvals, reverse=True),
            color="#4472C4", alpha=0.85, width=0.8)
    ax1.axhline(np.mean(h2_vals), color="#E74C3C", lw=1.2, ls="--",
                label=f"Mean univariate h²={np.mean(h2_vals):.3f}")
    ax1.axhline(max(eigvals), color="#27AE60", lw=1.2, ls=":",
                label=f"λ_max (canonical)={max(eigvals):.3f}")
    ax1.set_xlabel("Canonical dimension (ranked)", fontsize=FS_LABEL)
    ax1.set_ylabel("Canonical h²", fontsize=FS_LABEL)
    ax1.set_title("Eigenspectrum of G_g\n(Left Thalamus, top-K dims)", fontsize=9)
    ax1.legend(fontsize=6, frameon=False)
    ax1.tick_params(labelsize=FS_TICK)

    # ── Panel B: G_g heatmap ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    vmax = max(abs(Gg.min()), abs(Gg.max()))
    im = ax2.imshow(Gg, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xlabel("Dimension index", fontsize=FS_LABEL)
    ax2.set_ylabel("Dimension index", fontsize=FS_LABEL)
    ax2.set_title("Genetic covariance matrix G_g\n(Left Thalamus)", fontsize=9)
    ax2.tick_params(labelsize=FS_TICK)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label("Genetic covariance", fontsize=FS_LABEL)

    # ── Panel C: comparison bar ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    labels   = ["Mean\nunivariate\nh²", "Max\nunivariate\nh²", "λ_max\n(canonical)", "Sum h²\n(total)"]
    values   = [np.mean(h2_vals), max(h2_vals), max(eigvals), sum(h2_vals)]
    colors   = ["#95A5A6", "#7F8C8D", "#27AE60", "#4472C4"]
    bars = ax3.bar(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax3.set_ylabel("h²", fontsize=FS_LABEL)
    ax3.set_title("Multivariate h² vs univariate\n(Left Thalamus)", fontsize=9)
    ax3.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    save_mpl(fig, out_prefix)
    plt.close(fig)


def main():
    args = parse_args()

    # ── 1. Load per-dim h² for Left Thalamus ──────────────────────────────
    df_h2 = pd.read_csv(H2_CSV)
    thal  = df_h2[df_h2['reg'] == 'Left_Thalamus-Proper'].copy()
    if thal.empty:
        thal = df_h2[df_h2['reg'].str.contains('Left_Thalamus', case=False)].copy()
    print(f"Found {len(thal)} dims for Left Thalamus. Mean h²={thal['h2'].mean():.4f}")

    # select top-K by h²
    top = thal.nlargest(args.top_k, 'h2')
    dims    = sorted(top['qt'].astype(int).tolist())
    h2_dict = dict(zip(top['qt'].astype(int), top['h2']))
    h2_vals = [h2_dict[d] for d in dims]
    print(f"Top-{args.top_k} dims: {dims}")
    print(f"h² range: {min(h2_vals):.4f} – {max(h2_vals):.4f}")

    # ── 2. Run pairwise LDSC rg (or load cached) ──────────────────────────
    rg_csv = RES_DIR / "thalamus_pairwise_rg.csv"

    if args.rg_out and Path(args.rg_out).exists():
        print(f"Loading cached rg from {args.rg_out}")
        rg_df = pd.read_csv(args.rg_out)
    elif rg_csv.exists():
        print(f"Loading cached rg from {rg_csv}")
        rg_df = pd.read_csv(rg_csv)
    else:
        pairs = list(combinations(dims, 2))
        print(f"Running {len(pairs)} LDSC rg pairs with {args.ncpu} CPUs...")
        with tempfile.TemporaryDirectory() as tmpdir:
            job_args = [(di, dj, tmpdir) for di, dj in pairs]
            with Pool(args.ncpu) as pool:
                results = pool.map(run_ldsc_rg, job_args)
        rg_df = pd.DataFrame(results, columns=['dim_i', 'dim_j', 'rg', 'rg_se', 'error'])
        failed = rg_df[rg_df['error'].notna()]
        if len(failed):
            print(f"WARNING: {len(failed)} pairs failed: {failed['error'].value_counts().head()}")
        rg_df.to_csv(rg_csv, index=False)
        print(f"Saved pairwise rg to {rg_csv}")

    # ── 3. Build G_g ───────────────────────────────────────────────────────
    Gg = build_Gg(dims, h2_dict, rg_df)
    Gg = nearest_psd(Gg)  # ensure PSD (rg estimation noise can break this)

    # ── 4. Eigendecompose ──────────────────────────────────────────────────
    eigvals = np.linalg.eigvalsh(Gg)
    eigvals = np.sort(eigvals)[::-1]
    lambda_max = eigvals[0]
    total_h2   = np.sum(eigvals)

    print(f"\n=== Canonical Heritability (Left Thalamus, top-{args.top_k} dims) ===")
    print(f"  Mean univariate h²  : {np.mean(h2_vals):.4f}")
    print(f"  Max  univariate h²  : {max(h2_vals):.4f}")
    print(f"  λ_max (canonical h²): {lambda_max:.4f}")
    print(f"  Sum h² (total)      : {sum(h2_vals):.4f}")
    print(f"  tr(G_g) (PSD-corrected): {total_h2:.4f}")
    print(f"  Enrichment (λ_max / mean h²): {lambda_max / np.mean(h2_vals):.2f}×")

    # ── 5. Save summary ────────────────────────────────────────────────────
    summary = pd.DataFrame({
        'metric': ['mean_univ_h2', 'max_univ_h2', 'lambda_max', 'sum_h2', 'tr_Gg', 'enrichment_fold'],
        'value':  [np.mean(h2_vals), max(h2_vals), lambda_max,
                   sum(h2_vals), total_h2, lambda_max / np.mean(h2_vals)],
    })
    summary.to_csv(RES_DIR / "thalamus_canonical_h2.csv", index=False)

    # save eigenspectrum
    pd.DataFrame({'rank': range(1, len(eigvals)+1), 'eigenvalue': eigvals}).to_csv(
        RES_DIR / "thalamus_eigenspectrum.csv", index=False)

    # ── 6. Plot ────────────────────────────────────────────────────────────
    plot_results(dims, h2_vals, Gg, eigvals,
                 str(FIG_DIR / "thalamus_canonical_h2"))
    print(f"\nFigure saved to {FIG_DIR}/thalamus_canonical_h2.pdf/png")


if __name__ == "__main__":
    main()
