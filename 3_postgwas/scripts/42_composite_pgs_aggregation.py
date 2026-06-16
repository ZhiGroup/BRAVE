#!/usr/bin/env python3
"""
42_composite_pgs_aggregation.py
Aggregate SBayesRC PGS across top-5 h²-ranked dims per region; compare four flavors:
  (1) top-1 (rank-1)  PGS → top-1 dim target (Stage C; what's currently in the manuscript)
  (2) mean PGS over 5 dims → mean of 5 z-scored dim targets (unweighted composite)
  (3) h²-weighted mean PGS → h²-weighted mean of 5 z-scored dim targets
  (4) max PGS (rank-1 baseline; redundant with (1) but reported for completeness)

For each region, all four are computed in the replication cohort,
Pearson r and R² reported.

Outputs:
  results/sbayesrc/composite/per_region_composite_R2.csv   (one row per region × variant)
  results/sbayesrc/composite/headline_summary.csv          (best-variant per region + means)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

POST_BASE = Path(__file__).resolve().parents[1]
SWEEP_DIR = POST_BASE / "results" / "sbayesrc" / "sweep"
REP_IN = Path(str(cfg.gwas.pheno_dir)) / "replication" / "input"
H2_CSV = Path("<EXTERNAL: LDSC h2_table.csv (per-region per-dim canonical h2 table)>")
OUT_DIR = POST_BASE / "results" / "sbayesrc" / "composite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
    "Brain_Stem_or_4th_Ventricle", "CSF",
    "Left_Accumbens-area", "Right_Accumbens-area",
    "Left_Amygdala", "Right_Amygdala",
    "Left_Caudate", "Right_Caudate",
    "Left_Hippocampus", "Right_Hippocampus",
    "Left_Pallidum", "Right_Pallidum",
    "Left_Putamen", "Right_Putamen",
    "Left_Thalamus_Proper", "Right_Thalamus-Proper",
]


def get_top5_dims(h2_df, region):
    """Return list of (dim, h²) for region's top-5 h²-ranked dims."""
    h2_reg = region.replace("_Thalamus_Proper", "_Thalamus-Proper")
    sub = h2_df[h2_df["reg"] == h2_reg].sort_values("h2", ascending=False).head(5)
    return [(int(r["qt"]), float(r["h2"])) for _, r in sub.iterrows()]


def load_pgs(region, dim):
    f = SWEEP_DIR / f"{region}_QT{dim}.sscore"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep=r"\s+", engine="python")
    pgs_col = next((c for c in df.columns if c.endswith("_AVG")), df.columns[-1])
    out = df[["IID", pgs_col]].rename(columns={pgs_col: "PGS"})
    out["IID"] = out["IID"].astype(str)
    return out


def load_target(region, dim):
    f = REP_IN / f"replication_{region}_QT{dim}"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep=r"\s+", engine="python", header=0,
                     names=["FID", "IID", "PHENO"], skiprows=1)
    df["IID"] = df["IID"].astype(str)
    df["PHENO"] = pd.to_numeric(df["PHENO"], errors="coerce")
    return df.dropna(subset=["PHENO"])[["IID", "PHENO"]]


def zscore(x):
    """Z-score within a numpy array, robust to NaN."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x - mu


def r_and_R2(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 100:
        return np.nan, np.nan, np.nan, 0
    r = float(np.corrcoef(x[valid], y[valid])[0, 1])
    n = int(valid.sum())
    p = float(2 * stats.norm.sf(abs(r) * np.sqrt(n - 2)))
    return r, r * r, p, n


def main():
    h2 = pd.read_csv(H2_CSV)
    rows = []
    summary_rows = []
    for region in REGIONS:
        top5 = get_top5_dims(h2, region)
        print(f"\n=== {region} (top-5 dims: {[(d, round(h, 3)) for d, h in top5]}) ===")

        # Load all 5 PGS + targets and align on shared IID set
        pgs_dict = {dim: load_pgs(region, dim) for dim, _ in top5}
        tgt_dict = {dim: load_target(region, dim) for dim, _ in top5}
        # Find shared IIDs across all 10
        iid_sets = [set(p["IID"]) for p in pgs_dict.values() if p is not None] + \
                   [set(t["IID"]) for t in tgt_dict.values() if t is not None]
        if not iid_sets:
            print("  no data — skipping")
            continue
        shared = set.intersection(*iid_sets)
        iid_order = sorted(shared)
        print(f"  shared replication subjects: {len(iid_order):,}")

        # Stack PGS and target matrices indexed by shared IIDs
        idx = pd.Index(iid_order, name="IID")
        pgs_mat = np.column_stack([
            pgs_dict[dim].set_index("IID").reindex(idx)["PGS"].values
            for dim, _ in top5 if pgs_dict[dim] is not None
        ])
        tgt_mat = np.column_stack([
            tgt_dict[dim].set_index("IID").reindex(idx)["PHENO"].values
            for dim, _ in top5 if tgt_dict[dim] is not None
        ])
        h2_vec = np.array([h for _, h in top5])

        # Variant (1) top-h² rank-1: just dim 0
        r1, R2_1, p1, n1 = r_and_R2(pgs_mat[:, 0], tgt_mat[:, 0])
        rows.append(dict(region=region, variant="top1_h2_rank1",
                          dims=str([top5[0][0]]), r=r1, R2=R2_1, p=p1, N=n1))

        # Build composites — z-score each column then aggregate
        pgs_z = np.column_stack([zscore(pgs_mat[:, k]) for k in range(pgs_mat.shape[1])])
        tgt_z = np.column_stack([zscore(tgt_mat[:, k]) for k in range(tgt_mat.shape[1])])

        # Variant (2): unweighted mean composite
        pgs_unw = np.nanmean(pgs_z, axis=1)
        tgt_unw = np.nanmean(tgt_z, axis=1)
        r2_, R2_2, p2, n2 = r_and_R2(pgs_unw, tgt_unw)
        rows.append(dict(region=region, variant="composite_mean",
                          dims=str([d for d, _ in top5]), r=r2_, R2=R2_2, p=p2, N=n2))

        # Variant (3): h²-weighted mean composite
        w = h2_vec / h2_vec.sum()
        pgs_h2w = np.nansum(pgs_z * w[None, :], axis=1)
        tgt_h2w = np.nansum(tgt_z * w[None, :], axis=1)
        r3, R2_3, p3, n3 = r_and_R2(pgs_h2w, tgt_h2w)
        rows.append(dict(region=region, variant="composite_h2weighted",
                          dims=str([d for d, _ in top5]), r=r3, R2=R2_3, p=p3, N=n3))

        # Also: max R² across the 5 individual (PGS_k → target_k) per region
        per_dim_r2 = []
        for k in range(pgs_mat.shape[1]):
            _, R2_k, _, _ = r_and_R2(pgs_mat[:, k], tgt_mat[:, k])
            per_dim_r2.append(R2_k)
        best_rank = int(np.nanargmax(per_dim_r2)) + 1
        best_R2 = float(np.nanmax(per_dim_r2))

        print(f"  (1) top-h² rank-1   R² = {R2_1:.4f}  (dim {top5[0][0]})")
        print(f"  (2) composite_mean  R² = {R2_2:.4f}")
        print(f"  (3) composite_h2wt  R² = {R2_3:.4f}")
        print(f"  (best per-dim)      R² = {best_R2:.4f}  (rank {best_rank}, dim {top5[best_rank-1][0]})")

        summary_rows.append(dict(
            region=region, top1_R2=R2_1, composite_mean_R2=R2_2,
            composite_h2wt_R2=R2_3, best_per_dim_R2=best_R2,
            best_variant=max([
                ("top1", R2_1), ("composite_mean", R2_2),
                ("composite_h2wt", R2_3), (f"per_dim_rank{best_rank}", best_R2),
            ], key=lambda t: t[1] if np.isfinite(t[1]) else -1)[0],
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_region_composite_R2.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "headline_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("HEADLINE — per-region best PGS variant + means across 16 regions")
    print("=" * 80)
    for _, r in summary.iterrows():
        print(f"  {r['region']:32s}  top1={r['top1_R2']:.4f}  "
              f"mean={r['composite_mean_R2']:.4f}  "
              f"h2wt={r['composite_h2wt_R2']:.4f}  "
              f"best_per_dim={r['best_per_dim_R2']:.4f}  "
              f"best_variant={r['best_variant']}")
    print(f"\nMean R² across 16 regions:")
    print(f"  top-h² rank-1:       {summary['top1_R2'].mean():.4f}")
    print(f"  composite mean:      {summary['composite_mean_R2'].mean():.4f}")
    print(f"  composite h²-wt:     {summary['composite_h2wt_R2'].mean():.4f}")
    print(f"  best per-dim:        {summary['best_per_dim_R2'].mean():.4f}")
    print(f"\nBest-variant counts (across 16 regions):")
    print(summary["best_variant"].value_counts().to_string())
    print(f"\nWrote {OUT_DIR}/per_region_composite_R2.csv")
    print(f"Wrote {OUT_DIR}/headline_summary.csv")


if __name__ == "__main__":
    main()
