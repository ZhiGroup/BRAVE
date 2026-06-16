#!/usr/bin/env python3
"""
30_pops_per_region.py
PoPS (Polygenic Priority Score) [Weeks 2023] per JAGWAS region.

For each of the 16 JAGWAS regions, runs PoPS using the per-region MAGMA
gene-level z-scores produced by FUMA as the GWAS input + the PoPS feature
matrix as predictors. Outputs a ranked gene list per region; a "high-confidence
effector gene" shortlist is then defined as genes that are (a) FUMA-positionally
mapped to a JAGWAS aggregate locus AND (b) top-N PoPS-ranked within that locus
window.

Inputs:
  - PoPS tool:        third_party/pops/pops.py (bundled in this repo)
  - PoPS features:    <EXTERNAL: PoPS munged feature matrix prefix> .{mat.N.npy, cols.N.txt, rows.txt}
  - PoPS gene_annot:  third_party/pops/example/data/utils/gene_annot_jun10.txt
  - Per-region MAGMA: <cfg.postgwas.fuma_dir>/<region>/magma.genes.{out,raw}
  - Control features: third_party/pops/example/data/utils/features_jul17_control.txt
  - Aggregate loci:   <EXTERNAL: TableS1 aggregate 276-loci CSV> (for locus-window matching)

Per-region PoPS run produces (in OUT_DIR / <region> /):
  <region>.preds      — PoPS score per gene (ENSGID, PoPS_Score, ...)
  <region>.coefs      — fitted ridge coefficients per feature
  <region>.marginals  — per-feature marginal association table

Aggregate output (results/pops/):
  per_region_pops_scores.csv   — long-format: region × gene × PoPS_Score
  high_confidence_genes.csv    — per-locus top-PoPS positional-mapped gene
                                  (defines "high-confidence effector gene" shortlist)

Compute environment:
  Python 3.7. Verified compatible with PoPS pinned deps via example-data smoke test.

Reference:
  Weeks, E. M. et al. Leveraging polygenic enrichments of gene features to
  predict genes underlying complex traits and diseases. Nat. Genet. 55,
  1267–1276 (2023). doi:10.1038/s41588-023-01443-6
"""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

# ── Defaults (override via argparse) ─────────────────────────────────────────
# The PoPS tool is bundled in this repo under third_party/pops/.
POPS_REPO_DEFAULT = Path(__file__).resolve().parents[2] / "third_party" / "pops"
POPS_FEATURE_MUNGED_DEFAULT = Path(
    "<EXTERNAL: PoPS munged feature matrix prefix (pops_features.{mat.N.npy,cols.N.txt,rows.txt})>"
)
GENE_ANNOT_DEFAULT = POPS_REPO_DEFAULT / "example/data/utils/gene_annot_jun10.txt"
CONTROL_FEATURES_DEFAULT = POPS_REPO_DEFAULT / "example/data/utils/features_jul17_control.txt"
FUMA_BASE_DEFAULT = Path(str(cfg.postgwas.fuma_dir))
AGG_LOCI_CSV_DEFAULT = Path(
    "<EXTERNAL: TableS1 aggregate 276-loci CSV (locus-window matching)>"
)
OUT_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "pops"
PYTHON_DEFAULT = "python"

# All 16 regions (FUMA folder names)
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


def run_pops_one_region(
    region: str,
    fuma_base: Path,
    out_dir: Path,
    pops_repo: Path,
    python_bin: str,
    feature_mat_prefix: Path,
    gene_annot: Path,
    control_features: Path,
    num_chunks: int,
):
    """Run PoPS for one region; writes <region>.preds/.coefs/.marginals to out_dir/<region>/."""
    magma_prefix = fuma_base / region / "magma"
    if not magma_prefix.with_suffix(".genes.out").exists():
        print(f"  SKIP {region}: magma.genes.out missing")
        return None

    region_out = out_dir / region
    region_out.mkdir(parents=True, exist_ok=True)
    out_prefix = region_out / region

    cmd = [
        python_bin,
        str(pops_repo / "pops.py"),
        "--gene_annot_path", str(gene_annot),
        "--feature_mat_prefix", str(feature_mat_prefix),
        "--num_feature_chunks", str(num_chunks),
        "--magma_prefix", str(magma_prefix),
        "--control_features_path", str(control_features),
        "--out_prefix", str(out_prefix),
        "--verbose",
    ]
    print(f"  Running PoPS for {region} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ {region} FAILED")
        print("  stderr:", result.stderr[-500:])
        return None
    return out_prefix.with_suffix(".preds")


def aggregate_per_region(out_dir: Path) -> pd.DataFrame:
    """Concatenate per-region .preds into a long-format dataframe."""
    rows = []
    for r in REGIONS:
        preds_path = out_dir / r / f"{r}.preds"
        if not preds_path.exists():
            print(f"  SKIP aggregation for {r}: preds missing")
            continue
        df = pd.read_csv(preds_path, sep="\t")
        df["region"] = r
        rows.append(df[["region", "ENSGID", "PoPS_Score"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def define_high_confidence_genes(
    per_region_pops: pd.DataFrame,
    agg_loci_csv: Path,
    fuma_base: Path,
    top_n_per_locus: int = 5,
) -> pd.DataFrame:
    """For each aggregate locus × region, compute the top-N PoPS genes among
    FUMA-mapped genes (positional ∪ eQTL ∪ ciMap), annotate each gene with the
    locus-based method(s) that also nominate it, and flag agreement with the
    closest-gene method ("PoPS + local", Weeks 2023 §"Combining PoPS with
    locus-based methods", the recommended primary use case).

    Per-locus per-region output columns:
      al_id, chr, start, end, region, ensg, symbol, pops_score, pops_rank,
      is_closest_gene, in_positional, in_eqtl, in_ciMap, n_methods_agree,
      tier  (1 = PoPS top-1 ∩ closest gene; 2 = PoPS top-1 only with FUMA support;
             3 = PoPS in top-N with FUMA support)
    """
    agg = pd.read_csv(agg_loci_csv)
    leadsnps_cache = {}  # region_folder → leadSNPs df (FUMA)
    hits = []
    for _, locus in agg.iterrows():
        chr_, start, end = locus["chr"], locus["start"], locus["end"]
        regions_in_locus = (locus["regions"] or "").split(";")
        for region_label in regions_in_locus:
            region_label = region_label.strip()
            if not region_label:
                continue
            region_folder = _display_to_folder(region_label)
            if region_folder not in REGIONS:
                continue
            genes_path = fuma_base / region_folder / "genes.txt"
            if not genes_path.exists():
                continue
            # All FUMA-mapped genes in the locus window (union of positional + eQTL + ciMap)
            genes_df = pd.read_csv(genes_path, sep="\t")
            in_window = (
                (genes_df["chr"].astype(str) == str(chr_))
                & (genes_df["start"] >= start)
                & (genes_df["end"] <= end)
            )
            window_genes = genes_df[in_window].copy()
            if window_genes.empty:
                continue
            window_genes["in_positional"] = (window_genes["posMapSNPs"].fillna(0) > 0).astype(int)
            window_genes["in_eqtl"] = (window_genes["eqtlMapSNPs"].fillna(0) > 0).astype(int)
            ci_col = "ciMap" if "ciMap" in window_genes.columns else None
            window_genes["in_ciMap"] = ((window_genes[ci_col] == "Yes").astype(int)
                                       if ci_col else 0)
            any_method = (window_genes["in_positional"] + window_genes["in_eqtl"] + window_genes["in_ciMap"]) > 0
            window_genes = window_genes[any_method]
            if window_genes.empty:
                continue

            # PoPS scores for this region's full gene universe
            pops_region = per_region_pops[per_region_pops["region"] == region_folder]
            merged = window_genes.merge(
                pops_region, left_on="ensg", right_on="ENSGID", how="left"
            )
            merged = merged.dropna(subset=["PoPS_Score"])
            if merged.empty:
                continue
            merged = merged.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
            merged["pops_rank"] = merged.index + 1

            # Closest-gene determination using FUMA leadSNPs.txt:
            # For each lead SNP in this region's leadSNPs.txt that falls in the locus window,
            # the closest gene is the gene with minimum distance to the SNP pos.
            if region_folder not in leadsnps_cache:
                lead_path = fuma_base / region_folder / "leadSNPs.txt"
                leadsnps_cache[region_folder] = (
                    pd.read_csv(lead_path, sep="\t") if lead_path.exists()
                    else pd.DataFrame(columns=["chr", "pos", "rsID"])
                )
            leads = leadsnps_cache[region_folder]
            leads_in_locus = leads[
                (leads["chr"].astype(str) == str(chr_))
                & (leads["pos"] >= start) & (leads["pos"] <= end)
            ]
            closest_ensgs = set()
            for _, ls in leads_in_locus.iterrows():
                p = ls["pos"]
                d = merged.apply(
                    lambda r: 0 if r["start"] <= p <= r["end"]
                    else min(abs(r["start"] - p), abs(r["end"] - p)), axis=1,
                )
                if len(d) > 0:
                    closest_ensgs.add(merged.iloc[d.idxmin()]["ensg"])
            merged["is_closest_gene"] = merged["ensg"].isin(closest_ensgs).astype(int)
            merged["n_methods_agree"] = (
                merged["in_positional"] + merged["in_eqtl"]
                + merged["in_ciMap"] + merged["is_closest_gene"]
            )

            # Tier assignment per Weeks 2023 "PoPS + local" recommendation
            def _tier(row):
                if row["pops_rank"] == 1 and row["is_closest_gene"] == 1:
                    return 1  # strongest: top-1 PoPS AND closest gene
                if row["pops_rank"] == 1 and row["n_methods_agree"] >= 1:
                    return 2  # top-1 PoPS + ≥1 locus-method support
                if row["pops_rank"] <= top_n_per_locus and row["n_methods_agree"] >= 1:
                    return 3  # PoPS in top-N + ≥1 locus-method support
                return 4      # FUMA-supported but not in top-N PoPS

            merged["tier"] = merged.apply(_tier, axis=1)
            sub = merged[merged["pops_rank"] <= top_n_per_locus][[
                "ensg", "symbol", "PoPS_Score", "pops_rank",
                "is_closest_gene", "in_positional", "in_eqtl", "in_ciMap",
                "n_methods_agree", "tier",
            ]].rename(columns={"PoPS_Score": "pops_score"})
            for _, g in sub.iterrows():
                hits.append({
                    "al_id": locus["al_id"], "chr": chr_, "start": start, "end": end,
                    "region": region_folder,
                    **g.to_dict(),
                })
    return pd.DataFrame(hits)


def _display_to_folder(display: str) -> str:
    """Map 'L. Putamen' → 'Left_Putamen', etc."""
    mapping = {
        "Brain Stem / 4th V.": "Brain_Stem_or_4th_Ventricle",
        "CSF": "CSF",
        "L. Accumbens": "Left_Accumbens-area",
        "R. Accumbens": "Right_Accumbens-area",
        "L. Amygdala": "Left_Amygdala", "R. Amygdala": "Right_Amygdala",
        "L. Caudate": "Left_Caudate", "R. Caudate": "Right_Caudate",
        "L. Hippocampus": "Left_Hippocampus", "R. Hippocampus": "Right_Hippocampus",
        "L. Pallidum": "Left_Pallidum", "R. Pallidum": "Right_Pallidum",
        "L. Putamen": "Left_Putamen", "R. Putamen": "Right_Putamen",
        "L. Thalamus": "Left_Thalamus_Proper", "R. Thalamus": "Right_Thalamus-Proper",
    }
    return mapping.get(display, display)


def main():
    p = argparse.ArgumentParser(description="Run PoPS per JAGWAS region.")
    p.add_argument("--pops-repo", type=Path, default=POPS_REPO_DEFAULT)
    p.add_argument("--feature-mat-prefix", type=Path, default=POPS_FEATURE_MUNGED_DEFAULT)
    p.add_argument("--gene-annot", type=Path, default=GENE_ANNOT_DEFAULT)
    p.add_argument("--control-features", type=Path, default=CONTROL_FEATURES_DEFAULT)
    p.add_argument("--num-chunks", type=int, required=True,
                   help="Number of feature chunks produced by munge_feature_directory.py")
    p.add_argument("--fuma-base", type=Path, default=FUMA_BASE_DEFAULT)
    p.add_argument("--agg-loci-csv", type=Path, default=AGG_LOCI_CSV_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--python-bin", default=PYTHON_DEFAULT)
    p.add_argument("--top-n-per-locus", type=int, default=3,
                   help="Number of top PoPS-ranked positional genes per locus")
    p.add_argument("--only", nargs="+", default=None,
                   help="Optional subset of regions to run (folder names)")
    args = p.parse_args()

    regions = args.only if args.only else REGIONS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PoPS per-region run — {len(regions)} regions")
    print(f"  PoPS repo:          {args.pops_repo}")
    print(f"  Feature mat prefix: {args.feature_mat_prefix}")
    print(f"  Num feature chunks: {args.num_chunks}")
    print(f"  FUMA base:          {args.fuma_base}")
    print(f"  Output:             {args.out_dir}")
    print()

    succeeded = []
    for r in regions:
        out = run_pops_one_region(
            r, args.fuma_base, args.out_dir, args.pops_repo, args.python_bin,
            args.feature_mat_prefix, args.gene_annot, args.control_features,
            args.num_chunks,
        )
        if out is not None:
            succeeded.append(r)

    print(f"\nPoPS completed for {len(succeeded)}/{len(regions)} regions")

    # Aggregate per-region PoPS scores
    print("\nAggregating per-region PoPS scores ...")
    per_region_pops = aggregate_per_region(args.out_dir)
    if per_region_pops.empty:
        print("  No PoPS results to aggregate.")
        return
    per_region_pops_path = args.out_dir / "per_region_pops_scores.csv"
    per_region_pops.to_csv(per_region_pops_path, index=False)
    print(f"  Wrote {per_region_pops_path} ({len(per_region_pops):,} region×gene rows)")

    # Define high-confidence effector genes
    print("\nDefining high-confidence effector genes ...")
    hc_genes = define_high_confidence_genes(
        per_region_pops, args.agg_loci_csv, args.fuma_base, args.top_n_per_locus,
    )
    hc_path = args.out_dir / "high_confidence_genes.csv"
    hc_genes.to_csv(hc_path, index=False)
    print(f"  Wrote {hc_path}: {len(hc_genes)} locus-region-gene high-confidence triples")
    print(f"  Unique genes: {hc_genes['ensg'].nunique()}")
    print(f"  Unique loci with at least one HC gene: {hc_genes['al_id'].nunique()}")


if __name__ == "__main__":
    main()
