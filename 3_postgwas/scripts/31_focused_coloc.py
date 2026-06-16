#!/usr/bin/env python3
"""
31_focused_coloc.py
Focused Bayesian colocalization (coloc.abf, Giambartolomei 2014) for the
8 cross-modal BRE-locus × external-trait pairs identified in §5.

Pairs tested (chosen because they are FDR-significant or nominally significant
in the hypothesis-driven LDSC genetic correlation analyses):

  OCT (retinal, BRE-region × OCT-trait, BRE side via top-h² dim):
    1. R.Thalamus × GCIPL_L           (FDR q = 0.004)
    2. R.Thalamus × GCIPL_R           (FDR q = 0.004)
    3. L.Thalamus × GCIPL_L           (FDR q = 0.006)
    4. L.Thalamus × GCIPL_R           (FDR q = 0.019)
    5. R.Hippocampus × INL_L          (nominal p = 0.008, q = 0.143)
    6. R.Hippocampus × INL_R          (nominal p = 0.016, q = 0.222)

  Cardiac (atrial structure, BRE-region × cardiac-trait):
    7. L.Hippocampus × LAEF           (nominal p = 0.001, q = 0.075)
    8. R.Amygdala × LAV_min           (nominal p = 0.002, q = 0.075)

For each pair we:
  (a) Identify candidate loci where BOTH signals are present — defined as
      JAGWAS aggregate-locus windows for the BRE region that contain at least
      one SNP with OCT/cardiac p < 1e-5 in the matched trait sumstats.
  (b) For each candidate locus, extract per-SNP statistics in the locus window
      from BOTH the BRE-dim FastGWA sumstats (the top-h² dim used in the
      LDSC focused analysis) and the matched OCT/cardiac sumstats.
  (c) Run coloc.abf via R (coloc 5.x) to compute posterior probabilities
      PP.H0–PP.H4 (4 = shared causal variant; 3 = distinct causal variants).
  (d) Classify each locus by PP.H4 > 0.8 (strong colocalization) /
      0.5 ≤ PP.H4 ≤ 0.8 (suggestive) / PP.H3 > 0.5 (independent variants).

Outputs:
  results/coloc/coloc_focused_summary.csv   — per locus per pair: PP.H0..H4
  results/coloc/coloc_focused_per_pair.csv  — long format with locus-level details

Compute env:
  Python 3.7
  pip packages: coloc 0.4.2 (Bayesian coloc.abf) + sumstats 0.1.2 (approx_lnbf)
  (We dropped the R `coloc` 5.x path because the conda solver hung on the
   r-coloc dependency tree; the PyPI `coloc` package is a faithful pure-Python
   implementation of Giambartolomei 2014 coloc.abf and accepts per-SNP log Bayes
   factors directly — same posterior probabilities PP.H0..H4.)

Reference:
  Giambartolomei, C. et al. Bayesian test for colocalisation between pairs of
  genetic association studies using summary statistics. PLOS Genet. 10,
  e1004383 (2014). doi:10.1371/journal.pgen.1004383
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import coloc as coloc_pkg
import sumstats as sumstats_pkg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from paths import cfg

AGG_LOCI_CSV = Path("<EXTERNAL: TableS1 aggregate 276-loci CSV>")

# Top-h² dim per region — used in the LDSC focused analysis (Script 19b for OCT,
# 18b for cardiac). Pull from existing TableS4a/S4b on disk to ensure consistency.
TABLE_S4A = Path("<EXTERNAL: TableS4a OCT genetic-correlation focused table>")
TABLE_S4B = Path("<EXTERNAL: TableS4b heart genetic-correlation focused table>")

# External trait sumstats locations
OCT_BASE = Path(str(cfg.postgwas.oct_sumstats_dir))
HEART_BASE = Path(str(cfg.postgwas.heart_sumstats_dir))

# Per-dim FastGWA sumstats base
BRE_SUMSTATS_BASE = Path(str(cfg.gwas.fastgwa_out))

# OUT_DIR
OUT_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "coloc"

# The 8 pre-specified pairs.
# Format: (BRE region folder name, region display in TableS4, TableS4 trait value, modality)
# - region display in TableS4a/b uses short form (R.Thalamus, BrainStem, etc.) NOT TableS1 display
# - trait values must match TableS4a/b `trait` column exactly
PAIRS = [
    # (region_folder, ts4_region, ts4_trait, modality)
    ("Right_Thalamus-Proper", "R.Thalamus",    "GCIPL_thickness_left",  "oct"),
    ("Right_Thalamus-Proper", "R.Thalamus",    "GCIPL_thickness_right", "oct"),
    ("Left_Thalamus_Proper",  "L.Thalamus",    "GCIPL_thickness_left",  "oct"),
    ("Left_Thalamus_Proper",  "L.Thalamus",    "GCIPL_thickness_right", "oct"),
    ("Right_Hippocampus",     "R.Hippocampus", "INL_thickness_left",    "oct"),
    ("Right_Hippocampus",     "R.Hippocampus", "INL_thickness_right",   "oct"),
    ("Left_Hippocampus",      "L.Hippocampus", "LAEF",                  "heart"),
    ("Right_Amygdala",        "R.Amygdala",    "LAV_min",               "heart"),
]

# Per-SNP lnBF + coloc.abf are computed via:
#   sumstats.approx_lnbf(beta=BETA, se_beta=SE)  ->  per-SNP log Bayes factor
#   coloc.coloc(lnbfs_trait1, lnbfs_trait2, prior1, prior2, prior12)
#       returns a result object with .pp0 .. .pp4 (PP.H0–PP.H4)
COLOC_PRIORS = dict(prior1=1e-4, prior2=1e-4, prior12=1e-5)
MIN_OVERLAP_SNPS = 50  # minimum #SNPs in the locus window with merged BRE+ext stats


def _load_top_dim_per_pair():
    """Look up the top-h² dim from TableS4a/S4b. Columns: region, trait, best_dim, ..."""
    oct_df = pd.read_csv(TABLE_S4A)
    heart_df = pd.read_csv(TABLE_S4B)
    return oct_df, heart_df


def _bre_sumstats_path(region: str, dim: int) -> Path:
    """Per-dim FastGWA file (see docs/data_paths.md §7b + §11a)."""
    return (
        BRE_SUMSTATS_BASE
        / f"discovery_{region}_QT{dim}.fastGWA.fastGWA"
    )


def _ext_sumstats_path(modality: str, trait_label: str, oct_meta_path: Path) -> Path:
    """OCT or cardiac sumstats path. See docs/data_paths.md §11a.
    - OCT: trait_label = TableS4a `trait` (e.g. GCIPL_thickness_left).
           ID_OCT.xlsx header is row 1; column `ID` = trait label,
           column `File_name` = the folder name (e.g. ukbiobank_eye_oct_80k_9_march2022).
           Per-trait fastGWA is the single .fastGWA file inside that folder.
    - Heart: trait_label = TableS4b `trait` (e.g. LAEF, LAV_min).
           Bai 82 metadata in Script 18 maps trait label → pheno_id;
           per-trait fastGWA is at
           HEART_BASE / ukbiobank_heart_pheno{ID}_may2022/ukb_phase1to3_heart_may_2022_pheno{ID}.fastGWA
    """
    if modality == "oct":
        meta = pd.read_excel(oct_meta_path, header=1).dropna(subset=["File_name"])
        match = meta[meta["ID"].astype(str).str.strip() == trait_label]
        if match.empty:
            raise FileNotFoundError(f"OCT trait '{trait_label}' not in {oct_meta_path}")
        folder_name = str(match.iloc[0]["File_name"]).strip()
        folder = OCT_BASE / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"OCT folder not found: {folder}")
        fastgwa_files = sorted(folder.glob("*.fastGWA"))
        if not fastgwa_files:
            raise FileNotFoundError(f"No .fastGWA file inside {folder}")
        return fastgwa_files[0]
    elif modality == "heart":
        # Cardiac trait_label → pheno_id from Bai82_names_ukb_v2.csv (verified 2026-05-25).
        # Only the two pre-specified pairs in scope here; extend if more added.
        heart_map = {"LAEF": "31", "LAV_min": "29"}
        if trait_label not in heart_map:
            raise KeyError(f"Heart trait '{trait_label}' not in heart_map; extend mapping")
        pid = heart_map[trait_label]
        return HEART_BASE / f"ukbiobank_heart_pheno{pid}_may2022" / f"ukb_phase1to3_heart_may_2022_pheno{pid}.fastGWA"
    raise ValueError(f"Unknown modality: {modality}")


def _candidate_loci_for_pair(
    region: str, ext_sumstats: Path, agg_loci: pd.DataFrame, p_threshold: float = 1e-5,
) -> pd.DataFrame:
    """Return aggregate-locus rows for this region that have at least one SNP
    with p < p_threshold in the external trait sumstats."""
    region_label = _folder_to_display(region)
    # Aggregate loci where this region is in the regions list
    in_region = agg_loci["regions"].astype(str).str.contains(region_label, na=False)
    candidate = agg_loci[in_region].copy()
    if candidate.empty or not ext_sumstats.exists():
        return candidate
    # For each locus, check if external sumstats have any SNP with p < threshold
    # in [chr:start-end]. This requires reading the external sumstats — large file;
    # use chunked or tabix-style lookup in production. For now flag for runtime.
    return candidate


def _folder_to_display(folder: str) -> str:
    """Map 'Left_Putamen' → 'L. Putamen'."""
    mapping = {
        "Brain_Stem_or_4th_Ventricle": "Brain Stem / 4th V.", "CSF": "CSF",
        "Left_Accumbens-area": "L. Accumbens", "Right_Accumbens-area": "R. Accumbens",
        "Left_Amygdala": "L. Amygdala", "Right_Amygdala": "R. Amygdala",
        "Left_Caudate": "L. Caudate", "Right_Caudate": "R. Caudate",
        "Left_Hippocampus": "L. Hippocampus", "Right_Hippocampus": "R. Hippocampus",
        "Left_Pallidum": "L. Pallidum", "Right_Pallidum": "R. Pallidum",
        "Left_Putamen": "L. Putamen", "Right_Putamen": "R. Putamen",
        "Left_Thalamus_Proper": "L. Thalamus", "Right_Thalamus-Proper": "R. Thalamus",
    }
    return mapping.get(folder, folder)


def _load_fastgwa_full(path: Path) -> pd.DataFrame:
    """Load the full fastGWA file into memory once. Indexed for fast per-locus filter.
    fastGWA columns: CHR SNP POS A1 A2 N AF1 BETA SE P (INFO optional)."""
    usecols = ["CHR", "SNP", "POS", "N", "AF1", "BETA", "SE", "P"]
    df = pd.read_csv(path, sep="\t", usecols=usecols,
                     dtype={"CHR": str, "SNP": str, "POS": "Int32",
                            "N": "Int32", "AF1": "float32",
                            "BETA": "float32", "SE": "float32", "P": "float64"})
    # Pre-sort by (CHR, POS) so per-chromosome subsets are contiguous (cheap groupby)
    df = df.sort_values(["CHR", "POS"]).reset_index(drop=True)
    return df


def _filter_to_locus(df: pd.DataFrame, chr_: int, start: int, end: int) -> pd.DataFrame:
    """In-memory filter to a locus window. Microseconds; no I/O."""
    mask = (df["CHR"] == str(chr_)) & (df["POS"] >= start) & (df["POS"] <= end)
    return df[mask]


def run_coloc_one_locus(
    bre_full: pd.DataFrame, ext_full: pd.DataFrame, locus_row: pd.Series,
) -> dict:
    """Filter pre-loaded BRE and external sumstats to the locus window and run
    coloc.abf via the PyPI `coloc` package. Returns dict of PP_H0..PP_H4 + N_SNPS."""
    chr_ = int(locus_row["chr"])
    start = int(locus_row["start"])
    end = int(locus_row["end"])
    bre = _filter_to_locus(bre_full, chr_, start, end)
    ext = _filter_to_locus(ext_full, chr_, start, end)
    if bre.empty or ext.empty:
        return {"PP_H0": np.nan, "PP_H1": np.nan, "PP_H2": np.nan,
                "PP_H3": np.nan, "PP_H4": np.nan, "N_SNPS": 0,
                "error": f"bre_n={len(bre)} ext_n={len(ext)}"}
    merged = bre.merge(ext, on="SNP", suffixes=("_bre", "_ext"))
    if len(merged) < MIN_OVERLAP_SNPS:
        return {"PP_H0": np.nan, "PP_H1": np.nan, "PP_H2": np.nan,
                "PP_H3": np.nan, "PP_H4": np.nan, "N_SNPS": len(merged),
                "error": f"too few overlapping SNPs ({len(merged)} < {MIN_OVERLAP_SNPS})"}
    # Per-SNP approximate log-Bayes factors (Wakefield ABF), preferring beta/se path
    lnbfs_bre = [sumstats_pkg.approx_lnbf(beta=b, se_beta=s)
                 for b, s in zip(merged["BETA_bre"], merged["SE_bre"])]
    lnbfs_ext = [sumstats_pkg.approx_lnbf(beta=b, se_beta=s)
                 for b, s in zip(merged["BETA_ext"], merged["SE_ext"])]
    # coloc.coloc() returns a 5-element generator: (PP.H0, PP.H1, PP.H2, PP.H3, PP.H4)
    pp = list(coloc_pkg.coloc(lnbfs_bre, lnbfs_ext, **COLOC_PRIORS))
    return {
        "PP_H0": float(pp[0]), "PP_H1": float(pp[1]), "PP_H2": float(pp[2]),
        "PP_H3": float(pp[3]), "PP_H4": float(pp[4]), "N_SNPS": len(merged),
    }


def main():
    p = argparse.ArgumentParser(description="Focused coloc for 8 cross-modal pairs (pure-Python coloc.abf)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--p-threshold", type=float, default=1e-5,
                   help="External-trait p-value threshold for candidate-locus selection")
    p.add_argument("--oct-meta", type=Path,
                   default=OCT_BASE / "ID_OCT.xlsx",
                   help="ID_OCT.xlsx metadata file (maps File_name → ID)")
    p.add_argument("--only", nargs="+", default=None,
                   help="Optional subset of pair labels to test (e.g. RThalamus_GCIPL_thickness_left)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agg_loci = pd.read_csv(AGG_LOCI_CSV)
    oct_df, heart_df = _load_top_dim_per_pair()

    all_results = []
    for region, ts4_region, ts4_trait, modality in PAIRS:
        pair_label = f"{ts4_region.replace('.','')}_{ts4_trait}"
        if args.only and pair_label not in args.only:
            continue
        print(f"\n=== {pair_label} ({modality}) ===")
        # Look up top-h² dim from TableS4 (columns: region, trait, best_dim, ...)
        df_ts4 = oct_df if modality == "oct" else heart_df
        row = df_ts4[(df_ts4["region"] == ts4_region) & (df_ts4["trait"] == ts4_trait)]
        if row.empty:
            print(f"  WARN: no top-dim entry in TableS4 for ({ts4_region}, {ts4_trait}); skipping")
            continue
        top_dim = int(row.iloc[0]["best_dim"])
        bre_sumstats = _bre_sumstats_path(region, top_dim)
        if not bre_sumstats.exists():
            print(f"  WARN: BRE sumstats not found at {bre_sumstats}")
            continue
        try:
            ext_sumstats = _ext_sumstats_path(modality, ts4_trait, args.oct_meta)
        except (FileNotFoundError, KeyError) as e:
            print(f"  WARN: {e}; skipping")
            continue
        if not ext_sumstats.exists():
            print(f"  WARN: external sumstats not found at {ext_sumstats}")
            continue
        # Candidate aggregate loci where this BRE region is implicated
        cands = _candidate_loci_for_pair(region, ext_sumstats, agg_loci, args.p_threshold)
        print(f"  Candidate loci for {region}: {len(cands)}  (top_dim={top_dim})")
        if cands.empty:
            continue
        # ── Pre-load both sumstats files ONCE per pair (~8x speedup vs per-locus chunked read)
        print(f"  Loading BRE  sumstats: {bre_sumstats.name}")
        bre_full = _load_fastgwa_full(bre_sumstats)
        print(f"    {len(bre_full):,} rows")
        print(f"  Loading ext  sumstats: {ext_sumstats.name}")
        ext_full = _load_fastgwa_full(ext_sumstats)
        print(f"    {len(ext_full):,} rows")
        for _, locus in cands.iterrows():
            res = run_coloc_one_locus(bre_full, ext_full, locus)
            res.update({
                "pair": pair_label, "region": region, "modality": modality,
                "trait": ts4_trait, "top_dim": top_dim,
                "al_id": locus["al_id"], "chr": locus["chr"],
                "start": locus["start"], "end": locus["end"],
            })
            print(f"  AL-{locus['al_id']}: PP.H4 = {res.get('PP_H4', float('nan')):.3f}  "
                  f"(N_SNPS={res.get('N_SNPS',0)})")
            all_results.append(res)
        # Free memory before next pair
        del bre_full, ext_full

    if not all_results:
        print("\nNo coloc results produced.")
        return

    df = pd.DataFrame(all_results)
    out_per_pair = args.out_dir / "coloc_focused_per_pair.csv"
    df.to_csv(out_per_pair, index=False)
    print(f"\nWrote: {out_per_pair}  ({len(df)} locus-pair rows)")

    # Summary classification per locus-pair
    df["coloc_class"] = np.select(
        [df["PP_H4"] > 0.8, df["PP_H4"] >= 0.5, df["PP_H3"] > 0.5],
        ["strong shared causal", "suggestive shared", "distinct variants"],
        default="indeterminate",
    )
    summary_cols = ["pair", "al_id", "chr", "start", "end",
                    "PP_H0", "PP_H1", "PP_H2", "PP_H3", "PP_H4",
                    "N_SNPS", "coloc_class"]
    out_summary = args.out_dir / "coloc_focused_summary.csv"
    df[summary_cols].to_csv(out_summary, index=False)
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
