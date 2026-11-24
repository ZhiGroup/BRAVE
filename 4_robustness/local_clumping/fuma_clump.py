#!/usr/bin/env python3
"""
fuma_clump.py  –  Local FUMA-equivalent SNP clumping pipeline using plink 1.9

Algorithm mirrors FUMA SNP2GENE (getLD.py) with three stages:
  Stage 1 – Independent Significant SNPs (IndSigSNPs):
      Greedy LD-clumping via plink --clump at r2 threshold.
      Starting from the most significant SNP below leadP, all SNPs
      in LD (r2 >= r2) are grouped into its clump. Candidate SNPs
      are clump members with p <= gwasP.
  Stage 2 – Lead SNPs:
      Among IndSigSNPs, those with mutual LD r2 >= r2_2 are grouped
      together; the lowest-p IndSigSNP in each group is the LeadSNP.
  Stage 3 – Genomic Risk Loci:
      LeadSNP LD-blocks are merged if <= mergeDist bp apart.

Input:
  GWAS summary statistics in fastGWA format (tab-separated):
    CHR  SNP  POS  A1  A2  N  AF1  BETA  SE  P

  Per-chromosome plink 1.9 BED/BIM/FAM reference files.
  Use {} as chromosome placeholder in --bfile-pattern, e.g.:
    /path/to/1000G_EUR/EUR.chr{}   → EUR.chr1.bed, EUR.chr2.bed, ...

Output files (FUMA-compatible):
  IndSigSNPs.txt       – Independent significant SNPs
  leadSNPs.txt         – Lead SNPs (grouped IndSigSNPs at r2_2)
  GenomicRiskLoci.txt  – Merged genomic risk loci
  snps.txt             – All candidate SNPs with LD info
  ld.txt               – Pairwise r2 (IndSigSNP, candidate SNP)

Usage:
  python fuma_clump.py \\
      --gwas  my_gwas.fastGWA \\
      --bfile-pattern  /path/to/1000G_EUR/EUR.chr{} \\
      --plink <path/to>/plink \\
      --out   ./clump_results/

Reference panel download (1000G EUR, hg19):
  wget https://ctg.cncr.nl/software/MAGMA/ref_data/g1000_eur.zip
  unzip g1000_eur.zip
  # Rename: g1000_eur.bed -> EUR.chr1.bed etc. per chromosome if needed.
  # Alternatively use any per-chromosome plink BED files in LD reference.
"""

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FUMA default parameters (mirroring SNP2GENE defaults)
# ---------------------------------------------------------------------------
LEAD_P = 5e-8  # P-value threshold for independent significant SNPs
GWAS_P = 0.05  # P-value threshold for candidate SNPs
R2 = 0.6  # LD r2 for defining IndSigSNPs / candidate SNPs
R2_2 = 0.1  # LD r2 for grouping IndSigSNPs into LeadSNPs
MAF = 0.001  # Minimum MAF (applied via plink --maf)
MERGE_DIST = 250_000  # Max distance (bp) to merge adjacent loci
MHC_START = 29_614_758  # hg19
MHC_END = 33_170_276  # hg19
# plink --clump-kb / --ld-window-kb, in kb. 1000 matches FUMA, which builds its
# LD tables at plink's default 1000 kb window (`--r2 --ld-window 99999
# --ld-window-r2 0.05`, no --ld-window-kb) and therefore never links two SNPs
# more than 1 Mb apart.
#
# This was 50_000 -- effectively unbounded -- which merges blocks FUMA keeps
# separate wherever LD is long-range, the 8p23.1 inversion above all. Against the
# published FUMA output that cost 6 of 16 regions their exact agreement, and it
# was invisible: the flag had no CLI override, so nothing in a run log revealed
# which window had been used. The safe value is now the default; pass
# --clump-kb explicitly to widen it.
CLUMP_KB = 1_000
LD_WINDOW = 999_999  # plink --ld-window (variant count)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_uniq_id(chrom, pos, a1, a2):
    """FUMA uniqID: chr:pos:allele1:allele2 with alleles alphabetically sorted."""
    alleles = sorted([str(a1).upper(), str(a2).upper()])
    return f"{chrom}:{pos}:{alleles[0]}:{alleles[1]}"


def parse_sp2(sp2_str):
    """
    Parse plink .clumped SP2 field.
    Format: 'rs1(0.82),rs2(0.61)' or 'NONE'.
    Returns list of (rsID, r2_float) tuples.
    """
    if not sp2_str or str(sp2_str).strip() in ("NONE", "nan", ""):
        return []
    result = []
    for token in str(sp2_str).split(","):
        token = token.strip()
        if not token or token == "NONE":
            continue
        m = re.match(r"^(.+)\(([0-9.eE+\-]+)\)$", token)
        if m:
            result.append((m.group(1), float(m.group(2))))
    return result


# ---------------------------------------------------------------------------
# GWAS harmonization  (FUMA gwas_file.py equivalent)
# ---------------------------------------------------------------------------


def harmonize_gwas(df):
    """
    GWAS sanity checks and harmonization.

    Steps:
      1. Restrict alleles to {A, T, G, C} — removes indels and invalid codes.
         Ambiguous A/T and G/C SNPs are retained; their p-value signal is
         valid for clumping regardless of strand orientation.
      2. Remove duplicate uniqIDs — keep the SNP with the lowest p-value.

    Full rsID verification and strand harmonization are handled by
    harmonize_with_reference() when --dbsnp is supplied.

    Returns cleaned DataFrame.
    """
    n_start = len(df)

    # 1. Valid alleles only (keep A/T and G/C — valid GWAS signal)
    valid = {"A", "T", "G", "C"}
    mask_valid = df["A1"].str.upper().isin(valid) & df["A2"].str.upper().isin(valid)
    n_invalid = int((~mask_valid).sum())
    if n_invalid:
        print(f"[HARMONIZE] Removed {n_invalid:,} SNPs with non-ACGT alleles")
    df = df[mask_valid].copy()

    # 2. Duplicate uniqIDs — keep lowest p-value
    n_before = len(df)
    df = df.sort_values("P").drop_duplicates("uniqID").copy()
    n_dup = n_before - len(df)
    if n_dup:
        print(
            f"[HARMONIZE] Removed {n_dup:,} duplicate positions "
            f"(kept lowest p-value per uniqID)"
        )

    print(f"[HARMONIZE] {n_start:,} → {len(df):,} SNPs retained")
    return df


def _load_ref_as_dataframe(ref_path: str) -> pd.DataFrame:
    """
    Load a reference into a DataFrame with columns CHR, POS, SNP, A1, A2.

    Accepts two formats:
      - Parquet file (produced by build_ref_map.py) — fast, recommended
      - Plain or gzipped plink BIM  (CHR rsID cM POS A1 A2, no header) — slow
    """
    if ref_path.endswith(".parquet"):
        print(f"[DBSNP] Loading reference map (parquet): {ref_path}")
        ref = pd.read_parquet(ref_path)
        # Ensure correct column types
        ref["CHR"] = ref["CHR"].astype(str).str.lstrip("0").replace("", "0")
        ref["POS"] = ref["POS"].astype(int)
        ref["A1"] = ref["A1"].str.upper()
        ref["A2"] = ref["A2"].str.upper()
        # Preserve _ikey if pre-computed; pass through for use in harmonize_with_reference
        keep_cols = ["CHR", "POS", "SNP", "A1", "A2"]
        if "_ikey" in ref.columns:
            keep_cols.append("_ikey")
        ref = ref[keep_cols]
    else:
        print(
            f"[DBSNP] Loading reference BIM (slow — consider build_ref_map.py): {ref_path}"
        )
        ref = pd.read_csv(
            ref_path,
            sep=r"\s+",
            header=None,
            names=["CHR", "SNP", "cM", "POS", "A1", "A2"],
            dtype={"CHR": str, "POS": int},
            engine="c",
        )
        ref["CHR"] = ref["CHR"].astype(str).str.lstrip("0").replace("", "0")
        ref["A1"] = ref["A1"].str.upper()
        ref["A2"] = ref["A2"].str.upper()
        ref = ref[["CHR", "POS", "SNP", "A1", "A2"]]
        ref = ref.drop_duplicates(subset=["CHR", "POS"], keep="first")
    print(f"[DBSNP] Reference has {len(ref):,} variants")
    return ref


def harmonize_with_reference(df, ref_bim, _ref_df=None):
    """
    Harmonize GWAS SNPs against a reference (BIM file or pre-loaded DataFrame).

    Steps:
      1. Load reference (parquet fast-path or plain BIM slow-path).
      2. Merge GWAS on (CHR, POS) — SNPs absent from the reference are dropped.
      3. Update GWAS rsIDs to reference rsIDs where they differ.
      4. Detect strand flips (complement alleles): flip A1/A2 in-place so they
         match the reference strand, then recompute uniqID.
      5. Drop SNPs whose alleles are irreconcilable with the reference even
         after complement-flip (genuine allele mismatches).

    Args:
        df      : harmonized GWAS DataFrame (must have CHR, POS, SNP, A1, A2,
                  uniqID columns).
        ref_bim : path to reference file (.parquet from build_ref_map.py, or
                  plain/gzipped plink BIM).
        _ref_df : optional pre-loaded reference DataFrame (skips file I/O;
                  used internally when fuma_clump.py caches the reference).

    Returns:
        Updated DataFrame with reference-consistent rsIDs and alleles.
    """
    n_start = len(df)
    if _ref_df is not None:
        ref = _ref_df
        print(f"[DBSNP] Using cached reference ({len(ref):,} variants)")
    else:
        ref = _load_ref_as_dataframe(ref_bim)

    # Build a single int64 merge key: chr * 10^9 + pos
    # Direct int8 cast on already-clean '1'-'22' strings is much faster than
    # lstrip/replace/to_numeric on 8M+ rows.
    df = df.copy()
    df["_chr_norm"] = df["CHR"].astype(str).str.lstrip("0").replace("", "0")
    df["_ikey"] = pd.to_numeric(df["_chr_norm"], errors="coerce").fillna(0).astype(
        "int64"
    ) * 1_000_000_000 + df["POS"].astype("int64")

    ref_lookup = ref[["CHR", "POS", "SNP", "A1", "A2"]].copy()
    # Use pre-computed _ikey from parquet if available (saves ~5s on 8M-row string conversion)
    if "_ikey" in ref.columns:
        ref_lookup["_ikey"] = ref["_ikey"].values
    else:
        ref_lookup["_ikey"] = pd.to_numeric(
            ref_lookup["CHR"].astype(str), errors="coerce"
        ).fillna(0).astype("int64") * 1_000_000_000 + ref_lookup["POS"].astype("int64")
    ref_lookup = ref_lookup.rename(columns={"SNP": "SNP_R", "A1": "A1_R", "A2": "A2_R"})

    merged = df.merge(
        ref_lookup[["_ikey", "SNP_R", "A1_R", "A2_R"]],
        on="_ikey",
        how="inner",  # drops unmatched GWAS SNPs directly; count via n_start - len
    )

    n_not_found = n_start - len(merged)
    found = merged.copy()

    # ------------------------------------------------------------------
    # Update rsIDs
    # ------------------------------------------------------------------
    n_updated = int((found["SNP"] != found["SNP_R"]).sum())
    found["SNP"] = found["SNP_R"].values

    # ------------------------------------------------------------------
    # Allele harmonization — numpy uint8 for 100x speedup over str.translate
    # ------------------------------------------------------------------
    # Complement lookup table: A(65)<->T(84), C(67)<->G(71), lowercase too
    _COMP = np.arange(256, dtype=np.uint8)
    for _a, _b in [
        (65, 84),
        (84, 65),
        (67, 71),
        (71, 67),
        (97, 116),
        (116, 97),
        (99, 103),
        (103, 99),
    ]:
        _COMP[_a] = _b
        _COMP[_b] = _a

    def _to_upper_uint8(series):
        """
        Convert a Series of single-char allele strings to uppercase uint8.
        Uses np.frombuffer on the concatenated bytes — no Python loop.
        """
        vals = series.values.astype("U1")  # fixed-width 1-char unicode
        raw = vals.view(np.uint8)  # each char = 4 bytes (UCS-4)
        # Take only the first byte of each code point (ASCII range A/T/C/G = <128)
        b = raw[::4].copy()  # stride-4 slice
        # Uppercase: 'a'=97 -> 'A'=65 etc. (set bit 5 = 0)
        b[b >= 97] &= np.uint8(0xDF)
        return b

    b1 = _to_upper_uint8(found["A1"])
    b2 = _to_upper_uint8(found["A2"])
    b1r = _to_upper_uint8(found["A1_R"])
    b2r = _to_upper_uint8(found["A2_R"])

    # Direct match
    direct = ((b1 == b1r) & (b2 == b2r)) | ((b1 == b2r) & (b2 == b1r))
    # Complement match
    b1c = _COMP[b1]
    b2c = _COMP[b2]
    comp_match = ((b1c == b1r) & (b2c == b2r)) | ((b1c == b2r) & (b2c == b1r))
    n_flipped = int(comp_match.sum())

    # Irreconcilable allele mismatch
    mismatch = ~direct & ~comp_match
    n_mismatch = int(mismatch.sum())

    # Flip strand for complement-matching SNPs: vectorized uint8 -> char lookup
    # Build a 256-element char array indexed by ASCII byte value.
    _B2C = np.zeros(256, dtype="U1")
    _B2C[65] = "A"; _B2C[84] = "T"; _B2C[67] = "C"; _B2C[71] = "G"
    found = found.copy()
    flip_idx = np.where(comp_match)[0]
    if len(flip_idx):
        found.iloc[flip_idx, found.columns.get_loc("A1")] = _B2C[b1c[flip_idx]]
        found.iloc[flip_idx, found.columns.get_loc("A2")] = _B2C[b2c[flip_idx]]

    # Drop mismatches
    found = found[~mismatch].copy()

    # Recompute uniqID after possible strand flip
    a1_up = found["A1"].str.upper()
    a2_up = found["A2"].str.upper()
    allele_min = a1_up.where(a1_up <= a2_up, a2_up)
    allele_max = a2_up.where(a1_up <= a2_up, a1_up)
    found["uniqID"] = (
        found["CHR"].astype(str)
        + ":"
        + found["POS"].astype(str)
        + ":"
        + allele_min
        + ":"
        + allele_max
    )

    # Clean up temporary columns
    found = found.drop(
        columns=["_chr_norm", "_ikey", "SNP_R", "A1_R", "A2_R"],
        errors="ignore",
    )

    print(f"[DBSNP] {n_not_found:,} SNPs not found in reference (removed)")
    print(f"[DBSNP] {n_updated:,} SNP IDs updated to reference rsIDs")
    print(f"[DBSNP] {n_flipped:,} SNPs strand-flipped to match reference")
    if n_mismatch:
        print(f"[DBSNP] {n_mismatch:,} SNPs removed (irreconcilable allele mismatch)")
    print(f"[DBSNP] {n_start:,} → {len(found):,} SNPs retained after reference check")
    return found.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-computed LD table
# ---------------------------------------------------------------------------


class LDTable:
    """
    Per-chromosome sparse LD lookup table backed by a CSR numpy layout.

    Primary format (fast, ~2 s for chr17):
        ld_chr{N}.npz      – arrays: ib (int32), r2 (float32), indptr (int32)
        ld_chr{N}_snps.npy – string array mapping int index -> rsID

    Legacy fallback (slow, ~83 s for chr17):
        ld_chr{N}.tsv.gz   – tab-separated: SNP_A BP_A SNP_B BP_B R2

    Both directions are stored (symmetric): indptr has length n_snps+1,
    ib/r2 have length 2*n_pairs.
    """

    def __init__(self):
        self._snps = None  # np.ndarray of str, shape (n_snps,)
        self._s2i = {}  # str -> int index
        self._ib = None  # np.ndarray int32, neighbor indices (CSR data)
        self._r2 = None  # np.ndarray float32, r2 values (CSR data)
        self._indptr = None  # np.ndarray int32, shape (n_snps+1,)
        self._chrom = None

    def load(self, ld_dir, chrom):
        npz_file = os.path.join(ld_dir, f"ld_chr{chrom}.npz")
        snps_file = os.path.join(ld_dir, f"ld_chr{chrom}_snps.npy")
        # fallback: old tsv.gz format
        tsv_file = os.path.join(ld_dir, f"ld_chr{chrom}.tsv.gz")

        if os.path.exists(npz_file) and os.path.exists(snps_file):
            print(f"  [LD] Loading chr{chrom} (CSR npz)...", end=" ", flush=True)
            self._snps = np.load(snps_file, allow_pickle=True)
            d = np.load(npz_file)
            self._ib = d["ib"]
            self._r2 = d["r2"]
            self._indptr = d["indptr"]
            self._s2i = {s: i for i, s in enumerate(self._snps)}
            self._chrom = int(chrom)
            n_pairs = len(self._ib) // 2
            print(f"{n_pairs:,} pairs, {len(self._snps):,} SNPs")
        elif os.path.exists(tsv_file):
            # legacy fallback path (old gzip TSV format)
            print(
                f"  [LD] Loading chr{chrom} (legacy gzip TSV)...", end=" ", flush=True
            )
            import gzip as _gzip

            self._snps = None
            data = {}
            with _gzip.open(tsv_file, "rt") as fh:
                next(fh)
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    snp_a, snp_b = parts[0], parts[2]
                    try:
                        r2 = float(parts[4])
                    except ValueError:
                        continue
                    if snp_a not in data:
                        data[snp_a] = {}
                    if snp_b not in data:
                        data[snp_b] = {}
                    data[snp_a][snp_b] = r2
                    data[snp_b][snp_a] = r2
            # convert to CSR
            snp_list = list(data.keys())
            self._snps = np.array(snp_list)
            self._s2i = {s: i for i, s in enumerate(snp_list)}
            n = len(snp_list)
            ia_l, ib_l, r2_l = [], [], []
            for a, neighbors in data.items():
                ai = self._s2i[a]
                for b, r in neighbors.items():
                    ia_l.append(ai)
                    ib_l.append(self._s2i[b])
                    r2_l.append(r)
            ia_arr = np.array(ia_l, dtype=np.int32)
            ib_arr = np.array(ib_l, dtype=np.int32)
            r2_arr = np.array(r2_l, dtype=np.float32)
            order = np.argsort(ia_arr, kind="stable")
            self._ib = ib_arr[order]
            self._r2 = r2_arr[order]
            self._indptr = np.searchsorted(ia_arr[order], np.arange(n + 1)).astype(
                np.int32
            )
            self._chrom = int(chrom)
            print(f"{len(r2_arr) // 2:,} pairs, {n:,} SNPs")
        else:
            raise FileNotFoundError(
                f"LD table not found for chr{chrom} in {ld_dir}\n"
                f"Expected: {npz_file} + {snps_file}\n"
                f"Run build_ld_table.py --out-dir {ld_dir} first."
            )
        return self

    def get_neighbors(self, snp, r2_min):
        """Return list of (neighbor_snp, r2) tuples with r2 >= r2_min."""
        idx = self._s2i.get(snp, -1)
        if idx < 0:
            return []
        s, e = int(self._indptr[idx]), int(self._indptr[idx + 1])
        if s == e:
            return []
        r2_slice = self._r2[s:e]
        mask = r2_slice >= r2_min
        if not mask.any():
            return []
        ib_masked = self._ib[s:e][mask]
        return list(zip(self._snps[ib_masked], r2_slice[mask].tolist()))

    def get_r2(self, snp_a, snp_b):
        """Return r2 between two SNPs (0.0 if unknown)."""
        idx_a = self._s2i.get(snp_a, -1)
        if idx_a < 0:
            return 0.0
        idx_b = self._s2i.get(snp_b, -1)
        if idx_b < 0:
            return 0.0
        s, e = int(self._indptr[idx_a]), int(self._indptr[idx_a + 1])
        for j, r in zip(self._ib[s:e], self._r2[s:e]):
            if int(j) == idx_b:
                return float(r)
        return 0.0

    def is_loaded(self):
        return self._chrom is not None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_gwas(gwas_file, no_mhc, mhc_start, mhc_end):
    """
    Read fastGWA summary statistics.
    Returns a pandas DataFrame with standardized columns and a 'uniqID' column.
    """
    print(f"[INFO] Reading GWAS: {gwas_file}")
    df = pd.read_csv(
        gwas_file, sep="\t", comment="#",
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )

    # Normalise column names to upper-case for robust matching
    df.columns = [c.strip() for c in df.columns]

    required = {"CHR", "SNP", "POS", "A1", "A2", "P"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"[ERROR] Missing required columns in GWAS file: {missing}\n"
            f"Expected fastGWA format: CHR SNP POS A1 A2 N AF1 BETA SE P"
        )

    # Cast types
    df["CHR"] = (
        df["CHR"].str.replace("chr", "", case=False).str.replace("CHR", "").str.strip()
    )
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
    df["P"] = pd.to_numeric(df["P"], errors="coerce")

    # Remove rows with invalid p-values or positions
    df = df.dropna(subset=["POS", "P", "CHR"])
    df = df[(df["P"] > 0) & (df["P"] <= 1)]

    # Restrict to autosomes 1–22
    df = df[df["CHR"].isin([str(c) for c in range(1, 23)])]
    df["CHR"] = df["CHR"].astype(str)

    # Exclude MHC region (chr6) if requested
    if no_mhc:
        mask_mhc = (
            (df["CHR"] == "6") & (df["POS"] >= mhc_start) & (df["POS"] <= mhc_end)
        )
        n_mhc = int(mask_mhc.sum())
        df = df[~mask_mhc].copy()
        print(f"[INFO] Excluded {n_mhc} MHC SNPs (chr6:{mhc_start}-{mhc_end})")

    # Compute MAF from AF1 column (if present)
    if "AF1" in df.columns:
        af1 = pd.to_numeric(df["AF1"], errors="coerce")
        df["MAF"] = af1.where(af1 <= 0.5, 1.0 - af1)
    else:
        df["MAF"] = np.nan

    # Build FUMA-style unique ID (vectorized: chr:pos:sortedA:sortedB)
    a1_up = df["A1"].str.upper()
    a2_up = df["A2"].str.upper()
    allele_min = a1_up.where(a1_up <= a2_up, a2_up)
    allele_max = a2_up.where(a1_up <= a2_up, a1_up)
    df["uniqID"] = (
        df["CHR"].astype(str)
        + ":"
        + df["POS"].astype(str)
        + ":"
        + allele_min
        + ":"
        + allele_max
    )

    df = df.reset_index(drop=True)
    print(f"[INFO] {len(df):,} SNPs after filtering")
    return df


def write_plink_assoc(df, outfile):
    """Write a minimal plink-compatible association file: SNP, CHR, BP, P."""
    assoc = df[["SNP", "CHR", "POS", "P"]].copy()
    assoc.columns = ["SNP", "CHR", "BP", "P"]
    assoc.to_csv(outfile, sep="\t", index=False)


# ---------------------------------------------------------------------------
# plink wrappers
# ---------------------------------------------------------------------------


def _plink_maf_arg(maf):
    """FUMA's MAF=0 means "no filter"; plink rejects --maf 0 outright.

    The flag cannot simply be dropped either, because plink's own default is
    0.01. Substitute a threshold below the smallest frequency attainable in any
    reference panel, so only monomorphic variants are removed -- those carry no
    LD information.
    """
    return str(maf) if float(maf) > 0 else "0.000001"


def _run_plink(cmd, chrom, stage):
    """Run plink, surfacing failures instead of swallowing them.

    stdout and stderr were previously both sent to DEVNULL, which made a
    rejected argument indistinguishable from a chromosome carrying no signal:
    plink exits without writing its output file, and the caller reports "no
    significant SNPs".
    """
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        lines = proc.stdout.decode("utf-8", "replace").strip().splitlines()
        print(
            f"  [CHR{chrom}] plink {stage} FAILED (exit {proc.returncode}): "
            f"{lines[-1] if lines else 'no output'}"
        )
    return proc


def run_plink_clump(
    chrom, bfile, assoc_file, plink_bin, lead_p, gwas_p, r2, clump_kb, maf, tmpdir
):
    """
    Run plink --clump on one chromosome.
    Returns a DataFrame with the parsed .clumped output, or empty DataFrame.
    """
    out_prefix = os.path.join(tmpdir, f"chr{chrom}_clump")
    cmd = [
        plink_bin,
        "--bfile",
        bfile,
        "--clump",
        assoc_file,
        "--clump-snp-field",
        "SNP",
        "--clump-field",
        "P",
        "--clump-p1",
        str(lead_p),
        "--clump-p2",
        str(gwas_p),
        "--clump-r2",
        str(r2),
        "--clump-kb",
        str(clump_kb),
        "--maf",
        _plink_maf_arg(maf),
        "--chr",
        str(chrom),
        "--out",
        out_prefix,
        "--allow-no-sex",
        "--silent",
    ]
    _run_plink(cmd, chrom, "--clump")

    clumped = out_prefix + ".clumped"
    if not os.path.exists(clumped):
        return pd.DataFrame()

    try:
        df = pd.read_csv(clumped, sep=r"\s+", dtype=str)
    except Exception:
        return pd.DataFrame()

    if df.empty or "SNP" not in df.columns:
        return pd.DataFrame()

    return df


def run_plink_r2_among_snps(
    chrom, snp_list, bfile, plink_bin, r2_min, clump_kb, maf, tmpdir
):
    """
    Compute pairwise r2 for a set of SNPs on one chromosome using plink.
    Returns DataFrame with columns [SNP_A, SNP_B, R2], filtered to r2 >= r2_min.
    Only rows where BOTH SNP_A and SNP_B are in snp_list are returned.
    """
    if len(snp_list) <= 1:
        return pd.DataFrame(columns=["SNP_A", "SNP_B", "R2"])

    snp_file = os.path.join(tmpdir, f"chr{chrom}_ind_snps.txt")
    with open(snp_file, "w") as fh:
        fh.write("\n".join(snp_list) + "\n")

    out_prefix = os.path.join(tmpdir, f"chr{chrom}_r2")
    cmd = [
        plink_bin,
        "--bfile",
        bfile,
        "--r2",
        "--ld-snp-list",
        snp_file,
        "--ld-window-r2",
        str(r2_min),
        "--ld-window-kb",
        str(clump_kb),
        "--ld-window",
        str(LD_WINDOW),
        "--maf",
        _plink_maf_arg(maf),
        "--chr",
        str(chrom),
        "--out",
        out_prefix,
        "--allow-no-sex",
        "--silent",
    ]
    _run_plink(cmd, chrom, "--r2")

    ld_file = out_prefix + ".ld"
    if not os.path.exists(ld_file):
        return pd.DataFrame(columns=["SNP_A", "SNP_B", "R2"])

    try:
        ld = pd.read_csv(
            ld_file, sep=r"\s+", dtype=str, usecols=["SNP_A", "SNP_B", "R2"]
        )
    except Exception:
        return pd.DataFrame(columns=["SNP_A", "SNP_B", "R2"])

    ld["R2"] = pd.to_numeric(ld["R2"], errors="coerce")
    ld = ld.dropna(subset=["R2"])

    # Keep only pairs where both SNPs are in our target list
    snp_set = set(snp_list)
    ld = ld[ld["SNP_A"].isin(snp_set) & ld["SNP_B"].isin(snp_set)]
    ld = ld[ld["R2"] >= r2_min]
    return ld.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def process_chromosome(chrom, bfile, chr_gwas, plink_bin, params, tmpdir):
    """
    Stage 1 for one chromosome.
    Returns:
        ind_sig_list : list of dicts (one per IndSigSNP)
        candidates   : dict  rsID -> {r2, ind_sig_rsID, gwas_idx}
        ld_pairs     : list of (ind_sig_uniqID, candidate_uniqID, r2)
    """
    if chr_gwas.empty:
        return [], {}, []

    bfile_bed = bfile + ".bed"
    if not os.path.exists(bfile_bed):
        print(f"  [WARN] Reference BED not found: {bfile_bed} — skipping chr{chrom}")
        return [], {}, []

    assoc_file = os.path.join(tmpdir, f"chr{chrom}.assoc")
    write_plink_assoc(chr_gwas, assoc_file)

    print(f"  [CHR{chrom}] Running plink --clump ({len(chr_gwas):,} SNPs)...")
    clump_df = run_plink_clump(
        chrom,
        bfile,
        assoc_file,
        plink_bin,
        params["leadP"],
        params["gwasP"],
        params["r2"],
        params["clump_kb"],
        params["maf"],
        tmpdir,
    )

    if clump_df.empty:
        print(f"  [CHR{chrom}] No significant SNPs found.")
        return [], {}, []

    print(f"  [CHR{chrom}] {len(clump_df)} IndSigSNP(s) identified.")

    # Build fast lookup for this chromosome
    # Note: rsID may appear once per row; use first occurrence if duplicates
    snp_idx = chr_gwas.drop_duplicates("SNP").set_index("SNP")

    ind_sig_list = []
    candidates = {}  # rsID -> {r2, ind_sig_rsID}
    ld_pairs = []

    for _, row in clump_df.iterrows():
        index_rsid = row["SNP"]
        if index_rsid not in snp_idx.index:
            continue

        gwas_row = snp_idx.loc[index_rsid]
        ind_uid = make_uniq_id(
            chrom, int(gwas_row["POS"]), gwas_row["A1"], gwas_row["A2"]
        )

        # The IndSigSNP itself is always a candidate (r2 = 1.0)
        if index_rsid not in candidates:
            candidates[index_rsid] = {
                "r2": 1.0,
                "ind_sig_rsID": index_rsid,
                "gwas_row": gwas_row,
            }

        ld_pairs.append((ind_uid, ind_uid, 1.0))

        # Parse SP2 for clumped (candidate) SNPs
        sp2_entries = parse_sp2(row.get("SP2", "NONE"))
        n_gwas = 1  # count the index SNP itself

        for clump_rsid, clump_r2 in sp2_entries:
            if clump_rsid not in snp_idx.index:
                continue
            cs_row = snp_idx.loc[clump_rsid]
            cs_uid = make_uniq_id(chrom, int(cs_row["POS"]), cs_row["A1"], cs_row["A2"])
            ld_pairs.append((ind_uid, cs_uid, clump_r2))

            # Assign candidate to IndSigSNP with highest r2
            if clump_rsid not in candidates or candidates[clump_rsid]["r2"] < clump_r2:
                candidates[clump_rsid] = {
                    "r2": clump_r2,
                    "ind_sig_rsID": index_rsid,
                    "gwas_row": cs_row,
                }
            n_gwas += 1

        ind_sig_list.append(
            {
                "rsID": index_rsid,
                "chr": int(chrom),
                "pos": int(gwas_row["POS"]),
                "p": float(gwas_row["P"]),
                "A1": gwas_row["A1"],
                "A2": gwas_row["A2"],
                "uniqID": ind_uid,
                "nSNPs": n_gwas,
                "nGWASSNPs": n_gwas,  # all local candidates are GWAS-tagged
            }
        )

    return ind_sig_list, candidates, ld_pairs


def process_chromosome_with_ld(chrom, chr_gwas, ld_table, params):
    """
    Stage 1 for one chromosome using a pre-computed LDTable.
    Pure-Python greedy clumping — same algorithm as FUMA getLD.py.

    Iterates GWAS SNPs in ascending p-value order.  The first SNP below
    leadP that has not already been claimed becomes an IndSigSNP.  All
    GWAS SNPs in its LD block (r2 >= r2 threshold, p <= gwasP) are claimed
    as candidates.

    Returns same tuple as process_chromosome():
        ind_sig_list, candidates, ld_pairs
    """
    if chr_gwas.empty:
        return [], {}, []

    # Deduplicated SNP index for fast row lookup (pandas Series per SNP)
    snp_idx = chr_gwas.drop_duplicates("SNP").set_index("SNP")

    # SNPs eligible to become candidates (p <= gwasP)
    gwas_set = set(chr_gwas.loc[chr_gwas["P"] <= params["gwasP"], "SNP"])

    # Significant SNPs sorted by p-value (ascending)
    sig_df = (
        chr_gwas[chr_gwas["P"] <= params["leadP"]]
        .drop_duplicates("SNP")
        .sort_values("P")
    )
    if sig_df.empty:
        return [], {}, []

    print(
        f"  [CHR{chrom}] Using pre-computed LD table "
        f"({len(sig_df)} candidate index SNPs)..."
    )

    assigned = set()  # SNPs already claimed by an IndSigSNP
    ind_sig_list = []
    candidates = {}
    ld_pairs = []

    for _, index_snp in sig_df.iterrows():
        rsid = index_snp["SNP"]
        if rsid in assigned:
            continue

        assigned.add(rsid)
        ind_uid = make_uniq_id(
            chrom, int(index_snp["POS"]), index_snp["A1"], index_snp["A2"]
        )

        # IndSigSNP is its own candidate at r2 = 1.0
        candidates[rsid] = {"r2": 1.0, "ind_sig_rsID": rsid, "gwas_row": index_snp}
        ld_pairs.append((ind_uid, ind_uid, 1.0))
        n_clump = 1

        # Collect all GWAS-tagged LD neighbours above r2 threshold
        for nbr_rsid, r2 in ld_table.get_neighbors(rsid, params["r2"]):
            if nbr_rsid not in gwas_set:
                continue
            if nbr_rsid in assigned:
                continue
            if nbr_rsid not in snp_idx.index:
                continue

            assigned.add(nbr_rsid)
            nbr_row = snp_idx.loc[nbr_rsid]
            nbr_uid = make_uniq_id(
                chrom, int(nbr_row["POS"]), nbr_row["A1"], nbr_row["A2"]
            )
            ld_pairs.append((ind_uid, nbr_uid, r2))
            n_clump += 1

            # Assign candidate to IndSigSNP with highest r2
            if nbr_rsid not in candidates or candidates[nbr_rsid]["r2"] < r2:
                candidates[nbr_rsid] = {
                    "r2": r2,
                    "ind_sig_rsID": rsid,
                    "gwas_row": nbr_row,
                }

        ind_sig_list.append(
            {
                "rsID": rsid,
                "chr": int(chrom),
                "pos": int(index_snp["POS"]),
                "p": float(index_snp["P"]),
                "A1": index_snp["A1"],
                "A2": index_snp["A2"],
                "uniqID": ind_uid,
                "nSNPs": n_clump,
                "nGWASSNPs": n_clump,
            }
        )

    print(
        f"  [CHR{chrom}] {len(ind_sig_list)} IndSigSNP(s) "
        f"({len(candidates)} candidates)"
    )
    return ind_sig_list, candidates, ld_pairs


def define_lead_snps(ind_sig_list, pairwise_r2_df, r2_2):
    """
    Stage 2: Group IndSigSNPs into LeadSNPs via greedy union-find at r2_2.

    IndSigSNPs are processed in ascending p-value order (most significant first).
    An IndSigSNP becomes a LeadSNP if it hasn't already been assigned to a
    previously processed LeadSNP via LD >= r2_2.

    Returns dict: rsID -> lead_rsID
    """
    if not ind_sig_list:
        return {}

    # Sort by p-value ascending
    sorted_sigs = sorted(ind_sig_list, key=lambda x: x["p"])

    # Build adjacency set
    ld_neighbors = defaultdict(set)
    snp_set = {s["rsID"] for s in sorted_sigs}

    for _, row in pairwise_r2_df.iterrows():
        a, b, r = row["SNP_A"], row["SNP_B"], float(row["R2"])
        if a in snp_set and b in snp_set and a != b and r >= r2_2:
            ld_neighbors[a].add(b)
            ld_neighbors[b].add(a)

    assignment = {}  # rsID -> lead_rsID
    checked = set()

    for sig in sorted_sigs:
        rsid = sig["rsID"]
        if rsid in checked:
            continue
        # This SNP is a new LeadSNP; mark its LD neighbors as assigned to it
        assignment[rsid] = rsid
        checked.add(rsid)
        for neighbor in ld_neighbors.get(rsid, set()):
            if neighbor not in checked:
                checked.add(neighbor)
                assignment[neighbor] = rsid

    # Any remaining unassigned IndSigSNP is its own LeadSNP
    for sig in sorted_sigs:
        if sig["rsID"] not in assignment:
            assignment[sig["rsID"]] = sig["rsID"]

    return assignment


def merge_genomic_loci(raw_loci, merge_dist):
    """
    Stage 3: Merge raw loci (one per LeadSNP) that are on the same chromosome
    and within merge_dist bp of each other.

    raw_loci: list of dicts with keys:
      chr, start, end, top_rsID, top_p, top_uniqID, top_pos,
      lead_rsIDs, ind_sig_rsIDs, candidate_uids
    Returns merged list, sorted by chr then start.
    """
    if not raw_loci:
        return []

    # Sort by chromosome then start position
    loci = sorted(raw_loci, key=lambda x: (x["chr"], x["start"]))
    merged = [dict(loci[0])]  # deep-ish copy

    for locus in loci[1:]:
        prev = merged[-1]
        # Merge if same chromosome and gap <= merge_dist
        if locus["chr"] == prev["chr"] and locus["start"] - prev["end"] <= merge_dist:
            prev["end"] = max(prev["end"], locus["end"])
            prev["start"] = min(prev["start"], locus["start"])
            prev["lead_rsIDs"] += locus["lead_rsIDs"]
            prev["ind_sig_rsIDs"] += locus["ind_sig_rsIDs"]
            prev["candidate_uids"] += locus["candidate_uids"]
            # Update top SNP to the one with lowest p
            if locus["top_p"] < prev["top_p"]:
                prev["top_rsID"] = locus["top_rsID"]
                prev["top_p"] = locus["top_p"]
                prev["top_uniqID"] = locus["top_uniqID"]
                prev["top_pos"] = locus["top_pos"]
        else:
            merged.append(dict(locus))

    # De-duplicate lists inside each merged locus
    for locus in merged:
        locus["lead_rsIDs"] = list(dict.fromkeys(locus["lead_rsIDs"]))
        locus["ind_sig_rsIDs"] = list(dict.fromkeys(locus["ind_sig_rsIDs"]))
        locus["candidate_uids"] = list(dict.fromkeys(locus["candidate_uids"]))

    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Local FUMA-equivalent SNP clumping pipeline (plink 1.9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gwas",
        required=True,
        help="GWAS summary statistics (fastGWA format: "
        "CHR SNP POS A1 A2 N AF1 BETA SE P)",
    )
    parser.add_argument(
        "--bfile-pattern",
        default=None,
        help="Plink BED file prefix with {} as chromosome placeholder. "
        "Required only when --ld-dir is not provided (or an LD table is missing "
        "for a chromosome). E.g., /path/1000G_EUR/EUR.chr{}",
    )
    parser.add_argument(
        "--plink", default="<path/to>/plink", help="Path to plink 1.9 binary"
    )
    parser.add_argument("--out", default="./clump_output", help="Output directory")
    # Stage 1 parameters
    parser.add_argument(
        "--lead-p",
        type=float,
        default=LEAD_P,
        help="P-value threshold for IndSigSNPs (leadP)",
    )
    parser.add_argument(
        "--gwas-p",
        type=float,
        default=GWAS_P,
        help="P-value threshold for candidate SNPs (gwasP)",
    )
    parser.add_argument(
        "--r2", type=float, default=R2, help="LD r2 threshold for IndSigSNPs / clumping"
    )
    # Stage 2 parameters
    parser.add_argument(
        "--r2-2",
        type=float,
        default=R2_2,
        help="LD r2 threshold for grouping IndSigSNPs into LeadSNPs",
    )
    # Filters
    parser.add_argument(
        "--maf",
        type=float,
        default=MAF,
        help="Minimum MAF in reference panel (passed to plink)",
    )
    parser.add_argument(
        "--merge-dist",
        type=int,
        default=MERGE_DIST,
        help="Max bp gap between LD blocks to merge into one locus",
    )
    parser.add_argument(
        "--no-mhc",
        action="store_true",
        help="Exclude MHC region (chr6:29614758-33170276, hg19)",
    )
    parser.add_argument("--mhc-start", type=int, default=MHC_START)
    parser.add_argument("--mhc-end", type=int, default=MHC_END)
    parser.add_argument(
        "--chroms",
        nargs="+",
        type=int,
        default=list(range(1, 23)),
        help="Chromosomes to process",
    )
    parser.add_argument(
        "--ld-dir",
        default=None,
        help="Directory of pre-computed LD tables produced by "
        "build_ld_table.py (ld_chr{N}.tsv.gz). "
        "When provided, skips plink subprocess for LD "
        "computation (much faster). "
        "Falls back to plink if a chromosome table is missing.",
    )
    parser.add_argument(
        "--dbsnp",
        default=None,
        metavar="BIM_OR_PARQUET",
        help="Reference file for rsID harmonization. Accepts: "
        "(1) Parquet file from build_ref_map.py [fast, recommended], "
        "(2) plain or gzipped plink BIM (CHR rsID cM POS A1 A2, no header) [slow]. "
        "When provided: GWAS rsIDs are updated, strand flips corrected, "
        "SNPs absent from reference or with irreconcilable alleles removed. "
        "Alias for --ref-map (both flags are equivalent).",
    )
    parser.add_argument(
        "--ref-map",
        default=None,
        metavar="PARQUET",
        help="Pre-built Parquet reference map from build_ref_map.py. "
        "Equivalent to --dbsnp but makes the fast-path explicit. "
        "Build once with: python build_ref_map.py --bim ukb_mri.bim --out ref_map.parquet",
    )
    parser.add_argument(
        "--save-harmonized",
        default=None,
        metavar="PATH",
        help="After harmonization, write the harmonized GWAS "
        "DataFrame to this gzipped TSV file (e.g. "
        "harmonized.tsv.gz). Enables fast re-runs via "
        "--load-harmonized.",
    )
    parser.add_argument(
        "--load-harmonized",
        default=None,
        metavar="PATH",
        help="Load a previously saved harmonized GWAS file "
        "(written by --save-harmonized) and skip all "
        "harmonization steps (read_gwas, harmonize_gwas, "
        "harmonize_with_reference).",
    )
    parser.add_argument(
        "--clump-kb",
        type=float,
        default=CLUMP_KB,
        help="plink --clump-kb / --ld-window-kb, in kb. The default of 1000 "
        "matches FUMA, which builds its LD tables at plink's default 1000 kb "
        "window and so never links SNPs more than 1 Mb apart. Raising this "
        "merges blocks FUMA keeps separate across long-range LD (e.g. the "
        "8p23.1 inversion) and will not reproduce FUMA output.",
    )
    args = parser.parse_args()

    params = {
        "leadP": args.lead_p,
        "gwasP": args.gwas_p,
        "r2": args.r2,
        "r2_2": args.r2_2,
        "maf": args.maf,
        "merge_dist": args.merge_dist,
        "clump_kb": args.clump_kb,
    }

    os.makedirs(args.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="fuma_clump_")
    print(f"[INFO] Temp directory: {tmpdir}")

    try:
        # ------------------------------------------------------------------ #
        # Step 1: Read, filter, and harmonize GWAS                           #
        # ------------------------------------------------------------------ #
        if args.load_harmonized:
            print(f"[INFO] Loading pre-harmonized GWAS from {args.load_harmonized}")
            if args.load_harmonized.endswith(".parquet"):
                gwas_df = pd.read_parquet(args.load_harmonized)
                gwas_df["CHR"] = gwas_df["CHR"].astype(str)
            else:
                gwas_df = pd.read_csv(args.load_harmonized, sep="\t", dtype={"CHR": str})
            print(f"[INFO] Loaded {len(gwas_df):,} SNPs")
        else:
            gwas_df = read_gwas(args.gwas, args.no_mhc, args.mhc_start, args.mhc_end)
            gwas_df = harmonize_gwas(gwas_df)
            # --ref-map and --dbsnp are equivalent; --ref-map takes precedence
            ref_path = args.ref_map or args.dbsnp
            if ref_path:
                # Load reference once and pass as DataFrame to avoid re-reading
                # across any future per-chromosome calls
                _ref_df = _load_ref_as_dataframe(ref_path)
                gwas_df = harmonize_with_reference(gwas_df, ref_path, _ref_df=_ref_df)

        if args.save_harmonized:
            print(f"[INFO] Saving harmonized GWAS to {args.save_harmonized}")
            if args.save_harmonized.endswith(".parquet"):
                gwas_df.to_parquet(args.save_harmonized, index=False, compression="snappy")
            else:
                gwas_df.to_csv(
                    args.save_harmonized, sep="\t", index=False, compression="gzip"
                )
            print(f"[INFO] Saved {len(gwas_df):,} SNPs")

        # Global SNP lookup (by rsID); use first occurrence for duplicates
        snp_lookup = gwas_df.drop_duplicates("SNP").set_index("SNP")

        # ------------------------------------------------------------------ #
        # Step 2: Per-chromosome Stage 1 (IndSigSNPs + candidate SNPs)       #
        # ------------------------------------------------------------------ #
        all_ind_sig = []  # list of dicts
        all_candidates = {}  # rsID -> {r2, ind_sig_rsID, gwas_row}
        all_ld_pairs = []  # list of (ind_uid, cand_uid, r2)
        ld_cache = {}  # {chrom: LDTable} – reused in Stage 2

        # Only load LD tables for chromosomes that actually have SNPs at P < leadP.
        # Chromosomes with no genome-wide significant hits need no LD computation.
        chroms_with_sig = set(
            gwas_df.loc[gwas_df["P"] <= params["leadP"], "CHR"].unique()
        )
        print(
            f"[INFO] {len(chroms_with_sig)} chromosome(s) with P<{params['leadP']:.0e}: "
            f"{sorted(chroms_with_sig, key=int)}"
        )

        for chrom in args.chroms:
            if args.bfile_pattern:
                bfile = args.bfile_pattern.replace("{}", str(chrom))
            else:
                bfile = None
            chr_gwas = gwas_df[gwas_df["CHR"] == str(chrom)].copy()

            if chr_gwas.empty:
                continue

            # Skip chromosomes carrying no genome-wide significant SNP. Neither
            # path can yield a lead there -- plink --clump-p1 returns nothing --
            # so this is an efficiency change only. The `and not bfile` clause
            # this replaces disabled the skip whenever a reference BED was
            # supplied, making plink run on all 22 chromosomes every time.
            if str(chrom) not in chroms_with_sig:
                continue

            # Use pre-computed LD table when available; fall back to plink.
            # Accepts both new .npz format and legacy .tsv.gz format.
            if args.ld_dir:
                npz_exists = os.path.exists(
                    os.path.join(args.ld_dir, f"ld_chr{chrom}.npz")
                ) and os.path.exists(
                    os.path.join(args.ld_dir, f"ld_chr{chrom}_snps.npy")
                )
                tsv_exists = os.path.exists(
                    os.path.join(args.ld_dir, f"ld_chr{chrom}.tsv.gz")
                )
                if npz_exists or tsv_exists:
                    tbl = LDTable().load(args.ld_dir, chrom)
                    ld_cache[chrom] = tbl
                    ind_sigs, candidates, ld_pairs = process_chromosome_with_ld(
                        chrom, chr_gwas, tbl, params
                    )
                else:
                    print(
                        f"  [CHR{chrom}] LD table missing in {args.ld_dir}, "
                        f"falling back to plink"
                    )
                    if bfile is None:
                        raise RuntimeError(
                            f"LD table missing for chr{chrom} and --bfile-pattern "
                            f"not provided. Either supply --bfile-pattern or ensure "
                            f"all LD tables are present in {args.ld_dir}."
                        )
                    ind_sigs, candidates, ld_pairs = process_chromosome(
                        chrom, bfile, chr_gwas, args.plink, params, tmpdir
                    )
            else:
                if bfile is None:
                    raise RuntimeError(
                        "--bfile-pattern is required when --ld-dir is not provided."
                    )
                ind_sigs, candidates, ld_pairs = process_chromosome(
                    chrom, bfile, chr_gwas, args.plink, params, tmpdir
                )

            all_ind_sig.extend(ind_sigs)
            all_ld_pairs.extend(ld_pairs)

            for rsid, info in candidates.items():
                if (
                    rsid not in all_candidates
                    or all_candidates[rsid]["r2"] < info["r2"]
                ):
                    all_candidates[rsid] = info

        if not all_ind_sig:
            print("[INFO] No significant SNPs found across all chromosomes. Exiting.")
            return

        print(f"\n[INFO] Total IndSigSNPs: {len(all_ind_sig)}")
        print(f"[INFO] Total candidate SNPs: {len(all_candidates)}")

        # ------------------------------------------------------------------ #
        # Step 3: Stage 2 – define LeadSNPs per chromosome                   #
        # ------------------------------------------------------------------ #
        lead_assignment = {}  # rsID -> lead_rsID

        for chrom in args.chroms:
            chr_ind_sig = [s for s in all_ind_sig if s["chr"] == chrom]
            if not chr_ind_sig:
                continue

            if len(chr_ind_sig) == 1:
                s = chr_ind_sig[0]
                lead_assignment[s["rsID"]] = s["rsID"]
                continue

            snp_list = [s["rsID"] for s in chr_ind_sig]
            print(
                f"\n[INFO] Stage 2 – pairwise r2 among {len(snp_list)} "
                f"IndSigSNPs on chr{chrom}..."
            )

            if chrom in ld_cache:
                # Use pre-computed LD table (O(1) per pair)
                pairs_list = []
                for i, sna in enumerate(snp_list):
                    for snb in snp_list[i + 1 :]:
                        r2 = ld_cache[chrom].get_r2(sna, snb)
                        if r2 >= params["r2_2"]:
                            pairs_list.append({"SNP_A": sna, "SNP_B": snb, "R2": r2})
                pairwise_r2 = (
                    pd.DataFrame(pairs_list)
                    if pairs_list
                    else pd.DataFrame(columns=["SNP_A", "SNP_B", "R2"])
                )
            else:
                if not args.bfile_pattern:
                    raise RuntimeError(
                        "--bfile-pattern is required for Stage 2 plink fallback."
                    )
                bfile = args.bfile_pattern.replace("{}", str(chrom))
                pairwise_r2 = run_plink_r2_among_snps(
                    chrom,
                    snp_list,
                    bfile,
                    args.plink,
                    params["r2_2"],
                    params["clump_kb"],
                    params["maf"],
                    tmpdir,
                )

            assignments = define_lead_snps(chr_ind_sig, pairwise_r2, params["r2_2"])
            lead_assignment.update(assignments)

        # Any IndSigSNP with no assignment is its own LeadSNP
        for s in all_ind_sig:
            if s["rsID"] not in lead_assignment:
                lead_assignment[s["rsID"]] = s["rsID"]

        # ------------------------------------------------------------------ #
        # Step 4: Build per-LeadSNP loci, then merge (Stage 3)               #
        # ------------------------------------------------------------------ #

        # Group IndSigSNPs by their LeadSNP
        lead_to_indsig = defaultdict(list)
        for rsid, lead_rsid in lead_assignment.items():
            lead_to_indsig[lead_rsid].append(rsid)

        raw_loci = []
        for lead_rsid, indsig_rsids in lead_to_indsig.items():
            if lead_rsid not in snp_lookup.index:
                continue
            lead_row = snp_lookup.loc[lead_rsid]
            chrom = int(lead_row["CHR"])

            # Collect all candidate SNP uniqIDs for all IndSigSNPs in this group
            cand_uids = []
            for isid in indsig_rsids:
                for crsid, cinfo in all_candidates.items():
                    if cinfo["ind_sig_rsID"] == isid:
                        cr = cinfo["gwas_row"]
                        cand_uids.append(
                            make_uniq_id(chrom, int(cr["POS"]), cr["A1"], cr["A2"])
                        )

            # Determine locus boundaries from candidate positions
            pos_list = []
            for uid in cand_uids:
                parts = uid.split(":")
                if len(parts) >= 2:
                    try:
                        pos_list.append(int(parts[1]))
                    except ValueError:
                        pass
            if not pos_list:
                pos_list = [int(lead_row["POS"])]

            raw_loci.append(
                {
                    "chr": chrom,
                    "start": min(pos_list),
                    "end": max(pos_list),
                    "top_rsID": lead_rsid,
                    "top_p": float(lead_row["P"]),
                    "top_uniqID": lead_row["uniqID"],
                    "top_pos": int(lead_row["POS"]),
                    "lead_rsIDs": [lead_rsid],
                    "ind_sig_rsIDs": list(indsig_rsids),
                    "candidate_uids": cand_uids,
                }
            )

        print(
            f"\n[INFO] Merging {len(raw_loci)} raw loci "
            f"(merge_dist={params['merge_dist']} bp)..."
        )
        merged_loci = merge_genomic_loci(raw_loci, params["merge_dist"])
        print(f"[INFO] {len(merged_loci)} genomic risk loci after merging")

        # ------------------------------------------------------------------ #
        # Build reverse lookups for output                                    #
        # ------------------------------------------------------------------ #

        # rsID -> locus index (1-based)
        indsig_to_locus = {}
        for locus_i, locus in enumerate(merged_loci):
            for rsid in locus["ind_sig_rsIDs"]:
                indsig_to_locus[rsid] = locus_i + 1

        # rsID -> LeadSNP's IndSigSNPs
        lead_to_indsig_dedup = {
            k: list(dict.fromkeys(v)) for k, v in lead_to_indsig.items()
        }

        # ------------------------------------------------------------------ #
        # Step 5: Write output files                                          #
        # ------------------------------------------------------------------ #

        # --- IndSigSNPs.txt ---
        indsig_rows = []
        for i, sig in enumerate(all_ind_sig):
            gl = indsig_to_locus.get(sig["rsID"], 0)
            indsig_rows.append(
                {
                    "No": i + 1,
                    "GenomicLocus": gl,
                    "uniqID": sig["uniqID"],
                    "rsID": sig["rsID"],
                    "chr": sig["chr"],
                    "pos": sig["pos"],
                    "p": sig["p"],
                    "nSNPs": sig["nSNPs"],
                    "nGWASSNPs": sig["nGWASSNPs"],
                }
            )
        indsig_out = pd.DataFrame(indsig_rows)
        indsig_path = os.path.join(args.out, "IndSigSNPs.txt")
        indsig_out.to_csv(indsig_path, sep="\t", index=False)
        print(f"[OUT] IndSigSNPs.txt  ({len(indsig_out)} rows)")

        # --- leadSNPs.txt ---
        lead_rows = []
        lead_no = 1
        for locus_i, locus in enumerate(merged_loci):
            for lead_rsid in locus["lead_rsIDs"]:
                if lead_rsid not in snp_lookup.index:
                    continue
                lr = snp_lookup.loc[lead_rsid]
                isids = lead_to_indsig_dedup.get(lead_rsid, [lead_rsid])
                lead_rows.append(
                    {
                        "No": lead_no,
                        "GenomicLocus": locus_i + 1,
                        "uniqID": lr["uniqID"],
                        "rsID": lead_rsid,
                        "chr": int(lr["CHR"]),
                        "pos": int(lr["POS"]),
                        "p": float(lr["P"]),
                        "nIndSigSNPs": len(isids),
                        "IndSigSNPs": ";".join(isids),
                    }
                )
                lead_no += 1
        lead_out = pd.DataFrame(lead_rows)
        lead_path = os.path.join(args.out, "leadSNPs.txt")
        lead_out.to_csv(lead_path, sep="\t", index=False)
        print(f"[OUT] leadSNPs.txt    ({len(lead_out)} rows)")

        # --- GenomicRiskLoci.txt ---
        loci_rows = []
        for locus_i, locus in enumerate(merged_loci):
            unique_cands = list(dict.fromkeys(locus["candidate_uids"]))
            unique_ind_sigs = list(dict.fromkeys(locus["ind_sig_rsIDs"]))
            unique_leads = list(dict.fromkeys(locus["lead_rsIDs"]))
            loci_rows.append(
                {
                    "GenomicLocus": locus_i + 1,
                    "uniqID": locus["top_uniqID"],
                    "rsID": locus["top_rsID"],
                    "chr": locus["chr"],
                    "pos": locus["top_pos"],
                    "p": locus["top_p"],
                    "start": locus["start"],
                    "end": locus["end"],
                    "nSNPs": len(unique_cands),
                    "nGWASSNPs": len(unique_cands),
                    "nIndSigSNPs": len(unique_ind_sigs),
                    "IndSigSNPs": ";".join(unique_ind_sigs),
                    "nLeadSNPs": len(unique_leads),
                    "LeadSNPs": ";".join(unique_leads),
                }
            )
        loci_out = pd.DataFrame(loci_rows)
        loci_path = os.path.join(args.out, "GenomicRiskLoci.txt")
        loci_out.to_csv(loci_path, sep="\t", index=False)
        print(f"[OUT] GenomicRiskLoci.txt ({len(loci_out)} rows)")

        # --- snps.txt ---
        # All candidate SNPs with LD info
        snp_rows = []
        for rsid, info in all_candidates.items():
            gr = info["gwas_row"]
            isid = info["ind_sig_rsID"]
            gl = indsig_to_locus.get(isid, 0)
            uid = make_uniq_id(int(gr["CHR"]), int(gr["POS"]), gr["A1"], gr["A2"])
            row = {
                "uniqID": uid,
                "rsID": rsid,
                "chr": gr["CHR"],
                "pos": int(gr["POS"]),
                "non_effect_allele": gr["A2"],
                "effect_allele": gr["A1"],
                "MAF": gr.get("MAF", "NA"),
                "gwasP": float(gr["P"]),
                "r2": info["r2"],
                "IndSigSNP": isid,
                "GenomicLocus": gl,
            }
            # Optional effect columns
            for col in ("BETA", "SE"):
                if col in gr.index:
                    row[col.lower()] = gr[col]
            snp_rows.append(row)

        snps_out = pd.DataFrame(snp_rows)
        snps_out = snps_out.sort_values(["chr", "pos"])
        snps_path = os.path.join(args.out, "snps.txt")
        snps_out.to_csv(snps_path, sep="\t", index=False)
        print(f"[OUT] snps.txt        ({len(snps_out)} rows)")

        # --- ld.txt ---
        ld_out = pd.DataFrame(all_ld_pairs, columns=["SNP1", "SNP2", "r2"])
        ld_path = os.path.join(args.out, "ld.txt")
        ld_out.to_csv(ld_path, sep="\t", index=False)
        print(f"[OUT] ld.txt          ({len(ld_out)} rows)")

        print(f"\n[DONE] All output written to: {args.out}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
