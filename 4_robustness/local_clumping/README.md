# Local FUMA SNP Clumping Pipeline

**Genome build**: GRCh37 / hg19 throughout  
**Reference cohort**: UK Biobank MRI imaging (35,365 samples)  
**Purpose**: Reproducible GWAS LD clumping for UKB-derived summary statistics, mirroring FUMA SNP2GENE logic with pre-computed LD tables for fast repeat runs.

---

## Design Rationale

### Why pre-compute LD tables?

FUMA uses pre-computed LD tables (built once from a reference panel) rather than computing LD on the fly per-GWAS. We follow the same approach. Pre-computing LD once from the UKB MRI cohort allows:

- **Speed**: Each clumping run uses O(1) CSR lookups against the in-memory table instead of spawning a plink subprocess and waiting for pairwise r² computation.
- **Reproducibility**: All GWASes are clumped against the same LD structure, computed from the same reference samples under the same parameters.
- **Reuse**: The same 16 GB LD table serves every UKB phenotype analysis indefinitely. The one-time compute cost (~2–3 h) is amortized across dozens or hundreds of traits.

### Why filter UKB to rsID-only variants?

The UKB BIM contains a mix of rsID variants (`rs1234567`) and position-format variants (`1:12345_A_G`). The position-format variants arise when a genotyped site has no dbSNP entry. Cross-checking against dbSNP ensures every SNP in the pipeline has an rsID that can be looked up consistently — in the LD table, in the harmonized GWAS output, and in downstream annotation tools.

### Why FUMA-style harmonization (CHR:POS matching, not rsID matching)?

This applies only when processing **external / non-UKB sumstats** via `--ref-map`. GWAS software often reports rsIDs from the input BIM file, which may differ from the canonical dbSNP rsID for a given position (due to dbSNP merges, pipeline differences, etc.). Matching on CHR:POS instead of rsID avoids false misses from rsID discordance. After position matching, rsIDs are updated to the reference's canonical value and alleles are strand-checked.

For UKB sumstats (the primary use case), this step is **skipped**: the GWAS was run on the same `ukb_mri_dbsnp` BED used to build the LD tables, so rsIDs are already consistent and clumping is P-value–based (allele alignment is irrelevant).

### Why a separate ref map parquet?

Required only for external GWAS (`--ref-map`). The full dbSNP b151 BIM (27M variants) and even the UKB BIM (8.1M variants) are slow to merge in plain text. A pre-indexed parquet file allows the 8.1M-row CHR:POS → rsID+alleles lookup to complete in ~12s, versus minutes for text. The parquet also stores a pre-computed `_ikey = CHR×10⁹+POS` column to avoid recomputing it on every harmonization call.

---

## Architecture

```
One-time setup (run once for the UKB MRI cohort):
  ukb_ref/ukb_mri_dbsnp.{bed,bim,fam}   ← UKB ∩ dbSNP filtered BED
  ukb_ref/ukb_mri_dbsnp_ref_map.parquet  ← harmonization lookup (external GWAS only)
  ld_tables/ld_chr{1-22}.npz             ← per-chrom sparse LD matrices
  ld_tables/ld_chr{1-22}_snps.npy        ← SNP index arrays

Per-GWAS — UKB sumstats (fast path, ~1 min):
  read GWAS → basic QC (allele filter + dedup)
  clump     → IndSigSNPs → LeadSNPs → Genomic Risk Loci

Per-GWAS — external sumstats (with --ref-map):
  read GWAS → basic QC → harmonize (CHR:POS match, strand flip, rsID update)
  clump     → IndSigSNPs → LeadSNPs → Genomic Risk Loci
```

The key invariant is that **all three artifacts use the same rsID universe** (UKB ∩ dbSNP), so every SNP retained in a harmonized GWAS has a corresponding row in the LD table if it has any LD neighbor with r² ≥ 0.1.

| Artifact | Variant count | Format |
|----------|---------------|--------|
| UKB rsID-only BED | 8,123,147 | plink binary |
| Parquet ref map | 8,118,162 (4,985 duplicate-pos dropped) | Parquet (snappy, 137 MB) |
| LD table (all chrs) | 7,889,367 | CSR numpy, ~16 GB |
| GWAS → LD table coverage | **97.17%** | — |
| Remaining 2.83% | variants with no LD pair ≥ r²=0.1 | — |

All 2.83% of GWAS variants absent from the LD table are variants for which plink found no correlation partner above the r²=0.1 threshold in the 1 Mb window — they cannot be proxy SNPs and their absence is benign for clumping.

---

## File Inventory

| Path | Purpose |
|------|---------|
| `fuma_clump.py` | Main pipeline: harmonization + 3-stage FUMA clumping |
| `build_ld_table.py` | One-time LD table builder (plink → CSR npz) |
| `build_ref_map.py` | One-time ref map builder (BIM → parquet) |
| `build_dbsnp_table.py` | One-time dbSNP BIM builder from NCBI VCF |
| `_save_harmonized.py` | Utility: harmonize a GWAS and save to disk |
| `ukb_ref/ukb_mri.{bed,bim,fam}` | Full UKB MRI panel (8.1M variants including non-rsID) |
| `ukb_ref/ukb_mri_dbsnp_extract.txt` | plink extract list: 8,123,147 rsIDs |
| `ukb_ref/ukb_mri_dbsnp.{bed,bim,fam}` | **Filtered** UKB panel (rsID-only; 67 GB) |
| `ukb_ref/ukb_mri_dbsnp_ref_map.parquet` | Harmonization reference for external GWAS (CHR:POS → rsID + alleles + _ikey; 137 MB; external GWAS only) |
| `ld_tables/ld_chr{N}.npz` | Per-chromosome CSR LD table |
| `ld_tables/ld_chr{N}_snps.npy` | Per-chromosome SNP index (int index → rsID) |
| `dbsnp/dbsnp_b151_GRCh37_common.bim.gz` | Full dbSNP b151 common-variant BIM (used only for build_dbsnp_table.py) |

---

## One-Time Setup

### Step 1 — Build dbSNP BIM (optional; only if `dbsnp/` is missing)

Downloads and converts the NCBI dbSNP b151 common-variants VCF to a plink BIM. This is the original variant universe that defines "has an rsID."

```bash
python build_dbsnp_table.py \
    --out-dir <work_dir>/dbsnp/ \
    --local-vcf <work_dir>/dbsnp/00-common_all.vcf.gz
```

Result: `dbsnp/dbsnp_b151_GRCh37_common.bim.gz` — 27,443,307 biallelic ACGT SNPs on autosomes 1–22.

### Step 2 — Filter UKB BED to rsID-only variants

**Rationale**: Removes the ~3,600 UKB variants whose IDs are in `chr:pos_A_B` format (i.e., not in dbSNP). This ensures the filtered BED, the ref map, and the LD table all share the same rsID-based variant universe.

```bash
# Generate the extract list (rsIDs only: those starting with 'rs')
awk '$2 ~ /^rs/' ukb_ref/ukb_mri.bim | awk '{print $2}' \
    > ukb_ref/ukb_mri_dbsnp_extract.txt

# Filter BED with plink
<path/to>/plink \
    --bfile  ukb_ref/ukb_mri \
    --extract ukb_ref/ukb_mri_dbsnp_extract.txt \
    --make-bed \
    --out    ukb_ref/ukb_mri_dbsnp \
    --chr 1-22 --threads 22
```

Result: `ukb_ref/ukb_mri_dbsnp.{bed,bim,fam}` — 8,123,147 variants, 35,365 samples.

### Step 3 — Build parquet reference map

**Rationale**: The parquet format allows a fast in-memory indexed join (CHR:POS → rsID + alleles) in ~3 min during harmonization, versus ~30 min for line-by-line BIM parsing. The ref map is the single source of truth for rsID assignment in the pipeline.

```bash
python build_ref_map.py \
    --bim  ukb_ref/ukb_mri_dbsnp.bim \
    --out  ukb_ref/ukb_mri_dbsnp_ref_map.parquet
```

Result: `ukb_ref/ukb_mri_dbsnp_ref_map.parquet` — 8,118,162 variants (4,985 duplicate CHR:POS positions dropped), **137 MB**. Includes a pre-computed `_ikey = CHR×10⁹+POS` column for O(1) merge in `harmonize_with_reference()`.

### Step 4 — Build LD tables

**Rationale**: Pre-computing LD allows each clumping run to use O(1) CSR lookups instead of spawning plink. The 1 Mb window and r²≥0.1 threshold match FUMA defaults. Symmetry (both directions stored) allows neighbor queries in either direction without transposing.

```bash
python build_ld_table.py \
    --bfile-single <work_dir>/ukb_ref/ukb_mri_dbsnp \
    --plink <path/to>/plink \
    --out-dir <work_dir>/ld_tables/ \
    --maf 0.01 --r2-min 0.1 --kb-window 1000 \
    --threads 22
```

Runtime: ~2–3 hours (22 chromosomes in parallel). Output: ~16 GB total across 44 files (`.npz` + `_snps.npy` per chromosome).

| Chr | SNPs in LD table | Pairs stored |
|-----|------------------|--------------|
| 1 | 605,063 | 68,176,436 |
| 2 | 663,282 | 80,440,662 |
| 3 | 563,493 | 75,630,438 |
| 4 | 579,523 | 79,351,451 |
| 5 | 514,195 | 68,512,608 |
| 6 | 535,903 | 110,157,924 |
| 7 | 465,609 | 58,359,929 |
| 8 | 440,856 | 60,868,880 |
| 9 | 336,669 | 36,300,329 |
| 10 | 403,570 | 52,221,588 |
| 11 | 395,759 | 66,588,722 |
| 12 | 380,214 | 48,498,618 |
| 13 | 291,116 | 34,457,252 |
| 14 | 257,972 | 32,730,602 |
| 15 | 221,178 | 24,406,573 |
| 16 | 240,066 | 21,122,390 |
| 17 | 205,339 | 23,007,911 |
| 18 | 225,441 | 22,387,085 |
| 19 | 176,445 | 19,521,895 |
| 20 | 174,005 | 15,700,435 |
| 21 | 108,087 | 9,731,406 |
| 22 | 105,582 | 8,996,221 |
| **Total** | **7,889,367** | **1,017,169,355** |

---

## Per-GWAS Usage

Once setup is complete, running a new UKB GWAS takes **~1 minute** (benchmarked: 1m04s for a 9M-SNP fastGWA file with 4 chromosomes having genome-wide hits).

**Runtime breakdown (UKB fast path):**
| Step | Time |
|------|------|
| Read GWAS (9M rows, fastGWA) | ~30s |
| Basic QC (`harmonize_gwas`) | ~2s |
| Load LD tables (only chromosomes with p < leadP) | ~20s (4 chroms) |
| Clumping (Stages 1–3) | ~5s |
| Write output | <1s |

### Standard run (UKB sumstats)

```bash
python <work_dir>/fuma_clump.py \
    --gwas   /path/to/trait.fastGWA \
    --ld-dir <work_dir>/ld_tables/ \
    --out    /path/to/output/ \
    --no-mhc
```

`--no-mhc` excludes chr6:29,614,758–33,170,276 (hg19) to avoid the MHC cluster inflating locus counts.

**`--ref-map` / `--dbsnp` is not needed for UKB sumstats.** The UKB genotype was already QC'd and filtered to rsID-only variants when building the LD tables, so GWAS files produced from the same UKB BED already share the same rsID space. Clumping uses P-values, not effect direction, so allele convention alignment is irrelevant.

### External / non-UKB sumstats

For GWAS produced outside UKB (e.g., public summary statistics), pass `--ref-map` to filter and strand-align against the UKB ∩ dbSNP reference:

```bash
python fuma_clump.py \
    --gwas   /path/to/external_trait.fastGWA \
    --ref-map <work_dir>/ukb_ref/ukb_mri_dbsnp_ref_map.parquet \
    --ld-dir <work_dir>/ld_tables/ \
    --out    /path/to/output/ \
    --no-mhc
```

For repeat runs of the same external GWAS with different clumping parameters, save the harmonized output once:

```bash
# First run: harmonize and save
python fuma_clump.py --gwas /path/to/external_trait.fastGWA \
    --ref-map .../ukb_mri_dbsnp_ref_map.parquet \
    --ld-dir .../ld_tables/ --out .../output/ --no-mhc \
    --save-harmonized /path/to/trait_harmonized.parquet

# Subsequent runs: skip harmonization entirely
python fuma_clump.py \
    --load-harmonized /path/to/trait_harmonized.parquet \
    --ld-dir .../ld_tables/ --out .../output_v2/ --lead-p 1e-6
```

### GWAS input format (fastGWA)

```
CHR  SNP  POS  A1  A2  N  AF1  BETA  SE  P
```

All coordinates must be **GRCh37**. The `SNP` column should contain rsIDs (standard for GCTA/BOLT/regenie output from UKB BED input). Indels and multi-allelic variants are automatically dropped.

### Output files

| File | Description |
|------|-------------|
| `IndSigSNPs.txt` | Independent significant SNPs (p < leadP, pairwise r² < r2_2) |
| `leadSNPs.txt` | Lead SNPs after grouping IndSigSNPs by r² |
| `GenomicRiskLoci.txt` | Merged genomic risk loci with genomic boundaries |
| `snps.txt` | All candidate SNPs in LD with an IndSigSNP |
| `ld.txt` | Pairwise r² between IndSigSNPs and their candidates |

---

## Algorithm — FUMA Three-Stage Clumping

### GWAS harmonization (before clumping)

**Pass 1 — basic QC** (`harmonize_gwas`): always runs.

| Check | Action |
|-------|--------|
| Non-ACGT alleles | Remove (indels, structural, etc.) |
| Ambiguous SNPs (A/T or G/C) | **Kept** — P-value signal is independent of strand orientation, so strand ambiguity does not affect clumping |
| Duplicate CHR:POS | Keep lowest-p variant |

**Pass 2 — reference harmonization** (`harmonize_with_reference`): **optional, only needed for non-UKB sumstats** (enabled via `--ref-map` / `--dbsnp`).

For UKB sumstats this pass is skipped: the GWAS was run on the same `ukb_mri_dbsnp` BED that was used to build the LD tables, so rsIDs and alleles are already consistent. Clumping is P-value-based and does not require allele direction alignment.

For external sumstats, this pass merges on `CHR:POS` against the parquet ref map (position-based to handle rsID discordance):

| Step | Action |
|------|--------|
| Position match | Drop GWAS SNPs absent from the UKB ∩ dbSNP universe |
| rsID update | Assign canonical rsID from the ref map |
| Strand flip | If alleles are complement-matches (A↔T, G↔C), flip A1/A2 and recompute `uniqID` |
| Allele mismatch | Drop SNPs that don't match the reference alleles even after strand flip |

**`uniqID` format**: `chr:pos:allele_min:allele_max` with alleles alphabetically sorted. Strand-agnostic.

### Stage 1 — Independent Significant SNPs

Greedy selection using the pre-computed CSR LD table:
1. Sort GWAS SNPs by p-value ascending.
2. The top unassigned SNP below `leadP` (default 5×10⁻⁸) becomes an IndSigSNP.
3. All GWAS SNPs within its LD block (r² ≥ 0.6, p ≤ 0.05, within 1 Mb) are claimed as candidates via O(1) CSR row slice.
4. Repeat until no unassigned SNPs remain below `leadP`.

### Stage 2 — Lead SNPs

IndSigSNPs are grouped by inter-SNP LD. Processed in ascending p-value order: an IndSigSNP with r² ≥ 0.1 to an already-declared LeadSNP is merged into that LeadSNP's group.

### Stage 3 — Genomic Risk Loci

LeadSNP LD blocks on the same chromosome are merged if their outer boundaries are ≤ 250 kb apart. The LeadSNP with the lowest p-value in each merged locus is the locus representative.

### Default parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--lead-p` | 5×10⁻⁸ | P-value threshold for IndSigSNPs |
| `--gwas-p` | 0.05 | P-value threshold for candidate SNPs |
| `--r2` | 0.6 | LD r² for candidate assignment |
| `--r2-2` | 0.1 | LD r² for LeadSNP grouping |
| `--maf` | 0.01 | Minimum MAF in reference panel |
| `--merge-dist` | 250,000 bp | Max gap to merge adjacent loci |
| `--no-mhc` | off | Exclude chr6:25M–34M |
| LD window | 1,000 kb | Pairs beyond 1 Mb not stored |

---

## LD Table Format

### CSR npz (current format)

Two files per chromosome:

| File | Content |
|------|---------|
| `ld_chr{N}.npz` | Three numpy arrays: `ib` (int32, neighbor indices), `r2` (float32, r² values), `indptr` (int32, CSR row pointers; length = n_snps + 1) |
| `ld_chr{N}_snps.npy` | String array: integer index → rsID |

The CSR layout makes neighbor lookup `O(1)`: all neighbors of SNP `i` are at `ib[indptr[i]:indptr[i+1]]` with corresponding r² at `r2[indptr[i]:indptr[i+1]]`. Both directions are stored (symmetric), so a query for "what is in LD with rs123?" works regardless of which partner was listed first in the plink output.

**Load time comparison**:

| Format | Load time (chr17, 23M pairs) | File size |
|--------|------------------------------|-----------|
| gzip TSV (legacy) | ~85 s | 178 MB |
| **CSR npz (current)** | **~0.4 s** | 352 MB |

The 200× load speedup comes from eliminating the Python loop that parsed text and built nested dicts. CSR loads are two `np.load()` calls followed by one dict comprehension for the string→int index.

**Build time** (Python post-processing only, after plink completes): ~5s per chromosome (pandas C engine parse + Categorical encoding for SNP→int32). The plink `--r2` computation itself (~2–3 h for 22 chromosomes in parallel) dominates wall time and is irreducible.

### Rebuilding LD tables

```bash
# Rebuild from scratch (safe to re-run; skips existing .npz by default)
python build_ld_table.py \
    --bfile-single <work_dir>/ukb_ref/ukb_mri_dbsnp \
    --plink <path/to>/plink \
    --out-dir <work_dir>/ld_tables/ \
    --maf 0.01 --r2-min 0.1 --kb-window 1000 --threads 22

# Convert legacy tsv.gz to npz in-place (if applicable)
python build_ld_table.py \
    --convert-existing \
    --out-dir <work_dir>/ld_tables/ \
    --threads 22
```

---

## Validation Against FUMA

The pipeline was validated against public FUMA runs on 6 datasets. Results were compared at the genomic risk locus level.

| Dataset | FUMA ref panel | FUMA loci | Local loci | Lead SNP match | Note |
|---------|---------------|-----------|------------|----------------|------|
| cnn_random (Bonferroni leadP=3.9e-10) | 1KG/Phase3 EUR | 5 | 5 | 5/5 (100%) | Perfect |
| mesh_mask75 (Bonferroni leadP=3.9e-10) | 1KG/Phase3 EUR | 25 | 27 | 25/25 (100%) | 2 extra local |
| vit_rmheadsize (Bonferroni leadP=3.9e-10) | 1KG/Phase3 EUR | 63 | 63 | 63/63 (100%) | Perfect |
| VoxelAge meta (leadP=6.09e-12) | 1KG/Phase3 EUR | 80 | 81 | 80/80 (100%) | 1 extra local |
| 2ch 3DUiN heart (leadP=5e-8) | UKB/release2b 10k EUR | 118 | 139 | 50/118 lead SNP; **101/118 (86%) positional** | Ref panel diff: FUMA=UKB 10k, local=UKB 35k |
| 4ch 3DUiN heart (leadP=5e-8) | UKB/release2b 10k EUR | 155 | 142 | 56/155 lead SNP; **122/155 (79%) positional** | Same cause |

**Interpretation**: When FUMA used 1KG/Phase3 EUR (the standard FUMA LD panel), our UKB 35k-based LD table gives 100% lead SNP agreement. The 3DUiN cases are an exception: those FUMA runs used a UKB release2b 10k EUR subset — a different (smaller) UKB subsample — so LD estimates diverge at some loci, shifting the selected lead SNP within otherwise-overlapping regions (79–86% positional overlap).

The greedy Stage 1 algorithm (sort by P, claim candidates in descending significance order) matches FUMA's approach. FUMA assigns each candidate SNP to the IndSigSNP with maximum r²; our implementation uses first-claim order by ascending p-value. These differ only in how candidate SNPs shared between two nearby IndSigSNPs are attributed, and do not affect locus count.

---

## Known Limitations

1. **Reference-panel SNPs not in GWAS**: FUMA includes all reference-panel SNPs in LD with an IndSigSNP, even those absent from the GWAS input. This pipeline only includes GWAS-tagged SNPs in the output.
2. **LeadSNP assignment**: FUMA assigns each candidate SNP to the IndSigSNP with maximum r². This pipeline uses greedy first-claim order (ascending p-value). Practically equivalent for strong signals.
3. **Functional annotations**: FUMA's `snps.txt` includes CADD scores, RegulomeDB, chromatin state, nearest gene. Not computed here.
4. **X chromosome and non-autosomes**: Excluded (autosomes 1–22 only).
