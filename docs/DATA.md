# Input data: formats and layout

This repository ships code only. No subject-level data is included. This file
documents what each stage expects so you can point the code (via
`config/paths.yaml`) at your own approved UK Biobank copy.

Nothing below contains identifiers — only file layouts, column names, and the
public UK Biobank Data-Field codes the pipeline reads.

---

## The 16 regions

Throughout the pipeline a "region" is one of 16 subcortical labels. Directory
and file names use these exact strings:

```
Left_Accumbens-area      Right_Accumbens-area
Left_Amygdala            Right_Amygdala
Left_Caudate             Right_Caudate
Left_Hippocampus         Right_Hippocampus
Left_Pallidum            Right_Pallidum
Left_Putamen             Right_Putamen
Left_Thalamus-Proper     Right_Thalamus-Proper
Brain_Stem_or_4th_Ventricle
CSF
```

(14 lateralized structures + brainstem/4th-ventricle + CSF = 16.)

---

## Cohort CSVs (`cohorts.*`)

One CSV per split: encoder training/validation, GWAS discovery, GWAS
replication, and an inference/held-out set. Each row is one subject. Expected
columns:

| Column | Meaning |
|--------|---------|
| `eid` | UK Biobank subject identifier (your application's IDs) |
| `t1_path` | Path to the linear-MNI-registered T1 volume |
| `brain_mask_path` | Path to the whole-brain mask |
| `subcort_mask_path` | Path to the 16-region subcortical segmentation |

The splits are disjoint by subject. The replication set is additionally pruned
for relatedness against discovery before GWAS.

---

## UK Biobank fields used

These public Data-Field codes are read from your phenotype table
(`ukb.master_csv`). Listed so you can assemble the equivalent columns:

| Field | Use |
|-------|-----|
| `20252` | T1 NIfTI imaging bulk field |
| `21003-2.0` | Age at imaging visit |
| `31-0.0` | Reported sex |
| `22001-0.0` | Genetic sex |
| `21000-2.0` | Self-reported ethnicity |
| `22006-0.0` | Genetic-ancestry grouping indicator |
| `54-2.0` | Imaging assessment centre |
| `53-2.0` | Date of imaging visit (used linearly and squared) |
| `25000` | Volumetric scaling factor (head-size normalisation) |
| `25735` | Inverted T1 contrast-to-noise ratio (image quality) |
| `25756`–`25759` | Scanner table / head position parameters |
| Genetic PCs | First 10 ancestry principal components |

Consent-withdrawn subjects are dropped using the withdrawal list
(`ukb.withdrawal_list`).

---

## Genotypes (`ukb.bgen`, `ukb.bgen_sample`)

Imputed genotypes for the MRI subjects in Oxford BGEN v1.2 format, with the
matching `.sample` file and a `.bgi` index. The sparse GRM (`gwas.sparse_grm`)
is a GCTA sparse-GRM prefix (`.grm.sp` + `.grm.id`) built from the UK Biobank
KING kinship table at a third-degree-relative cutoff.

---

## GWAS phenotypes and covariates

- **Phenotypes** (`gwas.pheno_dir`): one GCTA-format quantitative-trait file per
  region × embedding dimension (128 dims). Three columns: FID, IID, value.
- **Covariates** (`gwas.covar_dir`): FastGWA `--qcovar` (quantitative: 10 PCs,
  age, age², sex×age terms, image-quality and scanner-position fields, visit
  date and its square) and `--covar` (categorical: sex, assessment centre).

---

## JAGWAS inputs

JAGWAS needs, per region: the per-dimension summary statistics and the residual
phenotype **correlation matrix** R (128 × 128), estimated from residualized
embeddings. The correlation matrices are produced by the R step in
[`../2_gwas/`](../2_gwas/) and stored under `gwas.bre_corr_dir`. See
[`../2_gwas/jagwas_eigenvalue_clamp.md`](../2_gwas/jagwas_eigenvalue_clamp.md)
for the conditioning fix applied to R.

---

## FUMA outputs (`postgwas.fuma_dir`)

The post-GWAS scripts read standard FUMA SNP2GENE / GENE2FUNC output. One
sub-folder per region (named exactly as in the region list above), each
containing:

```
GenomicRiskLoci.txt   leadSNPs.txt   IndSigSNPs.txt   snps.txt
genes.txt             magma.genes.out
magma_exp_gtex_v8_ts_avg_log2TPM.gsa.out
eqtl.txt              ci.txt         gwascatalog.txt
```

Cell-type results (GENE2FUNC) are downloaded per region into the same tree.

---

## Reference GWAS for comparison

Novelty and genetic-correlation analyses compare against external GWAS. Supply
LDSC-munged summary statistics (or, for novelty, FUMA loci tables) for:

- ENIGMA subcortical-volume GWAS (`postgwas.enigma_dir`)
- a subcortical-shape GWAS loci table (`postgwas.shape_gwas`)
- brain-disorder GWAS (`postgwas.disease_sumstats_dir`)
- cardiac-MRI trait GWAS (`postgwas.heart_sumstats_dir`)
- retinal-OCT trait GWAS (`postgwas.oct_sumstats_dir`)

Munging convention (variable N): `--N-col N --a1 A1 --a2 A2 --p P --frq AF1`,
no `--merge-alleles`.
