# Stage 2 — GWAS

Two association passes on the 128-dim BREs:

- **FastGWA** (GCTA), run per embedding dimension, with a per-SNP min-P across
  the 128 dimensions. This is the univariate baseline (60 loci).
- **JAGWAS**, a multivariate joint test of all 128 dimensions at once. This is
  the main analysis (276 loci).

The notebooks build the inputs; the Python/R/shell scripts run the association.

## Files

| File | Step |
|------|------|
| `bgen_process.ipynb` | Build the merged BGEN for the MRI subjects. |
| `sparse_grm.ipynb` | Build the GCTA sparse GRM from the kinship table. |
| `covariate_preparation.ipynb` | Assemble the FastGWA covariate files. |
| `proc_pheno.ipynb` | Split BREs into per-region × per-dim phenotype files. |
| `fastgwa_runner.py` | Run FastGWA per dimension and take the min-P. |
| `jagwas_phenotype_correlation.R` | Estimate the residual correlation matrix R per region. |
| `jagwas_phenotype_correlation_cli.R` | Command-line wrapper for the above. |
| `run_parallel_jagwas.py` | Run the R correlation step across all region × cohort pairs. |
| `jagwas_association_run.sh` | Launch JAGWAS for one region × cohort. |
| `jagwas_batch_run.py` | Batch-launch JAGWAS across regions. |
| `check_correlation_matrix.py` | Diagnostic: condition number / eigenvalue spectrum of R. |
| `jagwas_eigenvalue_clamp.md` | Note on the Tikhonov eigenvalue clamp added to JAGWAS. |

## A note on R conditioning

JAGWAS inverts the 128×128 residual correlation matrix R. Some regions have an
ill-conditioned R, which inflates the joint statistic. We clamp small
eigenvalues before inverting; the details and the relevant code are in
[`jagwas_eigenvalue_clamp.md`](jagwas_eigenvalue_clamp.md). Point
`tools.jagwas` at the patched binary.

The notebooks ship with all cell outputs cleared (they had executed on real
subject data). Set the `ukb.*`, `gwas.*`, and `cohorts.*` keys in
`config/paths.local.yaml` before running.
