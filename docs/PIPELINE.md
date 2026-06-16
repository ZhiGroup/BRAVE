# End-to-end run order

Each stage assumes the previous one has produced its outputs and that
`config/paths.local.yaml` points at the right locations. Most users will only
ever re-run stage 3 (post-GWAS), since stages 1–2 require the access-controlled
genotype/imaging data and substantial compute.

## Stage 1 — voxel embedding  (`1_embedding/`)

1. Train the contrastive encoder on the encoder train/val split
   (`train_pixpro.py`).
2. Extract per-voxel embeddings and mean-pool them within the 16 masks to get
   the 128-dim BRE per region per subject (`extract_regional_embeddings.py`).
3. (Optional) Extract regional shape features for the volume/shape comparison
   (`extract_region_shape.py`).

Output: a BRE matrix per region (subjects × 128), written under
`embedding.bre_out_dir`.

## Stage 2 — GWAS  (`2_gwas/`)

1. Build the merged BGEN for the MRI subjects (`bgen_process.ipynb`).
2. Build the sparse GRM from the kinship table (`sparse_grm.ipynb`).
3. Prepare covariate files (`covariate_preparation.ipynb`).
4. Split BREs into per-region × per-dim phenotype files (`proc_pheno.ipynb`).
5. Run FastGWA on each dimension and take the per-SNP min-P across dims — this is
   the univariate baseline (`fastgwa_runner.py`).
6. Estimate the residual phenotype correlation matrix R per region
   (`jagwas_phenotype_correlation.R`, driven by `run_parallel_jagwas.py`).
7. Run the JAGWAS multivariate joint test per region × cohort
   (`jagwas_batch_run.py`). See `jagwas_eigenvalue_clamp.md` for the R
   conditioning fix.

Output: per-region multivariate summary statistics, taken into FUMA for
annotation.

## Stage 3 — post-GWAS  (`3_postgwas/`)

Once FUMA outputs and munged summary statistics exist, the whole analysis layer
reproduces from one script:

```bash
cd 3_postgwas
bash run_all.sh
```

Scripts are numbered in dependency order and grouped (per-region → cross-region
→ novelty/replication → heritability/genetic-correlation → PGS). Each writes a
CSV under `results/` and a figure under `figures/`. See
[`../3_postgwas/README.md`](../3_postgwas/README.md) for the per-script table.
