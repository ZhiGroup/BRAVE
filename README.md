# BRAVE — Brain Region Aggregation of Voxel Embeddings

Code accompanying our study of subcortical genetic architecture through
self-supervised voxel embeddings and a multivariate GWAS test (JAGWAS).

The short version: we train a 3D contrastive encoder on T1 MRI, mean-pool the
per-voxel embeddings within 16 region masks to get a 128-dimensional
**Brain Region Embedding (BRE)** per region per subject, and run a multivariate
joint-test GWAS (JAGWAS) on those 128 dimensions. The 16 regions are 14 bilateral
subcortical structures (seven bilateral pairs), the brainstem with the fourth
ventricle, and the lateral ventricle. The last two are not subcortical, so we
say brain regions rather than subcortical regions throughout. The joint test
recovers far
more genetic signal than univariate GWAS on regional volume alone — 276 loci
versus 60 for a FastGWA min-P baseline on the same cohort, a 4.6-fold increase
without adding subjects.

We are **not** claiming finer spatial resolution; the embeddings are aggregated
to the region level. The claim is representational: a learned embedding carries
volume, shape, and additional morphological signal in one phenotype, and a
multivariate test can use all of it at once.

## Pipeline

```
T1 MRI ─▶ [1] voxel encoder ─▶ per-voxel embeddings
                                     │  mean-pool within 16 masks
                                     ▼
                        128-dim Brain Region Embedding (BRE)
                                     │
                 [2] GWAS ───────────┤  FastGWA per dim  (univariate baseline)
                                     └  JAGWAS           (multivariate joint test)
                                     │
                                     ▼
              [3] post-GWAS: loci, novelty, heritability, genetic
                  correlation (ENIGMA / shape / disease / cardiac / OCT),
                  cell-type & pathway enrichment, PGS validation,
                  and the figures/tables reported in the paper
```

Each numbered directory is a stage you can run independently once its inputs
exist:

| Dir | Stage | What it does |
|-----|-------|--------------|
| [`1_embedding/`](1_embedding/) | Voxel encoder | Train the contrastive encoder, extract BREs and regional shape features |
| [`2_gwas/`](2_gwas/) | GWAS | Build the BGEN/GRM/covariates, run FastGWA per dim, run JAGWAS multivariate test |
| [`3_postgwas/`](3_postgwas/) | Post-GWAS | All downstream analysis and figure scripts (the bulk of the repo) |

## Getting started

1. Install dependencies (see [`requirements.txt`](requirements.txt)). The
   embedding stage needs an older PyTorch/MONAI stack; the post-GWAS analyses
   run on a plain scientific-Python install.

2. Configure paths. The code never hard-codes a location — everything external
   lives in one file:

   ```bash
   cp config/paths.yaml config/paths.local.yaml
   # edit config/paths.local.yaml to point at your data and tools
   ```

   `paths.local.yaml` is git-ignored, so your real paths stay private. See
   [`config/paths.yaml`](config/paths.yaml) for every key.

3. Run a stage. Each stage directory has its own README with the order to run
   things in. The end-to-end run order is in [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Data availability

The individual-level UK Biobank imaging and genotype data used here are
access-controlled and are **not** included in this repository. No subject
identifiers, phenotypes, or genotypes ship with the code. To reproduce the
analyses you need your own approved UK Biobank application; the required input
formats (columns, file layout, region naming) are documented in
[`docs/DATA.md`](docs/DATA.md) so you can map the code onto your copy.

Summary-level outputs (per-region loci, genetic correlations) are reported in
the paper and its supplement.

## Repository layout

```
config/        paths.yaml + loader (the only place paths live)
docs/          DATA.md (input schemas), PIPELINE.md (run order)
1_embedding/   self-supervised encoder + BRE extraction
2_gwas/        BGEN/GRM/covariate prep, FastGWA, JAGWAS
3_postgwas/    numbered analysis scripts, shared figure style, run_all.sh
4_robustness/  replication in a genetically heterogeneous cohort; encoder
               retraining and stability of the reported loci
third_party/   vendored external tools (PoPS), under their own licenses
```

## Citing

If you use this code, please cite the accompanying paper (reference to be added
on publication). Third-party methods we build on — JAGWAS, FastGWA/GCTA, FUMA,
LDSC, MAGMA, PoPS — should be cited separately; see each stage's README.

## License

MIT for the code in this repository, except vendored third-party tools under
`third_party/`, which keep their original licenses. See [`LICENSE`](LICENSE).
