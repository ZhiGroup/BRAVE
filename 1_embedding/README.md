# Stage 1 — voxel encoder and Brain Region Embeddings

A 3D contrastive encoder is trained on T1 MRI so that two views of the same voxel
land close together in feature space and different voxels land apart. The trained
encoder produces a 128-dim vector per voxel; mean-pooling those vectors inside
each of the 16 subcortical masks gives the Brain Region Embedding (BRE) used for
GWAS.

## The objective

A joint InfoNCE loss with **three equally weighted terms**:

| Term | Positives | Negatives |
|------|-----------|-----------|
| Fine-scale voxel (48³) | 50 voxels per patch, at corresponding locations in two overlapping patches from the same scan | the most similar voxels outside a radius-3 neighbourhood of the positive, drawn from both views |
| Coarse-scale voxel (6³) | as above, on the coarse feature map | as above, on the coarse map |
| Instance | per-patch representations, obtained by projecting and average-pooling each resolution's voxel embeddings, from overlapping patches of the same scan | the remaining patches in the mini-batch |

InfoNCE temperatures are 0.3 (voxel) and 0.4 (instance). There is no
reconstruction term and no momentum encoder or memory bank in this model. The
`Model_Momentum` class in `models.py` is a separate variant that adds them; it is
not the published encoder.

Optimisation is AdamW (max learning rate 0.005, weight decay 0.008) under a
one-cycle schedule stepped once per mini-batch, batch size 4, gradients clipped
at 1.5 by value, single precision, distributed data-parallel across two GPUs.

## Files

| File | What it is |
|------|------------|
| `train_encoder.py` | Training entry point. |
| `models.py` | The encoder: 3D FPN over a modified 3D ResNet-18, projection and instance heads, and the objective in `Model.training_step`. |
| `preprocessing.py` | Patch-pair sampling, the two-view dataset, and the collate function. |
| `experiment_configurations.py` | Hyper-parameters for the published encoder (`conf_model_4_exp_3`). |
| `config.py` | Resolves stage-1 directories from `config/paths.yaml`. |
| `seeding.py` | Seeds every RNG the stack draws from and pins cuDNN behaviour. |
| `extract_regional_embeddings.py` | Runs the encoder over a subject and mean-pools voxel embeddings within the 16 masks → BRE. |
| `extract_region_shape.py` | Regional shape features, for the volume/shape comparison. |

## Run

```bash
# 1. train (published settings: batch size 4, 8 epochs, one-cycle spanning 40)
python train_encoder.py --seed 1

# 2. extract BREs for every subject in a cohort
python extract_regional_embeddings.py
```

Fill in the `embedding:` section of `config/paths.local.yaml` first — in
particular `train_dir` and `val_dir`, which hold one sub-folder per subject with
a `T1_brain.nii.gz` inside. Training uses two GPUs by default (`--devices 0,1`);
pass `--devices 0` for one. The MONAI / Lightning / PyTorch versions are pinned
in the top-level `requirements.txt`.

A handful of resource paths with no natural config key (FSL atlas, raw image
zips) are marked inline as `<EXTERNAL: ...>`; search for that string and point
each at your copy.

## Two settings that are easy to get wrong

**The schedule spans more epochs than you train.** `--scheduler-epochs` (40)
builds the one-cycle learning-rate schedule; `--train-epochs` (8) is where
training stops. The published checkpoint is epoch 7 of a 40-epoch schedule, so
the learning rate is still rising through warm-up at that point. Passing 8 to
both would complete an entire cycle in 8 epochs and give a very different
trajectory.

**Batch size 4.** Several negative-sampling counts are derived from it, so
changing the batch size changes the objective, not just the throughput.

## Reproducibility

Training is seeded end to end. `seeding.set_global_seed` covers the Python, NumPy
and torch RNGs, disables the cuDNN autotuner and pins cuDNN to deterministic
kernels. The dataset derives a per-item RNG from `(seed, index)`, so patch
sampling is invariant to worker count and batch ordering. MONAI's random
transforms are seeded explicitly, because their internal `RandomState` is
otherwise seeded from OS entropy and is not reached by seeding the global RNGs.

Bitwise determinism is not guaranteed on every stack: some 3D operations used
here have no deterministic CUDA kernel, and on PyTorch < 1.11
`torch.use_deterministic_algorithms` cannot be requested in warn-only mode. Seed
control makes runs comparable; it does not make them bit-identical.

A useful invariant if you modify the patch-pair code: the raw image intensities at
a positive pair's two coordinates must be identical, since both views are crops
of the same volume. If they are not, the voxel-level positives are mis-paired, and
the fine-scale term will sit at chance — ln(1 + n_negatives) — while the instance
term keeps learning normally, which makes the failure easy to miss.
