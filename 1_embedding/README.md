# Stage 1 — voxel embedding (BRE encoder)

A 3D contrastive encoder is trained on T1 MRI so that two augmented views of the
same voxel land close together in feature space and different voxels land apart
(voxel-level InfoNCE, in the spirit of self-supervised anatomical embeddings).
The trained encoder produces a 128-dim vector per voxel; mean-pooling those
vectors inside each of the 16 subcortical masks gives the Brain Region Embedding
(BRE) used for GWAS.

## Files

| File | What it is |
|------|------------|
| `train_pixpro.py` | Training entry point (PyTorch Lightning). |
| `pixpro2.py` | The contrastive model: backbone + projection/prediction heads + InfoNCE loss. |
| `resent_pixpro2.py` | 3D ResNet/FPN backbone. |
| `dataloader_pixpro.py` | Patch sampling and the two-view augmentation. |
| `config_pixpro.py` | Path/config shim — pulls everything from `config/paths.yaml`. |
| `config_loader_pixpro.py` | Hyper-parameter loading. |
| `extract_regional_embeddings.py` | Runs the encoder over a subject and mean-pools voxel embeddings within the 16 masks → BRE. |
| `extract_region_shape.py` | Regional shape features (for the volume/shape comparison). |
| `embedding_record.py` | Bookkeeping helper for the extraction pass. |

## Run

```bash
# 1. train
python train_pixpro.py
# 2. extract BREs for every subject in a cohort
python extract_regional_embeddings.py
```

Set `embedding.*` and `cohorts.*` in `config/paths.local.yaml` first. Training
needs a GPU and the older MONAI/Lightning stack pinned in the top-level
`requirements.txt`.

A handful of resource paths with no natural config key (FSL atlas, raw image
zips) are marked inline as `<EXTERNAL: ...>`; search for that string and point
each at your copy.
