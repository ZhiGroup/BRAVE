"""Path/config shim for the embedding stage.

Historically this module held all of the encoder's paths as module-level
constants. It still exposes the same names, but the values now come from the
central config (config/paths.yaml) so nothing is hard-coded here. Other modules
in this stage import it as `config_pixpro` and read these attributes.

Intermediate stores (per-sample embeddings, subject dicts) live under the BRE
output directory; the three image splits all read from the registered-T1
directory and are separated by the cohort CSVs rather than by folder.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg


def _ensure(path):
    """makedirs, but never on an unset placeholder value."""
    if isinstance(path, str) and not path.startswith("<"):
        os.makedirs(path, exist_ok=True)
    return path


# ── Model artifacts ───────────────────────────────────────────────────────────
model_weight_dir = _ensure(cfg.embedding.weights_dir)
model_log_dir    = _ensure(cfg.embedding.log_dir)

# ── GWAS hand-off (where embeddings/sumstats are written) ─────────────────────
gwas_dir      = _ensure(cfg.gwas.fastgwa_out)
embedding_dir = _ensure(cfg.embedding.bre_out_dir)
fuma_dir      = _ensure(cfg.postgwas.fuma_dir)
minp_dir      = _ensure(cfg.gwas.fastgwa_out)
fast_gwa_path = _ensure(cfg.gwas.fastgwa_out)
result_dir    = _ensure(os.path.join(gwas_dir, "results"))

# ── Intermediate embedding stores (derived from the BRE output dir) ───────────
persample_embed_dir = _ensure(os.path.join(embedding_dir, "per_sample"))
region_subject_dir  = _ensure(os.path.join(embedding_dir, "subject_dicts"))
pixpro_config_folder = _ensure(os.path.join(model_log_dir, "configs"))

# ── Cohort CSVs ───────────────────────────────────────────────────────────────
disc_path      = os.path.join(cfg.cohorts.dir, cfg.cohorts.discovery)
rep_path       = os.path.join(cfg.cohorts.dir, cfg.cohorts.replication)
train_datafile = os.path.join(cfg.cohorts.dir, cfg.cohorts.encoder_train)
val_datafile   = os.path.join(cfg.cohorts.dir, cfg.cohorts.encoder_val)
test_datafile  = os.path.join(cfg.cohorts.dir, cfg.cohorts.inference)

# ── Registered T1 images (splits defined by the cohort CSVs above) ────────────
train_img_dir = cfg.embedding.t1_dir
val_img_dir   = cfg.embedding.t1_dir
test_img_dir  = cfg.embedding.t1_dir


def _list_images(img_dir):
    """List image files in a split dir, or [] if it isn't set/present yet."""
    if isinstance(img_dir, str) and not img_dir.startswith("<") and os.path.isdir(img_dir):
        return [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    return []


train_img_paths = _list_images(train_img_dir)
val_img_paths   = _list_images(val_img_dir)
