"""Stage-1 directory layout, resolved from `config/paths.yaml`.

Every directory the encoder reads or writes comes from the repo's path
configuration, so no machine-specific location is committed. Copy
`config/paths.yaml` to `config/paths.local.yaml`, fill in the `embedding:`
section, and everything below resolves.

The function names are kept as-is because the model and dataset modules call
them directly.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg  # noqa: E402


def _check(path, key):
    """Reject placeholder values instead of creating directories named '<path'."""
    path = str(path)
    if "<" in path or ">" in path:
        raise ValueError(
            "embedding.{0} is still the placeholder {1!r}. Copy "
            "config/paths.yaml to config/paths.local.yaml and set it."
            .format(key, path))
    return path


def _ensure(path, key):
    """Validate, create if missing, and return the directory."""
    path = _check(path, key)
    os.makedirs(path, exist_ok=True)
    return path


def root_path():
    """Project root for stage 1. Only used to derive the paths below."""
    return _check(cfg.embedding.project_root, "project_root")


def data_path():
    return _ensure(os.path.join(root_path(), "data"), "project_root")


def train_dir():
    """One sub-folder per training subject, each holding T1_brain.nii.gz."""
    return _check(cfg.embedding.train_dir, "train_dir")


def val_dir():
    """As train_dir, for the held-out split used to monitor the objective."""
    return _check(cfg.embedding.val_dir, "val_dir")


def model_weight_path():
    """Encoder checkpoints are written here."""
    return _ensure(cfg.embedding.weights_dir, "weights_dir")


def log_path():
    """TensorBoard logs."""
    return _ensure(cfg.embedding.log_dir, "log_dir")


def checkpoints_path():
    return _ensure(os.path.join(root_path(), "checkpoints"), "project_root")


def result_dir_path():
    return _ensure(os.path.join(root_path(), "results"), "project_root")


def region_related_data_path():
    return _ensure(os.path.join(data_path(), "region_data"), "project_root")


def subject_dict_path():
    return _ensure(os.path.join(region_related_data_path(), "subject_dicts"),
                   "project_root")


def src_path():
    return _ensure(os.path.join(root_path(), "src"), "project_root")


def fast_gwa_path():
    return _ensure(os.path.join(region_related_data_path(), "fastgwa"),
                   "project_root")
