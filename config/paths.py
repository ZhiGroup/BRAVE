"""Path configuration loader.

Reads paths.yaml (or paths.local.yaml if present) and exposes the values as a
dotted-access object so scripts can write `cfg.postgwas.fuma_dir` instead of
hard-coding absolute paths.

Usage from any script in the repo:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "config"))
    from paths import cfg

    fuma = cfg.postgwas.fuma_dir

The loader prefers `paths.local.yaml` (your private, git-ignored copy) and falls
back to the committed `paths.yaml` template, so you never edit the template in
place and accidentally commit a real path.
"""

from pathlib import Path
import os
import yaml

_CONFIG_DIR = Path(__file__).resolve().parent


class _Section:
    """Thin attribute wrapper around a dict, with helpful errors."""

    def __init__(self, name, data):
        self._name = name
        for key, val in data.items():
            setattr(self, key, _Section(key, val) if isinstance(val, dict) else val)

    def __getattr__(self, key):
        raise AttributeError(
            f"'{self._name}' has no path '{key}'. Check config/paths.yaml."
        )

    def __repr__(self):
        return f"<paths:{self._name}>"


def _find_config():
    # Allow an explicit override for CI / alternate machines.
    env = os.environ.get("JAGWAS_PATHS")
    if env:
        return Path(env)
    local = _CONFIG_DIR / "paths.local.yaml"
    return local if local.exists() else _CONFIG_DIR / "paths.yaml"


def load(path=None):
    path = Path(path) if path else _find_config()
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return _Section("paths", data)


cfg = load()
