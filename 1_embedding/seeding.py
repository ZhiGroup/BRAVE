"""Global determinism controls for the seeded BRE encoder runs.

The original training stack contains no seed of any kind: `random.shuffle` over
an unsorted `os.listdir`, unseeded DataLoader workers, unseeded negative
sampling inside the InfoNCE loss, and cuDNN autotuning. This module pins every
one of those so that a run is a function of (seed, code, data) alone.

`CUBLAS_WORKSPACE_CONFIG` must be set before CUDA initialises, so the training
entry point sets it at import time rather than calling into here first.
"""
import os
import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=True, strict=False, verbose=True):
    """Seed every RNG the training stack draws from and pin cuDNN behaviour.

    Args:
        seed: integer seed.
        deterministic: pin cuDNN to deterministic kernels and disable the
            autotuner. This plus full RNG seeding is the practical recipe and
            is what the runs rely on.
        strict: additionally call `torch.use_deterministic_algorithms`. On
            torch 1.10.2 (this environment) there is no `warn_only` argument,
            so any op lacking a deterministic kernel — trilinear interpolation
            and scatter-style indexing, both used here — raises at runtime and
            aborts training. Left off by default; the smoke test verifies
            determinism empirically instead of asserting it via the API.
        verbose: print what was pinned.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False

    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if verbose:
            print("[seeding] strict deterministic algorithms ENABLED "
                  "(will raise on ops lacking a deterministic kernel)")

    if verbose:
        print("[seeding] seed={0} cudnn.deterministic={1} cudnn.benchmark={2} "
              "strict={3} CUBLAS_WORKSPACE_CONFIG={4}".format(
                  seed, torch.backends.cudnn.deterministic,
                  torch.backends.cudnn.benchmark, strict,
                  os.environ.get("CUBLAS_WORKSPACE_CONFIG")))


def environment_report():
    """Versions worth recording in the run manifest."""
    import platform
    rep = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import lightning
        rep["lightning"] = lightning.__version__
    except Exception:
        rep["lightning"] = None
    try:
        rep["gpu_names"] = [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
    except Exception:
        rep["gpu_names"] = []
    return rep
