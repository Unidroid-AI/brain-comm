# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeding for one brain replica. Torch is used only when it is installed."""

from __future__ import annotations

import os
import random

import numpy as np


def apply_determinism_flags() -> None:
    """Leave fast kernels enabled. Call this before the model is loaded.

    ``reseed`` still seeds each ``act`` from the episode seed and the step.
    Deterministic cuBLAS and cuDNN kernels are left off so attention and
    ``torch.compile`` can use the fast path. A brain image may still export
    ``CUBLAS_WORKSPACE_CONFIG``; drop it before torch initializes cuBLAS.
    """
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def mix_seed(seed: int, step: int) -> int:
    """Combine an episode seed and a control step into one 32-bit seed."""
    return (int(seed) * 1_000_003 + int(step)) & 0xFFFFFFFF


def reseed(seed: int, step: int) -> None:
    """Reseed Python, numpy, and torch from the episode seed and the step."""
    mixed = mix_seed(seed, step)
    random.seed(mixed)
    np.random.seed(mixed)
    torch = _torch()
    if torch is None:
        return
    torch.manual_seed(mixed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mixed)


def apply_memory_fraction(max_gpu_mem_gb: float | None) -> None:
    """Cap this process to the declared GPU budget, when CUDA is available."""
    if max_gpu_mem_gb is None:
        return
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (float(max_gpu_mem_gb) * (1024 ** 3)) / total)
    torch.cuda.set_per_process_memory_fraction(fraction)


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch
