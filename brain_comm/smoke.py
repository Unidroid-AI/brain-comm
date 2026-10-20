# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send one observation twice and require identical actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brain_comm.client import BrainClient, BrainError


class SmokeError(BrainError):
    """The live brain failed the smoke check."""


def smoke(socket_path: str | Path, timeout_s: float = 30.0) -> dict:
    """Handshake, then check determinism and the announced action spec.

    Returns:
        The hello spec.

    Raises:
        SmokeError: When the two calls differ or the action array does not match.
    """
    client = BrainClient(socket_path, timeout_s=timeout_s)
    try:
        spec = client.connect()
        obs = {"probe": np.zeros((1, 4), dtype=np.float32)}
        first = client.act(obs, step=0, seed=0)
        second = client.act(obs, step=0, seed=0)
    finally:
        client.close()
    if first.shape != second.shape or first.dtype != second.dtype or not np.array_equal(first, second):
        raise SmokeError("the same observation and seed produced different actions")
    expected_dtype = np.dtype(spec.get("action_dtype", "float32"))
    if first.dtype != expected_dtype:
        raise SmokeError(f"action dtype {first.dtype} != {expected_dtype}")
    horizon = int(spec["horizon"])
    action_dim = int(spec["action_dim"])
    if first.shape != (1, horizon, action_dim):
        raise SmokeError(f"action shape {tuple(first.shape)} != (1, {horizon}, {action_dim})")
    return spec
