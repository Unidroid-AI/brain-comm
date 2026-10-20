# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Participant brain interface.

A brain turns an observation dict into an action chunk. The server owns
seeding, so ``act`` itself must not draw unseeded randomness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Brain(ABC):
    """Inference entry point loaded from a submission."""

    session_safe: bool = False
    """Whether several simulator workers may share this replica.

    A shared replica interleaves calls from different workers. The server then
    repeats the caller's ``begin_episode`` whenever the active worker changes, so
    a brain may only set this when ``begin_episode`` is cheap and idempotent and
    ``act`` keeps no state besides what ``begin_episode`` stores.
    """

    batchable: bool = False
    """Whether the server may merge ``act`` calls from several workers.

    A batchable brain implements :meth:`act_rows`, and every output row must depend
    only on that row's inputs, not on which other rows share the call. The server
    then runs one forward pass for the workers that ask within ``batch_window_ms``
    (``agent.yaml``, default 20). ``begin_episode`` and ``end_episode`` are only
    notifications for such a brain, so it must not keep per-episode state in them.
    A batchable brain also serves several workers, as ``session_safe`` does.
    """

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        """Load weights. The default does nothing.

        Args:
            device: Torch device string for this replica, such as ``"cuda:0"``.
            weights_dir: Read-only directory of submitted weights.
            config: Parsed ``agent.yaml``.
        """

    @abstractmethod
    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        """Start a batch of episodes.

        Args:
            episode_ids: Stable id per environment.
            seeds: Per-environment seed.
            instructions: Natural-language instruction per environment.
        """

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        """Return an action chunk.

        Args:
            obs: Arrays batched on axis 0. Each value has shape ``[num_envs, ...]``.
            step: Control step at which this chunk starts.

        Returns:
            Actions with shape ``[num_envs, horizon, action_dim]``.
        """

    def act_rows(
        self,
        obs: dict[str, np.ndarray],
        steps: np.ndarray,
        seeds: np.ndarray,
        instructions: list[str],
    ) -> np.ndarray:
        """Return an action chunk for rows that may come from different workers.

        Only called when :attr:`batchable` is true.

        Args:
            obs: Arrays batched on axis 0. Each value has shape ``[rows, ...]``.
            steps: Control step at which each row's chunk starts, shape ``[rows]``.
            seeds: Seed per row, mixed from the episode seed and the step and in
                ``[0, 2**32)``, shape ``[rows]``. Draw any noise for a row from it.
            instructions: Natural-language instruction per row.

        Returns:
            Actions with shape ``[rows, horizon, action_dim]``.
        """
        raise NotImplementedError("a batchable brain must implement act_rows")

    def end_episode(self, episode_ids: list[str]) -> None:
        """Release per-episode state. The default does nothing."""
