# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A brain that returns zeros. Used by tests and the simulator smoke run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from brain_comm.brain import Brain


class ZeroBrain(Brain):
    """Action chunk of zeros with the tournament horizon and action size."""

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        return None

    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        count = 1
        for value in obs.values():
            count = int(np.asarray(value).shape[0])
            break
        return np.zeros((count, self.horizon, self.action_dim), dtype=np.float32)
