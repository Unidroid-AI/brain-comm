# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Client for one brain replica."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from brain_comm.protocol import PROTOCOL_VERSION, recv_message, send_message
from brain_comm.sockets import connect

_CONNECT_POLL_S = 0.5


class BrainError(RuntimeError):
    """The brain returned an error or a malformed reply."""


class BrainTimeout(BrainError):
    """A call exceeded its deadline. The caller records the step as failed."""


class BrainClient:
    """One connection to ``brain-<k>.sock``."""

    def __init__(self, socket_path: str | Path, timeout_s: float = 30.0, connect_wait_s: float = 0.0):
        """Create a client.

        Args:
            socket_path: Address file written by ``brain-comm serve``.
            timeout_s: Deadline [s] for one call.
            connect_wait_s: How long [s] ``connect`` keeps retrying while the brain is still
                loading and its socket does not exist yet or refuses connections.
        """
        self.socket_path = Path(socket_path)
        self.timeout_s = timeout_s
        self.connect_wait_s = connect_wait_s
        self._sock: socket.socket | None = None
        self.spec: dict[str, Any] = {}

    def connect(self) -> dict[str, Any]:
        """Connect and exchange ``hello``. Returns the server spec."""
        deadline = time.monotonic() + self.connect_wait_s
        while True:
            try:
                self._sock = connect(self.socket_path, self.timeout_s)
                break
            except socket.timeout as exc:
                raise BrainTimeout("timed out connecting to the brain") from exc
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_CONNECT_POLL_S)
        return self._call({"type": "hello", "protocol_version": PROTOCOL_VERSION})

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        self._call(
            {
                "type": "begin_episode",
                "episode_ids": list(episode_ids),
                "seeds": [int(seed) for seed in seeds],
                "instructions": list(instructions),
            }
        )

    def act(self, obs: dict[str, np.ndarray], step: int, seed: int) -> np.ndarray:
        """Request one action chunk.

        Args:
            obs: Observation arrays batched on axis 0.
            step: Control step.
            seed: Episode seed. The server reseeds from ``(seed, step)`` first.

        Returns:
            Array of shape ``[num_envs, horizon, action_dim]``.
        """
        reply = self._call({"type": "act", "obs": obs, "step": int(step), "seed": int(seed)})
        actions = reply.get("actions")
        if not isinstance(actions, np.ndarray):
            raise BrainError("act reply is missing an actions array")
        return actions

    def end_episode(self, episode_ids: list[str]) -> None:
        self._call({"type": "end_episode", "episode_ids": list(episode_ids)})

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._call({"type": "close"})
        except (BrainError, OSError, ConnectionError):
            pass
        finally:
            self._sock.close()
            self._sock = None

    def _call(self, message: dict) -> dict:
        if self._sock is None:
            raise BrainError("client is not connected")
        try:
            send_message(self._sock, message)
            reply = recv_message(self._sock)
        except socket.timeout as exc:
            raise BrainTimeout(f"timed out waiting for {message['type']}") from exc
        except ConnectionError as exc:
            raise BrainError(str(exc)) from exc
        if reply.get("type") == "error":
            raise BrainError(str(reply.get("message", "brain error")))
        if message["type"] == "hello":
            if reply.get("protocol_version") != PROTOCOL_VERSION:
                raise BrainError(
                    f"protocol {reply.get('protocol_version')!r} != {PROTOCOL_VERSION!r}"
                )
            self.spec = reply
        return reply
