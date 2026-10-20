# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Merge the ``act`` calls of several workers into one forward pass."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

import numpy as np

from brain_comm.brain import Brain
from brain_comm.determinism import mix_seed
from brain_comm.protocol import ProtocolError


class _Request:
    """One worker's act call, and where its answer goes."""

    def __init__(self, obs, step: int, seeds: list[int], instructions: list[str]) -> None:
        self.obs = {key: np.asarray(value) for key, value in obs.items()}
        self.step = step
        self.seeds = seeds
        self.instructions = instructions
        self.rows = len(instructions)
        self.done = False
        self.actions: np.ndarray | None = None
        self.error: BaseException | None = None

    def signature(self) -> tuple:
        """Requests with equal signatures can be stacked on axis 0."""
        return tuple((key, value.shape[1:], value.dtype.str) for key, value in sorted(self.obs.items()))

    def take(self) -> np.ndarray:
        if self.error is not None:
            raise self.error
        return self.actions


class BatchingArbiter:
    """Serve a batchable brain to several workers.

    The first worker to ask leads. It waits until every worker that has an episode
    running has asked, or until ``window_s`` has passed, then runs one ``act_rows``
    for all the rows that arrived. A worker that asks while a batch is running joins
    the next one. A lone worker is never delayed.

    Sessions are duck-typed: they need an ``episode`` attribute holding
    ``(ids, seeds, instructions)`` or ``None``.
    """

    def __init__(self, brain: Brain, window_s: float = 0.02) -> None:
        self.brain = brain
        self.window_s = window_s
        self._cond = threading.Condition()
        self._run_lock = threading.Lock()
        self._pending: list[_Request] = []
        self._active: set = set()
        self._leading = False

    def begin(self, session, ids, seeds, instructions) -> None:
        with self._run_lock:
            self.brain.begin_episode(ids, seeds, instructions)
        with self._cond:
            session.episode = (list(ids), list(seeds), list(instructions))
            self._active.add(session)
            self._cond.notify_all()

    def act(self, session, obs, step: int, seed: int) -> np.ndarray:
        """Return this worker's rows. ``seed`` is unused: rows carry their episode seeds."""
        del seed
        if session.episode is None:
            raise RuntimeError("act called before begin_episode")
        _ids, seeds, instructions = session.episode
        request = _Request(obs, int(step), seeds, instructions)
        first = next(iter(request.obs.values()), None)
        if first is None or first.shape[0] != request.rows or len(seeds) != request.rows:
            raise ValueError(f"observation rows do not match the {request.rows} episodes of this worker")
        batch = self._wait_for_turn(request)
        if batch is not None:
            self._lead(batch)
        return request.take()

    def end(self, session, ids) -> None:
        try:
            with self._run_lock:
                self.brain.end_episode(ids)
        finally:
            self.release(session)
            session.episode = None

    def release(self, session) -> None:
        """Stop waiting for this worker, for example when its connection closes."""
        with self._cond:
            self._active.discard(session)
            self._cond.notify_all()

    def _wait_for_turn(self, request: _Request) -> list[_Request] | None:
        """Queue the request. Return the batch to run, or None when another leader served it."""
        deadline = time.monotonic() + self.window_s
        with self._cond:
            self._pending.append(request)
            self._cond.notify_all()
            while not request.done:
                if self._leading:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or len(self._pending) >= len(self._active):
                    batch, self._pending = self._pending, []
                    self._leading = True
                    return batch
                self._cond.wait(remaining)
        return None

    def _lead(self, batch: list[_Request]) -> None:
        try:
            with self._run_lock:
                self._run(batch)
        finally:
            with self._cond:
                self._leading = False
                for request in batch:
                    request.done = True
                self._cond.notify_all()

    def _run(self, batch: list[_Request]) -> None:
        groups: dict[tuple, list[_Request]] = defaultdict(list)
        for request in batch:
            groups[request.signature()].append(request)
        for group in groups.values():
            try:
                self._run_group(group)
            except Exception as exc:  # delivered to the workers that asked
                for request in group:
                    request.error = exc

    def _run_group(self, group: list[_Request]) -> None:
        obs = {key: np.concatenate([request.obs[key] for request in group]) for key in group[0].obs}
        steps = np.concatenate([np.full(request.rows, request.step, dtype=np.int64) for request in group])
        seeds = np.array(
            [mix_seed(seed, request.step) for request in group for seed in request.seeds], dtype=np.int64
        )
        instructions = [text for request in group for text in request.instructions]
        actions = np.asarray(self.brain.act_rows(obs, steps, seeds, instructions), dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != len(instructions):
            raise ProtocolError(
                f"act_rows returned {tuple(actions.shape)} for {len(instructions)} rows; "
                "expected [rows, horizon, action_dim]"
            )
        start = 0
        for request in group:
            request.actions = actions[start : start + request.rows]
            start += request.rows
