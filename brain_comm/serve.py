# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve one brain replica per GPU.

``python -m brain_comm.serve --replicas N --socket-dir /run/brain``
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from brain_comm.batching import BatchingArbiter
from brain_comm.brain import Brain
from brain_comm.check import check_submission
from brain_comm.determinism import apply_determinism_flags, apply_memory_fraction, reseed
from brain_comm.protocol import PROTOCOL_VERSION, ProtocolError, recv_message, send_message
from brain_comm.sockets import bind_listener


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    replica = os.environ.get("BRAIN_COMM_REPLICA")
    if replica is None and args.replicas > 1:
        _spawn_replicas(args)
        return
    index = int(replica) if replica is not None else 0
    if replica is None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_replica(index, args)


def run_replica(index: int, args: argparse.Namespace) -> None:
    """Load the submission and serve ``brain-<index>.sock``."""
    apply_determinism_flags()
    config = check_submission(args.submission, max_gb=args.max_gb)
    entry = args.entry or config["entry"]
    brain_dir = Path(args.submission) / "brain"
    if str(brain_dir) not in sys.path:
        sys.path.insert(0, str(brain_dir))
    brain = _load_brain(entry)()
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "-1" else "cpu"
    memory = config.get("max_gpu_mem_gb")
    apply_memory_fraction(float(memory) if memory is not None else None)
    brain.setup(device, Path(args.submission) / "weights", config)
    # Workers share a replica when the brain says so; a batchable brain always does.
    shared = bool(brain.session_safe or brain.batchable)
    spec = {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "horizon": int(config["horizon"]),
        "action_dim": int(config["action_dim"]),
        "action_dtype": "float32",
        # Cameras and image size the brain reads. Empty means every camera at full size.
        "observation": _observation_spec(config.get("observation")),
        "concurrent": shared,
    }
    listener = bind_listener(Path(args.socket_dir) / f"brain-{index}.sock")
    if brain.batchable:
        window_s = float(config.get("batch_window_ms", 20)) / 1000.0
        arbiter = BatchingArbiter(brain, window_s)
    else:
        arbiter = _Arbiter(brain)
    try:
        while True:
            conn, _addr = listener.accept()
            conn.settimeout(None)
            if shared:
                threading.Thread(target=_serve_one, args=(conn, arbiter, spec), daemon=True).start()
            else:
                _serve_one(conn, arbiter, spec)
    finally:
        listener.close()


def _observation_spec(block) -> dict:
    """Normalize the optional ``observation`` block of ``agent.yaml``.

    ``cameras`` is a comma-separated list of observation names. ``image_hw`` is
    ``HxW`` [pixels]. Anything left out is absent from the result.
    """
    if not isinstance(block, dict):
        return {}
    spec: dict = {}
    cameras = block.get("cameras")
    if isinstance(cameras, str) and cameras.strip():
        spec["cameras"] = [name.strip() for name in cameras.split(",") if name.strip()]
    size = block.get("image_hw")
    if isinstance(size, str) and size.strip():
        height, _, width = size.lower().partition("x")
        spec["image_hw"] = [int(height), int(width)]
    return spec


def _serve_one(conn, arbiter: _Arbiter | BatchingArbiter, spec: dict) -> None:
    session = _Session()
    try:
        _serve_connection(conn, arbiter, spec, session)
    except (ConnectionError, ProtocolError, OSError):
        pass
    finally:
        arbiter.release(session)
        conn.close()


def _serve_connection(conn, arbiter: _Arbiter | BatchingArbiter, spec: dict, session: "_Session") -> None:
    while True:
        message = recv_message(conn)
        try:
            if not _dispatch(conn, arbiter, spec, message, session):
                return
        except Exception as exc:
            send_message(conn, {"type": "error", "message": str(exc)})


class _Session:
    """What one client last told the brain, so another client's call can be undone."""

    def __init__(self) -> None:
        self.episode: tuple[list, list, list] | None = None


class _Arbiter:
    """Serialize inference and keep the brain's episode state matched to the caller.

    The brain keeps one episode at a time. When a different client calls, its last
    ``begin_episode`` is repeated first. A sequential server sees one client at a
    time, so this never repeats anything there.
    """

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.lock = threading.Lock()
        self._active: _Session | None = None

    def begin(self, session: _Session, ids, seeds, instructions) -> None:
        with self.lock:
            self.brain.begin_episode(ids, seeds, instructions)
            session.episode = (list(ids), list(seeds), list(instructions))
            self._active = session

    def act(self, session: _Session, obs, step: int, seed: int) -> np.ndarray:
        with self.lock:
            self._attach(session)
            reseed(seed, step)
            return np.asarray(self.brain.act(obs, step), dtype=np.float32)

    def end(self, session: _Session, ids) -> None:
        with self.lock:
            self._attach(session)
            self.brain.end_episode(ids)
            session.episode = None
            self._active = None

    def release(self, session: _Session) -> None:
        with self.lock:
            if self._active is session:
                self._active = None

    def _attach(self, session: _Session) -> None:
        if self._active is session:
            return
        if session.episode is not None:
            self.brain.begin_episode(*session.episode)
        self._active = session


def _dispatch(conn, arbiter: _Arbiter | BatchingArbiter, spec: dict, message: dict, session: "_Session") -> bool:
    """Handle one message. Return False when the connection should close."""
    kind = message.get("type")
    if kind == "hello":
        if message.get("protocol_version") != PROTOCOL_VERSION:
            send_message(conn, {"type": "error", "message": "protocol version mismatch"})
            return False
        send_message(conn, spec)
    elif kind == "begin_episode":
        arbiter.begin(session, message["episode_ids"], message["seeds"], message["instructions"])
        send_message(conn, {"type": "begin_episode", "ok": True})
    elif kind == "act":
        actions = arbiter.act(session, message["obs"], int(message["step"]), int(message["seed"]))
        _check_actions(actions, spec)
        send_message(conn, {"type": "act", "actions": actions})
    elif kind == "end_episode":
        arbiter.end(session, list(message.get("episode_ids", [])))
        send_message(conn, {"type": "end_episode", "ok": True})
    elif kind == "close":
        send_message(conn, {"type": "close", "ok": True})
        return False
    else:
        send_message(conn, {"type": "error", "message": f"unknown message {kind!r}"})
        return False
    return True


def _check_actions(actions: np.ndarray, spec: dict) -> None:
    if actions.ndim != 3 or actions.shape[1] != spec["horizon"] or actions.shape[2] != spec["action_dim"]:
        raise ProtocolError(
            f"actions shape {tuple(actions.shape)} != "
            f"[num_envs, {spec['horizon']}, {spec['action_dim']}]"
        )


def _load_brain(entry: str) -> type[Brain]:
    module_name, class_name = entry.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, Brain):
        raise TypeError(f"{entry} is not a Brain subclass")
    return cls


def _spawn_replicas(args: argparse.Namespace) -> None:
    procs = []
    command = [
        sys.executable,
        "-m",
        "brain_comm.serve",
        "--replicas",
        str(args.replicas),
        "--socket-dir",
        args.socket_dir,
        "--submission",
        args.submission,
    ]
    if args.entry:
        command += ["--entry", args.entry]
    if args.max_gb is not None:
        command += ["--max-gb", str(args.max_gb)]
    for index in range(args.replicas):
        env = os.environ.copy()
        env["BRAIN_COMM_REPLICA"] = str(index)
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        procs.append(subprocess.Popen(command, env=env))
    code = 0
    try:
        for proc in procs:
            status = proc.wait()
            if status != 0:
                code = status
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
    raise SystemExit(code)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="brain-comm serve")
    parser.add_argument("--entry", default=None, help="Override agent.yaml entry, as module:Class")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--socket-dir", default=os.environ.get("UNIDROID_SOCKET_DIR", "/run/brain"))
    parser.add_argument("--submission", default=os.environ.get("UNIDROID_SUBMISSION", "/submission"))
    parser.add_argument("--max-gb", type=float, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
