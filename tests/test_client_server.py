# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from brain_comm.batching import BatchingArbiter
from brain_comm.brain import Brain
from brain_comm.client import BrainClient, BrainTimeout
from brain_comm.determinism import apply_determinism_flags, mix_seed
from brain_comm.serve import run_replica
from brain_comm.smoke import smoke

ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "examples" / "zero_brain"


def test_apply_determinism_flags_leaves_fast_kernels(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    apply_determinism_flags()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_zero_brain_smoke(tmp_path: Path):
    _start(tmp_path, ZERO, entry=None)
    spec = smoke(tmp_path / "brain-0.sock", timeout_s=5)
    assert spec["horizon"] == 1
    assert spec["action_dim"] == 8


def test_act_is_deterministic_for_the_same_seed(tmp_path: Path):
    submission = _noise_submission(tmp_path / "submission")
    _start(tmp_path, submission, entry="noise:NoiseBrain")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=5)
    client.connect()
    obs = {"probe": np.zeros((2, 3), dtype=np.float32)}
    client.begin_episode(["a", "b"], [7, 8], ["one", "two"])
    first = client.act(obs, step=1, seed=7)
    second = client.act(obs, step=1, seed=7)
    other = client.act(obs, step=2, seed=7)
    client.close()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert first.shape == (2, 1, 4)


def test_act_timeout(tmp_path: Path):
    submission = _slow_submission(tmp_path / "submission")
    _start(tmp_path, submission, entry="slow:SlowBrain")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=0.2)
    client.connect()
    with pytest.raises(BrainTimeout):
        client.act({"probe": np.zeros((1, 1), np.float32)}, step=0, seed=0)


def _start(socket_dir: Path, submission: Path, entry: str | None) -> None:
    args = argparse.Namespace(
        entry=entry,
        replicas=1,
        socket_dir=str(socket_dir),
        submission=str(submission),
        max_gb=None,
    )
    thread = threading.Thread(target=run_replica, args=(0, args), daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        if (socket_dir / "brain-0.sock").exists():
            return
        time.sleep(0.02)
    raise RuntimeError("brain socket was not created")


def _noise_submission(path: Path) -> Path:
    _write_submission(
        path,
        "noise:NoiseBrain",
        '''
import numpy as np
from brain_comm.brain import Brain

class NoiseBrain(Brain):
    def setup(self, device, weights_dir, config):
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step):
        count = next(iter(obs.values())).shape[0]
        return np.random.standard_normal((count, self.horizon, self.action_dim)).astype(np.float32)
''',
        horizon=1,
        action_dim=4,
    )
    return path


def _slow_submission(path: Path) -> Path:
    _write_submission(
        path,
        "slow:SlowBrain",
        '''
import time
import numpy as np
from brain_comm.brain import Brain

class SlowBrain(Brain):
    def setup(self, device, weights_dir, config):
        self.horizon = 1
        self.action_dim = 1

    def begin_episode(self, episode_ids, seeds, instructions):
        return None

    def act(self, obs, step):
        time.sleep(2)
        return np.zeros((1, 1, 1), dtype=np.float32)
''',
        horizon=1,
        action_dim=1,
    )
    return path


def _write_submission(path: Path, entry: str, source: str, horizon: int, action_dim: int) -> None:
    module, _class_name = entry.split(":")
    (path / "brain").mkdir(parents=True)
    (path / "weights").mkdir()
    (path / "train").mkdir()
    (path / "brain" / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (path / "agent.yaml").write_text(
        f"entry: {entry}\nhorizon: {horizon}\naction_dim: {action_dim}\nweights: {{}}\n",
        encoding="utf-8",
    )

def test_connect_waits_for_a_brain_that_is_still_loading(tmp_path: Path):
    submission = _noise_submission(tmp_path / "submission")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=5, connect_wait_s=5)
    threading.Timer(0.6, _start, args=(tmp_path, submission, "noise:NoiseBrain")).start()
    spec = client.connect()
    client.close()
    assert spec["action_dim"] == 4
    with pytest.raises(FileNotFoundError):
        BrainClient(tmp_path / "missing.sock", timeout_s=1).connect()


def test_hello_reports_the_observation_block(tmp_path: Path):
    submission = _noise_submission(tmp_path / "submission")
    text = (submission / "agent.yaml").read_text(encoding="utf-8")
    text += "observation:\n  cameras: cam_exterior_left, cam_wrist\n  image_hw: 180x320\n"
    (submission / "agent.yaml").write_text(text, encoding="utf-8")
    _start(tmp_path, submission, entry="noise:NoiseBrain")
    client = BrainClient(tmp_path / "brain-0.sock", timeout_s=5)
    spec = client.connect()
    client.close()
    assert spec["observation"] == {"cameras": ["cam_exterior_left", "cam_wrist"], "image_hw": [180, 320]}
    assert spec["concurrent"] is False


def test_shared_replica_keeps_each_client_instruction(tmp_path: Path):
    submission = tmp_path / "submission"
    _write_submission(
        submission,
        "echo:EchoBrain",
        """
        import numpy as np
        from brain_comm.brain import Brain

        class EchoBrain(Brain):
            session_safe = True

            def setup(self, device, weights_dir, config):
                self.horizon = 1
                self.action_dim = 1

            def begin_episode(self, episode_ids, seeds, instructions):
                self.text = list(instructions)

            def act(self, obs, step):
                return np.array([[[float(self.text[0])]]], dtype=np.float32)
        """,
        horizon=1,
        action_dim=1,
    )
    _start(tmp_path, submission, entry="echo:EchoBrain")
    first = BrainClient(tmp_path / "brain-0.sock", timeout_s=5)
    second = BrainClient(tmp_path / "brain-0.sock", timeout_s=5)
    assert first.connect()["concurrent"] is True
    second.connect()
    first.begin_episode(["a"], [1], ["1"])
    second.begin_episode(["b"], [2], ["2"])
    obs = {"probe": np.zeros((1, 1), dtype=np.float32)}
    seen = [
        (first.act(obs, 0, 1)[0, 0, 0], second.act(obs, 0, 2)[0, 0, 0], first.act(obs, 1, 1)[0, 0, 0])
        for _ in range(3)
    ]
    first.close()
    second.close()
    assert seen == [(1.0, 2.0, 1.0)] * 3


class _Worker:
    """Stands in for a connection session."""

    episode = None


class _RowBrain(Brain):
    """Answers each row with its instruction as a number."""

    batchable = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int], list[int]]] = []
        self.fail_next = False

    def begin_episode(self, episode_ids, seeds, instructions) -> None:
        return None

    def act(self, obs, step):
        raise AssertionError("a batchable brain is called through act_rows")

    def act_rows(self, obs, steps, seeds, instructions):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("model failed")
        self.calls.append((list(instructions), steps.tolist(), seeds.tolist()))
        return np.array([[[float(text)]] for text in instructions], dtype=np.float32)


def _ask_together(arbiter, asks):
    """Call ``arbiter.act`` from one thread per ask; return the results by name."""
    results: dict[str, object] = {}

    def call(name, worker, rows):
        try:
            obs = {"x": np.zeros((rows, 2), dtype=np.float32)}
            results[name] = arbiter.act(worker, obs, step=4, seed=0)
        except Exception as exc:
            results[name] = exc

    threads = [threading.Thread(target=call, args=ask) for ask in asks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    return results


def test_batching_runs_concurrent_workers_in_one_forward_pass():
    brain = _RowBrain()
    arbiter = BatchingArbiter(brain, window_s=5.0)
    first, second = _Worker(), _Worker()
    arbiter.begin(first, ["a0"], [5], ["1"])
    arbiter.begin(second, ["b0", "b1"], [6, 7], ["2", "3"])
    results = _ask_together(arbiter, [("a", first, 1), ("b", second, 2)])
    assert len(brain.calls) == 1
    instructions, steps, seeds = brain.calls[0]
    assert sorted(instructions) == ["1", "2", "3"]
    assert steps == [4, 4, 4]
    assert sorted(seeds) == sorted(mix_seed(seed, 4) for seed in (5, 6, 7))
    assert results["a"].tolist() == [[[1.0]]]
    assert results["b"].tolist() == [[[2.0]], [[3.0]]]


def test_batching_does_not_delay_a_lone_worker():
    arbiter = BatchingArbiter(_RowBrain(), window_s=5.0)
    worker = _Worker()
    arbiter.begin(worker, ["a0"], [1], ["1"])
    started = time.monotonic()
    results = _ask_together(arbiter, [("a", worker, 1)])
    assert results["a"].tolist() == [[[1.0]]]
    assert time.monotonic() - started < 2.0


def test_batching_waits_one_window_for_a_worker_that_is_not_asking():
    brain = _RowBrain()
    arbiter = BatchingArbiter(brain, window_s=0.2)
    asking, idle = _Worker(), _Worker()
    arbiter.begin(asking, ["a0"], [1], ["1"])
    arbiter.begin(idle, ["b0"], [2], ["2"])
    started = time.monotonic()
    results = _ask_together(arbiter, [("a", asking, 1)])
    elapsed = time.monotonic() - started
    assert results["a"].tolist() == [[[1.0]]]
    assert 0.15 <= elapsed < 3.0
    assert [call[0] for call in brain.calls] == [["1"]]
    arbiter.release(idle)


def test_batching_reports_a_model_error_to_every_worker_and_recovers():
    brain = _RowBrain()
    arbiter = BatchingArbiter(brain, window_s=5.0)
    first, second = _Worker(), _Worker()
    arbiter.begin(first, ["a0"], [1], ["1"])
    arbiter.begin(second, ["b0"], [2], ["2"])
    brain.fail_next = True
    failed = _ask_together(arbiter, [("a", first, 1), ("b", second, 1)])
    assert all(isinstance(value, RuntimeError) for value in failed.values())
    again = _ask_together(arbiter, [("a", first, 1), ("b", second, 1)])
    assert again["a"].tolist() == [[[1.0]]]
    assert again["b"].tolist() == [[[2.0]]]


def test_batching_rejects_observation_rows_that_do_not_match_the_episodes():
    arbiter = BatchingArbiter(_RowBrain(), window_s=0.1)
    worker = _Worker()
    arbiter.begin(worker, ["a0"], [1], ["1"])
    results = _ask_together(arbiter, [("a", worker, 3)])
    assert isinstance(results["a"], ValueError)


def test_batchable_replica_merges_two_clients_over_sockets(tmp_path: Path):
    submission = tmp_path / "submission"
    _write_submission(
        submission,
        "merge:MergeBrain",
        """
        import numpy as np
        from brain_comm.brain import Brain

        CALLS = []

        class MergeBrain(Brain):
            batchable = True

            def setup(self, device, weights_dir, config):
                self.horizon = 1
                self.action_dim = 1

            def begin_episode(self, episode_ids, seeds, instructions):
                return None

            def act(self, obs, step):
                raise AssertionError("act_rows is the entry point")

            def act_rows(self, obs, steps, seeds, instructions):
                CALLS.append(list(instructions))
                return np.array([[[float(text)]] for text in instructions], dtype=np.float32)
        """,
        horizon=1,
        action_dim=1,
    )
    with (submission / "agent.yaml").open("a", encoding="utf-8") as handle:
        handle.write("batch_window_ms: 2000\n")
    _start(tmp_path, submission, entry="merge:MergeBrain")
    clients = [BrainClient(tmp_path / "brain-0.sock", timeout_s=10) for _ in range(2)]
    assert all(client.connect()["concurrent"] is True for client in clients)
    clients[0].begin_episode(["a0"], [1], ["1"])
    clients[1].begin_episode(["b0", "b1"], [2, 3], ["2", "3"])
    gate = threading.Barrier(2)
    answers: dict[int, np.ndarray] = {}

    def ask(index: int, rows: int) -> None:
        gate.wait()
        answers[index] = clients[index].act({"probe": np.zeros((rows, 1), np.float32)}, step=0, seed=0)

    threads = [threading.Thread(target=ask, args=(0, 1)), threading.Thread(target=ask, args=(1, 2))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    for client in clients:
        client.close()
    merge_calls = sys.modules["merge"].CALLS
    assert len(merge_calls) == 1 and sorted(merge_calls[0]) == ["1", "2", "3"]
    assert answers[0].tolist() == [[[1.0]]]
    assert answers[1].tolist() == [[[2.0]], [[3.0]]]
