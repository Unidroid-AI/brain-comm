# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
from pathlib import Path

import pytest

from brain_comm.check import CheckError, check_submission

ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "examples" / "zero_brain"


def test_zero_submission_passes():
    config = check_submission(ZERO)
    assert config["entry"] == "zero:ZeroBrain"


def test_bad_hash(tmp_path: Path):
    _copy_zero(tmp_path)
    payload = b"weights"
    (tmp_path / "weights" / "model.bin").write_bytes(payload)
    digest = "0" * 64
    _set_weights(tmp_path, {"model.bin": digest})
    with pytest.raises(CheckError) as raised:
        check_submission(tmp_path)
    assert any("sha256 mismatch" in item for item in raised.value.errors)


def test_matching_hash(tmp_path: Path):
    _copy_zero(tmp_path)
    payload = b"weights"
    (tmp_path / "weights" / "model.bin").write_bytes(payload)
    _set_weights(tmp_path, {"model.bin": hashlib.sha256(payload).hexdigest()})
    check_submission(tmp_path)


def test_oversize(tmp_path: Path):
    _copy_zero(tmp_path)
    (tmp_path / "train" / "blob.bin").write_bytes(b"x" * 2048)
    with pytest.raises(CheckError) as raised:
        check_submission(tmp_path, max_gb=2048 / (1024 ** 3) / 2)
    assert any("limit" in item for item in raised.value.errors)


def test_hook_failure(tmp_path: Path, monkeypatch):
    _copy_zero(tmp_path)
    monkeypatch.setenv("BRAIN_COMM_CHECK", "tests.hooks:reject")
    with pytest.raises(CheckError) as raised:
        check_submission(tmp_path)
    assert any("rejected by hook" in item for item in raised.value.errors)
    monkeypatch.delenv("BRAIN_COMM_CHECK", raising=False)


def _copy_zero(dest: Path) -> None:
    for rel in ("agent.yaml", "brain/zero.py", "train/README.md", "weights/.gitkeep"):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ZERO / rel).read_bytes())


def _set_weights(dest: Path, weights: dict[str, str]) -> None:
    text = (dest / "agent.yaml").read_text(encoding="utf-8")
    text = text.replace("weights: {}", "weights:\n" + "".join(f"  {key}: {value}\n" for key, value in weights.items()))
    (dest / "agent.yaml").write_text(text, encoding="utf-8")


# Imported by BRAIN_COMM_CHECK in the hook test. Kept in this module's sibling.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
