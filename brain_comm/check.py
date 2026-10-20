# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic submission checks, plus the image-specific hook."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

from brain_comm.yaml_min import load_yaml

REQUIRED_DIRS = ("brain", "weights", "train")
_SHA256_HEX = 64


class CheckError(Exception):
    """One or more submission checks failed."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def check_submission(submission: Path, max_gb: float | None = None) -> dict:
    """Validate a submission directory.

    Args:
        submission: Directory mounted at ``/submission`` in a brain image.
        max_gb: Total size limit. ``UNIDROID_MAX_SUBMISSION_GB`` is used when omitted.

    Returns:
        The parsed ``agent.yaml``.

    Raises:
        CheckError: When any generic check or the image hook fails.
    """
    submission = Path(submission)
    errors: list[str] = []
    config = _load_config(submission, errors)
    if config is not None:
        _check_schema(config, errors)
        _check_weights(submission, config, errors)
    _check_layout(submission, errors)
    _check_size(submission, max_gb, errors)
    if not errors and config is not None:
        hook_error = _run_hook(submission, config)
        if hook_error:
            errors.append(hook_error)
    if errors:
        raise CheckError(errors)
    return config


def _load_config(submission: Path, errors: list[str]) -> dict | None:
    path = submission / "agent.yaml"
    if not path.is_file():
        errors.append("missing agent.yaml")
        return None
    try:
        config = load_yaml(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        errors.append(f"agent.yaml: {exc}")
        return None
    return config


def _check_layout(submission: Path, errors: list[str]) -> None:
    if not submission.is_dir():
        errors.append(f"submission is not a directory: {submission}")
        return
    for name in REQUIRED_DIRS:
        path = submission / name
        if not path.is_dir():
            errors.append(f"missing directory: {name}/")


def _check_schema(config: dict, errors: list[str]) -> None:
    entry = config.get("entry")
    if not isinstance(entry, str) or entry.count(":") != 1 or entry.startswith(":") or entry.endswith(":"):
        errors.append("entry must be 'module:Class'")
    for key in ("horizon", "action_dim"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"{key} must be an integer >= 1")
    weights = config.get("weights")
    if not isinstance(weights, dict):
        errors.append("weights must be a mapping of relative path to sha256")
    else:
        for rel, digest in weights.items():
            if not isinstance(rel, str) or not isinstance(digest, str):
                errors.append("weights entries must be strings")
                continue
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                errors.append(f"weights path must stay inside weights/: {rel}")
            if len(digest) != _SHA256_HEX or any(ch not in "0123456789abcdef" for ch in digest.lower()):
                errors.append(f"weights digest for {rel} is not 64 hex characters")
    memory = config.get("max_gpu_mem_gb", None)
    if memory is not None and (isinstance(memory, bool) or not isinstance(memory, (int, float)) or memory <= 0):
        errors.append("max_gpu_mem_gb must be a positive number when set")


def _check_weights(submission: Path, config: dict, errors: list[str]) -> None:
    declared = config.get("weights")
    if not isinstance(declared, dict):
        return
    weights_dir = submission / "weights"
    if not weights_dir.is_dir():
        return
    present = {
        path.relative_to(weights_dir).as_posix()
        for path in weights_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    declared_keys = set(declared)
    for rel in sorted(present - declared_keys):
        errors.append(f"weights file has no hash in agent.yaml: {rel}")
    for rel in sorted(declared_keys - present):
        errors.append(f"weights file listed in agent.yaml is missing: {rel}")
    for rel in sorted(present & declared_keys):
        digest = hashlib.sha256((weights_dir / rel).read_bytes()).hexdigest()
        if digest != str(declared[rel]).lower():
            errors.append(f"sha256 mismatch for weights/{rel}")


def _check_size(submission: Path, max_gb: float | None, errors: list[str]) -> None:
    if max_gb is None:
        raw = os.environ.get("UNIDROID_MAX_SUBMISSION_GB", "")
        max_gb = float(raw) if raw else None
    if max_gb is None or not submission.is_dir():
        return
    total = 0
    for path in submission.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    limit = int(max_gb * (1024 ** 3))
    if total > limit:
        errors.append(f"submission is {total} bytes; limit is {limit} bytes")


def _run_hook(submission: Path, config: dict) -> str | None:
    spec = os.environ.get("BRAIN_COMM_CHECK", "").strip()
    if not spec:
        return None
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        return "BRAIN_COMM_CHECK must be 'module:function'"
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        result = func(submission, config)
    except Exception as exc:
        return f"check hook failed: {exc}"
    if result is None or result == "":
        return None
    return str(result)
