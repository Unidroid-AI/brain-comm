# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A small YAML reader for the flat and one-level maps used by agent.yaml.

The brain image contract depends only on msgpack and numpy, so agent files are
parsed here instead of pulling in a YAML library.
"""

from __future__ import annotations

import re
from typing import Any


def load_yaml(text: str) -> dict[str, Any]:
    """Parse a mapping. Values are scalars or a nested mapping."""
    lines = _meaningful_lines(text)
    result, index = _parse_map(lines, 0, indent=0)
    if index != len(lines):
        raise ValueError(f"unexpected content at line {lines[index][0]}")
    if not isinstance(result, dict):
        raise ValueError("agent.yaml must be a mapping")
    return result


def _meaningful_lines(text: str) -> list[tuple[int, int, str]]:
    parsed = []
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        if "\t" in raw[:indent]:
            raise ValueError(f"tabs are not allowed in indentation (line {number})")
        parsed.append((number, indent, stripped.strip()))
    return parsed


def _parse_map(lines, index: int, indent: int):
    mapping: dict[str, Any] = {}
    while index < len(lines):
        number, level, content = lines[index]
        if level < indent:
            break
        if level > indent:
            raise ValueError(f"unexpected indent on line {number}")
        if content == "{}":
            raise ValueError(f"a bare mapping is not a document (line {number})")
        key, _, remainder = content.partition(":")
        key = key.strip()
        if not key:
            raise ValueError(f"missing key on line {number}")
        remainder = remainder.strip()
        index += 1
        if remainder == "":
            if index >= len(lines) or lines[index][1] <= level:
                mapping[key] = {}
                continue
            child, index = _parse_map(lines, index, indent=lines[index][1])
            mapping[key] = child
            continue
        if remainder == "{}":
            mapping[key] = {}
            continue
        mapping[key] = _scalar(remainder, number)
    return mapping, index


def _scalar(text: str, line: int):
    if text in ("null", "~", "Null", "NULL"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    if text[:1] in "{[":
        raise ValueError(f"unsupported value on line {line}: {text}")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text
