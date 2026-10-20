# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Length-prefixed msgpack frames with numpy arrays."""

from __future__ import annotations

import struct
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = "1.0"

# Two cameras at 720x1280 plus state, with headroom for a chunk of actions.
MAX_FRAME_BYTES = 256 * 1024 * 1024

_ARRAY_KEY = "__ndarray__"


class ProtocolError(ValueError):
    """A frame or message did not match the protocol."""


def encode_message(message: dict[str, Any]) -> bytes:
    """Serialize one message, embedding numpy arrays as raw bytes."""
    return msgpack.packb(message, default=_encode_object, use_bin_type=True)


def decode_message(payload: bytes | bytearray) -> dict[str, Any]:
    """Deserialize one message and restore numpy arrays."""
    message = msgpack.unpackb(payload, object_hook=_decode_object, raw=False)
    if not isinstance(message, dict) or "type" not in message:
        raise ProtocolError("message must be a dict with a type field")
    return message


def send_message(sock, message: dict[str, Any]) -> None:
    """Write one length-prefixed frame."""
    payload = encode_message(message)
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is {len(payload)} bytes; limit is {MAX_FRAME_BYTES}")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_message(sock) -> dict[str, Any]:
    """Read one length-prefixed frame."""
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame claims {length} bytes; limit is {MAX_FRAME_BYTES}")
    return decode_message(_recv_exact(sock, length))


def _recv_exact(sock, size: int) -> bytearray:
    """Read exactly ``size`` bytes into one preallocated buffer."""
    buf = bytearray(size)
    view = memoryview(buf)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:], size - received)
        if not count:
            raise ConnectionError("connection closed")
        received += count
    return buf


def _encode_object(obj: Any):
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {
            _ARRAY_KEY: True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            # A view lets msgpack copy the bytes once, into the frame.
            "data": memoryview(array.reshape(-1).view(np.uint8)),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot encode {type(obj).__name__}")


def _decode_object(obj: dict) -> Any:
    if obj.get(_ARRAY_KEY) is not True:
        return obj
    array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
    return array.reshape(obj["shape"]).copy()
