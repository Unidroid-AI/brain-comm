# Copyright (c) 2026, Unidroid
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from brain_comm.protocol import decode_message, encode_message
from brain_comm.yaml_min import load_yaml


def test_ndarray_roundtrip():
    original = {
        "type": "act",
        "step": 3,
        "obs": {
            "cam": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
            "joint_pos": np.array([[0.25, -1.5]], dtype=np.float32),
        },
    }
    restored = decode_message(encode_message(original))
    assert restored["step"] == 3
    assert restored["obs"]["cam"].dtype == np.uint8
    assert np.array_equal(restored["obs"]["cam"], original["obs"]["cam"])
    assert restored["obs"]["joint_pos"].dtype == np.float32
    assert np.allclose(restored["obs"]["joint_pos"], original["obs"]["joint_pos"])


def test_agent_yaml_subset():
    parsed = load_yaml(
        """
entry: zero:ZeroBrain
horizon: 8
action_dim: 8
max_gpu_mem_gb: 1.5
parent_submission: null
weights:
  model.bin: abcd
"""
    )
    assert parsed["entry"] == "zero:ZeroBrain"
    assert parsed["horizon"] == 8
    assert parsed["max_gpu_mem_gb"] == 1.5
    assert parsed["parent_submission"] is None
    assert parsed["weights"] == {"model.bin": "abcd"}


def test_empty_weights_mapping():
    assert load_yaml("weights: {}\n")["weights"] == {}
