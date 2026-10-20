#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Cast operation.

The Llama-3.1-8B model uses Cast i64 -> i32 in the attention mask subgraph.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

from framework.comparator import compare_outputs
from framework.onnx_utils import (
    make_model_from_nodes,
    np_to_onnx_type,
    onnx_type_to_np,
)

SEQ_LENS = [1, 128]


def _make_cast_model(from_dtype, to_onnx_type: int, shape: list[int]):
    """Build a Cast ONNX model."""
    tp_in = np_to_onnx_type(from_dtype)
    X = helper.make_tensor_value_info("X", tp_in, shape)
    Y = helper.make_tensor_value_info("Y", to_onnx_type, shape)
    node = helper.make_node("Cast", ["X"], ["Y"], to=to_onnx_type)
    return make_model_from_nodes([node], [X], [Y])


class TestCast:
    def test_byte_mask_to_bfloat16(self, model_runner):
        model = make_model_from_nodes(
            [
                helper.make_node("Cast", ["X"], ["bfloat"], to=TensorProto.BFLOAT16),
                helper.make_node("Cast", ["bfloat"], ["Y"], to=TensorProto.FLOAT),
            ],
            [helper.make_tensor_value_info("X", TensorProto.UINT8, [513])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [513])],
        )
        values = (np.arange(513) % 256).astype(np.uint8)
        actual, expected = model_runner.run_sample(model, [values], reference="cpu")
        np.testing.assert_array_equal(actual[0], values.astype(np.float32))
        np.testing.assert_array_equal(actual[0], expected[0])

    @pytest.mark.parametrize("input_dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("output_dtype", [np.float32, np.float16])
    def test_bfloat16_bridge(self, model_runner, input_dtype, output_dtype):
        inputs = [
            helper.make_tensor_value_info("X", np_to_onnx_type(input_dtype), [513])
        ]
        outputs = [
            helper.make_tensor_value_info("Y", np_to_onnx_type(output_dtype), [513])
        ]
        model = make_model_from_nodes(
            [
                helper.make_node("Cast", ["X"], ["bfloat"], to=TensorProto.BFLOAT16),
                helper.make_node(
                    "Cast", ["bfloat"], ["Y"], to=np_to_onnx_type(output_dtype)
                ),
            ],
            inputs,
            outputs,
        )
        values = np.linspace(-63.9375, 63.9375, 513).astype(input_dtype)
        actual, expected = model_runner.run_sample(model, [values], reference="cpu")
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_unsupported_cast_reports_execution_failure(self, model_runner, request):
        if request.config.getoption("--backend") != "ort_ep" or (
            "profile=hip" not in request.config.getoption("--ep-option")
        ):
            pytest.skip("HIP execution-status contract through the AMDGPU wrapper")
        model = _make_cast_model(np.float32, TensorProto.DOUBLE, [4])
        with pytest.raises(RuntimeException, match="inference_compute.*failed"):
            model_runner.run_sample(model, [np.arange(4, dtype=np.float32)])

    @pytest.mark.parametrize(
        "from_dtype,to_onnx_type,shape",
        [
            (np.int64, TensorProto.INT32, [2, 5]),
            (np.int64, TensorProto.INT32, [1, 10]),
        ],
    )
    def test_cast(self, model_runner, from_dtype, to_onnx_type, shape):
        model = _make_cast_model(from_dtype, to_onnx_type, shape)

        rng = np.random.default_rng(88)
        if np.issubdtype(from_dtype, np.integer):
            x = rng.integers(-50, 50, shape, dtype=from_dtype)
        else:
            x = rng.uniform(-10, 10, shape).astype(from_dtype)

        out_dtype = onnx_type_to_np(to_onnx_type)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 0 if np.issubdtype(out_dtype, np.integer) else 1e-5
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_cast_attention_mask_llama_shape(self, model_runner, seq_len):
        """Cast i64 -> i32 matching the Llama attention mask subgraph."""
        model = _make_cast_model(
            np.int64,
            TensorProto.INT32,
            [1, seq_len],
        )

        rng = np.random.default_rng(88)
        x = rng.integers(0, seq_len + 1, [1, seq_len], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
