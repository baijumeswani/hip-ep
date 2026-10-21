#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX TopK op (values + indices outputs).

K is supplied as a constant initializer so the ONNX shape-inference resolves
the output extent (k) statically -- the runtime still reads k from the K
buffer at run time (real/top_k.cpp does a D2H copy of k), so the GPU kernel is
exercised. `largest` and `sorted` are both covered.

Covers small and vocabulary-sized axes, lower-index ties, infinities and
legacy NaN ordering. Finite selections compare bit-exactly with ORT CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_topk_model(x_shape: list[int], k: int, axis: int, largest: int, sorted_: int):
    out_shape = list(x_shape)
    out_shape[axis] = k
    X = helper.make_tensor_value_info("x", np_to_onnx_type(np.float32), list(x_shape))
    values = helper.make_tensor_value_info(
        "values", np_to_onnx_type(np.float32), out_shape
    )
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(np.int64), out_shape
    )
    k_init = numpy_helper.from_array(np.array([k], dtype=np.int64), name="k")
    node = helper.make_node(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        axis=axis,
        largest=largest,
        sorted=sorted_,
    )
    return make_model_from_nodes([node], [X], [values, indices], initializers=[k_init])


def _distinct_rows(rng, rows: int, cols: int) -> np.ndarray:
    """Each row is a shuffled arange -> globally distinct within a row, so
    TopK has no ties to break."""
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        out[r] = rng.permutation(cols).astype(np.float32) + rng.uniform(
            -0.25, 0.25, cols
        ).astype(np.float32)
    return out


class TestTopK:
    @pytest.mark.parametrize("largest", [0, 1])
    @pytest.mark.parametrize("k", [3, 16])
    def test_wide_nan_preserves_legacy_scan(self, model_runner, largest, k):
        values = np.tile(np.arange(257, dtype=np.float32), (2, 1))
        values[0, 0] = values[1, 1] = np.nan
        model = _make_topk_model([2, 257], k, axis=1, largest=largest, sorted_=1)
        chosen = np.array(
            [[0, *range(256, 257 - k, -1)], list(range(256, 256 - k, -1))]
            if largest
            else [list(range(k)), list(range(k))],
            dtype=np.int64,
        )
        expected_values = np.take_along_axis(values, chosen, axis=1)
        reference = make_model_from_nodes(
            [
                helper.make_node("Identity", ["ref_values"], ["values"]),
                helper.make_node("Identity", ["ref_indices"], ["indices"]),
            ],
            list(model.graph.input),
            list(model.graph.output),
            initializers=[
                numpy_helper.from_array(expected_values, "ref_values"),
                numpy_helper.from_array(chosen, "ref_indices"),
            ],
        )
        actual, expected = model_runner.run_sample(
            model, [values], reference="cpu", reference_model=reference
        )
        for output, legacy in zip(actual, expected):
            np.testing.assert_array_equal(output, legacy)

    @pytest.mark.parametrize("largest", [0, 1])
    @pytest.mark.parametrize("sorted_", [0, 1])
    @pytest.mark.parametrize("shape,k", [([7, 248320], 16), ([2, 3, 513], 32)])
    def test_wide_last_axis(self, model_runner, largest, sorted_, shape, k):
        model = _make_topk_model(shape, k, axis=-1, largest=largest, sorted_=sorted_)
        rng = np.random.default_rng(942)
        values = rng.integers(-100, 101, shape).astype(np.float32)
        values[..., 0] = -np.inf
        values[..., -1] = np.inf
        actual, expected = model_runner.run_sample(model, [values], reference="cpu")
        if not sorted_:
            np.testing.assert_array_equal(
                np.sort(actual[1], axis=-1), np.sort(expected[1], axis=-1)
            )
            np.testing.assert_array_equal(
                actual[0], np.take_along_axis(values, actual[1], axis=-1)
            )
            return
        for output, reference in zip(actual, expected):
            np.testing.assert_array_equal(output, reference)

    @pytest.mark.parametrize("largest", [0, 1])
    def test_wide_equal_values_prefer_lower_indices(self, model_runner, largest):
        model = _make_topk_model([3, 257], 16, axis=1, largest=largest, sorted_=1)
        values = np.zeros((3, 257), dtype=np.float32)
        values[0, ::2] = -0.0
        values[1] = -np.inf
        values[2] = np.inf
        actual, expected = model_runner.run_sample(model, [values], reference="cpu")
        np.testing.assert_array_equal(actual[1], np.tile(np.arange(16), (3, 1)))
        for output, reference in zip(actual, expected):
            np.testing.assert_array_equal(output, reference)
        np.testing.assert_array_equal(
            actual[0].view(np.uint32), expected[0].view(np.uint32)
        )

    @pytest.mark.parametrize("largest", [0, 1])
    def test_non_last_axis(self, model_runner, largest):
        model = _make_topk_model([3, 4, 9], 3, axis=1, largest=largest, sorted_=1)
        values = np.random.default_rng(942).normal(size=(3, 4, 9)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [values], reference="cpu")
        for output, reference in zip(actual, expected):
            np.testing.assert_array_equal(output, reference)

    def test_rejected_selection_fails_inference(self, model_runner):
        from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

        model = _make_topk_model([2, 300], 257, axis=1, largest=1, sorted_=1)
        with pytest.raises(RuntimeException, match="inference_compute"):
            model_runner.run_sample(model, [np.zeros((2, 300), dtype=np.float32)])

    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize("sorted_", [1])
    def test_topk_axis1(self, model_runner, largest, sorted_):
        x_shape = [3, 8]
        k = 3
        model = _make_topk_model(x_shape, k, axis=1, largest=largest, sorted_=sorted_)
        rng = np.random.default_rng(601)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_full_k(self, model_runner):
        """k == axis length -> a full sort of each row."""
        x_shape = [4, 5]
        k = 5
        model = _make_topk_model(x_shape, k, axis=1, largest=1, sorted_=1)
        rng = np.random.default_rng(602)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
