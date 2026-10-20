#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization,
and the full LayerNormalization (added by qwen-vision-kernels PR).

These ops are used in Llama-3.1-8B / Qwen / GPT-OSS:
- SimplifiedLayerNormalization is registered in ORT's default domain.
- SkipSimplifiedLayerNormalization is a com.microsoft contrib op.
- LayerNormalization is the full ONNX-17 op:
      y = (x - mean) * rsqrt(var + eps) * scale + bias
  with optional `bias` input and optional `mean`/`inv_std` outputs. The
  runtime implements a block-per-row HIP kernel with fp32 internal math.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes
from framework.ort_ep_backend import _parse_ep_options

SEQ_LENS = [1, 128]
HIDDEN = 4096
# Odd row widths: the RMS-norm kernels only take their packed __half2 path
# when the row is even, so these pin down the scalar fp16 fallback. 33 also
# lands mid-block (block size 256), exercising the strided loop tail.
ODD_HIDDENS = [33, 4095]


_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


@pytest.mark.parametrize("operation", ["LayerNormalization", "Softmax"])
def test_bfloat16_untyped_arithmetic_rejected(request, tmp_path, operation):
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    nodes = [
        helper.make_node("Cast", ["X"], ["bf_X"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["scale"], ["bf_scale"], to=TensorProto.BFLOAT16),
        helper.make_node(
            operation,
            ["bf_X", "bf_scale"] if operation == "LayerNormalization" else ["bf_X"],
            ["bf_Y"],
            axis=-1,
        ),
        helper.make_node("Cast", ["bf_Y"], ["Y"], to=TensorProto.FLOAT),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "unsupported_bf16_layer_norm",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3]),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "unsupported.onnx"
    onnx.save(model, path)
    message = f"BF16 {'standalone ' if operation == 'Softmax' else ''}{operation} is not supported"
    # Strict compilation terminates on unsupported graphs; isolate the
    # expected rejection so it cannot abort the positive numeric suite.
    script = """
import json
import onnxruntime as ort
import sys
provider, library, path, options = sys.argv[1:]
ort.register_execution_provider_library(provider, library)
devices = [d for d in ort.get_ep_devices() if d.ep_name == provider]
assert len(devices) == 1
config = ort.SessionOptions()
config.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
config.add_provider_for_devices(devices, json.loads(options))
ort.InferenceSession(path, sess_options=config)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            request.config.getoption("--ep-name") + "_bf16_rejected",
            str(Path(library).resolve()),
            str(path),
            json.dumps(_parse_ep_options(request.config.getoption("--ep-option"))),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


@pytest.mark.parametrize("hidden", [33, 5120])
@pytest.mark.parametrize("skip", [False, True])
def test_bfloat16_rms_norm(model_runner, hidden, skip):
    names = ["X", "scale", *(["skip", "bias"] if skip else [])]
    shapes = {
        "X": [2, hidden],
        "scale": [hidden],
        "skip": [2, hidden],
        "bias": [hidden],
    }
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name])
        for name in names
    ]

    def make(reference, with_sum):
        nodes = []
        operands = {}
        for name in names:
            operand = name + "_bf16"
            nodes.append(
                helper.make_node("Cast", [name], [operand], to=TensorProto.BFLOAT16)
            )
            if reference:
                nodes.append(
                    helper.make_node(
                        "Cast", [operand], [name + "_fp32"], to=TensorProto.FLOAT
                    )
                )
                operand = name + "_fp32"
            operands[name] = operand
        if skip:
            nodes.append(
                helper.make_node(
                    "SkipSimplifiedLayerNormalization",
                    [operands[name] for name in ("X", "skip", "scale", "bias")],
                    ["normalized", "", "", "sum"] if with_sum else ["normalized"],
                    domain="com.microsoft",
                    epsilon=1e-5,
                )
            )
        else:
            nodes.append(
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    [operands["X"], operands["scale"]],
                    ["normalized"],
                    axis=-1,
                    epsilon=1e-5,
                    stash_type=1,
                )
            )
        output_names = ["normalized", *(["sum"] if with_sum else [])]
        outputs = []
        for name in output_names:
            source = name
            if reference:
                source = name + "_rounded"
                nodes.append(
                    helper.make_node("Cast", [name], [source], to=TensorProto.BFLOAT16)
                )
            nodes.append(
                helper.make_node(
                    "Cast", [source], [name + "_output"], to=TensorProto.FLOAT
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    name + "_output", TensorProto.FLOAT, [2, hidden]
                )
            )
        model = make_model_from_nodes(
            nodes,
            inputs,
            outputs,
            extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 10
        return model

    rng = np.random.default_rng(942)
    values = [
        (
            rng.uniform(-1, 1, shapes[name]) * (131072 if name in ("X", "skip") else 1)
        ).astype(np.float32)
        for name in names
    ]
    actual, expected = model_runner.run_sample(
        make(False, skip), values, reference_model=make(True, skip)
    )
    compare_outputs(actual[:1], expected[:1], atol=0.03125, rtol=0.008)
    assert np.all((actual[0].view(np.uint32) & 0xFFFF) == 0)
    if skip:
        np.testing.assert_array_equal(actual[1], expected[1])
        without_sum, _ = model_runner.run_sample(
            make(False, False),
            values,
            reference_model=make(True, False),
            name=f"bf16_without_sum_{hidden}",
        )
        np.testing.assert_array_equal(actual[0], without_sum[0])


def _make_simplified_layer_norm_model(
    input_shape: list[int], dtype: np.dtype = np.float16
):
    """Build a SimplifiedLayerNormalization ONNX model.

    ORT registers this op in the default ONNX domain (not com.microsoft).
    """
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)

    rng = np.random.default_rng(77)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
    scale_init = numpy_helper.from_array(scale_data, name="scale")

    node = helper.make_node(
        "SimplifiedLayerNormalization",
        ["X", "scale"],
        ["Y"],
        axis=-1,
        epsilon=1e-5,
        stash_type=1,
    )
    model = make_model_from_nodes(
        [node], [X], [Y], initializers=[scale_init], extra_opsets=[]
    )
    return model


def _make_skip_simplified_layer_norm_model(
    input_shape: list[int],
    dtype: np.dtype = np.float16,
    *,
    with_sum: bool = True,
    with_bias: bool = False,
):
    """Build a SkipSimplifiedLayerNormalization ONNX model.

    This op is in the com.microsoft domain. ORT CPU provider supports it.
    """
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    skip = helper.make_tensor_value_info("skip", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", tp, input_shape)

    rng = np.random.default_rng(88)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
    scale_init = numpy_helper.from_array(scale_data, name="scale")
    initializers = [scale_init]
    inputs = ["X", "skip", "scale"]
    if with_bias:
        bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(dtype)
        initializers.append(numpy_helper.from_array(bias_data, name="bias"))
        inputs.append("bias")

    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs,
        ["Y", "", "", "Y3"] if with_sum else ["Y"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3] if with_sum else [Y],
        initializers=initializers,
        extra_opsets=[ms_opset],
    )
    model.ir_version = 10
    return model


def _make_skip_layer_norm_model(input_shape: list[int]):
    """Build a standard com.microsoft.SkipLayerNormalization ONNX model (f16).

    This is mean-subtracting LayerNorm WITH a skip/residual add and bias (beta),
    distinct from SkipSimplifiedLayerNormalization (RMS norm). 4-input form
    (input, skip, gamma, beta), 4-output (output, mean, inv_std, sum) with only
    output[0] and output[3] consumed -- exactly the Whisper encoder pattern.

    Semantics (MS spec, 4-input / no input-bias):
        sum    = input + skip
        output = LayerNorm(sum, gamma, beta, epsilon)
        output[3] (input_skip_bias_sum) = sum

    ORT CPU provider supports it. Validates the converter's hip.add +
    hip.layer_norm composition end-to-end.
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    # output[3] = input_skip_bias_sum (Whisper consumes this as next residual).
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(66)
    gamma_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    beta_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    gamma_init = numpy_helper.from_array(gamma_data, name="gamma")
    beta_init = numpy_helper.from_array(beta_data, name="beta")

    node = helper.make_node(
        "SkipLayerNormalization",
        ["X", "skip", "gamma", "beta"],
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=[gamma_init, beta_init],
        extra_opsets=[ms_opset],
    )
    return model


class TestSkipLayerNorm:
    """Standard SkipLayerNormalization (Whisper encoder, 4-in / output[0]+[3])."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_skip_layer_norm(self, model_runner, input_shape):
        model = _make_skip_layer_norm_model(input_shape)
        rng = np.random.default_rng(77)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        # output[0] = LayerNorm(x+skip); output[3] = x+skip. Block-per-row fp32
        # reduction + fp16 I/O round-trip -> ~2e-3.
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    def test_skip_layer_norm_whisper_shape(self, model_runner):
        """SkipLayerNorm at the real Whisper-large-v3 encoder hidden size 1280."""
        input_shape = [1, 16, 1280]
        model = _make_skip_layer_norm_model(input_shape)
        rng = np.random.default_rng(78)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _make_skip_layer_norm_bias_model(input_shape: list[int]):
    """5-input SkipLayerNormalization: bias is added to input before the skip.

    Stable Diffusion's text encoder uses this form (SkipLayerNorm_AddBias):
        sum    = input + bias + skip
        output = LayerNorm(sum, gamma, beta, epsilon)
        output[3] = sum
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(66)
    gamma_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    beta_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    gamma_init = numpy_helper.from_array(gamma_data, name="gamma")
    beta_init = numpy_helper.from_array(beta_data, name="beta")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    node = helper.make_node(
        "SkipLayerNormalization",
        ["X", "skip", "gamma", "beta", "bias"],
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=[gamma_init, beta_init, bias_init],
        extra_opsets=[ms_opset],
    )


class TestSkipLayerNormInputBias:
    """5-input SkipLayerNormalization (Stable Diffusion text encoder)."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 1024],
        ],
    )
    def test_skip_layer_norm_input_bias(self, model_runner, input_shape):
        model = _make_skip_layer_norm_bias_model(input_shape)
        rng = np.random.default_rng(79)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


class TestSimplifiedLayerNorm:
    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_simplified_layer_norm(self, model_runner, input_shape):
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_simplified_layer_norm_llama_shape(self, model_runner, seq_len):
        """SimplifiedLayerNorm with shapes matching Llama input_layernorm.

        HIDDEN=4096 is also the case that would regress first if the kernel's
        sum-of-squares accumulator were ever narrowed from fp32 to fp16.
        """
        input_shape = [1, seq_len, HIDDEN]
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("hidden", ODD_HIDDENS)
    def test_simplified_layer_norm_odd_hidden(self, model_runner, hidden):
        """Odd hidden size forces the scalar fp16 body.

        The packed __half2 path needs an even row, so without this the scalar
        fp16 fallback is never exercised -- every other case here is even.
        """
        input_shape = [1, 4, hidden]
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(43)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("input_shape", [[1, 4, 16], [1, 8, 4096]])
    def test_simplified_layer_norm_fp32(self, model_runner, input_shape):
        model = _make_simplified_layer_norm_model(input_shape, np.float32)

        rng = np.random.default_rng(44)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5)


class TestSkipSimplifiedLayerNorm:
    @pytest.mark.parametrize("hidden", [33, 896])
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_optional_residual_preserves_normalized_output(
        self, model_runner, hidden, dtype, with_bias
    ):
        shape = [1, 4, hidden]
        rng = np.random.default_rng(109)
        inputs = [rng.uniform(-2, 2, shape).astype(dtype) for _ in range(2)]
        results = []
        for with_sum in (False, True):
            model = _make_skip_simplified_layer_norm_model(
                shape, dtype, with_sum=with_sum, with_bias=with_bias
            )
            actual, expected = model_runner.run_sample(
                model,
                inputs,
                name=f"optional_residual_{hidden}_{dtype.__name__}_{with_bias}_{with_sum}",
            )
            compare_outputs(
                actual, expected, atol=2e-3 if dtype == np.float16 else 1e-5
            )
            results.append(actual[0])
        np.testing.assert_array_equal(results[0], results[1])

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_skip_simplified_layer_norm(self, model_runner, input_shape):
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(55)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_skip_simplified_layer_norm_llama_shape(self, model_runner, seq_len):
        """SkipSimplifiedLayerNorm with shapes matching Llama post_attention_layernorm."""
        input_shape = [1, seq_len, HIDDEN]
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(55)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("hidden", ODD_HIDDENS)
    def test_skip_simplified_layer_norm_odd_hidden(self, model_runner, hidden):
        """Odd hidden size forces the scalar fp16 body (see the RMS-norm twin)."""
        input_shape = [1, 4, hidden]
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(56)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("input_shape", [[1, 4, 16], [1, 8, 4096]])
    def test_skip_simplified_layer_norm_fp32(self, model_runner, input_shape):
        model = _make_skip_simplified_layer_norm_model(input_shape, np.float32)

        rng = np.random.default_rng(57)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Full LayerNormalization (added by qwen-vision-kernels PR)
# ---------------------------------------------------------------------------


def _make_layer_norm_model(
    input_shape: list[int],
    dtype: np.dtype = np.float16,
    with_bias: bool = False,
):
    """Build a full LayerNormalization ONNX model.

    Output is single (no `mean` / `inv_std`) -- the runtime supports those
    optional outputs but they are emitted only when the producing graph
    requests them; this signature is the most common in practice.
    """
    hidden = input_shape[-1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)

    rng = np.random.default_rng(99)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
    scale_init = numpy_helper.from_array(scale_data, name="scale")
    initializers = [scale_init]
    input_names = ["X", "scale"]

    if with_bias:
        bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(dtype)
        bias_init = numpy_helper.from_array(bias_data, name="bias")
        initializers.append(bias_init)
        input_names.append("bias")

    node = helper.make_node(
        "LayerNormalization",
        input_names,
        ["Y"],
        axis=-1,
        epsilon=1e-5,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=initializers)


class TestLayerNormalization:
    """Full LN: (x - mean) * rsqrt(var + eps) * scale [+ bias]."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
            [2, 4, 64],
        ],
    )
    def test_layer_norm_no_bias(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=False)
        rng = np.random.default_rng(1101)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        # Block-per-row reduction in fp32, fp16 round-trip on I/O -> ~1e-3.
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 8, 32],
            [2, 4, 64],
        ],
    )
    def test_layer_norm_with_bias(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=True)
        rng = np.random.default_rng(1102)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_layer_norm_llama_shape(self, model_runner, seq_len):
        """fp16 LN with bias on a Llama-style [1, S, 4096] tensor."""
        input_shape = [1, seq_len, HIDDEN]
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=True)
        rng = np.random.default_rng(1103)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    def test_layer_norm_fp32(self, model_runner):
        """LN on f32 input with bias -- exercises the f32 dispatch path."""
        input_shape = [1, 8, 64]
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        rng = np.random.default_rng(1104)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("num_patches", [256, 1024])
    def test_layer_norm_qwen35b_vision_shape(self, model_runner, num_patches):
        """LN exactly as it appears 55x in Qwen3.5-35B vision.onnx:
        in:    f32 [num_patches, 1152]
        scale: f32 [1152]
        bias:  f32 [1152]
        axis=-1, epsilon=1e-6, stash_type=1
        """
        input_shape = [num_patches, 1152]
        hidden = input_shape[-1]
        tp = TensorProto.FLOAT
        X = helper.make_tensor_value_info("X", tp, input_shape)
        Y = helper.make_tensor_value_info("Y", tp, input_shape)
        rng = np.random.default_rng(1105)
        scale_init = numpy_helper.from_array(
            rng.uniform(0.5, 1.5, [hidden]).astype(np.float32), name="scale"
        )
        bias_init = numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [hidden]).astype(np.float32), name="bias"
        )
        node = helper.make_node(
            "LayerNormalization",
            ["X", "scale", "bias"],
            ["Y"],
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )
        model = make_model_from_nodes(
            [node], [X], [Y], initializers=[scale_init, bias_init]
        )
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)


def _make_instance_norm_model(input_shape: list[int], dtype=np.float32):
    """Build an ONNX InstanceNormalization model (scale/B as initializers)."""
    channels = input_shape[1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(2026)
    scale_init = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="scale"
    )
    bias_init = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="B"
    )
    node = helper.make_node(
        "InstanceNormalization",
        ["X", "scale", "B"],
        ["Y"],
        epsilon=1e-5,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=[scale_init, bias_init])


class TestInstanceNormalization:
    """Per-(N, C) mean/var over spatial axes, then channel affine."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 8],
            [2, 3, 8, 8],
            [1, 3, 4, 5, 6],
        ],
    )
    def test_instance_norm_f32(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float32)
        rng = np.random.default_rng(7)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [2, 3, 8, 8],
        ],
    )
    def test_instance_norm_f16(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float16)
        rng = np.random.default_rng(8)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _make_batch_norm_model(input_shape: list[int], dtype=np.float32):
    """Build an inference ONNX BatchNormalization model.

    Scale, bias, mean, and variance are per-channel initializers.
    """
    channels = input_shape[1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(2026)
    scale_init = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="scale"
    )
    bias_init = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="B"
    )
    mean_init = numpy_helper.from_array(
        rng.uniform(-1.0, 1.0, [channels]).astype(dtype), name="mean"
    )
    var_init = numpy_helper.from_array(
        rng.uniform(0.1, 1.5, [channels]).astype(dtype), name="var"
    )
    node = helper.make_node(
        "BatchNormalization",
        ["X", "scale", "B", "mean", "var"],
        ["Y"],
        epsilon=1e-5,
    )
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[scale_init, bias_init, mean_init, var_init],
    )


class TestBatchNormalization:
    """Inference BN: supplied per-channel mean and variance, one kernel."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 1, 1],
            [2, 3, 8],
            [1, 3, 4, 5],
            [1, 4, 2, 3, 3],
        ],
    )
    def test_batch_norm_f32(self, model_runner, input_shape):
        model = _make_batch_norm_model(input_shape, np.float32)
        rng = np.random.default_rng(9)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 1, 1],
            [2, 3, 8, 8],
        ],
    )
    def test_batch_norm_f16(self, model_runner, input_shape):
        model = _make_batch_norm_model(input_shape, np.float16)
        rng = np.random.default_rng(10)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _make_group_norm_model(
    input_shape: list[int],
    groups: int,
    *,
    channels_last: int = 0,
    activation: int = 0,
    dtype=np.float32,
    epsilon: float = 1e-5,
):
    """Build a com.microsoft.GroupNorm model. Channel axis is 1, or -1 when
    channels_last is set."""
    channels = input_shape[-1] if channels_last else input_shape[1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(11)
    gamma = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="gamma"
    )
    beta = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="beta"
    )
    node = helper.make_node(
        "GroupNorm",
        ["X", "gamma", "beta"],
        ["Y"],
        domain="com.microsoft",
        groups=groups,
        epsilon=epsilon,
        activation=activation,
        channels_last=channels_last,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[gamma, beta],
        extra_opsets=[ms_opset],
    )


class TestGroupNorm:
    """com.microsoft GroupNorm: per-group stats, optional SiLU."""

    def test_group_norm_nchw_f32(self, model_runner):
        shape = [1, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=4, activation=0)
        rng = np.random.default_rng(12)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nchw_silu_f32(self, model_runner):
        shape = [2, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=4, activation=1)
        rng = np.random.default_rng(13)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nhwc_f32(self, model_runner):
        shape = [1, 4, 4, 8]
        model = _make_group_norm_model(shape, groups=4, channels_last=1, activation=0)
        rng = np.random.default_rng(14)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nchw_silu_f16(self, model_runner):
        shape = [1, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=2, activation=1, dtype=np.float16)
        rng = np.random.default_rng(15)
        x = rng.uniform(-2, 2, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
