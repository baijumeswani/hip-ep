#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Native BF16 QK RMS/RoPE and noncausal head-128 attention on reused pages."""

import gc
import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from ml_dtypes import bfloat16
from onnx import TensorProto, helper

from framework.ort_ep_backend import (
    _parse_ep_options,
    numpy_from_ort_value,
    ort_value_from_numpy,
)


@pytest.fixture(scope="module")
def bf16_paged_device(request):
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    assert os.environ.get("HIPDNN_EP_STRICT") == "1"
    provider = request.config.getoption("--ep-name") + "_bf16_paged"
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    device = devices[0]
    yield device, provider
    del device, devices
    gc.collect()
    if request.session.testsfailed:
        warnings.warn(
            "Retaining EP registration: failed-test tracebacks may still own native sessions.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        ort.unregister_execution_provider_library(provider)


@pytest.fixture(scope="module", params=["packed", "separate"])
def bf16_paged_session(request, tmp_path_factory, bf16_paged_device):
    device, provider = bf16_paged_device
    separate = request.param == "separate"
    root = tmp_path_factory.mktemp("bf16_paged")
    cache_shape = ["blocks", "page_size", 8, 128]
    shapes = {
        "query": (TensorProto.BFLOAT16, ["tokens", 4096 if separate else 6144]),
        "past_key": (TensorProto.BFLOAT16, cache_shape),
        "past_value": (TensorProto.BFLOAT16, cache_shape),
        "cumulative": (TensorProto.INT32, ["batch_plus_one"]),
        "past": (TensorProto.INT32, ["batch"]),
        "table": (TensorProto.INT32, ["batch", "columns"]),
        "cos": (TensorProto.BFLOAT16, ["rotary_length", 64]),
        "sin": (TensorProto.BFLOAT16, ["rotary_length", 64]),
        "q_norm": (TensorProto.BFLOAT16, [128]),
        "k_norm": (TensorProto.BFLOAT16, [128]),
        "metadata": (TensorProto.INT32, [3]),
    }
    if separate:
        shapes["key"] = (TensorProto.BFLOAT16, ["key_tokens", 1024])
        shapes["value"] = (TensorProto.BFLOAT16, ["value_tokens", 1024])
    graph = helper.make_graph(
        [
            helper.make_node(
                "PagedAttention",
                [
                    "query",
                    "key" if separate else "",
                    "value" if separate else "",
                    "past_key",
                    "past_value",
                    "cumulative",
                    "past",
                    "table",
                    "cos",
                    "sin",
                    "",
                    "",
                    "q_norm",
                    "k_norm",
                    "",
                    "",
                    "metadata",
                ],
                ["output", "present_key", "present_value"],
                domain="com.microsoft",
                num_heads=32,
                kv_num_heads=8,
                is_causal=0,
                do_rotary=1,
                rotary_interleaved=0,
                local_window_size=2048,
                qk_norm_epsilon=1e-6,
            )
        ],
        "bf16_paged",
        [
            helper.make_tensor_value_info(name, dtype, shape)
            for name, (dtype, shape) in shapes.items()
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.BFLOAT16, ["tokens", 4096]
            ),
            helper.make_tensor_value_info(
                "present_key", TensorProto.BFLOAT16, cache_shape
            ),
            helper.make_tensor_value_info(
                "present_value", TensorProto.BFLOAT16, cache_shape
            ),
        ],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    onnx.checker.check_model(model)
    path = root / "paged.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.use_env_allocators", "1")
    options.add_provider_for_devices(
        [device], _parse_ep_options(request.config.getoption("--ep-option"))
    )
    options.enable_profiling = True
    options.profile_file_prefix = str(root / "ort")
    session = ort.InferenceSession(str(path), sess_options=options)
    context = {"session": session, "device": device, "separate": separate}
    yield context
    events = json.loads(Path(session.end_profiling()).read_text())
    assigned = {
        event["args"]["provider"]
        for event in events
        if "provider" in event.get("args", {})
    }
    context.clear()
    del session, options, device
    gc.collect()
    assert assigned == {provider}


def make_windowed_inputs(prefixes, lengths):
    rng = np.random.default_rng(942)
    batch = len(prefixes)
    total = max(p + q for p, q in zip(prefixes, lengths))
    columns = (total + 255) // 256 + 2
    # Every query needs its history until the entire append has completed.
    ring_pages = max(10, (2047 + max(lengths) + 255) // 256 + 1)
    pages = rng.permutation((batch + 1) * ring_pages).reshape(batch + 1, ring_pages)
    table = pages[:batch, np.arange(columns) % ring_pages].astype(np.int32)
    key = np.full(((batch + 1) * ring_pages, 256, 8, 128), np.nan, dtype=bfloat16)
    value = key.copy()
    for request, prefix in enumerate(prefixes):
        positions = np.arange(max(0, prefix - 2048), prefix)
        slots = table[request, positions // 256], positions % 256
        key[slots] = rng.normal(0, 0.3, (len(positions), 8, 128)).astype(bfloat16)
        value[slots] = rng.normal(0, 0.3, (len(positions), 8, 128)).astype(bfloat16)
    angle = np.arange(total, dtype=np.float32)[:, None] * (
        10000.0 ** (-np.arange(64, dtype=np.float32) / 64)
    )
    return {
        "query": rng.normal(0, 0.3, (sum(lengths), 6144)).astype(bfloat16),
        "past_key": key,
        "past_value": value,
        "cumulative": np.cumsum([0, *lengths], dtype=np.int32),
        "past": np.asarray(prefixes, dtype=np.int32),
        "table": table,
        "cos": np.cos(angle).astype(bfloat16),
        "sin": np.sin(angle).astype(bfloat16),
        "q_norm": rng.uniform(0.7, 1.3, 128).astype(bfloat16),
        "k_norm": rng.uniform(0.7, 1.3, 128).astype(bfloat16),
        "metadata": np.asarray(
            [max(lengths), total, min(p + q for p, q in zip(prefixes, lengths))],
            dtype=np.int32,
        ),
    }


def normalize_rotate(values, weight, cosine, sine):
    source = values.astype(np.float64)
    inverse = 1 / np.sqrt(
        np.mean(source * source, axis=-1, keepdims=True) + float(np.float32(1e-6))
    )
    normalized = (
        (source * inverse * weight.astype(np.float64))
        .astype(bfloat16)
        .astype(np.float32)
    )
    partner = np.concatenate([-normalized[..., 64:], normalized[..., :64]], axis=-1)
    first = (normalized * np.tile(cosine.astype(np.float32), 2)).astype(bfloat16)
    second = (partner * np.tile(sine.astype(np.float32), 2)).astype(bfloat16)
    return (first.astype(np.float32) + second.astype(np.float32)).astype(bfloat16)


def windowed_reference(inputs, selected=None):
    key, value = inputs["past_key"].copy(), inputs["past_value"].copy()
    source = inputs["query"].reshape(-1, 48, 128)
    cumulative = inputs["cumulative"]
    positions = np.concatenate(
        [
            np.arange(prefix, prefix + end - start)
            for prefix, start, end in zip(
                inputs["past"], cumulative[:-1], cumulative[1:]
            )
        ]
    )
    query = normalize_rotate(
        source[:, :32],
        inputs["q_norm"],
        inputs["cos"][positions, None],
        inputs["sin"][positions, None],
    )
    new_key = normalize_rotate(
        source[:, 32:40],
        inputs["k_norm"],
        inputs["cos"][positions, None],
        inputs["sin"][positions, None],
    )
    for request, (start, end) in enumerate(zip(cumulative[:-1], cumulative[1:])):
        slots = (
            inputs["table"][request, positions[start:end] // 256],
            positions[start:end] % 256,
        )
        key[slots] = new_key[start:end]
        value[slots] = source[start:end, 40:]
    selected = np.arange(len(source)) if selected is None else np.asarray(selected)
    outputs = []
    for token in selected:
        request = np.searchsorted(cumulative[1:], token, side="right")
        end = int(inputs["past"][request]) + int(
            cumulative[request + 1] - cumulative[request]
        )
        allowed = np.arange(max(0, int(positions[token]) - 2047), end)
        slots = inputs["table"][request, allowed // 256], allowed % 256
        keys = key[slots].astype(np.float64)
        values = value[slots].astype(np.float64)
        scores = np.einsum(
            "hgd,lhd->hgl", query[token].astype(np.float64).reshape(8, 4, 128), keys
        ) / np.sqrt(128)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        outputs.append(np.einsum("hgl,lhd->hgd", probabilities, values).reshape(4096))
    return np.asarray(outputs, dtype=bfloat16), key, value


@contextmanager
def bind_windowed(context, inputs):
    if context["separate"]:
        inputs = dict(inputs)
        query = inputs["query"]
        inputs["query"] = np.ascontiguousarray(query[:, :4096])
        inputs.setdefault("key", np.ascontiguousarray(query[:, 4096:5120]))
        inputs.setdefault("value", np.ascontiguousarray(query[:, 5120:]))
    memory = context["device"].memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    binding = context["session"].io_binding()
    tensors = {}
    for name, data in inputs.items():
        host = ort_value_from_numpy(data)
        if name == "metadata":
            tensor = host
        else:
            tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                data.shape, host.element_type(), memory_info=memory
            )
            tensor.update_inplace(host)
        binding.bind_ortvalue_input(name, tensor)
        tensors[name] = tensor
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        (len(inputs["query"]), 4096), TensorProto.BFLOAT16, memory_info=memory
    )
    output.update_inplace(
        ort_value_from_numpy(np.full(output.shape(), np.nan, bfloat16))
    )
    binding.bind_ortvalue_output("output", output)
    binding.bind_ortvalue_output("present_key", tensors["past_key"])
    binding.bind_ortvalue_output("present_value", tensors["past_value"])
    bound = {"binding": binding, "tensors": tensors, "output": output}
    try:
        yield bound
    finally:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        tensors.clear()
        bound.clear()
        del tensor, output, host, tensors, binding, memory


@pytest.mark.parametrize(
    "prefixes,lengths",
    [
        ([0], [8]),
        ([255, 2047], [3, 8]),
        ([4095], [8]),
        ([32760, 131064], [8, 8]),
        ([262143], [8]),
        ([65536], [2064]),
    ],
)
def test_bf16_windowed_attention(bf16_paged_session, prefixes, lengths):
    inputs = make_windowed_inputs(prefixes, lengths)
    selected = (
        np.r_[0, 7, 1023, 2047, np.arange(sum(lengths) - 8, sum(lengths))]
        if max(lengths) > 2048
        else None
    )
    expected, key, value = windowed_reference(inputs, selected)
    with bind_windowed(bf16_paged_session, inputs) as bound:
        bf16_paged_session["session"].run_with_iobinding(bound["binding"])
        actual = numpy_from_ort_value(bound["output"])
        assert actual.dtype == np.dtype(bfloat16)
        if selected is not None:
            actual = actual[selected]
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=0.001,
            rtol=0.008,
        )
        np.testing.assert_allclose(
            numpy_from_ort_value(bound["tensors"]["past_key"]).astype(np.float32),
            key.astype(np.float32),
            atol=0.001,
            rtol=0.008,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            numpy_from_ort_value(bound["tensors"]["past_value"]).view(np.uint16),
            value.view(np.uint16),
        )
        assert (
            bound["binding"].get_outputs()[1].data_ptr()
            == bound["tensors"]["past_key"].data_ptr()
        )
        assert (
            bound["binding"].get_outputs()[2].data_ptr()
            == bound["tensors"]["past_value"].data_ptr()
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "page",
        "cumulative",
        "negative_prefix",
        "write_alias",
        "own_write_alias",
        "own_history_alias",
        "table_shape",
        "sine_shape",
        "rotary_bound",
        "cumulative_shape",
        "key_shape",
        "value_shape",
    ],
)
def test_bf16_attention_failure_is_atomic(bf16_paged_session, invalid):
    if invalid in ("key_shape", "value_shape") and not bf16_paged_session["separate"]:
        pytest.skip("Separate key/value extent validation")
    inputs = make_windowed_inputs([255, 510], [3, 5])
    if invalid == "own_write_alias":
        inputs = make_windowed_inputs([0], [257])
        inputs["table"][0, 1] = inputs["table"][0, 0]
    elif invalid == "own_history_alias":
        inputs = make_windowed_inputs([256], [1])
        inputs["table"][0, 1] = inputs["table"][0, 0]
    if invalid == "page":
        inputs["table"][1, 2] = len(inputs["past_key"])
    elif invalid == "cumulative":
        inputs["cumulative"][1] = -1
    elif invalid == "negative_prefix":
        inputs["past"][1] = -1
    elif invalid == "write_alias":
        inputs["table"][1, 1] = inputs["table"][0, 0]
    elif invalid == "table_shape":
        inputs["table"] = inputs["table"][:1].copy()
    elif invalid == "sine_shape":
        inputs["sin"] = inputs["sin"][:-1].copy()
    elif invalid == "rotary_bound":
        inputs["cos"] = inputs["cos"][:-1].copy()
        inputs["sin"] = inputs["sin"][:-1].copy()
    elif invalid == "cumulative_shape":
        inputs["cumulative"] = inputs["cumulative"][:-1].copy()
    elif invalid == "key_shape":
        inputs["key"] = np.ascontiguousarray(inputs["query"][:-1, 4096:5120])
    elif invalid == "value_shape":
        inputs["value"] = np.ascontiguousarray(inputs["query"][:-1, 5120:])
    with bind_windowed(bf16_paged_session, inputs) as bound:
        with pytest.raises(RuntimeError, match="inference_compute .* failed"):
            bf16_paged_session["session"].run_with_iobinding(bound["binding"])
        for name in ("past_key", "past_value"):
            np.testing.assert_array_equal(
                numpy_from_ort_value(bound["tensors"][name]).view(np.uint16),
                inputs[name].view(np.uint16),
            )
        assert np.isnan(numpy_from_ort_value(bound["output"]).astype(np.float32)).all()
    valid = make_windowed_inputs([255, 510], [3, 5])
    expected, _, _ = windowed_reference(valid)
    with bind_windowed(bf16_paged_session, valid) as bound:
        bf16_paged_session["session"].run_with_iobinding(bound["binding"])
        np.testing.assert_allclose(
            numpy_from_ort_value(bound["output"]).astype(np.float32),
            expected.astype(np.float32),
            atol=0.001,
            rtol=0.008,
        )


def test_bf16_attention_ignores_pages_before_window(bf16_paged_session):
    inputs = make_windowed_inputs([8192], [8])
    inputs["table"][0, :24] = -1
    expected, _, _ = windowed_reference(inputs)
    with bind_windowed(bf16_paged_session, inputs) as bound:
        bf16_paged_session["session"].run_with_iobinding(bound["binding"])
        np.testing.assert_allclose(
            numpy_from_ort_value(bound["output"]).astype(np.float32),
            expected.astype(np.float32),
            atol=0.001,
            rtol=0.008,
        )


def test_bf16_qk_norm_preserves_exponent_range(bf16_paged_session):
    inputs = make_windowed_inputs([0], [8])
    source = inputs["query"].reshape(-1, 48, 128)
    source[:, :40] = (source[:, :40].astype(np.float32) * 1e30).astype(bfloat16)
    expected, key, _ = windowed_reference(inputs)
    with bind_windowed(bf16_paged_session, inputs) as bound:
        bf16_paged_session["session"].run_with_iobinding(bound["binding"])
        actual = numpy_from_ort_value(bound["output"]).astype(np.float32)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual, expected.astype(np.float32), atol=0.001, rtol=0.008
        )
        np.testing.assert_allclose(
            numpy_from_ort_value(bound["tensors"]["past_key"]).astype(np.float32),
            key.astype(np.float32),
            atol=0.001,
            rtol=0.008,
            equal_nan=True,
        )
