#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Compiled FP16 paged attention with real caller-bound GPU cache aliases."""

import gc
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from framework.ort_ep_backend import _parse_ep_options


@pytest.fixture(scope="module")
def paged_session(request, tmp_path_factory):
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    if os.environ.get("HIPDNN_EP_STRICT") != "1":
        pytest.fail("PagedAttention qualification requires HIPDNN_EP_STRICT=1")
    provider = request.config.getoption("--ep-name") + "_paged"
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    device = devices[0]
    root = tmp_path_factory.mktemp("paged_attention")
    shapes = {
        "query": (TensorProto.FLOAT16, ["tokens", 1152]),
        "past_key": (TensorProto.FLOAT16, ["blocks", "page_size", 2, 64]),
        "past_value": (TensorProto.FLOAT16, ["blocks", "page_size", 2, 64]),
        "cumulative": (TensorProto.INT32, ["batch_plus_one"]),
        "past": (TensorProto.INT32, ["batch"]),
        "table": (TensorProto.INT32, ["batch", 16]),
        "cos": (TensorProto.FLOAT16, ["rotary_length", 32]),
        "sin": (TensorProto.FLOAT16, ["rotary_length", 32]),
    }
    node = helper.make_node(
        "PagedAttention",
        [
            "query",
            "",
            "",
            "past_key",
            "past_value",
            "cumulative",
            "past",
            "table",
            "cos",
            "sin",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=14,
        kv_num_heads=2,
        do_rotary=1,
    )
    graph = helper.make_graph(
        [node],
        "paged_attention",
        [
            helper.make_tensor_value_info(name, dtype, shape)
            for name, (dtype, shape) in shapes.items()
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT16, ["tokens", 896]
            ),
            helper.make_tensor_value_info(
                "present_key", TensorProto.FLOAT16, ["blocks", "page_size", 2, 64]
            ),
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT16, ["blocks", "page_size", 2, 64]
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    model_path = root / "paged.onnx"
    onnx.save(model, model_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.use_env_allocators", "1")
    options.add_provider_for_devices(
        [device], _parse_ep_options(request.config.getoption("--ep-option"))
    )
    options.enable_profiling = True
    options.profile_file_prefix = str(root / "ort")
    session = ort.InferenceSession(str(model_path), sess_options=options)
    context = {"session": session, "device": device}
    yield context
    events = json.loads(Path(session.end_profiling()).read_text())
    assigned = {
        event["args"]["provider"]
        for event in events
        if "provider" in event.get("args", {})
    }
    assert assigned == {provider}
    context.clear()
    del session, options, device, devices
    gc.collect()
    ort.unregister_execution_provider_library(provider)


def rotate(values, cosine, sine):
    source = values.astype(np.float32)
    partner = np.concatenate([-source[..., 32:], source[..., :32]], axis=-1)
    return (source * np.tile(cosine, 2) + partner * np.tile(sine, 2)).astype(np.float16)


def make_inputs(prefixes, lengths):
    rng = np.random.default_rng(1701)
    table = rng.permutation(32).astype(np.int32).reshape(2, 16)[: len(prefixes)]
    key = np.full((32, 256, 2, 64), np.nan, np.float16)
    value = np.full_like(key, np.nan)
    for row, prefix in enumerate(prefixes):
        for position in range(prefix):
            slot = (table[row, position // 256], position % 256)
            key[slot] = rng.normal(0, 0.3, (2, 64))
            value[slot] = rng.normal(0, 0.3, (2, 64))
    angle = np.arange(4096, dtype=np.float32)[:, None] * (
        10000.0 ** (-np.arange(32, dtype=np.float32) / 32)
    )
    return {
        "query": rng.normal(0, 0.3, (sum(lengths), 1152)).astype(np.float16),
        "past_key": key,
        "past_value": value,
        "cumulative": np.cumsum([0, *lengths], dtype=np.int32),
        "past": np.array(prefixes, np.int32),
        "table": table,
        "cos": np.cos(angle).astype(np.float16),
        "sin": np.sin(angle).astype(np.float16),
    }


def reference(inputs):
    key = inputs["past_key"].copy()
    value = inputs["past_value"].copy()
    output = []
    for row, prefix in enumerate(inputs["past"]):
        start, end = inputs["cumulative"][row : row + 2]
        for token in range(start, end):
            position = int(prefix + token - start)
            qkv = inputs["query"][token].reshape(18, 64)
            cosine = inputs["cos"][position].astype(np.float32)
            sine = inputs["sin"][position].astype(np.float32)
            slot = (inputs["table"][row, position // 256], position % 256)
            key[slot] = rotate(qkv[14:16], cosine, sine)
            value[slot] = qkv[16:]
            positions = np.arange(position + 1)
            pages = inputs["table"][row, positions // 256]
            keys = key[pages, positions % 256].astype(np.float32).repeat(7, axis=1)
            values = value[pages, positions % 256].astype(np.float32).repeat(7, axis=1)
            query = rotate(qkv[:14], cosine, sine).astype(np.float32)
            scores = np.einsum("hd,lhd->hl", query, keys) * 0.125
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
            output.append(np.einsum("hl,lhd->hd", scores, values).reshape(896))
    return np.asarray(output, dtype=np.float16), key, value


def bind(session, device, inputs):
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    tensors = {}
    binding = session.io_binding()
    for name, data in inputs.items():
        tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            data.shape, data.dtype, memory_info=memory
        )
        tensor.update_inplace(data)
        binding.bind_ortvalue_input(name, tensor)
        tensors[name] = tensor
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        [len(inputs["query"]), 896], np.float16, memory_info=memory
    )
    output.update_inplace(np.full(output.shape(), np.nan, np.float16))
    binding.bind_ortvalue_output("output", output)
    binding.bind_ortvalue_output("present_key", tensors["past_key"])
    binding.bind_ortvalue_output("present_value", tensors["past_value"])
    return binding, tensors, output


@pytest.mark.parametrize(
    "prefixes,lengths", [([0], [7]), ([255, 510], [3, 5]), ([4095, 127], [1, 128])]
)
def test_paged_attention_aliases(paged_session, prefixes, lengths):
    session, device = paged_session["session"], paged_session["device"]
    inputs = make_inputs(prefixes, lengths)
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(session, device, inputs)
    session.run_with_iobinding(binding)
    actual = binding.get_outputs()
    assert actual[1].data_ptr() == tensors["past_key"].data_ptr()
    assert actual[2].data_ptr() == tensors["past_value"].data_ptr()
    np.testing.assert_allclose(output.numpy(), expected, atol=0.0005, rtol=0.005)
    np.testing.assert_allclose(
        tensors["past_key"].numpy(), key, atol=0.0005, rtol=0.005, equal_nan=True
    )
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)


@pytest.mark.parametrize(
    "invalid",
    [
        "page",
        "cumulative",
        "context",
        "write_alias",
        "cumulative_shape",
        "table_shape",
        "value_shape",
        "sine_shape",
    ],
)
def test_paged_attention_rejects_before_mutation(paged_session, invalid):
    session, device = paged_session["session"], paged_session["device"]
    inputs = make_inputs([255, 510], [3, 5])
    if invalid == "page":
        inputs["table"][1, 2] = 32
    elif invalid == "cumulative":
        inputs["cumulative"][1] = -1
    elif invalid == "context":
        inputs["past"][1] = 4095
    elif invalid == "write_alias":
        inputs["table"][1, 1] = inputs["table"][0, 0]
    elif invalid == "cumulative_shape":
        inputs["cumulative"] = inputs["cumulative"][:-1].copy()
    elif invalid == "table_shape":
        inputs["table"] = inputs["table"][:1].copy()
    elif invalid == "value_shape":
        inputs["past_value"] = inputs["past_value"][:-1].copy()
    else:
        inputs["sin"] = inputs["sin"][:-1].copy()
    binding, tensors, output = bind(session, device, inputs)
    with pytest.raises(RuntimeError, match="inference_compute .* failed with code: -1"):
        session.run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), inputs["past_key"])
    np.testing.assert_array_equal(tensors["past_value"].numpy(), inputs["past_value"])
    assert np.isnan(output.numpy()).all()
    valid = make_inputs([255, 510], [3, 5])
    binding, tensors, output = bind(session, device, valid)
    session.run_with_iobinding(binding)
    np.testing.assert_allclose(
        output.numpy(), reference(valid)[0], atol=0.0005, rtol=0.005
    )


def test_paged_attention_shared_read_only_prefix(paged_session):
    session, device = paged_session["session"], paged_session["device"]
    inputs = make_inputs([256, 256], [3, 5])
    inputs["table"][1, 0] = inputs["table"][0, 0]
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(session, device, inputs)
    session.run_with_iobinding(binding)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.0005, rtol=0.005)
    np.testing.assert_allclose(
        tensors["past_key"].numpy(), key, atol=0.0005, rtol=0.005, equal_nan=True
    )
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)
