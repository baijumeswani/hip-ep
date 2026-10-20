#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Compiled FP16/INT8 paged attention with caller-bound GPU cache aliases."""

import ctypes
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


@pytest.fixture(
    scope="module",
    params=[
        ("packed64", np.float16),
        ("separate256", np.float16),
        ("separate256", np.float32),
        ("int8", np.float32),
        ("int8_residual8", np.float32),
    ],
)
def paged_session(request, tmp_path_factory):
    yield from _paged_session(request, tmp_path_factory, request.param)


@pytest.fixture(scope="module")
def wide_paged_session(request, tmp_path_factory):
    yield from _paged_session(
        request, tmp_path_factory, ("int8_residual8", np.float32, 64, 8)
    )


def _paged_session(request, tmp_path_factory, parameters):
    layout, dtype = parameters[:2]
    separate = layout != "packed64"
    quantized = layout.startswith("int8")
    residual_bits = 8 if layout == "int8_residual8" else 0
    cache_dtype = TensorProto.INT8 if quantized else TensorProto.FLOAT16
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    heads, kv_heads, head_size = (24, 4, 256) if separate else (14, 2, 64)
    if len(parameters) == 4:
        heads, kv_heads = parameters[2:]
    output_width = heads * head_size
    cache_shape = [
        "blocks",
        "page_size",
        kv_heads,
        head_size * (2 if residual_bits else 1),
    ]
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    if os.environ.get("HIPDNN_EP_STRICT") != "1":
        pytest.fail("PagedAttention qualification requires HIPDNN_EP_STRICT=1")
    provider = request.config.getoption("--ep-name") + (
        f"_paged_{layout}_{dtype.__name__}_{heads}_{kv_heads}"
    )
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    device = devices[0]
    root = tmp_path_factory.mktemp("paged_attention")
    shapes = {
        "query": (activation, ["tokens", output_width if separate else 1152]),
        "past_key": (cache_dtype, cache_shape),
        "past_value": (cache_dtype, cache_shape),
        "cumulative": (TensorProto.INT32, ["batch_plus_one"]),
        "past": (TensorProto.INT32, ["batch"]),
        "table": (TensorProto.INT32, ["batch", "columns"]),
        "cos": (TensorProto.FLOAT16, ["rotary_length", 32]),
        "sin": (TensorProto.FLOAT16, ["rotary_length", 32]),
    }
    if separate:
        del shapes["cos"], shapes["sin"]
        shapes["key"] = (activation, ["key_tokens", kv_heads * head_size])
        shapes["value"] = (activation, ["value_tokens", kv_heads * head_size])
    attributes = {}
    if quantized:
        shapes["key_scale"] = (TensorProto.FLOAT, [kv_heads, 1, head_size])
        shapes["value_scale"] = (TensorProto.FLOAT, [kv_heads, 1, head_size])
        attributes = {"k_quant_type": "PER_CHANNEL", "v_quant_type": "PER_CHANNEL"}
        if residual_bits:
            attributes["kv_cache_residual_bits"] = residual_bits
    node = helper.make_node(
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
            "" if separate else "cos",
            "" if separate else "sin",
            *(["", "", "", "", "key_scale", "value_scale"] if quantized else []),
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=heads,
        kv_num_heads=kv_heads,
        do_rotary=0 if separate else 1,
        **attributes,
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
                "output", activation, ["tokens", output_width]
            ),
            helper.make_tensor_value_info("present_key", cache_dtype, cache_shape),
            helper.make_tensor_value_info("present_value", cache_dtype, cache_shape),
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
    context = {
        "session": session,
        "device": device,
        "separate": separate,
        "dtype": dtype,
        "quantized": quantized,
        "residual_bits": residual_bits,
    }
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


def make_inputs(
    prefixes,
    lengths,
    separate=False,
    dtype=np.float16,
    quantized=False,
    residual_bits=0,
    head_geometry=None,
):
    heads, kv_heads, head_size = head_geometry or (
        (24, 4, 256) if separate else (14, 2, 64)
    )
    rng = np.random.default_rng(1701)
    columns = max(16, (max(p + q for p, q in zip(prefixes, lengths)) + 255) // 256)
    capacity = max(2, len(prefixes))
    blocks = capacity * columns
    table = (
        rng.permutation(blocks)
        .astype(np.int32)
        .reshape(capacity, columns)[: len(prefixes)]
    )
    cache_dtype = np.int8 if quantized else np.float16
    cache_width = head_size * (2 if residual_bits else 1)
    key = np.full(
        (blocks, 256, kv_heads, cache_width), -128 if quantized else np.nan, cache_dtype
    )
    value = np.full_like(key, -128 if quantized else np.nan)
    for row, prefix in enumerate(prefixes):
        for position in range(prefix):
            slot = (table[row, position // 256], position % 256)
            if quantized:
                key[slot] = rng.integers(
                    -64, 64, (kv_heads, cache_width), dtype=np.int8
                )
                value[slot] = rng.integers(
                    -64, 64, (kv_heads, cache_width), dtype=np.int8
                )
            else:
                key[slot] = rng.normal(0, 0.3, (kv_heads, head_size))
                value[slot] = rng.normal(0, 0.3, (kv_heads, head_size))
    angle = np.arange(4096, dtype=np.float32)[:, None] * (
        10000.0 ** (-np.arange(32, dtype=np.float32) / 32)
    )
    inputs = {
        "query": rng.normal(
            0,
            0.3,
            (sum(lengths), (heads if separate else heads + 2 * kv_heads) * head_size),
        ).astype(dtype),
        "past_key": key,
        "past_value": value,
        "cumulative": np.cumsum([0, *lengths], dtype=np.int32),
        "past": np.array(prefixes, np.int32),
        "table": table,
        "cos": np.cos(angle).astype(np.float16),
        "sin": np.sin(angle).astype(np.float16),
    }
    if separate:
        del inputs["cos"], inputs["sin"]
        for name in ("key", "value"):
            inputs[name] = rng.normal(
                0, 0.3, (sum(lengths), kv_heads * head_size)
            ).astype(dtype)
    if quantized:
        for name in ("key_scale", "value_scale"):
            inputs[name] = rng.uniform(0.003, 0.03, (kv_heads, 1, head_size)).astype(
                np.float32
            )
    return inputs


def reference(inputs):
    separate = "key" in inputs
    quantized = "key_scale" in inputs
    heads, kv_heads, head_size = (24, 4, 256) if separate else (14, 2, 64)
    if separate:
        heads = inputs["query"].shape[1] // head_size
        kv_heads = inputs["past_key"].shape[2]
    residual = quantized and inputs["past_key"].shape[-1] == 2 * head_size
    key = inputs["past_key"].copy()
    value = inputs["past_value"].copy()
    output = []
    for row, prefix in enumerate(inputs["past"]):
        start, end = inputs["cumulative"][row : row + 2]
        for token in range(start, end):
            position = int(prefix + token - start)
            slot = (inputs["table"][row, position // 256], position % 256)
            if separate:
                for cache, kind in ((key, "key"), (value, "value")):
                    values = inputs[kind][token].reshape(kv_heads, head_size)
                    if quantized:
                        scale = inputs[f"{kind}_scale"].reshape(kv_heads, head_size)
                        if residual:
                            codes = np.rint(
                                np.clip(values / (scale / 256), -32768, 32767)
                            ).astype(np.int32)
                            values = np.concatenate(
                                [codes // 256, codes % 256 - 128], axis=-1
                            ).astype(np.int8)
                        else:
                            values = np.rint(np.clip(values / scale, -128, 127)).astype(
                                np.int8
                            )
                    cache[slot] = values
                query = (
                    inputs["query"][token].reshape(heads, head_size).astype(np.float32)
                )
            else:
                qkv = inputs["query"][token].reshape(18, 64)
                cosine = inputs["cos"][position].astype(np.float32)
                sine = inputs["sin"][position].astype(np.float32)
                key[slot] = rotate(qkv[14:16], cosine, sine)
                value[slot] = qkv[16:]
                query = rotate(qkv[:14], cosine, sine).astype(np.float32)
            positions = np.arange(position + 1)
            pages = inputs["table"][row, positions // 256]
            if quantized:
                keys = key[pages, positions % 256].astype(np.float32)
                values = value[pages, positions % 256].astype(np.float32)
                if residual:
                    keys = 256 * keys[..., :head_size] + keys[..., head_size:] + 128
                    values = (
                        256 * values[..., :head_size] + values[..., head_size:] + 128
                    )
                keys *= inputs["key_scale"].reshape(kv_heads, head_size) / (
                    256 if residual else 1
                )
                values *= inputs["value_scale"].reshape(kv_heads, head_size) / (
                    256 if residual else 1
                )
                scores = np.einsum(
                    "hgd,lhd->hgl",
                    query.reshape(kv_heads, heads // kv_heads, head_size),
                    keys,
                ) / np.sqrt(head_size)
                scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
                scores /= scores.sum(axis=-1, keepdims=True)
                output.append(
                    np.einsum("hgl,lhd->hgd", scores, values).reshape(heads * head_size)
                )
                continue
            keys = (
                key[pages, positions % 256]
                .astype(np.float32)
                .repeat(heads // kv_heads, axis=1)
            )
            values = (
                value[pages, positions % 256]
                .astype(np.float32)
                .repeat(heads // kv_heads, axis=1)
            )
            scores = np.einsum("hd,lhd->hl", query, keys) / np.sqrt(head_size)
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
            output.append(
                np.einsum("hl,lhd->hd", scores, values).reshape(heads * head_size)
            )
    return np.asarray(output, dtype=inputs["query"].dtype), key, value


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
        [len(inputs["query"]), inputs["query"].shape[1] if "key" in inputs else 896],
        inputs["query"].dtype,
        memory_info=memory,
    )
    output.update_inplace(np.full(output.shape(), np.nan, inputs["query"].dtype))
    binding.bind_ortvalue_output("output", output)
    binding.bind_ortvalue_output("present_key", tensors["past_key"])
    binding.bind_ortvalue_output("present_value", tensors["past_value"])
    return binding, tensors, output


@pytest.mark.parametrize("elements", [1024, 5 * 1024 * 1024])
@pytest.mark.parametrize(
    "memory_type,expected",
    [
        (ort.OrtDeviceMemoryType.DEFAULT, 2),
        (ort.OrtDeviceMemoryType.HOST_ACCESSIBLE, 1),
    ],
)
def test_allocator_memory_placement(paged_session, memory_type, expected, elements):
    inputs = make_inputs(
        [0],
        [1],
        paged_session["separate"],
        paged_session["dtype"],
        paged_session["quantized"],
        paged_session["residual_bits"],
    )
    binding, _, _ = bind(paged_session["session"], paged_session["device"], inputs)
    paged_session["session"].run_with_iobinding(binding)
    hip = ctypes.CDLL(
        os.environ.get(
            "HIP_RUNTIME_LIBRARY",
            "amdhip64.dll" if os.name == "nt" else "libamdhip64.so",
        )
    )
    hip.hipPointerGetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    memory = paged_session["device"].memory_info(memory_type)
    for iteration in range(2):
        tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            [elements], np.float32, memory_info=memory
        )
        actual = ctypes.c_int()
        # HIP_POINTER_ATTRIBUTE_MEMORY_TYPE distinguishes device from mapped host.
        assert (
            hip.hipPointerGetAttribute(
                ctypes.byref(actual), 2, ctypes.c_void_p(tensor.data_ptr())
            )
            == 0
        )
        assert actual.value == expected
        data = np.full(elements, iteration + 0.25, dtype=np.float32)
        if memory_type == ort.OrtDeviceMemoryType.HOST_ACCESSIBLE:
            host = np.ctypeslib.as_array(
                (ctypes.c_float * elements).from_address(tensor.data_ptr())
            )
            host[:] = data
            np.testing.assert_array_equal(host, data)
            del host
        else:
            tensor.update_inplace(data)
            np.testing.assert_array_equal(tensor.numpy(), data)
        del tensor
        gc.collect()


@pytest.mark.parametrize(
    "prefixes,lengths", [([0], [7]), ([255, 510], [3, 5]), ([4095, 127], [1, 128])]
)
def test_paged_attention_aliases(paged_session, prefixes, lengths):
    session, device = paged_session["session"], paged_session["device"]
    inputs = make_inputs(
        prefixes,
        lengths,
        paged_session["separate"],
        paged_session["dtype"],
        paged_session["quantized"],
        paged_session["residual_bits"],
    )
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(session, device, inputs)
    session.run_with_iobinding(binding)
    actual = binding.get_outputs()
    assert actual[1].data_ptr() == tensors["past_key"].data_ptr()
    assert actual[2].data_ptr() == tensors["past_value"].data_ptr()
    tolerance = 0.00002 if paged_session["dtype"] == np.float32 else 0.005
    np.testing.assert_allclose(
        output.numpy(), expected, atol=tolerance / 10, rtol=tolerance
    )
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
    inputs = make_inputs(
        [255, 510],
        [3, 5],
        paged_session["separate"],
        paged_session["dtype"],
        paged_session["quantized"],
        paged_session["residual_bits"],
    )
    if invalid == "page":
        inputs["table"][1, 2] = len(inputs["past_key"])
    elif invalid == "cumulative":
        inputs["cumulative"][1] = -1
    elif invalid == "context":
        inputs["past"][1] = 262144 if paged_session["quantized"] else 4095
    elif invalid == "write_alias":
        inputs["table"][1, 1] = inputs["table"][0, 0]
    elif invalid == "cumulative_shape":
        inputs["cumulative"] = inputs["cumulative"][:-1].copy()
    elif invalid == "table_shape":
        inputs["table"] = inputs["table"][:1].copy()
    elif invalid == "value_shape":
        inputs["past_value"] = inputs["past_value"][:-1].copy()
    elif not paged_session["separate"]:
        inputs["sin"] = inputs["sin"][:-1].copy()
    else:
        inputs["key"] = inputs["key"][:-1].copy()
    binding, tensors, output = bind(session, device, inputs)
    with pytest.raises(RuntimeError, match="inference_compute .* failed with code: -1"):
        session.run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), inputs["past_key"])
    np.testing.assert_array_equal(tensors["past_value"].numpy(), inputs["past_value"])
    assert np.isnan(output.numpy()).all()
    valid = make_inputs(
        [255, 510],
        [3, 5],
        paged_session["separate"],
        paged_session["dtype"],
        paged_session["quantized"],
        paged_session["residual_bits"],
    )
    binding, tensors, output = bind(session, device, valid)
    session.run_with_iobinding(binding)
    np.testing.assert_allclose(
        output.numpy(), reference(valid)[0], atol=0.0005, rtol=0.005
    )


def test_paged_attention_shared_read_only_prefix(paged_session):
    session, device = paged_session["session"], paged_session["device"]
    inputs = make_inputs(
        [256, 256],
        [3, 5],
        paged_session["separate"],
        paged_session["dtype"],
        paged_session["quantized"],
        paged_session["residual_bits"],
    )
    inputs["table"][1, 0] = inputs["table"][0, 0]
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(session, device, inputs)
    session.run_with_iobinding(binding)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.0005, rtol=0.005)
    np.testing.assert_allclose(
        tensors["past_key"].numpy(), key, atol=0.0005, rtol=0.005, equal_nan=True
    )
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)


@pytest.mark.parametrize(
    "prefix,length",
    [(0, 257), (4095, 65), (8191, 2), (16384, 1), (32767, 1), (131071, 1), (262143, 1)],
)
def test_int8_attention_tiled_and_long_context(paged_session, prefix, length):
    if not paged_session["quantized"]:
        pytest.skip("INT8 head-256 path only")
    inputs = make_inputs(
        [prefix], [length], True, np.float32, True, paged_session["residual_bits"]
    )
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.000002, rtol=0.00002)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), key)
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)


@pytest.mark.parametrize("invalid", ["duplicate", "out_of_range", "write_alias"])
def test_int8_long_page_validation_is_atomic(paged_session, invalid):
    if not paged_session["quantized"]:
        pytest.skip("INT8 head-256 path only")
    inputs = make_inputs(
        [32768, 0], [1, 1], True, np.float32, True, paged_session["residual_bits"]
    )
    if invalid == "duplicate":
        inputs["table"][0, 128] = inputs["table"][0, 0]
    elif invalid == "out_of_range":
        inputs["table"][0, 128] = inputs["past_key"].shape[0]
    else:
        inputs["table"][1, 0] = inputs["table"][0, 128]
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    with pytest.raises(RuntimeError, match="inference_compute .* failed with code: -1"):
        paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), inputs["past_key"])
    np.testing.assert_array_equal(tensors["past_value"].numpy(), inputs["past_value"])
    assert np.isnan(output.numpy()).all()


def test_int8_decode_tiled_ragged_requests(paged_session):
    if not paged_session["quantized"]:
        pytest.skip("INT8 head-256 path only")
    inputs = make_inputs(
        [16384, 0, 4097],
        [1, 1, 1],
        True,
        np.float32,
        True,
        paged_session["residual_bits"],
    )
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.000002, rtol=0.00002)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), key)
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)


def test_int8_attention_workspace_tile_downshift(wide_paged_session):
    """Exercise both workspace downshift loops, then revisit prefill."""
    for prefix, length in [(0, 257), (16384, 1), (0, 257)]:
        inputs = make_inputs(
            [prefix], [length], True, np.float32, True, 8, (64, 8, 256)
        )
        expected, key, value = reference(inputs)
        binding, tensors, output = bind(
            wide_paged_session["session"], wide_paged_session["device"], inputs
        )
        wide_paged_session["session"].run_with_iobinding(binding)
        np.testing.assert_allclose(
            output.numpy(), expected, atol=0.000002, rtol=0.00002
        )
        np.testing.assert_array_equal(tensors["past_key"].numpy(), key)
        np.testing.assert_array_equal(tensors["past_value"].numpy(), value)


@pytest.mark.parametrize(
    "name,bad_scale",
    [
        ("key_scale", 0),
        ("key_scale", np.inf),
        ("value_scale", -1),
        ("value_scale", np.nan),
        ("key_scale", np.nextafter(np.float32(0), np.float32(1))),
    ],
)
def test_int8_attention_invalid_scales_are_atomic(paged_session, name, bad_scale):
    if not paged_session["quantized"]:
        pytest.skip("INT8 head-256 path only")
    if 0 < bad_scale < np.finfo(np.float32).tiny and not paged_session["residual_bits"]:
        pytest.skip("Residual scale underflow only")
    inputs = make_inputs(
        [255], [3], True, np.float32, True, paged_session["residual_bits"]
    )
    inputs[name][2, 0, 17] = bad_scale
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    with pytest.raises(RuntimeError, match="inference_compute .* failed with code: -1"):
        paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), inputs["past_key"])
    np.testing.assert_array_equal(tensors["past_value"].numpy(), inputs["past_value"])
    assert np.isnan(output.numpy()).all()


def test_int8_attention_rounds_ties_and_saturates_signed_cache(paged_session):
    if not paged_session["quantized"]:
        pytest.skip("INT8 head-256 path only")
    residual = paged_session["residual_bits"]
    inputs = make_inputs([0], [3], True, np.float32, True, residual)
    pattern = np.array(
        [-1000, -128.5, -127.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, 127.5, 1000],
        np.float32,
    )
    if residual:
        pattern = np.array(
            [
                -1e6,
                -32768.5,
                -32767.5,
                -256.5,
                -255.5,
                -128.5,
                -2.5,
                -1.5,
                -0.5,
                0,
                0.5,
                1.5,
                2.5,
                127.5,
                255.5,
                256.5,
                32766.5,
                32767.5,
                1e6,
            ],
            np.float32,
        )
    for name in ("key", "value"):
        scales = np.resize(np.array([0.25, 0.5, 1, 2], np.float32), (4, 1, 256))
        if residual:
            scales *= 256
        inputs[f"{name}_scale"] = scales
        inputs[name] = np.resize(pattern, inputs[name].shape) * (
            scales.reshape(1, -1) / (256 if residual else 1)
        )
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), key)
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.00002, rtol=0.00002)


def test_residual_cache_covers_every_signed_code(paged_session):
    if not paged_session["residual_bits"]:
        pytest.skip("Two-byte residual format only")
    inputs = make_inputs([0], [64], True, np.float32, True, 8)
    inputs["query"].fill(0)
    codes = np.arange(-32768, 32768, dtype=np.float32).reshape(64, 1024)
    inputs["key"] = codes
    inputs["value"] = codes[:, ::-1].copy()
    inputs["key_scale"].fill(256)
    inputs["value_scale"].fill(256)
    expected, key, value = reference(inputs)
    binding, tensors, output = bind(
        paged_session["session"], paged_session["device"], inputs
    )
    paged_session["session"].run_with_iobinding(binding)
    np.testing.assert_array_equal(tensors["past_key"].numpy(), key)
    np.testing.assert_array_equal(tensors["past_value"].numpy(), value)
    np.testing.assert_allclose(output.numpy(), expected, atol=0.01, rtol=0.00002)
