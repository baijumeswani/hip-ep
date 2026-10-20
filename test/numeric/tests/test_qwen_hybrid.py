#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Compiled Qwen hybrid contracts with independent FP32 recurrence references."""

import gc
import json
import os
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from framework.ort_ep_backend import _parse_ep_options


@pytest.fixture(scope="module")
def hybrid_sessions(request, tmp_path_factory):
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    if os.environ.get("HIPDNN_EP_STRICT") != "1":
        pytest.fail("Hybrid qualification requires HIPDNN_EP_STRICT=1")
    provider = request.config.getoption("--ep-name") + "_hybrid"
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == provider]
    assert len(devices) == 1
    root = tmp_path_factory.mktemp("qwen_hybrid")
    sessions = {}

    def create(kind, shapes, outputs, **attrs):
        key = json.dumps([kind, shapes, outputs, attrs], sort_keys=True)
        if key not in sessions:
            infos = [
                helper.make_tensor_value_info(name, dtype, shape)
                for name, (dtype, shape) in shapes.items()
            ]
            result_infos = [
                helper.make_tensor_value_info(name, *spec)
                for name, spec in outputs.items()
            ]
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node(
                            kind,
                            list(shapes),
                            list(outputs),
                            domain="com.microsoft",
                            **attrs,
                        )
                    ],
                    kind,
                    infos,
                    result_infos,
                ),
                opset_imports=[
                    helper.make_opsetid("", 21),
                    helper.make_opsetid("com.microsoft", 1),
                ],
            )
            model.ir_version = 10
            onnx.checker.check_model(model)
            path = root / f"{kind}-{len(sessions)}.onnx"
            onnx.save(model, path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.add_session_config_entry("session.use_env_allocators", "1")
            options.add_provider_for_devices(
                devices, _parse_ep_options(request.config.getoption("--ep-option"))
            )
            options.enable_profiling = True
            options.profile_file_prefix = str(root / f"{kind}-{len(sessions)}")
            sessions[key] = ort.InferenceSession(str(path), sess_options=options)
        return sessions[key], devices[0]

    yield create
    for session in sessions.values():
        events = json.loads(Path(session.end_profiling()).read_text())
        assigned = {
            event["args"]["provider"]
            for event in events
            if "provider" in event.get("args", {})
        }
        assert assigned == {provider}
    if sessions:
        del session
    sessions.clear()
    devices.clear()
    gc.collect()
    if request.session.testsfailed:
        warnings.warn(
            "Retaining EP registration: failed-test tracebacks may still own native sessions.",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        ort.unregister_execution_provider_library(provider)


def run_bound(session, device, inputs, expected, alias=False, invalid=False):
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    binding = session.io_binding()
    tensors = {}
    for name, data in inputs.items():
        tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            data.shape, data.dtype, memory_info=memory
        )
        tensor.update_inplace(data)
        tensors[name] = tensor
        binding.bind_ortvalue_input(name, tensor)
    outputs = {}
    for name, data in expected.items():
        if name == "present" and alias:
            tensor = tensors["past"]
        else:
            tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                data.shape, data.dtype, memory_info=memory
            )
            tensor.update_inplace(np.full_like(data, np.nan))
        outputs[name] = tensor
        binding.bind_ortvalue_output(name, tensor)
    if invalid:
        with pytest.raises(RuntimeError, match="inference_compute .* failed"):
            session.run_with_iobinding(binding)
        for name, tensor in tensors.items():
            np.testing.assert_array_equal(tensor.numpy(), inputs[name])
        assert np.isnan(outputs["output"].numpy()).all()
    else:
        session.run_with_iobinding(binding)
        for name, data in expected.items():
            # FP16 outputs and FP32 recurrence are checked independently.
            tolerance = 0.002 if data.dtype == np.float16 else 0.00002
            np.testing.assert_allclose(
                outputs[name].numpy(), data, atol=tolerance, rtol=tolerance
            )
        if not alias and "past" in inputs:
            np.testing.assert_array_equal(tensors["past"].numpy(), inputs["past"])
    return {name: tensor.numpy() for name, tensor in outputs.items()}


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_varlen_convolution_carry(hybrid_sessions, alias, dtype):
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    i32 = TensorProto.INT32
    shapes = {
        "input": (activation, ["tokens", 10240]),
        "weight": (activation, [10240, 1, 4]),
        "cu": (i32, ["batch_plus_one"]),
        "bias": (activation, [10240]),
        "past": (activation, ["batch", 10240, 3]),
    }
    output_shapes = {
        "output": shapes["input"],
        "present": shapes["past"],
    }
    session, device = hybrid_sessions(
        "VarlenCausalConvWithState", shapes, output_shapes, activation="silu"
    )
    rng = np.random.default_rng(3001)
    inputs = {
        "input": rng.normal(0, 0.3, (8, 10240)).astype(dtype),
        "weight": rng.normal(0, 0.3, (10240, 1, 4)).astype(dtype),
        "cu": np.array([0, 3, 3, 8], np.int32),
        "bias": rng.normal(0, 0.1, 10240).astype(dtype),
        "past": rng.normal(0, 0.3, (3, 10240, 3)).astype(dtype),
    }
    present = inputs["past"].copy()
    output = np.empty_like(inputs["input"])
    for b, (start, end) in enumerate(zip(inputs["cu"][:-1], inputs["cu"][1:])):
        for t in range(start, end):
            window = np.concatenate([present[b], inputs["input"][t, :, None]], axis=1)
            value = (
                window.astype(np.float32) * inputs["weight"][:, 0].astype(np.float32)
            ).sum(axis=1) + inputs["bias"].astype(np.float32)
            output[t] = value / (1 + np.exp(-value))
            present[b] = window[:, 1:]
    expected = {"output": output, "present": present}
    run_bound(session, device, inputs, expected, alias)
    for malformed in (
        np.array([0, -1, 3, 8], np.int32),
        np.array([0, 3, 8], np.int32),
    ):
        run_bound(session, device, dict(inputs, cu=malformed), expected, alias, True)
    run_bound(session, device, inputs, expected, alias)


def delta_reference(inputs):
    state = inputs["past"].copy()
    output = np.empty_like(inputs["value"])
    q = inputs["query"].astype(np.float32).repeat(3, axis=1)
    k = inputs["key"].astype(np.float32).repeat(3, axis=1)
    q /= np.sqrt((q * q).sum(axis=-1, keepdims=True) + 1e-12)
    k /= np.sqrt((k * k).sum(axis=-1, keepdims=True) + 1e-12)
    for b, (start, end) in enumerate(zip(inputs["cu"][:-1], inputs["cu"][1:])):
        for t in range(start, end):
            z = inputs["decay"][t] + inputs["dt_bias"]
            gate = np.exp(-np.exp(inputs["a_log"]) * np.logaddexp(0, z))
            state[b] *= gate[:, None, None]
            retrieved = np.einsum("hvk,hk->hv", state[b], k[t])
            delta = (inputs["value"][t].astype(np.float32) - retrieved) / (
                1 + np.exp(-inputs["beta"][t, :, None])
            )
            state[b] += delta[:, :, None] * k[t, :, None, :]
            output[t] = np.einsum("hvk,hk->hv", state[b], q[t]) / np.sqrt(128)
    return {"output": output, "present": state}


@pytest.mark.parametrize("alias,lengths", [(False, [3, 0, 5]), (True, [65, 1])])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gated_delta_recurrence(hybrid_sessions, alias, lengths, dtype):
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    f32, i32 = TensorProto.FLOAT, TensorProto.INT32
    shapes = {
        "query": (activation, ["tokens", 16, 128]),
        "key": (activation, ["key_tokens", 16, 128]),
        "value": (activation, ["value_tokens", 48, 128]),
        "cu": (i32, ["batch_plus_one"]),
        "decay": (f32, ["decay_tokens", 48]),
        "beta": (f32, ["beta_tokens", 48]),
        "past": (f32, ["batch", 48, 128, 128]),
        "a_log": (f32, [48]),
        "dt_bias": (f32, [48]),
    }
    session, device = hybrid_sessions(
        "GatedDeltaNet",
        shapes,
        {"output": (activation, ["tokens", 48, 128]), "present": shapes["past"]},
        update_rule="gated_delta",
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        chunk_size=64,
        scale=0.0,
    )
    rng = np.random.default_rng(3002)
    tokens, batch = sum(lengths), len(lengths)
    inputs = {
        "query": rng.normal(0, 0.3, (tokens, 16, 128)).astype(dtype),
        "key": rng.normal(0, 0.3, (tokens, 16, 128)).astype(dtype),
        "value": rng.normal(0, 0.3, (tokens, 48, 128)).astype(dtype),
        "cu": np.cumsum([0, *lengths], dtype=np.int32),
        "decay": rng.normal(-1, 1, (tokens, 48)).astype(np.float32),
        "beta": rng.normal(0, 1, (tokens, 48)).astype(np.float32),
        "past": rng.normal(0, 0.03, (batch, 48, 128, 128)).astype(np.float32),
        "a_log": rng.normal(-2, 0.5, 48).astype(np.float32),
        "dt_bias": rng.normal(0, 0.5, 48).astype(np.float32),
    }
    expected = delta_reference(inputs)
    run_bound(session, device, inputs, expected, alias)
    for name in ("cu", "key", "value", "decay", "beta"):
        bad = dict(inputs)
        bad[name] = bad[name][:-1].copy()
        bad_expected = dict(expected)
        if name == "value":
            bad_expected["output"] = expected["output"][:-1]
        run_bound(session, device, bad, bad_expected, alias, True)
    run_bound(session, device, inputs, expected, alias)


def run_capture(session, device, inputs, output_shapes, *, alias=False, invalid=False):
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    binding = session.io_binding()
    tensors, outputs = {}, {}
    try:
        for name, data in inputs.items():
            if name == "active":
                tensor = ort.OrtValue.ortvalue_from_numpy(data)
            else:
                tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                    data.shape, data.dtype, memory_info=memory
                )
                tensor.update_inplace(data)
            tensors[name] = tensor
            binding.bind_ortvalue_input(name, tensor)
        for name, data in output_shapes.items():
            if name == "present" and alias:
                tensor = tensors["past"]
            else:
                tensor = ort.OrtValue.ortvalue_from_shape_and_type(
                    data.shape, data.dtype, memory_info=memory
                )
                tensor.update_inplace(np.full_like(data, np.nan))
            outputs[name] = tensor
            binding.bind_ortvalue_output(name, tensor)
        if invalid:
            with pytest.raises(RuntimeError, match="inference_compute .* failed"):
                session.run_with_iobinding(binding)
            for name, tensor in tensors.items():
                np.testing.assert_array_equal(tensor.numpy(), inputs[name])
            assert np.isnan(outputs["output"].numpy()).all()
            assert np.isnan(outputs["updates"].numpy()).all()
            return None
        session.run_with_iobinding(binding)
        return {name: value.numpy().copy() for name, value in outputs.items()}
    finally:
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        tensors.clear()
        outputs.clear()
        del binding, session


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("capacity", [1, 7])
def test_convolution_compact_prefixes(hybrid_sessions, dtype, alias, capacity):
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    channels = 10240
    shapes = {
        "input": (activation, ["tokens", channels]),
        "weight": (activation, [channels, 1, 4]),
        "cu": (TensorProto.INT32, ["batch_plus_one"]),
        "bias": (activation, [channels]),
        "past": (activation, ["batch", channels, 3]),
        "counts": (TensorProto.INT32, ["capture_batch"]),
    }
    results = {
        "output": shapes["input"],
        "present": shapes["past"],
        "updates": (activation, ["update_batch", capacity, channels]),
    }
    session, device = hybrid_sessions(
        "VarlenCausalConvWithState",
        shapes,
        results,
        activation="silu",
        state_update_capacity=capacity,
    )
    base_session, _ = hybrid_sessions(
        "VarlenCausalConvWithState",
        {name: spec for name, spec in shapes.items() if name != "counts"},
        {name: spec for name, spec in results.items() if name != "updates"},
        activation="silu",
    )
    rng = np.random.default_rng(5011)
    for lengths in ([8, 0, 11], [2, 8], [8, 0, 11]):
        batch, tokens = len(lengths), sum(lengths)
        inputs = {
            "input": rng.normal(0, 0.3, (tokens, channels)).astype(dtype),
            "weight": rng.normal(0, 0.2, (channels, 1, 4)).astype(dtype),
            "cu": np.cumsum([0, *lengths], dtype=np.int32),
            "bias": rng.normal(0, 0.1, channels).astype(dtype),
            "past": rng.normal(0, 0.3, (batch, channels, 3)).astype(dtype),
            "counts": np.full(batch, capacity, np.int32),
        }
        outputs = {
            "output": np.empty_like(inputs["input"]),
            "present": np.empty_like(inputs["past"]),
            "updates": np.empty((batch, capacity, channels), dtype),
        }
        captured = run_capture(session, device, inputs, outputs, alias=alias)
        baseline = run_capture(
            base_session,
            device,
            {name: data for name, data in inputs.items() if name != "counts"},
            {name: data for name, data in outputs.items() if name != "updates"},
        )
        for name in baseline:
            np.testing.assert_array_equal(captured[name], baseline[name])
        for b, (start, end) in enumerate(zip(inputs["cu"][:-1], inputs["cu"][1:])):
            count = min(capacity, end - start)
            np.testing.assert_array_equal(
                captured["updates"][b, :count], inputs["input"][start : start + count]
            )
            np.testing.assert_array_equal(captured["updates"][b, count:], 0)
            for kept in range(count + 1):
                replayed = np.concatenate(
                    [inputs["past"][b], captured["updates"][b, :kept].T], axis=1
                )[:, -3:]
                expected = np.concatenate(
                    [inputs["past"][b], inputs["input"][start : start + kept].T], axis=1
                )[:, -3:]
                np.testing.assert_array_equal(replayed, expected)
        zero = run_capture(
            session, device, dict(inputs, counts=np.full(batch, -3, np.int32)), outputs
        )
        np.testing.assert_array_equal(zero["updates"], 0)
        np.testing.assert_array_equal(zero["present"], captured["present"])
        mixed_counts = np.arange(batch, dtype=np.int32) * 2
        mixed = run_capture(session, device, dict(inputs, counts=mixed_counts), outputs)
        high = run_capture(
            session,
            device,
            dict(inputs, counts=np.full(batch, capacity + 9, np.int32)),
            outputs,
        )
        np.testing.assert_array_equal(high["updates"], captured["updates"])
        for b, length in enumerate(lengths):
            count = min(capacity, mixed_counts[b], length)
            np.testing.assert_array_equal(
                mixed["updates"][b, :count], captured["updates"][b, :count]
            )
            np.testing.assert_array_equal(mixed["updates"][b, count:], 0)
        for bad in (
            dict(inputs, counts=inputs["counts"][:-1]),
            dict(inputs, cu=np.array([0, -1, tokens], np.int32)),
        ):
            run_capture(session, device, bad, outputs, alias=alias, invalid=True)
        run_capture(session, device, inputs, outputs, alias=alias)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("capacity", [1, 7])
def test_gated_delta_compact_prefixes(hybrid_sessions, dtype, alias, capacity):
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    f32, i32 = TensorProto.FLOAT, TensorProto.INT32
    heads, key_heads, width = 48, 16, 128
    capsule_width = capacity * (heads + key_heads * width + heads * width)
    shapes = {
        "query": (activation, ["tokens", key_heads, width]),
        "key": (activation, ["key_tokens", key_heads, width]),
        "value": (activation, ["value_tokens", heads, width]),
        "cu": (i32, ["batch_plus_one"]),
        "decay": (f32, ["decay_tokens", heads]),
        "beta": (f32, ["beta_tokens", heads]),
        "past": (f32, ["batch", heads, width, width]),
        "a_log": (f32, [heads]),
        "dt_bias": (f32, [heads]),
    }
    base_results = {"output": shapes["value"], "present": shapes["past"]}
    attrs = dict(
        update_rule="gated_delta",
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        chunk_size=64,
        scale=0.0,
    )
    base_session, device = hybrid_sessions(
        "GatedDeltaNet", shapes, base_results, **attrs
    )
    session, _ = hybrid_sessions(
        "GatedDeltaNet",
        dict(shapes, counts=(i32, ["capture_batch"]), active=(i32, ["active_count"])),
        dict(base_results, updates=(f32, ["update_batch", capsule_width])),
        **attrs,
        state_update_capacity=capacity,
    )
    rng = np.random.default_rng(5012)
    lengths = [8, 0, 11]
    batch, tokens = len(lengths), sum(lengths)
    inputs = {
        "query": rng.normal(0, 0.3, (tokens, key_heads, width)).astype(dtype),
        "key": rng.normal(0, 0.3, (tokens, key_heads, width)).astype(dtype),
        "value": rng.normal(0, 0.3, (tokens, heads, width)).astype(dtype),
        "cu": np.cumsum([0, *lengths], dtype=np.int32),
        "decay": rng.normal(-1, 1, (tokens, heads)).astype(np.float32),
        "beta": rng.normal(0, 1, (tokens, heads)).astype(np.float32),
        "past": rng.normal(0, 0.03, (batch, heads, width, width)).astype(np.float32),
        "a_log": rng.normal(-2, 0.5, heads).astype(np.float32),
        "dt_bias": rng.normal(0, 0.5, heads).astype(np.float32),
    }
    expected = delta_reference(inputs)
    baseline = run_bound(base_session, device, inputs, expected)
    controlled = dict(
        inputs,
        counts=np.full(batch, capacity, np.int32),
        active=np.array([1], np.int32),
    )
    outputs = dict(expected, updates=np.empty((batch, capsule_width), np.float32))
    captured = run_capture(session, device, controlled, outputs, alias=alias)
    for name in baseline:
        np.testing.assert_array_equal(captured[name], baseline[name])
    decay_end = capacity * heads
    key_end = decay_end + capacity * key_heads * width
    decays = captured["updates"][:, :decay_end].reshape(batch, capacity, heads)
    keys = captured["updates"][:, decay_end:key_end].reshape(
        batch, capacity, key_heads, width
    )
    deltas = captured["updates"][:, key_end:].reshape(batch, capacity, heads, width)
    for kept in range(capacity + 1):
        replayed = inputs["past"].copy()
        prefix_indices, cumulative = [], [0]
        for b, (start, end) in enumerate(zip(inputs["cu"][:-1], inputs["cu"][1:])):
            count = min(kept, end - start)
            prefix_indices.extend(range(start, start + count))
            cumulative.append(cumulative[-1] + count)
            for t in range(count):
                decayed = replayed[b] * decays[b, t, :, None, None]
                key = keys[b, t].repeat(heads // key_heads, axis=0)[:, None, :]
                replayed[b] = (
                    key.astype(np.float64) * deltas[b, t, :, :, None].astype(np.float64)
                    + decayed.astype(np.float64)
                ).astype(np.float32)
        if not kept:
            np.testing.assert_array_equal(replayed, inputs["past"])
            continue
        prefix = dict(inputs, cu=np.asarray(cumulative, np.int32))
        for name in ("query", "key", "value", "decay", "beta"):
            prefix[name] = inputs[name][prefix_indices]
        reference = delta_reference(prefix)
        actual = run_bound(base_session, device, prefix, reference)
        np.testing.assert_array_equal(replayed, actual["present"])
    inactive = run_capture(
        session, device, dict(controlled, active=np.array([0], np.int32)), outputs
    )
    np.testing.assert_array_equal(inactive["updates"], 0)
    for name in baseline:
        np.testing.assert_array_equal(inactive[name], captured[name])
    high = run_capture(
        session,
        device,
        dict(
            controlled,
            counts=np.full(batch, capacity + 9, np.int32),
            active=np.array([-1], np.int32),
        ),
        outputs,
    )
    np.testing.assert_array_equal(high["updates"], captured["updates"])
    mixed_counts = np.arange(batch, dtype=np.int32) * 2
    mixed = run_capture(session, device, dict(controlled, counts=mixed_counts), outputs)
    for b, length in enumerate(lengths):
        count = min(capacity, mixed_counts[b], length)
        for section_begin, section_end, dimensions in (
            (0, decay_end, (capacity, heads)),
            (decay_end, key_end, (capacity, key_heads, width)),
            (key_end, capsule_width, (capacity, heads, width)),
        ):
            np.testing.assert_array_equal(
                mixed["updates"][b, section_begin:section_end].reshape(dimensions)[
                    :count
                ],
                captured["updates"][b, section_begin:section_end].reshape(dimensions)[
                    :count
                ],
            )
    for bad in (
        dict(controlled, counts=controlled["counts"][:-1]),
        dict(controlled, active=np.array([1, 0], np.int32)),
        dict(controlled, key=controlled["key"][:-1]),
        dict(controlled, cu=np.array([0, 8, 7, tokens], np.int32)),
    ):
        run_capture(session, device, bad, outputs, alias=alias, invalid=True)
    run_capture(session, device, controlled, outputs, alias=alias)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gated_rms_norm(hybrid_sessions, dtype):
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    shapes = {
        "input": (activation, ["tokens", 6144]),
        "weight": (activation, [128]),
        "gate": (activation, ["gate_tokens", 6144]),
    }
    session, device = hybrid_sessions(
        "GatedRMSNorm", shapes, {"output": shapes["input"]}, epsilon=1e-6
    )
    rng = np.random.default_rng(3003)
    inputs = {
        "input": rng.normal(0, 1, (9, 6144)).astype(dtype),
        "weight": rng.normal(1, 0.2, 128).astype(dtype),
        "gate": rng.normal(0, 1, (9, 6144)).astype(dtype),
    }
    x = inputs["input"].astype(np.float32).reshape(9, 48, 128)
    z = inputs["gate"].astype(np.float32).reshape(x.shape)
    out = (
        (
            x
            / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
            * inputs["weight"].astype(np.float32)
            * z
            / (1 + np.exp(-z))
        )
        .reshape(9, 6144)
        .astype(dtype)
    )
    expected = {"output": out}
    run_bound(session, device, inputs, expected)
    run_bound(
        session, device, dict(inputs, gate=inputs["gate"][:-1]), expected, invalid=True
    )


def test_partial_interleaved_mrope(hybrid_sessions):
    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    shapes = {
        "input": (f32, [1, "tokens", 6144]),
        "positions": (i64, [3, 1, "position_tokens"]),
        "cos": (f32, ["cos_length", 32]),
        "sin": (f32, ["sin_length", 32]),
    }
    session, device = hybrid_sessions(
        "MRotaryEmbedding",
        shapes,
        {"output": shapes["input"]},
        num_heads=24,
        rotary_embedding_dim=64,
        interleaved=0,
        mrope_layout=1,
        is_packed_batching=1,
        mrope_section=[11, 11, 10],
    )
    rng = np.random.default_rng(3004)
    angle = rng.normal(0, 1, (31, 32)).astype(np.float32)
    inputs = {
        "input": rng.normal(0, 1, (1, 9, 6144)).astype(np.float32),
        "positions": rng.integers(0, 31, (3, 1, 9), dtype=np.int64),
        "cos": np.cos(angle),
        "sin": np.sin(angle),
    }
    source = inputs["input"].reshape(9, 24, 256)
    output = source.copy()
    for col in range(32):
        axis = (
            1 if col % 3 == 1 and col < 33 else (2 if col % 3 == 2 and col < 30 else 0)
        )
        positions = inputs["positions"][axis, 0]
        cosine, sine = inputs["cos"][positions, col], inputs["sin"][positions, col]
        output[:, :, col] = (
            source[:, :, col] * cosine[:, None] - source[:, :, col + 32] * sine[:, None]
        )
        output[:, :, col + 32] = (
            source[:, :, col + 32] * cosine[:, None] + source[:, :, col] * sine[:, None]
        )
    expected = {"output": output.reshape(1, 9, 6144)}
    run_bound(session, device, inputs, expected)
    bad_positions = inputs["positions"].copy()
    bad_positions[2, 0, 8] = 31
    for bad in (
        dict(inputs, positions=bad_positions),
        dict(inputs, positions=inputs["positions"][:, :, :-1].copy()),
        dict(inputs, sin=inputs["sin"][:-1].copy()),
    ):
        run_bound(session, device, bad, expected, invalid=True)
    run_bound(session, device, inputs, expected)
