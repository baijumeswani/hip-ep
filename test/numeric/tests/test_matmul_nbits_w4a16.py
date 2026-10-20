#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Raw INT4 qualification using the pinned Qwen fixture's actual weights."""

import ctypes
import gc
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.ort_ep_backend import _parse_ep_options


@pytest.fixture(scope="module")
def w4a16_weights(request):
    graph_path = os.environ.get("HIPDNN_W4A16_MODEL")
    library = request.config.getoption("--ep-dll")
    if not graph_path or not library:
        pytest.skip("Set HIPDNN_W4A16_MODEL and --ep-dll for real-weight qualification")
    assert os.environ.get("HIPDNN_EP_STRICT") == "1"
    assert os.environ.get("HIPDNN_EP_MATMUL_DP4A") == "0"
    graph = onnx.load(graph_path).graph
    tensors = {tensor.name: tensor for tensor in graph.initializer}
    weights = {}
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        assert attrs["bits"] == 4 and attrs["block_size"] == 32
        assert attrs.get("weight_prepacked", 0) == 0 and not any(node.input[3:])
        geometry = (attrs["K"], attrs["N"])
        if geometry not in weights:
            weights[geometry] = tuple(tensors[name] for name in node.input[1:3])
    provider = request.config.getoption("--ep-name") + "_w4a16"
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    yield weights, devices[0], provider
    del devices, tensors, weights, graph
    gc.collect()
    ort.unregister_execution_provider_library(provider)


def unpack_weight(packed, scales, k, n):
    data = numpy_helper.to_array(packed)
    scale = numpy_helper.to_array(scales).reshape(n, -1, 1).astype(np.float32)
    groups = (k + 31) // 32
    assert data.shape == (n, groups, 16) and scale.shape == (n, groups, 1)
    nibbles = np.stack((data & 15, data >> 4), axis=-1).reshape(n, groups, 32)
    return ((nibbles.astype(np.float32) - 8) * scale).reshape(n, -1)[:, :k]


def test_qwen_w4a16_runtime_error_reaches_caller(w4a16_weights, request):
    weights, device, _ = w4a16_weights
    k = n = 896
    packed, scales = weights[k, n]
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["X", packed.name, scales.name, "", "g_idx"],
                ["Y"],
                domain="com.microsoft",
                K=k,
                N=n,
                bits=4,
                block_size=32,
            )
        ],
        "qwen_w4a16_unsupported_g_idx",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["rows", k]),
            helper.make_tensor_value_info("g_idx", TensorProto.INT32, [k]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["rows", n])],
        initializer=[packed, scales],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        [device], _parse_ep_options(request.config.getoption("--ep-option"))
    )
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options)
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    source = np.ones((1, k), dtype=np.float16)
    groups = np.arange(k, dtype=np.int32) // 32
    inputs = []
    binding = session.io_binding()
    for name, data in (("X", source), ("g_idx", groups)):
        value = ort.OrtValue.ortvalue_from_shape_and_type(
            data.shape, data.dtype, memory_info=memory
        )
        value.update_inplace(data)
        inputs.append(value)
        binding.bind_ortvalue_input(name, value)
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        (1, n), np.float16, memory_info=memory
    )
    output.update_inplace(np.full((1, n), np.nan, dtype=np.float16))
    binding.bind_ortvalue_output("Y", output)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="inference_compute .* failed"):
            session.run_with_iobinding(binding)
        assert np.isnan(output.numpy()).all()
        np.testing.assert_array_equal(inputs[0].numpy(), source)
        np.testing.assert_array_equal(inputs[1].numpy(), groups)
    del binding, output, value, inputs, session, options
    gc.collect()


@pytest.mark.parametrize(
    "dtype,k,n",
    [
        (dtype, k, n)
        for dtype in (np.float16, np.float32)
        for k, n in ((896, 1152), (896, 896), (896, 4864), (4864, 896), (896, 151936))
    ]
    + [(np.float32, 883, 263)],
)
def test_qwen_w4a16_dynamic_shapes(w4a16_weights, request, tmp_path, dtype, k, n):
    weights, device, provider = w4a16_weights
    if k % 32:
        packed, scales = weights[896, 896]
        packed = numpy_helper.from_array(
            numpy_helper.to_array(packed)[:n, : (k + 31) // 32].copy(), packed.name
        )
        scales = numpy_helper.from_array(
            numpy_helper.to_array(scales).reshape(896, -1)[:n, : (k + 31) // 32].copy(),
            scales.name,
        )
    else:
        packed, scales = weights[k, n]
    if dtype == np.float32:
        scales = numpy_helper.from_array(
            numpy_helper.to_array(scales).astype(np.float32) * np.float32(1.000123),
            scales.name,
        )
    activation = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    dense = unpack_weight(packed, scales, k, n)
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMulNBits",
                ["X", packed.name, scales.name],
                ["Y"],
                domain="com.microsoft",
                K=k,
                N=n,
                bits=4,
                block_size=32,
            )
        ],
        "qwen_w4a16",
        [helper.make_tensor_value_info("X", activation, ["rows", k])],
        [helper.make_tensor_value_info("Y", activation, ["rows", n])],
        initializer=[packed, scales],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    model_path = tmp_path / "matmul.onnx"
    onnx.save(model, model_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        [device], _parse_ep_options(request.config.getoption("--ep-option"))
    )
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "profile")
    session = ort.InferenceSession(str(model_path), sess_options=options)
    rng = np.random.default_rng(27)
    # Revisit both dispatch sides after changing M, on the same compiled graph.
    for rows in (1, 2, 3, 7, 8, 16, 17, 128, 3, 16, 1, 128):
        source = rng.uniform(-1, 1, (rows, k)).astype(dtype)
        expected = source.astype(np.float32) @ dense.T
        actual = session.run(None, {"X": source})[0].astype(np.float32)
        assert np.isfinite(actual).all()
        difference = actual - expected
        if dtype == np.float32:
            np.testing.assert_allclose(actual, expected, atol=0.00003, rtol=0.00002)
        else:
            np.testing.assert_array_less(
                np.abs(difference), 0.02 + 0.003 * np.abs(expected)
            )
        normalized_rms = np.sqrt(np.mean(difference.astype(np.float64) ** 2)) / max(
            np.sqrt(np.mean(expected.astype(np.float64) ** 2)), 1e-12
        )
        assert normalized_rms <= (0.000002 if dtype == np.float32 else 0.003)
    events = json.loads(Path(session.end_profiling()).read_text())
    assigned = {
        event["args"]["provider"]
        for event in events
        if "provider" in event.get("args", {})
    }
    assert assigned == {provider}
    del session, options
    gc.collect()


@pytest.mark.parametrize("k", [32, 65])
@pytest.mark.parametrize("scale_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("offsets", [(0, 0), (4, 1)])
def test_fp32_small_m_element_aligned_inputs(w4a16_weights, k, scale_dtype, offsets):
    """Caller-owned FP32/byte tensors need not have float4/uint32 alignment."""
    _, device, _ = w4a16_weights
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    assert memory.device_id == 0 and memory.device_vendor_id == 0x1002
    library = os.environ.get("HIP_CUSTOM_KERNEL_LIBRARY")
    if not library:
        pytest.skip(
            "Set HIP_CUSTOM_KERNEL_LIBRARY to the device's custom-kernel library"
        )
    hip = ctypes.CDLL(
        os.environ.get(
            "HIP_RUNTIME_LIBRARY",
            "amdhip64.dll" if os.name == "nt" else "libamdhip64.so",
        )
    )
    kernels = ctypes.CDLL(str(Path(library).resolve()))
    kernel = kernels.hip_matmul_nbits_gemv_fp32
    kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
    kernel.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    assert hip.hipStreamCreate(ctypes.byref(stream)) == 0
    try:
        rng = np.random.default_rng(91)
        n = 17
        groups = (k + 31) // 32
        packed = rng.integers(0, 256, (n, groups, 16), dtype=np.uint8)
        scales = rng.uniform(-0.01, 0.01, (n, groups)).astype(scale_dtype)
        codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(n, groups, 32)
        dense = ((codes.astype(np.float32) - 8) * scales[:, :, None]).reshape(n, -1)[
            :, :k
        ]
        for rows in (1, 3, 16):
            source = rng.uniform(-1, 1, (rows, k)).astype(np.float32)
            buffers = []
            pointers = []
            for data, offset in (
                (source, offsets[0]),
                (packed, offsets[1]),
                (scales, 0),
            ):
                storage = np.zeros(data.nbytes + offset, dtype=np.uint8)
                storage[offset:] = data.view(np.uint8).reshape(-1)
                value = ort.OrtValue.ortvalue_from_shape_and_type(
                    storage.shape, np.uint8, memory_info=memory
                )
                value.update_inplace(storage)
                buffers.append(value)
                pointers.append(ctypes.c_void_p(value.data_ptr() + offset))
            output = ort.OrtValue.ortvalue_from_shape_and_type(
                (rows, n), np.float32, memory_info=memory
            )
            assert pointers[0].value % 16 == offsets[0]
            assert pointers[1].value % 4 == offsets[1]
            assert (
                kernel(
                    stream,
                    *pointers,
                    None,
                    ctypes.c_void_p(output.data_ptr()),
                    rows,
                    n,
                    k,
                    np.dtype(scale_dtype).itemsize,
                )
                == 0
            )
            assert hip.hipStreamSynchronize(stream) == 0
            np.testing.assert_allclose(
                output.numpy(), source @ dense.T, atol=3e-5, rtol=3e-5
            )
    finally:
        assert hip.hipStreamDestroy(stream) == 0
