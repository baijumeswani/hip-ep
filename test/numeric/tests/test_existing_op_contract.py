#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Dynamic existing-op state aliases and fail-stop completion through the umbrella."""

import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def create_session(library, path, provider, ep_options):
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    device = devices[0]
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    assert memory.device_id == 0
    assert memory.device_vendor_id == 0x1002
    weights = np.eye(8, dtype=np.float16)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("MatMul", ["X", "W"], ["product"]),
                helper.make_node("Gather", ["state", "order"], ["reordered"], axis=0),
                helper.make_node("Add", ["product", "reordered"], ["present"]),
            ],
            "existing_op_state",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["rows", 8]),
                helper.make_tensor_value_info(
                    "state", TensorProto.FLOAT16, ["rows", 8]
                ),
                helper.make_tensor_value_info("order", TensorProto.INT32, ["rows"]),
            ],
            [
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT16, ["rows", 8]
                )
            ],
            [numpy_helper.from_array(weights, "W")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.use_env_allocators", "1")
    options.add_provider_for_devices([device], ep_options)
    options.enable_profiling = True
    options.profile_file_prefix = str(path.with_suffix(".ort"))
    return ort.InferenceSession(str(path), sess_options=options), device


def run_alias(session, device, rows, before_run=None):
    values = np.arange(rows * 8, dtype=np.float16).reshape(rows, 8) / 8
    state = (values + 1).astype(np.float16)
    order = np.arange(rows - 1, -1, -1, dtype=np.int32)
    memory = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    tensors = {}
    binding = session.io_binding()
    for name, data in {"X": values, "state": state, "order": order}.items():
        tensor = ort.OrtValue.ortvalue_from_shape_and_type(
            data.shape, data.dtype, memory_info=memory
        )
        tensor.update_inplace(data)
        binding.bind_ortvalue_input(name, tensor)
        tensors[name] = tensor
    binding.bind_ortvalue_output("present", tensors["state"])
    if before_run is not None:
        before_run()
    session.run_with_iobinding(binding)
    assert binding.get_outputs()[0].data_ptr() == tensors["state"].data_ptr()
    np.testing.assert_array_equal(tensors["state"].numpy(), values + state[order])


def test_existing_op_state_alias_and_failed_drain(request, tmp_path):
    from framework.ort_ep_backend import _parse_ep_options

    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    if os.name != "posix" or not Path("/proc/self/maps").exists():
        pytest.skip("Linux fail-stop interposer qualification")
    assert os.environ.get("HIPDNN_EP_STRICT") == "1"
    compiler = shutil.which("cc")
    assert compiler, "The drain contract test requires a C compiler"
    provider = request.config.getoption("--ep-name") + "_contract"
    ep_options = _parse_ep_options(request.config.getoption("--ep-option"))
    session, device = create_session(
        library, tmp_path / "state.onnx", provider, ep_options
    )
    for rows in [1, 2, 7, 8, 16, 2, 1]:
        run_alias(session, device, rows)
    events = json.loads(Path(session.end_profiling()).read_text())
    assigned = {
        event["args"]["provider"]
        for event in events
        if "provider" in event.get("args", {})
    }
    assert assigned == {provider}
    del session, device
    gc.collect()
    ort.unregister_execution_provider_library(provider)

    interposer = tmp_path / "drain.so"
    source = Path(__file__).with_name("compute_drain_interposer.c")
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-ldl", "-o", str(interposer)],
        check=True,
        capture_output=True,
        text=True,
    )
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(library),
            str(tmp_path / "failure.onnx"),
            provider,
            json.dumps(ep_options),
            str(interposer),
        ],
        env={**os.environ, "LD_PRELOAD": str(interposer)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == -signal.SIGABRT, child.stdout + child.stderr
    assert "HIP compute drain failed; device buffers cannot be released" in child.stderr

    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(library),
            str(tmp_path / "allocation.onnx"),
            provider,
            json.dumps(ep_options),
            str(interposer),
            "allocation",
        ],
        env={**os.environ, "LD_PRELOAD": str(interposer)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert (
        "hipMalloc(512) failed" in child.stderr
        or "allocation injection skipped" in child.stdout
    )


def run_allocation_failure(library, path, provider, ep_options, hook):
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    ["X", "W", "scales"],
                    ["Y"],
                    domain="com.microsoft",
                    K=32,
                    N=8,
                    bits=4,
                    block_size=32,
                )
            ],
            "scratch_failure",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, ["rows", 32])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["rows", 8])],
            [
                numpy_helper.from_array(np.full((8, 1, 16), 0x99, np.uint8), "W"),
                numpy_helper.from_array(np.ones((8, 1), np.float16), "scales"),
            ],
        ),
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices, ep_options)
    session = ort.InferenceSession(str(path), sess_options=options)
    session.run(None, {"X": np.ones((1, 32), np.float16)})
    runtime = ctypes.CDLL("libamdhip64.so")
    device = ctypes.c_int()
    wave_size = ctypes.c_int()
    assert runtime.hipGetDevice(ctypes.byref(device)) == 0
    # hipDeviceAttributeWarpSize is value 87 in the HIP C ABI.
    assert runtime.hipDeviceGetAttribute(ctypes.byref(wave_size), 87, device) == 0
    if wave_size.value < 64:
        print("FP16 expansion scratch is wave64-only; allocation injection skipped")
        return
    hook.arm_scratch_allocation_failure()
    with pytest.raises(Exception, match="inference_compute"):
        session.run(None, {"X": np.ones((16, 32), np.float16)})
    np.testing.assert_array_equal(
        session.run(None, {"X": np.ones((16, 32), np.float16)})[0],
        np.full((16, 8), 32, np.float16),
    )


if __name__ == "__main__":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    library, model_path, provider, options, interposer = sys.argv[1:6]
    hook = ctypes.CDLL(interposer)
    if sys.argv[6:] == ["allocation"]:
        run_allocation_failure(
            library, Path(model_path), provider, json.loads(options), hook
        )
        sys.exit(0)
    session, device = create_session(
        library, Path(model_path), provider, json.loads(options)
    )
    run_alias(session, device, 2)
    run_alias(session, device, 2, before_run=hook.arm_compute_drain_failure)
    raise AssertionError(
        "Failed drain returned to a caller that could release its allocations"
    )
