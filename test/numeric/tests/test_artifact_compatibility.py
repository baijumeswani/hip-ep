#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Reusable artifacts must fail closed before incompatible code is loaded."""

import gc
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.ort_ep_backend import _parse_ep_options


def session_options(device, provider_options, *, export=None, allow_cpu_fallback=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    if not allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices([device], provider_options)
    if export is not None:
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_embed_mode", "1")
        options.add_session_config_entry("ep.context_file_path", str(export))
    return options


def write_source(
    path, *, weight_offset=0.0, graph_name="artifact_identity", external=True
):
    weight = np.arange(128, dtype=np.float16).reshape(16, 8) / 128 + weight_offset
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        graph_name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["rows", 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["rows", 8])],
        [numpy_helper.from_array(weight.astype(np.float16), "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    onnx.save_model(
        model,
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    return weight


def context_archive(path):
    model = onnx.load(path, load_external_data=False)
    context_node = next(
        node for node in model.graph.node if node.op_type == "EPContext"
    )
    archive = next(
        attr for attr in context_node.attribute if attr.name == "ep_cache_context"
    )
    with tarfile.open(fileobj=io.BytesIO(archive.s)) as source:
        files = {
            entry.name: source.extractfile(
                source.getmember(entry.linkname) if entry.issym() else entry
            ).read()
            for entry in source.getmembers()
            if entry.isfile() or entry.issym()
        }
    return model, archive, files


def source_fingerprint(path):
    _, _, files = context_archive(path)
    return json.loads(
        next(data for name, data in files.items() if name.endswith("/context.json"))
    )["sourceModelSha256"]


def rewrite_context(source, destination, mutate):
    model, archive, files = context_archive(source)
    context_name = next(name for name in files if name.endswith("/context.json"))
    context = json.loads(files[context_name])
    metadata = context["metaDef"][0]["param"]
    mutate(context, metadata, files)
    files[context_name] = json.dumps(context).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as target:
        for name, data in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            target.addfile(entry, io.BytesIO(data))
    archive.s = output.getvalue()
    onnx.save(model, destination)


@pytest.fixture(scope="module")
def artifact_fixture(request, tmp_path_factory):
    library = request.config.getoption("--ep-dll")
    if not library:
        pytest.skip("--ep-dll is required")
    if os.environ.get("HIPDNN_EP_STRICT") != "1":
        pytest.fail("Artifact qualification requires HIPDNN_EP_STRICT=1")
    provider = request.config.getoption("--ep-name") + "_artifact"
    ort.register_execution_provider_library(provider, str(Path(library).resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider]
    assert len(devices) == 1
    device = devices[0]
    root = tmp_path_factory.mktemp("artifact_compatibility")
    provider_options = _parse_ep_options(request.config.getoption("--ep-option"))
    source = root / "source.onnx"
    exported = root / "context.onnx"
    weight = write_source(source)
    options = session_options(device, provider_options, export=exported)
    session = ort.InferenceSession(str(source), sess_options=options)
    inputs = np.ones((2, 16), np.float16)
    np.testing.assert_allclose(session.run(None, {"x": inputs})[0], inputs @ weight)
    del session, options
    gc.collect()
    assert exported.is_file()
    yield device, provider_options, root, exported, weight
    del device, devices
    gc.collect()
    ort.unregister_execution_provider_library(provider)


def test_valid_artifact_persistent_dynamic_shapes(artifact_fixture):
    device, provider_options, _, exported, weight = artifact_fixture
    session = ort.InferenceSession(
        str(exported), sess_options=session_options(device, provider_options)
    )
    for rows in (1, 2, 7, 16, 1):
        values = np.full((rows, 16), 0.25, np.float16)
        np.testing.assert_allclose(session.run(None, {"x": values})[0], values @ weight)


def write_constant_free_source(source):
    def info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT16, ["rows", 16])

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["x", "z"], ["y"])],
            "constant_free",
            [info("x"), info("z")],
            [info("y")],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.save(model, source)


def test_constant_free_artifact_round_trip(artifact_fixture):
    device, provider_options, root, _, _ = artifact_fixture
    source, exported = root / "add.onnx", root / "add-context.onnx"
    write_constant_free_source(source)
    original = ort.InferenceSession(
        str(source),
        sess_options=session_options(device, provider_options, export=exported),
    )
    values = np.arange(32, dtype=np.float16).reshape(2, 16)
    feeds = {"x": values, "z": -values / 2}
    np.testing.assert_array_equal(original.run(None, feeds)[0], values / 2)
    del original
    restored = ort.InferenceSession(
        str(exported), sess_options=session_options(device, provider_options)
    )
    np.testing.assert_array_equal(restored.run(None, feeds)[0], values / 2)


def test_source_identity_covers_free_dimension_specialization(artifact_fixture):
    device, provider_options, root, _, _ = artifact_fixture
    source = root / "specialized-add.onnx"
    write_constant_free_source(source)
    fingerprints = []
    for index, rows in enumerate((4, 2, 4)):
        exported = root / f"specialized-add-context-{index}.onnx"
        options = session_options(
            device, {**provider_options, "cache_key": "same-key"}, export=exported
        )
        options.add_free_dimension_override_by_name("rows", rows)
        session = ort.InferenceSession(str(source), sess_options=options)
        data = np.ones((rows, 16), np.float16)
        np.testing.assert_array_equal(
            session.run(None, {"x": data, "z": data})[0], data * 2
        )
        del session, options
        fingerprints.append(source_fingerprint(exported))
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[0] == fingerprints[2]


@pytest.mark.parametrize("allow_cpu_fallback", [False, True])
def test_nested_external_weight_artifact_fails_closed(
    artifact_fixture, allow_cpu_fallback
):
    device, provider_options, root, _, _ = artifact_fixture
    variant = root / f"nested-{allow_cpu_fallback}"
    variant.mkdir()
    source = variant / "source.onnx"
    branches = {}
    for name in ("then_branch", "else_branch"):
        branches[name] = helper.make_graph(
            [helper.make_node("MatMul", ["x", "weight"], ["result"])],
            name,
            [],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT16, [2, 8])],
            [numpy_helper.from_array(np.ones((16, 8), np.float16), "weight")],
        )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("If", ["condition"], ["y"], **branches)],
            "nested_external_weight",
            [
                helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 16]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 8])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.save_model(
        model,
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    assert (variant / "weights.bin").is_file()
    with pytest.raises(Exception, match="nested graph weight identity"):
        ort.InferenceSession(
            str(source),
            sess_options=session_options(
                device,
                provider_options,
                export=variant / "context.onnx",
                allow_cpu_fallback=allow_cpu_fallback,
            ),
        )


@pytest.mark.parametrize("allow_cpu_fallback", [False, True])
def test_external_tensor_attribute_artifact_fails_closed(
    artifact_fixture, allow_cpu_fallback
):
    device, provider_options, root, _, _ = artifact_fixture
    variant = root / f"tensor-attribute-{allow_cpu_fallback}"
    variant.mkdir()
    source = variant / "source.onnx"
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["y"],
                    value=numpy_helper.from_array(np.array([0.25], np.float32)),
                )
            ],
            "external_fill_value",
            [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["rows", "cols"])],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 10
    onnx.save_model(
        model,
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="value.bin",
        size_threshold=0,
        convert_attribute=True,
    )
    assert (variant / "value.bin").is_file()
    with pytest.raises(Exception, match="tensor-attribute weight identity"):
        ort.InferenceSession(
            str(source),
            sess_options=session_options(
                device,
                provider_options,
                export=variant / "context.onnx",
                allow_cpu_fallback=allow_cpu_fallback,
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "manifest_version",
        "runtime_abi",
        "compiler_build",
        "llvm_version",
        "runtime_sha256",
        "compiler_config_sha256",
        "gpu_isa",
        "optimization_level",
        "env.HIPDNN_EP_MATMUL_DP4A",
    ],
)
def test_incompatible_manifest_rejected(artifact_fixture, field):
    device, provider_options, root, exported, _ = artifact_fixture
    changed = root / f"changed-{field}.onnx"

    def mutate(_context, metadata, _files):
        metadata["compatibility"][field] += "-incompatible"

    rewrite_context(exported, changed, mutate)
    with pytest.raises(Exception, match="HIP artifact compatibility mismatch"):
        ort.InferenceSession(
            str(changed), sess_options=session_options(device, provider_options)
        )


@pytest.mark.parametrize(
    "component",
    [
        "source",
        "code",
        "weights",
        "missing_manifest",
        "missing_source_identity",
        "missing_code",
        "missing_weights",
        "missing_weight_identity",
    ],
)
def test_payload_identity_rejected(artifact_fixture, component):
    device, provider_options, root, exported, _ = artifact_fixture
    changed = root / f"changed-{component}.onnx"

    def mutate(context, metadata, files):
        if component == "source":
            context["sourceModelSha256"] = "0" * 64
        elif component == "missing_manifest":
            del metadata["compatibility"]
        elif component == "missing_source_identity":
            del context["sourceModelSha256"]
        elif component == "missing_weight_identity":
            del metadata["fileSha256"]
        else:
            suffix = (
                "/model_compiled"
                if component in ("code", "missing_code")
                else ".constants.bin"
            )
            name = next(name for name in files if name.endswith(suffix))
            if component.startswith("missing_"):
                del files[name]
            else:
                data = bytearray(files[name])
                data[len(data) // 2] ^= 1
                files[name] = bytes(data)

    rewrite_context(exported, changed, mutate)
    with pytest.raises(
        Exception, match="HIP artifact|HIP compiled artifact|HIP source"
    ):
        ort.InferenceSession(
            str(changed), sess_options=session_options(device, provider_options)
        )


def test_changed_requested_compiler_options_rejected(artifact_fixture):
    device, provider_options, _, exported, _ = artifact_fixture
    requested = {**provider_options, "optimization_level": "1"}
    with pytest.raises(Exception, match="HIP artifact compatibility mismatch"):
        ort.InferenceSession(
            str(exported), sess_options=session_options(device, requested)
        )


def test_changed_session_options_rejected(artifact_fixture):
    device, provider_options, _, exported, _ = artifact_fixture
    requested = session_options(device, provider_options)
    requested.add_session_config_entry("session.use_env_allocators", "1")
    with pytest.raises(Exception, match="EPContext session options"):
        ort.InferenceSession(str(exported), sess_options=requested)


def test_incompatible_artifact_cannot_fall_back_to_cpu(artifact_fixture):
    device, provider_options, root, _, _ = artifact_fixture
    exported = root / "fallback-context.onnx"
    session = ort.InferenceSession(
        str(root / "source.onnx"),
        sess_options=session_options(
            device, provider_options, export=exported, allow_cpu_fallback=True
        ),
    )
    del session
    changed = root / "fallback-incompatible.onnx"

    def mutate(_context, metadata, _files):
        metadata["compatibility"]["runtime_abi"] = "incompatible"

    rewrite_context(exported, changed, mutate)
    with pytest.raises(Exception, match="HIP artifact compatibility mismatch"):
        ort.InferenceSession(
            str(changed),
            sess_options=session_options(
                device, provider_options, allow_cpu_fallback=True
            ),
        )


@pytest.mark.parametrize("component", ["graph", "external_weights"])
def test_source_identity_covers_graph_and_external_weights(artifact_fixture, component):
    device, provider_options, root, exported, _ = artifact_fixture
    variant = root / component
    variant.mkdir()
    source = variant / "source.onnx"
    target = variant / "context.onnx"
    write_source(
        source,
        weight_offset=0.25 if component == "external_weights" else 0.0,
        graph_name="changed_graph" if component == "graph" else "artifact_identity",
    )
    session = ort.InferenceSession(
        str(source),
        sess_options=session_options(device, provider_options, export=target),
    )
    del session
    assert source_fingerprint(exported) != source_fingerprint(target)


def test_source_identity_hashes_only_referenced_external_ranges(artifact_fixture):
    device, provider_options, root, exported, _ = artifact_fixture
    variant = root / "unreferenced-external-bytes"
    variant.mkdir()
    source = variant / "source.onnx"
    target = variant / "context.onnx"
    write_source(source)
    with (variant / "weights.bin").open("ab") as weights:
        weights.write(b"not part of an initializer")
    session = ort.InferenceSession(
        str(source),
        sess_options=session_options(device, provider_options, export=target),
    )
    del session
    assert source_fingerprint(exported) == source_fingerprint(target)


@pytest.mark.parametrize("threshold", ["127", "4096"])
def test_inline_initializer_artifact_round_trip(artifact_fixture, threshold):
    device, provider_options, root, _, _ = artifact_fixture
    provider_options = {
        **provider_options,
        "XLNX_model_clone_external_data_threshold": threshold,
    }
    source = root / f"inline-{threshold}.onnx"
    exported = root / f"inline-context-{threshold}.onnx"
    weight = write_source(source, external=False)
    session = ort.InferenceSession(
        str(source),
        sess_options=session_options(device, provider_options, export=exported),
    )
    del session
    restored = ort.InferenceSession(
        str(exported), sess_options=session_options(device, provider_options)
    )
    inputs = np.ones((2, 16), np.float16)
    np.testing.assert_allclose(restored.run(None, {"x": inputs})[0], inputs @ weight)
