<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Compiler–Runtime Contract
**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [constant-handling-design.md](constant-handling-design.md), [morphizen-ep-integration.md](morphizen-ep-integration.md)

---

## Table of Contents

- [Overview](#overview)
- [Schema: Single Source of Truth](#schema-single-source-of-truth)
- [Embedding Mechanism](#embedding-mechanism)
- [Current Fields](#current-fields)
- [Generated-Code Runtime Inputs](#generated-code-runtime-inputs)
- [Consumers](#consumers)
- [Design Principles](#design-principles)
- [Related Documents](#related-documents)

---

## Overview

`schemas/model_metadata.fbs` is the **single source of truth for serialized
model metadata** shared with artifact loaders and external consumers. It does
not contain every compile-time value passed to runtime functions by generated
LLVM IR. Transient memory planning is the canonical generated-code-only
contract; see [Generated-Code Runtime Inputs](#generated-code-runtime-inputs).

```
  Compiler (GenerateInterface pass)
        │
        │  serializes model_metadata into:
        │    __metadata_blob  (binary, for runtime)
        │    __metadata_json  (text, for inspection)
        ▼
  ┌───────────────────────────────────────────┐
  │            model.bc (bitcode)             │
  │  __metadata_blob       (internal)         │  ← baked into bitcode data
  │  __metadata_json       (internal)         │  ← human-readable copy
  │                                           │
  │  inference_init(out_state, fs, config)    │  ← public API
  │  inference_get_metadata_json()            │  ← public API
  └───────────────────────────────────────────┘
        │
        │  LlvmIrJit parses bytes → llvm::orc::LLJIT codegen
        │  inference_init reads __metadata_blob directly
        │  (same LLVM module — hidden from public API)
        ▼
  Runtime (EP DLL hosts the JIT)
```

This contract is **format-independent**. The same metadata blob, the same
`inference_*` / `hipdnn_ep_runtime_begin_compute` symbols, and the same JSON
mirror are produced for both artifact formats
(`CompilationOptions.output_mode`):

- **LLVM IR** (default): the symbols are JIT-resolved by `LlvmIrJit` in-process.
- **Native**: the same symbols are exported from the per-model `.dll`/`.so` and
  resolved via `morphizen::Plugin` (`GetProcAddress` / `dlsym`). The runtime is
  merged into the module at producer time so `hipdnn_ep_runtime_begin_compute`
  is present in the native artifact too.

The EP selects the loader from `mlir_metadata::Metadata.artifact_format` before
opening the artifact. `NATIVE` is available only when the EP is built with
`-DHIPDNN_EP_ENABLE_NATIVE_ARTIFACTS=ON` (default `OFF`); otherwise session
setup throws `std::runtime_error`. See [native-vs-ir-comparison.md](../native-vs-ir-comparison.md).

`Metadata.rocmlir_arch` records what the embedded rocMLIR code objects were
built for, not only the GPU arch: it is `hip::compiler::rocMlirArtifactTarget()`,
which is empty when `HIPDNN_EP_ROCMLIR` was off, the arch alone when rocMLIR
compiled with default perfConfigs (`gfx1151`), and the arch with an
`+autotune=<space>` suffix when `HIPDNN_EP_ROCMLIR_AUTOTUNE` selected
benchmarked ones (`gfx1151+autotune=quick`). The perfConfig a kernel was tuned
to is baked into its code object, so a tuned and an untuned artifact are not
interchangeable even on the same GPU.

The EP refuses to load an artifact whose recorded value differs from the current
one. Compilation fails outright if autotuning was requested but no HIP device
could be reached, so the suffix never claims tuning that did not happen.

### Reusable artifact compatibility

Reusable prebuilt/EPContext compilations additionally require a versioned
compatibility manifest in `backend-mlir-compiler/proto/metadata.proto`. The
manifest binds the compiled code and externalized constants with SHA-256 and
records compiler build identity, LLVM version, runtime-bitcode identity, runtime
ABI, selected GPU ISA/HIP runtime, compiler/provider options, and precision and
pipeline environment switches. Changing any recorded execution contract
requires recompilation; unknown or missing manifest fields fail closed.
Increment `runtime_abi` when changing an incompatible generated-code contract.

`ContextProto.source_model_sha256` hashes the source ONNX file, effective graph
input/output element types and shapes, and sorted initializer contents, including
externally stored weights. Effective interfaces bind free-dimension overrides
that specialize a graph without changing the source file or session config
entries. This covers graph
attributes such as packing, dtypes, scales and capture capacity as well as
topology. Prebuilt reuse compares this identity with the requested source.
EPContext restoration verifies the preserved identity against its compiled
metadata, then checks the code and constant payloads before loading or executing
code. This is compatibility/corruption detection, not authentication of untrusted
artifacts.

The metadata records the expected external dependency count independently of
the checksum map. Constant-free graphs may have an empty dependency set; a
missing checksum for an expected dependency is an error.

Reusable compilation requires the MLIR backend and a file-backed source model
without nested graphs or tensor-valued operation attributes.
Control-flow artifacts fail closed because the initializer API currently exposes
only the owning graph's weights, not recursive branch/body identities.
Tensor-attribute weights must first be represented as graph initializers; their
external payloads are not covered by source-protobuf hashing. The alternative
ONNX backend's external-location API does not yet provide the byte-range contract
needed for reusable identity. Ordinary in-memory
session compilation is unchanged, but cannot export or request reusable
artifacts without a source-file identity. Hashing occurs at compilation/session
creation, never per inference; constant payloads are streamed through bounded
host scratch. Session options are also compared, excluding `ep.context_*`
export-location/mode settings. The old structural cache key remains only a
namespace, not permission to reuse an incompatible artifact. Ordinary
non-prebuilt file-cache lookup still misses.

Compatibility failures reject session creation even when normal unsupported
subgraphs are allowed to fall back. Because ONNX Runtime treats `GetCapability`
errors as unsupported graphs, the bridge retains hard compilation failures and
returns them from `Compile` instead. Regenerate older artifacts that lack this
manifest rather than interpreting them with a newer runtime.

---

## Schema: Single Source of Truth

`schemas/model_metadata.fbs` defines the schema compiled by FlatBuffers into
`__metadata_blob`. The compiler writes the blob from MLIR module attributes;
the runtime reads it directly.

```
table HipModelMetaInfo {
  version:            int32 = 1;
  constants_filename: string;
  constants:          [ConstantInfo];
  input_count:        int64;
  output_count:       int64;
  inputs:             [TensorInfo];
  outputs:            [TensorInfo];
  // ...
}
```

See [Current Fields](#current-fields) for the purpose of each field.

---

## Embedding Mechanism

The `GenerateInterface` pass (`lib/Dialect/Transforms/GenerateInterface.cpp`)
builds the FlatBuffers blob from MLIR module attributes and emits two globals
into the LLVM module with **internal linkage** (not visible to JITted callers
outside `model.bc`):

| Global | Content | Linkage |
|--------|---------|---------|
| `__metadata_blob` | FlatBuffers binary bytes | Internal |
| `__metadata_json` | Human-readable JSON representation | Internal |

Both are compile-time constants baked directly into the bitcode's data section
and end up in JITted memory after ORC codegen. No file I/O is needed to access
them at runtime.

The generated `inference_init` accesses `__metadata_blob` directly — they are
in the same LLVM module and JITted together. This is entirely hidden from the
public interface.

### inference_get_metadata_json

`inference_get_metadata_json()` is a public export that returns a pointer to
`__metadata_json`. The caller must not free the pointer. Safe to call from
multiple threads.

```c
const char* inference_get_metadata_json(void);
```

The JSON mirrors the FlatBuffers schema. `constant_sizes` is included for
external consumers; `constant_offsets` is omitted (internal to the runtime).
Dynamic dimensions are encoded as `-1`.

```json
{
  "version": 1,
  "constants_filename": "constants.bin",
  "constant_sizes": [int64, ...],
  "input_count": int64,
  "output_count": int64,
  "inputs":  [ { "shape": [int64, ...], "element_size": int64 }, ... ],
  "outputs": [ { "shape": [int64, ...], "element_size": int64 }, ... ]
}
```

---

## Current Fields

| Field | Type | Purpose |
|-------|------|---------|
| `version` | `int32` | Schema version |
| `constants_filename` | `string` | Name of the constants file to open via `fs` |
| `constants[i].size` | `int64` | Byte size of constant `i` |
| `constants[i].offset` | `int64` | Byte offset of constant `i` in the constants file |
| `input_count` | `int64` | Number of model inputs |
| `output_count` | `int64` | Number of model outputs |
| `inputs[i].shape` | `[int64]` | Tensor dimensions of input `i`; `-1` = dynamic |
| `inputs[i].element_size` | `int64` | Bytes per element of input `i` |
| `outputs[i].shape` | `[int64]` | Tensor dimensions of output `i`; `-1` = dynamic |
| `outputs[i].element_size` | `int64` | Bytes per element of output `i` |

---

## Generated-Code Runtime Inputs

Some compiler/runtime coupling is encoded directly in generated LLVM calls
rather than serialized in `model_metadata.fbs`. These contracts are internal to
the compiled model and are documented by the design that owns the lowering.

Pool planning is the canonical example:

- `hip-pool-allocs` writes `hipdnn.pool_size`,
  `hipdnn.buffer_count`, `hipdnn.buffer_offsets`, and optional
  multi-domain attributes on the MLIR module.
- `generate-interface` consumes the legacy attributes at compile time and emits
  `hipdnn_ep_pool_init` only when all three are present and
  `hipdnn.pool_size > 0`.
- Each `hip.get_pool` lowers directly to
  `hipdnn_ep_get_pool_base(state, domain_id, needed_size)`.
- None of the `hipdnn.pool_*` or `hipdnn.buffer_*` attributes are fields in the
  FlatBuffers schema or the JSON mirror.

The generated LLVM IR plus `RuntimeState` therefore carry pool behavior. See
[pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) for the
attribute, lowering, and grow-on-demand runtime contract.

### Execution failures

Generated graph calls do not consume individual operator return codes.
`inference_compute` resets the session's device error flag before graph
execution and checks it afterward, synchronizing the runtime stream during
readback. Returning a nonzero operator status alone is therefore insufficient
to fail inference.

The real MatMul and MatMulNBits wrappers use `ExecutionStatusGuard` to record
every unsuccessful exit in that flag, including argument validation and
hipBLASLt failures. Successful calls do not clear earlier errors. If recording
the failure itself fails, the process aborts instead of returning apparent
success. Hybrid wrappers and metadata-validation kernels also report through
the same session flag. Output buffers from a failed inference must not be
consumed as successful results.

### Hybrid activation precision

Qwen hybrid convolution, gated-delta and grouped gated normalization accept
matching FP16 or FP32 activation/weight tensors. The generated wrapper ABI
carries the activation dtype explicitly; BF16 and mixed activation types are
rejected. Recurrent state and gate arithmetic remain FP32. Separate head-256
paged attention accepts either activation type but always stores FP16 KV:
FP32 Q and attention output are not narrowed, and K/V are rounded only when
written to the cache. Packed head-64 attention remains FP16.

Symmetric raw block-32 W4A32 MatMulNBits preserves FP32 activations, outputs,
dequantized weights and accumulation. Up to 16 flattened input rows use a
fused INT4/FP32 GEMV with independent 32-lane reductions on both wave32 and
wave64, without expanded-weight scratch or runtime algorithm tuning. Blocks
reuse packed weights across four rows for batches of three through sixteen;
one- and two-row calls retain independent rows with their original reductions
and activation loads. FP16 and
FP32 scales are consumed in their declared types. Larger row counts retain
FP32 dequantization and hipBLASLt GEMM in tiles of at most 16384 output channels,
using session-owned weight scratch independently of hipBLASLt workspace.
Neither path retains a full expanded model or uses the legacy FP16 activation
adapter. GEMM algorithms are selected for each exact matrix geometry, including
tail tiles and the full output leading dimension.
Caller-bound tensors need only their element alignment: FP32 activation
vector loads additionally check the base address, and packed byte weights
are loaded without imposing a wider alignment requirement.

The typed hybrid wrapper signatures were introduced in runtime ABI version 2.
Runtime ABI version 3 adds session-owned paged-attention scratch and requires
recompilation of older artifacts. Runtime ABI version 4 replaces byte-width
dispatch with explicit `HIPDNN_EP_DATATYPE_*` arguments for dense MatMul,
RMS normalization and skip-RMS normalization. BF16 operands use native BF16
storage and FP32 accumulation; descriptor caches distinguish BF16 from FP16.
ONNX import and reverse type mapping preserve BF16 rather than promoting it to
FP32 or confusing it with FP16. Native casts connect BF16 with FP16 and FP32.
Cast and RMS wrapper failures set the session error flag even when generated
graph calls discard the individual return value.
Native BF16 Add/Mul perform FP32 arithmetic with BF16 output rounding and
per-axis broadcasting, without FP16 reinterpretation. Byte-valued masks
(including boolean storage) can be cast directly to BF16. Runtime Slice
preserves BF16 bits for positive and negative steps. Elementwise and Slice
wrappers also retain rejected contracts and launch failures in the session
error flag; unsupported operations must not return an apparently successful
graph with untouched outputs.
Legacy byte-width arithmetic ABIs do not distinguish BF16 from FP16.
LayerNormalization (including SkipLayerNormalization), standalone Softmax/RoPE, TopK, GQA,
MultiHeadAttention, CausalConvWithState and QMoE therefore reject BF16 at
lowering rather than reinterpret its bits. The explicitly typed RMS and
windowed paged-attention operations below are separate supported contracts.
ScatterElements permits BF16 bitwise assignment but rejects BF16 arithmetic
reductions on its byte-width ABI.

Slice bounds that are all compile-time constants survive conversion and
bufferization as `hipdnn.constant_{starts,ends,axes,steps}` arrays. LLVM
lowering materializes host INT64 arrays and calls `wrap_slice_host`; other
slices retain `wrap_slice` and its synchronized device readback. Both
entrypoints use the same clamping, logical-extent, zero-tail, and
empty-accumulator-capacity behavior. Static positive-stride slices still
prefer zero-copy views. The additional symbol does not change the existing
runtime entrypoint or artifact ABI.

Runtime ABI version 5 replaces MatMulNBits activation and scale byte widths
with explicit dtype arguments. Raw symmetric INT4/block-32 supports native
BF16 activations, bias and output, with FP16, BF16 or FP32 scales. Small-M
fused projection and bounded prefill expansion both round dequantized weights
to BF16 before FP32 accumulation; neither reinterprets BF16 as FP16 or retains
a model-wide expanded copy. Prefill scratch is bounded to at most 16384
output channels, as on the FP32 path. BF16 zero points, group indices, other
bit widths or block sizes, and CUDA-prepacked weights remain unsupported and
are rejected. FP16 activations with BF16 scales are also rejected.
Older artifacts must be recompiled, not passed through the typed ABI.
These contracts establish supported
computation, not full-model accuracy or a 24-GB residency guarantee.

### Compact hybrid transition capture

The existing convolution and Qwen gated-delta operators additionally support
compact prefixes with capacity at most seven. Separate internal
`hip.varlen_causal_conv_updates` and `hip.gated_delta_net_updates` operations
preserve the capture-disabled wrapper ABI and compile out capture work on
that path. FP16/FP32 activation contracts and FP32 recurrent arithmetic are
unchanged; captured and capture-disabled full-step outputs must agree.

Convolution returns raw input values `[batch,capacity,channels]`, with unused
entries zeroed. Gated delta returns FP32
`[batch,capacity*(value_heads + key_heads*128 + value_heads*128)]` rows:
all decay factors first, then normalized keys, then per-value-row deltas.
These are transitions, not full-state snapshots. Capture starts at each
packed sequence's first token, including a verification anchor. Counts clamp
to `[0,min(capacity,sequence_length)]`, matching the operator schema.

The GDN `[1]` int32 activity input remains a host graph control; normal
compiled-input staging uploads it before the kernel reads it. Zero disables
capture and clears the capsule without changing full-step computation.
While active, entries outside the captured prefix remain unspecified.
Dynamic activity extents are checked before execution, and invalid packed
lengths or dynamic extents prevent output/state mutation. Capture outputs
cannot overlap inputs, full-step outputs or persistent state.
Every retained prefix must reproduce an independent capture-disabled
execution; committing all eight verification inputs uses the full staged
state rather than asking a capacity-seven capsule to represent eight steps.

### Compiled execution status

Compiled wrapper return statuses are recorded after each call, including
wrappers without a local guard. The session preserves host-contract rejection
separately from explicitly tagged device execution API failures; later host
errors cannot downgrade a device failure. Native argument checks are never
classified from the sign or magnitude of a HIP error code.

The bridge drains before exposing either error. Tagged device failures use
`ORT_EP_FAIL`; host-contract errors retain `ORT_RUNTIME_EXCEPTION`. A failed
drain remains fail-stop. Runtime ABI 6 artifacts must export the execution-status hook;
older artifacts without it are rejected before runtime state allocation.

### Cooperative vocabulary selection

Wide FP32 last-axis TopK with K at most 32 uses one cooperative block per
row instead of one serial thread per row. Selection preserves exact values,
lower-index ties, smallest/largest modes and sorted output. Rows containing
NaNs retain the legacy scan's ordering; other geometries and types retain
their existing dispatch. Rejected contracts and launch failures are recorded
in the session error flag.

### BF16 windowed attention

`hip.paged_attention_bf16` accepts packed native BF16 QKV, head size 128,
BF16 per-head Q/K RMS weights and BF16 half-rotated RoPE tables. It implements
noncausal query-block attention with a positive left window of at most 4096
positions; the window includes the current position, while the block's future
keys remain visible. `hip.paged_attention_bf16_separate` accepts separate
BF16 Q, K and V matrices with the same semantics and uses their native row
strides without allocating or copying a packed QKV tensor. Sinks, explicit
slot mappings, quantization and interleaved/offset RoPE are rejected.

Q/K RMS uses a widened reduction and normalization before rounding to BF16.
This prevents FP32 rounding near BF16 halfway points from being amplified by
cancellation in RoPE, and preserves BF16's exponent range when squaring.
RoPE preserves BF16 intermediate rounding. Attention uses FP32 dot products
and online softmax in fixed 128-key tiles, with native BF16 cache and output.
Workspace contains only the packed normalized queries, not full-context
expanded KV or a full score matrix.

Validation completes before any cache mutation. Repeated read-only ring pages
are valid; overlapping new writes and write/read aliases (including live
history in the same request) are
rejected. Only live window pages are inspected, and all positions still use
their absolute RoPE indices. Past/present cache aliases and metadata extents
are checked at runtime, including equal Q/K/V token extents for separate
inputs before any mutation. Existing packed runtime entrypoints retain their
signatures. This operator qualification does not enable the
complete BF16 drafter or speculative Engine by itself.

### Static INT8 paged target attention

`hip.paged_attention_int8` uses FP32 separate Q/K/V and output, signed INT8
`[blocks,page_size,kv_heads,256]` caches, and finite positive FP32
`[kv_heads,1,256]` dequantization scales. Writes divide by the channel scale,
clamp to `[-128,127]`, and round ties to even. Reads multiply signed cache
values by that same scale. Packed INT4, dynamic scales, rotation, BF16,
Q/K normalization and drafter attention are not part of this operation.
Past/present cache pointers must alias exactly.

Device validation checks metadata, page bounds, conflicting writes and every
scale before appending any cache data. Long INT8 page tables distribute the
unchanged page checks across 128-thread blocks; FP16 launch geometry is unchanged.
Only cumulative/past control vectors and
the error flag are read back, once per attention call, with explicit stream
completion. Failure to drain host readback aborts before releasing host storage.
The optional ONNX `attention_metadata` remains a shape-checked hint; it does not
replace validation of the actual device lengths and page table.

Attention processes each request through FP32 batched hipBLASLt GEMMs and online
softmax. Decode-only calls use one query and up to 16384 KV tokens per tile;
prefill and mixed calls use up to 256 queries and 4096 KV tokens. Tile sizes
adapt down toward the previous 64-query/4096-key envelope when necessary to
preserve the workspace limit for larger head geometries.
K/V are dequantized once per KV head in a tile, not expanded to query
heads or to the full context. A session shares one attention scratch allocation,
independent of BLAS workspace and weight scratch. Its request is
`4*(3*heads*query_tile*256 + 2*kv_heads*key_tile*256 +
heads*query_tile*key_tile + 3*heads*query_tile)`
bytes, independent of context length and active request count; excessive
requests above 256 MiB fail explicitly. For 24 query heads and four KV heads
this is at most 153,165,824 bytes for prefill/mixed calls and 135,864,608 bytes
for decode-only calls. Retained BLAS algorithm workspace is additional and
must be included in the measured memory ledger.

The context ceiling is 262144 tokens including appended queries; page capacity
and actual allocation remain separate limits. The legacy FP16-cache packed and
separate operations retain their 4096-token envelope. The INT8 operation reuses
the ordinary Matmul per-session descriptor/algo table, with exact QK and PV
geometries and strides. A runtime error is recorded in the session flag, not
returned only to generated code that would discard it.

The optional `kv_cache_residual_bits=8` ONNX attribute selects an explicit
two-byte correction format; its default of zero preserves plain INT8. The
physical cache width is 512 for a logical head of 256. For each head, the
first 256 bytes store a signed high component and the next 256 bytes store
the low component biased by -128. With calibrated channel scale `S`, encode
`q = round_ties_even(clamp(x/(S/256), -32768, 32767))`, `high = floor(q/256)`,
and `low = q - 256*high - 128`. Decode the integer expression
`256*high + low + 128` before multiplying once by `S/256`.

Scales must be finite and positive, and `S/256` must remain positive in FP32.
Validation precedes all writes. Zero is encoded as high zero and low -128;
unpopulated cache storage must not be read as an implicit zero prefix.
This uses **16 bits total per value**, not eight-bit cache compression.
Graph-derived physical widths determine allocation and capacity accounting;
262144 tokens across sixteen four-head layers require 16 GiB for K/V.
Dequantized tile geometry and the bounded attention scratch are unchanged.
Residual arithmetic and provider support require separate numerical and
capacity qualification; successful conversion alone establishes neither.

---

## Consumers

| Consumer | Fields used | How |
|----------|-------------|-----|
| Runtime (`inference_init`) | `constants_filename`, `constants[i].offset` | Reads `__metadata_blob`; opens constants file via `fs`; indexes into blob by offset |
| Any caller | all | Reads `__metadata_json` via `inference_get_metadata_json()` |

---

## Design Principles

**Durable serialized metadata goes through this schema.** Add a schema field
when information must be inspected by the loader or external consumers, or
must remain available independently of generated code. Compile-time values
used only to form calls inside the generated model may remain an internal
lowering contract, but must be documented by the owning design and must not be
described as part of `__metadata_blob`.

---

## Related Documents

- [constant-handling-design.md](constant-handling-design.md) — how `constants_filename`, `sizes`, and `offsets` fields are produced and consumed
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) — generated-code-only pool attributes and runtime calls
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — JIT loader contract; `inference_init` public interface
- [compilation-options.md](compilation-options.md) — `constants_file` compilation option
