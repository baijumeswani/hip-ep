<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Supported operations

This reference summarizes ONNX operations handled by the built-in ONNX-to-HIP pipeline. Support may be provided by a vendor library, a custom HIP kernel, decomposition into other operations, or standard MLIR tensor transformations.

The conversion registrations in `lib/Conversion/OnnxToHip/OnnxToHip.cpp` and the corresponding LIT/numeric tests are the source of truth. Individual operations may have data-type, rank, attribute, or dynamic-shape restrictions that are documented in their conversion and test files.

### Paged attention qualification

`com.microsoft::PagedAttention` currently accepts packed FP16 QKV, head size 64,
grouped query heads, full causal attention, and fused non-interleaved RoPE with
FP16 cosine/sine tables. Its int32 cumulative lengths, past lengths and block
table are device-resident. FP16-cache sequence length is limited to 4,096 tokens; page size
must be a power of two from 16 through 4,096. A second route accepts separate
packed FP16 or FP32 Q/K/V with head size 256 and pre-applied RoPE; cosine/sine inputs
must then be absent. Both routes accept an optional `int32[3]` attention
metadata hint, while validating the authoritative device sequence/page metadata.
The separate-input route also accepts signed INT8 caches with positive finite
FP32 per-channel scales. Optional residual8 stores a signed INT8 correction plane
after the high plane, doubling the last cache dimension; readers reconstruct
`scale * (high + (residual + 128) / 256)`. FP32 activations are not narrowed before
attention. The INT8-cache context ceiling is 262,144 tokens. Local windows, sinks,
QK normalization inputs, softcap, and rotated KV are rejected rather
than silently approximated.

The caller must bind each present cache to the corresponding past allocation.
Before mutation, a validation kernel checks sequence/page bounds, metadata
extents, duplicate pages, and cross-request writes that would overlap a peer's
live state. Shared read-only prefix pages are allowed. Malformed metadata returns
an execution error without modifying caches; the same session can then run valid
inputs.

The output-allocator execution boundary drains its inference stream before returning,
including when the compiled function fails or throws, and clears its borrowed
allocation callback before leaving scope. It does not fence unrelated streams.
A failed drain prints a diagnostic and
aborts directly: fatal logging alone is insufficient because a host logger can
throw and turn it into a recoverable error. Linux numeric contract tests use a
test-only interposer to verify this fail-stop behavior without launching a
destructive GPU fault.
Swish launch failures are retained in that session status, including their
device-failure classification; generated graph calls discard individual
operator return values.

`SkipSimplifiedLayerNormalization` accumulates and normalizes its residual sum
in FP32, rounding only when storing FP16 outputs. Requesting the optional residual
output does not change the normalized output; scalar and packed paths are tested
with and without bias against the ORT CPU implementation.

### Bounded FP32 quantized-cache attention

The separate-QKV, per-channel INT8-cache route retains FP32 unpacking, score
GEMMs, online softmax and value GEMMs. Its optional eight residual bits make
each cache value sixteen physical bits; unpacking never rounds through FP16.
Aligned cache data, scales and output use four-channel vector loads/stores;
unaligned buffers retain the scalar path with identical arithmetic.

Query tiles use the actual packed token count, capped at one for single-token
decode and 512 otherwise. Up to 32 packed query tokens use 16,384-key tiles;
larger prefills use 3,072-key tiles. Tiles shrink as needed to preserve the
existing 256 MiB scratch ceiling, and unsupported geometry fails before cache
mutation. FP32 softmax uses wave reductions plus a bounded cross-wave reduction;
small-row, long-key tiles use more threads without changing precision.
Ragged verification, page boundaries, cache aliases and invalid-metadata
atomicity remain covered by the existing numeric tests.

### Experimental Qwen hybrid target operators

The target-only hybrid path additionally implements these narrow contracts:

These are packed/variable-length extensions, not replacements for the
upstream dense operators with the same ONNX names. They lower to
`hip.varlen_gated_delta_net`, `hip.hybrid_gated_rms_norm`, and
`hip.packed_mrotary_embedding`; dense forms retain their existing conversion
and runtime entrypoints. In particular, the upstream dense GatedRMSNorm
continues to support sigmoid gating.

| Operation (`com.microsoft`) | Contract |
|---|---|
| VarlenCausalConvWithState | Matching FP16 or FP32 `[tokens, channels]`, kernel-4 depthwise weights, bias, SiLU, int32 cumulative lengths, and same-type `[batch, channels, 3]` carry. |
| GatedDeltaNet | Matching FP16 or FP32 Q/K/V, head size 128, value-head count divisible by key-head count, Qwen decay and sigmoid beta, FP32 V-major `[batch, value_heads, 128, 128]` recurrent state. Q/K L2 normalization uses the ORT operator's `1e-12` epsilon. |
| GatedRMSNorm | Matching FP16 or FP32 input/gate with a statically known common width, 128-element groups, FP32 variance/SiLU gating arithmetic, and a positive finite epsilon. Sigmoid gating is unsupported. |
| MRotaryEmbedding | Packed FP32 `[1, tokens, heads * 256]`, int64 `[3, 1, tokens]` positions, partial rotary width 64, unit scale, layout 1, sections `[11, 11, 10]`, and FP32 caches of statically known width 32. Remaining head dimensions pass through unchanged. |

Convolution and recurrence validate packed lengths before any output/state
write; MRoPE validates every position before writing. Dynamic output extents are
checked against the inputs. Convolution and recurrent present state may alias
past state, although the transactional Engine uses separate staged output banks.
These kernels retain only per-channel convolution carry and per-wave recurrent
rows, not a quadratic token workspace. Unsupported BF16 or transition-capture
forms are rejected rather than reinterpreted as FP16.

Compiled numerical/failure-path operator tests and real fixed-state
rollback/cancellation checks do not qualify end-to-end model accuracy. Full
hybrid model accuracy remains unqualified until independent long-prompt
matching-weight logit checks pass.

## Runtime-backed operations

| Operation | Backend or lowering |
|---|---|
| Conv | Custom HIP kernel. Rank-3 (NCL) is rewritten to a unit-height 2D conv; rank-4 (NCHW) and rank-5 (NCDHW) lower directly |
| NhwcConv (`com.microsoft`) | Rank-4 NHWC activations and `[M, kH, kW, C/group]` weights are transposed into `hip.conv` and the result is transposed back. Static weights are permuted while lowering |
| ConvTranspose | Custom HIP kernel |
| MatMul | hipBLASLt |
| Einsum | Static binary contraction decomposed to Transpose + MatMul (hipBLASLt). An equation with no contracted index, such as `bc,d->bcd`, is a reshape plus `hip.mul` when the element type is f16 or f32 and the rank is at most 4 |
| Gemm | hipBLASLt |
| Transpose | Custom HIP kernel |
| Mul | Custom HIP kernel |
| Add | Custom HIP kernel |
| ImageScaler | Decomposed to `hip.mul`, or `hip.add` then `hip.mul` when the per-channel bias is non-zero. NCHW only; bias broadcasts as `1xCx1x1`. |
| Sum | Decomposed to Add. Variadic elementwise sum with NumPy broadcast; same rank and dtype limits as Add |
| Softmax | Custom HIP kernel |
| Sigmoid | Custom HIP kernel |
| Tanh | Custom HIP kernel |
| Softplus | Custom HIP kernel (f32/f16) |
| Gelu | Custom HIP kernel; `com.microsoft` Gelu is the erf form of `onnx.Gelu` |
| Swish | Custom HIP kernel; `com.microsoft` QuickGelu is this formula with alpha default 1.702 |
| ScaledSiLU (`com.microsoft`) | Decomposed to Mul + Swish (alpha 1). f16/f32 |
| GatedAdd (`com.microsoft`) | Decomposed to Mul + Add. f16/f32, rank 1-4 |
| SwiGLU (`hip.sigmoid` followed by two `hip.mul`) | Fused by `hip-fusion-transform` into one custom HIP kernel when the tensors are identical f16/f32/bf16/f64, each dynamic axis traces to the same runtime extent, and the intermediates have no other value uses |
| BiasGelu (`com.microsoft`) | Custom HIP kernel |
| FastGelu (`com.microsoft`) | Custom HIP kernel |
| Reciprocal | Custom HIP kernel |
| Sqrt | Custom HIP kernel |
| Exp | Custom HIP kernel |
| Log | Custom HIP kernel |
| Pow | Decomposed to Mul / Sqrt / Reciprocal for supported constant scalar exponents. Other constant scalar exponents use a custom HIP kernel |
| Sub | Custom HIP kernel |
| Cast | Custom HIP kernel |
| CastLike | Simplified to Cast |
| Ceil | Custom HIP kernel |
| Round | Custom HIP kernel |
| Floor | Custom HIP kernel |
| Neg | Custom HIP kernel |
| Equal | Custom HIP kernel |
| Not | Custom HIP kernel |
| IsNaN | Custom HIP kernel |
| And | Custom HIP kernel |
| Or | Custom HIP kernel |
| Abs | Custom HIP kernel |
| Cos | Custom HIP kernel |
| Erf | Custom HIP kernel |
| Sin | Custom HIP kernel |
| Atan | Custom HIP kernel |
| Div | Custom HIP kernel |
| Mod | Custom HIP kernel |
| Sign | Custom HIP kernel |
| Where | Custom HIP kernel |
| Less | Custom HIP kernel |
| Greater | Decomposed to `Less(B, A)` |
| GreaterOrEqual | Decomposed to `Not(Less(A, B))` |
| LessOrEqual | Decomposed to `Not(Less(B, A))` |
| Min | Custom HIP kernel |
| Max | Custom HIP kernel |
| ReduceSum | Custom HIP kernel |
| ReduceMax | Custom HIP kernel |
| ReduceMin | Custom HIP kernel |
| ReduceProd | Custom HIP kernel |
| ReduceMean | Custom HIP kernel |
| CumSum | Custom HIP kernel |
| Pad | Custom HIP kernel. The opset<11 form, with `pads` and the fill `value` as attributes, is rewritten to the operand form first |
| Tile | Custom HIP kernel |
| Expand | Custom HIP kernel |
| GatherND | Custom HIP kernel |
| ScatterND | Custom HIP kernel (reductions: none / add / mul / min / max) |
| ScatterElements | Custom HIP kernel (reductions: none / add / mul / min / max) |
| Range | Custom HIP kernel |
| Size | Custom HIP kernel; folds to a constant for static shapes |
| NonZero | Custom HIP kernel |
| Gather | Custom HIP kernel |
| GatherElements | Custom HIP kernel |
| TopK | Custom HIP kernel |
| RandomNormalLike | Custom HIP kernel. Output f16, bf16, f32, and f64; rank 0–8. The input contributes only its shape, including dynamic dimensions. `seed` makes the fill reproducible inside hip-ep; the generator is not bit-identical to another runtime. An absent seed comes from the clock. |
| ArgMax | Custom HIP kernel. f16, bf16, f32, f64, i8, ui8, i16, ui16, i32, and i64; rank 1–8. ui32 and ui64 are not supported. An empty reduction axis fails. NaN outranks every number; the first NaN wins unless select_last_index keeps the last. |
| Compress | Custom HIP kernel; a dynamic selected extent is scanned and read back before allocation |
| OneHot | Custom HIP kernel |
| LayerNormalization | Custom HIP kernel |
| InstanceNormalization | Custom HIP kernel |
| BatchNormalization | Custom HIP kernel; inference only (`training_mode` stays on ONNX) |
| GroupNorm (`com.microsoft`) | Custom HIP kernel; optional SiLU, NCHW or NHWC |
| SkipLayerNormalization (`com.microsoft`) | Decomposed to Add + LayerNormalization, including the optional input bias |
| EmbedLayerNormalization (`com.microsoft`) | Decomposed to Gather + Add + LayerNormalization. f16/f32 tables, i32 indices. A missing position-id input uses `0 .. sequence-1` when the sequence length is static. `mask_index` is the count of ones along the sequence; with no mask it is zeros |
| RMSNormalization | Custom HIP kernel |
| SimplifiedLayerNormalization | Custom HIP kernel |
| SkipSimplifiedLayerNormalization (`com.microsoft`) | Custom HIP kernel, add and norm fused |
| BranchwiseRMSNorm (`com.microsoft`) | Lowered to RMSNormalization over the branch axis. f16/f32; the scale input is required |
| GatedRMSNorm (`com.microsoft`) | Custom HIP kernel; silu, swish or sigmoid gate. f16/f32 |
| EngramGate (`com.microsoft`) | Custom HIP kernel, including the optional `gated_value_normed` output. f16/f32 |
| HyperConnectionPreMix (`com.microsoft`) | Custom HIP kernel; folds a preceding Sigmoid. f16/f32 |
| HyperConnectionPostMix (`com.microsoft`) | Custom HIP kernel; optional stream mix, fused with a following BranchwiseRMSNorm. f16/f32 |
| LpNormalization | Decomposed to Mul / ReduceSum / Sqrt / Div |
| RotaryEmbedding (`com.microsoft`) | Custom HIP kernel |
| RotaryEmbedding (`ai.onnx`) | Custom HIP kernel |
| MRotaryEmbedding (`com.microsoft`) | Custom HIP kernel. Rank-3 input, `interleaved=0`, `is_packed_batching=0`, f16/f32 |
| GroupQueryAttention (`com.microsoft`) | Custom HIP kernels and hipBLASLt |
| SparseAttentionIndexer (`com.microsoft`) | Custom HIP kernels, wave32 (RDNA) only. `qsa` policy with packed QK, head size 128, up to 8 heads, `token_budget / compress_ratio` ≤ 512, f16/f32. With `past_sequence_length`, `past_key` / `present_key` are max-capacity caches that may share one buffer. An MRoPE cos/sin table subgraph with constant section masks is folded into the op, which then reads the base tables through the position ids. The exported per-row counts subgraph (`ReduceSum(Where(selected_indices >= 0, 1, 0))`) is folded into an extra op output. Without a mask, an indexer whose selection only feeds DynamicSparseAttention is paired with it: rows that select every position skip scoring, and the rest are listed in block order |
| DynamicSparseAttention (`com.microsoft`) | Custom HIP kernel (WMMA), wave32 (RDNA) only. `selected_only` over the main KV cache, head size 128 or 256, up to 16 query heads per KV head, f16; the op's own rotary, q/k norm and `head_sink` inputs are not supported. A surrounding packed-QKV `Split`, per-head `SimplifiedLayerNormalization` + `MRotaryEmbedding` of q/k and `Mul(output, Sigmoid(gate))` are folded into the kernel. Paired with its indexer, an unsplit call runs the dense rows of a long enough prompt on the GQA flash prefill |
| PagedAttention (`com.microsoft`) | FP16 packed rank-2 QKV with head size 64 and fused non-interleaved RoPE, or matching FP16/FP32 separate Q/K/V with head size 256 and pre-applied RoPE. FP16 or signed INT8 per-channel caches (optionally residual8) must be caller-bound as identical past/present GPU buffers. Int32 cumulative lengths, past lengths and block table reside on device. Context is limited to 4096 tokens for FP16 caches or 262144 for INT8 caches, with power-of-two page sizes 16–4096. Metadata is validated before any cache write. QK norm inputs, sinks, windows and rotated caches are rejected. |
| MultiHeadAttention (`com.microsoft`) | Lowered to GroupQueryAttention or decomposed hipBLASLt/custom-kernel paths |
| Attention (`com.microsoft`) | Fused QKV split into GroupQueryAttention. Equal Q/K/V sizes may be inferred from a `[H, 3H]` weight. Bidirectional and causal, with static or dynamic batch and sequence. A rank-1 length or rank-2 padding mask becomes an additive attention bias. |
| Attention (`ai.onnx`, opset 23/24) | Lowered to GroupQueryAttention for supported rank-3/rank-4, causal/masked, output, and KV-cache forms. With `is_causal = 1`, a mask built as `Where(And(key padding, causal [, window]), 0, fill)` keeps the fused flash kernels: only the key-padding conjunct is kept, as a per-key bias. The kernels clamp that bias at -65504, so a `-inf` fill (including fp32 `-3.4e38` cast to fp16) masks a key the way -65504 does and a fully masked row stays finite. The causal conjunct is dropped only when both of its sides read one position `CumSum` or `Range` with the same constant offset. Fused prefill applies that bias at head dim 128; other head dims run the decomposed path. On a sliding-window layer with a separate past buffer, `present` rows older than the window (rounded down to 64 keys) are not written, since no later step reads them |
| MatMulNBits (`com.microsoft`) | Raw blockwise weights; CUDA `weight_prepacked` layouts are rejected. Custom HIP kernels, with hipBLASLt for eligible wave64 INT4 prefill using one reusable FP16 weight scratch per session. Set `HIPDNN_EP_MATMUL_DP4A=0` before initialization to preserve FP16 activations instead of default-on narrow-M activation requantization. |
| MatMulBnb4 (`com.microsoft`) | HIP dequant of FP4 (`quant_type` 0) or NF4 (`quant_type` 1) into `[N, K]`, then `hip.matmul` with `transB = 1`. The first packed value is the high nibble. Static shapes pull that dequant into the rocMLIR matmul kernel. A dynamic activation leaves the dequant on HIP and the GEMM on hipBLASLt. Block size 16, 32, 64, 128, or 256. `transB` must be 1 and `training_mode` must be 0 |
| QMoE (`com.microsoft`) | Custom HIP kernel. FP16 activations with 2-bit or 4-bit expert weights in the schema `[E, N, K/pack]` layout. Block size must be a power of two and at least 32. SwiGLU intermediates beyond the fp16 range do not overflow, as in the CPU reference: decode keeps them in fp32, and prefill stores such a row divided by a power of two, which is exact except for elements pushed into fp16 subnormals. The fc2 bias of a scaled row is added after the rescale. Sigmoid + correction-bias routing built outside the op (`TopK` over `Sigmoid(logits) + bias`, renormalized and scaled weights, two `ScatterElements` into the dense router inputs) is folded into the op and routed on the device from the fp32 logits |
| PagedAttention (`com.microsoft`) | FP16 packed rank-2 QKV, head size 64, grouped heads, causal attention and non-interleaved fused RoPE. FP16 `[blocks, page_size, kv_heads, 64]` caches must be caller-bound as identical past/present GPU buffers. Int32 cumulative lengths, past lengths and block table reside on device; rotary tables are FP16. Context is limited to 4096 tokens, with power-of-two page sizes 16–4096. Metadata is validated before any cache write. Separate Q/K/V, QK norm, sinks, windows and quantized/rotated caches are rejected. |
| GatherBlockQuantized (`com.microsoft`) | Custom HIP kernel |
| QuantizeLinear | Custom HIP kernel |
| DynamicQuantizeLinear | Decomposed to single-axis ReduceMin/ReduceMax, scale and zero-point math, and `hip.quantize_linear` with axis 0. A zero input range selects scale 1 |
| DequantizeLinear | Custom HIP kernel |
| ConvInteger | Decomposed to Cast, Sub, and `hip.conv`, then Cast back to i32 when `65025 * (C/group) * product(kernel_shape)` fits in the f32 mantissa. A longer reduction accumulates in i32 and is not rewritten to `hip.gemm`. Rank-3 NCL and rank-4 NCHW with a static or dynamic batch; channels, spatial dimensions, and weights must be static. Rank-3 is expanded to the shared 2D path. Scalar or per-output-channel weight zero point |
| MatMulInteger | Decomposed to Cast, Sub, and `hip.matmul`, then Cast back to i32 when `65025 * K` fits in the f32 mantissa. A longer K accumulates in i32. Rank 2–4. A static rank-2 B may be shared across dynamic batch and sequence dimensions, including a dynamic M. B's K and N stay static. An uninferred K on A still lowers and accumulates in i32. A per-row zero point requires a static M. A size-1 batch on A is expanded to the result batch. A rank-2 B stays one shared matrix, and an i32 `hip.matmul` passes batch stride 0 for it. Scalar or per-row / per-column zero point |
| QLinearConv | Custom HIP kernel. 8-bit grouped NCHW, optional int32 bias. `NOTSET` uses the pads attribute. `VALID`, `SAME_UPPER`, and `SAME_LOWER` are resolved to explicit pads. Separate from the W4A16 `hip.qconv` fusion |
| QLinearAdd (`com.microsoft`) | Decomposed to DequantizeLinear + Add + QuantizeLinear; per-tensor scales fuse to `hip.qadd` |
| QLinearMul (`com.microsoft`) | Decomposed to DequantizeLinear + Mul + QuantizeLinear; per-tensor scales fuse to `hip.qmul` |
| QLinearConcat (`com.microsoft`) | Decomposed to DequantizeLinear + Concat + QuantizeLinear |
| QLinearGlobalAveragePool (`com.microsoft`) | Decomposed to DequantizeLinear + GlobalAveragePool + QuantizeLinear. `channels_last` is transposed around the pool |
| QLinearAveragePool (`com.microsoft`) | Decomposed to DequantizeLinear + AveragePool + QuantizeLinear. `channels_last` is transposed around the pool. Rank 3–5, int8 or uint8. Omitted strides are 1. `y_zero_point` may be omitted |
| QLinearLeakyRelu (`com.microsoft`) | Decomposed to DequantizeLinear + LeakyRelu + QuantizeLinear. int8 or uint8. `alpha` defaults to 0.01. `Y_zero_point` may be omitted |
| QLinearMatMul | Custom HIP kernel. Rank-2 8-bit. A and Y scales and zero points are per-tensor. B's scale and zero point are per-tensor or one value per column. Separate from the `hip.qmatmul` QDQ fusion |
| LinearAttention (`com.microsoft`) | Custom HIP kernel |
| GatedDeltaNet (`com.microsoft`) | Custom HIP kernel. Packed QKV, `gate_activation=qwen`, `beta_activation=sigmoid`, `qk_l2_norm=1`, f16/f32; no `cu_seqlens` or state capture |
| CausalConvWithState (`com.microsoft`) | Custom HIP kernel |
| NGramHashMapping (`com.microsoft`) | Custom HIP kernel. int32/int64 |
| VarlenCausalConvWithState (`com.microsoft`) | Matching FP16/FP32 kernel-4 SiLU convolution with three-element carry; see hybrid target contracts above. |
| GatedDeltaNet (`com.microsoft`) | Matching FP16/FP32 Qwen gated-delta with FP32 V-major recurrent state; no transition capture. |
| GatedRMSNorm (`com.microsoft`) | Matching FP16/FP32 grouped width-128 SiLU-gated RMS normalization. |
| MRotaryEmbedding (`com.microsoft`) | Packed FP32 head-256 partial rotary embedding, layout 1 and sections `[11,11,10]`. |
| Relu | Decomposed to Max |
| LeakyRelu | Custom HIP kernel |
| PRelu | Custom HIP kernel. Slope broadcasts onto X; f16/bf16/f32/f64 |
| LRN | Custom HIP kernel. Channel window, rank >= 2, f16/bf16/f32/f64 |
| Trilu | Custom HIP kernel. Keeps the upper or lower triangle of the last two dimensions; `k` must be a constant scalar |
| Clip | Decomposed to Max + Min |
| HardSigmoid | Decomposed to Mul + Add + Clip (`alpha` defaults to 0.2, `beta` to 0.5). f16 and f32 only; bf16 and f64 are left to another EP, since the emitted elementwise ops have no runtime path for them |
| MaxPool | Custom HIP kernel (f16/bf16/f32/f64, and i8/ui8) |
| AveragePool | Custom HIP kernel |
| LpPool | Custom HIP kernel |
| Resize | Custom HIP kernel. nearest and linear on f16/bf16/f32/f64 and on int8/uint8. Integer linear interpolates the stored codes and truncates toward zero. A dynamic extent comes from a constant `scales` vector, or from a `sizes` vector read back to a host index when `scales` is absent |
| Upsample | Lowered through Resize (`asymmetric`, nearest `floor`). Schema 9 passes `scales` as an input; opset 7–8 stores them as an f32 attribute |
| GridSample | Custom HIP kernel |
| RoiAlign | Custom HIP kernel. avg and max modes, `half_pixel` and `output_half_pixel` (opset 10 has no attribute and uses `output_half_pixel`), f16/bf16/f32/f64. An out-of-range batch index yields zeros rather than an error. The number of regions and every input extent may be dynamic |
| GlobalAveragePool | Custom HIP kernel |
| GlobalMaxPool | Custom HIP kernel |
| GlobalLpPool | Custom HIP kernel |

## Control flow

| Operation | Implementation |
|---|---|
| If | Outlined then/else functions with runtime branch dispatch. The condition is a single-element `i1`, `i8`, or `ui8` tensor, including a unit shape such as `tensor<1xui8>` |
| Loop | Outlined loop body with runtime loop dispatch |
| Scan | Outlined body with runtime scan dispatch. Supports ranked scan inputs, state variables, outer-scope captures, dynamic or zero sequence lengths, and stacked outputs. Input/output axes must be 0 and directions must be forward |

## Compiler-optimized operations

These operations are handled through standard MLIR transformations and generally do not require operation-specific runtime backend support.

| Operation | Implementation | Notes |
|---|---|---|
| Reshape | `tensor.expand_shape` / `tensor.collapse_shape` | Metadata-only where representable; dynamic dimensions are supported |
| Unsqueeze | `tensor.expand_shape` | Inserts size-one axes |
| Squeeze | `tensor.collapse_shape` | Removes size-one axes |
| Split | `tensor.extract_slice` | Produces tensor slices that bufferize to views |
| Slice | `tensor.extract_slice` or `hip.slice` | Constant positive-stride forms decompose to tensor slices; runtime indices or negative steps use the runtime path. The opset<10 form, with starts/ends/axes as attributes, is rewritten to the operand form first |
| Concat | `tensor.empty` + `tensor.insert_slice` | Bufferizes to destination subviews and copies |
| Shape | `tensor.dim` + `tensor.from_elements` | Static shapes fold to constants; dynamic dimensions remain runtime SSA |
| Constant | `arith.constant` or external constants file | Large values are externalized |
| ConstantOfShape | `arith.constant` when foldable | Produces a splat constant for constant shape inputs |
| Identity | SSA value forwarding | No runtime operation |
| Flatten | `tensor.collapse_shape` and, where needed, `tensor.expand_shape` | Metadata-only where representable |
| DepthToSpace | `tensor.expand_shape`, `hip.transpose`, `tensor.collapse_shape` | Rank-4 NCHW. `blocksize >= 1`, `C` divisible by `blocksize^2`, and output `[N, C / blocksize^2, H * blocksize, W * blocksize]`. Modes `DCR` (default) and `CRD`. `blocksize == 1` is the input |
| SpaceToDepth | `tensor.expand_shape`, `hip.transpose`, `tensor.collapse_shape` | Rank-4 NCHW. `blocksize >= 1`, `H` and `W` divisible by `blocksize`, and output `[N, C * blocksize^2, H / blocksize, W / blocksize]`. Inverse of DepthToSpace `DCR`. `blocksize == 1` is the input |

## Fusion and preprocessing

The frontend also recognizes or simplifies selected patterns before operation conversion, including approximate/FastGelu chains, Erf-based Gelu chains, supported Pow forms, LpNormalization decompositions, CastLike simplification, and model-exporter cleanup patterns.
