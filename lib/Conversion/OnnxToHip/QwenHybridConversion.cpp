/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir::hip {
namespace {

bool tensorMatches(Value value, Type element, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getElementType() != element ||
      type.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [index, size] : llvm::enumerate(shape))
    if (size >= 0 && type.getDimSize(index) != size)
      return false;
  return true;
}

bool compatibleShape(Value lhs, Value rhs) {
  auto a = dyn_cast<RankedTensorType>(lhs.getType());
  auto b = dyn_cast<RankedTensorType>(rhs.getType());
  if (!a || !b || a.getElementType() != b.getElementType() ||
      a.getRank() != b.getRank())
    return false;
  for (int64_t i = 0; i < a.getRank(); ++i)
    if (!a.isDynamicDim(i) && !b.isDynamicDim(i) &&
        a.getDimSize(i) != b.getDimSize(i))
      return false;
  return true;
}

struct QwenHybridToHip : RewritePattern {
  explicit QwenHybridToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto name = op->getAttrOfType<StringAttr>("function_name");
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!name || !domain || domain.getValue() != "com.microsoft")
      return failure();
    StringRef kind = name.getValue();
    if (kind != "VarlenCausalConvWithState" && kind != "GatedDeltaNet" &&
        kind != "GatedRMSNorm" && kind != "MRotaryEmbedding")
      return failure();
    const auto reject = [&](StringRef reason) {
      return op->emitError("HIP ") << kind << ": " << reason;
    };
    const auto integer = [&](StringRef key, int64_t fallback) {
      auto attr = op->getAttrOfType<IntegerAttr>(key);
      return attr ? attr.getInt() : fallback;
    };
    const auto floating = [&](StringRef key, double fallback) {
      auto attr = op->getAttrOfType<FloatAttr>(key);
      return attr ? attr.getValueAsDouble() : fallback;
    };
    const auto stringIs = [&](StringRef key, StringRef expected,
                              StringRef fallback) {
      auto attr = op->getAttrOfType<StringAttr>(key);
      return (attr ? attr.getValue() : fallback) == expected;
    };
    const auto f32 = rewriter.getF32Type();
    const auto i32 = rewriter.getI32Type();
    if (op->getNumOperands() == 0)
      return reject("requires an activation input");
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!inputType || (!inputType.getElementType().isF16() &&
                       !inputType.getElementType().isF32()))
      return reject("requires FP16 or FP32 activation tensors");
    const auto activationType = inputType.getElementType();
    SmallVector<unsigned> anchors;
    SmallVector<NamedAttribute> attributes;
    StringRef target;

    if (kind == "VarlenCausalConvWithState") {
      if (op->getNumOperands() != 5 || op->getNumResults() != 2)
        return reject("requires five inputs and two outputs; state windows "
                      "are not supported");
      auto x = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!x || x.getRank() != 2 || x.getDimSize(1) <= 0)
        return reject("requires a statically known channel count");
      const int64_t channels = x.getDimSize(1);
      if (!tensorMatches(op->getOperand(0), activationType, {-1, channels}) ||
          !tensorMatches(op->getOperand(1), activationType, {channels, 1, 4}) ||
          !tensorMatches(op->getOperand(2), i32, {-1}) ||
          !tensorMatches(op->getOperand(3), activationType, {channels}) ||
          !tensorMatches(op->getOperand(4), activationType,
                         {-1, channels, 3}) ||
          !compatibleShape(op->getResult(0), op->getOperand(0)) ||
          !compatibleShape(op->getResult(1), op->getOperand(4)) ||
          !stringIs("activation", "silu", ""))
        return reject(
            "requires matching FP16/FP32 packed kernel-4 convolution, SiLU, "
            "int32 cumulative lengths and three-element carry");
      target = "hip.varlen_causal_conv";
      anchors = {0, 4};
    } else if (kind == "GatedDeltaNet") {
      if (op->getNumOperands() != 9 || op->getNumResults() != 2)
        return reject("requires nine inputs and two outputs; state windows "
                      "are not supported");
      auto q = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto v = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
      if (!q || !v || q.getRank() != 3 || v.getRank() != 3)
        return reject("requires rank-three packed Q/K/V");
      const int64_t kh = q.getDimSize(1), vh = v.getDimSize(1);
      const double scale = floating("scale", 0);
      if (kh <= 0 || vh <= 0 || kh > vh || vh % kh != 0 ||
          !std::isfinite(scale) || scale < 0 ||
          !tensorMatches(op->getOperand(0), activationType, {-1, kh, 128}) ||
          !tensorMatches(op->getOperand(1), activationType, {-1, kh, 128}) ||
          !tensorMatches(op->getOperand(2), activationType, {-1, vh, 128}) ||
          !tensorMatches(op->getOperand(3), i32, {-1}) ||
          !tensorMatches(op->getOperand(4), f32, {-1, vh}) ||
          !tensorMatches(op->getOperand(5), f32, {-1, vh}) ||
          !tensorMatches(op->getOperand(6), f32, {-1, vh, 128, 128}) ||
          !tensorMatches(op->getOperand(7), f32, {vh}) ||
          !tensorMatches(op->getOperand(8), f32, {vh}) ||
          !compatibleShape(op->getResult(0), op->getOperand(2)) ||
          !compatibleShape(op->getResult(1), op->getOperand(6)) ||
          !stringIs("update_rule", "gated_delta", "gated_delta") ||
          !stringIs("gate_activation", "qwen", "none") ||
          !stringIs("beta_activation", "sigmoid", "none") ||
          integer("qk_l2_norm", 0) != 1 || integer("chunk_size", 64) != 64)
        return reject(
            "requires Qwen gated-delta, matching normalized FP16/FP32 Q/K, "
            "head size 128 and FP32 V-major recurrence");
      target = "hip.gated_delta_net";
      anchors = {2, 6};
      attributes = {
          rewriter.getNamedAttr("key_heads", rewriter.getI64IntegerAttr(kh)),
          rewriter.getNamedAttr("value_heads", rewriter.getI64IntegerAttr(vh)),
          rewriter.getNamedAttr(
              "scale", rewriter.getF32FloatAttr(
                           scale == 0 ? 1.0 / std::sqrt(128.0) : scale))};
    } else if (kind == "GatedRMSNorm") {
      if (op->getNumOperands() != 3 || op->getNumResults() != 1)
        return reject("requires input, scale, gate and one output");
      auto x = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      double epsilon = floating("epsilon", 1e-5);
      if (!stringIs("activation", "silu", "silu") &&
          !stringIs("activation", "swish", "silu"))
        return reject("requires SiLU gating");
      if (!x || x.getRank() != 2 || x.getDimSize(1) <= 0 ||
          x.getDimSize(1) % 128 != 0 || !std::isfinite(epsilon) ||
          epsilon <= 0 ||
          !tensorMatches(op->getOperand(0), activationType,
                         {-1, x.getDimSize(1)}) ||
          !tensorMatches(op->getOperand(1), activationType, {128}) ||
          !tensorMatches(op->getOperand(2), activationType,
                         {-1, x.getDimSize(1)}) ||
          !compatibleShape(op->getOperand(0), op->getOperand(2)) ||
          !compatibleShape(op->getOperand(0), op->getResult(0)))
        return reject(
            "requires matching FP16/FP32 packed input/gate, group width 128 "
            "and positive finite epsilon");
      target = "hip.gated_rms_norm";
      anchors = {0};
      attributes.push_back(
          rewriter.getNamedAttr("epsilon", rewriter.getF32FloatAttr(epsilon)));
    } else {
      if (op->getNumOperands() != 4 || op->getNumResults() != 1)
        return reject("requires input, positions, two caches and one output");
      int64_t heads = integer("num_heads", 0);
      if (floating("scale", 1) != 1)
        return reject("requires unit rotary scale");
      auto section = op->getAttrOfType<ArrayAttr>("mrope_section");
      bool sectionsMatch = section && section.size() == 3;
      if (sectionsMatch)
        for (auto [index, attr] : llvm::enumerate(section)) {
          auto value = dyn_cast<IntegerAttr>(attr);
          sectionsMatch &= value && value.getInt() == (index == 2 ? 10 : 11);
        }
      if (heads <= 0 || heads > 65535 ||
          !tensorMatches(op->getOperand(0), f32, {1, -1, heads * 256}) ||
          !tensorMatches(op->getOperand(1), rewriter.getI64Type(),
                         {3, 1, -1}) ||
          !tensorMatches(op->getOperand(2), f32, {-1, 32}) ||
          !tensorMatches(op->getOperand(3), f32, {-1, 32}) ||
          !compatibleShape(op->getOperand(2), op->getOperand(3)) ||
          !compatibleShape(op->getOperand(0), op->getResult(0)) ||
          !sectionsMatch || integer("interleaved", 0) != 0 ||
          integer("rotary_embedding_dim", 0) != 64 ||
          integer("mrope_layout", 0) != 1 ||
          integer("is_packed_batching", 0) != 1)
        return reject("requires packed FP32 head-256 partial MRoPE with "
                      "layout 1 and sections [11,11,10]");
      target = "hip.mrotary_embedding";
      anchors = {0};
      attributes.push_back(rewriter.getNamedAttr(
          "num_heads", rewriter.getI64IntegerAttr(heads)));
    }

    auto context = getContextArg(op, rewriter);
    if (failed(context))
      return reject("missing runtime context");
    OperationState state(op->getLoc(), target);
    state.addOperands(*context);
    state.addOperands(op->getOperands());
    for (auto [result, anchor] : llvm::zip(op->getResults(), anchors)) {
      auto type = cast<RankedTensorType>(result.getType());
      state.addTypes(type);
      state.addOperands(createEmptyTensor(rewriter, op->getLoc(), type,
                                          op->getOperand(anchor)));
    }
    state.addAttributes(attributes);
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

} // namespace

void populateQwenHybridConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<QwenHybridToHip>(ctx);
}

} // namespace mlir::hip
