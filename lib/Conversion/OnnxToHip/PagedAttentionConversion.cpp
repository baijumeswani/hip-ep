/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir::hip {
namespace {

struct PagedAttentionToHip : RewritePattern {
  explicit PagedAttentionToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto name = op->getAttrOfType<StringAttr>("function_name");
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!name || name.getValue() != "PagedAttention" || !domain ||
        domain.getValue() != "com.microsoft")
      return failure();

    const auto reject = [&](StringRef reason) {
      return op->emitError("HIP FP16 PagedAttention: ") << reason;
    };
    if (op->getNumOperands() < 10 || op->getNumOperands() > 17 ||
        op->getNumResults() != 3)
      return reject("expected packed QKV, rotary caches and three outputs");
    for (unsigned index : {1u, 2u, 10u, 11u, 12u, 13u, 14u, 15u}) {
      if (index < op->getNumOperands() &&
          !isa<NoneType>(op->getOperand(index).getType()))
        return reject("separate QKV, sinks, QK norm and quantized caches are "
                      "not supported");
    }
    const auto integer = [&](StringRef key, int64_t fallback) {
      auto attr = op->getAttrOfType<IntegerAttr>(key);
      return attr ? attr.getInt() : fallback;
    };
    const auto floating = [&](StringRef key, float fallback) {
      auto attr = op->getAttrOfType<FloatAttr>(key);
      return attr ? attr.getValueAsDouble() : fallback;
    };
    const int64_t heads = integer("num_heads", 0);
    const int64_t kvHeads = integer("kv_num_heads", 0);
    if (heads <= 0 || heads > 65535 || kvHeads <= 0 || kvHeads > heads ||
        heads % kvHeads != 0 || integer("is_causal", 1) != 1 ||
        integer("do_rotary", 0) != 1 || integer("rotary_interleaved", 0) != 0 ||
        integer("local_window_size", -1) != -1 || floating("softcap", 0) != 0)
      return reject("requires grouped heads, full causal attention and "
                    "non-interleaved fused RoPE");
    for (StringRef key :
         {"k_quant_type", "v_quant_type", "qk_rotation", "v_rotation"}) {
      if (auto attr = op->getAttrOfType<StringAttr>(key);
          attr && attr.getValue() != "NONE")
        return reject("cache quantization/rotation is not supported");
    }
    const auto typeAt = [&](unsigned index) {
      return dyn_cast<RankedTensorType>(op->getOperand(index).getType());
    };
    const auto queryType = typeAt(0);
    const auto cacheType = typeAt(3);
    const auto valueType = typeAt(4);
    const auto cumulativeType = typeAt(5);
    const auto pastType = typeAt(6);
    const auto tableType = typeAt(7);
    const auto cosType = typeAt(8);
    const auto sinType = typeAt(9);
    if (!queryType || queryType.getRank() != 2 ||
        !queryType.getElementType().isF16() ||
        queryType.getDimSize(1) != (heads + 2 * kvHeads) * 64 || !cacheType ||
        cacheType.getRank() != 4 || !cacheType.getElementType().isF16() ||
        cacheType.getDimSize(2) != kvHeads || cacheType.getDimSize(3) != 64 ||
        valueType != cacheType || !cumulativeType ||
        cumulativeType.getRank() != 1 ||
        !cumulativeType.getElementType().isInteger(32) || !pastType ||
        pastType.getRank() != 1 || !pastType.getElementType().isInteger(32) ||
        !tableType || tableType.getRank() != 2 ||
        !tableType.getElementType().isInteger(32) || !cosType ||
        cosType.getRank() != 2 || !cosType.getElementType().isF16() ||
        cosType.getDimSize(1) != 32 || sinType != cosType)
      return reject("requires FP16 packed QKV/cache with head size 64, FP16 "
                    "rotary tables and int32 metadata");

    auto context = getContextArg(op, rewriter);
    if (failed(context))
      return reject("missing runtime context");
    SmallVector<Type> types;
    SmallVector<Value> operands{*context};
    for (unsigned index : {0u, 3u, 4u, 5u, 6u, 7u, 8u, 9u})
      operands.push_back(op->getOperand(index));
    for (unsigned i = 0; i < 3; ++i) {
      auto type = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!type ||
          (i == 0 && (type.getRank() != 2 || !type.getElementType().isF16() ||
                      type.getDimSize(1) != heads * 64 ||
                      (!type.isDynamicDim(0) && !queryType.isDynamicDim(0) &&
                       type.getDimSize(0) != queryType.getDimSize(0)))) ||
          (i != 0 && type != cacheType))
        return reject("present cache shape must match past cache");
      types.push_back(type);
      operands.push_back(createEmptyTensor(rewriter, op->getLoc(), type,
                                           op->getOperand(i == 0 ? 0 : i + 2)));
    }
    OperationState state(op->getLoc(), "hip.paged_attention");
    state.addOperands(operands);
    state.addTypes(types);
    state.addAttribute("num_heads", rewriter.getI64IntegerAttr(heads));
    state.addAttribute("kv_num_heads", rewriter.getI64IntegerAttr(kvHeads));
    state.addAttribute("scale",
                       rewriter.getF32FloatAttr(floating("scale", 0.125f)));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

} // namespace

void populatePagedAttentionConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<PagedAttentionToHip>(ctx);
}

} // namespace mlir::hip
