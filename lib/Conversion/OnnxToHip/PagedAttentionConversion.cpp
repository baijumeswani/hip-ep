/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

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
      return op->emitError("HIP PagedAttention: ") << reason;
    };
    if (op->getNumOperands() < 10 || op->getNumOperands() > 17 ||
        op->getNumResults() != 3)
      return reject("expected Q/K/V, paged caches and three outputs");
    const bool separate = !isa<NoneType>(op->getOperand(1).getType()) ||
                          !isa<NoneType>(op->getOperand(2).getType());
    const auto keyCache =
        dyn_cast<RankedTensorType>(op->getOperand(3).getType());
    const bool quantized = keyCache && keyCache.getElementType().isInteger(8);
    if (quantized && cast<IntegerType>(keyCache.getElementType()).isUnsigned())
      return reject("quantized caches must be signed INT8, not UINT8");
    for (unsigned index : {10u, 11u, 12u, 13u}) {
      if (index < op->getNumOperands() &&
          !isa<NoneType>(op->getOperand(index).getType()))
        return reject("sinks and QK norm are not supported");
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
    const int64_t residualBits = integer("kv_cache_residual_bits", 0);
    if ((residualBits != 0 && residualBits != 8) ||
        (residualBits != 0 && !quantized))
      return reject("residual bits must be 0, or 8 with signed INT8 caches");
    if (heads <= 0 || heads > 65535 || kvHeads <= 0 || kvHeads > heads ||
        heads % kvHeads != 0 || integer("is_causal", 1) != 1 ||
        integer("do_rotary", 0) != (separate ? 0 : 1) ||
        integer("rotary_interleaved", 0) != 0 ||
        integer("local_window_size", -1) != -1 || floating("softcap", 0) != 0)
      return reject("requires grouped heads, full causal attention and "
                    "supported RoPE layout");
    if (auto attr = op->getAttrOfType<StringAttr>("kv_cache_layout");
        attr && attr.getValue() != "SEPARATE")
      return reject("requires separate K/V cache storage");
    for (StringRef key : {"qk_rotation", "v_rotation"}) {
      if (auto attr = op->getAttrOfType<StringAttr>(key);
          attr && attr.getValue() != "NONE")
        return reject("cache quantization/rotation is not supported");
    }
    for (StringRef key : {"k_quant_type", "v_quant_type"}) {
      const auto attr = op->getAttrOfType<StringAttr>(key);
      if (quantized ? (!attr || attr.getValue() != "PER_CHANNEL")
                    : (attr && attr.getValue() != "NONE"))
        return reject("INT8 caches require static PER_CHANNEL scales");
    }
    for (StringRef key : {"k_cache_dtype", "v_cache_dtype"}) {
      if (auto attr = op->getAttrOfType<StringAttr>(key);
          attr && !attr.getValue().empty() &&
          attr.getValue() != (quantized ? "int8" : "float16"))
        return reject("cache dtype attribute must match FP16 or signed INT8 "
                      "storage: ")
               << key << " = " << attr;
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
    const int64_t headSize = separate ? 256 : 64;
    for (unsigned index : {14u, 15u}) {
      const bool present = index < op->getNumOperands() &&
                           !isa<NoneType>(op->getOperand(index).getType());
      if (quantized) {
        const auto type = present ? typeAt(index) : RankedTensorType{};
        if (!separate || !queryType || !queryType.getElementType().isF32() ||
            !type || !type.getElementType().isF32() ||
            type.getShape() != ArrayRef<int64_t>({kvHeads, 1, 256}))
          return reject("INT8 requires separate FP32 Q/K/V and FP32 "
                        "[kv_heads,1,256] K/V scales");
      } else if (present) {
        return reject("unquantized caches must not supply scales");
      }
    }
    const auto validMatrix = [&](RankedTensorType type, int64_t width) {
      return type && type.getRank() == 2 && queryType &&
             type.getElementType() == queryType.getElementType() &&
             type.getDimSize(1) == width;
    };
    if (separate && (!validMatrix(typeAt(1), kvHeads * headSize) ||
                     !validMatrix(typeAt(2), kvHeads * headSize) ||
                     !isa<NoneType>(op->getOperand(8).getType()) ||
                     !isa<NoneType>(op->getOperand(9).getType())))
      return reject(
          "separate Q/K/V requires head size 256 and pre-applied RoPE");
    if (op->getNumOperands() == 17 &&
        !isa<NoneType>(op->getOperand(16).getType())) {
      auto metadata = typeAt(16);
      if (!metadata || metadata.getRank() != 1 ||
          !metadata.getElementType().isInteger(32) ||
          metadata.getDimSize(0) != 3)
        return reject("optional attention metadata must be int32[3]");
    }
    const double scale = floating("scale", separate ? 0.0625f : 0.125f);
    if (!std::isfinite(scale) || scale < 0)
      return reject("scale must be finite and nonnegative");
    if (!queryType || queryType.getRank() != 2 ||
        !(queryType.getElementType().isF16() ||
          (separate && queryType.getElementType().isF32())) ||
        queryType.getDimSize(1) !=
            (separate ? heads : heads + 2 * kvHeads) * headSize ||
        !cacheType || cacheType.getRank() != 4 ||
        !(quantized ? cacheType.getElementType().isInteger(8)
                    : cacheType.getElementType().isF16()) ||
        cacheType.getDimSize(2) != kvHeads ||
        cacheType.getDimSize(3) != headSize * (residualBits ? 2 : 1) ||
        valueType != cacheType || !cumulativeType ||
        cumulativeType.getRank() != 1 ||
        !cumulativeType.getElementType().isInteger(32) || !pastType ||
        pastType.getRank() != 1 || !pastType.getElementType().isInteger(32) ||
        !tableType || tableType.getRank() != 2 ||
        !tableType.getElementType().isInteger(32) ||
        (!separate && (!cosType || cosType.getRank() != 2 ||
                       !cosType.getElementType().isF16() ||
                       cosType.getDimSize(1) != 32 || sinType != cosType)))
      return reject("requires FP16 QKV or separate FP32 Q/K/V, "
                    "matching static cache geometry, "
                    "supported head size/rotary "
                    "tables and int32 metadata");

    auto context = getContextArg(op, rewriter);
    if (failed(context))
      return reject("missing runtime context");
    SmallVector<Type> types;
    SmallVector<Value> operands{*context};
    SmallVector<unsigned> inputs =
        separate ? SmallVector<unsigned>{0, 1, 2, 3, 4, 5, 6, 7}
                 : SmallVector<unsigned>{0, 3, 4, 5, 6, 7, 8, 9};
    for (unsigned index : inputs)
      operands.push_back(op->getOperand(index));
    if (quantized) {
      operands.push_back(op->getOperand(14));
      operands.push_back(op->getOperand(15));
    }
    for (unsigned i = 0; i < 3; ++i) {
      auto type = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!type ||
          (i == 0 && (type.getRank() != 2 ||
                      type.getElementType() != queryType.getElementType() ||
                      type.getDimSize(1) != heads * headSize ||
                      (!type.isDynamicDim(0) && !queryType.isDynamicDim(0) &&
                       type.getDimSize(0) != queryType.getDimSize(0)))) ||
          (i != 0 && type != cacheType))
        return reject("present cache shape must match past cache");
      types.push_back(type);
      operands.push_back(createEmptyTensor(rewriter, op->getLoc(), type,
                                           op->getOperand(i == 0 ? 0 : i + 2)));
    }
    OperationState state(op->getLoc(), quantized ? "hip.paged_attention_int8"
                                       : separate
                                           ? "hip.paged_attention_separate"
                                           : "hip.paged_attention");
    state.addOperands(operands);
    state.addTypes(types);
    state.addAttribute("num_heads", rewriter.getI64IntegerAttr(heads));
    state.addAttribute("kv_num_heads", rewriter.getI64IntegerAttr(kvHeads));
    state.addAttribute("scale", rewriter.getF32FloatAttr(scale));
    if (quantized)
      state.addAttribute("residual_bits",
                         rewriter.getI64IntegerAttr(residualBits));
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
