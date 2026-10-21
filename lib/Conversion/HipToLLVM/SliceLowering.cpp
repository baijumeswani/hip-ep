/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.slice(ctx, data, starts, ends, [axes], [steps], output)
//   -> wrap_slice(state, data_ptr, starts_ptr, ends_ptr,
//                 axes_ptr (or null), steps_ptr (or null), out_ptr,
//                 data_shape_ptr, data_rank,
//                 output_shape_ptr, output_rank,
//                 starts_num_elements, axes_num_elements,
//                 steps_num_elements, data_type)
//
// Constant bounds on dynamic axes still need runtime clamping. Materialize
// those bounds on the host instead of reading immutable GPU tensors back.
struct SliceOpLowering : public ConvertOpToLLVMPattern<SliceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto startsType = cast<MemRefType>(op.getStarts().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value startsPtr =
        extractContiguousMemRefPtr(adaptor.getStarts(), rewriter, loc);
    Value endsPtr =
        extractContiguousMemRefPtr(adaptor.getEnds(), rewriter, loc);
    Value axesPtr = extractOptionalMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value stepsPtr =
        extractOptionalMemRefPtr(adaptor.getSteps(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value startsNum =
        computeNumElements(startsType, adaptor.getStarts(), rewriter, loc);
    Value axesNum;
    if (op.getAxes()) {
      auto axesT = cast<MemRefType>(op.getAxes().getType());
      axesNum = computeNumElements(axesT, adaptor.getAxes(), rewriter, loc);
    } else {
      axesNum = createI64Const(0);
    }
    Value stepsNum;
    if (op.getSteps()) {
      auto stepsT = cast<MemRefType>(op.getSteps().getType());
      stepsNum = computeNumElements(stepsT, adaptor.getSteps(), rewriter, loc);
    } else {
      stepsNum = createI64Const(0);
    }

    Value dataRank = createI64Const(dataType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(hipDtype);
    auto constantStarts =
        op->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_starts");
    auto constantEnds =
        op->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_ends");
    auto constantAxes =
        op->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_axes");
    auto constantSteps =
        op->getAttrOfType<DenseI64ArrayAttr>("hipdnn.constant_steps");
    bool hostIndices =
        constantStarts && constantEnds && constantAxes && constantSteps;
    if ((constantStarts || constantEnds || constantAxes || constantSteps) &&
        !hostIndices)
      return rewriter.notifyMatchFailure(op,
                                         "incomplete constant Slice bounds");
    if (hostIndices) {
      auto count = constantStarts.size();
      if (!count || constantEnds.size() != count ||
          (op.getAxes() ? constantAxes.size() != count
                        : !constantAxes.empty()) ||
          (op.getSteps() ? constantSteps.size() != count
                         : !constantSteps.empty()))
        return rewriter.notifyMatchFailure(
            op, "inconsistent constant Slice bounds");
      auto emitIndices = [&](DenseI64ArrayAttr values) -> Value {
        if (values.empty())
          return LLVM::ZeroOp::create(rewriter, loc, ptrType);
        auto arrayType = LLVM::LLVMArrayType::get(i64Type, values.size());
        Value array =
            LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType, one, 8);
        for (auto [index, value] : llvm::enumerate(values.asArrayRef())) {
          Value offset = createI64Const(index);
          Value pointer = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                              array, offset);
          LLVM::StoreOp::create(rewriter, loc, createI64Const(value), pointer);
        }
        return array;
      };
      startsPtr = emitIndices(constantStarts);
      endsPtr = emitIndices(constantEnds);
      axesPtr = emitIndices(constantAxes);
      stepsPtr = emitIndices(constantSteps);
      startsNum = createI64Const(count);
      axesNum = createI64Const(constantAxes.size());
      stepsNum = createI64Const(constantSteps.size());
    }

    SmallVector<Type, 16> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, data, starts, ends
        ptrType, ptrType, ptrType,          // axes, steps, output
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // out_shape,  out_rank
        i64Type, i64Type, i64Type,          // starts_num, axes_num,
                                            // steps_num
        i64Type};                           // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, hostIndices ? kWrapSliceHost : kWrapSlice, paramTypes,
        i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {statePtr, dataPtr,  startsPtr,  endsPtr,
                                   axesPtr,  stepsPtr, outPtr,     dataShape,
                                   dataRank, outShape, outRank,    startsNum,
                                   axesNum,  stepsNum, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateSliceLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<SliceOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
