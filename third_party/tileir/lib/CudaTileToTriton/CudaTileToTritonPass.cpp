//===- CudaTileToTritonPass.cpp - Convert CUDA Tile IR to Triton IR -------===//
//
// POC reverse translation of `convert-triton-to-cuda-tile`. Supports the
// sum-of-two-tiles round trip:
//
//   iota            -> tt.make_range
//   reshape (s ->1) -> tt.splat
//   broadcast (1->N)-> tt.broadcast
//   offset          -> tt.addptr
//   load_ptr_tko    -> tt.load (token dropped, must be unused)
//   store_ptr_tko   -> tt.store (token dropped, must be unused)
//   addf            -> arith.addf
//   assume          -> erased (POC: assumptions are not reconstructed)
//   entry           -> tt.func
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CudaTileToTriton/Passes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// MLIR pass TableGen uses per-pass macros (GEN_PASS_DEF_*).
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTCUDATILETOTRITON
#include "CudaTileToTriton/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

//===----------------------------------------------------------------------===//
// Type conversion: tile<SxE> -> tensor<SxE'> (or scalar E' for rank-0 tiles),
// where ptr<E> elements map to !tt.ptr<E>.
//===----------------------------------------------------------------------===//

class CudaTileToTritonTypeConverter : public TypeConverter {
public:
  explicit CudaTileToTritonTypeConverter(MLIRContext *context) {
    // Types that are already in Triton form (e.g. operands of ops whose
    // producers were converted first) pass through unchanged. Registered
    // first so it is consulted last.
    addConversion([](Type type) -> std::optional<Type> {
      if (isa<RankedTensorType, triton::PointerType>(type))
        return type;
      return std::nullopt;
    });
    addConversion([](cuda_tile::TileType tileType) -> std::optional<Type> {
      Type elementType = tileType.getElementType();
      Type tritonElementType = elementType;
      if (auto ptrType = dyn_cast<cuda_tile::PointerType>(elementType))
        tritonElementType = triton::PointerType::get(ptrType.getPointeeType());
      if (tileType.getShape().empty())
        return tritonElementType;
      return RankedTensorType::get(tileType.getShape(), tritonElementType);
    });
  }
};

//===----------------------------------------------------------------------===//
// Op conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertIota : public OpConversionPattern<cuda_tile::IotaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::IotaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newResultType =
        dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!newResultType || newResultType.getRank() != 1 ||
        !isa<IntegerType>(newResultType.getElementType()))
      return op.emitOpError("POC: iota is only supported on 1-d integer tiles");
    auto elemType = cast<IntegerType>(newResultType.getElementType());
    if (elemType.getWidth() != 32)
      return op.emitOpError("POC: iota is only supported on i32 tiles");
    int64_t n = newResultType.getShape().front();
    rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(op, newResultType,
                                                     /*start=*/0, /*end=*/n);
    return success();
  }
};

// A rank-0 -> rank-1 reshape is the scalar-to-tensor step of the splat that
// `convert-triton-to-cuda-tile` emits for `tt.splat`.
struct ConvertReshape : public OpConversionPattern<cuda_tile::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType();
    auto resultType = op.getResult().getType();
    if (!srcType.getShape().empty() || resultType.getShape().size() != 1)
      return op.emitOpError("POC: only scalar -> 1-d reshape is supported");
    Type newResultType = getTypeConverter()->convertType(resultType);
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, newResultType,
                                                 adaptor.getSource());
    return success();
  }
};

struct ConvertBroadcast : public OpConversionPattern<cuda_tile::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<triton::BroadcastOp>(op, newResultType,
                                                     adaptor.getSource());
    return success();
  }
};

struct ConvertOffset : public OpConversionPattern<cuda_tile::OffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::OffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, newResultType,
                                                  adaptor.getPtr(),
                                                  adaptor.getOffset());
    return success();
  }
};

struct ConvertLoadPtrTko : public OpConversionPattern<cuda_tile::LoadPtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // POC: token-threaded programs are rejected.
    if (!op.getResultToken().use_empty())
      return op.emitOpError("POC: used token results are not supported");
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    auto newLoad = triton::LoadOp::create(
        rewriter, op.getLoc(), newResultType, adaptor.getSource(), Value(),
        Value(), Attribute(), /*isVolatile=*/false);
    // Replace the value result with the tt.load result and drop the (unused)
    // token result.
    rewriter.replaceOpWithMultiple(op,
                                   {{newLoad.getResult()}, {}});
    return success();
  }
};

struct ConvertStorePtrTko
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // POC: token-threaded programs are rejected.
    if (!op.getResultToken().use_empty())
      return op.emitOpError("POC: used token results are not supported");
    triton::StoreOp::create(rewriter, op.getLoc(), adaptor.getDestination(),
                            adaptor.getValue());
    // Drop the (unused) token result.
    rewriter.replaceOpWithMultiple(op, {{}});
    return success();
  }
};

struct ConvertAddF : public OpConversionPattern<cuda_tile::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getFlushToZero())
      return op.emitOpError("POC: flush_to_zero modifier is not supported");
    auto roundingMode = op->getAttrOfType<cuda_tile::RoundingModeAttr>(
        cuda_tile::AddFOp::getRoundingModeAttrName(op->getName()));
    if (roundingMode &&
        roundingMode.getValue() != cuda_tile::RoundingMode::NEAREST_EVEN)
      return op.emitOpError("POC: rounding mode modifiers are not supported");
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

// POC: `assume` carries Triton arg-attr information (e.g. divisibility) that
// has no home in the reverse translation yet, so it is dropped.
struct ConvertAssume : public OpConversionPattern<cuda_tile::AssumeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {

struct ConvertCudaTileToTriton
    : public mlir::triton::impl::ConvertCudaTileToTritonBase<
          ConvertCudaTileToTriton> {
  using mlir::triton::impl::ConvertCudaTileToTritonBase<
      ConvertCudaTileToTriton>::ConvertCudaTileToTritonBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    CudaTileToTritonTypeConverter typeConverter(context);

    // 1. Lower every `entry` into a fresh `tt.func` at the top-level module
    //    (signature converted, body cloned), remembering the originals.
    SmallVector<cuda_tile::EntryOp> entries;
    module->walk([&](cuda_tile::EntryOp entry) { entries.push_back(entry); });
    if (entries.empty())
      return;

    SmallVector<triton::FuncOp> funcs;
    for (cuda_tile::EntryOp entry : entries) {
      triton::FuncOp func = lowerEntry(module, entry, typeConverter);
      if (!func)
        return signalPassFailure();
      funcs.push_back(func);
    }

    // 2. Erase the `cuda_tile.module` wrappers (this also erases the original
    //    entries cloned in step 1).
    SmallVector<Operation *> cudaModules;
    module->walk([&](Operation *op) {
      if (isa<cuda_tile::ModuleOp>(op))
        cudaModules.push_back(op);
    });
    for (Operation *op : llvm::reverse(cudaModules))
      op->erase();

    // 3. Convert the cloned bodies from cuda_tile ops to Triton ops.
    ConversionTarget target(*context);
    target.addLegalDialect<triton::TritonDialect, arith::ArithDialect>();
    target.addIllegalDialect<cuda_tile::CudaTileDialect>();

    RewritePatternSet patterns(context);
    patterns.add<
        /* clang-format off */
        ConvertAssume,
        ConvertIota,
        ConvertReshape,
        ConvertBroadcast,
        ConvertOffset,
        ConvertLoadPtrTko,
        ConvertStorePtrTko,
        ConvertAddF
        /* clang-format on */
        >(typeConverter, context);
    FrozenRewritePatternSet frozen(std::move(patterns));

    for (triton::FuncOp func : funcs) {
      if (failed(applyFullConversion(func, target, frozen)))
        return signalPassFailure();
    }
  }

private:
  triton::FuncOp lowerEntry(ModuleOp module, cuda_tile::EntryOp entry,
                            TypeConverter &typeConverter) const {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    Location loc = entry.getLoc();

    SmallVector<Type> argTypes;
    if (failed(typeConverter.convertTypes(entry.getArgumentTypes(), argTypes))) {
      entry.emitOpError("failed to convert entry signature");
      return {};
    }

    auto funcType =
        FunctionType::get(module.getContext(), argTypes, /*results=*/{});
    auto func = triton::FuncOp::create(builder, loc, entry.getSymName(),
                                       funcType);
    func->setAttr("sym_visibility", builder.getStringAttr("public"));
    builder.setInsertionPointToEnd(&func.getBody().emplaceBlock());

    // Map the (tile-typed) entry block arguments to the converted tt.func
    // block arguments, then clone the body. The clone still contains
    // cuda_tile ops with tile types; step 3 converts them.
    auto &body = func.getBody().front();
    IRMapping mapping;
    for (auto [oldArg, newType] :
         llvm::zip(entry.getBody().getArguments(), argTypes))
      mapping.map(oldArg, body.addArgument(newType, loc));

    for (Operation &op : entry.getBody().front().without_terminator())
      builder.clone(op, mapping);
    triton::ReturnOp::create(builder, loc);
    return func;
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::triton::createConvertCudaTileToTritonPass() {
  return std::make_unique<ConvertCudaTileToTriton>();
}
