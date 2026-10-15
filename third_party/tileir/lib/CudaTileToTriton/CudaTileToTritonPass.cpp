//===- CudaTileToTritonPass.cpp - Convert CUDA Tile IR to Triton IR -------===//
//
// POC reverse translation of `convert-triton-to-cuda-tile`. Supported ops:
//
//   iota            -> tt.make_range
//   constant        -> arith.constant
//   reshape (s ->1) -> tt.splat
//   broadcast (1->N)-> tt.broadcast
//   offset          -> tt.addptr
//   load_ptr_tko    -> tt.load (token dropped)
//   store_ptr_tko   -> tt.store (token dropped)
//   addf/mulf/divf  -> arith.addf/mulf/divf (rounding/ftz modifiers ignored)
//   negf            -> arith.negf
//   exp             -> math.exp
//   addi/muli       -> arith.addi/muli
//   exti            -> arith.extsi/extui
//   ftof            -> arith.extf/truncf
//   get_tile_block_id -> 3x tt.get_program_id
//   assume          -> erased (POC: assumptions are not reconstructed)
//   entry           -> tt.func
//
// Tokens: CUDA Tile uses tokens to order token-ordered (tko) ops; Triton IR
// has no token concept. Since Triton loads/stores are ordered by program
// order, token inputs are dropped and unused token results are discarded. A
// pre-pass rewrites loads/stores to not consume tokens and erases
// `make_token`. Programs that depend on token results (joins/fences) are
// rejected.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
namespace tt = ::mlir::triton;

namespace {

//===----------------------------------------------------------------------===//
// Type conversion: tile<SxE> -> tensor<SxE'> (or scalar E' for rank-0 tiles),
// where ptr<E> elements map to !tt.ptr<E>. Token types pass through (they are
// dropped by the op patterns, never converted values are produced).
//===----------------------------------------------------------------------===//

class CudaTileToTritonTypeConverter : public TypeConverter {
public:
  explicit CudaTileToTritonTypeConverter(MLIRContext *context) {
    // Types that are already in Triton form (e.g. operands of ops whose
    // producers were converted first) pass through unchanged. Registered
    // first so it is consulted last.
    addConversion([](Type type) -> std::optional<Type> {
      if (isa<RankedTensorType, triton::PointerType, cuda_tile::TokenType,
              IntegerType, FloatType>(type))
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
// Token pre-pass patterns (run before dialect conversion, on tile-typed IR)
//===----------------------------------------------------------------------===//

// Rebuild a load without its token input. The token result is preserved so
// that the op shape is unchanged; the conversion drops it later.
struct DropLoadTokenInput : public OpRewritePattern<cuda_tile::LoadPtrTkoOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cuda_tile::LoadPtrTkoOp op,
                  PatternRewriter &rewriter) const override {
    if (!op.getToken())
      return failure();
    auto newOp = cuda_tile::LoadPtrTkoOp::create(
        rewriter, op.getLoc(), op.getResult().getType(),
        op.getResultToken().getType(), op.getMemoryOrderingSemanticsAttr(),
        op.getMemoryScopeAttr(), op.getSource(), op.getMask(),
        op.getPaddingValue(), /*token=*/nullptr,
        op.getOptimizationHintsAttr());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct DropStoreTokenInput
    : public OpRewritePattern<cuda_tile::StorePtrTkoOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op,
                  PatternRewriter &rewriter) const override {
    if (!op.getToken())
      return failure();
    auto newOp = cuda_tile::StorePtrTkoOp::create(
        rewriter, op.getLoc(), op.getResultToken().getType(),
        op.getMemoryOrderingSemanticsAttr(), op.getMemoryScopeAttr(),
        op.getDestination(), op.getValue(), op.getMask(), /*token=*/nullptr,
        op.getOptimizationHintsAttr());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// Once no load/store consumes tokens, `make_token` is dead.
struct EraseMakeToken : public OpRewritePattern<cuda_tile::MakeTokenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MakeTokenOp op,
                  PatternRewriter &rewriter) const override {
    if (!op->getResult(0).use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
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

struct ConvertConstant : public OpConversionPattern<cuda_tile::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getResult().getType());
    DenseTypedElementsAttr value = op.getValue();
    Attribute newAttr;
    if (auto tensorType = dyn_cast<RankedTensorType>(newType)) {
      // Element type is unchanged, so the payload transfers directly.
      TypedAttr attr = value.isSplat()
                           ? DenseElementsAttr::get(
                                 tensorType, value.getSplatValue<Attribute>())
                           : DenseElementsAttr::getFromRawBuffer(
                                 tensorType, value.getRawData());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newType, attr);
      return success();
    }
    // Rank-0 tile: scalar constant.
    TypedAttr splat = cast<TypedAttr>(value.getSplatValue<Attribute>());
    if (!splat)
      return op.emitOpError("POC: non-splat rank-0 constant");
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newType, splat);
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
    // The token input is dropped by the pre-pass; a token result must not be
    // consumed by anything (joins/fences are unsupported).
    if (!op.getResultToken().use_empty())
      return op.emitOpError("POC: used token results are not supported");
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    // Padding only applies where the mask is false: without a mask it is a
    // no-op, so it is dropped.
    Value mask = adaptor.getMask();
    Value padding = mask ? adaptor.getPaddingValue() : Value();
    auto newLoad = triton::LoadOp::create(
        rewriter, op.getLoc(), newResultType, adaptor.getSource(), mask,
        padding, Attribute(), /*isVolatile=*/false);
    // Replace the value result with the tt.load result and drop the token.
    rewriter.replaceOpWithMultiple(op, {{newLoad.getResult()}, {}});
    return success();
  }
};

struct ConvertStorePtrTko
    : public OpConversionPattern<cuda_tile::StorePtrTkoOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::StorePtrTkoOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getResultToken().use_empty())
      return op.emitOpError("POC: used token results are not supported");
    if (adaptor.getMask())
      triton::StoreOp::create(rewriter, op.getLoc(), adaptor.getDestination(),
                              adaptor.getValue(), adaptor.getMask());
    else
      triton::StoreOp::create(rewriter, op.getLoc(), adaptor.getDestination(),
                              adaptor.getValue());
    // Drop the token result.
    rewriter.replaceOpWithMultiple(op, {{}});
    return success();
  }
};

// POC: divf rounding modes (approx/full) and flush_to_zero are dropped;
// arith.divf is IEEE-accurate, which is a strict upgrade over `approx`.
struct ConvertDivF : public OpConversionPattern<cuda_tile::DivFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertAddF : public OpConversionPattern<cuda_tile::AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertMulF : public OpConversionPattern<cuda_tile::MulFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertNegF : public OpConversionPattern<cuda_tile::NegFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    // Lower to `0 - x` rather than `arith.negf`: stock Triton (e.g. 3.5.1)
    // rejects `arith.negf` in convert-triton-to-tritongpu, and its own
    // frontend emits `subf 0, x` for unary minus.
    auto zero = arith::ConstantOp::create(
        rewriter, op.getLoc(), newResultType,
        rewriter.getZeroAttr(newResultType));
    rewriter.replaceOpWithNewOp<arith::SubFOp>(op, newResultType,
                                               zero.getResult(),
                                               adaptor.getSource());
    return success();
  }
};

struct ConvertExp : public OpConversionPattern<cuda_tile::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<math::ExpOp>(op, newResultType,
                                             adaptor.getSource());
    return success();
  }
};

struct ConvertAddI : public OpConversionPattern<cuda_tile::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertMulI : public OpConversionPattern<cuda_tile::MulIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, newResultType,
                                               adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertExtI : public OpConversionPattern<cuda_tile::ExtIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::ExtIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    if (op.getSignedness() == cuda_tile::Signedness::Signed)
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, newResultType,
                                                  adaptor.getFrom());
    else
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, newResultType,
                                                  adaptor.getFrom());
    return success();
  }
};

struct ConvertFToF : public OpConversionPattern<cuda_tile::FToFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::FToFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fromType = cast<cuda_tile::TileType>(op.getFrom().getType());
    auto toType = cast<cuda_tile::TileType>(op.getResult().getType());
    unsigned fromWidth = fromType.getElementType().getIntOrFloatBitWidth();
    unsigned toWidth = toType.getElementType().getIntOrFloatBitWidth();
    Type newResultType = getTypeConverter()->convertType(op.getResult().getType());
    if (toWidth > fromWidth)
      rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, newResultType,
                                                 adaptor.getFrom());
    else
      rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, newResultType,
                                                   adaptor.getFrom());
    return success();
  }
};

struct ConvertGetTileBlockId
    : public OpConversionPattern<cuda_tile::GetTileBlockIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cuda_tile::GetTileBlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto x = tt::GetProgramIdOp::create(rewriter, op.getLoc(), 0);
    auto y = tt::GetProgramIdOp::create(rewriter, op.getLoc(), 1);
    auto z = tt::GetProgramIdOp::create(rewriter, op.getLoc(), 2);
    rewriter.replaceOpWithMultiple(op, {{x.getResult()},
                                        {y.getResult()},
                                        {z.getResult()}});
    return success();
  }
};

// POC: `assume` carries information (divisibility, bounds) that has no home
// in the reverse translation yet, so it is dropped.
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

    // 3. Drop token inputs from loads/stores and erase dead `make_token`s so
    //    the dialect conversion below never sees live token values.
    for (triton::FuncOp func : funcs) {
      RewritePatternSet tokenPatterns(func.getContext());
      tokenPatterns.add<DropLoadTokenInput, DropStoreTokenInput,
                        EraseMakeToken>(func.getContext());
      if (failed(applyPatternsGreedily(func, std::move(tokenPatterns)))) {
        func->emitOpError("POC: failed to drop token inputs");
        return signalPassFailure();
      }
    }

    // 4. Convert the cloned bodies from cuda_tile ops to Triton ops.
    ConversionTarget target(*context);
    target.addLegalDialect<triton::TritonDialect, arith::ArithDialect,
                           math::MathDialect>();
    target.addIllegalDialect<cuda_tile::CudaTileDialect>();

    RewritePatternSet patterns(context);
    patterns.add<
        /* clang-format off */
        ConvertAssume,
        ConvertIota,
        ConvertConstant,
        ConvertReshape,
        ConvertBroadcast,
        ConvertOffset,
        ConvertLoadPtrTko,
        ConvertStorePtrTko,
        ConvertAddF,
        ConvertDivF,
        ConvertMulF,
        ConvertNegF,
        ConvertExp,
        ConvertAddI,
        ConvertMulI,
        ConvertExtI,
        ConvertFToF,
        ConvertGetTileBlockId
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
    // cuda_tile ops with tile types; step 4 converts them.
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
