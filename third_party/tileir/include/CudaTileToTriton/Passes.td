#ifndef CUDA_TILE_TO_TRITON_CONVERSION_PASSES
#define CUDA_TILE_TO_TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCudaTileToTriton : Pass<"convert-cuda-tile-to-triton", "mlir::ModuleOp"> {
    let summary = "Convert CUDA Tile IR back to the Triton dialect";
    let description = [{
        POC pass that performs the reverse translation of
        `convert-triton-to-cuda-tile`: it rewrites `cuda_tile` operations into
        Triton/arith operations. Currently supports the sum-of-two-tiles
        round trip (iota/reshape/broadcast/offset, loads and stores, addf,
        assume). Token results must be unused; anything else is rejected.
    }];
    let constructor = "mlir::triton::createConvertCudaTileToTritonPass()";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::cuda_tile::CudaTileDialect"];
}

def CudaTileDropTokens : Pass<"cuda-tile-drop-tokens", "mlir::ModuleOp"> {
    let summary = "Remove token usage from CUDA Tile IR";
    let description = [{
        Drops the `token=` input from token-ordered loads/stores
        (`load_ptr_tko`/`store_ptr_tko`) and erases `make_token` ops once
        they have no remaining uses. Token *results* are left in place but
        must not be consumed by anything (joins/fences are unsupported):
        Triton IR has no token concept, and program order gives the same
        ordering guarantees.
    }];
    let constructor = "mlir::triton::createCudaTileDropTokensPass()";
    let dependentDialects = ["mlir::cuda_tile::CudaTileDialect"];
}

#endif
