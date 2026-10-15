#ifndef CUDA_TILE_TO_TRITON_CONVERSION_PASSES_H
#define CUDA_TILE_TO_TRITON_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createConvertCudaTileToTritonPass();

// Generate the pass class declarations (and options structs).
#define GEN_PASS_DECL
#include "CudaTileToTriton/Passes.h.inc"

// Generate the pass registration.
#define GEN_PASS_REGISTRATION
#include "CudaTileToTriton/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // CUDA_TILE_TO_TRITON_CONVERSION_PASSES_H
