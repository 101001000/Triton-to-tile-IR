set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(TritonCudaTileToTriton
    CudaTileToTritonPass.cpp

    DEPENDS
    CudaTileToTritonIncGen

    LINK_LIBS PUBLIC
    # order is important
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
)
