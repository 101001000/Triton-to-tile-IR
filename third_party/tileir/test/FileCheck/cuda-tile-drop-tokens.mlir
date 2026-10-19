// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(cuda-tile-drop-tokens)" | FileCheck %s

// Standalone pass that removes token usage from CUDA Tile IR: drops the
// `token=` input from token-ordered loads/stores and erases `make_token`
// ops once dead. Token *results* are left in place (they are unused) so
// the op shapes don't change; a later pass/consumer can ignore them.

module {
  cuda_tile.module @kernels {
    entry @tokens(%arg0: tile<ptr<f16>>) {
      %tok = make_token : token
      %iota = iota : tile<4xi32>
      %mask = constant <i1: 1> : tile<4xi1>
      %pad = constant <f16: 0.000000e+00> : tile<4xf16>
      %r = reshape %arg0 : tile<ptr<f16>> -> tile<1xptr<f16>>
      %b = broadcast %r : tile<1xptr<f16>> -> tile<4xptr<f16>>
      %p = offset %b, %iota : tile<4xptr<f16>>, tile<4xi32> -> tile<4xptr<f16>>
      %v, %ltok = load_ptr_tko weak %p, %mask, %pad token=%tok : tile<4xptr<f16>>, tile<4xi1>, tile<4xf16> -> tile<4xf16>, token
      %stok = store_ptr_tko weak %p, %v token=%tok : tile<4xptr<f16>>, tile<4xf16> -> token
      return
    }
  }
}

// CHECK-LABEL: entry @tokens
// The token input is gone from both memory ops...
// CHECK-NOT: token=
// CHECK: {{%.*}} = load_ptr_tko weak {{%.*}}, {{%.*}}, {{%.*}} : tile<4xptr<f16>>, tile<4xi1>, tile<4xf16> -> tile<4xf16>, token
// CHECK: store_ptr_tko weak {{%.*}}, {{%.*}} : tile<4xptr<f16>>, tile<4xf16> -> token
// ...and make_token is erased once nothing consumes it.
// CHECK-NOT: make_token
// CHECK: return
