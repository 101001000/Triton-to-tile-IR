// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-cuda-tile-to-triton,reconcile-unrealized-casts)" | FileCheck %s

// POC: reverse translation of the sum of two tiles (CUDA Tile IR -> Triton IR).
// The input is the (assume-stripped) output of `convert-triton-to-cuda-tile`
// for a kernel that loads two 128-element tiles, adds them, and stores the
// result. The reverse pass must produce the equivalent Triton IR.

module {
  cuda_tile.module @cuda_tile_module {
    entry @sum(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>) {
      // Matches the shape of convert-triton-to-cuda-tile output, including
      // the assume ops it emits for tt.divisibility arg attributes.
      %assume_a = assume div_by<16>, %arg0 : tile<ptr<f32>>
      %assume_b = assume div_by<16>, %arg1 : tile<ptr<f32>>
      %assume_o = assume div_by<16>, %arg2 : tile<ptr<f32>>
      %iota = iota : tile<128xi32>
      %ra = reshape %assume_a : tile<ptr<f32>> -> tile<1xptr<f32>>
      %ba = broadcast %ra : tile<1xptr<f32>> -> tile<128xptr<f32>>
      %rb = reshape %assume_b : tile<ptr<f32>> -> tile<1xptr<f32>>
      %bb = broadcast %rb : tile<1xptr<f32>> -> tile<128xptr<f32>>
      %ro = reshape %assume_o : tile<ptr<f32>> -> tile<1xptr<f32>>
      %bo = broadcast %ro : tile<1xptr<f32>> -> tile<128xptr<f32>>
      %pa = offset %ba, %iota : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
      %pb = offset %bb, %iota : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
      %po = offset %bo, %iota : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
      %va, %ta = load_ptr_tko weak %pa : tile<128xptr<f32>> -> tile<128xf32>, token
      %vb, %tb = load_ptr_tko weak %pb : tile<128xptr<f32>> -> tile<128xf32>, token
      %s = addf %va, %vb : tile<128xf32>
      %to = store_ptr_tko weak %po, %s : tile<128xptr<f32>>, tile<128xf32> -> token
      return
    }
  }
}

// CHECK-LABEL: tt.func public @sum
// CHECK: %[[OFF:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK: %[[SA:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[BA:.*]] = tt.broadcast %[[SA]] : tensor<1x!tt.ptr<f32>> -> tensor<128x!tt.ptr<f32>>
// CHECK: %[[SB:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[BB:.*]] = tt.broadcast %[[SB]] : tensor<1x!tt.ptr<f32>> -> tensor<128x!tt.ptr<f32>>
// CHECK: %[[SO:.*]] = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %[[BO:.*]] = tt.broadcast %[[SO]] : tensor<1x!tt.ptr<f32>> -> tensor<128x!tt.ptr<f32>>
// CHECK: %[[AA:.*]] = tt.addptr %[[BA]], %[[OFF]] : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK: %[[AB:.*]] = tt.addptr %[[BB]], %[[OFF]] : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK: %[[AO:.*]] = tt.addptr %[[BO]], %[[OFF]] : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
// CHECK: %[[VA:.*]] = tt.load %[[AA]] : tensor<128x!tt.ptr<f32>>
// CHECK: %[[VB:.*]] = tt.load %[[AB]] : tensor<128x!tt.ptr<f32>>
// CHECK: %[[S:.*]] = arith.addf %[[VA]], %[[VB]] : tensor<128xf32>
// CHECK: tt.store %[[AO]], %[[S]] : tensor<128x!tt.ptr<f32>>
// CHECK: tt.return
// CHECK-NOT: cuda_tile
