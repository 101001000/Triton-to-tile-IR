// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-cuda-tile-to-triton,reconcile-unrealized-casts)" -split-input-file | FileCheck %s

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

// -----

// Test: a more complete kernel (swiglu forward) exercising tokens, constants,
// get_tile_block_id, integer/float conversions and arithmetic, plus loads
// with mask/padding. Tokens are dropped: Triton IR has no token concept, and
// program order gives the same ordering guarantees for these loads/stores.
// Note: the loads carry an all-true mask plus a padding value, matching the
// original `tl.load(..., mask=..., other=0.0)`.
module {
  cuda_tile.module @kernels {
    entry @_swiglu_forward_kernel_Kt2_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l0_2t1_3i16_p16_I1024_B0(%arg0: tile<ptr<f16>>, %arg1: tile<i32>, %arg2: tile<i32>, %arg3: tile<i32>, %arg4: tile<i32>, %arg5: tile<ptr<f16>>, %arg6: tile<i32>, %arg7: tile<i32>, %arg8: tile<i32>, %arg9: tile<i32>, %arg10: tile<ptr<f16>>, %arg11: tile<i32>, %arg12: tile<i32>, %arg13: tile<i32>, %arg14: tile<i32>) optimization_hints=<default = {num_worker_warps_per_cta = 8, occupancy = 1}> {
      %0 = make_token : token
      %assume = assume div_by<16>, %arg0 : tile<ptr<f16>>
      %assume_0 = assume div_by<8>, %arg3 : tile<i32>
      %assume_1 = assume div_by<16>, %arg5 : tile<ptr<f16>>
      %assume_2 = assume div_by<8>, %arg8 : tile<i32>
      %assume_3 = assume div_by<16>, %arg10 : tile<ptr<f16>>
      %assume_4 = assume div_by<8>, %arg13 : tile<i32>
      %cst_1_i64 = constant <i64: 1> : tile<1024xi64>
      %cst_1_i64_5 = constant <i64: 1> : tile<1024xi64>
      %cst_1_i64_6 = constant <i64: 1> : tile<1024xi64>
      %assume_assume = assume bounded<0, ?>, %assume_0 : tile<i32>
      %assume_assume_7 = assume bounded<0, ?>, %assume_2 : tile<i32>
      %assume_assume_8 = assume bounded<0, ?>, %assume_4 : tile<i32>
      %blockId_x, %blockId_y, %blockId_z = get_tile_block_id : tile<i32>
      %1 = iota : tile<1024xi32>
      %true = constant <i1: true> : tile<1024xi1>
      %2 = exti %blockId_x signed : tile<i32> -> tile<i64>
      %reshape = reshape %2 : tile<i64> -> tile<1xi64>
      %bcast = broadcast %reshape : tile<1xi64> -> tile<1024xi64>
      %3 = exti %assume_assume signed : tile<i32> -> tile<i64>
      %reshape_9 = reshape %3 : tile<i64> -> tile<1xi64>
      %bcast_10 = broadcast %reshape_9 : tile<1xi64> -> tile<1024xi64>
      %4 = muli %bcast, %bcast_10 : tile<1024xi64>
      %5 = exti %1 signed : tile<1024xi32> -> tile<1024xi64>
      %6 = muli %5, %cst_1_i64 : tile<1024xi64>
      %7 = addi %4, %6 : tile<1024xi64>
      %reshape_11 = reshape %assume : tile<ptr<f16>> -> tile<1xptr<f16>>
      %bcast_12 = broadcast %reshape_11 : tile<1xptr<f16>> -> tile<1024xptr<f16>>
      %8 = offset %bcast_12, %7 : tile<1024xptr<f16>>, tile<1024xi64> -> tile<1024xptr<f16>>
      %cst_0_f16 = constant <f16: 0.000000e+00> : tile<f16>
      %reshape_13 = reshape %cst_0_f16 : tile<f16> -> tile<1xf16>
      %bcast_14 = broadcast %reshape_13 : tile<1xf16> -> tile<1024xf16>
      %result, %result_token = load_ptr_tko weak %8, %true, %bcast_14 token=%0 : tile<1024xptr<f16>>, tile<1024xi1>, tile<1024xf16> -> tile<1024xf16>, token
      %9 = exti %blockId_x signed : tile<i32> -> tile<i64>
      %reshape_15 = reshape %9 : tile<i64> -> tile<1xi64>
      %bcast_16 = broadcast %reshape_15 : tile<1xi64> -> tile<1024xi64>
      %10 = exti %assume_assume_7 signed : tile<i32> -> tile<i64>
      %reshape_17 = reshape %10 : tile<i64> -> tile<1xi64>
      %bcast_18 = broadcast %reshape_17 : tile<1xi64> -> tile<1024xi64>
      %11 = muli %bcast_16, %bcast_18 : tile<1024xi64>
      %12 = exti %1 signed : tile<1024xi32> -> tile<1024xi64>
      %13 = muli %12, %cst_1_i64_5 : tile<1024xi64>
      %14 = addi %11, %13 : tile<1024xi64>
      %reshape_19 = reshape %assume_1 : tile<ptr<f16>> -> tile<1xptr<f16>>
      %bcast_20 = broadcast %reshape_19 : tile<1xptr<f16>> -> tile<1024xptr<f16>>
      %15 = offset %bcast_20, %14 : tile<1024xptr<f16>>, tile<1024xi64> -> tile<1024xptr<f16>>
      %cst_0_f16_21 = constant <f16: 0.000000e+00> : tile<f16>
      %reshape_22 = reshape %cst_0_f16_21 : tile<f16> -> tile<1xf16>
      %bcast_23 = broadcast %reshape_22 : tile<1xf16> -> tile<1024xf16>
      %result_24, %result_token_25 = load_ptr_tko weak %15, %true, %bcast_23 token=%0 : tile<1024xptr<f16>>, tile<1024xi1>, tile<1024xf16> -> tile<1024xf16>, token
      %16 = ftof %result  : tile<1024xf16> -> tile<1024xf32>
      %cst_1_f32 = constant <f32: 1.000000e+00> : tile<f32>
      %17 = negf %16 : tile<1024xf32>
      %18 = exp %17  : tile<1024xf32>
      %reshape_26 = reshape %cst_1_f32 : tile<f32> -> tile<1xf32>
      %bcast_27 = broadcast %reshape_26 : tile<1xf32> -> tile<1024xf32>
      %19 = addf %bcast_27, %18  : tile<1024xf32>
      %cst_1_f32_28 = constant <f32: 1.000000e+00> : tile<f32>
      %reshape_29 = reshape %cst_1_f32_28 : tile<f32> -> tile<1xf32>
      %bcast_30 = broadcast %reshape_29 : tile<1xf32> -> tile<1024xf32>
      %20 = divf %bcast_30, %19 rounding<approx> flush_to_zero : tile<1024xf32>
      %21 = mulf %16, %20  : tile<1024xf32>
      %22 = ftof %21  : tile<1024xf32> -> tile<1024xf16>
      %23 = mulf %22, %result_24  : tile<1024xf16>
      %24 = exti %blockId_x signed : tile<i32> -> tile<i64>
      %reshape_31 = reshape %24 : tile<i64> -> tile<1xi64>
      %bcast_32 = broadcast %reshape_31 : tile<1xi64> -> tile<1024xi64>
      %25 = exti %assume_assume_8 signed : tile<i32> -> tile<i64>
      %reshape_33 = reshape %25 : tile<i64> -> tile<1xi64>
      %bcast_34 = broadcast %reshape_33 : tile<1xi64> -> tile<1024xi64>
      %26 = muli %bcast_32, %bcast_34 : tile<1024xi64>
      %27 = exti %1 signed : tile<1024xi32> -> tile<1024xi64>
      %28 = muli %27, %cst_1_i64_6 : tile<1024xi64>
      %29 = addi %26, %28 : tile<1024xi64>
      %reshape_35 = reshape %assume_3 : tile<ptr<f16>> -> tile<1xptr<f16>>
      %bcast_36 = broadcast %reshape_35 : tile<1xptr<f16>> -> tile<1024xptr<f16>>
      %30 = offset %bcast_36, %29 : tile<1024xptr<f16>>, tile<1024xi64> -> tile<1024xptr<f16>>
      %31 = store_ptr_tko weak %30, %23 token=%0 : tile<1024xptr<f16>>, tile<1024xf16> -> token
      return
    }
  }
}

// CHECK-LABEL: tt.func public @_swiglu_forward_kernel_Kt2_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l0_2t1_3i16_p16_A2f16_1v8l0_2t1_3i16_p16_I1024_B0
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[Z16:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[MSK:.*]] = arith.constant dense<true> : tensor<1024xi1>
// CHECK: %[[CST_STRIDE:.*]] = arith.constant dense<1> : tensor<1024xi64>
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[PIDY:.*]] = tt.get_program_id y : i32
// CHECK: %[[PIDZ:.*]] = tt.get_program_id z : i32
// CHECK: %[[RANGE:.*]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK: %[[PID64:.*]] = arith.extsi %[[PID]] : i32 to i64
// CHECK: %[[SP:.*]] = tt.splat %[[PID64]] : i64 -> tensor<1xi64>
// CHECK: %[[BP:.*]] = tt.broadcast %[[SP]] : tensor<1xi64> -> tensor<1024xi64>
// CHECK: %[[S64:.*]] = arith.extsi %arg3 : i32 to i64
// CHECK: %[[SS:.*]] = tt.splat %[[S64]] : i64 -> tensor<1xi64>
// CHECK: %[[BS:.*]] = tt.broadcast %[[SS]] : tensor<1xi64> -> tensor<1024xi64>
// CHECK: %[[M1:.*]] = arith.muli %[[BP]], %[[BS]] : tensor<1024xi64>
// CHECK: %[[RI:.*]] = arith.extsi %[[RANGE]] : tensor<1024xi32> to tensor<1024xi64>
// CHECK: %[[M2:.*]] = arith.muli %[[RI]], %[[CST_STRIDE]] : tensor<1024xi64>
// CHECK: %[[OFF1:.*]] = arith.addi %[[M1]], %[[M2]] : tensor<1024xi64>
// CHECK: %[[SA:.*]] = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>>
// CHECK: %[[BA:.*]] = tt.broadcast %[[SA]] : tensor<1x!tt.ptr<f16>> -> tensor<1024x!tt.ptr<f16>>
// CHECK: %[[PA:.*]] = tt.addptr %[[BA]], %[[OFF1]] : tensor<1024x!tt.ptr<f16>>, tensor<1024xi64>
// CHECK: %[[SZ16:.*]] = tt.splat %[[Z16]] : f16 -> tensor<1xf16>
// CHECK: %[[BZ16:.*]] = tt.broadcast %[[SZ16]] : tensor<1xf16> -> tensor<1024xf16>
// CHECK: %[[LA:.*]] = tt.load %[[PA]], %[[MSK]], %[[BZ16]] : tensor<1024x!tt.ptr<f16>>
// CHECK: %[[LB:.*]] = tt.load %{{.*}}, %[[MSK]], %{{.*}} : tensor<1024x!tt.ptr<f16>>
// CHECK: %[[AF:.*]] = arith.extf %[[LA]] : tensor<1024xf16> to tensor<1024xf32>
// NOTE: negf is lowered to subf 0, x because stock Triton (e.g. 3.5.1)
// rejects arith.negf in convert-triton-to-tritongpu.
// CHECK: %[[Z32:.*]] = arith.constant dense<0.000000e+00> : tensor<1024xf32>
// CHECK: %[[NEG:.*]] = arith.subf %[[Z32]], %[[AF]] : tensor<1024xf32>
// CHECK: %[[EXP:.*]] = math.exp %[[NEG]] : tensor<1024xf32>
// CHECK: %[[SONE:.*]] = tt.splat %[[ONE]] : f32 -> tensor<1xf32>
// CHECK: %[[BONE:.*]] = tt.broadcast %[[SONE]] : tensor<1xf32> -> tensor<1024xf32>
// CHECK: %[[DEN:.*]] = arith.addf %[[BONE]], %[[EXP]] : tensor<1024xf32>
// CHECK: %[[REC:.*]] = arith.divf %{{.*}}, %[[DEN]] : tensor<1024xf32>
// CHECK: %[[SIG:.*]] = arith.mulf %[[AF]], %[[REC]] : tensor<1024xf32>
// CHECK: %[[SIG16:.*]] = arith.truncf %[[SIG]] : tensor<1024xf32> to tensor<1024xf16>
// CHECK: %[[PROD:.*]] = arith.mulf %[[SIG16]], %[[LB]] : tensor<1024xf16>
// CHECK: tt.store %{{.*}}, %[[PROD]] : tensor<1024x!tt.ptr<f16>>
// CHECK: tt.return
// CHECK-NOT: cuda_tile
