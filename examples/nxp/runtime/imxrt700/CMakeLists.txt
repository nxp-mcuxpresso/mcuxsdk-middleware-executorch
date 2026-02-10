if (CONFIG_MCUX_COMPONENT_middleware.eiq.executorch)
  mcux_add_source(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    SOURCES backends/nxp/runtime/*.h
    codegen/*.h
    extension/data_loader/*.h
    extension/evalue_util/*.h
    extension/runner_util/*.h
    kernels/prim_ops/*.h
    runtime/backend/*.h
    runtime/core/*.h
    runtime/core/exec_aten/*.h
    runtime/core/exec_aten/util/*.h
    runtime/core/portable_type/*.h

    runtime/core/portable_type/c10/c10/macros/*.h
    runtime/core/portable_type/c10/c10/util/*.h
    runtime/core/portable_type/c10/torch/headeronly/macros/*.h
    runtime/core/portable_type/c10/torch/headeronly/util/*.h

    runtime/executor/*.h
    runtime/kernel/*.h
    kernels/portable/cpu/*.h
    kernels/portable/cpu/pattern/*.h
    kernels/portable/cpu/util/*.h
    runtime/platform/*.h
    kernels/quantized/cpu/*.h
    schema/*.h
    third-party/flatbuffers/include/flatbuffers/*.h
  )
  mcux_add_include(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    INCLUDES .
    ../
    ../..
    third-party/flatbuffers/include
    runtime/core/portable_type/c10/
  )

  mcux_add_mdk_configuration(
    CX "-std=gnu++17"
    LD "--library_type=standardlib"
  )
  mcux_add_armgcc_configuration(
    CC "-std=c99"
    CX "-std=c++17"
  )
  mcux_add_macro(
    CC "NO_HEAP_USAGE=1"
    CX "NO_HEAP_USAGE=1"
  )
endif()

if (CONFIG_MCUX_COMPONENT_middleware.eiq.executorch.neutron)
  mcux_add_source(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    SOURCES backends/nxp/runtime/NeutronBackend.cpp
  )
  mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    LIBS ../neutron/rt700/cm33/libNeutronDriver.a ../neutron/rt700/cm33/libNeutronFirmware.a
    TOOLCHAINS armgcc mcux
    DEVICE_IDS MIMXRT798S
  )
  mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    LIBS ../neutron/rt700/hifi4/libNeutronDriver.a ../neutron/rt700/hifi4/libNeutronFirmware.a
    TOOLCHAINS xcc xtensa
    DEVICE_IDS MIMXRT798S
  )
endif()

if (CONFIG_MCUX_COMPONENT_middleware.eiq.executorch.binary)
  mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    LIBS lib/cm33/armgcc/libexecutorch.a
    TOOLCHAINS armgcc mcux
    CORES cm33 cm33f
  )
  mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    LIBS lib/hifi4/xcc/imxrt700/libexecutorch.a
    TOOLCHAINS xcc xtensa
    CORE_IDS hifi4
    DEVICE_IDS MIMXRT798S
  )
endif()

if (CONFIG_MCUX_COMPONENT_middleware.eiq.executorch.source)
  mcux_add_source(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../
    SOURCES kernels/prim_ops/et_copy_index.cpp
    kernels/prim_ops/et_view.cpp
    kernels/prim_ops/register_prim_ops.cpp

    extension/runner_util/inputs.cpp
    extension/runner_util/inputs_portable.cpp

    runtime/backend/interface.cpp
    runtime/core/evalue.cpp
    runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp
    runtime/core/exec_aten/util/tensor_util_portable.cpp
    runtime/core/portable_type/tensor_impl.cpp
    runtime/core/tag.cpp
    runtime/core/tensor_layout.cpp

    runtime/executor/method.cpp
    runtime/executor/method_meta.cpp
    runtime/executor/program.cpp
    runtime/executor/tensor_parser_exec_aten.cpp
    
    runtime/executor/tensor_parser_portable.cpp
    runtime/executor/pte_data_map.cpp
    runtime/kernel/operator_registry.cpp
    runtime/platform/abort.cpp
    runtime/platform/default/posix.cpp
    runtime/platform/log.cpp
    runtime/platform/platform.cpp
    runtime/platform/profiler.cpp
    runtime/platform/runtime.cpp
    schema/extended_header.cpp
  
    kernels/portable/cpu/op__clone_dim_order.cpp
    kernels/portable/cpu/op__empty_dim_order.cpp
    kernels/portable/cpu/op__to_dim_order_copy.cpp
    kernels/portable/cpu/op_abs.cpp
    kernels/portable/cpu/op_acos.cpp
    kernels/portable/cpu/op_acosh.cpp
    kernels/portable/cpu/op_add.cpp
    kernels/portable/cpu/op_addmm.cpp
    kernels/portable/cpu/op_alias_copy.cpp
    kernels/portable/cpu/op_allclose.cpp
    kernels/portable/cpu/op_amax.cpp
    kernels/portable/cpu/op_amin.cpp
    kernels/portable/cpu/op_any.cpp
    kernels/portable/cpu/op_arange.cpp
    kernels/portable/cpu/op_argmax.cpp
    kernels/portable/cpu/op_argmin.cpp
    kernels/portable/cpu/op_as_strided_copy.cpp
    kernels/portable/cpu/op_asin.cpp
    kernels/portable/cpu/op_asinh.cpp
    kernels/portable/cpu/op_atan.cpp
    kernels/portable/cpu/op_atan2.cpp
    kernels/portable/cpu/op_atanh.cpp
    kernels/portable/cpu/op_avg_pool2d.cpp
    kernels/portable/cpu/op_bitwise_and.cpp
    kernels/portable/cpu/op_bitwise_not.cpp
    kernels/portable/cpu/op_bitwise_or.cpp
    kernels/portable/cpu/op_bitwise_xor.cpp
    kernels/portable/cpu/op_bmm.cpp
    kernels/portable/cpu/op_cat.cpp
    kernels/portable/cpu/op_cdist_forward.cpp
    kernels/portable/cpu/op_ceil.cpp
    kernels/portable/cpu/op_clamp.cpp
    kernels/portable/cpu/op_clone.cpp
    kernels/portable/cpu/op_constant_pad_nd.cpp
    kernels/portable/cpu/op_convolution.cpp
    kernels/portable/cpu/op_convolution_backward.cpp
    kernels/portable/cpu/op_copy.cpp
    kernels/portable/cpu/op_cos.cpp
    kernels/portable/cpu/op_cosh.cpp
    kernels/portable/cpu/op_cumsum.cpp
    kernels/portable/cpu/op_detach_copy.cpp
    kernels/portable/cpu/op_diagonal_copy.cpp
    kernels/portable/cpu/op_div.cpp
    kernels/portable/cpu/op_elu.cpp
    kernels/portable/cpu/op_embedding.cpp
    kernels/portable/cpu/op_empty.cpp
    kernels/portable/cpu/op_eq.cpp
    kernels/portable/cpu/op_erf.cpp
    kernels/portable/cpu/op_exp.cpp
    kernels/portable/cpu/op_expand_copy.cpp
    kernels/portable/cpu/op_expm1.cpp
    kernels/portable/cpu/op_fill.cpp
    kernels/portable/cpu/op_flip.cpp
    kernels/portable/cpu/op_floor.cpp
    kernels/portable/cpu/op_floor_divide.cpp
    kernels/portable/cpu/op_fmod.cpp
    kernels/portable/cpu/op_full.cpp
    kernels/portable/cpu/op_full_like.cpp
    kernels/portable/cpu/op_gather.cpp
    kernels/portable/cpu/op_ge.cpp
    kernels/portable/cpu/op_gelu.cpp
    kernels/portable/cpu/op_glu.cpp
    kernels/portable/cpu/op_gt.cpp
    kernels/portable/cpu/op_hardtanh.cpp
    kernels/portable/cpu/op_index.cpp
    kernels/portable/cpu/op_index_put.cpp
    kernels/portable/cpu/op_index_select.cpp
    kernels/portable/cpu/op_isinf.cpp
    kernels/portable/cpu/op_isnan.cpp
    kernels/portable/cpu/op_le.cpp
    kernels/portable/cpu/op_leaky_relu.cpp
    kernels/portable/cpu/op_lift_fresh_copy.cpp
    # kernels/portable/cpu/op_linear_scratch_example.cpp
    kernels/portable/cpu/op_log.cpp
    kernels/portable/cpu/op_log10.cpp
    kernels/portable/cpu/op_log1p.cpp
    kernels/portable/cpu/op_log2.cpp
    kernels/portable/cpu/op_log_softmax.cpp
    kernels/portable/cpu/op_logical_and.cpp
    kernels/portable/cpu/op_logical_not.cpp
    kernels/portable/cpu/op_logical_or.cpp
    kernels/portable/cpu/op_logical_xor.cpp
    kernels/portable/cpu/op_logit.cpp
    kernels/portable/cpu/op_lt.cpp
    kernels/portable/cpu/op_masked_fill.cpp
    kernels/portable/cpu/op_masked_scatter.cpp
    kernels/portable/cpu/op_masked_select.cpp
    kernels/portable/cpu/op_max.cpp
    kernels/portable/cpu/op_max_pool2d_with_indices.cpp
    kernels/portable/cpu/op_max_pool2d_with_indices_backward.cpp
    kernels/portable/cpu/op_maximum.cpp
    kernels/portable/cpu/op_mean.cpp
    kernels/portable/cpu/op_min.cpp
    kernels/portable/cpu/op_minimum.cpp
    kernels/portable/cpu/op_mm.cpp
    kernels/portable/cpu/op_mul.cpp
    kernels/portable/cpu/op_native_batch_norm.cpp
    kernels/portable/cpu/op_native_dropout.cpp
    kernels/portable/cpu/op_native_group_norm.cpp
    kernels/portable/cpu/op_native_layer_norm.cpp
    kernels/portable/cpu/op_narrow_copy.cpp
    kernels/portable/cpu/op_ne.cpp
    kernels/portable/cpu/op_neg.cpp
    kernels/portable/cpu/op_nonzero.cpp
    kernels/portable/cpu/op_ones.cpp
    kernels/portable/cpu/op_pdist_forward.cpp
    kernels/portable/cpu/op_permute_copy.cpp
    kernels/portable/cpu/op_pixel_shuffle.cpp
    kernels/portable/cpu/op_pixel_unshuffle.cpp
    kernels/portable/cpu/op_pow.cpp
    kernels/portable/cpu/op_prod.cpp
    kernels/portable/cpu/op_rand.cpp
    kernels/portable/cpu/op_randn.cpp
    kernels/portable/cpu/op_reciprocal.cpp
    kernels/portable/cpu/op_reflection_pad1d.cpp
    kernels/portable/cpu/op_reflection_pad2d.cpp
    kernels/portable/cpu/op_reflection_pad3d.cpp
    kernels/portable/cpu/op_relu.cpp
    kernels/portable/cpu/op_remainder.cpp
    kernels/portable/cpu/op_repeat.cpp
    kernels/portable/cpu/op_repeat_interleave.cpp
    kernels/portable/cpu/op_replication_pad1d.cpp
    kernels/portable/cpu/op_replication_pad2d.cpp
    kernels/portable/cpu/op_replication_pad3d.cpp
    kernels/portable/cpu/op_roll.cpp
    kernels/portable/cpu/op_round.cpp
    kernels/portable/cpu/op_rsqrt.cpp
    kernels/portable/cpu/op_rsub.cpp
    kernels/portable/cpu/op_scalar_tensor.cpp
    kernels/portable/cpu/op_scatter.cpp
    kernels/portable/cpu/op_scatter_add.cpp
    kernels/portable/cpu/op_select_copy.cpp
    kernels/portable/cpu/op_select_scatter.cpp
    kernels/portable/cpu/op_sigmoid.cpp
    kernels/portable/cpu/op_sign.cpp
    kernels/portable/cpu/op_sin.cpp
    kernels/portable/cpu/op_sinh.cpp
    kernels/portable/cpu/op_slice_copy.cpp
    kernels/portable/cpu/op_slice_scatter.cpp
    kernels/portable/cpu/op_softmax.cpp
    kernels/portable/cpu/op_split_copy.cpp
    kernels/portable/cpu/op_split_with_sizes_copy.cpp
    kernels/portable/cpu/op_sqrt.cpp
    kernels/portable/cpu/op_squeeze_copy.cpp
    kernels/portable/cpu/op_stack.cpp
    kernels/portable/cpu/op_sub.cpp
    kernels/portable/cpu/op_sum.cpp
    kernels/portable/cpu/op_t_copy.cpp
    kernels/portable/cpu/op_tan.cpp
    kernels/portable/cpu/op_tanh.cpp
    kernels/portable/cpu/op_to_copy.cpp
    kernels/portable/cpu/op_topk.cpp
    kernels/portable/cpu/op_transpose_copy.cpp
    kernels/portable/cpu/op_tril.cpp
    kernels/portable/cpu/op_trunc.cpp
    kernels/portable/cpu/op_unbind_copy.cpp
    kernels/portable/cpu/op_unfold_copy.cpp
    kernels/portable/cpu/op_unsqueeze_copy.cpp
    kernels/portable/cpu/op_upsample_bilinear2d.cpp
    kernels/portable/cpu/op_upsample_bilinear2d_aa.cpp
    kernels/portable/cpu/op_upsample_nearest2d.cpp
    kernels/portable/cpu/op_var.cpp
    kernels/portable/cpu/op_view_as_real_copy.cpp
    kernels/portable/cpu/op_view_copy.cpp
    kernels/portable/cpu/op_where.cpp
    kernels/portable/cpu/op_zeros.cpp
    kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_bool.cpp
    kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp
    kernels/portable/cpu/pattern/unary_ufunc_realhbf16.cpp
    kernels/portable/cpu/util/activation_ops_util.cpp
    kernels/portable/cpu/util/advanced_index_util.cpp
    kernels/portable/cpu/util/arange_util.cpp
    kernels/portable/cpu/util/broadcast_util.cpp
    kernels/portable/cpu/util/copy_ops_util.cpp
    kernels/portable/cpu/util/delinearize_index.cpp
    kernels/portable/cpu/util/distance_util.cpp
    kernels/portable/cpu/util/dtype_util.cpp
    kernels/portable/cpu/util/index_util.cpp
    kernels/portable/cpu/util/kernel_ops_util.cpp
    kernels/portable/cpu/util/matmul_ops_util.cpp
    kernels/portable/cpu/util/normalization_ops_util.cpp
    kernels/portable/cpu/util/padding_util.cpp
    kernels/portable/cpu/util/reduce_util.cpp
    kernels/portable/cpu/util/repeat_util.cpp
    kernels/portable/cpu/util/select_copy_util.cpp
    kernels/portable/cpu/util/slice_util.cpp
    kernels/portable/cpu/util/upsample_util.cpp

    kernels/quantized/cpu/embeddingxb.cpp
    kernels/quantized/cpu/op_add.cpp
    kernels/quantized/cpu/op_choose_qparams.cpp
    kernels/quantized/cpu/op_dequantize.cpp
    kernels/quantized/cpu/op_embedding.cpp
    kernels/quantized/cpu/op_embedding2b.cpp
    kernels/quantized/cpu/op_embedding4b.cpp
    kernels/quantized/cpu/op_mixed_linear.cpp
    kernels/quantized/cpu/op_mixed_mm.cpp
    kernels/quantized/cpu/op_quantize.cpp
  )
endif()