# RMSNorm

近日被问及 RMSNorm，复盘时发觉有些细节还有待改进。Flashinfer 中有实现 RMSNorm 这一算子，现在学习一下。

## cpp 接口

```cpp
void rmsnorm(TensorView output, TensorView input, TensorView weight, double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(1, weight);  // weight: (hidden_size)

  auto input_ndim = input.ndim();
  if (input_ndim == 2) {
    // Normal RMSNorm: [batch_size, hidden_size]
    // Use CTA parallelization for better parallelism
    CHECK_DIM(2, output);
    TVM_FFI_ICHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size = input.size(0);
    unsigned int hidden_size = input.size(1);
    TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
      cudaError_t status = norm::RMSNorm(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
          static_cast<c_type*>(output.data_ptr()), batch_size, hidden_size, input.stride(0),
          output.stride(0), eps, enable_pdl, stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "RMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else if (input_ndim == 3) {
    // QK RMSNorm: [batch_size, num_heads, head_dim]
    // Use warp-level parallization
    CHECK_DIM(3, output);  // output: (batch_size, num_heads, hidden_size)
    TVM_FFI_ICHECK_EQ(input.size(2), weight.size(0));
    unsigned int batch_size = input.size(0);
    unsigned int num_heads = input.size(1);
    unsigned int hidden_size = input.size(2);
    TVM_FFI_ICHECK_EQ(output.size(0), batch_size);
    TVM_FFI_ICHECK_EQ(output.size(1), num_heads);
    TVM_FFI_ICHECK_EQ(output.size(2), hidden_size);

    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
      cudaError_t status = norm::QKRMSNorm(
          static_cast<c_type*>(input.data_ptr()), static_cast<c_type*>(weight.data_ptr()),
          static_cast<c_type*>(output.data_ptr()), batch_size, num_heads, hidden_size,
          input.stride(0), input.stride(1), output.stride(0), output.stride(1), eps, enable_pdl,
          stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "QKRMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported input dimension: " << input_ndim;
  }
}
```

