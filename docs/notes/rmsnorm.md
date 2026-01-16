# RMSNorm

近日被问及 RMSNorm，复盘时发觉有些细节还有待改进。Flashinfer 中有实现 RMSNorm 这一算子，现在学习一下。

## CPP

`norm.cu` 中实现了基于 TensorView 的接口，提供对接其他框架（pytorch）的 binding。

rmsnorm 支持二维和三维的输入，二维会调用 norm::RMSNorm，三维会调用 norm::QKRMSNorm。
这里我们关注二维的 norm::RMSNorm。

```cpp
// norm.cu
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

## Host

RMSNorm 先计算向量化访存的向量长度。最长是 16 字节，但要确保和 d 长度对齐，所以和 d 取最大公约数。

> 令人感叹，手撕时被问到有何优化空间时，竟然忘记了向量化访存。复盘后，最理想的回答应该是，先判断这是一个 memory bound 的算子，所以从优化访存的角度上，联想到向量化访存，减少访存的总指令数。

每个线程处理最长 16 字节的向量，就能计算出 block_size 和 num_warps。
这里，launch kernel 所用的 block_size 是两维的 (32, num_warps)，方便 kernel 内做 warp-level reduce。

```cpp
// norm.cuh
template <typename T>
cudaError_t RMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t d,
                    uint32_t stride_input, uint32_t stride_output, float eps = 1e-5,
                    bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, d, stride_input,
                                            stride_output, weight_bias, eps));
  });
  return cudaSuccess;
}
```

## Device

Kernel 实现也很简单：每个线程算各自的平方和，Block-level reduce，最后就是 element-wise 操作。
注意几点：

1. 向量化访存：使用了 `vec_t<T, VEC_SIZE>` 类，封装了 load 和 store 操作。
2. Block-level reduce：先做 warp-level reduce，最后一个 block 中的每个 warp 再做一次 reduce。显然，有 warp_nums 不超过 32，因此第二次 reduce 也能用 warp-level reduce 去做。

> 难绷的是，我说出了第一次要用 warp-level reduce，但是第二次没想到用 warp-level reduce，手写了个在 SMEM 中的蝴蝶相加。

```cpp
// norm.cuh
template <uint32_t VEC_SIZE, typename T>
__global__ void RMSNormKernel(T* __restrict__ input, T* __restrict__ weight, T* __restrict__ output,
                              const uint32_t d, const uint32_t stride_input,
                              const uint32_t stride_output, float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<T, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
```

## Epilogue

其实，RMSNorm 作为一个简单算子，需要注意的点并不多。
但在紧张刺激的环境中，一时也会想不起来。
还需多读代码，多学多练。