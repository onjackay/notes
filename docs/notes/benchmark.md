# Benchmarks

## Triton 09-persistent-matmul.py

TFLOPS on H100:

```
492.465 9767.938 ROOT
├─ 449.932 1527.330 matmul_kernel [M=8192, N=8192, K=512]
├─ 500.842 1372.080 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=512]
├─ 501.773 1369.533 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=512]
├─ 490.207 1401.846 matmul_kernel_persistent [M=8192, N=8192, K=512]
├─ 442.730 1552.175 matmul_kernel_tma [M=8192, N=8192, K=512]
├─ 527.416 1302.946 matmul_kernel_tma_persistent [M=8192, N=8192, K=512]
└─ 553.285 1242.028 torch [M=8192, N=8192, K=512]
   └─ nan 1242.028 nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN
```

在 H100 上，persistent 表现是比 non-persistent 好的

TFLOPS on 5090D:

```
141.592 48533.589 ROOT
├─ 141.233 4865.698 cuBLAS [M=8192, N=8192, K=512]
│  └─ nan 4865.698 _ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8EEvNT_6ParamsE
├─ 157.919 4351.576 matmul_kernel [M=8192, N=8192, K=512]
├─ 135.810 5059.964 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=512]
├─ 141.585 4853.572 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=512]
├─ 144.860 4743.870 matmul_kernel_persistent [M=8192, N=8192, K=512]
├─ 154.013 4461.935 matmul_kernel_tma [M=8192, N=8192, K=512]
├─ 136.668 5028.210 matmul_kernel_tma_persistent [M=8192, N=8192, K=512]
├─ 142.390 4826.136 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=512]
├─ 125.341 5482.589 matmul_kernel_tma_ws [M=8192, N=8192, K=512]
└─ 141.397 4860.038 torch [M=8192, N=8192, K=512]
   └─ nan 4860.038 _ZN7cutlass7Kernel2I66cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8EEvNT_6ParamsE
```

```
152.105 361432.388 ROOT
├─ 163.805 33561.663 cuBLAS [M=8192, N=8192, K=4096]
│  └─ nan 33561.663 _ZN7cutlass7Kernel2I65cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_tn_align8EEvNT_6ParamsE
├─ 164.485 33422.771 matmul_kernel [M=8192, N=8192, K=4096]
├─ 142.180 38666.284 matmul_kernel_descriptor_persistent [M=8192, N=8192, K=4096]
├─ 145.775 37712.702 matmul_kernel_descriptor_persistent_ws [M=8192, N=8192, K=4096]
├─ 153.797 35745.615 matmul_kernel_persistent [M=8192, N=8192, K=4096]
├─ 159.969 34366.454 matmul_kernel_tma [M=8192, N=8192, K=4096]
├─ 142.289 38636.550 matmul_kernel_tma_persistent [M=8192, N=8192, K=4096]
├─ 145.831 37698.195 matmul_kernel_tma_persistent_ws [M=8192, N=8192, K=4096]
├─ 144.405 38070.502 matmul_kernel_tma_ws [M=8192, N=8192, K=4096]
└─ 163.854 33551.651 torch [M=8192, N=8192, K=4096]
   └─ nan 33551.651 _ZN7cutlass7Kernel2I65cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x256_32x4_tn_align8EEvNT_6ParamsE
```

### Base

--- Testing float16 ---
GEMM average time over 10 iterations: 3.9855 ms
GEMM average throughput over 10 iterations: 275877.2548 GFLOPS

--- Testing bfloat16 ---
GEMM average time over 10 iterations: 6.6677 ms
GEMM average throughput over 10 iterations: 164901.8401 GFLOPS

--- Testing float16 tma ---
GEMM average time over 10 iterations: 3.9904 ms
GEMM average throughput over 10 iterations: 275537.4408 GFLOPS

--- Testing bfloat16 tma ---
GEMM average time over 10 iterations: 6.6888 ms
GEMM average throughput over 10 iterations: 164381.7855 GFLOPS

### 去除tma 版本的访存:

--- Testing float16 ---
GEMM average time over 10 iterations: 3.9882 ms
GEMM average throughput over 10 iterations: 275693.0684 GFLOPS

--- Testing bfloat16 ---
GEMM average time over 10 iterations: 6.6730 ms
GEMM average throughput over 10 iterations: 164770.2607 GFLOPS

--- Testing float16 tma ---
GEMM average time over 10 iterations: 3.9108 ms
GEMM average throughput over 10 iterations: 281150.4731 GFLOPS (+2.0%)

--- Testing bfloat16 tma ---
GEMM average time over 10 iterations: 6.5445 ms
GEMM average throughput over 10 iterations: 168005.7973 GFLOPS (+2.2%)

5090 为何 persistent 表现比 non-persistent 要更差？

```
==================================================
MATRIX BENCHMARK SUMMARY
==================================================
M values: [64, 256, 1024, 4096]
N: 4096
K: 4096
Layout: TT
Data type: torch.bfloat16

Operation                            Time (ms)        flops (TF/s)     MFU (%)
--------------------------------------------------------------------------------
(64,4096,4096) base                  0.092            23.384           22.31
(64,4096,4096) persistent            0.091            23.656           22.57
(64,4096,4096) tma                   0.103            20.757           19.81
(64,4096,4096) tma_persistent        0.103            20.893           19.94
--------------------------------------------------------------------------------
(256,4096,4096) base                  0.096            89.673           85.57
(256,4096,4096) persistent            0.093            92.186           87.96
(256,4096,4096) tma                   0.107            80.558           76.87
(256,4096,4096) tma_persistent        0.106            81.118           77.40
--------------------------------------------------------------------------------
(1024,4096,4096) base                  0.300            114.426          109.19
(1024,4096,4096) persistent            0.300            114.698          109.44
(1024,4096,4096) tma                   0.317            108.548          103.58
(1024,4096,4096) tma_persistent        0.315            108.914          103.93
--------------------------------------------------------------------------------
(4096,4096,4096) base                  0.907            151.473          144.54
(4096,4096,4096) persistent            0.906            151.688          144.74
(4096,4096,4096) tma                   0.919            149.537          142.69
(4096,4096,4096) tma_persistent        0.918            149.722          142.86
--------------------------------------------------------------------------------
```

## 5090D

### SPECS


SM Count: 170 

L1 Cache: 128 KB (per SM) 

L2 Cache: 96 MB 

FP16: 104.8 TFLOPS (1:1)  ???
