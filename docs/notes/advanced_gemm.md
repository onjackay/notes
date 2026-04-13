# GEMM SM120

## Theoretical
 
**Specs**

- Arch: SM120
- SM Count: 170 
- L1 Cache: 128 KB (per SM) 
- L2 Cache: 96 MB 

**Base GEMM:**

- Average SM Active Cycles: 1340238.56
- Tensor Pipe utilization: 92.04%
- Problem Size: 5120x5120x4096
- MMA Ops per SM: 5120/16 x 5120/8 x 4096/16 / 170 = 308404.7
- MMA Latency = 1340238.56 x 0.9204 / 308404.7 = 4 cycles
- Theoretical FLOPS = 16x8x16x2/4 x 170 x 1.82 Ghz = 316.8 TFLOPS

## Base

commit: f570bd12275eff736d3e0b9d5804a788b57a3a99

fp16: 0.783 us (274.2 TFLOPS)

- Register: 168
- SMEM: 98.3k (blk_size: 128,128,64)

bf16: 1.39 ms (155.6 TFLOPS)

- Register: 234
- SMEM: 98.3k (blk_size: 128,128,64)

tma fp16: 0.784 us

- Register: 166
- SMEM: 49.28k (blk_size: 128,128,32)

tma bf16: 1.38 ms

- Register: 230
- SMEM: 49.28k (blk_size: 128,128,32)

## 0x01. Epilogue use SMEM

1. Use STSM for R2S copy
2. Use TMA Store for S2G copy
3. For bf16, convert accumulator fp32 to bf16 in register before STSM

commit: 7f4bcf4d21156f329f8c0f4a87d6a6f537f48282

tma fp16: 0.830 us ()

- Register: 166
- SMEM: 82.0k (blk_size: 128,128,32)

tma bf16: 1.43 ms

- Register: 230
- SMEM: 82.0k (blk_size: 128,128,32)

SMEM 大小增加，使 1 个 SM 上只能调度 1 个 block，性能下降 5.8% (fp16) / 3.6% (bf16)