# Data formats

## Tensor Cores Supported Data Formats

| Architecture | Supported Formats |
|--------------|-------------------|
| Blackwell | FP64, TF32, BF16, FP16, FP8, INT8, FP6, FP4 |
| Hopper | FP64, TF32, BF16, FP16, FP8, INT8 |
| Ampere | FP64, TF32, BF16, FP16 |

## 4-bit Data Formats

| Format | Description |
|--------|-------------|
| FP4 (E2M1) | 4-bit floating point with 2 exponent bits and 1 mantissa bit. Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6. |
| MXFP4 | FP4_E2M1. 1 FP8_E4M3 scale per 32 elements. |
| NVFP4 | FP4_E2M1. 1 FP8_E4M3 scale per 16 elements. 1 FP32 scale per tensor. |
