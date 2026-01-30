# Fused MoE Kernels

## Sglang

Sglang 的 MoE kernel 选取可见 `python/sglang/srt/layers/moe/ep_moe/layer.py` 的 `get_moe_impl_class()`.

- DeepEPMoE：需要 DeepEP 或 Mooncake 的 A2A backend
- FlashInferFP4MoE：需要 SM100 和 ModelOptNvFp4FusedMoEMethod 量化，后端调用 trtllm_fp4_block_scale_moe
- FlashInferFusedMoE：需要 SM100，后端调用 trtllm_bf16_moe
- FlashInferCutlass：可能暂未支持，目前路由到 FusedMoE
- FusedMoE：Triton kernel

## vLLM

vLLM 的 MoE 封装成 Fused MoE Modular Kernel, 包含三个模块：

1. TopKWeightAndReduce：A2A combine 之前的 reduce
2. FusedMoEPrepareAndFinalize：动态激活量化，A2A dispatch 和 combine
3. FusedMoEPermuteExpertsUnpermute: 一般意义上的 Fused MoE Kernel

这里，我们仅关注 FusedMoEPermuteExpertsUnpermute 模块，它包含了 Fused MoE 的核心逻辑。

From `moe_kernel_features.md` in v0.15.0:

| Kernel | Input act. format | Quant. types | Quant. format | Activation function | Apply Weight On Input | Modular | Source |
|--------|-------------------|--------------|---------------|---------------------|-----------------------|---------|--------|
| triton | standard | all<sup>1</sup> | G,A,T | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y | Y | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.fused_moe.TritonExperts] |
| triton (batched) | batched | all<sup>1</sup> | G,A,T | silu, gelu | <sup>6</sup> | Y | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.fused_batched_moe.BatchedTritonExperts] |
| deep gemm | standard,</br>batched | fp8 | G(128),A,T | silu, gelu | <sup>6</sup> | Y | </br>[`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.deep_gemm_moe.DeepGemmExperts],</br>[`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4 | standard,</br>batched | nvfp4 | A,T | silu | Y | Y | [`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassExpertsFp4] |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | silu, gelu | Y | Y | [`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassExpertsFp8],</br>[`CutlasBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.cutlass_moe.CutlassBatchedExpertsFp8] |
| flashinfer | standard | nvfp4,</br>fp8 | T | <sup>5</sup> | N | Y | [`FlashInferExperts`][vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe.FlashInferExperts] |
| gpt oss triton | standard | N/A | N/A | <sup>5</sup> | Y | Y | [`triton_kernel_fused_experts`][vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts],</br>[`OAITritonExperts`][vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe.OAITritonExperts] |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.fused_marlin_moe.BatchedMarlinExperts] |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>5</sup> | N | Y | [`TrtLlmGenExperts`][vllm.model_executor.layers.fused_moe.trtllm_moe.TrtLlmGenExperts] |
| rocm aiter moe | standard | fp8 | G(128),A,T | silu, gelu | Y | N | [`rocm_aiter_fused_experts`][vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts] |
| cpu_fused_moe | standard | N/A | N/A | silu | N | N | [`CPUFusedMOE`][vllm.model_executor.layers.fused_moe.cpu_fused_moe.CPUFusedMOE] |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.fused_batched_moe.NaiveBatchedExperts] |

1. All types: mxfp4, nvfp4, int4, int8, fp8
2. A dispatcher wrapper around triton and deep gemm experts. Will select based on type + shape + quantization params
3. uint4, uint8, fp8, fp4
4. This is a naive implementation of experts that supports batched format. Mainly used for testing.
5. The `activation` parameter is ignored and SwiGlu is used by default instead.
6. Only handled by or supported when used with modular kernels.

可用组合：

| backend | `FusedMoEPrepareAndFinalize` subclasses | `FusedMoEPermuteExpertsUnpermute` subclasses |
|---------|-----------------------------------------|----------------------------------------------|
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` |  `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts`,</br>`CutlassExpertsFp8`, </br>`MarlinExperts` |
| deepep_low_latency,</br>pplx | `DeepEPLLPrepareAndFinalize`,</br>`PplxPrepareAndFinalize` |  `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts`,</br>`CutlassBatchedExpertsFp8`,</br>`BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |

