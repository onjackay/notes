# Fused MoE Kernels

## Sglang

Sglang 的 MoE kernel 选取可见 `python/sglang/srt/layers/moe/ep_moe/layer.py` 的 `get_moe_impl_class()`.

- DeepEPMoE
- FlashInferFP4MoE
- FlashInferFusedMoE
- FusedMoE