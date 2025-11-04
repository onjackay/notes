# Papers

**Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding**

Decode Context Parallelism in vLLM. 

每个 DCP rank 连续存放若干个（16）tokens 的 KV cache。也就是说，前 16 个 decode steps 把新增的 KV cache 都存在 DCP rank 0 上，后 16 个 decode steps 把新增的 KV cache 都存在 DCP rank 1 上。

> <https://arxiv.org/html/2507.07120v1>
> <https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html#decode-context-parallel>