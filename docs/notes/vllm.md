# Experiment

## Environment

4 x NVIDIA L40 GPUs

```
$ nvidia-smi
Mon Nov  3 14:56:35 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40                     Off |   00000000:21:00.0 Off |                    0 |
| N/A   41C    P0             83W /  300W |     448MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L40                     Off |   00000000:61:00.0 Off |                    0 |
| N/A   34C    P8             36W /  300W |      17MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L40                     Off |   00000000:81:00.0 Off |                    0 |
| N/A   29C    P8             36W /  300W |      17MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L40                     Off |   00000000:E1:00.0 Off |                    0 |
| N/A   31C    P8             37W /  300W |      17MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

Topology information:

```
$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	NIC0	NIC1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NODE	SYS	SYS	SYS	SYS	0-127,256-383	0		N/A
GPU1	NODE	 X 	SYS	SYS	SYS	SYS	0-127,256-383	0		N/A
GPU2	SYS	SYS	 X 	NODE	NODE	NODE	128-255,384-511	1		N/A
GPU3	SYS	SYS	NODE	 X 	NODE	NODE	128-255,384-511	1		N/A
NIC0	SYS	SYS	NODE	NODE	 X 	PIX				
NIC1	SYS	SYS	NODE	NODE	PIX	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```

GPU0 和 GPU1，以及 GPU2 和 GPU3 之间的通信方式是 NODE，需要跨过 CPU cluster 但是不跨过 NUMA node。
GPU0/1 和 GPU2/3 之间的通信方式是 SYS，需要跨过 NUMA 节点间的 QPI/UPI 总线。

## gpt-oss-120b mxfp4

### Test TP2

运行 server 脚本：

```
vllm serve ../models/gpt-oss-120b \
    -tp 2
```

运行 client benchmark 脚本：

```
vllm bench serve --model ../models/gpt-oss-120b
```

默认配置下，request rate 为无限制，burstiness 为 1.0。
burstiness 为 1.0 表示请求到达服从泊松分布，否则服从 Gamma 分布，burstiness 越大则请求越均匀，越小则请求越集中。
共 1000 个请求，每个请求输入长度 1024，输出长度 128。
结果如下：

```
./bench_server.sh 
INFO 11-03 16:49:51 [__init__.py:216] Automatically detected platform cuda.
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x77995f557380>, seed=0, num_prompts=1000, dataset_name='random', no_stream=False, dataset_path=None, no_oversample=False, custom_output_len=256, custom_skip_chat_template=False, spec_bench_output_len=256, spec_bench_category=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, blazedit_min_distance=0.0, blazedit_max_distance=1.0, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_name=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, label=None, backend='openai', endpoint_type=None, base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', header=None, max_concurrency=None, model='../models/gpt-oss-120b', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
INFO 11-03 16:49:55 [datasets.py:507] Sampling input_len from [1024, 1024] and output_len from [128, 128]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                  | 00:01 elapsed, 140:06:18 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:32<00:00, 10.79it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  92.72     
Total input tokens:                      1022592   
Total generated tokens:                  52897     
Request throughput (req/s):              10.79     
Output token throughput (tok/s):         570.52    
Peak output token throughput (tok/s):    1799.00   
Peak concurrent requests:                1000.00   
Total Token throughput (tok/s):          11599.66  
---------------Time to First Token----------------
Mean TTFT (ms):                          44668.95  
Median TTFT (ms):                        44620.76  
P99 TTFT (ms):                           88297.92  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          162.63    
Median TPOT (ms):                        170.06    
P99 TPOT (ms):                           172.12    
---------------Inter-token Latency----------------
Mean ITL (ms):                           156.00    
Median ITL (ms):                         170.00    
P99 ITL (ms):                            173.31    
==================================================
```