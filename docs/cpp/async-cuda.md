# Asynchronous CUDA Programming

## Async Thread and Async Proxy

> [Async Thread and Async Proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-asynchronous-execution-features-async-thread-proxy)

- 一般的同步访存操作在 **Generic Proxy**
- 异步的 LDGSTS，STAS/REDAS 在 **Generic Proxy** 的 **Async Thread**
- 异步的 TMA 和 Tensor Core 指令（wgmma, tcgen5）在 **Async Proxy** 中的 **Async Thread**

**Async thread operating in generic proxy**: 在异步访存之前的同步访存操作，保证有序。
在在异步访存之后的同步访存操作，不保证有序，需使用 Async Barrier。

**Async thread operating in async proxy**: 在异步访存之前或之后的同步访存操作，都不保证有序，需使用 Proxy Fence。

## LDGSTS (From CC80)

> [Using LDGSTS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-ldgsts)

从 cc80 开始，LDGSTS (Load GMEM Store SMEM) 指令异步的从 GMEM 拷贝到 SMEM，支持拷贝 4，8，16 字节的数据。
拷贝 4 或 8 字节使用 L1 ACCESS mode，数据会存在于 L1 Cache 中。
拷贝 16 字节时启用 L1 BYPASS mode，L1 Cache 不会被污染。

**异步性**：LDGSTS 是一个 Async thread，可以使用 Shared memory barrier 或 pipeline 来获取完成信号。
完成信号的范围是单个线程，需要 `__syncthreads()` 才能看到其他线程用 LDGSTS 读取的数据。

## TMA (From CC90)

> [Using TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma)

TMA 支持拷贝 1 维至 5 维的 tensor。
在拷贝大于等于 2 维的 tensor 时，需要提供 tensor map，用以描述该 tensor 在 GMEM 和 SMEM 中的 layout。
这个 tensor map 一般在 host 端创建，通过 `const __grid_constant__` 修饰的 kernel 参数传入 device。
拷贝 1 维的 tensor 不需要 tensor map。

TMA 支持 GMEM -> SMEM, SMEM -> GMEM, 和 SMEM -> 同个 cluster 其他 block 的 Distributed SMEM。
在一个 cluster 里，GMEM -> SMEM 的拷贝可以是 multicast，即向这个 cluster 的多个 block 的 SMEM 拷贝。

TMA 拷贝操作基于 Async Proxy。
当拷贝 GMEM -> SMEM 时，同一 block 内的线程通过 shared memory barrier 可以等待 SMEM 中的数据可用。
当拷贝 SMEM -> GMEM 时，只有发起该操作的线程能通过基于 *Bulk Async-group* 的同步机制。

### 1 维拷贝

TMA 要求 src 和 dst 地址和大小都满足 16 Byte 对齐。

在 GMEM -> SMEM 拷贝时，发起 TMA Store 的线程还需要向 Barrier 更新传输数据的 Byte 数。
Barrier 在 wait 时，不仅需要满足所有线程到达 arrive，还需要已完成相应 Byte 数大小的传输。

在 SMEM -> GMEM 拷贝前，需要 `ptx::fence_proxy_async(ptx::space_shared)` + `__syncthreads()` 使得此前的 SMEM 写入对 TMA 可见。
拷贝后则需要通过 bulk async-group 等待拷贝完成。

```cpp
#include <cuda/barrier>
#include <cuda/ptx>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;

__device__ inline bool is_elected()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0.
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF)); // Elect a leader thread among warp 0.
}

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operation should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory from a single thread.
  if (is_elected()) {
    // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).
    
    // Version 1: cuda::memcpy_async
    // memcpy_async 自动更新 barrier 的传输 byte 数
    cuda::memcpy_async(
        smem_data, data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar);
    
    // Version 2: cuda::device::memcpy_async_tx
    // cuda::device::memcpy_async_tx(
    //   smem_data, data + offset, 
    //   cuda::aligned_size_t<16>(sizeof(smem_data)),
    //   bar);
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));

    // Version 3: cuda::ptx::cp_async_bulk
    // ptx::cp_async_bulk(
    //     ptx::space_shared, ptx::space_global,
    //     smem_data, data + offset, 
    //     sizeof(smem_data), 
    //     cuda::device::barrier_native_handle(bar));
    // cuda::device::barrier_expect_tx(
    //     cuda::device::barrier_native_handle(bar),
    //     sizeof(smem_data));
  }
  
  // 3a. All threads arrive on the barrier.
  barrier::arrival_token token = bar.arrive();
  
  // 3b. Wait for the data to have arrived.
  // 当所有线程 arrive 且已完成相应 Byte 大小的传输
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory.
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory.
  if (is_elected()) {
    ptx::cp_async_bulk(
        ptx::space_global, ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}
```

> It is recommended to initiate TMA operations by a single thread in the block. While using if (threadIdx.x == 0) might seem sufficient, the compiler cannot verify that indeed only one thread is initiating the copy and may insert a peeling loop over all active threads, which results in warp serialization and reduced performance. To prevent this, we define the is_elected() helper function that uses cuda::ptx::elect_sync to select one thread from warp 0 – which is known to the compiler – to execute the copy allowing it to generate more efficient code. Alternatively, the same effect can be achieved with cooperative_groups::invoke_one.

## Asynchronous Barrier (From CC80)

> [10.26 Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-barrier)

### Arrive-Wait 分离

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* code before arrive */
       barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
       compute(data, curr_iter);
       bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/
        /* code after wait */
    }
}
```

这里的同步操作分成了两步：`bar.arrive()` 和 `bar.wait()`。在 `bar.arrive()` 之前的内存更新能确保被 `bar.wait()` 之后的代码看见。
可以理解为，`bar.wait()` 会一直阻塞当前线程，直到 `block` 内所有线程（`block.size` 个线程）到达 `bar.arrive()`。

为什么 `bar.wait()` 要接受一个右值引用的 `arrival_token`？ 这是为了确保使用者必须先使用 `bar.arrive()`，再使用 `bar.wait()`，强制配对。

### Producer-Consumer Model

```cpp
#include <cuda/barrier>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__device__ void produce(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *in, int N)
{
  for (int i = 0; i < N / buffer_len; ++i)
  {
    ready[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
    /* produce, i.e., fill in, buffer_(i%2)  */
    barrier_t::arrival_token token = filled[i % 2].arrive(); /* buffer_(i%2) is filled */
  }
}

__device__ void consume(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *out, int N)
{
  barrier_t::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
  barrier_t::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
  for (int i = 0; i < N / buffer_len; ++i)
  {
    filled[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
    /* consume buffer_(i%2) */
    barrier_t::arrival_token token3 = ready[i % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */
  }
}

__global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
{
  constexpr int warpSize = 32;

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
     while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier_t bar[4];

  if (threadIdx.x < 4)
  {
    init(bar + threadIdx.x, blockDim.x);
  }
  __syncthreads();

  if (threadIdx.x < warpSize)
  { produce(bar, bar + 2, buffer, buffer_len, in, N); }
  else
  { consume(bar, bar + 2, buffer, buffer_len, out, N); }
}

```

## Pipeline

> (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html#pipelines)

### Multi-Stages (Unified Pipeline)

在 Unified Pipeline 中，每个线程都既是生产者，同时也都是消费者。
多 stage 的异步拷贝可以使用 unified pipeline。

```cpp
#include <cuda/pipeline>

__global__ void example_kernel(const float *in)
{
    constexpr int block_size = 128;
    __shared__ __align__(sizeof(float)) float buffer[4 * block_size];

    // Create a unified pipeline per thread
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // First stage of memory copies
    pipeline.producer_acquire();
    // Every thread fetches one element of the first block
    cuda::memcpy_async(buffer, in, sizeof(float), pipeline);
    pipeline.producer_commit();

    // Second stage of memory copies
    pipeline.producer_acquire();
    // Every thread fetches one element of the second and third block
    cuda::memcpy_async(buffer + block_size, in + block_size, sizeof(float), pipeline);
    cuda::memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float), pipeline);
    pipeline.producer_commit();

    // Third stage of memory copies
    pipeline.producer_acquire();
    // Every thread fetches one element of the last block
    cuda::memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float), pipeline);
    pipeline.producer_commit();

    // Wait for the oldest stage (waits for first stage)
    pipeline.consumer_wait();
    pipeline.consumer_release();

    // __syncthreads();
    // Use data from the first stage

    // Wait for the oldest stage (waits for second stage)
    pipeline.consumer_wait();
    pipeline.consumer_release();

    // __syncthreads();
    // Use data from the second stage

    // Wait for the oldest stage (waits for third stage)
    pipeline.consumer_wait();
    pipeline.consumer_release();

    // __syncthreads();
    // Use data from the third stage
}
```

### Producer-Consumer Model (Partitioned Pipeline)

在 Paritioned Pipeline 中，一些线程属于生产者，而另一些线程属于消费者。
Warp-specialize 就属于这种情况。
这时，初始化 pipeline 对象时需要提供生产者线程的个数（如下例），或者提供当前线程的生产者/消费者角色：

```cpp
// Create a pipeline at block scope
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto pipeline = cuda::make_pipeline(group, &shared_state);
```

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

#pragma nv_diag_suppress static_var_with_dynamic_init

using pipeline = cuda::pipeline<cuda::thread_scope_block>;

__device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *in, int N)
{
  if (batch < num_batches)
  {
    pipe.producer_acquire();
    /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
    pipe.producer_commit();
  }
}

__device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *out, int N)
{
  pipe.consumer_wait();
  /* consume buffer(stage) and update out(batch) */
  pipe.consumer_release();
}

__global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
{
  auto block = cooperative_groups::this_thread_block();

  /* Shared memory buffer declared below is of size 2 * buffer_len
     so that we can alternatively work between two buffers.
     buffer_0 = buffer and buffer_1 = buffer + buffer_len */
  __shared__ extern float buffer[];

  const int num_batches = N / buffer_len;

  // Create a partitioned pipeline with 2 stages where half the threads are producers and the other half are consumers.
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int num_stages = 2;
  cuda::std::size_t producer_count = block.size() / 2;
  __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
  pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);

  // Fill the pipeline
  if (block.thread_rank() < producer_count)
  {
    for (int s = 0; s < num_stages; ++s)
    {
      produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);
    }
  }

  // Process the batches
  int stage = 0;
  for (size_t b = 0; b < num_batches; ++b)
  {
    if (block.thread_rank() < producer_count)
    {
      // Prefetch the next batch
      produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
    }
    else
    {
      // Consume the oldest batch
      consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
    }
    stage = (stage + 1) % num_stages;
  }
}
```