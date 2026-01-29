# Asynchronous CUDA Programming

## LDGSTS

> [4.11.1. Using LDGSTS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-ldgsts)

从 cc80 开始，LDGSTS (Load GMEM Store SMEM) 指令异步的从 GMEM 拷贝到 SMEM，支持拷贝 4，8，16 字节的数据。
拷贝 4 或 8 字节使用 L1 ACCESS mode，数据会存在于 L1 Cache 中。
拷贝 16 字节时启用 L1 BYPASS mode，L1 Cache 不会被污染。

**异步性**：LDGSTS 是一个 Async thread，可以使用 Shared memory barrier 或 pipeline 来获取完成信号。
完成信号的范围是单个线程，需要 `__syncthreads()` 才能看到其他线程用 LDGSTS 读取的数据。

## Asynchronous Barrier

> [10.26 Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-barrier)

### Simple Synchronization Pattern

```cu
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}
```

这里的 `block` 的范围就是一般意义上的 block，`block.sync()` 等价于 `__syncthreads()`。

### Temporal Splitting and Five Stages of Synchronization

```cu
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
