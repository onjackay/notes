# Parallel Thread Execution (PTX)

## 9.7.12 Synchronization and Communication

### 9.7.12.13 griddepcontrol

```
griddepcontrol.action;

.action   = { .launch_dependents, .wait }
```

The griddepcontrol instruction allows the dependent grids and prerequisite grids as defined by the runtime, to control execution in the following way:

`.launch_dependents` modifier signals that specific dependents the runtime system designated to react to this instruction can be scheduled as soon as all other CTAs in the grid issue the same instruction or have completed. The dependent may launch before the completion of the current grid. There is no guarantee that the dependent will launch before the completion of the current grid. Repeated invocations of this instruction by threads in the current CTA will have no additional side effects past that of the first invocation.

`.wait` modifier causes the executing thread to wait until all prerequisite grids in flight have completed and all the memory operations from the prerequisite grids are performed and made visible to the current grid.

使用这两条 ptx 指令的目的是启用 [PDL (Programmatic Dependent Kernel Launch)](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#programmatic-dependent-kernel-launch)。
使得同一个 stream 中，前后两个有依赖关系，但仍有可重叠部分的 kernel，可以一部分重叠。
需要 SM90 及以上 CC。
例如以下例子：

```cpp
__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel

    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
    // Initialization, Independent work, etc.

    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();

    // Dependent work
}

// Launch the secondary kernel with the special attribute

// Set Up the attribute
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;

// Set the attribute in a kernel launch configuration
 cudaLaunchConfig_t config = {0};

// Base launch configuration
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes= 0;
config.stream = stream;

// Add special attribute for PDL
config.attrs = attribute;
config.numAttrs = 1;

// Launch primary kernel
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

// Launch secondary (dependent) kernel using the configuration with
// the attribute
cudaLaunchKernelEx(&config, secondary_kernel);
```

`cudaGridDependencySynchronize()` 函数封装的就是 `griddepcontrol.wait` 指令，等待第一个 kernel 被依赖的部分执行完成。
`cudaTriggerProgrammaticLaunchCompletion()` 函数封装的是 `griddepcontrol.launch_dependents` 指令，提示第二个 kernel 可以往下运行。