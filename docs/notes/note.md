# Note

IBGDA: InfiniBand GPUDirect Async.  IBGDA实现了GPU SM到NIC的直接控制，无需CPU介入

# Blackwell

## Tensor Memory

每个 SM 有 256KB 大小的 TMEM，128 行 (rows/lanes) 512 列 (columns)，每个单元 32 bits，每行 2KB。
32 位地址的 31-16 位为 lane ID，15-0 位位 column ID。

TMEM 由 `tcgen05.alloc` 指令动态分配，一次分配若干列上的所有 128 行，列数为 2 的幂且最小为 32。
TMEM 由 `tcgen05.dealloc` 显式释放。
`tcgen05.alloc` 和 `tcgen05.dealloc` 需要一个 warp 发起，分配和释放需要由同一个 warp 发起。

- `tcgen05.cp`: SMEM -> TMEM，由一个线程发起
- `tcgen05.ld`: TMEM -> RMEM
- `tcgen05.st`: RMEM -> TMEM
- 一个 warp 通过 `tcgen05.ld` 和 `tcgen05.st` 只能访问 32 个 lanes (warp 0 访问 lane 0-31)，一个 warpgroup 访问完整的 128 lanes。

# DeepEP

DeepEP 是一个 MoE 专家并行（EP）通信库，三种模式本质上是 **dispatch（分发）** → GEMM 计算 → **combine（合并）** 的 pipeline 中不同通信路径的优化。

---

## 1️⃣ **Intranode 模式**（单节点内 NVLink）

单节点内 8 GPU，只用 NVLink 通信。

### 输入

| 参数 | 形状 | 含义 |
|---|---|---|
| `x` | `(num_tokens, hidden)` | 输入 token 张量，默认 `4096 × 7168`，dtype 为 BF16 或 FP8 |
| `topk_idx` | `(num_tokens, num_topk)` | 每个 token 选中的专家索引，默认 `4096 × 8` |
| `topk_weights` | `(num_tokens, num_topk)` | 每个 token 对选中专家的门控权重 |
| `num_tokens_per_rank` | `(num_ranks,)` | 每个 rank 将接收的 token 数 |
| `is_token_in_rank` | `(num_tokens, num_ranks)` | bool mask = token i 是否发送到 rank j |
| `num_tokens_per_expert` | `(num_experts,)` | 每个专家接收的 token 总数（全局） |
| `handle` | tuple | dispatch 返回后透传给 combine，含 `rank_prefix_matrix` |

### 输出（dispatch）

| 参数 | 形状 | 含义 |
|---|---|---|
| `recv_x` | `(recv_num_tokens, hidden)` | 当前 rank 收到的 token，`recv_num_tokens` = 选中且路由到本 rank 的 token 总数 |
| `recv_topk_idx` | `(recv_num_tokens, num_topk)` | 专家索引已映射为 **本地偏移**（范围 `[0, num_experts_per_rank)`） |
| `recv_topk_weights` | `(recv_num_tokens, num_topk)` | 对应 token 的门控权重 |
| `recv_num_tokens_per_expert_list` | `List[int]`, 长度 `num_experts_per_rank` | 本 rank 每个本地专家收到的 token 数 |
| `handle` | tuple | 含 `rank_prefix_matrix`(num_ranks, num_ranks)，combine 用 |

### 输出（combine）

| 参数 | 形状 | 含义 |
|---|---|---|
| `combined_x` | `(num_tokens, hidden)` | 合并回原始形状的张量 |
| `combined_topk_weights` | `(num_tokens, num_topk)` | 合并后的权重 |

### 关键：handle 内容

- **`rank_prefix_matrix`**: `(num_ranks, num_ranks)` 的 int 矩阵，`rank_prefix_matrix[i][rank]` 表示第 i 个 rank 发送到当前 rank 的数据在当前 rank 接收区中的起始偏移。combine 时用于反向映射。

---

## 2️⃣ **Internode 模式**（跨节点 NVLink + RDMA）

多节点场景，NVLink 负责节点内，RDMA 负责节点间。

### 输入

与 intranode 完全相同，**额外增加**：

| 参数 | 形状 | 含义 |
|---|---|---|
| `num_tokens_per_rdma_rank` | `(num_nodes,)` | 每个节点（RDMA rank）将接收的 token 数 |

默认配置（line 27-28）：`num_tokens=4096, hidden=7168, num_experts=256, num_topk=8, num_topk_groups=4`。

### 输出（dispatch）

与 intranode 的 dispatch 输出完全相同：

| 参数 | 形状 | 含义 |
|---|---|---|
| `recv_x` | `(recv_num_tokens, hidden)` | |
| `recv_topk_idx` | `(recv_num_tokens, num_topk)` | |
| `recv_topk_weights` | `(recv_num_tokens, num_topk)` | |
| `recv_num_tokens_per_expert_list` | `List[int]` | |
| `handle` | tuple | 含 `recv_gbl_rank_prefix_sum` `(num_ranks,)` |

### 输出（combine）

与 intranode 的 combine 输出完全相同：

| 参数 | 形状 | 含义 |
|---|---|---|
| `combined_x` | `(num_tokens, hidden)` | |
| `combined_topk_weights` | `(num_tokens, num_topk)` | |

### 关键：handle 内容

- **`recv_gbl_rank_prefix_sum`**: `(num_ranks,)` 数组，`recv_gbl_rank_prefix_sum[i]` 表示 rank 0..i 的累积收到 token 数。combine 时用于定位每个 rank 数据在接收区中的偏移。

---

## 3️⃣ **Low-Latency 模式**（纯 RDMA，低延迟解码）

专为 inference decoding 优化，批量小、延迟极低。

### 输入

| 参数 | 形状 | 含义 |
|---|---|---|
| `hidden_states` (`current_x`) | `(num_tokens, hidden)` | **批量小得多**，默认 `128 × 7168`（而非 4096） |
| `topk_idx` | `(num_tokens, num_topk)` | 默认 `128 × 8` |
| `topk_weights` | `(num_tokens, num_topk)` | 门控权重 |
| `num_max_dispatch_tokens_per_rank` | int | 最大 dispatch token 数（预分配用），通常 = `num_tokens` |
| `num_experts` | int | 全局专家数，默认 288 |
| `cumulative_local_expert_recv_stats` | `(num_local_experts,)` | 可选输出，每个本地专家的累计收到 token 数 |

### 输出（`low_latency_dispatch`）

| 参数 | 形状 | 含义 |
|---|---|---|
| `packed_recv_x` | `(num_local_experts, num_tokens, hidden)` (BF16)<br>或 FP8 tuple | 按**本地专家**分组的张量。每个专家独占一个 "slot"，大小预分配为 `num_tokens`（即使实际收到的更少） |
| `packed_recv_count` | `(num_local_experts,)` | 每个专家实际收到的有效 token 数 |
| `handle` | tuple | 包含 `recv_src_info` 和 `recv_layout_range`，见下文 |
| `event` | `EventOverlap` | CUDA 同步事件 |
| `hook` | callable | **接收完成回调**，调用后才真正接收数据，用于计算/通信重叠（不占 SM） |

### 输出（`low_latency_combine`）

| 参数 | 形状 | 含义 |
|---|---|---|
| `combined_x` | `(num_tokens, hidden)` | 合并回原始形状 |
| `event` | `EventOverlap` | 同步事件 |
| `hook` | callable | 同上，用于通信/计算重叠 |

### 关键：handle 内容

- **`recv_src_info`**: `(num_local_experts, num_tokens)` 每个专家 slot 中，每个 token 来自哪个 rank 的哪个 token 位置
- **`recv_layout_range`**: `(num_local_experts, num_nodes)` 每个专家在每个节点上的数据范围（高 32 位 = 起始偏移，低 32 位 = count）

---

## 🔑 三种模式的核心差异对比

| 维度 | Intranode | Internode | Low-Latency |
|---|---|---|---|
| **通信通道** | NVLink only | NVLink + RDMA | RDMA only |
| **典型 token 数** | 4096 | 4096 | **128** |
| **dispatch 输出按什么组织** | token 列表（连续打包） | token 列表（连续打包） | **按本地专家**预分配 slot |
| **dispatch 输出 `[0]` 维度** | `recv_num_tokens`（动态） | `recv_num_tokens`（动态） | `num_local_experts`（固定） |
| **dispatch 输出 `[0, 0]` 维度** | `hidden` | `hidden` | `num_tokens`（预分配） |
| **是否有 hook 机制** | ❌ | ❌ | ✅ 通信/计算重叠 |
| **combine 输出** | `(num_tokens, hidden)` | `(num_tokens, hidden)` | `(num_tokens, hidden)` 或写入 pre-allocated `out` |
| **支持 FP8 dispatch** | ✅ (SM90) | ✅ | ✅ |
| **支持 LogFMT combine** | ❌ | ❌ | ✅ |