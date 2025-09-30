# From Online Softmax to Flash Attention

## Online Softmax

### 3-Pass Safe Softmax

Softmax 的标准定义是
$$
y_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$
但是当 $x_i$ 很大时，$\exp(x_i)$ 会溢出。为了避免这种情况，通常会先计算输入的最大值 $m = \max_i (x_i)$，然后使用
$$
y_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}
$$
来计算 softmax，确保 $x_i - m \le 0$，从而避免溢出。
朴素的实现这种方法需要遍历三次输入：

1. 计算最大值 $m_i = \max(m_{i-1}, x_i)$。
2. 计算归一化因子 $d_i = d_{i-1} + \exp(x_j - m_N)$。
3. 计算最终的 softmax 输出 $y_i = \exp(x_i - m_N) / d_N$。

### 2-Pass Safe Online Softmax

我们可以把前两次遍历融合成一次，在一次遍历中同时更新最大值和归一化因子，实现 2-Pass Safe Softmax：

1. 计算最大值和归一化因子
    - $m_i = \max(m_{i-1}, x_i)$
    - $d_i = d_{i-1} \cdot \exp(m_{i-1} - m_i) + \exp(x_i - m_i)$
2. 计算最终的 softmax 输出 $y_i = \exp(x_i - m_N) / d_N$。

> 把 3-Pass 优化成 2-Pass 有什么收益吗？从计算量上看，2-Pass 甚至还要比 3-Pass 多一些计算。但从全局内存访问的角度来看，2-Pass 只需要遍历两次输入，而 3-Pass 需要遍历三次，减少了一次内存访问。对于 softmax 这种 memory-bound 的操作来说，减少内存访问往往能带来更大的性能提升。

## FlashAttention V1

### 1-Pass Attention

忽略缩放常数 $\sqrt{d}$，标准 Attention 的计算公式是
$$
O = \text{Softmax}(QK^T)V
$$
其中 $Q, K, V$ 的形状分别是 $(M, D)$，$(N, D)$，$(N, D)$。
那么在 2-Pass Softmax 的基础上，每一行 $O_i$ 的计算可以分解成以下两步：

1. 计算 $QK^T$ 的最大值和归一化因子
    - $x_j = Q_i K_j^T$
    - $m_j = \max(m_{j-1}, x_j)$
    - $d_j = d_j \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$
2. 计算 $O$
    - $O_j = O_{j-1} + \frac{\exp(x_j - m_D)}{d_N} V_j$

虽然 Softmax 不能被进一步优化成 1-Pass，但是 Attention 只需要得到最终的输出 $O$，而不需要中间的 Softmax 结果。实际上 Attention 是可以被优化成 1-Pass 的。
定义 $O'_j$

$$
O'_j = \sum_{k=1}^{j} \frac{\exp(x_k-m_j)}{d_j} V_k
$$

可以得到递推式：

$$
O'_j = O'_{j-1} \cdot \exp(m_{j-1}-m_j) \cdot \frac{d_{j-1}}{d_j} + \frac{\exp(x_j - m_j)}{d_j} V_j
$$

由此，我们可以在一次遍历中同时计算 $m_j, d_j, O'_j$，从而实现 1-Pass Attention：

1. 计算 $QK^T$ 的最大值、归一化因子和输出
    - $x_j = Q_i K_j^T$
    - $m_j = \max(m_{j-1}, x_j)$
    - $d_j = d_j \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$
    - $O'_j = O'_{j-1} \cdot \exp(m_{j-1}-m_j) \cdot \frac{d_{j-1}}{d_j} + \frac{\exp(x_j - m_j)}{d_j} V_j$

### Tiling

我们把 K 和 V 分块，每块大小为 $(n, D)$，再把 Q 每块分为 $(m, D)$。
我们先分析对于 $Q_i, K_j, V_j$ 分块的计算过程：

1. 计算 $QK^T$
    - $X = Q_i K_j^T \in \R^{(m,n)}$
2. 对每行求
    - $\tilde{m_i}[k] = \max_l(X_{k,l}) \in \R^m$
    - $P_{k,l} = \exp(X_{k,l} - \tilde{m_i}) \in \R^{(m,n)}$
    - $d_i[k] = \sum_l P_{k,l} \in \R^m$
3. 对每行更新最大值和归一化因子
    - $m_i[k] = \max(m'_i[k], \tilde{m_i}[k])$
    - $d_i[k] = d'_i[k] \cdot \exp(m'_i[k] - m_i[k]) + d_i[k] \cdot \exp(\tilde{m_i}[k] - m_i[k])$
4. 对每行计算输出
    - $O_i[k] = d_i^{-1} ((d'_i \cdot \exp(m'_i - m_i)) \cdot O_i[k] + \exp(\tilde{m_i} - m_i) \cdot PV) $
5. 给下个块更新这个块的最大值和归一化因子
    - $m'_i = m$
    - $d'_i = d$

FlashAttention V1 把 K 和 V 的分块遍历放在外层循环，把 Q 的分块遍历放在内层循环。

## FlashAttention V2

1. 减少 Cuda Core 计算

    在 V1，每个 tile 中 $O_i$ 的计算都包含两次 rescale 操作：乘 $d'$ 和除 $d$。
    V2 在每个 tile 中仅保留乘 $d'$ 的 rescale，把所有的 $d$ 的 rescale 一起放在最后。

2. 调换循环顺序

    V1 先循环 K 和 V，后循环 Q 的做法使得每次内循环都需要反复向 HBM 访存 $O_i, m'_i, d'_i$。
    V2 调换了内外循环的顺序，先循环 Q，再循环 K 和 V，只在每次外循环中向 HBM 访存 $O_i, m'_i, d'_i$。

3. 增加 Sequence Length 维度并行

    V1 只在 Batch Size 和 Head Num 维度上并行，V2 增加了 Sequence Length 维度的并行

## FlashAttention V3

TODO

## FlashAttention vs SDPA

FlashAttention 是否一定比标准的 SDPA 快？答案是否定的。
FlashAttention 希望通过减少全局内存访问来提升性能，FA1 论文中其实给出了 FlashAttention 和 SDPA 访存量的渐进分析：

| Method         | Memory Access             |
|----------------|---------------------------|
| SDPA           | $\Theta(Nd + N^2)$        | 
| FlashAttention | $\Theta(N^2 d^2 M^{-1})$  |

其中 $N$ 是 `seq_len`，$d$ 是 `head_dim`，$M$ 是 GPU 的 SRAM 大小。
对于主流的模型和推理框架，`seq_len` 一般在几千以内（8192），`head_dim` 是 64/128，GPU 的 SRAM 大概是 200kB（A100: 192kB, H100: 228kB）。
这时 $d^2 M^{-1}$ 是远小于 1 的，所以 FlashAttention 的访存量要远小于 SDPA。

但是如果 `head_dim` 比较大，比如 256 或 512，或者 SRAM 比较小，比如在旧架构的 GPU 上运行，FlashAttention 的访存量可能会大于 SDPA。

## FlashAttention V4

Tri Dao 已经预告 FlashAttention V4 了，但还未正式发布。