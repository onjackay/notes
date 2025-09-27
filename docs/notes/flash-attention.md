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
    - $\tilde{m}_k = \max_l(X_{k,l}) \in \R^m$
    - $P_{k,l} = \exp(X_{k,l} - \tilde{m}) \in \R^{(m,n)}$
    - $d_k = \sum_l P_{k,l} \in \R^m$
3. 对每行更新最大值和归一化因子
    - $m_k = \max(m'_k, \tilde{m})$
    - $d_k = d'_k \cdot \exp(m'_k - m_k) + d_k \cdot \exp(\tilde{m}_k - m_k)$
4. 对每行计算输出
    - $O_i[k] = d^{-1} ((d' \cdot \exp(m' - m)) \cdot O_i[k] + \exp(\tilde{m} - m) \cdot PV) $
5. 给下个块更新这个块的最大值和归一化因子
    - $m'_k = m_k$
    - $d'_k = d_k$

FlashAttention V1 把 K 和 V 的分块遍历放在外层循环，把 Q 的分块遍历放在内层循环