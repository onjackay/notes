# CuTe

## CuTe Layout

Layout 用于描述 Tensor 在内存中的排布，提供了从逻辑坐标到物理坐标的映射。
对于二维矩阵，我们可以简单的用 row-major 或 column-major 来描述其物理映射。
但是对于更高维的张量，这样的描述方式显然不够用了。
于是，我们有了使用 Shape 和 Stride 来描述物理映射的方式。

### Layout in PyTorch

PyTorch 中，Tensor 类包含有 `shape` 属性 和 `stride()` 方法。
shape 描述了 tensor 在逻辑上各个维度的大小。
stride 描述了 tensor 在各个维度上的相邻元素在物理坐标上的差值。
对于一个大小（shape）为 `[3, 4]` 的矩阵，row-major 排布所对应的 stride 为 `[4, 1]`，
表示同一行的相邻两个元素在物理坐标上相差 1，同一列的相邻两个元素在物理坐标上相差 4。
如果这个矩阵是 column-major 排布的，则 stride 为 `[1, 3]`。
这里的 shape 和 stride 可以自然的拓展至高维 tensor。
每个逻辑坐标对应的物理坐标也很容易得到：

$$
Id_{physical} = \sum_i Coord_{i} Stride_{i}.
$$

用 shape 和 stride 来描述 tensor 很好的刻画了一位存储结构和高维逻辑结构的关系，
同时使得在一些 reshape 操作时，避免了对数据本身进行操作，仅需通过改写 shape 和 stride 属性（PyTorch 中的 `view`）。
然而，在这套排布描述中，每一个维度只能有一个 stride 值，使得更复杂的多层次排布无法被描述。

### Hierarchical Layout in CuTe

有时，我们会希望一个大 tensor 按某种排布被分块成若干小 tensor，而每个小 tensor 内部又遵循某种排布。
拿 GEMM 作为例子，我们会在输出矩阵 C 上做分块，使得每个 SM 有足够资源计算一个块。
为了提升 L2 Cache 命中率，快与块之间的排布可能并不是单纯的 row-major 或者 column-major，而是采用 Threadblock Swizzling：
若干个块先按 row-major 的排布组成一个较大的 block，然后所有的 block 再通过 row-major 的排布组成整个大 tensor。
这种复合的排布无法再被 shape 和 stride 的语言描述了，CuTe 所采用的多层级排布描述应运而生。

![Layout Example](cute/layout_example.jpg)

在 CuTe Layout 中，每个维度对应的 shape 不再必须为一个整数，也可以包含多个整数。
在上图 c 的例子中，行维度上 shape 是 4，stride 是 2，行为与上节所介绍的一致。
列维度上，shape 是 (2, 4)，stride 是 (1, 8)，这里的 shape 和 stride 是按照从里往外的顺序：
连续的 2 列先构成一个小 tensor，连续 4 个小 tensor 再构成所有列。
在每个小 tensor 中，同一行相邻两列的物理坐标相差 1。
对于在列上相邻的两个小 tensor，相同位置的元素物理坐标相差 8。

```cpp
/** crd2idx(c,s,d) maps a coordinate within <Shape,Stride> to an index
 *
 * This is computed as follows:
 *  [coord, shape, and stride are all integers => step forward by stride]
 * op(c, s, d)             => c * d
 *  [coord is integer, shape and stride are tuple => divmod coord for each mode]
 * op(c, (s,S), (d,D))     => op(c % prod(s), s, d) + op(c / prod(s), (S), (D))
 *  [coord, shape, and stride are all tuples => consider each mode independently]
 * op((c,C), (s,S), (d,D)) => op(c, s, d) + op((C), (S), (D))
 */
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord  const& coord,
        Shape  const& shape,
        Stride const& stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
CUTE_HOST_DEVICE constexpr
auto
crd2idx_ttt(Coord  const& coord,
            Shape  const& shape,
            Stride const& stride, seq<Is...>)
{
  return (... + crd2idx(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
CUTE_HOST_DEVICE constexpr
auto
crd2idx_itt(CInt   const& coord,
            STuple const& shape,
            DTuple const& stride, seq<I0,Is...>)
{
  if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
    return crd2idx(coord, get<I0>(shape), get<I0>(stride));
  } else if constexpr (is_constant<0, CInt>::value) {
    return crd2idx(_0{}, get<I0>(shape), get<I0>(stride))
         + (_0{} + ... + crd2idx(_0{}, get<Is>(shape), get<Is>(stride)));
  } else {                             // General case
    auto [div, mod] = divmod(coord, product(get<I0>(shape)));
    return crd2idx(mod, get<I0>(shape), get<I0>(stride))
         + crd2idx_itt(div, shape, stride, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord  const& coord,
        Shape  const& shape,
        Stride const& stride)
{
  if constexpr (is_tuple<Coord>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple tuple
      static_assert(tuple_size<Coord>::value == tuple_size< Shape>::value, "Mismatched Ranks");
      static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return detail::crd2idx_ttt(coord, shape, stride, tuple_seq<Coord>{});
    } else {                                     // tuple "int" "int"
      static_assert(sizeof(Coord) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {      // "int" tuple tuple
      static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return detail::crd2idx_itt(coord, shape, stride, tuple_seq<Shape>{});
    } else {                                     // "int" "int" "int"
      return coord * stride;
    }
  }

  CUTE_GCC_UNREACHABLE;
}
```