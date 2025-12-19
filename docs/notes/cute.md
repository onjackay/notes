# CuTe Basic

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
Id_{physical} = \sum_i Coord_{i} \times Stride_{i}.
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
连续的 2 列先构成一个块，连续 4 个块 再构成所有列。
在每个块中，同一行相邻两列的物理坐标相差 1。
对于在列上相邻的两个块，相同位置的元素物理坐标相差 8。
类似的，上图 d 的例子在行维度上切成了 2 个由连续 2 行构成的块。
块内的 stride 为 1，块间的 stride 为 4。
在列维度上切成了 4 个由连续 2 列构成的块。
块内的 stride 为 2，块间的 stride 为 8。

接下来，我们来分析 CuTe 中 Layout 的实现，给出 Layout 较为严谨的定义。

### Shape, Stride, Layout

前置知识：静态整数 `Int<2>{}` (`_2`) 和动态整数 `int{2}` (`2`)，统称整数 (Integer)。

定义：`IntTuple` 是一个整数，或者是由 IntTuple 组成的元组。

例如，`2`, `_3`, `make_tuple(2, _3)`, `make_tuple(42, make_tuple(_1, 3), _17)` 都是 IntTuple。

在 CuTe 中，`Shape` 和 `Stride` 都是 IntTuple。
`Layout` 是一个 `Shape` 和 `Stride` 组成的元组，描述了逻辑坐标 `Coord` 到一维物理坐标的映射。
这个映射由下面的代码实现，总结成三条规则：

1. 当 Coord, Shape, Stride 都是整数时，物理坐标为 `Coord x Stride`。
2. 当 Coord 是整数，Shape 和 Stride 是元组时，则当前维度上需要分块。按照最低维在先的规则，从最里面的块开始，需要从整数 Coord 计算出块内偏移和块的坐标（第几块），向外迭代，累加物理坐标。
3. 当 Coord, Shape, Stride 都是元组时，则迭代元组内的每一项，累加物理坐标。

下面代码的注释解释的比较清楚：

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

相反，也有从一维物理坐标到逻辑坐标到映射，同样按照最低维在先的规则：

```cpp
/** idx2crd(i,s,d) splits an index into a coordinate within <Shape,Stride>.
 *
 * This is computed as follows:
 *  [index, shape, and stride are all integers => determine 1D coord]
 * op(i, s, d)             => (i / d) % s
 *  [index is integer, shape and stride are tuple => determine component for each mode]
 * op(i, (s,S), (d,D))     => (op(i, s, d), op(i, S, D)...)
 *  [index, shape, and stride are all tuples => consider each mode independently]
 * op((i,I), (s,S), (d,D)) => (op(i, s, d), op((I), (S), (D)))
 *
 * NOTE: This only works for compact shape+stride layouts. A more general version would
 *       apply to all surjective layouts
 */
template <class Index, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
idx2crd(Index  const& idx,
        Shape  const& shape,
        Stride const& stride)
{
  if constexpr (is_tuple<Index>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple tuple
      static_assert(tuple_size<Index>::value == tuple_size< Shape>::value, "Mismatched Ranks");
      static_assert(tuple_size<Index>::value == tuple_size<Stride>::value, "Mismatched Ranks");
      return transform(idx, shape, stride, [](auto const& i, auto const& s, auto const& d){ return idx2crd(i,s,d); });
    } else {                                     // tuple "int" "int"
      static_assert(sizeof(Index) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {
      if constexpr (is_tuple<Stride>::value) {   // "int" tuple tuple
        static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
        return transform(shape, stride, [&](auto const& s, auto const& d){ return idx2crd(idx,s,d); });
      } else {                                   // "int" tuple "int"
        return transform(shape, compact_col_major(shape, stride), [&](auto const& s, auto const& d){ return idx2crd(idx,s,d); });
      }
    } else {                                     // "int" "int" "int"
      if constexpr (is_constant<1, Shape>::value) {
        // Skip potential stride-0 division
        return Int<0>{};
      } else {
        return (idx / stride) % shape;
      }
    }
  }

  CUTE_GCC_UNREACHABLE;
}

/** idx2crd(i,s) splits an index into a coordinate within Shape
 * via a colexicographical enumeration of coordinates in Shape.
 * c0 = (idx / 1) % s0
 * c1 = (idx / s0) % s1
 * c2 = (idx / (s0 * s1)) % s2
 * ...
 */
template <class Index, class Shape>
CUTE_HOST_DEVICE constexpr
auto
idx2crd(Index const& idx,
        Shape const& shape)
{
  if constexpr (is_tuple<Index>::value) {
    if constexpr (is_tuple<Shape>::value) {      // tuple tuple
      static_assert(tuple_size<Index>::value == tuple_size<Shape>::value, "Mismatched Ranks");
      return transform(idx, shape, [](auto const& i, auto const& s) { return idx2crd(i,s); });
    } else {                                     // tuple "int"
      static_assert(sizeof(Index) == 0, "Invalid parameters");
    }
  } else {
    if constexpr (is_tuple<Shape>::value) {      // "int" tuple
      return transform_leaf(as_arithmetic_tuple(crd2idx(idx, shape, make_basis_like(shape))), identity{});
    } else {                                     // "int" "int"
      return idx;
    }
  }
}
```

Layout 的构造也可以只提供 Shape，不提供 Stride。
这时会按照 LayoutLeft （最里的维度在先，即 Column major）构造 Stride。
具体实例可见下节。

### Layout Manipulation

#### Sublayouts

多层单点选取：

```cpp
Layout a   = Layout<Shape<_4,Shape<_3,_6>>>{}; // (4,(3,6)):(1,(4,12))
Layout a0  = layout<0>(a);                     // 4:1
Layout a1  = layout<1>(a);                     // (3,6):(4,12)
Layout a10 = layout<1,0>(a);                   // 3:4
Layout a11 = layout<1,1>(a);                   // 6:12
```

单层多点选取：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = select<1,3>(a);                   // (3,7):(2,30)
Layout a01 = select<0,1,3>(a);                 // (2,3,7):(1,2,30)
Layout a2  = select<2>(a);                     // (5):(6)
```

单层范围选取：

```cpp
Layout a   = Layout<Shape<_2,_3,_5,_7>>{};     // (2,3,5,7):(1,2,6,30)
Layout a13 = take<1,3>(a);                     // (3,5):(2,6)
Layout a14 = take<1,4>(a);                     // (3,5,7):(2,6,30)
// take<1,1> not allowed. Empty layouts not allowed.
```

#### Concatenation

```cpp
Layout a = Layout<_3,_1>{};                     // 3:1
Layout b = Layout<_4,_3>{};                     // 4:3
Layout row = make_layout(a, b);                 // (3,4):(1,3)
Layout col = make_layout(b, a);                 // (4,3):(3,1)
Layout q   = make_layout(row, col);             // ((3,4),(4,3)):((1,3),(3,1))
Layout aa  = make_layout(a);                    // (3):(1)
Layout aaa = make_layout(aa);                   // ((3)):((1))
Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
```

#### Grouping and Flattening

```cpp
Layout a = Layout<Shape<_2,_3,_5,_7>>{};  // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout b = group<0,2>(a);                 // ((_2,_3),_5,_7):((_1,_2),_6,_30)
Layout c = group<1,3>(b);                 // ((_2,_3),(_5,_7)):((_1,_2),(_6,_30))
Layout f = flatten(b);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
Layout e = flatten(c);                    // (_2,_3,_5,_7):(_1,_2,_6,_30)
```

## Layout Algebra

### Coalesce

Coalesce 操作是对 Layout 的简化。例如：

```cpp
auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(layout);    // _12:_1
```

考虑 Coalesce 一个两维的 Layout `(s0, s1):(d0, d1)`，存在四种情况：

1. `(s0, _1):(d0, d1) => s0:d0`，移除大小为 1 的维度
2. `(_1, s1):(d0, d1) => s1:d1`，移除大小为 1 的维度
3. `(s0, s1):(d0, s0*d0) => s0*s1:d0`
4. 其他情况则无法合并

支持按维度 Coalesce:

```cpp
auto a = Layout<Shape <_2,Shape <_1,_6>>,
                Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(a, Step<_1,_1>{});   // (_2,_6):(_1,_2)
// Identical to
auto same_r = make_layout(coalesce(layout<0>(a)),
                          coalesce(layout<1>(a)));
```

### Composition

既然 Layout 是从整数到整数的映射，我们可以复合两个 Layout 成一个新的 Layout，就像复合函数。

```cpp
// @post compatible(@a layout_b, @a result)
// @post for all i, 0 <= i < size(@a layout_b), @a result(i) == @a layout_a(@a layout_b(i)))
Layout composition(LayoutA const& layout_a, LayoutB const& layout_b)
```

也可以按维度复合：

```cpp
// (12,(4,8)):(59,(13,1))
auto a = make_layout(make_shape (12,make_shape ( 4,8)),
                     make_stride(59,make_stride(13,1)));
// <3:4, 8:2>
auto tiler = make_tile(Layout<_3,_4>{},  // Apply 3:4 to mode-0
                       Layout<_8,_2>{}); // Apply 8:2 to mode-1

// (_3,(2,4)):(236,(26,1))
auto result = composition(a, tiler);
// Identical to
auto same_r = make_layout(composition(layout<0>(a), get<0>(tiler)),
                          composition(layout<1>(a), get<1>(tiler)));
```

### Complement

一个 Layout 可以不是单射的，这时物理坐标的范围（codomain）会大于逻辑坐标的范围（domain）。
Layouot 的补集 Complement 用来描述 codomain 中没被映射到的位置。

- `complement(4:1, 24) => 6:4`，因为 `(4, 6):(1, 4)` 有 codomain 大小为 24
- `complement(6:4, 24) => 6:4`，因为 `(6, 4):(4, 1)` 有 codomain 大小为 24

![Complement Example](cute/complement1.png)

### Division (Tiling)

Layout 的除法运算表示按照除数 Layout 对被除数 Layout 的划分。
除法的结果是一个两维的 Layout。
可以把除数 Layout 理解成 Tile 的 Layout，除法结果的第一维是 Tile 内的 Layout，第二维是被除数 Layout 中 Tile 间的 Layout。

> Layout 除法共有四种：logical, zipped, tiled, flat。下面先介绍的除法指的是 logical divide。

具体的，先把被除数和除数 Layout 都展平成一维，除法结果 Layout 的每个 Tile 按照除数 Layout 的索引，在被除数中取值。
可以参考下图中的例子：

![Divide Example 1](cute/divide1.png)

实际上，我们可以给出上述除法的严格定义：

$$
A / B := A \circ (B, B^*)
$$

A 与 B 的 logical divide，是 B 与 B 的补集 (complement) 的拼接 (concatenation)，再和 A 的复合 (composition)。

考虑当 B 是一个单射时，有

$$
A / B = A \circ (B, B^*) = (A \circ B, A \circ B^*)
$$

可以注意到结果的第一维是单个 Tile 内的 Layout，而第二维是被除数 A 中各个 Tile 的 Layout。

上述的 logical divide 也可以扩展至多维，对每一维分别做一维的 logical divide。见下图：

> 这里出现的尖括号 `B = <3:3, (2,4):(1,8)>` 表示 B 是一个 Tiler，用来代表运算是对每个维度分别操作的。

![2D Logical Divide](cute/divide2.png)

前文提到，除了 logical divide，还有 zipped，tiled，flat divide 三种除法运算。
这三种除法与 logical divide 的区别仅在于结果 layout 的排布：

```
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

### Product (Tiling)

有了除法之后，我们可以把乘法想象成除法的逆运算。
除法是将一个大的 Layout 按照某种 Tiling 分块，乘法则是将小的分块 Layout 组合成整个 Layout。
Logical product 的严格定义为：

$$
A \otimes B = (A, A^* \circ B)
$$

A 与 B 的乘积有两维，第一维是 A 本身，第二维是 A 的补集和 B 的复合。
以下是两个一维乘法的例子：

![1D Logical Product 1](cute/product1.png)

![1D Logical Product 2](cute/product2.png)

从二维开始，logical product 变得反直觉了。
在下图中，我们希望把 2x5 的 Tile 在列方向上重复 3 次，在行方向上重复 4 次。
然而，这个操作所需的 B Layout 是 `<3:5, 4:6>`，这需要从 A 的 Layout 中推导出来，且并不直接。

![2D Logical Product](cute/product2d.png)

CuTe 提供了 `blocked_product` 和 `raked_product`，来简化这样的乘法操作。

![Blocked Product](cute/productblocked2d.png)

Blocked product 把 Layout A 连续的按照 Layout B 进行排布。

![Raked Product](cute/productraked2d.png)

Raked product 把 Layout A 中每个元素按 Layout B 进行排布，再按 Layout A 进行排布。 

## Tensor

### Tensor Creation

Tensor 的创建分为两种：

1. 在已有的栈上数据（memory）上创建，Tensor 不拥有 memory。
2. 在堆上创建静态大小的 memory，Tensor 拥有该 memory。

```cpp
// 栈上对象：需同时指定类型和Layout，layout必须是静态shape
Tensor make_tensor<T>(Layout layout);

// 堆上对象：需指定pointer和Layout，layout可动可静
Tensor make_tensor(Pointer pointer, Layout layout);
 
// 栈上对象，tensor的layout必须是静态的
Tensor make_tensor_like(Tensor tensor); 

// 栈上对象，tensor的layout必须是静态的
Tensor make_fragment_like(Tensor tensor);
```

### Tensor Partition

在 GPU 上，我们需要将一个 Tensor 划分成多个块，让每个 SM 每次处理一个块。
例如，我们把一个大小为 [8, 24] 的 Tensor 划分成若干个大小为 [4, 8] 的块：

```cpp
Tensor A = make_tensor(ptr, make_shape(8,24));  // (8,24)
auto tiler = Shape<_4,_8>{};                    // (_4,_8)

Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
```

在 kernel 内可以根据 block ID 索引当前 SM 的数据：

```cpp
Tensor cta_a = tiled_a(make_coord(_,_), make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

这种索引方式也被封装进了 `inner_partition(Tensor, Tiler, Coord)` 或者 `local_tile(Tensor, Tiler, Coord)`。

另外一种情况是，假设有 32 个线程，每个线程处理第一维中 (4, 8) 对应的某一块：

```cpp
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_,_)); // (2,3)
```

也可以写成 `outer_partition(Tensor, Layout, Idx)` 或 `local_partition(Tensor, Layout, Idx)`。

### Thread-Value partitioning

假设我们有一个将线程映射到坐标的 Layout，
我们可以将一个 Tensor 与这个 Layout 复合，构成能将线程映射到值的 Tensor。

```cpp
// Construct a TV-layout that maps 8 thread indices and 4 value indices
//   to 1D coordinates within a 4x8 tensor
// (T8,V4) -> (M4,N8)
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

// Construct a 4x8 tensor with any layout
Tensor A = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
// Compose A with the tv_layout to transform its shape and order
Tensor tv = composition(A, tv_layout);                           // (8,4)
// Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
Tensor  v = tv(threadIdx.x, _);                                  // (4)
```

## References

https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/

https://www.zhihu.com/people/reed-84-49/posts