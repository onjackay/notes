# CuTe 之 Copy

## Prologue

1. `thrCopy` 的 `partition_S`, `partition_D`, `retile` 分别接受怎样的输入，做了什么操作，目的是什么？
2. 从而可以回答：在 partition 前，dst 和 src 的 layout 分别需要满足什么限制，才能产生我们期望的 per-thread 的 layout？

## Copy Operation

和 mma 一样，copy 抽象的最底层也是 Copy Operation。

```cpp
// cute/arch/copy_sm75.hpp
struct SM75_U32x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};
```

Copy Operation 位置在 `cute/arch/mma_sm*` 文件中，是 ptx 指令的封装。
SRegisters/DRegisters 指明了 `copy()` 方法的参数类型与数量。

```cpp
template <>
struct Copy_Traits<SM75_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

Copy Trait 为 Operation 描述了指令对应的 layout 信息。
`ThrId` 表示指令 issue 所需的线程数量。
`SrcLayout` 是在 bit 视角下，src 的 TV-layout。
`DstLayout` 是在 bit 视角下，dst 的 TV-layout。

对于 `UniversalCopy` 和 `cp.async` 这类一般的 Op，`SrcLayout` 和 `DstLayout` 是相同的。
但是像 `ldmatrix` 这类，每个线程的 src 和 dst 的视图不同，我们需要确定一个“坐标系“作为参照，这就是 `RefLayout`。
对于 `ldmatrix` 这个例子，`RefLayout` 等于 `DstLayout`。

## Copy Atom

```cpp
// cute/atom/copy_atom.hpp
template <class CopyOperation, class CopyInternalType>
struct Copy_Atom<CopyOperation, CopyInternalType> : Copy_Atom<Copy_Traits<CopyOperation>, CopyInternalType>
{};

template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType;

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));

  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}), "CopyOperation is not valid for Src of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}), "CopyOperation is not valid for Dst of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}), "CopyOperation is not valid for Ref of ValType.");

  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});
```

Copy Atom 是 Copy Operation 与 dtype 的组合。
Copy Atom 通过 Copy Trait 获取 Copy Op 的 layout 信息，把 Op 中 bit 视角下的 TV-layout 转换为特定 dtype 视角下的 TV-layout，
即 `ValLayoutSrc`, `ValLayoutDst`, 和 `ValLayoutRef`。
`NumValSrc` 和 `NumValDst` 取对应 TV-layout rank-1 的大小，即每个线程负责的 src/dst 元素数量。

```cpp
  // Check and call instruction, or recurse
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_HOST_DEVICE
  void
  call(Tensor<SEngine,SLayout> const& src,
       Tensor<DEngine,DLayout>      & dst) const
  {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack to execute instruction
      return copy_unpack(static_cast<Traits const&>(*this), src, dst);
    } else if constexpr (is_tuple<decltype(shape(src))>::value &&
                         is_tuple<decltype(shape(dst))>::value) {
      // If the size of the src/dst doesn't match the instruction,
      //   recurse this rank-1 layout by peeling off the mode
      //   ((A,B,C,...)) -> (A,B,C,...)
      return copy(*this, tensor<0>(src), tensor<0>(dst));
    } else {
      static_assert(dependent_false<SEngine>,
                    "CopyAtom: Src/Dst partitioning does not match the instruction requirement.");
    }
  }
```

`Copy_Atom::call()` 要求 src/dst 都是 1-rank tensor，然后判断直接发射硬件 copy 指令或者递归分解。

如果 src/dst 的大小恰好满足 Copy Atom 的大小：调用 `copy_unpack(static_cast<Traits const&>(*this), src, dst)` 直接发射硬件 copy 指令。

```cpp
// cute/atom/copy_trait.hpp
template <class AnyCPYTraits,
          class SEngine, class SLayout,
          class DEngine, class DLayout>
CUTE_HOST_DEVICE constexpr
void
copy_unpack(AnyCPYTraits            const&,
            Tensor<SEngine,SLayout> const& src,
            Tensor<DEngine,DLayout>      & dst)
{
  using CopyOp       = typename CPY_Op<AnyCPYTraits>::type;  
  using RegistersSrc = typename CopyOp::SRegisters;
  using RegistersDst = typename CopyOp::DRegisters;
  using RegTypeSrc   = typename remove_extent<RegistersSrc>::type;
  using RegTypeDst   = typename remove_extent<RegistersDst>::type;
  constexpr int RegNumSrc = extent<RegistersSrc>::value;
  constexpr int RegNumDst = extent<RegistersDst>::value;

  Tensor rS = recast<RegTypeSrc>(src);
  Tensor rD = recast<RegTypeDst>(dst);

  CUTE_STATIC_ASSERT_V(size(rS) == Int<RegNumSrc>{},
    "Copy_Traits: src failed to vectorize into registers. Layout is incompatible with this CopyOp.");
  CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumDst>{},
    "Copy_Traits: dst failed to vectorize into registers. Layout is incompatible with this CopyOp.");

  detail::explode(detail::CallCOPY<CopyOp>{},
                  rS, make_int_sequence<RegNumSrc>{},
                  rD, make_int_sequence<RegNumDst>{});
}
```

`copy_unpack` 将 src/dst 转换为 Copy Op 中 SRegisters/DRegisters 的类型，然后进入 explode 把 tensor 展开，调用 Copy Op 的 `copy()`。

否则，调用 `copy(*this, tensor<0>(src), tensor<0>(dst))`，去掉最外层的 1-rank shape 后回到 `copy.hpp` 中的自由函数：

```cpp
// cute/algorithm/copy.hpp
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
     Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");

  if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
    copy_atom.call(src, dst);
  } else {                                // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    Tensor src_v = group_modes<1,R>(src);
    Tensor dst_v = group_modes<1,R>(dst);

    if constexpr (is_static<decltype(shape(src_v))>::value && is_static<decltype(shape(dst_v))>::value) {
      // AutoFilter on the Rest-mode
      auto dst_null = nullspace(layout<1>(dst_v));

      Tensor dst_n = zipped_divide(dst_v, make_tile(shape<0>(dst_v), dst_null));  // ((V, NLL), (_1, Rest))
      Tensor src_n = zipped_divide(src_v, make_tile(shape<0>(src_v), dst_null));  // ((V, NLL), (_1, Rest))
      Tensor dst_c = dst_n(make_coord(_,Int<0>{}),make_coord(Int<0>{},_));        // (V, Rest)
      Tensor src_c = src_n(make_coord(_,Int<0>{}),make_coord(Int<0>{},_));        // (V, Rest)

      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_c); ++i) {
        copy_atom.call(src_c(_,i), dst_c(_,i));
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.call(src_v(_,i), dst_v(_,i));
      }
    }
  }
}
```

如果这时 rank 仍等于 1，则直接进入 `Copy_Atom::call()`。
否则，会对除了 mode-0 以外的 mode 循环，然后对 mode-0 调用 `Copy_Atom::call()`。
从此处可见，调用 `cute::copy()` 始终需要将每次 atom 操作的 src/dst layout 放在 mode-0，其他 mode 代表 atom 循环次数。

在 src/dst 的 layout 是静态的时候，layout 中的 nullspace（stride=0 的空间）会被过滤掉，跳过在重复地址上的循环。
对于动态 layout 的情况，编译期无法计算 nullspace，此时只能朴素地遍历所有的元素。

## Tiled Copy

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
  using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
  using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN       = ShapeTiler_MN;
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));

  CUTE_STATIC_ASSERT_V(TiledNumThr{} % AtomNumThr{} == Int<0>{}, "TiledCopy uses too few thrs for selected CopyAtom");
  CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{}, "TiledCopy uses too few vals for selected CopyAtom");
```

对单个 Copy Atom 进行扩展，就得到了 Tiled Copy。
扩展发生在两个维度上：线程数量上的扩展 (AtomNumThr -> TiledNumThr)，和单一线程上多次 Atom 的扩展 (AtomNumVal -> TiledNumVal)。
这两个扩展表现在构造参数 TV-Layout 上，T 的 size 代表线程数量，而 V 的 size 代表单一线程处理的元素个数。

另一个构造参数 TilerMN 表示整个 Tiled Copy 的块大小。

构造参数使用的 TV-Layout 是相对于 atom 的 ref layout 的，那么当 src/dst layout 不等于 ref layout 时，相对于 src/dst layout 的 TV-Layout 则需要进行转换。
以 src 为例：

```cpp
  // Tile a tensor or a layout from shape
  //   ((TileM,TileN,...), (RestM,RestN,...))
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static
  auto
  tile2thrfrg(Tensor&& tensor, Ref2TrgLayout const& ref2trg)
  {
    // Take the thrs/vals that the atom is interested in
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto atom_layout_TV = zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

    // Transform to the trg layout
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

    // Transform the thrs mode from thrid to thr_idx
    // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1,Shape<_1,_1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

    /// ==================

    // Transform the tile mode
    auto tv_tensor = tensor.compose(thrval2mn, _);
    // ((thrid,val),(RestM,RestN,...))

    // Unfold and return
    return tv_tensor(make_coord(_,_), _);
  }
  
  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   (Thr,(FrgV,FrgX),(RestM,RestN,...))
  // where
  //   Thr:   The logical threads within the tiled copy.
  //   FrgV:  The values local to a COPY_ATOM Src.
  //   FrgX:  The values tiled across COPY_ATOMs Src.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class STensor>
  CUTE_HOST_DEVICE constexpr static
  auto
  tidfrg_S(STensor&& stensor)
  {
    CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

    // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout
    return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
  }
```