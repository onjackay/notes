# CuTe Detail

## CuTe MMA

### MMA Operation

MMA Operation 位置在 `cute/arch/mma_sm*` 文件中，是 PTX 指令的封装。
类的名称里指定了 SM 计算能力，MNK 的形状，`D = A x B + C` 的数据类型，和 AB 矩阵是否转置。
N (normal) 表示 Col-major，T (transpose) 表示 Row-major。
`fma` 接口中调用 PTX 指令运算 `D = A x B + C`。

```cpp
// MMA 16x8x8 TN
struct SM80_16x8x8_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x8_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```

对于 SM90 wgmma，AB 矩阵的转置情况变为在模版参数中传入。
类的名称中增加了 AB 矩阵是在寄存器 (R) 或 SMEM (S) 的标识。
可以注意到仅有 RS 和 SS 两种情况。
如果 A 矩阵在寄存器中，则只能是 K-major 排布 (Row-major)。

```cpp
// GMMA 64x8x16 F32+=F16*F16
template <
  GMMA::Major tnspA,
  GMMA::Major tnspB,
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
struct MMA_64x8x16_F32F16F16_RS
{
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  static_assert(tnspA == GMMA::Major::K,
      "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint64_t const& desc_b,
      float         & d0, float         & d1, float         & d2, float         & d3,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    cutlass::arch::synclog_emit_wgmma_reg_smem(__LINE__, desc_b);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %9, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      " %8,"
      " p,   %10, %11, %12;\n"
    "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "l"(desc_b),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspB)));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_64x8x16_F32F16F16_RS without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
};
```

### MMA Traits

MMA Traits 中提供每种 MMA Operation 的基本信息：

```cpp
using ElementDVal =  // Logical D-value type
using ElementAVal =  // Logical A-value type
using ElementBVal =  // Logical B-value type
using ElementCVal =  // Logical C-value type

using Shape_MNK =    // Logical MxNxK shape of the MMA

using ThrID     =    // Logical thread id (tid) -> tidx

using ALayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MK-coord
using BLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat NK-coord
using CLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MN-coord
```

ThrID 表示了需要参与 MMA 指令的线程的 Layout。
例如，在 SM70 上，有 `ThrID = (_4, _2):(_1, _16)`，即 8 个线程发起一个 MMA 指令。
而在 SM90 WGMMA 上，有 `ThrID = (_128):(_1)`，即一个 warp group 128 个线程发起一个 WGMMA 指令。
ABC 的 Layout 是一个映射 `(T, V) -> Idx`，将第 T 个逻辑线程中的第 V 个值的逻辑坐标映射至按 Col-major 排布展平的物理坐标。

```cpp
// (T32,V2) -> (M8,N8)
using SM80_8x8_Row  = Layout<Shape <Shape < _4,_8>,_2>,
                             Stride<Stride<_16,_1>,_8>>;
// (T32,V4) -> (M16,N8)
using SM80_16x8_Row = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                             Stride<Stride<_32,_1>,Stride<_16,_8>>>;
template <>
struct MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_8>;
  using ThrID   = Layout<_32>;
  using ALayout = SM80_16x8_Row;
  using BLayout = SM80_8x8_Row;
  using CLayout = SM80_16x8_Row;
};
```

### MMA Atom

MMA Atom 继承 MMA Traits，作为一条 MMA 指令的最终封装。

```cpp
template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
{
  using MMA_Op = MMAOperation;
  using Traits = MMA_Traits<MMAOperation, Args...>;

  // Element value types from the MMA_Traits
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  // Thr-Val layouts from the MMA_Traits
  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // Fragment value types from the MMA_Traits (optional, defaults to Val type)
  using FrgTypeD = typename detail::FrgTypeC_or_Default<Traits>::type;
  using FrgTypeA = typename detail::FrgTypeA_or_Default<Traits>::type;
  using FrgTypeB = typename detail::FrgTypeB_or_Default<Traits>::type;
  using FrgTypeC = typename detail::FrgTypeC_or_Default<Traits>::type;
...
```

### TiledMMA

TiledMMA 将多个 MMA Atom 组合起来，表达使用多个 MMA 指令来计算更大的一块 GEMM。

```cpp
// @tparam MMA_Atom The MMA_Atom to use in the TiledMMA
// @tparam AtomLayoutMNK The MNK-tiling of the Atom to be performed.
// @tparam PermuationsMNK Permutations to apply to each MNK-mode before tiling for the Atom.
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;
...
```

只传入 MMA Atom 模板参数时，默认 AtomLayoutMNK 是 `(_1, _1, _1)`，即只有一个 MMA Atom。

```cpp
MMA_Atom mma = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{};
```

等价于 MNK 大小为 `(_8, _8, _4)` 的一个 MMA 指令：

```cpp
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape<_1,_1,_1>>{},   // Layout of Atoms
                              Tile<_8,_8,_4>{});           // Tiler
```