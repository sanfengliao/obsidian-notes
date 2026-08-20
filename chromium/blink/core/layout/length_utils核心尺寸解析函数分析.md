
# 0. 这组函数解决什么问题

CSS 的 `width` / `height` / `min-width` / `max-width` 等属性的值（`Length`）可以是：固定长度、百分比、`calc()`、`auto`、`none`、`stretch`、`min-content`、`max-content`、`fit-content`、`min-intrinsic`、`content` 等。把它们解析成具体的 `LayoutUnit`（带 border/padding、受 box-sizing 影响、受可用尺寸与百分比解析尺寸约束）是布局的第一步。

`length_utils` 把这件事分成 **6 层**，从底向上的调用关系是：

```
Layer 5  CalculateInitialFragmentGeometry        (入口：算 border-box 尺寸 + border/padding/scrollbar)
Layer 4  ComputeInlineSizeForFragment / ComputeBlockSizeForFragment   (解析主尺寸 + clamp min/max)
Layer 3  ComputeMinMaxInlineSizes / ComputeMinMaxBlockSizes           (解析 min/max 这一对)
Layer 2  ResolveMin/Max/Main Inline/BlockLength   (薄包装：选 LengthTypeInternal + 处理 kIndefiniteSize)
Layer 1  ResolveInlineLengthInternal / ResolveBlockLengthInternal    (核心：switch on Length 类型)
Layer 0  MinimumValueForLength / ValueForLength   (平台层：fixed/percent/calc → LayoutUnit)
```

理解这组函数的关键有三点：

1. **`LengthTypeInternal { kMin, kMain, kMax }`**：同一个 `ResolveInlineLengthInternal` 既能解析 `min-width`、也能解析 `width`、`max-width`。三者对「百分比解析尺寸不确定」和 `auto` 的处理不同（见 1.x）。
2. **`kIndefiniteSize` 哨兵**：内部函数用 `kIndefiniteSize` 表示「解析不出来」，外层包装再把它转成语义化的默认值（min → border+padding，max → `LayoutUnit::Max()`，main → 透传给调用方决定）。
3. **回调驱动**：intrinsic 关键字（`min-content` 等）不自己算，而是通过 `MinMaxSizesFunctionRef` / `BlockSizeFunctionRef` 回调按需获取，避免提前触发昂贵的 min/max 计算。

---

# 1. Layer 0/1：长度解析引擎

## 1.1 平台层 `MinimumValueForLength` / `ValueForLength`

`core/platform/geometry/length_functions.{h,cc}`，处理最底层的 fixed/percent/calc：

```cpp
inline LayoutUnit MinimumValueForLength(const Length& length,
                                        LayoutUnit maximum_value,
                                        const EvaluationInput& input = {}) {
  if (length.IsFixed()) [[likely]]
    return LayoutUnit(length.Pixels());
  return MinimumValueForLengthInternal(length, maximum_value, input);
}

// MinimumValueForLengthInternal:
//   kPercent   -> maximum_value * percent / 100
//   kCalculated-> NonNanCalculatedValue(maximum_value, input)
//   kStretch/kAuto -> 0
//   其余 intrinsic 类型 -> NOTREACHED（不该走到这里）

// ValueForLength: 与 MinimumValueForLength 的唯一区别是
//   kStretch/kAuto -> maximum_value（返回完整可用尺寸，而非 0）
```

- `maximum_value` 就是百分比解析尺寸（100% 的基准）。
- `MinimumValueForLength` 用于 min-size 解析（CSS 规定 min-size 的 `auto` 当 0）；`ValueForLength` 用于需要 `auto = 完整尺寸` 的场景（如 SVG replaced 的百分比高度）。
- intrinsic 关键字在这里是错误——它们由 Layer 1 在调用 `MinimumValueForLength` **之前**通过 `intrinsic_evaluator` 回调拦截处理（用于 `calc(min-content + 10px)` 这类）。

## 1.2 `ResolveInlineLengthInternal`（核心，inline 轴）

```cpp
LayoutUnit ResolveInlineLengthInternal(
    const ConstraintSpace& constraint_space,
    const ComputedStyle& style,
    const BoxStrut& border_padding,
    MinMaxSizesFunctionRef min_max_sizes_func,
    const Length& original_length,
    const Length* auto_length,
    LengthTypeInternal length_type,           // kMin / kMain / kMax
    FitContentMode fit_content_mode,
    LayoutUnit override_available_size,
    CalcSizeKeywordBehavior calc_size_keyword_behavior);
```

逻辑（switch on `length.GetType()`）：

**前置**：若 `original_length` 是 `auto` 且提供了 `auto_length`，则用 `auto_length` 替换。这就是「`width: auto` 时实际用什么长度」的入口——`auto_length` 由上层（`ComputeInlineSizeForFragmentInternal`）根据可用尺寸、aspect-ratio、auto 行为决定（见 4.2）。

```cpp
const Length& length =
    original_length.IsAuto() && auto_length ? *auto_length : original_length;
```

**`kStretch`**：用 `AvailableSize().inline_size`（或 override）减去 margin（除非 `IgnoreMarginsForStretch`），下限为 `border_padding.InlineSum()`。若可用尺寸 indefinite → 返回 indefinite。

**`kPercent` / `kFixed` / `kCalculated`**：

```cpp
LayoutUnit percentage_resolution_size =
    constraint_space.PercentageResolutionInlineSize();
if (length.HasPercent() && percentage_resolution_size == kIndefiniteSize) {
  if (length_type != LengthTypeInternal::kMin) {
    return kIndefiniteSize;          // main/max：百分比基准不定 → 解析不出
  }
  percentage_resolution_size = LayoutUnit();  // min：百分比基准不定 → 当 0
}
```

> 这是 `LengthTypeInternal` 最关键的区别：**min 长度的百分比在基准 indefinite 时按 0 解析**（CSS 规范要求 min-size 的 indefinite 百分比视为 0），而 main/max 直接返回 indefinite。`kFixed` 不受影响（无需基准）。

然后调 `MinimumValueForLength`，并通过 `intrinsic_evaluator` 回调处理 calc 里的 intrinsic 关键字；最后按 `box-sizing` 加/ clamp `border_padding.InlineSum()`：

```cpp
if (style.BoxSizing() == EBoxSizing::kBorderBox)
  value = std::max(border_padding.InlineSum(), value);   // border-box：下限为 border+padding
else
  value += border_padding.InlineSum();                    // content-box：加上 border+padding
```

**intrinsic 关键字**：

```cpp
case Length::kContent:
case Length::kMaxContent:  
  return min_max_sizes_func(SizeType::kContent).sizes.max_size;
case Length::kMinContent:  
  return min_max_sizes_func(SizeType::kContent).sizes.min_size;
case Length::kMinIntrinsic:
  return min_max_sizes_func(SizeType::kIntrinsic).sizes.min_size;
```

`kContent` / `kMaxContent` 都取 max-content；`kMinContent` 取 min-content（`kContent` 类型）；`kMinIntrinsic` 取 `kIntrinsic` 类型下的 min（**不**考虑 aspect-ratio，见 `SizeType` 文档）。这里全靠回调 `min_max_sizes_func` 按需计算，不在解析时强制算。

**`kFitContent`**：可用尺寸 indefinite 时按 `fit_content_mode` / `length_type` 退化（min → min-content，max → max-content，main → indefinite）；可用尺寸确定时用 `ShrinkToFit(available - margins)`。

**`kAuto` / `kNone`**：

```cpp
case Length::kAuto:
  if (length_type == LengthTypeInternal::kMin)
    return border_padding.InlineSum();   // min-size 的 auto = border+padding
  [[fallthrough]];                       // main/max 的 auto 透传 indefinite
case Length::kNone:
  return kIndefiniteSize;
```

> 又一处 `LengthTypeInternal` 的差异：min 长度的 `auto` = `border+padding`（CSS: `min-width: auto` 解析为内容最小尺寸的下限，这里简化为 border+padding；实际自动最小尺寸由上层 `apply_automatic_min_size` 控制）；main/max 的 `auto` 由前面的 `auto_length` 替换处理，走到这里的 `auto` 通常是没给 `auto_length`，返回 indefinite。

## 1.3 `ResolveBlockLengthInternal`（核心，block 轴）

结构与 inline 版本对称，差异点：

1. **百分比基准**：用 `PercentageResolutionBlockSize()` 或 `*override_percentage_resolution_size`。
2. **indefinite 百分比的 main 分支**：不是返回 indefinite，而是**回退到 `block_size_func(SizeType::kContent)`**——block 轴的 `height: 50%` 在基准不定时退化成内容高度（这是 CSS block 高度的特性）。

```cpp
if (length.HasPercent() && percentage_resolution_size == kIndefiniteSize) {
  switch (length_type) {
    case LengthTypeInternal::kMin:   
      percentage_resolution_size = LayoutUnit();
      break;
    case LengthTypeInternal::kMain:  
      return block_size_func(SizeType::kContent); // 退化为内容高度
    case LengthTypeInternal::kMax:   
      return kIndefiniteSize;
  }
}
```

3. **`kStretch` indefinite 时**：main → `block_size_func(kContent)`（退化为内容高度），min/max → indefinite。
4. **intrinsic 关键字**：`kContent`/`kMinContent`/`kMaxContent`/`kMinIntrinsic`/`kFitContent` **全部**走 `block_size_func`（不像 inline 那样区分 min/max-content），因为 block 方向的 intrinsic size 是单一值（内容高度），由回调统一给出。`kMinIntrinsic` 用 `SizeType::kIntrinsic`，其余用 `kContent`。

---

# 2. Layer 2：`Resolve*` 公共包装（薄适配器）

这些都在 `length_utils.h` 里 inline 定义，本质是「选 `LengthTypeInternal` + 调 Internal + 把 `kIndefiniteSize` 转成语义默认值」。

## 2.1 inline 轴三兄弟

```cpp
// min-width：kMin，indefinite → border_padding.InlineSum()
inline LayoutUnit ResolveMinInlineLength(..., const Length& length,
        const Length* auto_length = nullptr, ...,
        FitContentMode fit_content_mode = FitContentMode::kNormal) {
  const LayoutUnit result = ResolveInlineLengthInternal(
      ..., length, auto_length, LengthTypeInternal::kMin, fit_content_mode, ...,
      CalcSizeKeywordBehavior::kAsSpecified);
  return result == kIndefiniteSize ? border_padding.InlineSum() : result;
}

// max-width：kMax，indefinite → LayoutUnit::Max()（即「无上限」）
inline LayoutUnit ResolveMaxInlineLength(..., const Length& length, ...,
        FitContentMode fit_content_mode = FitContentMode::kNormal) {
  const LayoutUnit result = ResolveInlineLengthInternal(
      ..., length, /*auto_length*/ nullptr, LengthTypeInternal::kMax, ...);
  return result == kIndefiniteSize ? LayoutUnit::Max() : result;
}

// width（主长度）：kMain，indefinite → 透传（调用方决定怎么处理）
inline LayoutUnit ResolveMainInlineLength(..., const Length& length,
        const Length* auto_length, ...,
        CalcSizeKeywordBehavior calc_size_keyword_behavior = kAsSpecified) {
  return ResolveInlineLengthInternal(
      ..., length, auto_length, LengthTypeInternal::kMain,
      FitContentMode::kNormal, ..., calc_size_keyword_behavior);
}
```

| 函数 | LengthType | `auto_length` | indefinite 时的返回 |
|------|-----------|---------------|---------------------|
| `ResolveMinInlineLength` | `kMin` | 可选 | `border_padding.InlineSum()` |
| `ResolveMaxInlineLength` | `kMax` | 无（nullptr） | `LayoutUnit::Max()` |
| `ResolveMainInlineLength` | `kMain` | 必须 | 透传 `kIndefiniteSize` |

注意 max 长度没有 `auto_length`——`max-width: auto` 等价于 `none`（无上限），由 Internal 的 `kAuto` fallthrough 到 `kNone` 返回 indefinite，再被转成 `LayoutUnit::Max()`。

## 2.2 block 轴六兄弟

block 轴多一组「Initial」变体：

```cpp
// min-height：kMin，indefinite → border_padding.BlockSum()
inline LayoutUnit ResolveMinBlockLength(..., BlockSizeFunctionRef block_size_func,
        const Length& length, const Length* auto_length = nullptr, ...,
        const LayoutUnit* override_percentage_resolution_size = nullptr);

// max-height：kMax，indefinite → LayoutUnit::Max()
inline LayoutUnit ResolveMaxBlockLength(..., const Length& length,
        BlockSizeFunctionRef block_size_func, ...,
        const LayoutUnit* override_percentage_resolution_size = nullptr);

// height（主长度）：kMain，两个重载
//   重载1：传入 intrinsic_size（捕获成 lambda 作为 block_size_func）
inline LayoutUnit ResolveMainBlockLength(..., const Length& length,
        const Length* auto_length, LayoutUnit intrinsic_size, ...);
//   重载2：传入 block_size_func 回调
inline LayoutUnit ResolveMainBlockLength(..., const Length& length,
        const Length* auto_length, BlockSizeFunctionRef block_size_func, ...);
```

**「Initial」变体** `ResolveInitialMinBlockLength` / `ResolveInitialMaxBlockLength`：

```cpp
inline LayoutUnit ResolveInitialMinBlockLength(..., const Length& length, ...) {
  const LayoutUnit result = ResolveBlockLengthInternal(
      ..., length, /*auto_length*/ &Length::Auto(), LengthTypeInternal::kMin,
      ..., /*override_pct*/ nullptr,
      [](SizeType) { return kIndefiniteSize; });   // 永远返回 indefinite 的回调
  return result == kIndefiniteSize ? border_padding.BlockSum() : result;
}
```

区别在于用了一个**永远返回 `kIndefiniteSize` 的 `block_size_func`**。效果：intrinsic 关键字（`min-content`/`max-content`/`fit-content`）不会被真正解析——它们会拿到 indefinite，再被外层转成默认值。用途：在还没法做内容布局的阶段（比如算 aspect-ratio transferred size 时）先拿一个「不考虑内容」的 min/max。

| 函数 | LengthType | block_size_func | indefinite 返回 |
|------|-----------|-----------------|-----------------|
| `ResolveInitialMinBlockLength` | `kMin` | 永远 indefinite | `border_padding.BlockSum()` |
| `ResolveInitialMaxBlockLength` | `kMax` | 永远 indefinite | `LayoutUnit::Max()` |
| `ResolveMinBlockLength` | `kMin` | 真实回调 | `border_padding.BlockSum()` |
| `ResolveMaxBlockLength` | `kMax` | 真实回调 | `LayoutUnit::Max()` |
| `ResolveMainBlockLength` | `kMain` | 真实回调 / intrinsic_size | 透传 |

---

# 3. Layer 3：`ComputeMinMaxInlineSizes` / `ComputeMinMaxBlockSizes`

把 min-length 和 max-length 解析成一对 `MinMaxSizes`，并叠加 aspect-ratio transferred size 和 table 规则。

## 3.1 `ComputeMinMaxBlockSizes`

```cpp
MinMaxSizes ComputeMinMaxBlockSizes(const ConstraintSpace& space,
                                    const BlockNode& node,
                                    const BoxStrut& border_padding,
                                    const Length* auto_min_length,
                                    BlockSizeFunctionRef block_size_func,
                                    LayoutUnit override_available_size) {
  const ComputedStyle& style = node.Style();
  MinMaxSizes sizes = {
      ResolveMinBlockLength(space, style, border_padding, block_size_func,
                            style.LogicalMinHeight(), auto_min_length,
                            override_available_size),
      ResolveMaxBlockLength(space, style, border_padding,
                            style.LogicalMaxHeight(), block_size_func,
                            override_available_size)};

  // 1) 用 max 限制 auto min-size
  if (auto_min_length && style.LogicalMinHeight().HasAuto()) {
    sizes.min_size = std::min(sizes.min_size, sizes.max_size);
  }
  // 2) table 不能低于 min-intrinsic size
  if (node.IsTable()) {
    sizes.Encompass(block_size_func(SizeType::kIntrinsic));
  }
  // 3) max 至少和 min 一样大
  sizes.max_size = std::max(sizes.max_size, sizes.min_size);
  return sizes;
}
```

逻辑很直白：解析 `min-height` / `max-height` → 三步修正。

- **`auto_min_length`**：当 `min-height` 本身是 `auto` 时，用上层传入的 `auto_min_length`（通常是 `Length::MinIntrinsic()`，即 aspect-ratio 的 automatic minimum）作为替代。解析后再用 max 把它压住（`min = min(min, max)`），避免自动最小尺寸超过 max。
- **table 规则**：`block_size_func(SizeType::kIntrinsic)` 取表格的 min-intrinsic block size，`Encompass` 保证不小于它。

## 3.2 `ComputeMinMaxInlineSizes`

```cpp
MinMaxSizes ComputeMinMaxInlineSizes(
    const ConstraintSpace& space, const BlockNode& node,
    const BoxStrut& border_padding, const Length* auto_min_length,
    MinMaxSizesFunctionRef min_max_sizes_func,
    TransferredSizesMode transferred_sizes_mode,
    FitContentMode fit_content_mode,
    LayoutUnit override_available_size) {
  const ComputedStyle& style = node.Style();
  MinMaxSizes sizes = {
      ResolveMinInlineLength(space, style, border_padding, min_max_sizes_func,
                             style.LogicalMinWidth(), auto_min_length,
                             override_available_size, fit_content_mode),
      ResolveMaxInlineLength(space, style, border_padding, min_max_sizes_func,
                             style.LogicalMaxWidth(), override_available_size,
                             fit_content_mode)};

  // 1) 用 max 限制 auto min-size
  if (auto_min_length && style.LogicalMinWidth().HasAuto()) {
    sizes.min_size = std::min(sizes.min_size, sizes.max_size);
  }

  // 2) aspect-ratio transferred min/max（css-sizing-4 #aspect-ratio-size-transfers）
  if (transferred_sizes_mode == TransferredSizesMode::kNormal &&
      !style.AspectRatio().IsAuto() && style.LogicalWidth().HasAuto() &&
      space.InlineAutoBehavior() != AutoSizeBehavior::kStretchExplicit) {
    MinMaxSizes transferred_sizes =
        ComputeMinMaxInlineSizesFromAspectRatio(space, node, border_padding);
    sizes.min_size = std::max(
        sizes.min_size, std::min(transferred_sizes.min_size, sizes.max_size));
    sizes.max_size = std::min(sizes.max_size, transferred_sizes.max_size);
  }

  // 3) table 不能低于 min-intrinsic size
  if (node.IsTable()) {
    sizes.Encompass(min_max_sizes_func(SizeType::kIntrinsic).sizes.min_size);
  }
  sizes.max_size = std::max(sizes.max_size, sizes.min_size);
  return sizes;
}
```

比 block 版多第 2 步——**transferred sizes**，这是 inline 轴特有的（因为 `width: auto` + aspect-ratio 时，inline 尺寸会从 block 的 min/max 推导）。条件：有 aspect-ratio、`width` 是 auto、且不是显式 stretch。转移规则见 3.3。

`transferred_sizes_mode == kIgnore` 时跳过这步——flex 在测量 item 的 min/max-content contribution 时会忽略 transferred size（因为它要的是纯内容贡献）。

## 3.3 transferred sizes 链：`ComputeMinMaxInlineSizesFromAspectRatio` → `ComputeInitialMinMaxBlockSizes` → `ComputeTransferredMinMaxInlineSizes`

```cpp
MinMaxSizes ComputeMinMaxInlineSizesFromAspectRatio(
    const ConstraintSpace& constraint_space, const BlockNode& node,
    const BoxStrut& border_padding) {
  const ComputedStyle& style = node.Style();
  DCHECK(!style.AspectRatio().IsAuto());
  // 先算 block 方向的 min/max（用 Initial 变体，不解析 intrinsic 关键字）
  const MinMaxSizes block_min_max =
      ComputeInitialMinMaxBlockSizes(constraint_space, node, border_padding);
  // 通过 aspect-ratio 转成 inline 方向
  return ComputeTransferredMinMaxInlineSizes(style.LogicalAspectRatio(),
                                             block_min_max, border_padding,
                                             style.BoxSizingForAspectRatio());
}
```

`ComputeInitialMinMaxBlockSizes` 用 `ResolveInitialMinBlockLength` / `ResolveInitialMaxBlockLength`（1.x 的 Initial 变体），所以不依赖内容布局：

```cpp
MinMaxSizes ComputeInitialMinMaxBlockSizes(...) {
  MinMaxSizes sizes = {ResolveInitialMinBlockLength(..., style.LogicalMinHeight(), ...),
                       ResolveInitialMaxBlockLength(..., style.LogicalMaxHeight(), ...)};
  sizes.max_size = std::max(sizes.max_size, sizes.min_size);
  return sizes;
}
```

`ComputeTransferredMinMaxInlineSizes` 把每个 block 尺寸通过 `InlineSizeFromAspectRatio` 投影：

```cpp
MinMaxSizes ComputeTransferredMinMaxInlineSizes(
    const LogicalSize& ratio, const MinMaxSizes& block_min_max,
    const BoxStrut& border_padding, const EBoxSizing sizing) {
  MinMaxSizes transferred = {LayoutUnit(), LayoutUnit::Max()};
  if (block_min_max.min_size > LayoutUnit())
    transferred.min_size = InlineSizeFromAspectRatio(border_padding, ratio, sizing, block_min_max.min_size);
  if (block_min_max.max_size != LayoutUnit::Max())
    transferred.max_size = InlineSizeFromAspectRatio(border_padding, ratio, sizing, block_min_max.max_size);
  transferred.max_size = std::max(transferred.max_size, transferred.min_size);  // min 胜出
  return transferred;
}
```

语义：把 `min-height` / `max-height` 通过 aspect-ratio 换算成等价的 `min-width` / `max-width` 约束，按 [css-sizing-4 #aspect-ratio-size-transfers](https://drafts.csswg.org/css-sizing-4/#aspect-ratio-size-transfers)。

**关键语义：这组 transferred min/max 是「约束区间」，用来 clamp min-content / max-content，而非重新计算它们。**

整条链的产物 `transferred` 来自元素的 **min-block-size / max-block-size**（`min-height` / `max-height` 解析后经 AR 投影到 inline 轴），它本身**不涉及子节点内容**。它的用途是作为 `MinMaxSizes` 区间，对已经从内容算出的 min-content / max-content 做 `ClampSizeToMinAndMax`：

```cpp
// 调用方（ComputeMinMaxInlineSizes 或 BlockNode::ComputeMinMaxSizes 3.9）：
result.min_size = transferred.ClampSizeToMinAndMax(result.min_size);  // min-content 不低于 transferred min
result.max_size = transferred.ClampSizeToMinAndMax(result.max_size);  // max-content 不高于 transferred max
```

即：当 `width: auto` + aspect-ratio 且 block-size 不确定时，inline 方向的 min-content / max-content 必须落在「由 min-block-size / max-block-size 经 AR 换算出的 inline 范围」内。block 方向的 min/max 约束通过 size-transfer 机制变成 inline 方向的约束，再夹住内容算出的 min/max-content。

注意 `ComputeInitialMinMaxBlockSizes` 用的是 **Initial 变体**（`ResolveInitialMinBlockLength` / `ResolveInitialMaxBlockLength`，回调永远返回 indefinite），所以这里的 block min/max 只来自 `min-height` / `max-height` 的显式长度（fixed/percent/calc），**不会**因为 `min-height: min-content` 之类去触发内容布局——这正是它能在「测量阶段」安全调用的原因。

---

# 4. Layer 4：`ComputeInlineSizeForFragment` / `ComputeBlockSizeForFragment`

解析**主尺寸**（`width` / `height`），并用 Layer 3 的 min/max clamp。

## 4.1 `ComputeInlineSizeForFragment`（入口）

```cpp
LayoutUnit ComputeInlineSizeForFragment(
    const ConstraintSpace& space, const BlockNode& node,
    const BoxStrut& border_padding, MinMaxSizesFunctionRef min_max_sizes_func) {
  if (space.IsFixedInlineSize() || space.IsAnonymous())
    return space.AvailableSize().inline_size;          // 固定尺寸/匿名：直接用可用尺寸
  if (node.IsTable())
    return To<TableNode>(node).ComputeTableInlineSize(space, border_padding);  // table 专用
  return ComputeInlineSizeForFragmentInternal(space, node, border_padding, min_max_sizes_func);
}
```

短路：固定 inline 尺寸（父级强制）或匿名 box → 直接返回可用尺寸；table → 走 table 专用逻辑（table 的 inline size 由 table-layout 算法决定）。其余进 Internal。

## 4.2 `ComputeInlineSizeForFragmentInternal`（核心）

三件事：决定 `auto_length`、决定是否应用 automatic minimum、解析主长度并 clamp。

```cpp
LayoutUnit ComputeInlineSizeForFragmentInternal(
    const ConstraintSpace& space, const BlockNode& node,
    const BoxStrut& border_padding, MinMaxSizesFunctionRef min_max_sizes_func) {
  const auto& style = node.Style();
  const Length& logical_width = style.LogicalWidth();

  // (1) aspect-ratio 是否可能生效
  const bool may_apply_aspect_ratio = ([&]() {
    if (style.AspectRatio().IsAuto()) return false;
    // height:auto 且非显式 stretch 时，优先用 inline 轴尺寸，不走 AR
    if (style.LogicalHeight().HasAuto() &&
        space.BlockAutoBehavior() != AutoSizeBehavior::kStretchExplicit)
      return false;
    // 能在不依赖 intrinsic-size 的情况下解析出 block-size 才用 AR
    return ComputeBlockSizeForFragment(space, node, border_padding,
                                       kIndefiniteSize, kIndefiniteSize) != kIndefiniteSize;
  })();

  // (2) 决定 width:auto 时的替代长度
  const Length& auto_length = ([&]() {
    if (space.AvailableSize().inline_size == kIndefiniteSize) return Length::MinContent();
    if (space.InlineAutoBehavior() == AutoSizeBehavior::kStretchExplicit) return Length::Stretch();
    if (may_apply_aspect_ratio) return Length::FitContent();
    if (space.InlineAutoBehavior() == AutoSizeBehavior::kStretchImplicit) return Length::Stretch();
    DCHECK_EQ(space.InlineAutoBehavior(), AutoSizeBehavior::kFitContent);
    return Length::FitContent();
  })();

  // (3) 是否应用 automatic minimum（css-sizing-4 #aspect-ratio-minimum）
  bool apply_automatic_min_size = ([&]() {
    if (style.IsScrollContainer()) return false;     // 滚动容器不应用
    if (!may_apply_aspect_ratio) return false;
    if (logical_width.HasContentOrIntrinsic()) return true;
    if (logical_width.HasAuto() && auto_length.HasContentOrIntrinsic()) return true;
    return false;
  })();

  // (4) 解析主长度，再用 min/max clamp
  const LayoutUnit extent = ResolveMainInlineLength(
      space, style, border_padding, min_max_sizes_func, logical_width, &auto_length);
  return ComputeMinMaxInlineSizes(
             space, node, border_padding,
             apply_automatic_min_size ? &Length::MinIntrinsic() : nullptr,
             min_max_sizes_func)
      .ClampSizeToMinAndMax(extent);
}
```

**`auto_length` 决策表**（`width: auto` 时用什么）：

| 条件 | auto_length |
|------|-------------|
| 可用 inline 尺寸 indefinite | `MinContent()`（shrink-to-fit 的最小） |
| 显式 stretch | `Stretch()`（撑满） |
| aspect-ratio 可能生效 | `FitContent()` |
| 隐式 stretch | `Stretch()` |
| 否则（fit-content 行为） | `FitContent()` |

**automatic minimum**：当 width 是 intrinsic 关键字或 auto+intrinsic auto_length，且有 aspect-ratio（非滚动容器）时，min 用 `MinIntrinsic()`（参考 aspect-ratio 的 automatic minimum，参考 css-sizing-4#aspect-ratio-minimum）。传给 `ComputeMinMaxInlineSizes` 的 `auto_min_length`。

## 4.3 `ComputeBlockSizeForFragment`（入口）

```cpp
LayoutUnit ComputeBlockSizeForFragment(const ConstraintSpace& constraint_space,
                                       const BlockNode& node,
                                       const BoxStrut& border_padding,
                                       LayoutUnit intrinsic_size,
                                       LayoutUnit inline_size,
                                       LayoutUnit override_available_size) {
  DCHECK(override_available_size == kIndefiniteSize || node.IsTable());  // override 仅 table 用

  if (constraint_space.IsFixedBlockSize()) {
    LayoutUnit block_size = override_available_size == kIndefiniteSize
                                ? constraint_space.AvailableSize().block_size
                                : override_available_size;
    if (constraint_space.MinBlockSizeShouldEncompassIntrinsicSize())
      return std::max(intrinsic_size, block_size);     // 取大
    return block_size;
  }
  if (constraint_space.IsTableCell() && intrinsic_size != kIndefiniteSize)
    return intrinsic_size;                             // table-cell：用 intrinsic
  if (constraint_space.IsAnonymous())
    return intrinsic_size;                             // 匿名：用 intrinsic
  return ComputeBlockSizeForFragmentInternal(
      constraint_space, node, border_padding, intrinsic_size, inline_size,
      override_available_size);
}
```

短路：固定 block 尺寸 → 用可用尺寸（必要时取 `max(intrinsic, available)`）；table-cell / 匿名 → 直接用 `intrinsic_size`（block 高度由内容决定）。其余进 Internal。

## 4.4 `ComputeBlockSizeForFragmentInternal`（核心，匿名 namespace）

与 inline 版对称，但 block 轴的 intrinsic size 是**外部传入**的（`intrinsic_size` 参数，因为 block 高度要等子节点布局完才知道），而 inline 轴是**回调按需算**的。

```cpp
LayoutUnit ComputeBlockSizeForFragmentInternal(
    const ConstraintSpace& space, const BlockNode& node,
    const BoxStrut& border_padding, LayoutUnit intrinsic_size,
    LayoutUnit inline_size, LayoutUnit override_available_size) {
  const ComputedStyle& style = node.Style();

  // table-cell 的百分比尺寸子元素特殊处理
  if (space.IsRestrictedBlockSizeTableCellChild())
    return ResolveInitialMinBlockLength(space, style, border_padding,
                                        style.LogicalMinHeight(), override_available_size);

  const Length& logical_height = style.LogicalHeight();
  const bool has_aspect_ratio = !style.AspectRatio().IsAuto();
  const bool may_apply_aspect_ratio = has_aspect_ratio && inline_size != kIndefiniteSize;

  // (1) auto_length（与 inline 对称，但 block 默认 FitContent）
  const Length& auto_length = ([&]() {
    if (space.AvailableSize().block_size == kIndefiniteSize) return Length::FitContent();
    if (space.BlockAutoBehavior() == AutoSizeBehavior::kStretchExplicit) return Length::Stretch();
    if (may_apply_aspect_ratio) return Length::FitContent();
    if (space.BlockAutoBehavior() == AutoSizeBehavior::kStretchImplicit) return Length::Stretch();
    return Length::FitContent();
  })();

  // (2) automatic minimum（需要 intrinsic_size 已知）
  bool apply_automatic_min_size = ([&]() {
    if (intrinsic_size == kIndefiniteSize) return false;
    if (style.IsScrollContainer()) return false;
    if (!may_apply_aspect_ratio) return false;
    if (logical_height.HasContentOrIntrinsic()) return true;
    if (logical_height.HasAuto() && auto_length.HasContentOrIntrinsic()) return true;
    return false;
  })();

  // (3) block_size_func：有 AR 且 inline_size 已知时，从 inline 推 block；否则用 intrinsic_size
  auto BlockSizeFunc = [&](SizeType type) {
    if (type == SizeType::kContent && has_aspect_ratio && inline_size != kIndefiniteSize) {
      return BlockSizeFromAspectRatio(
          border_padding, style.LogicalAspectRatio(),
          style.BoxSizingForAspectRatio(), inline_size);
    }
    return intrinsic_size;
  };

  // (4) 解析主长度
  const LayoutUnit extent = ResolveMainBlockLength(
      space, style, border_padding, logical_height, &auto_length, BlockSizeFunc,
      override_available_size);
  if (extent == kIndefiniteSize) {
    DCHECK_EQ(intrinsic_size, kIndefiniteSize);
    return extent;                                     // 解析不出，透传
  }

  // (5) 用 min/max clamp
  MinMaxSizes min_max = ComputeMinMaxBlockSizes(
      space, node, border_padding,
      apply_automatic_min_size ? &Length::MinIntrinsic() : nullptr,
      BlockSizeFunc, override_available_size);

  // 分页时可能需要把 intrinsic 纳入 min
  if (space.MinBlockSizeShouldEncompassIntrinsicSize() &&
      intrinsic_size != kIndefiniteSize) {
    min_max.Encompass(std::min(intrinsic_size, min_max.max_size));
  }
  return min_max.ClampSizeToMinAndMax(extent);
}
```

**关键差异**（vs inline）：

- `BlockSizeFunc` 是个 lambda：有 aspect-ratio 且 inline_size 已知时，用 `BlockSizeFromAspectRatio` 从 inline 反推 block（这样 `height: auto` + aspect-ratio 时能用上 inline 尺寸）；否则返回外部传入的 `intrinsic_size`。
- `extent == kIndefiniteSize` 时直接返回（block 高度未知是常态，调用方会继续走内容布局）。
- 多一个分页相关的 `MinBlockSizeShouldEncompassIntrinsicSize` 处理。

---

# 5. Layer 5：`CalculateInitialFragmentGeometry`

布局算法的入口前奏：在真正跑布局前，先算出节点的 border-box 尺寸和 border/scrollbar/padding 四个 `BoxStrut`。

## 5.1 两个重载

```cpp
// 重载 A：显式传 MinMaxSizesFunctionRef
FragmentGeometry CalculateInitialFragmentGeometry(
    const ConstraintSpace& space, const BlockNode& node,
    const BlockBreakToken* break_token,
    MinMaxSizesFunctionRef min_max_sizes_func,
    bool is_intrinsic = false);

// 重载 B：用节点自己的 ComputeMinMaxSizes 包成回调
FragmentGeometry CalculateInitialFragmentGeometry(
    const ConstraintSpace& space, const BlockNode& node,
    const BlockBreakToken* break_token,
    bool is_intrinsic = false) {
  auto MinMaxSizesFunc = [&](SizeType type) -> MinMaxSizesResult {
    return node.ComputeMinMaxSizes(space.GetWritingMode(), type, space);
  };
  return CalculateInitialFragmentGeometry(space, node, break_token, MinMaxSizesFunc, is_intrinsic);
}
```

重载 B 是便利接口：把 `node.ComputeMinMaxSizes` 包成 `MinMaxSizesFunctionRef` 再委托给 A。大多数调用方用 B；需要自定义 min/max 行为的（如 flex/grid 测量）用 A。

## 5.2 重载 A 实现

```cpp
FragmentGeometry CalculateInitialFragmentGeometry(
    const ConstraintSpace& space, const BlockNode& node,
    const BlockBreakToken* break_token,
    MinMaxSizesFunctionRef min_max_sizes_func, bool is_intrinsic) {
  const auto& style = node.Style();

  // (1) frameset 特殊处理
  if (node.IsFrameSet()) {
    if (node.IsParentNGFrameSet()) {
      const auto size = space.AvailableSize();
      DCHECK_NE(size.inline_size, kIndefiniteSize);
      DCHECK_NE(size.block_size, kIndefiniteSize);
      return {size, {}, {}, {}};
    }
    const auto size = node.InitialContainingBlockSize();
    return {ToLogicalSize(size, style.GetWritingMode()), {}, {}, {}};
  }

  // (2) 算 border / padding / scrollbar
  const auto border = ComputeBorders(space, node);
  const auto padding = ComputePadding(space, style);
  auto scrollbar = ComputeScrollbars(space, node);
  const auto border_padding = border + padding;
  const auto border_scrollbar_padding = border_padding + scrollbar;

  // (3) replaced 元素：走 ComputeReplacedSize
  if (node.IsReplaced()) {
    const auto border_box_size = ComputeReplacedSize(
        node, space, border_padding,
        is_intrinsic ? ReplacedSizeMode::kIgnoreInlineLengths
                     : ReplacedSizeMode::kNormal);
    return {border_box_size, border, scrollbar, padding};
  }

  // (4) inline size（intrinsic 模式下跳过，留 kIndefiniteSize）
  const LayoutUnit inline_size =
      is_intrinsic ? kIndefiniteSize
                   : ComputeInlineSizeForFragment(space, node, border_padding, min_max_sizes_func);

  // (5) scrollbar 比 content 还宽时的钳制（左侧 scrollbar 不破坏 scrollWidth）
  if (inline_size != kIndefiniteSize &&
      inline_size < border_scrollbar_padding.InlineSum() &&
      scrollbar.InlineSum() && !space.IsAnonymous()) [[unlikely]] {
    const auto content_box_inline_size = inline_size - border_padding.InlineSum();
    if (scrollbar.InlineSum() > content_box_inline_size) {
      // 把 scrollbar 钳到 content box 宽度，按 start/end 分配
      if (scrollbar.inline_start && scrollbar.inline_end) {
        scrollbar.inline_start = content_box_inline_size / 2;
        scrollbar.inline_end = content_box_inline_size - scrollbar.inline_start;
      } else if (scrollbar.inline_end) {
        scrollbar.inline_end = content_box_inline_size;
      } else {
        scrollbar.inline_start = content_box_inline_size;
      }
    }
  }

  // (6) block size（quirks 模式下 html/body 的默认高度）
  const auto default_block_size = CalculateDefaultBlockSize(
      space, node, break_token, border_scrollbar_padding);
  const auto block_size = ComputeInitialBlockSizeForFragment(
      space, node, border_padding, default_block_size, inline_size);

  return {LogicalSize(inline_size, block_size), border, scrollbar, padding};
}
```

**步骤要点**：

1. **frameset**：直接用可用尺寸或 ICB 尺寸，无 border/padding/scrollbar。
2. **border/padding/scrollbar**：三个 `BoxStrut` 是后续所有尺寸解析的基础（`ResolveInlineLengthInternal` 等都要 `border_padding`）。
3. **replaced**：替换元素尺寸由 `ComputeReplacedSize` 算（用 natural size + aspect-ratio，独立于普通流）。
4. **`is_intrinsic`**：为 `true` 时 inline_size 留 `kIndefiniteSize`。这是 `BlockNode::ComputeMinMaxSizes` 调用时的模式——算 min/max-content 时不能预设 inline 约束（否则循环依赖：算 inline size 要 min/max，算 min/max 又调本函数）。
5. **scrollbar 钳制**：处理「scrollbar 比内容区还宽」的边界情况（左侧 scrollbar 会破坏 scrollWidth，见 crbug.com/724255）。
6. **block size**：通过 `ComputeInitialBlockSizeForFragment` 算，quirks 模式下 html/body 用 `CalculateDefaultBlockSize`（填满 ICB）。

## 5.3 `ComputeInitialBlockSizeForFragment`（初始 block size 包装）

```cpp
LayoutUnit ComputeInitialBlockSizeForFragment(...) {
  if (space.IsInitialBlockSizeIndefinite())
    return intrinsic_size;                  // 大多数情况：block size 初始不定
  return ComputeBlockSizeForFragment(space, node, border_padding,
                                     intrinsic_size, inline_size, override_available_size);
}
```

**关键**：布局开始时 block size 通常是 `kIndefiniteSize`（要等子节点布局完才知道高度）。`IsInitialBlockSizeIndefinite()` 为 true 时直接返回 `intrinsic_size`（往往也是 indefinite），让后续布局去确定。只有少数情况（固定高度、table-cell 等）会在初始阶段就解析出 block size。

## 5.4 `CalculateDefaultBlockSize`（quirks 模式 html/body）

```cpp
LayoutUnit CalculateDefaultBlockSize(const ConstraintSpace& space, const BlockNode& node,
                                     const BlockBreakToken* break_token,
                                     const BoxStrut& border_scrollbar_padding) {
  // quirks 模式下 html/body 填满 ICB，百分比高度据此解析
  if (node.IsQuirkyAndFillsViewport() && !IsBreakInside(break_token)) {
    LayoutUnit block_size = space.AvailableSize().block_size;
    block_size -= ComputeMarginsForSelf(space, node.Style()).BlockSum();
    return std::max(block_size.ClampNegativeToZero(),
                    border_scrollbar_padding.BlockSum());
  }
  return kIndefiniteSize;
}
```

只在 quirks 模式的 html/body 上生效（历史上 body 百分比高度要相对 viewport 解析），其余返回 indefinite。

## 5.5 `is_intrinsic` 参数详解（及与 `kContent` / `kIntrinsic` 的关系）

`is_intrinsic` 是 `CalculateInitialFragmentGeometry` 的 `bool` 参数（默认 `false`）。它和 `SizeType` 的 `kIntrinsic` 都叫 "intrinsic" 但**不是一回事**，需要严格区分。

### 直接作用

在 `CalculateInitialFragmentGeometry` 内只门控三处：

```cpp
// (1) inline_size：is_intrinsic=true 时跳过，留 kIndefiniteSize
const LayoutUnit inline_size =
    is_intrinsic ? kIndefiniteSize
                 : ComputeInlineSizeForFragment(space, node, border_padding, min_max_sizes_func);

// (2) replaced 元素：改用 kIgnoreInlineLengths（忽略 inline 长度，只用 transferred/natural size）
if (node.IsReplaced()) {
  const auto border_box_size = ComputeReplacedSize(
      node, space, border_padding,
      is_intrinsic ? ReplacedSizeMode::kIgnoreInlineLengths
                   : ReplacedSizeMode::kNormal);
}
// (3) block_size 不直接受 is_intrinsic 门控（见下）
```

核心是第 (1) 条：**intrinsic 模式下不算 inline-size，留作 `kIndefiniteSize`**。

### 为什么需要它：打破循环依赖

算 inline-size 和算 min/max-content 互相依赖：

```
ComputeInlineSizeForFragmentInternal
  → 若 width 是 min-content/max-content/fit-content
    → 调 min_max_sizes_func 回调
      → node.ComputeMinMaxSizes(...)
        → CalculateInitialFragmentGeometry(...)   ← 又回来了
```

若 `ComputeMinMaxSizes` 里也以 `is_intrinsic=false` 调本函数，会再触发 `ComputeInlineSizeForFragment`，无限递归。所以 `BlockNode::ComputeMinMaxSizes` 调用时**强制传 `is_intrinsic=true`**：

```cpp
// block_node.cc，ComputeMinMaxSizes 内
cached_fragment_geometry = CalculateInitialFragmentGeometry(
    constraint_space, *this, /*break_token*/ nullptr, /*is_intrinsic*/ true);
```

inline-size 留 indefinite，不再回头调 `ComputeInlineSizeForFragment`，循环断开。这正是「intrinsic 测量不施加 inline 约束」的实现——min/max-content 本来就要求「不限定宽度」。正常布局 `BlockNode::Layout` 调本函数时用默认 `false`，inline-size 正常算。

### 三个 "intrinsic" 的命名辨析

| 名字 | 层面 | 含义 |
|------|------|------|
| `is_intrinsic`（bool） | 测量**阶段** | 「正在做 intrinsic 测量，别算 inline-size」 |
| `SizeType::kIntrinsic`（枚举） | 测量**结果变体** | 「给我不考虑 AR 反推的那个 min/max 变体」 |
| `Length::kMinIntrinsic` | **Length 关键字** | 请求 `kIntrinsic` 变体的那个 length |

三者都挂在 CSS「intrinsic sizing」大概念下，但分别在阶段、结果、关键字三个层面，互不相同。

### 与 `kContent` / `kIntrinsic` 的关系

**正交**：`is_intrinsic` 作用在 fragment geometry 层（控制 inline-size 算不算）；`type`(kContent/kIntrinsic) 作用在 `ComputeMinMaxSizes` 层（控制 3.6 的 AR 反推走不走）。两者作用层不同。

**同源**：都为「intrinsic 测量」服务——`is_intrinsic=true` 保证测量不被 inline 约束污染，`type` 选择测量结果是否含 AR 推导。而且 `type` 所在的 `ComputeMinMaxSizes` 正是 `is_intrinsic=true` 的调用方，二者同属一次测量调用。

**间接耦合**（最关键）：`is_intrinsic=true` 让 inline_size indefinite，进而影响「`height:auto` + AR」时 block_size 能否用 AR 推出，从而间接影响 3.6（看 block_size）是否触发。分两种情况：

**情况 A：`height` 确定或可独立解析（如 `height: 100px`）**

`ComputeBlockSizeForFragmentInternal` 走 fixed/percent 分支，block_size 算出确定值（100px）。即使 `is_intrinsic=true`，block_size 仍确定。

→ 3.6 触发条件（block_size 确定）满足：**`type=kContent` 走 AR 反推，`type=kIntrinsic` 不走**。两者差异显现。

**情况 B：`height: auto` + aspect-ratio**

正常布局（`is_intrinsic=false`）时 inline_size 被算出（如 width:100px），`ComputeBlockSizeForFragmentInternal` 里：

```cpp
const bool may_apply_aspect_ratio = has_aspect_ratio && inline_size != kIndefiniteSize;  // true
auto BlockSizeFunc = [&](SizeType type) {
  if (type == kContent && has_aspect_ratio && inline_size != kIndefiniteSize)
    return BlockSizeFromAspectRatio(..., inline_size);  // 用 AR 从 inline 推 block
  return intrinsic_size;
};
```

block_size 由 AR 从 inline 推出，确定。

但 `is_intrinsic=true` 时 inline_size = indefinite → `may_apply_aspect_ratio = false` → `BlockSizeFunc` 退化成返回 `intrinsic_size`（kIndefiniteSize）→ block_size 解析不出 → **block_size 也变 indefinite**。

→ 3.6 触发条件不满足：**`kContent` 和 `kIntrinsic` 行为一致**（都走算法路径 + 3.9 transferred clamp）。

### 交互一览表

| 场景 | is_intrinsic | inline_size | block_size | 3.6 触发？ | kContent vs kIntrinsic |
|------|-------------|-------------|-----------|-----------|----------------------|
| `height:100px` + AR，测量 min/max | true | indefinite | **100px（确定）** | 看 type | **有差异**（kContent 走 AR） |
| `height:auto` + AR，测量 min/max | true | indefinite | indefinite（AR 推不出） | 否 | 无差异 |
| 正常布局（非测量） | false | 算出 | 算出（可能用 AR） | — | — |

### 精确结论

1. **正交**：`is_intrinsic` 门控 fragment geometry 的 inline-size；`type` 门控 `ComputeMinMaxSizes` 内 3.6 的 AR 反推。
2. **同源**：都为 intrinsic 测量服务，且 `type` 所在函数正是 `is_intrinsic=true` 的调用方。
3. **间接耦合**：`is_intrinsic=true` 让 inline_size indefinite，进而使「`height:auto`+AR」的 block_size 也无法用 AR 推出（保持 indefinite），从而让 3.6 不触发、`kContent`/`kIntrinsic` 趋同。但当 `height` 本身确定时，block_size 与 `is_intrinsic` 无关，3.6 仍由 `type` 决定。

即：**`is_intrinsic` 不改变 `type` 的语义，但通过让 inline_size indefinite，间接缩小了 `kContent` 与 `kIntrinsic` 产生差异的范围**——只剩「height 可独立解析 + 有 AR」这一种情况两者才不同。

---

# 6. aspect-ratio 尺寸换算辅助

被上述函数反复调用的两个工具：

```cpp
// block_size → inline_size
LayoutUnit InlineSizeFromAspectRatio(const BoxStrut& border_padding,
                                     const LogicalSize& aspect_ratio,
                                     EBoxSizing box_sizing, LayoutUnit block_size) {
  if (box_sizing == EBoxSizing::kBorderBox) {
    return std::max(border_padding.InlineSum(),
        block_size.MulDiv(aspect_ratio.inline_size, aspect_ratio.block_size));
  }
  block_size -= border_padding.BlockSum();
  return block_size.MulDiv(aspect_ratio.inline_size, aspect_ratio.block_size) +
         border_padding.InlineSum();
}

// inline_size → block_size（对称）
LayoutUnit BlockSizeFromAspectRatio(const BoxStrut& border_padding,
                                    const LogicalSize& aspect_ratio,
                                    EBoxSizing box_sizing, LayoutUnit inline_size) {
  DCHECK_GE(inline_size, border_padding.InlineSum());
  if (box_sizing == EBoxSizing::kBorderBox) {
    return std::max(border_padding.BlockSum(),
        inline_size.MulDiv(aspect_ratio.block_size, aspect_ratio.inline_size));
  }
  inline_size -= border_padding.InlineSum();
  return inline_size.MulDiv(aspect_ratio.block_size, aspect_ratio.inline_size) +
         border_padding.BlockSum();
}
```

- `border-box`：直接按比例缩放 border-box，下限为对应方向的 border+padding。
- `content-box`：先减去 block/inline 方向的 border+padding 算出 content-box，按比例缩放，再加回另一方向的 border+padding。

---

# 7. 类型与枚举速查

```cpp
enum class SizeType { kContent, kIntrinsic };   // kIntrinsic 不考虑 aspect-ratio
using MinMaxSizesFunctionRef = base::FunctionRef<MinMaxSizesResult(SizeType)>;
using BlockSizeFunctionRef   = base::FunctionRef<LayoutUnit(SizeType)>;

enum class LengthTypeInternal { kMin, kMain, kMax };  // 解析 min / 主 / max 长度
enum class FitContentMode { kNormal, kMinContribution, kMaxContribution };  // fit-content 在 indefinite 时的退化
enum class ReplacedSizeMode { kNormal, kIgnoreInlineLengths, kIgnoreBlockLengths };
enum class TransferredSizesMode { kNormal, kIgnore };  // 是否应用 aspect-ratio transferred size
```

`LengthTypeInternal` 是整组函数的灵魂——同一个 `Resolve*LengthInternal` 通过它复用三种语义：

| LengthType | 用途 | indefinite 百分比 | `auto` |
|-----------|------|-------------------|--------|
| `kMin` | min-width/min-height | 当 0 | border+padding |
| `kMain` | width/height | inline→indefinite；block→退化为内容 | 由 `auto_length` 替换 |
| `kMax` | max-width/max-height | indefinite | indefinite（即「无上限」） |

---

# 8. 关键设计要点

1. **分层 + 回调**：底层只懂 fixed/percent/calc，intrinsic 关键字靠回调按需计算，避免解析阶段强制触发 min/max 测量。`is_intrinsic` 模式下连 inline_size 都不算，打破「算 inline 要 min/max、算 min/max 要 inline」的循环依赖。
2. **`LengthTypeInternal` 复用**：一个 Internal 函数覆盖 min/main/max 三种语义，差异体现在 indefinite 百分比和 `auto` 的处理上——min 宽容（当 0 / border+padding），max 严格（indefinite = 无上限），main 居中（inline 透传、block 退化内容）。
3. **`kIndefiniteSize` 哨兵 + 包装层转译**：内部统一用 indefinite 表示「算不出」，外层 `Resolve*` 按 min/max/main 各自转成语义默认值，调用方拿到的永远是合法值（除 main 透传外）。
4. **transferred sizes**：`width: auto` + aspect-ratio 时，把 block 的 min/max 通过 AR 换算成 inline 的 min/max 约束，实现 css-sizing-4 的 size transfer。`Initial` 变体保证这步不依赖内容布局。
5. **automatic minimum**：aspect-ratio 元素的 min-size 有一个「自动最小尺寸」（基于 AR），通过 `auto_min_length = &Length::MinIntrinsic()` 注入 `ComputeMinMax*Sizes`，并用 max 压住避免超过 max-size。滚动容器跳过此规则。
6. **block 与 inline 的不对称**：inline 的 intrinsic size 由回调按需算（`MinMaxSizesFunctionRef`），block 的 intrinsic size 是外部传入的单一值（`intrinsic_size` 参数）——因为 block 高度要等子节点布局完才确定，不能在解析阶段回调。
7. **短路优先**：`Compute*SizeForFragment` 在固定尺寸 / table-cell / 匿名 / table 等情况下直接返回，避免不必要的 Internal 调用。
8. **box-sizing 贯穿**：每个 percent/fixed/calc 分支末尾都按 `border-box`（clamp）/ `content-box`（加）处理 border+padding，aspect-ratio 换算也区分两种。

---

# 附：相关文件索引

| 文件                                               | 内容                                                                                                                                                                         |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `core/layout/length_utils.h`                     | `Resolve*` 包装（inline）、枚举、`Compute*SizeForFragment` / `ComputeMinMax*Sizes` 声明                                                                                              |
| `core/layout/length_utils.cc`                    | `ResolveInlineLengthInternal` / `ResolveBlockLengthInternal`、`Compute*SizeForFragmentInternal`、`CalculateInitialFragmentGeometry`、transferred size 链、`ComputeReplacedSize` |
| `core/platform/geometry/length_functions.{h,cc}` | `MinimumValueForLength` / `ValueForLength`（fixed/percent/calc → LayoutUnit）                                                                                                |
|                                                  |                                                                                                                                                                            |
