# 0. 这是什么

`ComputeMinMaxSizes` 计算 CSS intrinsic sizing 中的 **min-content / max-content 内联尺寸（inline size）**，供 shrink-to-fit、表格列宽、flex/grid item 测量等场景使用。

它解决两件事：

1. **算出** `{min_size, max_size}`——这是相对父容器书写模式的 inline 方向上的 min/max-content。
2. **算出** `depends_on_block_constraints`——结果是否依赖于 block 方向约束。因为 inline size 有时会依赖 block size（最典型是 `aspect-ratio` 宽高比，其次是百分比高度的传递），这个标记决定上层能否在 block size 变化时复用缓存。

```css
/* min/max-content 可由 height + aspect-ratio 反推 */
width: auto;
height: 100px;
aspect-ratio: 1 / 2;
```

---

# 1. 类型定义（结合源码）

## 1.1 `MinMaxSizes` — `core/layout/min_max_sizes.h`

```cpp
struct MinMaxSizes {
  LayoutUnit min_size;
  LayoutUnit max_size;

  void Encompass(const MinMaxSizes& other);  // min/max 各取 max
  void Encompass(LayoutUnit value);          // min/max 都至少为 value
  void Constrain(LayoutUnit value);          // min/max 都至多为 value
  LayoutUnit ShrinkToFit(LayoutUnit available) const;            // shrink-to-fit
  LayoutUnit ClampSizeToMinAndMax(LayoutUnit size) const;        // clamp 到 [min,max]
  MinMaxSizes& operator+=(MinMaxSizes extra);
  MinMaxSizes& operator+=(LayoutUnit length);   // 同时加到 min/max
  MinMaxSizes& operator-=(LayoutUnit length);
  void operator=(LayoutUnit value);             // min=max=value
};
```

> 注意：`MinMaxSizes` **没有** `InlineSum` 方法。代码里看到的 `BorderScrollbarPadding().InlineSum()` 是 `BoxStrut::InlineSum()`，而 `sizes += BorderScrollbarPadding().InlineSum()` 走的是 `operator+=(LayoutUnit)`。

## 1.2 `MinMaxSizesResult` — `core/layout/min_max_sizes.h`

```cpp
struct MinMaxSizesResult {
  MinMaxSizesResult(MinMaxSizes sizes, bool depends_on_block_constraints);
  // 仅 BlockNode::ComputeMinMaxSizes 内部使用（aspect-ratio 直接推导时）：
  MinMaxSizesResult(MinMaxSizes sizes, bool depends_on_block_constraints,
                    bool applied_aspect_ratio);

  MinMaxSizes sizes;
  bool depends_on_block_constraints = false;
  bool applied_aspect_ratio = false;
};
```

源码注释明确：`applied_aspect_ratio` **不向上传播**，仅本层使用；`depends_on_block_constraints` 才是向上传播的标记。

## 1.3 `SizeType` — `core/layout/length_utils.h`

```cpp
// min/max-content take the CSS aspect-ratio property into account.
// In some cases that's undesirable; this enum lets you choose not
// to do that using |kIntrinsic|.
enum class SizeType { kContent, kIntrinsic };
```

- `kContent`：min/max-content，**考虑** `aspect-ratio`（用于 `width: min-content` 等显式 intrinsic 长度）。
- `kIntrinsic`：**不考虑** `aspect-ratio`（用于 flex base size 等内部计算）。

#### 精确语义：`type` 只门控 3.6

`type` 在整个 `ComputeMinMaxSizes` 里**只作用在一处**——3.6 节的 aspect-ratio 直接反推：

```cpp
if (has_aspect_ratio && type == SizeType::kContent) {   // 只 kContent 进
  if (fragment_geometry.border_box_size.block_size != kIndefiniteSize) {  // 且 block 确定
    return MinMaxSizesResult({inline_size_from_ar, inline_size_from_ar}, ...);  // min = max = block/ratio
  }
}
```

所以 `kContent` 与 `kIntrinsic` 的差异**只在「block-size 确定 + 有 aspect-ratio」时才显现**：`kContent` 直接用 AR 反推（min = max = `block_size / ratio`），`kIntrinsic` 跳过 3.6、坚持从子节点内容算。

注意 3.9 节（block-size 不确定时的 transferred min/max clamp）**不分** `type` **都会跑**。因此「kIntrinsic 不考虑 aspect-ratio」是简化说法——准确讲是「跳过 AR 的直接推导（3.6）」，transfer/clamp（3.9）仍照常。

四种情况一览：

| block-size | type | 行为 |
| --- | --- | --- |
| 确定 | `kContent` | 走 3.6，min = max = `block_size / ratio` |
| 确定 | `kIntrinsic` | 跳过 3.6，走完整算法从子节点内容算 |
| 不确定 | `kContent` | 3.6 本就不触发，走算法 + 3.9 transferred clamp |
| 不确定 | `kIntrinsic` | 同上——与 `kContent` **行为一致** |

> 何时 block-size 确定、何时不定，还受 `is_intrinsic`（见文档 2 Layer 5）间接影响——`is_intrinsic=true` 会让 `height:auto`+AR 的 block-size 也解析不出，从而让 3.6 不触发、两种 type 趋同。详见文档 2。

## 1.4 `MinMaxSizesFloatInput` — `core/layout/layout_input_node.h`

```cpp
struct MinMaxSizesFloatInput {
  LayoutUnit float_left_inline_size;
  LayoutUnit float_right_inline_size;
};
```

同一 BFC 内，子节点需要知道「旁边有哪些 float」才能正确计算 inline 占用。父节点把当前行已累计的左右 float 宽度通过它传给子节点。

## 1.5 函数签名

```cpp
MinMaxSizesResult BlockNode::ComputeMinMaxSizes(
    WritingMode container_writing_mode,
    const SizeType type,
    const ConstraintSpace& constraint_space,
    const MinMaxSizesFloatInput float_input) const;
```

| 参数 | 含义 |
| --- | --- |
| `container_writing_mode` | 父容器书写模式，用于判断正交流。 |
| `type` | `kContent` / `kIntrinsic`，决定 3.6 的 aspect-ratio 直接反推走不走（详见 1.3 精确语义）。 |
| `constraint_space` | 当前约束空间。 |
| `float_input` | 旁边 float 的累计 inline size，默认 0。 |

---

# 2. 整体流程

```
ComputeMinMaxSizes
 ├─ 0. ListItem marker 文本更新
 ├─ 1. 惰性 IntrinsicFragmentGeometry
 ├─ 2. Grid / Grid-Lanes / 列方向 Flex 早期返回   (仅 border+padding)
 ├─ 3. 正交流根 → 完整 Layout 取 inline size
 ├─ 4. DependsOnBlockConstraints lambda
 ├─ 5. Replaced → border_box inline size
 ├─ 6. aspect-ratio + kContent + block 确定 → 反推 inline size
 ├─ 7. 缓存查询（indefinite / definite-block-size 两层）
 ├─ 8. 未命中 → 算法计算 + ContentMinimumInlineSize + 写缓存
 ├─ 9. aspect-ratio 最终 clamp（block 不确定时）
 └─ 10. 汇总 depends_on_block_constraints
```

---

# 3. 逐步实现

## 3.0 ListItem marker

```cpp
if (IsListItem())
  To<LayoutListItem>(box_.Get())->UpdateMarkerTextIfNeeded();
```

marker 宽度影响 inline size，必须先刷新。

## 3.1 惰性几何 `IntrinsicFragmentGeometry`

```cpp
std::optional<FragmentGeometry> cached_fragment_geometry;
auto IntrinsicFragmentGeometry = [&]() -> FragmentGeometry& {
  if (!cached_fragment_geometry) {
    cached_fragment_geometry = CalculateInitialFragmentGeometry(
        constraint_space, *this, /* break_token */ nullptr,
        /* is_intrinsic */ true);
  }
  return *cached_fragment_geometry;
};
```

惰性求值：`FragmentGeometry`（border/padding/border*box*size 等）较贵，而第 2/5/6 步可能直接返回，根本不需要它，所以只在首次访问时算一次。

`is_intrinsic = true` 的语义：intrinsic 模式下把 inline size 当作 `kIndefiniteSize`（不预设宽度约束），这正是 min/max-content「不限定宽度」的要求。

## 3.2 Grid / Grid-Lanes / 列方向 Flex 早期返回

```cpp
const bool is_in_perform_layout = box_->GetFrameView()->IsInPerformLayout();
if (!is_in_perform_layout &&
    (IsGrid() || IsGridLanes() ||
     (IsFlexibleBox() && Style().ResolvedIsColumnFlexDirection()))) {
  const FragmentGeometry& fragment_geometry = IntrinsicFragmentGeometry();
  const BoxStrut border_padding = fragment_geometry.border + fragment_geometry.padding;
  MinMaxSizes sizes;
  sizes.min_size = border_padding.InlineSum();
  sizes.max_size = sizes.min_size;
  return MinMaxSizesResult(sizes, /* depends_on_block_constraints */ false);
}
```

**为什么**：Grid / Grid-Lanes / 列方向 Flex 在 MinMax 计算时会对其 item 跑布局。若当前**不在** `PerformLayout`（只是测量），跑子项布局可能产生并缓存错误结果。因此这种情况下只返回 `border + padding`，把真正的计算推迟到这些算法在正式布局阶段内部完成。

## 3.3 正交流根（Orthogonal Flow Root）

```cpp
bool is_orthogonal_flow_root =
    !IsParallelWritingMode(container_writing_mode, Style().GetWritingMode());

if (is_orthogonal_flow_root) {
  MinMaxSizes sizes;
  CHECK(is_in_perform_layout);

  std::optional<DisableLayoutSideEffectsScope> disable_side_effects;
  if (!GetLayoutBox()->NeedsLayout())
    disable_side_effects.emplace();

  const LayoutResult* layout_result = Layout(constraint_space);
  DCHECK_EQ(layout_result->Status(), LayoutResult::kSuccess);
  sizes = LogicalFragment({container_writing_mode, TextDirection::kLtr},
                          layout_result->GetPhysicalFragment())
              .InlineSize();
  const bool depends_on_block_constraints =
      Style().LogicalWidth().HasAuto() ||
      Style().LogicalWidth().HasPercentOrStretch() ||
      Style().LogicalMinWidth().HasPercentOrStretch() ||
      Style().LogicalMaxWidth().HasPercentOrStretch();
  return MinMaxSizesResult(sizes, depends_on_block_constraints);
}
```

正交流时（父 `horizontal-tb`，自身 `vertical-lr`），父要的「inline size」其实是自身的「block size」，intrinsic 公式推不出，**必须跑一次完整 Layout**，再按父容器书写模式从片段取 inline size。

三个关键点：

1. `CHECK(is_in_perform_layout)`：正交流根的 MinMax 不允许在纯测量阶段触发。
2. `DisableLayoutSideEffectsScope`：若元素本身 clean，Layout 期间禁用副作用，避免中间结果写回 `LayoutObject` 树污染全局状态（`FinishLayout` 里会走 `AddMeasureLayoutResult` 分支）。
3. `depends_on_block_constraints` 仅由自身 `width` 相关样式判定。

## 3.4 `DependsOnBlockConstraints` lambda

```cpp
auto DependsOnBlockConstraints = [&]() -> bool {
  return Style().LogicalHeight().HasPercentOrStretch() ||
         Style().LogicalMinHeight().HasPercentOrStretch() ||
         Style().LogicalMaxHeight().HasPercentOrStretch() ||
         (Style().LogicalHeight().HasAuto() &&
          constraint_space.IsBlockAutoBehaviorStretch());
};
```

自身 `height` / `min-height` / `max-height` 为百分比或 stretch，或 `height: auto` 且 auto 行为是 stretch（如 `align-items: stretch`）→ 自身 inline 尺寸**可能**依赖 block 约束。在第 5、6、10 步复用。

## 3.5 替换元素

```cpp
if (IsReplaced()) {
  MinMaxSizes sizes;
  sizes = IntrinsicFragmentGeometry().border_box_size.inline_size;
  return {sizes, DependsOnBlockConstraints()};
}
```

替换元素（`<img>`、`<video>` 等）尺寸由内容固有尺寸决定，min = max = border\_box inline size。不缓存——大多数调用方要的是 min/max content contribution（直接走 `ComputeReplacedSize`），缓存收益小；此处主要服务 flex。

## 3.6 aspect-ratio + block\_size 确定 → 直接反推

```cpp
const bool has_aspect_ratio = !Style().AspectRatio().IsAuto();
if (has_aspect_ratio && type == SizeType::kContent) {
  const FragmentGeometry& fragment_geometry = IntrinsicFragmentGeometry();
  const BoxStrut border_padding = fragment_geometry.border + fragment_geometry.padding;
  if (fragment_geometry.border_box_size.block_size != kIndefiniteSize) {
    const LayoutUnit inline_size_from_ar = InlineSizeFromAspectRatio(
        border_padding, Style().LogicalAspectRatio(),
        Style().BoxSizingForAspectRatio(),
        fragment_geometry.border_box_size.block_size);
    return MinMaxSizesResult({inline_size_from_ar, inline_size_from_ar},
                             DependsOnBlockConstraints(),
                             /* applied_aspect_ratio */ true);
  }
}
```

条件：有 `aspect-ratio` + `kContent` + block*size 确定。此时 min = max = 由 block*size 通过 aspect-ratio 反推的 inline*size，并标记 \`applied*aspect\_ratio = true\`（第 9 步据此跳过重复 clamp）。

反推实现 `InlineSizeFromAspectRatio`（`core/layout/length_utils.cc`）：

```cpp
LayoutUnit InlineSizeFromAspectRatio(const BoxStrut& border_padding,
                                     const LogicalSize& aspect_ratio,
                                     EBoxSizing box_sizing,
                                     LayoutUnit block_size) {
  if (box_sizing == EBoxSizing::kBorderBox) {
    return std::max(border_padding.InlineSum(),
        block_size.MulDiv(aspect_ratio.inline_size, aspect_ratio.block_size));
  }
  block_size -= border_padding.BlockSum();
  return block_size.MulDiv(aspect_ratio.inline_size, aspect_ratio.block_size) +
         border_padding.InlineSum();
}
```

- `border-box`：`max(border+padding, block_size * inline/block)`，下限为 border+padding。
- `content-box`：先把 block\_size 减去 block 方向 border+padding，再按比例缩放，最后加回 inline 方向 border+padding。

## 3.7 缓存查询

```cpp
bool can_use_cached_intrinsic_inline_sizes =
    CanUseCachedIntrinsicInlineSizes(constraint_space, float_input, *this);

if (!can_use_cached_intrinsic_inline_sizes) {
  box_->SetIntrinsicLogicalWidthsDirty(kMarkOnlyThis);
}

std::optional<MinMaxSizesResult> result;

// 第一层：indefinite 缓存
if (can_use_cached_intrinsic_inline_sizes &&
    !box_->IntrinsicLogicalWidthsDependsOnBlockConstraints()) {
  result = box_->CachedIndefiniteIntrinsicLogicalWidths();
}

// 第二层：按确定的 block-size 查
if (!result && can_use_cached_intrinsic_inline_sizes &&
    !UseParentPercentageResolutionBlockSizeForChildren()) {
  result = box_->CachedIntrinsicLogicalWidths(
      IntrinsicFragmentGeometry().border_box_size.block_size);
}
```

#### `CanUseCachedIntrinsicInlineSizes` 判定（`block_node.cc`）

返回 `false`（不可用）的情形：

1. `IntrinsicLogicalWidthsDirty()`——已脏。
2. `float_input` 非零——不存储 float inline size，无法比对。
3. 存在百分比 padding。
4. 是 table-cell 且其 border sizes 与 `constraint_space.TableCellBorders()` 不一致。
5. 是 Grid / Grid-Lanes 且 `min/max-width` 为百分比或 stretch；或有 aspect-ratio 且 `min/max-height` 为百分比或 stretch（此时 min/max 依赖 `min-width` 解析值，是 grid 特有情况）。

判定不可用即 `SetIntrinsicLogicalWidthsDirty(kMarkOnlyThis)` 使缓存失效。

#### `UseParentPercentageResolutionBlockSizeForChildren()`（`block_node.cc`）

一般为 `false`。为 `true` 表示子元素的百分比 block-size 应使用**父级**的解析 block-size（quirks mode 下跳过 auto-height 容器、匿名 block、`<input type=range>` 的 thumb 等）。此时无法用本元素的 block-size 作 key 查第二层缓存，跳过。

## 3.8 未命中：算法计算 + 修正 + 写缓存

```cpp
if (!result) {
  const FragmentGeometry& fragment_geometry = IntrinsicFragmentGeometry();
  result = ComputeMinMaxSizesWithAlgorithm(
      LayoutAlgorithmParams(*this, fragment_geometry, constraint_space),
      float_input);

  const BoxStrut border_padding =
      fragment_geometry.border + fragment_geometry.padding;
  if (auto min_size = ContentMinimumInlineSize(*this, border_padding)) {
    result->sizes.min_size = *min_size;
  }

  box_->SetIntrinsicLogicalWidths(
      fragment_geometry.border_box_size.block_size, *result);
  if (IsTableCell()) {
    To<LayoutTableCell>(box_.Get())
        ->SetIntrinsicLogicalWidthsBorderSizes(
            constraint_space.TableCellBorders());
  }
}
```

三步：

1. `ComputeMinMaxSizesWithAlgorithm`（`block_node.cc`）：通过 `DetermineAlgorithmAndRun` 按 `display` 分发到具体算法，调其 `ComputeMinMaxSizes(float_input)`。
2. `ContentMinimumInlineSize` **修正**：部分元素的 min 不能低于某下限。
3. **写缓存** + table-cell 记录 border sizes。

### 3.8.1 算法分发表（`block_node.cc` `DetermineAlgorithmAndRun`）

| 谓词 | 算法类 |
| --- | --- |
| `IsFlexibleBox()` | `FlexLayoutAlgorithm` |
| `IsTable()` | `TableLayoutAlgorithm` |
| `IsTableRow()` / `IsTableSection()` | `TableRow/SectionLayoutAlgorithm` |
| `IsLayoutCustom()` | `CustomLayoutAlgorithm` |
| `IsMathML()` | `DetermineMathMLAlgorithmAndRun`（再分发） |
| `IsLayoutGrid()` | `GridLayoutAlgorithm` |
| `IsLayoutGridLanes()` | `GridLanesLayoutAlgorithm` |
| `IsLayoutReplaced()` | `ReplacedLayoutAlgorithm` |
| `IsFieldset()` / `IsFrameSet()` / `IsMulticolContainer()` | 对应算法 |
| 无父 + `IsPaginatedRoot()` [[unlikely]] | `PaginatedRootLayoutAlgorithm` |
| else（兜底） | `BlockLayoutAlgorithm` |

> `CreateAlgorithmAndRun` 用模板 + `NOINLINE`，目的是避免每个节点布局时在栈上分配所有算法对象。

### 3.8.2 `ContentMinimumInlineSize`（`block_node.cc`）

返回 `std::nullopt` → 不特殊处理；否则返回一个强制下限覆盖 `min_size`。规则：

- table：返回 `nullopt`（table 不允许低于 min-intrinsic size）。
- 水平 `<marquee>`：返回 `border_padding.InlineSum()`。
- `LogicalWidth()` 为百分比时：将其对 0 解析（`MinimumValueForLength`），按 `box-sizing` 加 border/padding 得下限。
- 部分表单控件返回该下限：text control、`<select>`、`<input type=file>`（需 `apply_form_sizing`）、`<input type=range>`。

语义：某些控件即使内容很小，也不应收缩到比「百分比 width 对 0 解析」更小。

## 3.9 aspect-ratio 最终 clamp

```cpp
if (has_aspect_ratio) {
  const FragmentGeometry& fragment_geometry = IntrinsicFragmentGeometry();
  if (fragment_geometry.border_box_size.block_size == kIndefiniteSize) {
    const BoxStrut border_padding =
        fragment_geometry.border + fragment_geometry.padding;
    const MinMaxSizes min_max = ComputeMinMaxInlineSizesFromAspectRatio(
        constraint_space, *this, border_padding);
    result->sizes.min_size = min_max.ClampSizeToMinAndMax(result->sizes.min_size);
    result->sizes.max_size = min_max.ClampSizeToMinAndMax(result->sizes.max_size);
  }
}
```

与 3.6 互补：3.6 处理「block 确定 → 反推」并已 `return`；这里处理「block 不确定 → block 将由 aspect-ratio 推导」，需用 aspect-ratio 自身的 min/max inline 约束对结果 clamp。

`ComputeMinMaxInlineSizesFromAspectRatio`（`core/layout/length_utils.cc`）：

```cpp
MinMaxSizes ComputeMinMaxInlineSizesFromAspectRatio(...) {
  // 仅在 width 不确定时使用，故无需按 preferred size clamp。
  const MinMaxSizes block_min_max =
      ComputeInitialMinMaxBlockSizes(constraint_space, node, border_padding);
  return ComputeTransferredMinMaxInlineSizes(style.LogicalAspectRatio(),
                                             block_min_max, border_padding,
                                             style.BoxSizingForAspectRatio());
}
```

即：先算 block 方向的 min/max，再通过 aspect-ratio **transfer** 成 inline 方向的 min/max，用作 clamp 区间。参考 css-sizing-4 §aspect-ratio。

**关键语义**：这里算出的 `min_max` 是一组 **min/max inline 约束区间**，它的来源是元素的 **min-block-size / max-block-size**（即 `min-height` / `max-height` 解析后经 aspect-ratio 换算到 inline 轴）。它**不是**重新计算 min-content/max-content，而是用这组 block 方向的约束去 **clamp 已经算好的 min-content / max-content**：

```cpp
result->sizes.min_size = min_max.ClampSizeToMinAndMax(result->sizes.min_size);  // clamp min-content
result->sizes.max_size = min_max.ClampSizeToMinAndMax(result->sizes.max_size);  // clamp max-content
```

换句话说，当 `width: auto` + aspect-ratio 且 block-size 不确定时，inline 方向的 min-content / max-content 不能超过「由 min-height / max-block-size 经 AR 换算出的 inline 范围」——min-content 不低于 transferred min，max-content 不高于 transferred max。这正是 css-sizing-4 §aspect-ratio-size-transfers 要求的 size transfer。

## 3.10 汇总 `depends_on_block_constraints`

```cpp
result->depends_on_block_constraints =
    (DependsOnBlockConstraints() ||
     UseParentPercentageResolutionBlockSizeForChildren()) &&
    (result->depends_on_block_constraints || has_aspect_ratio);
return *result;
```

需**同时**满足：

1. **自身层面**：`DependsOnBlockConstraints()` 或 `UseParentPercentageResolutionBlockSizeForChildren()`（自身/子级会用父级 block-size 解析百分比高度）；**且**
2. **结果层面**：子级算法报告了 `depends_on_block_constraints`，或自身有 `aspect-ratio`。

含义：只有当「自身可能因 block-size 变化而改 inline size」**且**「计算确实用到这种依赖」时才向上报告，避免对固定尺寸元素的无谓重算。

---

# 4. 缓存机制（结合 `LayoutBox` / `LayoutObject` / `MinMaxSizesCache`）

## 4.1 存储成员（`core/layout/layout_box.h`）

```cpp
MinMaxSizes intrinsic_logical_widths_;        // 单值：indefinite 槽
Member<MinMaxSizesCache> min_max_sizes_cache_;  // N-way LRU：definite 槽
```

## 4.2 `MinMaxSizesCache` — `core/layout/min_max_sizes_cache.h`

8 路 LRU，按 `initial_block_size` 索引：

```cpp
class MinMaxSizesCache final : public GarbageCollected<MinMaxSizesCache> {
  static constexpr unsigned kMaxCacheEntries = 8;  // grid 多遍可达 10，取 8
  struct Entry { MinMaxSizes sizes; LayoutUnit initial_block_size; bool depends_on_block_constraints; };

  std::optional<MinMaxSizesResult> Find(LayoutUnit initial_block_size) {
    // 命中则移到末尾（LRU），返回结果
  }
  void Add(...) {
    if (cache_.size() == kMaxCacheEntries) cache_.EraseAt(0);  // 淘汰最旧
    cache_.push_back(Entry{...});
  }
  Vector<Entry, 2> cache_;
};
```

设计动机（源码注释）：grid 等算法会对同一元素用**不同** initial block-size 多次查询 MinMax；当存在 aspect-ratio + 百分比高度这类依赖时，不同 block-size 会得到不同结果，故按 block-size 做 LRU。

## 4.3 写入：`SetIntrinsicLogicalWidths`（`core/layout/layout_box.h`）

```cpp
void SetIntrinsicLogicalWidths(LayoutUnit initial_block_size, const MinMaxSizesResult& result) {
  if (initial_block_size == kIndefiniteSize || !result.depends_on_block_constraints) {
    // → indefinite 槽
    intrinsic_logical_widths_ = result.sizes;
    SetIntrinsicLogicalWidthsDependsOnBlockConstraints(result.depends_on_block_constraints);
    SetIndefiniteIntrinsicLogicalWidthsDirty(false);
  } else {
    // → definite LRU 槽（按 initial_block_size 索引）
    if (!min_max_sizes_cache_) min_max_sizes_cache_ = MakeGarbageCollected<MinMaxSizesCache>();
    else if (DefiniteIntrinsicLogicalWidthsDirty()) min_max_sizes_cache_->Clear();
    min_max_sizes_cache_->Add(result.sizes, initial_block_size, result.depends_on_block_constraints);
    SetDefiniteIntrinsicLogicalWidthsDirty(false);
  }
  ClearIntrinsicLogicalWidthsDirty();
}
```

分流条件：`block_size 不确定` 或 `结果不依赖 block 约束` → 单值槽；否则 → LRU 槽。

## 4.4 读取

```cpp
// indefinite 槽
MinMaxSizesResult CachedIndefiniteIntrinsicLogicalWidths() const {
  DCHECK(!IntrinsicLogicalWidthsDirty());
  return {intrinsic_logical_widths_, IntrinsicLogicalWidthsDependsOnBlockConstraints()};
}

// 按 block-size 查
std::optional<MinMaxSizesResult> CachedIntrinsicLogicalWidths(LayoutUnit initial_block_size) const {
  if (initial_block_size == kIndefiniteSize) {
    if (IndefiniteIntrinsicLogicalWidthsDirty()) return std::nullopt;
    return MinMaxSizesResult(intrinsic_logical_widths_, IntrinsicLogicalWidthsDependsOnBlockConstraints());
  }
  if (min_max_sizes_cache_) {
    if (DefiniteIntrinsicLogicalWidthsDirty()) return std::nullopt;
    return min_max_sizes_cache_->Find(initial_block_size);
  }
  return std::nullopt;
}
```

## 4.5 脏标记与传播（`core/layout/layout_object.cc`）

`SetIntrinsicLogicalWidthsDirty` 会同时置 4 个 bit 并向上传播：

```cpp
void LayoutObject::SetIntrinsicLogicalWidthsDirty(MarkingBehavior mark_parents) {
  bitfields_.SetIntrinsicLogicalWidthsDirty(true);
  bitfields_.SetIntrinsicLogicalWidthsDependsOnBlockConstraints(true);
  bitfields_.SetIndefiniteIntrinsicLogicalWidthsDirty(true);
  bitfields_.SetDefiniteIntrinsicLogicalWidthsDirty(true);
  if (mark_parents == kMarkContainerChain && (IsText() || !StyleRef().HasOutOfFlowPosition()))
    InvalidateContainerIntrinsicLogicalWidths();  // 向上传播
}
```

注意 `ComputeMinMaxSizes` 里调用的是 `SetIntrinsicLogicalWidthsDirty(kMarkOnlyThis)`（不向上传播），仅在判定缓存不可用时清理本节点。

---

# 5. 下游算法的 `ComputeMinMaxSizes`

`ComputeMinMaxSizesWithAlgorithm` 分发后，具体算法各自实现 `ComputeMinMaxSizes(float_input)`，递归合并子节点。下面以 Block 和 Flex 为例，对照真实源码。

## 5.0 心智模型 vs 实现

min-content / max-content 的 CSS 规范直觉：

- **Block**：子节点垂直堆叠，每行一个 in-flow 子节点 → 容器 inline size = **各子节点 contribution 的最大值**（min 用子节点 min，max 用子节点 max）。
- **Flex-row**：max-content = 所有 item 在一行的 max 之和；min-content = nowrap 时全部相加，wrap 时取单个最大。
- **Flex-column**（单行）：与 block 一致，取最大。

这个直觉**是对的**。代码只是在此基础上叠加了 flex-basis / `cant_move` 等修正（见 5.2），并未推翻模型本身。

## 5.1 Block — `BlockLayoutAlgorithm::ComputeMinMaxSizes`（`core/layout/block_layout_algorithm.cc`）

**核心思想**：block 中每个 in-flow 子节点各占一行，父的 MinMax = **所有子节点 contribution 的最大者**，再叠加 border/scrollbar/padding。

### 步骤 1：尝试忽略子节点

```cpp
if (auto result = CalculateMinMaxSizesIgnoringChildren(node_, BorderScrollbarPadding()))
  return *result;
```

`CalculateMinMaxSizesIgnoringChildren`（`core/layout/length_utils.cc`）在三种情况下直接返回（`depends_on_block_constraints = false`）：

1. **OverrideIntrinsicContentInlineSize** 非无限——有显式 intrinsic 覆盖。
2. **DefaultIntrinsicContentInlineSize** 非无限——如 `<textarea>` 的默认尺寸（且 textarea 会减去 scrollbar 宽度）。
3. **inline-size containment** 或无子节点——返回 `border+padding`。

否则返回 `nullopt`，进入子节点遍历。

### 步骤 2：遍历子节点

跳过：OOF 定位、column-span-all（在 column bfc 中）、text control placeholder（`ApplyControlFixedSize` 时）。

对每个子节点：

1. **clear 处理**：float / new-FC 子节点若有 `clear`，先把它之前的 float 累计宽度纳入 `max_size`，再按 `clear` 方向清零对应 float 追踪器。
2. **构造子 constraint space**（`MinMaxConstraintSpaceBuilder`），传百分比解析 block-size、replaced 百分比等。
3. **递归算子节点 MinMax**：inline 子走 `InlineNode::ComputeMinMaxSizes`；block 子走 `ComputeMinAndMaxContentContribution`（内部递归 `BlockNode::ComputeMinMaxSizes`）。
4. **算 margin**：inline 子无 margin；block 子用 `ComputeMarginsFor`。
5. **算 max inline contribution**，按子节点类型分三种：

```cpp
if (child.IsFloating()) {
  // float 累加到当前行；max_contribution = 左右 float 之和
  LayoutUnit float_inline_size = child_result.sizes.max_size + margins.InlineSum();
  if (float_inline_size > 0) {
    if (child_style.Floating(Style()) == EFloat::kLeft) float_left_inline_size += float_inline_size;
    else float_right_inline_size += float_inline_size;
  }
  max_inline_contribution = float_left_inline_size + float_right_inline_size;
} else if (child_is_new_fc) {
  // new-FC 的 margin 与同行 float 取 max/相加，得到 line_left/right_inset
  LayoutUnit line_left_inset = margin_line_left > 0
      ? std::max(float_left_inline_size, margin_line_left)
      : float_left_inline_size + margin_line_left;
  LayoutUnit line_right_inset = margin_line_right > 0
      ? std::max(float_right_inline_size, margin_line_right)
      : float_right_inline_size + margin_line_right;
  // 顺序重要：先合并 inset 再加，避免 max_size 饱和时触发 DCHECK
  max_inline_contribution = child_result.sizes.max_size + (line_left_inset + line_right_inset);
} else {
  // 普通 in-flow 子：sizes + margin
  max_inline_contribution = child_result.sizes.max_size + margins.InlineSum();
}
sizes.max_size = std::max(sizes.max_size, max_inline_contribution);
```

1. **min inline contribution**：假设所有 float 各自独占一行，故 `min = child.min + margins.InlineSum()`（不叠加 float），取 max。
2. **行结束**：非 float 子节点开新行，清零 float 追踪器。
3. **依赖传播**：`depends_on_block_constraints |= child_result.depends_on_block_constraints`。

最后 `sizes += BorderScrollbarPadding().InlineSum()` 返回。

## 5.2 Flex — `FlexLayoutAlgorithm::ComputeMinMaxSizes`（`core/layout/flex/flex_layout_algorithm.cc`）

先同样尝试 `CalculateMinMaxSizesIgnoringChildren`，然后按方向分发：

```cpp
if (!is_column_) return ComputeMinMaxSizeOfRowContainer();          // 行方向
if (is_multi_line_) return ComputeMinMaxSizeOfMultilineColumnContainer(); // 列+换行
// 列+不换行：fast path
```

> `is_column_` = `ResolvedIsColumnFlexDirection()`（`column` / `column-reverse`）；`is_multi_line_` = `!ResolvedIsFlexNowrap()`（`wrap` / `wrap-reverse`）。

### 5.2.1 列 + 不换行（fast path）

与 block 一致——取所有子节点 contribution 的最大值：

```cpp
for (BlockNode child = iterator.NextChild(); child; ...) {
  if (child.IsOutOfFlowPositioned()) continue;
  MinMaxSizesResult child_result = ComputeMinAndMaxContentContribution(Style(), child, space);
  child_result.sizes += child_margins.InlineSum();
  sizes.min_size = std::max(sizes.min_size, child_result.sizes.min_size);
  sizes.max_size = std::max(sizes.max_size, child_result.sizes.max_size);
}
sizes.max_size = std::max(sizes.max_size, sizes.min_size);
sizes.Encompass(LayoutUnit());                  // 负 margin 防护
sizes += BorderScrollbarPadding().InlineSum();
```

### 5.2.2 列 + 换行（`ComputeMinMaxSizeOfMultilineColumnContainer`）

换行列容器无法用简单遍历，需跑 flex line-breaking：

```cpp
FlexLineVector flex_lines;
PlaceFlexItems(Phase::kColumnWrapIntrinsicSize, &flex_lines);
min_max_sizes.min_size = largest_min_content_contribution_;
for (const auto& line : flex_lines)
  min_max_sizes.max_size += line.line_cross_size;   // 各行 cross size 之和
min_max_sizes.max_size += (flex_lines.size() - 1) * gap_between_lines_;
// ...
return {min_max_sizes, /* depends_on_block_constraints */ true};  // 恒为 true
```

`depends_on_block_constraints` **恒为 true**：因为 block 约束变化会改变列数（换行数），进而改变 inline（cross）尺寸。

### 5.2.3 行方向（`ComputeMinMaxSizeOfRowContainer`）

最复杂的一条。先 `ConstructAndAppendFlexItems(Phase::kRowIntrinsicSize)` 算出每个 item 的 flex base size、hypothetical main size 等，再对每行累加 item contribution。

对每个 item，先用 `ComputeMinAndMaxContentContribution` 拿到 min/max contribution，再用 `cant_move` **启发式**决定最终 contribution：

```cpp
// min_size contribution
const bool cant_move = (min_contribution > flex_base_size_border_box && item.flex_grow == 0.f) ||
                       (min_contribution < flex_base_size_border_box && item.flex_shrink == 0.f);
if (cant_move && !item.is_used_flex_basis_indefinite)
  item_final_contribution.min_size = hypothetical_main_size_border_box;  // 用 hypothetical
else
  item_final_contribution.min_size = min_contribution;                   // 用原始 min-content

// max_size contribution：同样的 cant_move 逻辑
```

`cant_move` 语义：当 item 的 contribution 与 flex base size 的偏离方向**与 flex 因子相反**（即 grow=0 却要变大，或 shrink=0 却要变小），且 flex basis 确定，则用 hypothetical main size 代替原始 contribution（兼容性约束的「新算法」）。

行内累加：

```cpp
line_sizes += item_final_contribution;
line_sizes += main_axis_margins;
line_sizes += (line.count - 1) * gap_between_items_;   // 行内 gap
container_sizes.Encompass(line_sizes);                  // 跨行取 max
```

最终：

- `max_size`：各行 `line_sizes` 的 `Encompass`（取 max）。
- `min_size`：
  - `is_multi_line_`：取所有 item 中最大的 outer min-content contribution（每个 item 可独占一行）。
  - 单行：行内累加（含 `cant_move` 调整）。
- `container_sizes.max_size = max(max, min)`，`Encompass(0)` 防负，加 border/padding。

> 这与「row 方向 max 全相加、min nowrap 全相加 / wrap 取最大」的简化描述不同——真实实现引入了 flex base size / hypothetical size / `cant_move` 启发式，比单纯累加子节点 `min/max_size` 复杂得多。

---

# 6. 关键设计要点

1. **惰性几何**：`IntrinsicFragmentGeometry` 只算一次，让早期返回分支（Grid/Flex、Replaced、aspect-ratio 反推）零几何开销。
2. **副作用隔离**：正交流根走完整 Layout，但用 `DisableLayoutSideEffectsScope` 防止中间结果污染 `LayoutObject` 树。
3. **两层缓存 + LRU**：indefinite 单值槽 / definite 8 路 LRU，按「是否依赖 block 约束」分流；grid 多遍查询正是 LRU 的设计动机。
4. **依赖传播**：`depends_on_block_constraints` 由子级向上传播，本层结合自身样式（百分比高度、aspect-ratio）做最终判定，决定缓存可否跨 block-size 复用。
5. **aspect-ratio 两段处理**：block 确定 → 反推（3.6，已 return）；block 不确定 → transfer 后 clamp（3.9）。二者互斥。
6. **特殊控件下限**：`ContentMinimumInlineSize` 保证部分表单控件不收缩到不合理小尺寸。
7. **分发即模板 + NOINLINE**：`CreateAlgorithmAndRun` 避免每节点栈上分配所有算法对象。
8. **Block 的 float/clear/new-FC 处理**：float 累加到当前行并经 `float_input` 传给后续兄弟；clear 重置 float 追踪器；new-FC 的 margin 与同行 float 取 max 形成 inset。这是 block intrinsic size 计算的难点。

---

## 附：相关文件索引

| 文件 | 内容 |
| --- | --- |
| `core/layout/block_node.cc` | `BlockNode::ComputeMinMaxSizes`、`CanUseCachedIntrinsicInlineSizes`、`ContentMinimumInlineSize`、`DetermineAlgorithmAndRun` |
| `core/layout/min_max_sizes.h` | `MinMaxSizes`、`MinMaxSizesResult` |
| `core/layout/min_max_sizes_cache.h` | `MinMaxSizesCache`（8 路 LRU） |
| `core/layout/layout_box.h` | `SetIntrinsicLogicalWidths`、`Cached*`、缓存成员 |
| `core/layout/layout_object.cc` | `SetIntrinsicLogicalWidthsDirty`（脏标记传播） |
| `core/layout/length_utils.h` / `.cc` | `SizeType`、`InlineSizeFromAspectRatio`、`ComputeMinMaxInlineSizesFromAspectRatio`、`CalculateMinMaxSizesIgnoringChildren` |
| `core/layout/block_layout_algorithm.cc` | `BlockLayoutAlgorithm::ComputeMinMaxSizes` |
| `core/layout/flex/flex_layout_algorithm.cc` | `FlexLayoutAlgorithm::ComputeMinMaxSizes`、行/列容器实现 |
