---
_organized: true
---
# Chromium Flex 布局算法分析

通过 Chromium 源码（flex*layout*algorithm.cc、line\_flexer.cc）深入了解 CSS Flexbox 从规范到实现的完整过程。

## 整体流程

Flex 容器的布局完整流程可以分为两个主要阶段：

```cpp
// flex_layout_algorithm.cc
const LayoutResult* FlexLayoutAlgorithm::LayoutInternal() {
  // 防止滚条状态变化导致重排循环
  if (ignore_child_scrollbar_changes_)
    freeze_scrollbars.emplace();
  
  // 第一阶段：计算所有 flex items 的尺寸
  PlaceFlexItems(Phase::kLayout, &flex_lines, ...);
  
  // 第二阶段：计算位置和对齐
  GiveItemsFinalPositionAndSize(&flex_lines, ...);
}
```

## Flex Base Size 计算

在 `ConstructAndAppendFlexItems` 中，第一件事是遍历并初始化每个 flex item。

### 绝对定位元素的处理

```cpp
// flex_layout_algorithm.cc
for (BlockNode child = iterator.NextChild(); child;
     child = iterator.NextChild()) {
  if (child.IsOutOfFlowPositioned()) {
    oof_children->emplace_back(child.GetLayoutBox());
    continue;
  }
  // 处理 in-flow 子元素...
}
```

首先来看 flex item 的初始化阶段。遍历所有子元素时，首先要过滤掉 `position: absolute/fixed` 的元素，因为它们不参与 flex 布局，需要单独处理。

### 确定 Flex Base Size

flex-basis 的优先级很清晰。首先检查 flex-basis 是否为 auto，如果是则回退到 width/height，最后才使用内容尺寸。

```cpp
// flex_layout_algorithm.cc
const LayoutUnit flex_base_border_box = ([&]() -> LayoutUnit {
  if (flex_basis.HasAuto()) {
    // 检查 width/height 属性
    const Length& specified_length_in_main_axis =
        is_horizontal_flow_ ? child_style.Width() : child_style.Height();
    LayoutUnit auto_flex_basis_size = resolve_main_length(
        specified_length_in_main_axis, &auto_size_length);
  }
  
  // 解析 flex-basis 的实际值
  LayoutUnit main_size = resolve_main_length(flex_basis, ...);
  
  // 表格有额外处理：加上 caption 的高度
  if (const auto* table_child = DynamicTo<TableNode>(&child)) {
    main_size += table_child->ComputeCaptionBlockSize(child_space);
  }
  
  return main_size;
})();
```

关键的是 `resolve_main_length` 闭包，它处理三种情况：

- 具体长度（200px）：直接使用
- 百分比（50%）：相对容器主轴尺寸计算
- auto/content：调用 `MinMaxSizesFunc` 获取内容的 min-content 尺寸

注意表格这类特殊元素需要额外处理，要加上 caption 的高度。

### Automatic Minimum Size 机制

规范定义 flex items 的默认 `min-width: auto`（而非 0），这防止内容溢出：

```cpp
// flex_layout_algorithm.cc
std::optional<Length> auto_min_length;
if (ShouldApplyAutoMinSize(child)) {
  LayoutUnit content_size_suggestion = ...;    // min-content 尺寸
  LayoutUnit specified_size_suggestion = ...;  // width/height 值
  
  // automatic minimum size 取两者的较小值
  LayoutUnit auto_min_size =
      std::min(specified_size_suggestion, content_size_suggestion);
  
  auto_min_length = Length::Fixed(auto_min_size);
}
```

规范定义 flex items 的默认 `min-width: auto`（而非 0）。这个机制防止内容溢出，但也带来了一些常见问题。

这里的 auto minimum size 取两者的较小值：内容尺寸建议（min-content）和显式指定的尺寸（width/height 值）。为什么 `flex: 1` 无法收缩到小于内容宽度？正因为这个 auto minimum size 的存在。要克服这个限制，需要显式 `min-width: 0`。

### Hypothetical Main Size

在 flex base size 上应用 min/max 约束：

```cpp
// 对应 W3C Flexbox Section 9.2.3.E
hypothetical_main_size = clamp(flex_base_size, min_main_size, max_main_size)
```

最后一步是应用 min/max 约束。这对应 W3C 规范 Section 9.2.3.E，公式是 `clamp(flex_base_size, min_main_size, max_main_size)`。应用约束后，FlexItem 的核心字段就都有值了：

- `base_content_size`：flex base size（border-box）
- `hypothetical_content_size`：应用 min/max 后的尺寸
- `flex_grow / flex_shrink`：弹性因子
- `main_axis_border_padding`：border + padding

## 分行算法

分行算法决定哪些 items 在同一行，核心逻辑很简单。单行容器（`flex-wrap: nowrap`）把所有 items 放在一行。多行容器按规则分行：

```cpp
// flex_line_breaker.cc
for (auto& item : items) {
  LayoutUnit item_size = item.hypothetical_content_size +
                         item.main_axis_border_padding +
                         item.MainAxisMarginExtent();
  
  // 检查这个 item 是否应该换行
  bool should_break = is_multi_line &&
                      current_line_count > 0 &&
                      current_line_size + gap + item_size > line_break_size;
  
  if (should_break) {
    result.flex_lines.push_back({current_line_count, current_line_size});
    current_line_size = LayoutUnit();
    current_line_count = 0;
  }
  
  current_line_size += item_size;
}
```

判断逻辑就是：累计尺寸（包括 gap）超过容器了吗？超过就换行。这是很直观的处理方式。

## 弹性长度解析 - LineFlexer 核心算法

这是 Flexbox 实现的心脏。`LineFlexer::Run()` 处理有剩余空间时的分配（flex-grow）和空间不足时的收缩（flex-shrink）。核心思路是通过迭代和冻结机制，逐步逼近满足所有 min/max 约束的最终尺寸。

### 初始状态：冻结不灵活的 items

在初始化时就要识别出哪些 items 无法改变，直接冻结它们：

```cpp
// line_flexer.cc
LineFlexer::LineFlexer(base::span<FlexItem> line_items, ...)
    : mode_(sum_hypothetical_main_size < main_axis_inner_size ? kGrow
                                                              : kShrink) {
  // 冻结那些无法改变的 items
  FreezeItems([mode = mode_](const FlexItem& item) {
    const float flex_factor =
        (mode == kGrow) ? item.flex_grow : item.flex_shrink;
    
    if (flex_factor == 0.f) return true;  // flex-grow/shrink 为 0
    
    // 已经被 min/max 限制，无法继续调整
    return mode == kGrow
        ? item.base_content_size > item.hypothetical_content_size
        : item.base_content_size < item.hypothetical_content_size;
  });
}
```

第一次初始化时就冻结所有不能变的 items。无法增长/收缩的直接排除，避免后续无用计算。判断标准很清晰：flex-grow/shrink 为 0 的无法改变，或者已经被 min/max 限制的也无法继续调整。

### 核心迭代：ResolveFlexibleLengths

```cpp
// line_flexer.cc
bool LineFlexer::ResolveFlexibleLengths() {
  // 处理 flex-factor < 1 的情况：只分配比例空间
  if (total_flex_factor_ > 0.0 && total_flex_factor_ < 1.0) {
    LayoutUnit fractional(initial_free_space_ * total_flex_factor_);
    if (fractional.Abs() < free_space_.Abs()) {
      free_space_ = fractional;
    }
  }
  
  // 没有空间可分配，提前退出
  if ((mode_ == kGrow && free_space_ <= LayoutUnit()) ||
      (mode_ == kShrink && free_space_ >= LayoutUnit())) {
    return false;
  }
  
  // ★ 从后向前分配，避免浮点数累积误差
  LayoutUnit total_violation;
  for (auto& item : base::Reversed(line_items_)) {
    if (item.state == FlexerState::kFrozen) continue;
    
    // 最后一个 item 获得剩余的所有空间
    const LayoutUnit extra_size = (item.free_space_fraction == 1.0)
        ? free_space_
        : LayoutUnit::FromDoubleRound(free_space_ * item.free_space_fraction);
    free_space_ -= extra_size;
    
    const LayoutUnit item_size = item.base_content_size + extra_size;
    const LayoutUnit adjusted_item_size =
        item.main_axis_min_max_sizes.ClampSizeToMinAndMax(item_size);
    
    item.flexed_content_size = adjusted_item_size;
    
    // 记录是否违反 min/max 约束
    const LayoutUnit violation = adjusted_item_size - item_size;
    if (violation) {
      item.state = violation < LayoutUnit() ? FlexerState::kMaxViolation
                                            : FlexerState::kMinViolation;
    }
    total_violation += violation;
  }
  
  // 根据违规冻结对应的 items，继续迭代
  if (total_violation) {
    const FlexerState state = total_violation < LayoutUnit()
                                  ? FlexerState::kMaxViolation
                                  : FlexerState::kMinViolation;
    FreezeItems([state](const FlexItem& item) { return item.state == state; });
    return true;  // 需要继续迭代
  }
  
  return false;  // 所有 items 都满足约束，算法结束
}
```

这是整个算法的核心逻辑。首先处理 flex-factor < 1 的特殊情况：只分配比例空间，不分配全部剩余空间。然后检查是否有空间可分配，没有的话提前退出。

最关键的是从后向前分配空间。最后一个 item 获得剩余的所有空间，这样可以避免浮点数累积误差。每个 item 的新尺寸应用 min/max 约束后，如果违反约束就记录违规状态。最后根据违规情况冻结对应的 items，然后继续迭代。

这种反复冻结/解冻的机制能保证所有 min/max 约束都被满足。

### 三个关键设计要点

**从后向前分配**

这是避免浮点累积误差的技巧。最后一个 item 获得剩余的所有空间，保证总和精确。

例如 100px 分配给 3 个 flex-grow: 1 的 items，从后向前分配的过程是：

- 第 3 个：100 × 1/3 = 33px，剩余 67px
- 第 2 个：67 × 1/2 = 34px，剩余 33px
- 第 1 个：33 × 1 = 33px，完全消耗
- 总和：33 + 34 + 33 = 100px（精确）

这样就避免了正向分配中可能积累的舍入误差。

**flex-shrink 是加权分配**

增长模式按 `flex-grow` 比例分配，收缩模式按 `flex-shrink × base_size` 比例分配：

```cpp
// line_flexer.cc
if (mode_ == kGrow) {
  item.free_space_fraction = flex_factor / total_flex_factor_;
} else {
  // 收缩时：大 item 承担更多
  const double weighted_flex_shrink = flex_factor * item.base_content_size;
  item.free_space_fraction = weighted_flex_shrink / total_weighted_flex_shrink;
}
```

增长模式按 `flex-grow` 比例分配，收缩模式则不同。收缩时使用加权分配，权重是 `flex-shrink × base_size`。这意味着大 item 会承担更多的收缩。

实际例子：两个 item 都设 `flex-shrink: 1`，但一个 100px、一个 50px，收缩时 100px 那个会损失更多。这符合直觉，因为更大的元素有更多的空间可以缩放。

**迭代与冻结**

每次应用 min/max 约束后，有些 item 被「卡住」（触及 min 或 max）。这些 item 被冻结，下次迭代只计算未冻结的 items，避免无用计算。这个过程最多迭代 N 次（N = items 数量），最坏情况下每轮冻结一个 item。

### 具体例子

容器宽度 400px，三个 items：

- A: `flex: 1 1 100px; min-width: 80px`
- B: `flex: 2 1 100px`
- C: `flex: 1 1 100px; max-width: 120px`

**第一轮：**

```
总 hypothetical size = 300px
剩余空间 = 400 - 300 = 100px
模式 = kGrow

分配：
- A: 100 + 100×(1/4) = 125px ✓
- B: 100 + 100×(2/4) = 150px ✓
- C: 100 + 100×(1/4) = 125px → 被 max-width 限制到 120px (MaxViolation)

总违规 = -5px (负数)
冻结所有 MaxViolation items（即 C）
```

**第二轮：**

```
剩余空间 = 400 - 125 - 150 - 120 = 5px
只有 A、B 参与，总 flex-grow = 3

分配：
- A: 125 + 5×(1/3) ≈ 127px ✓
- B: 150 + 5×(2/3) ≈ 153px ✓

无违规，算法结束
```

**最终结果：** A=127px, B=153px, C=120px

## 对齐与定位

`GiveItemsFinalPositionAndSize` 处理 justify-content、align-items 等对齐属性。

### 主轴对齐（justify-content）

```cpp
// flex_layout_algorithm.cc
LayoutUnit InitialContentPositionOffset(const StyleContentAlignmentData& data,
                                       LayoutUnit free_space,
                                       unsigned number_of_items) {
  switch (data.Distribution()) {
    case kSpaceBetween:
      // 首末贴边，中间均分
      return free_space > 0 && number_of_items > 1 ? 0 : free_space;
      
    case kSpaceAround:
      // 每个 item 两侧各半份
      return free_space > 0 && number_of_items ? free_space / (2 * number_of_items) : 0;
      
    case kSpaceEvenly:
      // n+1 个间隙均分（首尾也有间隙）
      return free_space > 0 && number_of_items ? free_space / (number_of_items + 1) : 0;
      
    case kCenter:
      return free_space / 2;
  }
}
```

**核心目标：** 计算第一个 item 的起始位置。后续 items 根据对齐模式间隔排列。Chromium 使用 `LayoutUnitDiffuser` 来确保像素级精确，避免舍入误差累积。

### Auto Margins 的优先级

交叉轴上的 auto margin 优先级最高，吃掉所有剩余空间：

```cpp
// flex_layout_algorithm.cc
const LayoutUnit margin_space = cross_axis_space.ClampNegativeToZero();
if (is_margin_auto.CrossStart() && is_margin_auto.CrossEnd()) {
  // 两侧都 auto：居中
  margin.CrossStart() = margin_space / 2;
  margin.CrossEnd() = margin_space / 2;
} else if (is_margin_auto.CrossStart()) {
  margin.CrossStart() = margin_space;
} else if (is_margin_auto.CrossEnd()) {
  margin.CrossEnd() = margin_space;
}
```

交叉轴上的 auto margin 优先级最高，会吃掉所有剩余空间。如果两侧都是 auto，就平均分配实现居中效果。如果只有一侧 auto，就把所有空间分给它。

主轴的 auto margin 也遵循同样逻辑，消耗所有自由空间。这就是为什么 `margin-left: auto` 能把 item 推到右边的原因。

### Baseline 对齐

Baseline 对齐是对齐中最复杂的部分。规范定义了两组 baseline（major/minor），还要处理 writing mode、wrap-reverse 等各种情况。核心逻辑是计算行内最大 ascent，然后让每个 item 的 ascent 与之对齐。

```cpp
// flex_layout_algorithm.cc
case ItemPosition::kBaseline: {
  const bool is_major = item.baseline_group == BaselineGroup::kMajor;
  const LayoutUnit ascent = BaselineAscent(item, physical_fragment);
  const LayoutUnit max_ascent = is_major ? flex_line.major_baseline 
                                         : flex_line.minor_baseline;
  
  // 偏移 = 行内最大 ascent - 当前 item 的 ascent
  const LayoutUnit baseline_delta = max_ascent - ascent;
  
  offset = is_major ? baseline_delta : space - baseline_delta;
  break;
}
```

## 常见问题

### Q: 为什么 `flex: 1` 无法收缩到小于内容宽度？

**A:** automatic minimum size。`flex: 1` 隐含 `min-width: auto`，而非 `min-width: 0`。

```cpp
// 源码判断
auto_min_size = min(content_size_suggestion, specified_size_suggestion);
```

**解决方案：** `flex: 1; min-width: 0;`

### Q: Column flex 容器中为什么百分比高度不生效？

**A:** Column flex 的高度通常是 indefinite（由内容决定），百分比相对 indefinite 无法解析。

```cpp
// flex_layout_algorithm.cc
const bool is_initial_block_size_indefinite =
    is_column_ && !is_main_axis_inline_axis &&
    ChildAvailableSize().block_size == kIndefiniteSize &&
    is_used_flex_basis_indefinite;
```

**解决方案：** 给容器明确高度，或嵌套另一个 flex 容器。

### Q: 为什么多个 `flex-shrink: 1` 的 items 收缩不等比？

**A:** flex-shrink 是加权分配，乘以 base size：

```cpp
weighted_flex_shrink = flex_shrink × base_size
```

100px 的 item 比 50px 的 item 收缩两倍。

### Q: `flex-grow < 1` 时会怎样？

**A:** 只分配比例空间。

```cpp
if (total_flex_factor_ < 1.0) {
  free_space = initial_free_space * total_flex_factor;
}
```

`flex: 0.5 0 100px` 在 300px 容器中，剩余 200px 只分配 200 × 0.5 = 100px。

## 实现细节

### 坐标转换抽象

Chromium 用模板类 `PhysicalToFlex` 处理 row/column 的坐标差异：

```cpp
// flex_layout_algorithm.cc
template <typename Value>
class PhysicalToFlex {
  Value MainStart() const {
    return is_column_ ? logical_.BlockStart() : logical_.InlineStart();
  }
  Value CrossStart() const {
    return is_column_ ? logical_.InlineStart() : logical_.BlockStart();
  }
};
```

这个坐标变换模板很巧妙，通过条件判断选择正确的坐标轴。这样 row/column 的逻辑就能用统一的代码处理，无需大量条件分支，代码简洁高效。

### LayoutUnitDiffuser 的像素精度

这是处理像素级舍入的工具类。避免浮点舍入误差，通过逐步消耗空间并将余数分给最后一份，确保像素级精确分配：

```cpp
// geometry/layout_unit_diffuser.h
class LayoutUnitDiffuser {
  LayoutUnit Next() {
    // 返回一份空间，逐步消耗，最后一个包含余数
  }
};
```

### Gap 的处理

`gap` 属性在分行、flex-grow/shrink 计算时都需要考虑，不是简单地加在 items 之间。在计算累计尺寸、判断是否换行时都要包括 gap，这样才能准确决定分行的位置。

## 总结

Chromium 的 Flexbox 实现忠实于规范，同时做了工程优化：

1. **浮点精度：** 从后向前分配，最后一个 item 消耗余数
2. **迭代收敛：** 反复冻结/解冻，保证所有 min/max 约束满足
3. **坐标抽象：** PhysicalToFlex 统一 row/column 逻辑
4. **滚条管理：** FreezeScrollbarsScope 防止 relayout 循环
5. **缓存机制：** 避免重复计算

这些细节虽然不在规范里，但是实现稳定高效浏览器的关键。
