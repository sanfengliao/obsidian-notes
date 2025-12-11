`CalculateMargins` 是 Chromium 渲染引擎 Blink 中块级布局算法(`BlockLayoutAlgorithm`)的一个关键函数，它计算块级子元素的边距，并处理特殊的对齐和自动边距情况。

## 函数签名
```cpp
BoxStrut BlockLayoutAlgorithm::CalculateMargins(
    LayoutInputNode child,
    bool is_new_fc,
    LayoutUnit* additional_line_offset) 
```

## 参数说明
- `child`: 要计算边距的布局节点（通常是一个块级子元素）
- `is_new_fc`: 布尔值，表示子元素是否创建了一个新的格式化上下文(formatting context)
- `additional_line_offset`: 输出参数，用于返回计算出的额外行偏移量

## 返回值
- `BoxStrut`: 表示元素的四个方向的边距（上、右、下、左）的结构体

## 函数详解

### 1. 处理特殊情况

```cpp
if (child.IsInline())
  return {};
```
如果子元素是内联元素，不需要计算边距，直接返回空的边距结构。

### 2. 计算基本边距

```cpp
const ComputedStyle& child_style = child.Style();
BoxStrut margins =
    ComputeMarginsFor(child_style, child_percentage_size_.inline_size,
                      GetConstraintSpace().GetWritingDirection());
```
调用 `ComputeMarginsFor` 计算子元素四个方向的基本边距，传入：
- 子元素的样式 (`child_style`)
- 用于百分比计算的内联尺寸 (`child_percentage_size_.inline_size`)
- 当前写作模式 (`GetConstraintSpace().GetWritingDirection()`)

### 3. 对于新格式化上下文的特殊处理

```cpp
if (is_new_fc) {
  return margins;
}
```
如果子元素创建新的格式化上下文(比如 `float`、`position: absolute` 等)，就直接返回计算出的边距，不进行后续的对齐或自动边距处理。

### 4. 惰性计算子元素内联尺寸

```cpp
std::optional<LayoutUnit> child_inline_size;
auto ChildInlineSize = [&]() -> LayoutUnit {
  if (!child_inline_size) {
    // 创建约束空间构建器
    ConstraintSpaceBuilder builder(GetConstraintSpace(),
                                   child_style.GetWritingDirection(),
                                   /* is_new_fc */ false);
    builder.SetAvailableSize(ChildAvailableSize());
    builder.SetPercentageResolutionSize(child_percentage_size_);

    // 根据子元素的 justify-self 属性设置内联尺寸自动行为
    const ItemPosition justify_self =
        child_style
            .ResolvedJustifySelf(
                {ItemPosition::kNormal, OverflowAlignment::kDefault},
                &Style())
            .GetPosition();

    if (child.IsAnonymousBlockFlow()) {
      builder.SetInlineAutoBehavior(AutoSizeBehavior::kStretchImplicit);
    } else if (justify_self == ItemPosition::kStretch) {
      builder.SetInlineAutoBehavior(AutoSizeBehavior::kStretchExplicit);
    } else if (justify_self != ItemPosition::kNormal) {
      builder.SetInlineAutoBehavior(AutoSizeBehavior::kFitContent);
    } else {
      builder.SetInlineAutoBehavior(AutoSizeBehavior::kStretchImplicit);
    }
    ConstraintSpace space = builder.ToConstraintSpace();

    // 计算子元素的边框和内边距
    const auto block_child = To<BlockNode>(child);
    BoxStrut child_border_padding = ComputeBorders(space, block_child) +
                                    ComputePadding(space, child_style);
    // 计算子元素的内联尺寸
    child_inline_size = ComputeInlineSizeForFragment(space, block_child,
                                                     child_border_padding);
  }
  return *child_inline_size;
};
```

这是一个惰性计算函数，只有在实际需要时才会计算子元素的内联尺寸。它为子元素创建约束空间，并根据子元素的 `justify-self` 属性确定内联尺寸自动行为。

### 5. 处理自动边距和文本对齐

```cpp
const auto& style = Style();
const bool is_rtl = IsRtl(style.Direction());
const LayoutUnit available_space = ChildAvailableSize().inline_size;

LayoutUnit text_align_offset;
if (child_style.MarginInlineStartUsing(style).IsAuto() ||
    child_style.MarginInlineEndUsing(style).IsAuto()) {
  // 解析自动边距
  ResolveInlineAutoMargins(child_style, style, available_space,
                           ChildInlineSize(), &margins);
} else {
  // 处理 -webkit- 文本对齐和 justify-self 属性
  text_align_offset = WebkitTextAlignAndJustifySelfOffset(
      child_style, style, available_space, margins, ChildInlineSize);
}
```

这部分代码处理两种情况：
1. 如果子元素有自动边距（`margin-left: auto` 或 `margin-right: auto`），调用 `ResolveInlineAutoMargins` 解析自动边距
2. 否则，调用 `WebkitTextAlignAndJustifySelfOffset` 处理文本对齐和 `justify-self` 属性

### 6. 计算额外行偏移量

```cpp
if (is_rtl) {
  *additional_line_offset = ChildAvailableSize().inline_size -
                            text_align_offset - ChildInlineSize() -
                            margins.InlineSum();
} else {
  *additional_line_offset = text_align_offset;
}
```

根据方向性（LTR 或 RTL）计算额外的行偏移量：
- 对于 RTL 布局，从可用空间减去文本对齐偏移量、子元素内联尺寸和边距总和
- 对于 LTR 布局，直接使用文本对齐偏移量

### 7. 返回计算出的边距

```cpp
return margins;
```

## 函数工作流程

1. 对于内联元素，直接返回空边距
2. 计算基本边距
3. 如果是新格式化上下文，直接返回基本边距
4. 定义惰性计算子元素内联尺寸的函数
5. 处理两种对齐情况：
   - 自动边距
   - 文本对齐和 justify-self
6. 根据布局方向计算额外行偏移量
7. 返回最终的边距结构

## 关键技术点

1. **自动边距处理**：当元素有 `margin: auto` 时，`ResolveInlineAutoMargins` 使用可用空间和元素尺寸来计算真实的边距值，实现元素的水平居中等效果
2. **文本对齐处理**：`WebkitTextAlignAndJustifySelfOffset` 处理 `-webkit-` 文本对齐属性和 `justify-self` 属性，计算额外的偏移量
3. **方向感知**：考虑了 LTR 和 RTL 布局的区别，计算不同的行偏移量
4. **惰性计算**：通过闭包(lambda)实现子元素内联尺寸的惰性计算，避免不必要的计算
5. **动态约束空间**：为每个子元素创建专门的约束空间，以正确处理盒模型和布局环境
这个函数是实现 CSS 布局中块级元素边距和对齐计算的核心部分，特别是对于实现自动边距、水平对齐等功能至关重要。