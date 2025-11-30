
# isRooted
判断是否在`RootLayer` 下面，也就是判断他是否在`LayoutView` 下面
```cpp
bool LayoutObject::IsRooted() const {
  NOT_DESTROYED();
  const LayoutObject* object = this;
  while (object->Parent() && !object->HasLayer())
    object = object->Parent();
  if (object->HasLayer())
    return To<LayoutBoxModelObject>(object)->Layer()->Root()->IsRootLayer();
  return false;
}
```
# ContainingBlock 函数详解

`ContainingBlock` 是 Chromium 的 Blink 渲染引擎中一个非常重要的布局函数，它用于确定一个布局对象(LayoutObject)的"包含块"(containing block)。该函数返回一个"**非匿名**的 LayoutBlock"，它是当前元素的某种包含元素。

## 功能概述

在 CSS 布局模型中，"包含块"是一个核心概念，它是元素用于计算以下内容的参考框架：

1. 百分比值的计算基准（如宽度、高度、margin、padding等）
2. 定位元素的定位参考点
3. 限制元素溢出的边界

## 实现细节

`ContainingBlock` 函数根据不同的布局情况返回不同的包含块：

```cpp
LayoutBlock* LayoutObject::ContainingBlock(AncestorSkipInfo* skip_info) const {
  NOT_DESTROYED();
  if (!IsTextOrSVGChild()) {
    if (style_->GetPosition() == EPosition::kFixed)
      return ContainingBlockForFixedPosition(skip_info);
    if (style_->GetPosition() == EPosition::kAbsolute)
      return ContainingBlockForAbsolutePosition(skip_info);
  }
  LayoutObject* object;
  if (IsColumnSpanAll()) {
    object = SpannerPlaceholder()->ContainingBlock();
  } else {
    object = Parent();
    // 处理特殊情况...
    while (object && ((object->IsInline() && !object->IsAtomicInlineLevel()) ||
                      !object->IsLayoutBlock())) {
      if (skip_info)
        skip_info->Update(*object);
      object = object->Parent();
    }
  }
  return DynamicTo<LayoutBlock>(object);
}
```

## 包含块的确定规则

函数根据不同情况确定包含块：
1. **固定定位元素（position: fixed）**：
   - 调用 `ContainingBlockForFixedPosition`，通常是视口或具有转换属性的祖先
1. **绝对定位元素（position: absolute）**：
   - 调用 `ContainingBlockForAbsolutePosition`，是最近的非static定位的祖先
1. **跨列元素（column-span: all）**：
   - 使用 `SpannerPlaceholder` 的包含块
1. **普通元素**：
   - 向上查找父链，直到找到第一个块级元素（即，跳过内联元素，除非是原子内联级）

## 与 Container 函数的区别

`ContainingBlock` 与 `Container` 函数略有不同：

- `Container` 更关注于确定DOM和布局树的容器关系，通常返回直接的父对象
- `ContainingBlock` 严格遵循CSS规范中的包含块概念，特别是在处理定位元素时


## 特殊情况与CSS差异
注释特别强调，这个函数**不总是返回CSS规范中定义的标准"containing block"**：

1. **相对定位的内联元素情况**：
   - 按CSS规范，相对定位的内联元素可以作为包含块
   - 但此函数会跳过这些内联元素，返回它们所在的非匿名LayoutBlock
   - 这会导致跳过LayoutInline元素，也会跳过内联的LayoutNGTable或LayoutBlockFlow

2. **匿名包含块情况**：
   - 如果CSS包含块是匿名的，函数会继续向上查找，直到找到非匿名的LayoutBlock
   - 在这些特殊情况下，返回的LayoutBlock可能与原始元素没有直接的逻辑关系

## 实际用途
函数的设计目的是为布局过程服务：
- LayoutBlock负责处理定位元素(positioned elements)的布局
- 这个函数确保定位元素能够被正确地插入到适当的LayoutBlock中进行布局处理

## 实际应用场景

在渲染引擎中，`ContainingBlock` 被用于：

1. 计算元素的大小（特别是百分比值）
2. 定位绝对定位和固定定位元素
3. 处理布局失效（invalidation）和重新布局
4. 确定溢出裁剪的边界

这个函数是CSS布局系统的基础部分，确保元素能够根据CSS规范正确地确定自己的大小和位置。


# Container 函数解析

`Container()` 是 Blink 渲染引擎中 `LayoutObject` 类的一个重要方法，用于按照 CSS 规范准确地确定一个布局对象的"包含块"(containing block)。这个函数完全遵循 CSS 规范中对包含块的定义，即使返回的不是一个 `LayoutBlock`。

## 代码实现与逻辑

```cpp
LayoutObject* LayoutObject::Container(AncestorSkipInfo* skip_info) const {
  NOT_DESTROYED();

  if (IsTextOrSVGChild())
    return Parent();

  EPosition pos = style_->GetPosition();
  if (pos == EPosition::kFixed)
    return ContainerForFixedPosition(skip_info);

  if (pos == EPosition::kAbsolute) {
    return ContainerForAbsolutePosition(skip_info);
  }

  if (IsColumnSpanAll()) {
    LayoutObject* multicol_container = SpannerPlaceholder()->Container();
    if (skip_info) {
      // 处理跳跃直接从spanner到多列容器的情况
      for (LayoutObject* walker = Parent();
           walker && walker != multicol_container; walker = walker->Parent())
        skip_info->Update(*walker);
    }
    return multicol_container;
  }

  return Parent();
}
```

函数逻辑：

1. **文本或SVG子元素**：直接返回父节点
2. **固定定位元素(position: fixed)**：通过特殊处理确定其容器
3. **绝对定位元素(position: absolute)**：寻找合适的非静态定位祖先作为容器
4. **跨列元素(column-span: all)**：使用其占位符的容器，并记录中间跳过的元素
5. **普通流中的元素**：直接返回父节点

## 与 ContainingBlock 的区别

`Container()` 与 `ContainingBlock()` 有关键区别：

1. **准确的 CSS 实现**：`Container()` 完全遵循 CSS 规范中的包含块概念
2. **保留内联定位祖先**：对于绝对定位元素，会返回相对定位的内联元素作为容器，而不跳过它们
3. **返回类型**：可以返回任何 `LayoutObject`，不限于 `LayoutBlock`
4. **普通流元素（Normal Flow Elements）**：
   - `Container()`: 简单地返回父元素
   - `ContainingBlock()`: 寻找最近的非匿名块级祖先
5. **绝对定位元素（Absolute Positioned Elements）**：
   - `Container()`: 会返回相对定位的内联元素（如果它是最近的定位祖先）
   - `ContainingBlock()`: 会跳过相对定位的内联元素，而是找到包含它的块级元素

这导致CSS的"computePositionedLogicalWidth"和"computePositionedLogicalHeight"计算必须使用`Container()`而不是`ContainingBlock()`，否则当包含块是相对定位的内联元素时会计算错误。
## 实际应用

这个方法在以下场景中使用：

1. **布局无效化**：通过 `markContainerChainForLayout` 等函数触发容器链的重新布局
2. **尺寸计算**：如 `computePositionedLogicalWidth` 和 `computePositionedLogicalHeight`
3. **可视区域变更**：处理元素溢出和滚动时的逻辑
4. **事件冒泡和捕获**：确定正确的事件传播路径
5. **调整绝对定位元素尺寸**：当需要精确按照CSS规范确定绝对定位元素的包含块时

## 总结

`Container()` 是一个严格遵循 CSS 规范的函数，确保元素能够正确地识别其真正的包含块，这对于精确的布局计算和渲染至关重要。它与更面向实现的 `ContainingBlock()` 相比，更准确地反映了 CSS 布局模型中的包含块概念。

