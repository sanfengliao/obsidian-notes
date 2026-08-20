> 源码：`core/layout/scrollable_overflow_calculator.{h,cc}`、`core/layout/physical_box_fragment.{h,cc}`、`core/layout/layout_box.cc
> `规范：css-overflow-3 §scrollable-overflow

---

# 0. scrollable overflow 是什么

scrollable overflow（也叫 layout overflow）是一个矩形区域，表示「为了让所有内容可见，滚动容器需要能滚动到的范围」。它决定滚动条大小、滚动范围、以及 `scrollWidth/scrollHeight`。

它与 ink overflow（paint overflow）不同：

- **scrollable overflow**：只包含**影响滚动**的内容。被 `overflow: hidden` 裁剪掉的、transform 移出可视区但仍需滚动看到的，都算。
- **ink overflow**：包含所有**影响绘制**的内容（如阴影、outline），即使不滚动也要留出绘制空间。

## 直觉 vs 精确

**直觉理解**：对滚动容器来说，scrollable overflow 大致就是「内容总共需要多大空间才能全部看见」——即滚动范围的尺寸。`scrollWidth` / `scrollHeight` 就是它的宽高。这个近似能帮你快速建立概念，但要理解后续计算逻辑，需要记住它**不等于「内容本身的大小」**，而是一个**带方向、有基准、有排除规则的矩形**：

1. **它是矩形（有位置），不只是尺寸**。它包含 offset，表示内容在哪个方向、多远地超出。比如 LTR 容器里内容只往右溢出，矩形的左边界贴着 padding 盒左边、右边界往外扩——这个「方向性」是 `AdjustOverflowForScrollOrigin` 算出来的，单纯的「内容大小」表达不了。
2. **基准是 padding 盒，不是 0**。即使内容没溢出，scrollable overflow 也不是 0，而是 padding 盒的大小（border + scrollbar 之内）。所以准确说法是「padding 盒子内容溢出」，不是「内容大小」。
3. **非滚动容器也有 scrollable overflow，但语义不同**。非滚动容器不滚动，它的 scrollable overflow 纯粹是「内容几何上超出自身多少」的记录，不对应任何滚动行为。这时叫它「滚动容器内容大小」就不合适——它根本不滚。
4. **它受滚动方向裁剪**。滚动容器只在自己能滚的方向扩展溢出（LTR 不左滚、垂直书写模式不上滚）。所以它不是「内容的真实外接矩形」，而是「按滚动方向裁剪后的外接矩形」。比如内容往左伸 100px，LTR 容器不左滚，这 100px 不进 scrollable overflow。
5. **它有包含/排除规则**：包含 OOF 元素、transform 移出的内容（仍需滚动看到）；排除 hanging 标点、`contain:layout` 的内部溢出（虽然伸出但不该参与滚动）。

**一句话精确表述**：对滚动容器，scrollable overflow 是「按滚动方向裁剪后的、 padding 盒的外接矩形」，尺寸约等于 `scrollWidth/scrollHeight`，但它是带方向的矩形，且排除了不该参与滚动的部分——不是单纯的内容尺寸。

本文分析 `ScrollableOverflowCalculator`——LayoutNG 中计算 scrollable overflow 的核心类。

---

# 1. 计算的两个阶段

scrollable overflow 有两次计算时机：

## 阶段一：初始计算（布局时，`PhysicalBoxFragment::Create`）

`Create` 在造片段时，如果 `builder->ShouldCalculateScrollableOverflow()`，就建一个 `ScrollableOverflowCalculator`，把所有子片段和行内 items 喂给它，得到 overflow 后存进片段的 `rare_data_`：

```cpp
// physical_box_fragment.cc，Create 内
PhysicalRect scrollable_overflow = {PhysicalOffset(), physical_size};
if (builder->ShouldCalculateScrollableOverflow()) {
  ScrollableOverflowCalculator calculator(
      To<BlockNode>(builder->node_),
      /* is_css_box */ !builder->IsFragmentainerBoxType(),
      builder->GetConstraintSpace().HasBlockFragmentation(),
      borders, scrollbar, padding, physical_size, writing_direction);

  if (FragmentItemsBuilder* items_builder = builder->ItemsBuilder())
    calculator.AddItems(builder->GetLayoutObject(), items_builder->Items(physical_size));

  for (auto& child : builder->children_) {
    const auto* box_fragment = DynamicTo<PhysicalBoxFragment>(*child.fragment);
    if (!box_fragment) continue;
    calculator.AddChild(*box_fragment,
        child.offset.ConvertToPhysical(writing_direction, physical_size, box_fragment->Size()));
  }

  if (builder->table_collapsed_borders_)
    calculator.AddTableSelfRect();

  scrollable_overflow = calculator.Result(inflow_bounds);
}
```

注意：这个循环遍历 `builder->children_` 是为了**算 overflow**，不是为了填 `children_`（填 `children_` 是构造函数的事，见 fragment builder 文档）。

## 阶段二：重算（布局后）

某些场景下，初始计算的结果会过时，需要重算。`RecalculateScrollableOverflowForFragment` 静态方法从一个**已有的片段**重建 calculator，重新累加 items + children，返回新 overflow（不写回，由调用方决定）：

```cpp
PhysicalRect ScrollableOverflowCalculator::RecalculateScrollableOverflowForFragment(
    const PhysicalBoxFragment& fragment, bool has_block_fragmentation) {
  // ...建 calculator，用 fragment 自己的 borders/scrollbar/padding/size...
  if (const FragmentItems* items = fragment.Items())
    calculator.AddItems(fragment, *items);
  for (const auto& child : fragment.PostLayoutChildren()) {
    // fragmentainer 子片段递归重算
    if (box_fragment->IsFragmentainerBox()) {
      PhysicalRect child_overflow = RecalculateScrollableOverflowForFragment(*box_fragment, ...);
      calculator.AddOverflow(child_overflow, /* child_is_fragmentainer */ true);
    } else {
      calculator.AddChild(*box_fragment, child.offset);
    }
  }
  return calculator.Result(fragment.InflowBounds());
}
```

两个调用方：

1. `PhysicalBoxFragment::MutableForOofFragmentation::UpdateOverflow`——OOF 分页后，片段内容变了，重算并 `SetScrollableOverflow` 写回。
2. `LayoutBox::UpdateLayoutOverflow`**（**`layout_box.cc`**）**——布局后比对新旧 overflow，变了就标记 `scrollable_overflow_changed` / `rebuild_fragment_tree`，触发滚动条更新。

---

# 2. `ScrollableOverflowCalculator` 结构

```cpp
class ScrollableOverflowCalculator {
  const BlockNode node_;
  const WritingDirectionMode writing_direction_;
  const bool is_scroll_container_;      // 是滚动容器
  const bool is_view_;                  // 是 LayoutView
  const bool scrolls_all_directions_;   // 四向滚动（忽略 has_left/top_overflow）
  const bool has_left_overflow_;        // 左溢出（RTL 等）
  const bool has_top_overflow_;         // 上溢出（垂直书写模式等）
  const bool has_non_visible_overflow_;
  const bool has_block_fragmentation_;

  const PhysicalBoxStrut padding_;
  const PhysicalSize size_;

  PhysicalRect padding_rect_;           // padding 盒（溢出基准）
  PhysicalRect scrollable_overflow_;    // 累积结果
};
```

## 构造函数：建立基准

```cpp
ScrollableOverflowCalculator::ScrollableOverflowCalculator(...) {
  const auto border_scrollbar = borders + scrollbar;
  padding_rect_ = {PhysicalOffset(border_scrollbar.left, border_scrollbar.top),
                   PhysicalSize((size_.width - border_scrollbar.HorizontalSum()).ClampNegativeToZero(),
                                (size_.height - border_scrollbar.VerticalSum()).ClampNegativeToZero())};
  scrollable_overflow_ = padding_rect_;   // ← 初始值 = padding 盒
}
```

**关键**：scrollable overflow 的**初始值是 padding 盒**（border + scrollbar 之内、padding 之外的区域），不是 `{0,0}`。因为即使没有溢出内容，滚动范围也至少是 padding 盒。后续 `UniteEvenIfEmpty` 把子内容的溢出并进来。

> 注释提到 fieldset 的 legend 可能比 block-start border 高，这里的 padding*rect* 对 fieldset 不完全正确。

---

# 3. 三个输入源

计算器有三个 `Add*` 方法吸收溢出来源：

## 3.1 `AddChild`——块级子片段

```cpp
void AddChild(const PhysicalBoxFragment& child_fragment, PhysicalOffset offset) {
  if (is_view_ && child_fragment.IsFixedPositioned())
    return;   // LayoutView 不吸收 fixed 子元素的溢出
  PhysicalRect child_overflow = ScrollableOverflowForPropagation(child_fragment);
  child_overflow.offset += offset;       // 平移到父坐标系
  AddOverflow(child_overflow, child_fragment.IsFragmentainerBox());
}
```

## 3.2 `AddItems`——行内内容

行内格式化上下文的内容通过 `AddItems` 吸收（见第 6 节）。

## 3.3 `AddTableSelfRect`——表格自身

```cpp
void AddTableSelfRect() {
  AddOverflow({PhysicalOffset(), size_});   // 表格自己的 border box
}
```

表格的 collapsed border 可能超出 border box，所以表格自身整块都算溢出。

---

# 4. `ScrollableOverflowForPropagation`——单个子片段的溢出提取

这是最复杂的方法，决定「一个子片段贡献多少溢出给父容器」。逻辑：

```cpp
PhysicalRect ScrollableOverflowForPropagation(const PhysicalBoxFragment& child_fragment) {
  // (1) 完全不传播的情况
  if (child_fragment.IsHiddenForPaint() ||                          // empty-cells 等
      IsA<ViewTransitionTransitionElement>(child_fragment.GetNode()))  // ::view-transition
    return {};

  // (2) 匿名片段：直接返回它的 scrollable_overflow，不做变换
  if (!child_fragment.IsCSSBox())
    return child_fragment.ScrollableOverflow();

  // (3) 起点 = 子片段自身的 border box（在子片段坐标系）
  PhysicalRect overflow = {{}, child_fragment.Size()};

  // (4) 决定是否忽略子片段的内部溢出
  bool ignore_scrollable_overflow =
      child_fragment.ShouldApplyLayoutContainment() ||   // contain:layout 不传播内部溢出
      child_fragment.IsInlineBox() ||                    // inline-box 只用自身尺寸
      (child_fragment.ShouldClipOverflowAlongBothAxis() &&
       !child_fragment.ShouldApplyOverflowClipMargin());  // 两轴都裁剪且无 clip-margin

  // (5) 若不忽略，取子的 scrollable_overflow，按裁剪轴收缩
  if (!ignore_scrollable_overflow) {
    PhysicalRect child_overflow = child_fragment.ScrollableOverflow();
    if (child_fragment.HasNonVisibleOverflow()) {
      const OverflowClipAxes overflow_clip_axes = child_fragment.GetOverflowClipAxes();
      if (child_fragment.ShouldApplyOverflowClipMargin()) {
        // overflow-clip-margin：把裁剪区扩大 margin outset
        PhysicalRect clip_rect({}, child_fragment.Size());
        clip_rect.Expand(child_fragment.OverflowClipMarginOutsets());
        child_overflow.Intersect(clip_rect);
      } else {
        // 按轴裁剪回 border box
        if (overflow_clip_axes & kOverflowClipX) { child_overflow.offset.left = 0; child_overflow.size.width = child_fragment.Size().width; }
        if (overflow_clip_axes & kOverflowClipY) { child_overflow.offset.top = 0; child_overflow.size.height = child_fragment.Size().height; }
      }
    }
    overflow.UniteEvenIfEmpty(child_overflow);
  }

  // (6) 应用 transform（子片段可能被 transform 移动/缩放）
  if (std::optional<gfx::Transform> transform = GetTransformForChildFragment(...))
    overflow = PhysicalRect::EnclosingRect(transform->MapRect(gfx::RectF(overflow)));

  // (7) 分页 + OOF：若 OOF 的 containing block 在裁剪容器内，沿裁剪轴收缩到 1px
  if (has_block_fragmentation_ && child_fragment.IsOutOfFlowPositioned() && !FragmentedOofInCbEnabled()) {
    // 遍历 containing block 链，遇裁剪轴就把 overflow 沿该轴缩到 ≤1px
  }

  return overflow;
}
```

要点：

- **不传播**：hidden-for-paint（table empty-cells）、`::view-transition`。
- **匿名片段**：直接用其 `ScrollableOverflow()`，不施加 transform/裁剪（匿名片段的几何已是父级处理过的）。
- **layout containment**：`contain:layout` 的元素不传播内部溢出，只用自身 border box。
- **overflow 裁剪**：`overflow: hidden/auto/clip` 的子元素，其内部溢出按裁剪轴收缩回 border box；`overflow-clip-margin` 则把裁剪区扩大。
- **transform**：子片段若有 transform，溢出经 transform 映射后可能扩大/移位。
- **分页 OOF**：OOF 元素的 containing block 若在裁剪链内，溢出沿裁剪轴缩到 ≤1px（保留非裁剪轴的传播）。

返回的 `overflow` 是**子片段在父坐标系下贡献的溢出矩形**（还未经 `AddOverflow` 的 scroll-origin 调整）。

---

# 5. `AddOverflow` 与 `AdjustOverflowForScrollOrigin`——滚动方向裁剪

### `AddOverflow`

```cpp
void AddOverflow(PhysicalRect child_overflow, bool child_is_fragmentainer = false) {
  if (is_scroll_container_)
    child_overflow = AdjustOverflowForScrollOrigin(child_overflow);

  // fragmentainer 即使空也可能贡献溢出（如非零 column-gap）
  if (!child_overflow.IsEmpty() || child_is_fragmentainer)
    scrollable_overflow_.UniteEvenIfEmpty(child_overflow);
}
```

只有**滚动容器**才做 scroll-origin 调整；非滚动容器直接 unite。

## `AdjustOverflowForScrollOrigin`——核心

```cpp
PhysicalRect AdjustOverflowForScrollOrigin(const PhysicalRect& overflow) {
  LayoutUnit left_offset =
      scrolls_all_directions_ || has_left_overflow_
          ? std::min(padding_rect_.Right(), overflow.offset.left)   // 左溢出：left 可往左扩
          : std::max(padding_rect_.offset.left, overflow.offset.left);  // 否则 left 钳到 padding 左边
  LayoutUnit right_offset =
      scrolls_all_directions_ || !has_left_overflow_
          ? std::max(padding_rect_.offset.left, overflow.Right())   // 右溢出：right 可往右扩
          : std::min(padding_rect_.Right(), overflow.Right());      // 否则 right 钳到 padding 右边
  // top/bottom 同理，用 has_top_overflow_
  return {PhysicalOffset(left_offset, top_offset),
          PhysicalSize(right_offset - left_offset, bottom_offset - top_offset)};
}
```

**这是 scrollable overflow 计算的灵魂**：滚动容器只在自己**会滚动的方向**上扩展溢出。

- `has_left_overflow_` / `has_top_overflow_`：由 `HasLeftOverflow()` / `HasTopOverflow()` 决定，与书写方向、`direction` 相关。例如 LTR 水平书写模式下 `has_left_overflow_ = false`（内容只会往右溢出，不往左），垂直书写模式下 `has_top_overflow_ = false`。
- `scrolls_all_directions_`：`IsOverscrollAreaParent()`，四向都可滚（如某些触屏场景），此时忽略 left/top 标志，四向都扩展。

效果：LTR 容器里，子内容往左伸出 padding 盒时，**左边界被钳回 padding*****rect*****.left**（不会产生左滚）；往右伸出时，右边界扩展。这样滚动范围只反映「真正需要滚动才能看到的方向」。

非滚动容器跳过这步，直接 unite 子内容真实溢出（因为不滚动，溢出纯粹是几何记录）。

---

# 6. `AddItems`——行内内容的溢出

行内内容（文本、inline-box、atomic inline）通过 `AddItemsInternal` 处理：

```cpp
template <typename Items>
void AddItemsInternal(const LayoutObject* layout_object, const Items& items) {
  if (IsA<LayoutTextCombine>(layout_object))   // text-combine 不产生溢出（缩进 1em）
    return;

  bool has_hanging = false;
  PhysicalRect line_rect;
  for (const auto& item : items) {
    if (item->IsHiddenForPaint()) continue;

    if (const auto* line_box = item->LineBoxFragment()) {
      has_hanging = line_box->HasHanging();
      line_rect = item->RectInContainerFragment();
      if (line_rect.IsEmpty()) continue;
      scrollable_overflow_.UniteEvenIfEmpty(line_rect);   // 行盒本身 unite
      continue;
    }
    if (item->IsText()) {
      PhysicalRect child_overflow = item->RectInContainerFragment();
      if (has_hanging)   // hanging 字符不增加溢出
        child_overflow = AdjustOverflowForHanging(line_rect, child_overflow);
      AddOverflow(child_overflow);
      continue;
    }
    if (const auto* child_box_fragment = item->BoxFragment()) {
      PhysicalRect child_overflow = ScrollableOverflowForPropagation(*child_box_fragment);
      child_overflow.offset += item->OffsetInContainerFragment();
      if (child_box_fragment->IsInlineBox() && has_hanging)
        child_overflow = AdjustOverflowForHanging(line_rect, child_overflow);
      AddOverflow(child_overflow);
      continue;
    }
  }
}
```

要点：

- **行盒**：直接 unite 行盒矩形（行盒的尺寸已包含其内容）。
- **文本**：用 item 在容器片段里的矩形。若行有 hanging（`text-indent: hanging` 或行尾悬挂标点），用 `AdjustOverflowForHanging` 把文本溢出钳回行盒边界——hanging 字符不增加滚动范围。
- **inline box / atomic inline**：走 `ScrollableOverflowForPropagation`（复用块级逻辑），inline-box 还要考虑 hanging。
- `LayoutTextCombine`：跳过（竖排中横排文本缩进 1em，不溢出）。

## `AdjustOverflowForHanging`

```cpp
PhysicalRect AdjustOverflowForHanging(const PhysicalRect& line_rect, PhysicalRect overflow) {
  if (writing_direction_.IsHorizontal()) {
    if (overflow.offset.left < line_rect.offset.left) overflow.offset.left = line_rect.offset.left;
    if (overflow.Right() > line_rect.Right()) overflow.ShiftRightEdgeTo(line_rect.Right());
  } else {
    // 垂直书写模式：钳 top/bottom
  }
  return overflow;
}
```

把溢出钳到行盒边界。hanging 的本意是「这个字符挂在外面但不应该触发滚动」。

---

# 7. `Result`——inflow\_bounds 最终调整

所有子内容吸收完后，`Result(inflow_bounds)` 给出最终矩形：

```cpp
const PhysicalRect Result(const std::optional<PhysicalRect> inflow_bounds) {
  if (!inflow_bounds || !is_scroll_container_)
    return scrollable_overflow_;

  // 用 inflow_bounds 扩 padding 计算「inflow 溢出」
  PhysicalOffset start_offset = inflow_bounds->MinXMinYCorner() - PhysicalOffset(padding_.left, padding_.top);
  PhysicalOffset end_offset = inflow_bounds->MaxXMaxYCorner() + PhysicalOffset(padding_.right, padding_.bottom);
  PhysicalRect inflow_overflow = {start_offset, PhysicalSize(end_offset.left - start_offset.left, end_offset.top - start_offset.top)};
  inflow_overflow = AdjustOverflowForScrollOrigin(inflow_overflow);

  scrollable_overflow_.UniteEvenIfEmpty(inflow_overflow);
  return scrollable_overflow_;
}
```

`inflow_bounds` 是 builder 在 `AddChild` 时累积的「所有 inflow 子片段的外接矩形」（见 fragment builder 文档的 `inflow_bounds_` 字段）。滚动容器用它来保证滚动范围至少覆盖 inflow 内容 + padding（OOF 不算 inflow，不参与这里的扩展）。

只有滚动容器且 inflow*bounds 存在时才做这步。\`inflow*overflow` 同样要过 `AdjustOverflowForScrollOrigin\`（受滚动方向限制）。

---

# 8. 存储与访问

## `PhysicalBoxFragment` 上的存储

scrollable overflow 和 inflow*bounds 都存在 \`rare*data\_`（`PhysicalFragmentRareData\`）里，按需分配：

```cpp
const PhysicalRect ScrollableOverflow() const {
  if (const auto* field = GetRareField(FieldId::kScrollableOverflow))
    return field->scrollable_overflow;
  return {{}, Size()};   // 无存储时默认 = 自身 border box
}
const std::optional<PhysicalRect> InflowBounds() const { ... }
```

`Create` 时只有当 `scrollable_overflow != PhysicalRect({}, physical_size)`（与自身 border box 不同）才存进 `rare_data_`：

```cpp
bool has_scrollable_overflow = scrollable_overflow != PhysicalRect({}, physical_size);
// 传给构造函数，非默认时存 rare_data_
```

`HasScrollableOverflow()` 判断是否存了，`ScrollableOverflow()` 取值（无则返回 `{{}, Size()}`）。

---

# 9. 完整数据流

```typescript
Create(builder)
  │
  ├─ calculator 构造：scrollable_overflow_ = padding_rect_
  │
  ├─ AddItems(...)              ← 行内内容（文本/inline-box/行盒）
  │     ├─ 行盒 unite
  │     ├─ 文本 RectInContainerFragment（hanging 钳制）
  │     └─ inline-box ScrollableOverflowForPropagation（hanging 钳制）
  │
  ├─ for each child in builder->children_:
  │     AddChild(box_fragment, offset)
  │       ├─ ScrollableOverflowForPropagation(child)   ← 提取子溢出（containment/裁剪/transform）
  │       └─ AddOverflow(child_overflow, is_fragmentainer)
  │             ├─ if 滚动容器: AdjustOverflowForScrollOrigin  ← 按滚动方向钳制
  │             └─ scrollable_overflow_.UniteEvenIfEmpty
  │
  ├─ AddTableSelfRect()（表格）
  │
  └─ Result(inflow_bounds)
        ├─ if 滚动容器 && inflow_bounds: 算 inflow_overflow + AdjustOrigin + unite
        └─ return scrollable_overflow_
              ↓
       存进 PhysicalBoxFragment::rare_data_（仅当 ≠ border box）
```

---

# 10. 关键设计要点

1. **基准是 padding 盒**：初始 `scrollable_overflow_ = padding_rect_`（border+scrollbar 之内），子内容溢出在其上 unite。这保证滚动范围至少覆盖 padding 区域。
2. **滚动方向裁剪（**`AdjustOverflowForScrollOrigin`**）是灵魂**：滚动容器只在自己能滚动的方向扩展溢出。LTR 水平模式不左滚、垂直模式不上滚，由 `has_left_overflow_` / `has_top_overflow_` / `scrolls_all_directions_` 控制。非滚动容器跳过这步。
3. **per-child 提取（**`ScrollableOverflowForPropagation`**）处理 5 类情况**：hidden-for-paint / view-transition 不传播；匿名直接用；containment/inline-box/双轴裁剪忽略内部；`overflow` 裁剪按轴收缩（clip-margin 扩大）；transform 映射；分页 OOF 裁剪链。
4. **块级与行内复用同一套**：行内 box 片段（inline-box / atomic inline）也走 `ScrollableOverflowForPropagation`，只是额外处理 hanging。
5. **hanging 不增加滚动范围**：`text-indent: hanging` / 行尾悬挂标点用 `AdjustOverflowForHanging` 钳回行盒边界。
6. **inflow\_bounds 兜底**：滚动容器用 builder 累积的 inflow 子内容外接矩形 + padding 保证滚动范围覆盖正常流内容（OOF 不算）。
7. **两阶段计算**：初始在 `Create` 里算一次存 `rare_data_`；OOF 分页或布局后变化时由 `RecalculateScrollableOverflowForFragment` 重算，调用方比对后决定是否更新滚动条 / 重建片段树。
8. **懒存储**：overflow 与 border/padding/scrollbar 一样懒分配到 `rare_data_`，等于自身 border box 时不占空间。
9. **fragmentainer 特殊**：空 fragmentainer 也可能贡献溢出（非零 column-gap）；重算时 fragmentainer 子片段递归重算（因为它不直接关联 LayoutObject，初始 overflow 不会被正常更新）。
10. **LayoutView 不吸收 fixed 子元素溢出**：`is_view_ && child.IsFixedPositioned()` 跳过——fixed 元素相对 viewport 定位，不该撑大 view 的滚动范围。

---

# 附：相关文件索引

| 文件 | 内容 |
| --- | --- |
| `core/layout/scrollable_overflow_calculator.{h,cc}` | `ScrollableOverflowCalculator` 全部逻辑 |
| `core/layout/physical_box_fragment.{h,cc}` | `Create` 调用计算器、`ScrollableOverflow()`/`InflowBounds()` 访问器、`MutableForOofFragmentation::UpdateOverflow` |
| `core/layout/layout_box.cc` | `RecalculateScrollableOverflowForFragment` 用于布局后比对溢出变化 |
| `core/layout/fragment_builder.{h,cc}` | builder 累积 `inflow_bounds_`（`AddChild` 时） |
| `core/layout/transform_utils.{h,cc}` | `GetTransformForChildFragment`（子片段 transform 提取） |
