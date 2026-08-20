# 目标：绘制一个红色 div

本文追踪一个简单元素的完整绘制流程：

```html
<div style="width: 100px; height: 100px; background-color: red"></div>
```

这个 div 没有子元素、没有边框、没有阴影，只有一个纯红色背景。我们将看到它如何从 DOM 节点变成屏幕上的像素。

## 完整调用链预览

```
PaintLayerPainter::PaintWithPhase
  ↓ 遍历 Fragments (我们只有 1 个)
  ↓ fragment_idx = 0, 不需要 ScopedDisplayItemFragment
  │
  ├→ PaintFragmentWithPhase
  │    ↓ CullRect 检查（可见）
  │    ↓ ScopedPaintChunkProperties（设置 Transform/Clip/Effect）
  │    │
  │    ├→ BoxFragmentPainter::Paint
  │    │    ↓ 不是原子绘制
  │    │    │
  │    │    ├→ PaintInternal
  │    │    │    ↓ phase = kBlockBackground
  │    │    │    ↓ 转换为 kSelfBlockBackgroundOnly
  │    │    │    ↓ 没有滚动，只调用一次 PaintObject
  │    │    │    │
  │    │    │    ├→ PaintObject
  │    │    │    │    ↓ 调用 PaintBoxDecorationBackground
  │    │    │    │    ↓ 然后直接返回（SelfBlockBackgroundOnly）
  │    │    │    │    │
  │    │    │    │    ├→ PaintBoxDecorationBackground
  │    │    │    │    │    ↓ 计算 paint_rect = {0, 0, 100, 100}
  │    │    │    │    │    ↓ BoxDecorationData: 只需要画背景
  │    │    │    │    │    ↓ DrawingRecorder 开始录制
  │    │    │    │    │    │
  │    │    │    │    │    ├→ PaintBoxDecorationBackgroundWithRectImpl
  │    │    │    │    │    │    ↓ 跳过阴影、裁剪
  │    │    │    │    │    │    ↓ 调用 PaintBackground
  │    │    │    │    │    │    │
  │    │    │    │    │    │    ├→ PaintBackground
  │    │    │    │    │    │    │    ↓ PaintFillLayers
  │    │    │    │    │    │    │    │
  │    │    │    │    │    │    │    ├→ PaintFillLayer
  │    │    │    │    │    │    │    │    ↓ 纯色背景，走快速路径
  │    │    │    │    │    │    │    │    ↓ GraphicsContext::DrawRect
  │    │    │    │    │    │    │    │    │
  │    │    │    │    │    │    │    │    └→ 生成 DrawRectOp
  │    │    │    │    │    │    │    │       {rect={0,0,100,100}, color=red}
  │    │    │    │    │    │    │    │       追加到 PaintOpBuffer
  │    │    │    │    │    │
  │    │    │    │    │    └→ DrawingRecorder 析构
  │    │    │    │    │       封装 PaintRecord (包含 DrawRectOp)
  │    │    │    │    │       创建 DrawingDisplayItem
  │    │    │    │    │       追加到 DisplayItemList
  │    │    │    │    │
  │    │    │    │    └→ 返回 (phase 是 SelfBlockBackgroundOnly)
  │    │    │    │
  │    │    │    └→ 不继续绘制子元素
  │    │    │
  │    │    └→ 完成
  │    │
  │    └→ ScopedPaintChunkProperties 析构（恢复属性）
  │
  └→ Fragment 循环结束（只循环 1 次）

最终结果：
DisplayItemList 中有 1 个 DrawingDisplayItem
  ├─ client: div 的 DisplayItemClient
  ├─ type: kBoxDecorationBackground
  ├─ visual_rect: {0, 0, 100, 100}
  └─ record: PaintRecord 包含 1 个 DrawRectOp
```

---

# PaintLayerPainter::PaintWithPhase - Fragment 协调器

当渲染引擎决定要绘制这个 div 时，会调用它所属 PaintLayer 的 `PaintWithPhase` 方法。

## Fragment 的概念

在讲具体流程前，先理解 Fragment。对于复杂布局（如多列、分页），一个元素的内容可能被拆成多个片段（Fragment）。比如跨列的文本会被拆成多个 Fragment，每个列一个。

**但对于我们的红色 div**，它是个简单的块级元素，没有分页或多列，只会有 **一个 Fragment**。

## 针对红色 div 的执行流程

```cpp
void PaintLayerPainter::PaintWithPhase(PaintPhase phase,
                                       GraphicsContext& context,
                                       PaintFlags paint_flags) {
  const auto* layout_box_with_fragments =
      paint_layer_.GetLayoutBoxWithBlockFragments();

  // 遍历所有 Fragment
  for (const FragmentData& fragment :
       FragmentDataIterator(paint_layer_.GetLayoutObject())) {
    const PhysicalBoxFragment* physical_fragment =
        layout_box_with_fragments->GetPhysicalFragment(fragment_idx);

    // 只有第二个及以后的片段才需要标记 fragment_idx
    std::optional<ScopedDisplayItemFragment> scoped_display_item_fragment;
    if (fragment_idx)
      scoped_display_item_fragment.emplace(context, fragment_idx);

    // 委托给 PaintFragmentWithPhase 绘制单个片段
    PaintFragmentWithPhase(phase, fragment, fragment_idx, physical_fragment,
                           context, paint_flags);

    ++fragment_idx;
  }
}
```

**对于我们的红色 div**：

1. 它使用 NG 布局引擎，`layout_box_with_fragments` 不为空
2. `FragmentDataIterator` 只会返回 **一个** FragmentData（因为没有分页/多列）
3. 循环只执行 **一次**
4. `fragment_idx = 0`，不需要创建 `ScopedDisplayItemFragment`
5. 直接调用 `PaintFragmentWithPhase` 处理这唯一的片段

**Fragment ID 的作用**：DisplayItem 需要唯一标识符来做缓存。对于有多个 Fragment 的元素（如跨列内容），每个片段生成的 DisplayItem 需要不同的 ID，否则缓存会冲突。第一个片段默认 ID 为 0，无需额外标记。

## 示例对比：单片段 vs 多片段

**我们的红色 div（单片段）**：

```
循环第 1 次: fragment_idx=0, 不创建 ScopedDisplayItemFragment
→ 生成 DisplayItem: {client=div, type=kBoxDecorationBackground, fragment=0}
```

**假设是多列内容（三片段）**：

```
循环第 1 次: fragment_idx=0, 不创建 ScopedDisplayItemFragment
→ 生成 DisplayItem: {client=div, type=kDrawRect, fragment=0}

循环第 2 次: fragment_idx=1, 创建 ScopedDisplayItemFragment(context, 1)
→ 生成 DisplayItem: {client=div, type=kDrawRect, fragment=1}

循环第 3 次: fragment_idx=2, 创建 ScopedDisplayItemFragment(context, 2)
→ 生成 DisplayItem: {client=div, type=kDrawRect, fragment=2}
```

## 与 Paint 架构的关联

回到之前的类图，现在能看清楚 `PaintWithPhase` 在整个流程中的位置了：

```
PaintLayerPainter::Paint (主入口)
    ↓
PaintWithPhase (Fragment 协调)
    ↓ 遍历每个 fragment
PaintFragmentWithPhase (单片段绘制)
    ↓ 设置 Paint Chunk 属性
ScopedPaintChunkProperties
    ↓ 影响
PaintController (维护 paint chunks 和缓存)
    ↓ 追加
DisplayItemList (最终的绘制指令序列)
```

它是承上启下的中间层：往上对接 PaintLayer 的总体控制，往下协调各个 Fragment 的具体绘制，同时确保缓存系统正常工作。

---

# PaintFragmentWithPhase：单个片段的绘制准备

`PaintWithPhase` 把红色 div 的唯一片段交给 `PaintFragmentWithPhase` 处理。这个函数做两件核心的事：**可见性检查** 和 **设置绘制属性**。

## 第一步：CullRect 可见性检查

```cpp
void PaintLayerPainter::PaintFragmentWithPhase(...) {
  CullRect cull_rect = fragment_data.GetCullRect();
  if (cull_rect.Rect().IsEmpty())
    return;  // 不可见，直接跳过
}
```

**对于我们的红色 div**：

- 假设它在视口内，CullRect 是一个包含 div 区域的矩形（例如 `{0, 0, 100, 100}`）
- 不为空，继续执行

CullRect **是在 Paint 阶段开头计算的**，而非 PrePaint 阶段。具体流程如下：

在 `LocalFrameView::PaintTree()` 被调用时（这是 Paint 阶段的入口），第一件事就是运行：

```cpp
// LocalFrameView::PaintTree()
CullRectUpdater(*layout_view->Layer()).Update();

// 之后才推进生命周期到 kInPaint
frame_view.Lifecycle().AdvanceTo(DocumentLifecycle::kInPaint);
```

`CullRectUpdater::Update()` 会遍历整个 PaintLayer 树，利用属性树（transform、clip、scroll 节点）计算每个 layer 各 fragment 的裁剪矩形，并将结果写入：

- `FragmentData::cull_rect_` —— layer 自身的裁剪矩形
- `FragmentData::contents_cull_rect_` —— layer 内容区域的裁剪矩形

`PaintFragmentWithPhase` 中的 `fragment_data.GetCullRect()` 只是**读取这个刚在 Paint 阶段开头写好的缓存值**，然后构造 `PaintInfo` 并向下传递：

```cpp
CullRect cull_rect = fragment_data.GetCullRect();  // 读缓存
// ...
PaintInfo paint_info(context, cull_rect, phase, ...);  // 传入 PaintInfo
```

对于可滚动容器，CullRect 还会在视口基础上向外**扩展**（expansion ratio），以覆盖即将滚动进入视口的内容，减少滚动时的重绘频率。如果 div 在视口外且超出扩展范围（比如页面滚动到很远处），CullRect 为空，直接跳过绘制，节省性能。

## 第二步：设置 Paint Chunk 属性

```cpp
auto chunk_properties = fragment_data.LocalBorderBoxProperties();

ScopedPaintChunkProperties fragment_paint_chunk_properties(
    context.GetPaintController(), chunk_properties, paint_layer_,
    DisplayItem::PaintPhaseToDrawingType(phase));
```

这是关键一步——设置当前绘制上下文的 **Transform/Clip/Effect 属性**。

**对于我们的红色 div**：

- `LocalBorderBoxProperties()` 返回这个 div 的属性树节点：
- **Transform**: 如果有 `transform: translate(...)` 等，这里包含；我们的例子没有，是单位矩阵
- **Clip**: 如果有 `overflow: hidden` 等，这里包含；我们的例子没有，无裁剪
- **Effect**: 如果有 `opacity` 等，这里包含；我们的例子没有，是不透明

`ScopedPaintChunkProperties` 是 RAII 对象：

- **构造时**：在 PaintController 中开启一个新的 Paint Chunk，记录 `chunk_properties`
- **析构时**：自动恢复之前的属性

这样设计的好处是：这个作用域内生成的所有 DisplayItem（比如背景、边框、子元素）都会关联到同一个 Paint Chunk，共享相同的属性。合成线程后续可以批量应用 transform/clip/effect，而不用每个 DisplayItem 单独存一份。

**实际效果**：对于我们的例子，这会创建一个新的 Paint Chunk，标记为 “这是 div 的背景绘制阶段，没有特殊的 transform/clip/effect”。

## 第三步：委托给 BoxFragmentPainter

```cpp
if (physical_fragment) {
  BoxFragmentPainter(*physical_fragment).Paint(paint_info);
}
```

**对于我们的红色 div**：

- `physical_fragment` 存在（NG 布局路径）
- 创建 `BoxFragmentPainter`，传入红色 div 的 PhysicalBoxFragment
- 调用它的 `Paint` 方法，进负责实际的绘制工作，它有典型的三层结构：

```
Paint()         - 入口，判断绘制模式
   ↓
PaintInternal() - 阶段分发器，决定画什么
   ↓
PaintObject()   - 执行实际绘制
```

## 第一层：Paint() - 判断是否原子绘制

```cpp
void BoxFragmentPainter::Paint(const PaintInfo& paint_info) {
  if (GetPhysicalFragment().IsPaintedAtomically() &&
      !box_fragment_.HasSelfPaintingLayer() &&
      paint_info.phase != PaintPhase::kOverlayOverflowControls) {
    PaintAllPhasesAtomically(paint_info);  // 原子绘制路径
  } else {
    PaintInternal(paint_info);  // 正常分阶段绘制
  }
}
```

**对于我们的红色 div**：

- `IsPaintedAtomically()` 返回 false（不是 `<img>` 这样的 replaced element）
- 走 `PaintInternal` 路径

**什么是原子绘制？** 像 `<img>`、`<video>` 这样的元素渲染很简单，不需要分阶段，可以一次性画完所有内容（背景→前景→边框→outline）。这是性能优化，避免多次函数调用按顺序执行:

1. kSelfBlockBackgroundOnly
2. kDescendantBlockBackgroundsOnly
3. kForeground
4. kSelfOutlineOnly

相当于把正常的多次 Paint 调用合并成一次。

大部分元素走正常的 `PaintInternal` 路径。

## 第二层：PaintInternal() - 阶段路由器

这个函数有 150+ 行,核心是根据 `paint_info.phase` 分发到不同逻辑。

这个函数根据 `paint_info.phase` 决定画什么内容。我们的红色 div 会经历 `PaintPhase::kBlockBackground` 阶段。

```cpp
void BoxFragmentPainter::PaintInternal(const PaintInfo& paint_info) {
  STACK_UNINITIALIZED ScopedPaintState paint_state(box_fragment_, paint_info);
  if (!ShouldPaint(paint_state))
    return;

  PaintInfo& info = paint_state.MutablePaintInfo();
  const PhysicalOffset paint_offset = paint_state.PaintOffset();
  const PaintPhase original_phase = info.phase;  // kBlockBackground

  if (ShouldPaintSelfBlockBackground(original_phase)) {
    info.phase = PaintPhase::kSelfBlockBackgroundOnly;

    const LayoutBox& box = To<LayoutBox>(*box_fragment_.GetLayoutObject());
    auto paint_location = box.GetBackgroundPaintLocation();
    bool has_overflow = box.ScrollsOverflow();

    // 第一次调用：绘制 border box 背景
    info.SetSkipsGapDecorations(has_overflow);
    PaintObject(info, paint_offset);
    info.SetSkipsGapDecorations(false);

    // 如果有滚动，还需要第二次调用绘制滚动内容背景
    if (box.ScrollsOverflow() ||
        (paint_location & kBackgroundPaintInContentsSpace)) {
      PaintOverflowControls(info, paint_offset);
      info.SetIsPaintingBackgroundInContentsSpace(true);
      PaintObject(info, paint_offset);
      info.SetIsPaintingBackgroundInContentsSpace(false);
    }
  }

  // 继续绘制子元素（如果有）
  if (original_phase != PaintPhase::kSelfBlockBackgroundOnly) {
    ScopedBoxContentsPaintState contents_paint_state(...);
    PaintObject(contents_paint_state.GetPaintInfo(), ...);
  }
}
```

**对于我们的红色 div**（没有滚动，没有子元素）：

1. `original_phase = PaintPhase::kBlockBackground`
2. `ShouldPaintSelfBlockBackground()` 返回 true
3. 转换 phase 为 `kSelfBlockBackgroundOnly`
4. `box.ScrollsOverflow()` 返回 false（没有 overflow: scroll）
5. **只调用一次 `PaintObject`**，绘制背景
6. `PaintObject` 返回后，因为 phase 是 `kSelfBlockBackgroundOnly`，不继续绘制子元素

**为什么有些元素需要画两次背景？** 对于有 `overflow: scroll` 的容器，背景可能有两种行为：

- **Border box space**：背景固定，滚动时不动（`background-attachment: fixed`）
- **Scrolling contents space** 滚动执行实际绘制

`PaintObject` 根据 phase 执行具体绘制。对于我们的红色 div，phase 是 `kSelfBlockBackgroundOnly`。

```cpp
void BoxFragmentPainter::PaintObject(
    const PaintInfo& paint_info,
    const PhysicalOffset& paint_offset,
    bool suppress_box_decoration_background) {

  const PaintPhase paint_phase = paint_info.phase;  // kSelfBlockBackgroundOnly
  const PhysicalBoxFragment& fragment = GetPhysicalFragment();
  const ComputedStyle& style = fragment.Style();
  const bool is_visible = IsVisibleToPaint(fragment, style);  // true

  if (ShouldPaintSelfBlockBackground(paint_phase)) {
    if (is_visible) {
      // 关键调用！绘制背景、边框、阴影
      PaintBoxDecorationBackground(paint_info, paint_offset,
                                   suppress_box_decoration_background);
    }

    if (paint_phase == PaintPhase::kSelfBlockBackgroundOnly) {
      return;  // 画完背景就结束，不处理子元素
    }
  }

  // 下面是其他 phase 的处理逻辑...
  if (paint_phase == PaintPhase::kMask && is_visible) {
    PaintMask(paint_info, paint_offset);
    return;
  }

  if (paint_phase == PaintPhase::kForeground) {
    // 绘制文本、图片等前景内容
    if (items_) {
      PaintLineBoxes(paint_info, paint_offset);
    } else if (!fragment.IsInlineFormattingContext()) {
      PaintBlockChildren(paint_info, paint_offset);
    }
  }
}
```

**对于我们的红色 div 的执行路径**：

1. `paint_phase = kSelfBlockBackgroundOnly`
2. `is_visible = true`（有 background-color）
3. 调用 `PaintBoxDecorationBackground` 绘制红色背景
4. 检查 `paint_phase == kSelfBlockBackgroundOnly` → **返回**
5. 不执行后面的前景/子元素绘制逻辑

**为什么叫 “SelfBlockBackgroundOnly”？** 这个 phase 只画 **自己** 的背景（box decoration），不画子元素。这是绘制顺序的一部分——先画父元素背景，再画子元素，最后画前景和边框。

## 完整的绘制阶段（一般情况）

虽然我们的例子只用到 `kSelfBlockBackgroundOnly`，但一个完整的元素（有子元素、边框等）会经历多个阶段：

1. **PaintPhase::kBlockBackground**
    - `kSelfBlockBackgroundOnly`：画自己的背景
    - `kDescendantBlockBackgroundsOnly`：递归画子元素背景
2. **PaintPhase::kForeground**
    - 画文本、图片、子元素内容
3. **PaintPhase::kFloat**
    - 画浮动元素
4. **PaintPhase::kOutline**
    - 画 outline（最上层）

每个阶段生成一批 DisplayItem，按顺序追加到 DisplayItemList，最终交给合成线程