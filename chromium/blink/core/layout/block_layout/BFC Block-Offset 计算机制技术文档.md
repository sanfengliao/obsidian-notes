> 本文基于 `third_party/blink/renderer/core/layout/` 下的`block_layout_algorithm.cc`、`block_layout_algorithm.h`、`constraint_space.h`等源码，系统阐述 LayoutNG 块级布局算法中**BFC block-offset（块格式化上下文块向偏移）** 的计算、解析（resolve）、传播与中止重排机制。注：文中只标注函数/方法名，不标注行号——行号会随源码演进而漂移。

---

# 1. 概述

## 1.1 什么是 BFC block-offset

在 Chromium 的 LayoutNG 布局系统中，**BFC block-offset** 指的是一个元素在
其所在的**块格式化上下文（Block Formatting Context, BFC）**中的绝对块方向坐标
（block-start 边缘相对于 BFC 根的偏移）。

它是 LayoutNG 坐标体系的核心锚点。每个块级容器在 `BlockNode` 上经过
`BlockLayoutAlgorithm::Layout()` 后，会产生一个 `LayoutResult`，其中
`BfcBlockOffset()` 字段就携带了该值（`std::optional<LayoutUnit>`，可能为空）。

### 1.2 为什么 BFC block-offset 难以一开始就确定

CSS 的 **margin 折叠（margin collapsing）** 是核心难点。考虑：

```html height="320"
<div class="parent">   <!-- parent 的 block-start margin 可能与 child1 折叠 -->
  <div class="child1"></div>
  <div class="child2"></div>
</div>
```

- parent 自身的 `margin-top` 与 child1 的 `margin-top` 会折叠；
- child1 若是"自折叠"的（如空 div，且无 height），其 margin 会继续穿透，与 child2 折叠；
- 只有遇到一个 \*\*"实体"（有 border/padding、有非零 block-size、建立新 BFC、被浮动清除）

时，margin 折叠链才会终止，BFC block-offset 才能最终确定。

因此 LayoutNG 采用了 **延迟解析 + 子元素反馈 + 中止重排** 的策略：
parent 在布局开始时通常**不知道**自己的 BFC block-offset（`std::nullopt`），
需要等第一个"非自折叠、非清除"的子元素把自己的 BFC block-offset 算出来，
再**反馈**给 parent，由 parent 调用 `ResolveBfcBlockOffset()` 采用。

---

# 2. 核心概念与数据结构

## 2.1 ConstraintSpace（约束空间）

父元素在布局子元素前会为其创建一个 `ConstraintSpace`（约束空间），其中携带
与 BFC block-offset 相关的几个关键字段（定义于 `constraint_space.h`）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `BfcOffset()` | `BfcOffset{line, block}` | **估计值**：当前在 BFC 中放置的位置，不是元素自身的最终位置（详见 2.1.1） |
| `ForcedBfcBlockOffset()` | `std::optional<LayoutUnit>` | **强制** BFC 偏移。当祖先有越过邻接浮动的清除，或二轮布局需要时设置。此值是确定性的，子布局必须遵守 |
| `OptimisticBfcBlockOffset()` | `std::optional<LayoutUnit>` | **乐观** BFC 偏移。仅是一个提示（来自前一次布局的缓存位置），不保证正确 |
| `ExpectedBfcBlockOffset()` | `LayoutUnit` | 期望值 = `Forced` ?? `Optimistic` ?? `BfcOffset().block`，用于判断实际解析值是否与估计一致（决定是否需要中止重排） |
| `ClearanceOffset()` | `std::optional<LayoutUnit>` | 浮动清除偏移 |
| `IsNewFormattingContext()` | `bool` | 是否建立新的格式化上下文 |
| `GetMarginStrut()` | `MarginStrut` | 传入的 margin 折叠链 |

> 四者优先级关系（`ExpectedBfcBlockOffset`）：
> \`cpp
> return ForcedBfcBlockOffset().value\_or(
> OptimisticBfcBlockOffset().value*or(GetBfcOffset().block*offset));
> 

### 2.1.1 `BfcOffset` 为何是"估计值"——它基于什么估计

这是理解整个 BFC block-offset 机制最关键的一点。源码中对 `BfcOffset` 的
注释定义是：

> \*\*"The BfcOffset is where the MarginStrut is placed within the block
> formatting context."\*\*
> （BfcOffset 是 MarginStrut 在块格式化上下文中放置的位置）

也就是说，`BfcOffset.block` **并不是元素自身 border edge 的最终位置**，而是
**margin 折叠链（MarginStrut）当前被放置的位置**。元素真正的 block-start border
edge 要在它之上再叠加折叠后的 margin：

```cpp
// 解析 BFC block-offset 的标准公式
bfc_block_offset =
    space.GetBfcOffset().block_offset + space.GetMarginStrut().Sum();
```

正因为如此，`BfcOffset.block` 在元素真正"落地"之前只是一个**估计值**，原因有三：

**① 它依赖父链的 BFC offset，而父链可能尚未解析。
**`BlockLayoutAlgorithm::BfcBlockOffset()` 的取值逻辑是：

```cpp
LayoutUnit BfcBlockOffset() const {
  // 若自身已解析，用真实值
  if (container_builder_.BfcBlockOffset())
    return *container_builder_.BfcBlockOffset();
  // 否则回退到父算法分配下来的 BfcOffset（同样是估计值）
  return GetConstraintSpace().GetBfcOffset().block_offset;
}
```

当 parent 自己的 BFC offset 还是 `std::nullopt` 时，它传给 child 的
`BfcOffset.block` 其实是从**它的**父元素继承下来的估计值——这是一个**递归的估计链**。
链上任何一个祖先解析了真实 BFC offset（例如遇到 border/padding、新 BFC、清除），
整条链都会平移，所有未解析后代的估计值都会跟着变。

**② MarginStrut 仍可能变化。
**此时累积的 `MarginStrut` 还不是最终值：后续的自折叠后代（空块）的 margin 仍可能
折叠进来使其增长；负 margin 会抵消；遇到 clearance 会把 margin 链"切断"分离。
所以 `BfcOffset.block + MarginStrut.Sum()` 这个候选位置随时可能被改写。

**③ 浮动 / 清除可能下推。
**即便 margin 链稳定，`ApplyClearance` 也可能把元素往下推到一个高于估计的位置。

**那这个估计值是怎么算出来的？** 在 `ComputeChildData` 中，parent 为每个 child
计算 `bfc_offset_estimate`：

```cpp
BfcOffset child_bfc_offset = {
    GetConstraintSpace().GetBfcOffset().line_offset +
        BorderScrollbarPadding().LineLeft(direction) +
        additional_line_offset + margins.LineLeft(direction),
    BfcBlockOffset() + logical_block_offset};   // ← block 部分
```

其中：

- `BfcBlockOffset()` = parent 的 BFC block（已解析则真实，未解析则继承自祖先的估计）；
- `logical_block_offset` = parent 内部已累积的块向游标（block-start border/padding +

已布局兄弟节点消耗的 block-size）。

随后在 `LayoutInflow` 之前，再叠加 margin 折叠得到 child 的候选 BFC 偏移：

```cpp
adjoining_bfc_offset_estimate =
    child_data.bfc_offset_estimate.block_offset + adjoining_margin_strut.Sum();
```

即 **child 候选 BFC = (parent BFC 估计 + parent 内部累积偏移) + 折叠 margin 之和**。

**为什么"最后可能就是它"？
**当 margin 折叠链在本元素处终止（本元素有 border/padding、非零 block-size、建立新 BFC、
或被清除），`ResolveBfcBlockOffset` 写入的最终值正是按上面那个公式算出来的
`BfcOffset.block + MarginStrut.Sum()`（可能再经 `ApplyClearance` 调整）。
只要估计过程中没有发生意外——没有后续 margin 突变、没有浮动下推、祖先解析后没有平移
——**最终解析值就等于这个估计值**。

这正是 `NeedsAbortOnBfcBlockOffsetChange` 的判定基础：它把 `ResolveBfcBlockOffset
`刚写入的真实值与 `ExpectedBfcBlockOffset`（即估计值）做比较，**相等就不必重排**。
换句话说，估计值在多数情况下"恰好正确"，只有当它与最终解析值不一致时才触发回滚重排。
几个偏移字段的分工也由此清晰：

| 字段 | 性质 | 作用 |
| --- | --- | --- |
| `BfcOffset.block` | **基线估计**（始终存在） | 递归继承自父链，是"如果没有更精确信息时的默认猜测" |
| `OptimisticBfcBlockOffset` | **精炼估计**（可选，来自缓存） | 上一轮布局落点，覆盖基线用于放置邻接对象（浮动等） |
| `ForcedBfcBlockOffset` | **权威值**（可选） | 已知正确的强制位置，子布局必须遵守，无需再猜 |
| `ExpectedBfcBlockOffset` | **合并后的期望** | `Forced ?? Optimistic ?? BfcOffset.block`，作为"是否需要重排"的比对基准 |

## 2.2 Container Fragment Builder

`BlockLayoutAlgorithm` 内部维护一个 `container_builder_`（`BlockFragmentBuilder`），
它在布局过程中逐步累积结果。与 BFC block-offset 相关的状态：

- `container_builder_.BfcBlockOffset()` → `std::optional<LayoutUnit>`，

初始为 `std::nullopt`，被 `ResolveBfcBlockOffset()` 设置后变为有值。

- `container_builder_.BfcLineOffset()` → 行向 BFC 偏移（通常布局一开始即已知）。
- `container_builder_.IsPushedByFloats()` → 是否被浮动下推。
- `container_builder_.IsSelfCollapsing()` → 是否自折叠。

## 2.3 PreviousInflowPosition（前置流内位置）

布局子元素循环中维护的游标：

```cpp
PreviousInflowPosition previous_inflow_position = {
    LayoutUnit(),                      // logical_block_offset
    constraint_space.GetMarginStrut(), // margin_strut（折叠链）
    is_resuming_ ? LayoutUnit() : container_builder_.Padding().block_start,
    /* self_collapsing_child_had_clearance */ false};
```

其中 `margin_strut` 累积了未终止的 margin 折叠链，是估计子元素 BFC 偏移的依据。

## 2.4 MarginStrut（margin 折叠链）

记录正/负 margin 的最大值、是否 discard 等。`Sum()` 给出折叠后的净 margin 值，
用于将"估计偏移"叠加 margin 得到子元素的实际候选位置。

## 2.5 自折叠（self-collapsing）

一个块若 **margin 可以穿透它**，即为自折叠。典型情况：

- 空块（无 children、无 height、无 border/padding），block-size 为 0；
- 此时它的 BFC block-offset **无法确定**（`std::nullopt`），需要推迟。

**关键判别**：有确定非零 block-size 的块**不是**自折叠的——margin 无法穿透一个
有高度的块，因此它必须在自己 `Layout()` 末尾确定 BFC offset。

---

# 3. BFC block-offset 的解析时机

LayoutNG 中 BFC block-offset 的解析发生在多个"锚点"，概括为以下五类：

## 3.1 布局开始时立即解析

当满足以下任一条件，parent 在布局一开始就能（且必须）解析 BFC offset：

- **A. 有 block-start border/padding**（`content_edge` 非零）：border/padding 阻断了

parent 与 child 之间的 margin 折叠；

- **B. 建立新 BFC**（`IsNewFormattingContext()`，如 `overflow:hidden`）；
- **C. 从 break token 恢复布局**（`is_resuming_`）：分片场景下 margin strut 不能跨片传递。

```cpp
if (content_edge || is_resuming_ ||
    constraint_space.IsNewFormattingContext()) {
  ...
  if (!ResolveBfcBlockOffset(&previous_inflow_position)) {
    return container_builder_.Abort(LayoutResult::kBfcBlockOffsetResolved);
  }
  previous_inflow_position.logical_block_offset = content_edge;
}
```

## 3.2 子元素反馈解析（`HandleInflowChildResult`）

parent 在布局子元素后，根据子元素结果反向解析自身 BFC offset。这是**最常见的路径**，
详见第 5 节。

## 3.3 有清除（clearance）时解析

子元素被浮动清除时，清除（clearance）会阻断 margin 折叠，使 parent 的 BFC offset
可解析：

```cpp
if (child_had_clearance) {
  if (!ResolveBfcBlockOffset(previous_inflow_position))
    return LayoutResult::kBfcBlockOffsetResolved;
}
```

## 3.4 布局末尾、有非零 block-size 时解析

当块自身（即使无 children）有确定的非零 block-size、或有 break token、或被 column
spanner 中断时，在 `Layout()` 末尾解析：

```cpp
if (!container_builder_.BfcBlockOffset() &&
    (border_box_size.block_size || GetBreakToken() ||
     container_builder_.FoundColumnSpanner())) {
  if (!ResolveBfcBlockOffset(previous_inflow_position))
    return container_builder_.Abort(LayoutResult::kBfcBlockOffsetResolved);
}
```

> 注释说明："If we have a non-zero block-size (margins don't collapse through us)"。

> 这正是 `<div style="height:100px">`（无 children）能在自身 Layout() 末尾确定

> BFC offset、且**不**被标记为自折叠的根本原因。

## 3.5 布局末尾、穿透到底的兜底解析

如果一路穿透到块末尾仍未解析（既无 border/padding、子元素全自折叠、无清除），
则在收尾阶段兜底解析（依赖 clearance 或 block-end border 等实体）：

```cpp
if (!container_builder_.BfcBlockOffset()) {
  DCHECK(!constraint_space.IsNewFormattingContext());
  if (!ResolveBfcBlockOffset(previous_inflow_position)) {
    return container_builder_.Abort(LayoutResult::kBfcBlockOffsetResolved);
  }
}
```

---

# 4. 核心函数 `ResolveBfcBlockOffset`

该函数是所有解析路径的**汇聚点**，最终通过
`container_builder_.SetBfcBlockOffset()` 写入 BFC offset。

```cpp
bool BlockLayoutAlgorithm::ResolveBfcBlockOffset(
    PreviousInflowPosition* previous_inflow_position,
    LayoutUnit bfc_block_offset,
    std::optional<LayoutUnit> forced_bfc_block_offset) {
  // 1. 若祖先已预应用清除（前一轮已解析），继承该状态
  if (GetConstraintSpace().IsPushedByFloats())
    container_builder_.SetIsPushedByFloats();

  // 2. 幂等：若已设置直接返回
  if (container_builder_.BfcBlockOffset())
    return true;

  // 3. 优先采用 forced 值，否则采用子元素反馈的 bfc_block_offset
  bfc_block_offset = forced_bfc_block_offset.value_or(bfc_block_offset);

  // 4. 应用浮动清除规则，可能抬高 offset
  if (ApplyClearance(GetConstraintSpace(), &bfc_block_offset))
    container_builder_.SetIsPushedByFloats();

  // 5. ★ 写入 BFC block-offset
  container_builder_.SetBfcBlockOffset(bfc_block_offset);

  // 6. 若实际值与期望值（估计）不一致，且存在需中止的邻接对象，则要求重排
  if (NeedsAbortOnBfcBlockOffsetChange()) {
    DCHECK(!GetConstraintSpace().IsNewFormattingContext());
    return false;
  }

  // 7. 重置 logical_block_offset 到 block-start border edge
  previous_inflow_position->logical_block_offset = LayoutUnit();

  // 8. 解析完成意味着 margin 折叠链终止，重置 margin strut
  //    （resuming 场景例外：保留空 strut 用于与 fragmentainer 边界折叠）
  if (!is_resuming_)
    previous_inflow_position->margin_strut = MarginStrut();
  else
    DCHECK(previous_inflow_position->margin_strut.IsEmpty());

  return true;
}
```

## 4.1 中止判定 `NeedsAbortOnBfcBlockOffsetChange`

```cpp
bool BlockLayoutAlgorithm::NeedsAbortOnBfcBlockOffsetChange() const {
  DCHECK(container_builder_.BfcBlockOffset());
  if (!abort_when_bfc_block_offset_updated_)
    return false;
  // 实际位置与（乐观）估计不同 → 必须回滚重排已布局的邻接对象
  return *container_builder_.BfcBlockOffset() !=
         GetConstraintSpace().ExpectedBfcBlockOffset();
}
```

当 `abort_when_bfc_block_offset_updated_` 为真（存在前序邻接对象，或处于分片流中），
且最终解析值与 `ExpectedBfcBlockOffset` 不一致时，返回 `false` 触发调用方中止当前布局
（`Abort(kBfcBlockOffsetResolved)`），由上层重启布局并带上正确的 `ForcedBfcBlockOffset`。

---

# 5. 子元素反馈与传播机制（核心流程）

本节用一个三段式示例串联全流程：

```html height="320"
<div class="parent">
  <div class="child1"></div>
  <div class="child2"></div>
</div>
```

## 5.1 阶段 1：Parent 布局开始

- `container_builder_.BfcBlockOffset()` → `std::nullopt`（未知）；
- `BfcLineOffset()` 已知（从约束空间复制）；
- `content_edge = BorderScrollbarPadding().block_start`；
- 通常 `content_edge==0`、非新 BFC、非 resuming → **不**立即解析，等待 child1。

## 5.2 阶段 2：为 child1 创建约束空间

`CreateConstraintSpaceForChild` 依据 parent 当前是否已知 BFC offset，
决定向 child1 传递何种偏移信息：

```cpp
bool has_bfc_block_offset = container_builder_.BfcBlockOffset().has_value();
// parent 此时仍为 nullopt

if (!has_bfc_block_offset && constraint_space.ForcedBfcBlockOffset()) {
  builder.SetForcedBfcBlockOffset(*constraint_space.ForcedBfcBlockOffset());
} else if (constraint_space.OptimisticBfcBlockOffset()) {
  builder.SetOptimisticBfcBlockOffset(
      *constraint_space.OptimisticBfcBlockOffset());
}
```

同时在 `LayoutInflow` 之前，基于 margin strut 估计 child1 的候选 BFC 偏移：

```cpp
LayoutUnit adjoining_bfc_offset_estimate =
    child_data.bfc_offset_estimate.block_offset + adjoining_margin_strut.Sum();
LayoutUnit non_adjoining_bfc_offset_estimate =
    child_data.bfc_offset_estimate.block_offset +
    previous_inflow_position->margin_strut.Sum();
LayoutUnit child_bfc_offset_estimate = adjoining_bfc_offset_estimate;
```

## 5.3 阶段 3：child1 内部布局并返回结果

`LayoutInflow(child1_space, ...)` 递归进入 child1 的 `BlockLayoutAlgorithm::Layout()`。
child1 可能：

- **自身决定 BFC offset**（建立新 BFC / 有非零 block-size / 有清除）→

返回 `result->BfcBlockOffset()` 有值，`Status()==kSuccess`；

- **无法决定**（自折叠空块）→ 返回 `BfcBlockOffset()` 为 `std::nullopt`，

`IsSelfCollapsing()==true`；

- **因解析了 BFC offset 而中止自身布局** → `Status()==kBfcBlockOffsetResolved`，

但 `BfcBlockOffset()` 一定有值（这是给上层用的信号）。

## 5.4 阶段 4：Parent 处理 child1 结果（⭐ 关键）

`HandleInflowChildResult` 是反馈机制的枢纽。

**路径 A：child1 以** `kBfcBlockOffsetResolved` **中止，且 parent 尚无 BFC offset**

```cpp
if (layout_result->Status() == LayoutResult::kBfcBlockOffsetResolved &&
    !container_builder_.BfcBlockOffset()) {
  DCHECK(child_bfc_block_offset);
  abort_when_bfc_block_offset_updated_ = true;

  LayoutUnit bfc_block_offset = *child_bfc_block_offset;

  if (normal_child_had_clearance) {
    if (GetConstraintSpace().ClearanceOffset() ==
        child_space.ClearanceOffset()) {
      container_builder_.SetIsPushedByFloats();
    } else {
      bfc_block_offset = NextBorderEdge(*previous_inflow_position);
    }
  }
  // 用 child1 的值解析 parent
  if (!ResolveBfcBlockOffset(previous_inflow_position, bfc_block_offset,
                             /* forced */ std::nullopt)) {
    return LayoutResult::kBfcBlockOffsetResolved;
  }
}
```

**路径 B：child1 非自折叠、非清除，正常成功**

```cpp
} else if (!child_had_clearance && !is_self_collapsing) {
  // child1 有 BFC offset → 直接用它解析 parent
  if (!ResolveBfcBlockOffset(previous_inflow_position,
                             *child_bfc_block_offset))
    return LayoutResult::kBfcBlockOffsetResolved;
}
```

无论走 A 还是 B，最终都调用 `ResolveBfcBlockOffset`，于是：

```
✅ container_builder_.SetBfcBlockOffset(child1_bfc_block_offset);
   Parent.BfcBlockOffset = child1 的值
```

> **设计要点**：parent 的 BFC block-offset 通常由**第一个非自折叠、无清除的子元素**决定。

## 5.5 阶段 5：Parent 已知 BFC，继续布局 child2

此时 `container_builder_.BfcBlockOffset()` 有值。`CreateConstraintSpaceForChild` 走
另一分支（`has_bfc_block_offset` 为真），可利用前次缓存做相对位移优化：

```cpp
if (has_bfc_block_offset) {
  if (child.IsBlock()) {
    if (const LayoutResult* cached_result =
            child.GetLayoutBox()->GetCachedLayoutResult(...)) {
      const auto& prev_space = cached_result->GetConstraintSpaceForCaching();
      LayoutUnit bfc_block_delta =
          child_data.bfc_offset_estimate.block_offset -
          prev_space.GetBfcOffset().block_offset;
      if (prev_space.ForcedBfcBlockOffset())
        builder.SetOptimisticBfcBlockOffset(
            *prev_space.ForcedBfcBlockOffset() + bfc_block_delta);
      else if (prev_space.OptimisticBfcBlockOffset())
        builder.SetOptimisticBfcBlockOffset(
            *prev_space.OptimisticBfcBlockOffset() + bfc_block_delta);
    }
  }
}
```

子元素最终位置由 `CalculateLogicalOffset` 确定：当 parent 与 child 都有
BFC offset 时，用两者 BFC 偏移之差算出绝对逻辑位置：

```cpp
if (child_bfc_block_offset && container_builder_.BfcBlockOffset()) {
  return LogicalFromBfcOffsets(
      {child_bfc_line_offset, *child_bfc_block_offset},
      ContainerBfcOffset(), fragment.InlineSize(), inline_size, direction);
}
// 否则 block 方向相对偏移为 0（推迟定位）
return {inline_offset, LayoutUnit()};
```

---

# 6. 中止与重排机制（kBfcBlockOffsetResolved）

`LayoutResult::kBfcBlockOffsetResolved` 是 LayoutNG 协调延迟解析与一致性的关键状态。

## 6.1 触发场景

1. **乐观估计错误**：用 `OptimisticBfcBlockOffset` 乐观放置了邻接对象（如浮动），

最终解析的真实 BFC offset 与估计不同 → `NeedsAbortOnBfcBlockOffsetChange()` 返回
`false` → `ResolveBfcBlockOffset` 返回 `false` → 调用方 `Abort`。

1. **子元素自身解析后中止**：子元素解析了自己的 BFC offset，但还需要带上该信息重排

（例如发现新 BFC 子树在当前 float 旁放不下），返回 `kBfcBlockOffsetResolved`。

1. **二轮布局仍中止**：父元素用已知 BFC offset 给 child 重排，child 仍可能因新 BFC

后代与浮动冲突而二次中止（允许"再放一程"）。

## 6.2 重启布局

中止信号会沿调用栈向上传播，直到到达一个**已经解析了 BFC offset 的节点
**（能提供 `ForcedBfcBlockOffset` 的节点）来重启布局。这正是 `ForcedBfcBlockOffset
`的两大来源之一（另一个是祖先的越界清除）。

## 6.3 流程示意

```
child1 布局结果
   │
   ├─ Status()==kBfcBlockOffsetResolved && parent 无 BFC
   │     → 用 child1 的值 ResolveBfcBlockOffset 解析 parent
   │     → 若 NeedsAbort → parent 也 Abort，向上传播
   │
   ├─ Status()==kSuccess && child1 有 BFC && 非清除 && 非自折叠
   │     → ResolveBfcBlockOffset(child1_value) 解析 parent
   │
   └─ child1 自折叠 (BFC==nullopt)
         → parent 仍未知，推迟到 child2 或布局末尾
```

---

# 7. 特殊场景分析

## 7.1 自折叠空块（`<div></div>`，无 height 无 children）

- block-size = 0，margin 可穿透 → `IsSelfCollapsing()==true`；
- 自身 `BfcBlockOffset()` 始终为 `std::nullopt`；
- parent 无法借此解析，需推迟到下一个非自折叠子元素或布局末尾；
- 若 parent 后续已知 BFC，则通过

`PositionSelfCollapsingChildWithParentBfc()` 反向计算其位置。

## 7.2 有确定高度无 children（`<div style="height:100px">`）

与 7.1 **本质不同**：

1. child1 的子元素循环不执行，`intrinsic_block_size_ = 0`；
2. `ComputeBlockSizeForFragment` 根据 style 的 `height` 得到

`border_box_size.block_size = 100px`；

1. 末尾检查 `!BfcBlockOffset() && block_size(非零)` → 进入分支，

`ResolveBfcBlockOffset` 解析自身 BFC offset；

1. 随后检查 `BfcBlockOffset()` 已知 → **不**进入 `else`，

**不**调用 `SetIsSelfCollapsing()` → `IsSelfCollapsing()==false`；

1. parent 走正常路径用 child1 的 BFC offset 解析自身。

**对比表：**

| 情况 | block-size | IsSelfCollapsing | BFC 在哪设置 | parent BFC 由谁决定 |
| --- | --- | --- | --- | --- |
| `height:100px` 无 children | 100px | false | child1 自身（末尾非零 block-size 解析） | child1（正常路径） |
| 空 div 无 height 无 children | 0 | true | 不设置（nullopt） | 推迟到 parent 末尾/child2 |
| 新 BFC（`overflow:hidden`） | 任意 | false | child1 内部（布局开始立即解析） | child1（中止或正常路径） |
| 有清除（`clear:both`） | 任意 | false | child1 clearance 调整 | child1（调整后） |

## 7.3 建立新 BFC（`overflow:hidden`）

- `IsNewFormattingContext()==true`；
- child1 在自身 `Layout()` 一开始（`IsNewFormattingContext` 分支）立即解析 BFC offset；
- parent 接收时通常走中止传播或正常路径解析。

## 7.4 有浮动清除（`clear:both`）

- child1 经 `ApplyClearance` 调整 offset，`IsPushedByFloats()==true`；
- parent 处理：若 child1 与 parent 的 `ClearanceOffset` 相同，parent 也标记

`SetIsPushedByFloats()`；否则用 `NextBorderEdge` 重新计算；

- 清除会阻断 margin 折叠，使 parent 的 BFC offset 可立即解析。

---

# 8. 完整流程图

```
时间轴
│
├─ 1. Parent Layout() 开始
│     BfcBlockOffset: ❌ unknown
│     创建 ConstraintSpace for child1（传递 Forced/Optimistic 估计）
│
├─ 2. child1 创建约束空间
│     估计 BFC = parent_content_edge + margin_strut.Sum()
│
├─ 3. child1 Layout() 执行
│     内部可能解析自身 BFC（新 BFC / 非零 block-size / 清除）
│     返回 BfcBlockOffset 或 std::nullopt，或以 kBfcBlockOffsetResolved 中止
│
├─ 4. Parent 处理 child1 结果   ⭐ 关键
│     ├─ kBfcBlockOffsetResolved && parent 无 BFC
│     │     → ResolveBfcBlockOffset(child1_value)
│     ├─ kSuccess && child1 有 BFC && 非清除 && 非自折叠
│     │     → ResolveBfcBlockOffset(child1_value)
│     └─ child1 自折叠
│           → 推迟
│     → container_builder_.SetBfcBlockOffset(...)  ✅ NOW SET
│     → 若 NeedsAbort → Abort(kBfcBlockOffsetResolved) 向上传播
│
├─ 5. child2 创建约束空间
│     parent BFC 已知 → 可用缓存优化（OptimisticBfcBlockOffset 相对位移）
│
├─ 6. child2 Layout() 执行
│     CalculateLogicalOffset 直接用 parent.BfcBlockOffset 定位
│
└─ 7. Parent Layout() 结束
      输出 Fragment，BfcBlockOffset = child1 解析的值
```

---

# 9. 代码追踪路径

```typescript
BlockLayoutAlgorithm::Layout()
  ├─ 布局开始：判断是否立即解析 parent BFC
  │   （content_edge / new FC / resuming）
  │
  ├─ for child in children:
  │    ├─ CreateConstraintSpaceForChild()      传递 Forced/Optimistic 估计
  │    ├─ 估计 child BFC offset                （bfc_offset_estimate + margin strut）
  │    ├─ LayoutInflow(child_space) → 递归 BlockLayoutAlgorithm::Layout()
  │    │    ├─ 布局开始：新 BFC / content_edge 立即解析
  │    │    ├─ 子元素循环（无 children 跳过）
  │    │    ├─ 布局末尾：穿透到底兜底解析
  │    │    └─ 布局末尾：非零 block-size → ResolveBfcBlockOffset ⭐
  │    │
  │    └─ HandleInflowChildResult()
  │         ├─ 处理 kBfcBlockOffsetResolved → ResolveBfcBlockOffset ⭐
  │         ├─ 有清除 → ResolveBfcBlockOffset ⭐
  │         ├─ 自折叠子元素 + parent 已知 BFC
  │         │   → PositionSelfCollapsingChildWithParentBfc()
  │         ├─ 二轮重排（带已知 BFC 重新 CreateConstraintSpaceForChild）
  │         └─ 非自折叠直接传递 → ResolveBfcBlockOffset ⭐
  │
  ├─ CalculateLogicalOffset()                  用 parent/child BFC offset 定位
  └─ return LayoutResult (含 container_builder_.BfcBlockOffset())
```

---

# 10. 核心要点总结

1. **延迟解析**：parent 的 BFC block-offset 通常一开始未知（`std::nullopt`），

由第一个**非自折叠、无清除**的子元素反馈决定。

1. **统一汇聚点**：所有解析路径最终都调用 `ResolveBfcBlockOffset()`，

由它写入 `container_builder_.SetBfcBlockOffset()` 并判定是否需要中止重排。

1. **几个偏移字段**：`ForcedBfcBlockOffset`（强制）、`OptimisticBfcBlockOffset`

（乐观缓存）、`BfcOffset.block`（基线估计），优先级 Forced > Optimistic > 基线，
共同构成 `ExpectedBfcBlockOffset`。

1. **margin 折叠驱动**：BFC offset 的"何时能解析"本质上由 margin 折叠链是否被

实体（border/padding、非零 block-size、新 BFC、clearance）阻断决定。

1. **自折叠 vs 有高度**：有确定非零 block-size 的块**不是**自折叠，会在自身

`Layout()` 末尾解析 BFC offset；真正空块（block-size=0）是自折叠，
BFC offset 保持 `nullopt`，推迟到 parent。

1. **中止重排保一致性**：乐观估计可能出错，`NeedsAbortOnBfcBlockOffsetChange`

比对实际值与期望值，不一致则 `Abort(kBfcBlockOffsetResolved)`，由已解析节点
带 `ForcedBfcBlockOffset` 重启布局。

1. **幂等性**：`ResolveBfcBlockOffset` 开头检查

`if (container_builder_.BfcBlockOffset()) return true;`，
多次调用安全，BFC offset 一旦设置后后续子元素直接复用，无需重新计算。

1. **位置计算**：当 parent 与 child 都已知 BFC offset 时，`CalculateLogicalOffset`

用两者 BFC 偏移之差算绝对逻辑位置；否则 child 的 block 偏移临时为 0，推迟定位。

---

## 附：关键源码索引

| 函数/逻辑 | 所在文件 | 作用 |
| --- | --- | --- |
| `BfcBlockOffset()` | `block_layout_algorithm.h` | 取自身 BFC block（已解析用真实值，否则回退父链估计） |
| `BfcOffset()` / `GetBfcOffset()` | `constraint_space.h` | margin 折叠链放置位置（基线估计） |
| `ForcedBfcBlockOffset()` | `constraint_space.h` | 强制偏移访问器 |
| `OptimisticBfcBlockOffset()` | `constraint_space.h` | 乐观偏移访问器 |
| `ExpectedBfcBlockOffset()` | `constraint_space.h` | 期望偏移（优先级合并） |
| `ResolveBfcBlockOffset()` | `block_layout_algorithm.cc` | 核心解析汇聚点 |
| `NeedsAbortOnBfcBlockOffsetChange()` | `block_layout_algorithm.cc` | 中止判定 |
| `ComputeChildData()` | `block_layout_algorithm.cc` | 为 child 计算 bfc*offset*estimate |
| `CreateConstraintSpaceForChild()` | `block_layout_algorithm.cc` | 为 child 创建约束空间、传递偏移 |
| `HandleInflowChildResult()` | `block_layout_algorithm.cc` | 处理子元素结果、反馈解析 parent |
| `CalculateLogicalOffset()` | `block_layout_algorithm.cc` | 用 BFC offset 定位子元素 |
| `PositionSelfCollapsingChildWithParentBfc()` | `block_layout_algorithm.cc` | 反向计算自折叠子元素位置 |
| `NextBorderEdge()` | `block_layout_algorithm.h` | 提交待定 margin 后的下一个 border edge |
