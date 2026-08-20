这份文档聚焦 Blink 里 CSS scroll snap 的主链路：样式变化如何被收集、如何转换成 snap container / snap area 数据、以及事件如何派发到脚本侧。

# 总览

可以把实现理解成四层：

1. 样式层：`scroll-snap-type`、`scroll-snap-align`、`scroll-snap-stop` 这些属性在 computed style 中保存。
2. DOM / layout 层：Element 和 PaintLayer 负责识别“谁是 snap container，谁是 snap area”。
3. 计算层：`SnapCoordinator` 把 DOM/layout 信息转换成 `cc::SnapContainerData` / `cc::SnapAreaData`。
4. 事件层：`ScrollableArea` 和 `Document` 负责把 `scrollsnapchange` / `scrollsnapchanging` 事件送到脚本。

# 关键源码与分析

## 滚动层创建与 CSS target 重传播 — [paint_layer.cc](third_party/blink/renderer/core/paint/paint_layer.cc)

```cpp
void PaintLayer::UpdateScrollableArea() {
  if (RequiresScrollableArea() == !!scrollable_area_)
    return;

  if (!scrollable_area_) {
    scrollable_area_ = MakeGarbageCollected<PaintLayerScrollableArea>(*this);
    const ComputedStyle& style = GetLayoutObject().StyleRef();
    // A newly created snap container may need to be made aware of snap areas
    // within it which are targeted or contain a targeted element. Such a
    // container may also change the snap areas associated with snap containers
    // higher in the DOM.
    if (!style.GetScrollSnapType().is_none) {
      if (Element* css_target = GetLayoutObject().GetDocument().CssTarget()) {
        css_target->SetTargetedSnapAreaIdsForSnapContainers();
      }
    }
  }
}
```

这里的重点是：新建滚动层时，如果这个元素本身就是 snap container，Blink 会立刻把当前 CSS target 对应的 snap 目标重新传播一遍。也就是说，snap 不是只靠一次样式计算完成的，它和滚动层的创建时机强相关。

这一段也说明了一个实际的实现策略：当容器出现或切换为可 snap 状态时，必须主动重新标记与它相关的 snap area，而不是等到下一次完整刷新。

## CSS target 的 snap 归属与重绑定 — [element.cc](third_party/blink/renderer/core/dom/element.cc)

```cpp
void Element::SetTargetedSnapAreaIdsForSnapContainers() {
  std::optional<cc::ElementId> targeted_area_id = std::nullopt;
  const LayoutBox* box = GetLayoutBox();
  while (box) {
    if (const ComputedStyle* style = box->Style()) {
      // If this is a snap area, associate it with the first snap area we
      // encountered, if any, since the previous snap container.
      if (box->IsScrollContainer() && !style->GetScrollSnapType().is_none) {
        if (auto* scrollable_area = box->GetScrollableArea()) {
          scrollable_area->SetTargetedSnapAreaId(targeted_area_id);
          GetDocument().View()->AddPendingSnapUpdate(scrollable_area);
        }
        targeted_area_id.reset();
      }
      const auto& snap_align = style->GetScrollSnapAlign();
      if (!targeted_area_id &&
          (snap_align.alignment_block != cc::SnapAlignment::kNone ||
           snap_align.alignment_inline != cc::SnapAlignment::kNone)) {
        if (Node* node = box->GetNode()) {
          targeted_area_id =
              CompositorElementIdFromDOMNodeId(node->GetDomNodeId());
        }
        box = box->ContainingScrollContainer();
        continue;
      }
    }
    box = box->ContainingBlock();
  }
}
```

这一段不是后代 snap area 的主收集器，而是 CSS target 的重绑定入口：它从当前元素往上走，遇到 snap area 就记住最近的目标 id，遇到 snap container 就把这个目标 id 绑定给容器，然后把这个容器加入 pending snap update。

代码里有一个很重要的行为：`box = box->ContainingScrollContainer(); continue;`。这意味着一旦找到一个 snap area，Blink 会直接跳到它外层的滚动容器，避免把中间不相关的 snap area 重复算进去。这个策略和“只把 snap area 贡献给最近祖先 scroll container”的规则是一致的。

`SetTargetedSnapAreaId` 存的是一个 `std::optional<cc::ElementId>`——`:target` 元素所在 snap area 的 ElementId。它经 `SnapCoordinator::UpdateSnapContainerData`（snap_coordinator.cc）写进 `cc::SnapContainerData::targeted_area_id_`，传到 compositor 后在 `FindSnapPosition`（cc/input/scroll_snap_data.cc）里参与候选选择，匹配该 id 的 snap area 会被优先选中。也就是说，它是“导航到 `#fragment` 后 `:target` 元素在 snap 时胜出”的实现机制。

注意这趟 walk 会对路径上**每一个**祖先 snap container 都调一次 `SetTargetedSnapAreaId`，而不只是最内层那个。关键在 `reset()` 之后：`targeted_area_id` 变回 `nullopt`，`!targeted_area_id` 重新为 `true`，所以 walk **不会**就此停下——继续往上时，下一个遇到的 snap area 会再次把 `targeted_area_id` 设成它自己的 id，归给更外层的 container。也就是说，每个 snap container 绑定的是“自上一个内层 snap container 以来遇到的第一个 snap area”（即代码注释那句）；只有当两层 container 之间没有别的 snap area 时，外层 container 才拿到 `std::nullopt`。

`reset()` 的作用是实现这种“按段归属”的语义，而不是清旧值。真正负责清旧 id 的是 `Document::SetCSSTarget` 里对旧 target 调的 `ClearTargetedSnapAreaIdsForSnapContainers`（见下）——它把旧 target 祖先链上的 container 全置 `nullopt`，避免新 target 的 walk 走不到的 container 里残留旧 id。



## 后代 snap area 的真正收集链路

如果把“谁是 snap area”拆开看，会发现它有两条不同的路径：

1. `SetTargetedSnapAreaIdsForSnapContainers()` 处理的是 CSS target 场景，作用是把当前 target 重新映射到它上方每个 snap container。
2. `FragmentBuilder::PropagateSnapAreas()` 处理的是常规后代收集，作用是在 layout fragment 构建过程中，把当前 fragment 里的 snap area 和子 fragment 里已经收集到的后代 snap area 一起向上挂。

真正的后代节点收集点就在 `FragmentBuilder::PropagateSnapAreas()`，它不是事后扫描 DOM，而是在每个 child fragment 进入父 fragment 时顺手完成的：

```cpp
GCedHeapVector<Member<Element>>& FragmentBuilder::EnsureSnapAreas() {
  if (!snap_areas_) {
    snap_areas_ = MakeGarbageCollected<GCedHeapVector<Member<Element>>>();
  }
  return *snap_areas_;
}

void FragmentBuilder::PropagateSnapAreas(const PhysicalFragment& child) {
  auto get_insertion_pos = [&](Element* snap_area) {
    auto& snap_areas = EnsureSnapAreas();
    const LayoutBox* new_box = snap_area->GetLayoutBox();
    if (!new_box) {
      return snap_areas.size();
    }
    for (wtf_size_t i = snap_areas.size(); i >= 1; i--) {
      const LayoutBox* existing_box = snap_areas.at(i - 1)->GetLayoutBox();
      if (existing_box && existing_box->IsBeforeInPreOrder(*new_box)) {
        return i;
      }
    }
    return 0u;
  };
  if (child.IsSnapArea()) {
    if (!To<PhysicalBoxFragment>(child).GetBreakToken()) {
      auto* snap_area = To<Element>(child.GetLayoutObject()->GetNode());
      EnsureSnapAreas().insert(get_insertion_pos(snap_area), snap_area);
    }
  }

  if (const auto* child_snap_areas = child.PropagatedSnapAreas()) {
    EnsureSnapAreas().InsertVector(get_insertion_pos(child_snap_areas->at(0)),
                                   *child_snap_areas);
  }

  if (child.IsSnapArea() && child.PropagatedSnapAreas()) {
    child.GetDocument().CountUse(WebFeature::kScrollSnapNestedSnapAreas);
  }
}
```

```cpp
const GCedHeapVector<Member<Element>>* SnapAreas() const {
  return propagated_data_ ? propagated_data_->snap_areas.Get() : nullptr;
}
const GCedHeapVector<Member<Element>>* PropagatedSnapAreas() const {
  return IsScrollContainer() ? nullptr : SnapAreas();
}
```

这一段里最关键的是 `PropagateSnapAreas(const PhysicalFragment& child)`，它做了三件事：

1. 如果 `child.IsSnapArea()`，并且它是最后一个 fragment，就把这个 child 自己插入当前 fragment 的 `snap_areas_`。
2. 如果 `child.PropagatedSnapAreas()` 非空，就把 child 已经收集好的后代 snap areas 一并插到当前 fragment 里。
3. 插入位置按 DOM 前序排序，保证后面 `SnapCoordinator` 读取时顺序稳定。

`PropagatedSnapAreas()` 只对非 scroll container 返回值，这个限制很重要：它让 scroll container 成为 snap area 传播链上的终点。注意这里说的是被传播的 child fragment——`PropagateSnapAreas(child)` 处理的就是这个 child，父 fragment 通过 `child.PropagatedSnapAreas()` 决定要不要把 child 收集到的 snap areas 并入自己。一个 child fragment 在布局过程中收集到的 snap areas（自己的以及从更深层后代冒泡上来的）只会在两种去向里选其一——

- 如果 child 不是 scroll container，这些 snap areas 通过 `PropagatedSnapAreas()` 返回出去，继续并入上层父 fragment。
- 如果 child 是 scroll container，`PropagatedSnapAreas()` 返回 `nullptr`，传播在此中断；child 收集到的这些 snap areas 由它自己通过 `SnapAreas()` 消费（`SnapCoordinator::UpdateSnapContainerData()` 会读取），**既不会把 child 自己、也不会把它收集到的后代集合再往更上层传**。

这与“每个 snap area 归属其最近祖先 scroll container”的规则一致：嵌套 scroll container 内部的 snap area 只属于内层 container，不应泄漏到外层 container 的 `SnapAreas()` 里。所以父 fragment 从一个 scroll-container child 那里拿到的始终是 `nullptr`，而不是重新遍历整棵子树。

这一点还能从 `FragmentBuilder` 的调用位置看出来：它是在处理 child fragment 的过程中调用 `PropagateSnapAreas(child)` 的，所以收集发生在布局树逐层合并 fragment 数据的时刻，而不是在 `SnapCoordinator` 里临时查找后代。

最终在 `SnapCoordinator::UpdateSnapContainerData()` 里，Blink 只是读取 `fragment.SnapAreas()`：

```cpp
for (auto& fragment : snap_container.PhysicalFragments()) {
  if (auto* snap_areas = fragment.SnapAreas()) {
    for (Element* snap_area : *snap_areas) {
      cc::SnapAreaData snap_area_data =
          CalculateSnapAreaData(*snap_area, snap_container);
      snap_container_data.AddSnapAreaData(snap_area_data);
    }
  }
}
```

这说明收集和消费是分离的：layout 阶段负责把后代节点整理成 fragment 上的 snap area 列表，snap coordinator 阶段只负责读取并计算最终数据。也因此，真正决定“一个后代是否能传到某个祖先容器”的代码，不在 target 归属逻辑里，而在 fragment 传播逻辑里。

对应测试也能证明这一点：

```cpp
unsigned SizeOfSnapAreas(const ContainerNode& node) {
  for (auto& fragment : node.GetLayoutBox()->PhysicalFragments()) {
    if (fragment.PropagatedSnapAreas()) {
      return 0u;
    }
    if (auto* snap_areas = fragment.SnapAreas()) {
      return snap_areas->size();
    }
  }
  return 0u;
}
```

这个测试 helper 明确区分了 `SnapAreas()` 和 `PropagatedSnapAreas()`：前者是当前 container 自己最终消费的 snap areas，后者是还要继续往上层传递的后代集合。

## snap 更新的入队与执行 — [local_frame_view.cc](third_party/blink/renderer/core/frame/local_frame_view.cc)

snap 更新不是在样式/布局变更的当场立即重算的，而是分成"入队"和"执行"两步：变更发生时只把对应的 scroll container 登记下来，等布局收尾时再统一重算。`LocalFrameView` 维护了两个集合：

- `pending_snap_updates_`：已登记、等待重算 `SnapContainerData` 的滚动容器。
- `pending_perform_snap_`：重算后数据确实变了、还需要执行 `SnapAfterLayout()`（真正把滚动位置吸到新 snap 点）的滚动容器。

两个集合都是 per-`LocalFrameView` 的，所以入队和执行都只涉及本 frame 自己的 snap container，不跨 frame 汇总。

### 入队：`EnqueueForSnapUpdateIfNeeded`

入队的真正闸门是 `PaintLayerScrollableArea::EnqueueForSnapUpdateIfNeeded`——它判断"这个滚动容器是否需要被重算"，需要才调 `LocalFrameView::AddPendingSnapUpdate` 把自己塞进 `pending_snap_updates_`：

```cpp
void PaintLayerScrollableArea::EnqueueForSnapUpdateIfNeeded() {
  auto* box = GetLayoutBox();
  // Not all PLSAs are scroll containers!
  if (!box->IsScrollContainer()) {
    return;
  }

  if (box->IsOverscrollAreaParent()) {
    // ::-internal-overscroll-area-parent has implicit snap areas and should
    // always be enqueued for pending snap updates.
    box->GetFrameView()->AddPendingSnapUpdate(this);
  } else {
    // Enqueue ourselves for a snap update if we have any snap-areas, or if we
    // currently have snap-data (and it needs to be cleared).
    for (const auto& fragment : box->PhysicalFragments()) {
      if (fragment.SnapAreas() || GetSnapContainerData()) {
        box->GetFrameView()->AddPendingSnapUpdate(this);
        break;
      }
    }
  }
}
```

判断逻辑很直接：非 scroll container 直接返回；overscroll-area-parent 因为有隐式 snap area 无条件入队；其余的只要 fragment 上有 `SnapAreas()`、或已有待清理的旧 `SnapContainerData`，就入队。也就是说，每个需要（重新）算 snap 的容器都会主动报到自己，不需要 `SnapCoordinator` 来扫整棵树。

`EnqueueForSnapUpdateIfNeeded` 有三类触发路径，对应布局、样式变更、CSS target 重绑定三种时机：

1. **布局完成时（主路径）**——NG 布局在每个 box 收尾时走 `BlockNode::FinishLayout`，链路如下：

   ```
   BlockNode::FinishLayout                                   // block_node.cc
     → CopyFragmentDataToLayoutBox
       → LayoutBox::UpdateAfterLayout    (仅最后一个 fragment) // layout_box.cc
         → PaintLayer::UpdateScrollingAfterLayout  (HasLayer 时)
           → PaintLayerScrollableArea::UpdateAfterLayout    // paint_layer_scrollable_area.cc
             → EnqueueForSnapUpdateIfNeeded
   ```

2. **样式变更时**——`LayoutBox::UpdateScrollSnapMappingAfterStyleChange`（layout_box.cc）：`scroll-snap-type` / `scroll-padding` 变化时，若该 box 已是 scroll container 且不需要重新布局，就直接 `EnqueueForSnapUpdateIfNeeded`；`scroll-snap-stop` / `scroll-margin` / `transform` 变化时，则找到 `ContainingScrollContainer()` 把那个容器入队。这条路径处理"样式变了但还没走到布局"的情况。

3. **CSS target 重绑定时**——`Element::SetTargetedSnapAreaIdsForSnapContainers()`（见上文 element.cc 一节）：从当前元素往上走，每命中一个 snap container 就 `AddPendingSnapUpdate` 一次。

对应地，滚动容器销毁时会调 `RemovePendingSnapUpdate` 把自己摘掉，避免悬挂引用。

### 执行：`ExecutePendingSnapUpdates`

入队把snap container塞进 `pending_snap_updates_`，执行则发生在 frame 生命周期 post-layout 阶段——主路径是 `LocalFrameView::RunStyleAndLayoutLifecyclePhases`：在 `UpdateStyleAndLayoutIfNeededRecursive()` 跑完布局之后调用一次，统一消费队列。（`Document::UpdateStyleAndLayout` 在 forced-layout 时也会调一次，但同样只刷本 frame 队列，不是常规渲染的主驱动点。）

```cpp
void LocalFrameView::ExecutePendingSnapUpdates() {
  if (pending_snap_updates_) {
    // Some scroll containers might be mid-scroll animation. Defer snapping
    // those containers until after the scroll animation is done.
    HeapHashSet<Member<PaintLayerScrollableArea>> deferred_updates;
    for (PaintLayerScrollableArea* scrollable_area : *pending_snap_updates_) {
      auto* snap_container = scrollable_area->GetLayoutBox();
      DCHECK(snap_container->IsScrollContainer());
      if (SnapCoordinator::UpdateSnapContainerData(*snap_container)) {
        if (!pending_perform_snap_) {
          pending_perform_snap_ = MakeGarbageCollected<
              GCedHeapHashSet<Member<PaintLayerScrollableArea>>>();
        }
        if (scrollable_area->HasRunningAnimation()) {
          deferred_updates.insert(scrollable_area);
        } else {
          pending_perform_snap_->insert(scrollable_area);
        }
      }
    }
    pending_snap_updates_->swap(deferred_updates);
  }

  if (pending_perform_snap_ && !ShouldDeferLayoutSnap()) {
    for (PaintLayerScrollableArea* scrollable_area : *pending_perform_snap_) {
      scrollable_area->SnapAfterLayout();
    }
    pending_perform_snap_->clear();
  }
}
```

这里分两步，对应两种不同的"延后"：

1. **遍历 `pending_snap_updates_` 重算数据**：对每个容器调 `UpdateSnapContainerData()`。返回 `true` 表示数据变了，需要进一步 `SnapAfterLayout`；返回 `false`（包括 `is_none` 早返回）则就此打住。注意这一趟**无论容器是否在动画中都会跑**——容器数据该更新就更新，延后的只是"吸到 snap 点"这一步。
2. **决定 `SnapAfterLayout` 的时机**：数据变了的容器进入 `pending_perform_snap_`，但真正调 `SnapAfterLayout()` 还要满足 `!ShouldDeferLayoutSnap()`。

两种延后机制容易混淆，区分开看：

- **`HasRunningAnimation()`（per-scroller）**：某个容器自己正在跑滚动动画时，它的 `SnapAfterLayout` 不在这一帧执行，而是通过 `pending_snap_updates_->swap(deferred_updates)` 把它**留回** `pending_snap_updates_`，下一帧再走一遍。数据已经更新好了，只是吸附动作等动画结束。
- **`ShouldDeferLayoutSnap()`（全局）**：只要当前有活跃的滚动手势（`widget->IsScrollGestureActive()`），就**整体**推迟所有 `pending_perform_snap_` 的 `SnapAfterLayout`，等用户松手。目的是避免布局触发的 re-snap 和 compositor 线程上的 snap 动画互相打架。

也就是说，`pending_snap_updates_` 会被 `swap` 保留下来等下一帧（针对动画中的容器），而 `pending_perform_snap_` 在条件不满足时则原样留着、等下次 `ExecutePendingSnapUpdates` 再尝试（针对用户还在滚动的场景）。这体现了 snap 计算和滚动动画/手势不是同步的：Blink 会先把容器数据算好，但把"真正吸过去"的动作推迟到动画结束或用户松手之后。

## SnapCoordinator：容器与 snap area 数据的计算 — [snap_coordinator.cc](third_party/blink/renderer/core/page/scrolling/snap_coordinator.cc)

```cpp
bool SnapCoordinator::UpdateSnapContainerData(LayoutBox& snap_container) {
  ScrollableArea* scrollable_area =
      ScrollableArea::GetForScrolling(&snap_container);
  const auto* old_snap_container_data = scrollable_area->GetSnapContainerData();
  auto snap_type = GetPhysicalSnapType(snap_container);

  if (snap_type.is_none) {
    if (old_snap_container_data) {
      snap_container.SetNeedsPaintPropertyUpdate();
      scrollable_area->SetScrollsnapchangingTargetIds(std::nullopt);
      scrollable_area->SetScrollsnapchangeTargetIds(std::nullopt);
      scrollable_area->SetSnappedQueryTargetIds(std::nullopt);
      if (RuntimeEnabledFeatures::CSSScrollSnapChangeEventEnabled()) {
        scrollable_area->EnqueueScrollSnapChangeEvent();
      }
      scrollable_area->SetSnapContainerData(std::nullopt);
    }
    return false;
  }
```

这里是 snap 的主控制器。第一步先拿旧数据，再把 `scroll-snap-type` 归一化成物理轴。若结果是 `none`，就清空旧状态并触发相应事件。

这段体现了实现上的一个关键原则：snap 不是只在“有 snap”时工作，Blink 也必须处理“从有 snap 变成没 snap”的反向路径，包括清理 target ids、更新 paint property、以及在需要时派发 snap change 事件。

过了 `is_none` 早返回之后，才是主分支——构建新的 `cc::SnapContainerData` 并填充容器侧几何，然后遍历 snap areas：

```cpp
cc::SnapContainerData snap_container_data(snap_type);
// 最大可滚动位置 + 当前 target snap area id（来自 CSS :target 或脚本）。
snap_container_data.set_max_position(...);
snap_container_data.set_targeted_area_id(
    scrollable_area->GetTargetedSnapAreaId());

// scrollport = padding box，再按 scroll-padding 向内收缩。
// 注意 scroll-padding 的百分比是相对 scrollport 对应维度解析，
// 而非普通 padding 那样相对 width，所以这里用 MinimumValueForLength。
PhysicalRect container_rect(snap_container.OverflowClipRect(PhysicalOffset()));
container_rect.ContractEdges(
    MinimumValueForLength(container_style->ScrollPaddingTop(), container_rect.Height()),
    MinimumValueForLength(container_style->ScrollPaddingRight(), container_rect.Width()),
    MinimumValueForLength(container_style->ScrollPaddingBottom(), container_rect.Height()),
    MinimumValueForLength(container_style->ScrollPaddingLeft(), container_rect.Width()));
snap_container_data.set_rect(gfx::RectF(container_rect));

// proximity 严格度下，按容器尺寸 × kProximityRatio 给出吸附范围。
if (snap_container_data.scroll_snap_type().strictness ==
    cc::SnapStrictness::kProximity) {
  PhysicalSize size = container_rect.size;
  size.Scale(kProximityRatio);
  snap_container_data.set_proximity_range(
      gfx::PointF(size.width.ToFloat(), size.height.ToFloat()));
}
```

容器数据准备好后，就开始逐个收集 snap area——这里正是 `fragment.SnapAreas()` 被消费的地方，也是 `CalculateSnapAreaData` 被调用的入口：

```cpp
cc::TargetSnapAreaElementIds new_target_ids;
const cc::TargetSnapAreaElementIds old_target_ids =
    old_snap_container_data
        ? old_snap_container_data->GetTargetSnapAreaElementIds()
        : cc::TargetSnapAreaElementIds();

if (snap_container.IsOverscrollAreaParent()) {
  AddOverscrollSnapAreas(snap_container, snap_container_data,
                         new_target_ids, old_target_ids);
} else {
  for (auto& fragment : snap_container.PhysicalFragments()) {
    if (auto* snap_areas = fragment.SnapAreas()) {
      for (Element* snap_area : *snap_areas) {
        cc::SnapAreaData snap_area_data =
            CalculateSnapAreaData(*snap_area, snap_container);
        // 旧 target id 只有在对应 snap area 仍存在时才保留。
        if (old_target_ids.x == snap_area_data.element_id)
          new_target_ids.x = old_target_ids.x;
        if (old_target_ids.y == snap_area_data.element_id)
          new_target_ids.y = old_target_ids.y;
        snap_container_data.AddSnapAreaData(snap_area_data);
      }
    }
  }
}
snap_container_data.SetTargetSnapAreaElementIds(new_target_ids);

// 仅当数据真的变化时才写入并请求 paint property 更新。
if (!old_snap_container_data ||
    *old_snap_container_data != snap_container_data) {
  snap_container.SetNeedsPaintPropertyUpdate();
  scrollable_area->SetSnapContainerData(snap_container_data);
  return true;
}
return false;
```

主分支里有几个值得注意的点：

- **容器几何与 area 几何是分开算的**：scroll-padding、proximity range 这些容器侧属性在这里一次性算好，snap area 的几何则交给 `CalculateSnapAreaData`。
- **target id 的保留是有条件的**：旧的 x/y target 只有在本次仍能找到对应 snap area 时才迁移到新数据里，否则丢弃——这处理了**当前 snap target 被移除”的场景。**
- **overscroll area parent 走单独路径**：`::internal-overscroll-area-parent` 伪元素会构造初始位置 + 内容两个特殊 snap area，不收集普通后代 snap area。
- **只在数据变化时写入**：`*old != new` 的判断避免无谓的 paint property 更新。

下面就看 `CalculateSnapAreaData` 怎么把单个 snap area 转成 `cc::SnapAreaData`——也就是上面循环体里每一行 `CalculateSnapAreaData(*snap_area, snap_container)` 的具体实现：

```cpp
cc::SnapAreaData SnapCoordinator::CalculateSnapAreaData(
    Element& snap_area,
    const LayoutBox& snap_container) {
  const ComputedStyle* container_style = snap_container.Style();
  const ComputedStyle* area_style = snap_area.GetComputedStyle();
  cc::SnapAreaData snap_area_data;

  const MapCoordinatesFlags mapping_mode =
      kTraverseDocumentBoundaries | kIgnoreScrollOffset;
  Vector<gfx::QuadF> quads;
  if (const LayoutBox* box = snap_area.GetLayoutBox()) {
    box->QuadsInAncestor(quads, &snap_container, mapping_mode);
  }

  PhysicalRect area_rect;
  for (const gfx::QuadF& quad : quads) {
    area_rect.UniteIfNonZero(PhysicalRect::EnclosingRect(quad.BoundingBox()));
  }

  PhysicalBoxStrut area_margin = area_style->ScrollMarginStrut();
  area_rect.Expand(area_margin);
  snap_area_data.rect = gfx::RectF(area_rect);
  snap_area_data.scroll_snap_align = GetPhysicalAlignment(
      *area_style, *container_style, area_rect, container_rect);
  snap_area_data.must_snap =
      (area_style->ScrollSnapStop() == EScrollSnapStop::kAlways);
```

这里做的是“把 DOM 几何和 CSS 规则变成 compositor 认识的数据”。它先求 snap area 的几何包围盒，再加上 `scroll-margin`，然后把逻辑方向的 `scroll-snap-align` 转成物理方向，最后计算 `scroll-snap-stop: always` 是否成立。

也就是说，`SnapCoordinator` 的职责不是做最终滚动，而是把“可 snap 的描述”标准化成一份统一的数据结构，供后续选择和对齐使用。

## scrollsnapchange 事件的构造 — [scrollable_area.cc](third_party/blink/renderer/core/scroll/scrollable_area.cc)

```cpp
void ScrollableArea::EnqueueScrollSnapChangeEvent() const {
  DCHECK(RuntimeEnabledFeatures::CSSScrollSnapChangeEventEnabled());
  Node* target_node = EventTargetNode();
  if (!target_node) {
    return;
  }
  Member<Node> block_target = GetSnapEventTargetAlongAxis(
      event_type_names::kScrollsnapchange, cc::SnapAxis::kBlock);
  Member<Node> inline_target = GetSnapEventTargetAlongAxis(
      event_type_names::kScrollsnapchange, cc::SnapAxis::kInline);
  target_node->GetDocument().EnqueueScrollSnapChangeEvent(
      target_node, block_target, inline_target);
}
```

这一层是从滚动容器拿到 target，再把事件转交给 `Document`。它本身不决定 snap 结果，只负责把计算好的 block / inline 目标节点整理出来。

## 事件入队与 per-frame 调度 — [document.cc](third_party/blink/renderer/core/dom/document.cc)

```cpp
void Document::EnqueueScrollSnapChangeEvent(Node* target,
                                            Member<Node>& block_target,
                                            Member<Node>& inline_target) {
  Event* scrollsnapchange_event = SnapEvent::Create(
      event_type_names::kScrollsnapchange,
      (target->IsDocumentNode() ? Event::Bubbles::kYes : Event::Bubbles::kNo),
      block_target, inline_target);
  scrollsnapchange_event->SetTarget(target);
  scripted_animation_controller_->EnqueuePerFrameEvent(scrollsnapchange_event);
}
```

`Document` 只做事件对象创建和入队，不参与 snap 选择。这里能看出 snap 事件是 per-frame event，也就是最后会被脚本动画控制器统一调度。

## compositor 侧的数据模型 — [cc/input/scroll_snap_data.h](cc/input/scroll_snap_data.h)

```cpp
struct ScrollSnapType {
  ScrollSnapType()
      : is_none(true),
        axis(SnapAxis::kBoth),
        strictness(SnapStrictness::kProximity) {}

  bool is_none;
  SnapAxis axis;
  SnapStrictness strictness;
};

struct ScrollSnapAlign {
  ScrollSnapAlign()
      : alignment_block(SnapAlignment::kNone),
        alignment_inline(SnapAlignment::kNone) {}

  SnapAlignment alignment_block;
  SnapAlignment alignment_inline;
};
```

```cpp
struct SnapAreaData {
  ScrollSnapAlign scroll_snap_align;
  gfx::RectF rect;
  bool must_snap = false;
  bool has_focus_within = false;
  ElementId element_id;
};
```

前面那些都是容器内部或单个 area 的描述，真正把它们装起来、并在 cc 层承担 snap 计算的是 `SnapContainerData`（`cc/input/scroll_snap_data.h`）。它在本文里反复出现却一直没展开，这里补齐它的成员：

```cpp
class CC_EXPORT SnapContainerData {
 public:
  // 构造/比较/赋值等略；operator== 逐字段比较下面所有成员。

  // —— 对外接口 ——
  SnapPositionData FindSnapPosition(const SnapSelectionStrategy& strategy) const;
  const TargetSnapAreaElementIds& GetTargetSnapAreaElementIds() const;
  bool SetTargetSnapAreaElementIds(TargetSnapAreaElementIds ids);  // 返回是否变化
  void AddSnapAreaData(SnapAreaData snap_area_data);
  void UpdateSnapAreaFocus(size_t index, bool has_focus_within);
  void set_scroll_snap_type(ScrollSnapType type);
  void set_rect(const gfx::RectF& rect);                 // snapport
  void set_max_position(gfx::PointF position);           // 最大可滚动位置
  void set_proximity_range(const gfx::PointF& range);    // proximity 吸附范围
  void set_targeted_area_id(const std::optional<ElementId>& id);
  void set_has_horizontal_writing_mode(bool);

 private:
  // snap-type：是否 snap / 哪个轴 / 严格度。is_none 时整份数据视为无效（见 SnapCoordinator 的早返回）。
  ScrollSnapType scroll_snap_type_;

  // snapport：Blink 下发的容器 rect（padding box 再按 scroll-padding 收缩）。
  // GetSnapSearchResult 里 area.rect - rect_ 算对齐 offset 就用它。
  gfx::RectF rect_;

  // 容器最大可滚动位置，Blink 侧 set_max_position 下发。
  gfx::PointF max_position_;

  // proximity 严格度下的吸附范围（容器尺寸 × kProximityRatio）。
  gfx::PointF proximity_range_;

  // 本容器所有 snap area 的列表，即 layout 阶段收集、SnapCoordinator 读取的那份。
  std::vector<SnapAreaData> snap_area_list_;

  // 当前 snap 在哪个 area 上（x / y 各一个）。compositor 写回、Blink 保留，驱动事件。
  TargetSnapAreaElementIds target_snap_area_element_ids_;

  // 浏览器控件显隐期间对 snapport 高度的临时修正，只在 FindSnapPosition 执行期间有效。
  double snapport_height_adjustment_ = 0;

  // 容器是否横向 writing mode，影响逻辑轴→物理轴的换算。
  bool has_horizontal_writing_mode_ = true;

  // :target 元素所在 snap area 的 ElementId（单向下发，cc 只读不回写）。
  std::optional<ElementId> targeted_area_id_;

  // 由 UpdateExtremes() 维护的各轴 snap offset 上下界，加速搜索。
  std::optional<float> min_snap_offset_x_;
  std::optional<float> max_snap_offset_x_;
  std::optional<float> min_snap_offset_y_;
  std::optional<float> max_snap_offset_y_;
};
```

把字段和前文对应起来看：

- `scroll_snap_type_` / `rect_` / `max_position_` / `proximity_range_`：就是 `SnapCoordinator::UpdateSnapContainerData` 主分支里 `set_*` 的那几个，容器侧几何一次性算好。
- `snap_area_list_`：`AddSnapAreaData(snap_area_data)` 一条条塞进来，元素来自 `fragment.SnapAreas()` → `CalculateSnapAreaData`。
- `target_snap_area_element_ids_`：上文强调的"cc 回写"字段，`FindSnapPosition` 产出后由 `SetTargetSnapAreaElementIds` 写回，Blink 侧只做"area 仍存在才保留"的迁移。
- `targeted_area_id_`：`:target` 的单向下发字段，`FindSnapPosition` 里给它优先级。
- `has_horizontal_writing_mode_`：和 `CalculateSnapAreaData` 里 `GetPhysicalAlignment` 的逻辑轴→物理轴换算对齐。
- `snapport_height_adjustment_`：这就是 `FindSnapPositionWithViewportAdjustment(strategy, snapport_height_adjustment)` 多带的那一参，对应浏览器控件伸缩 snapport 的场景。

顺带补一个上面没贴、但被两个 target 字段都用到的辅助结构 `TargetSnapAreaElementIds`（`scroll_snap_data.h`）：

```cpp
struct TargetSnapAreaElementIds {
  ElementId x;
  ElementId y;
};
```

它就是 `target_snap_area_element_ids_` 的类型——每轴各一个 `ElementId`，这正是后文"为什么常常只设一个 elementId"那一段讨论的结构基础。

`SnapContainerData` 上还有两个容易混淆的"target"字段，分清它们很关键：

- `targeted_area_id_`（`std::optional<ElementId>`，单个）：CSS `:target` 元素所在 snap area 的 id。由 `ScrollableArea::SetTargetedSnapAreaId` 设置（见上文 element.cc 一节），在 `FindSnapPosition` 里让匹配的候选优先胜出——即"导航到 `#fragment` 后 `:target` 在 snap 时优先"。
- `target_snap_area_element_ids_`（`TargetSnapAreaElementIds{ x, y }`，每轴一个）：**用户当前 snap 在哪个 snap area 上**，每轴各记一个。权威设置点是 compositor——`InputHandler` 在 snap 完成（scroll end / snap fling）时把 `FindSnapPosition` 的结果写回（input_handler.cc），再经 `ScrollingCoordinator::DidCompositorScroll`（scrolling_coordinator.cc）回传 Blink，用于派发 `scrollsnapchange`/`scrollsnapchanging` 事件目标。`SnapCoordinator::UpdateSnapContainerData` 里那处只是**保留**旧值（且只保留仍存在的 area），不计算新目标。

为什么 `target_snap_area_element_ids_` 常常"只设一个 elementId"：结构能存两个（x 和 y），但 snap 是按轴进行的——单轴 snap-type 只有一轴会有值；即使 `both`，一次手势通常只滚一个轴，`FindSnapPosition` 只更新那个轴，另一轴 retain 或为空；另外同一元素可同时是两轴目标（x == y），看起来也像"一个"。所以这是 snap 行为通常单轴的结果，不是结构限制。

如果只记一个点：Blink 的大部分复杂性都在“把 CSS 语义和布局几何转换成这些简单结构”，真正的 snap 选择逻辑是在后续基于这些数据完成的。

# cc 层：snap 选择算法

数据结构准备好之后，"该 snap 到哪个 area、哪个 offset"由 cc 层的 `SnapContainerData::FindSnapPosition` 决定。入口在 [cc/input/scroll_snap_data.cc](cc/input/scroll_snap_data.cc)，策略类在 [cc/input/snap_selection_strategy.h](cc/input/snap_selection_strategy.h)。

## SnapSelectionStrategy：把"滚动场景"抽象成策略

不同滚动场景对 snap 的要求不同（方向性滚动只接受前进方向的 snap 点；程序化滚动以终点为基准），cc 用策略模式封装这种差异。基类 `SnapSelectionStrategy` 提供关键钩子：

- `base_position()`：搜索距离的基准点。
- `intended_position()`：预期目标位置，用于方向判断。
- `ShouldPrioritizeSnapTargets()`：是否优先沿用当前已 snap 的目标（snapAfterLayout有明确目标"的场景为 true）。
- `IsValidSnapArea` / `IsValidSnapPosition`：过滤不该考虑的候选。

两个主要子类：

- `DirectionStrategy`（`snap_selection_strategy.h`）：手势滑动、键盘方向键、惯性 fling。`intended_position = current + step`，`IsValidSnapPosition` 只接受与 `step` 同向的 snap 点（往右滚只接受更靠右的点）。
- `EndPositionStrategy`（`snap_selection_strategy.h`）：`scrollTo` / `scrollIntoView` 等程序化滚动。`base_position == intended_position == current`，`ShouldPrioritizeSnapTargets()` 返回 true，`PickBestResult` 让覆盖型 snap（area 比 snapport 大）优先于最近 snap。

工厂方法 `CreateForDirection / CreateForEndPosition / CreateForDisplacement / CreateForPageScroll / CreateForTargetElement` 分别对应不同滚动来源。

## FindSnapPosition：主流程

```cpp
SnapPositionData SnapContainerData::FindSnapPosition(
    const SnapSelectionStrategy& strategy) const {
  // 1. 按 snap-type 决定本趟在 x / y / 两轴搜索
  // 2. 在主轴上 FindClosestValidArea 选出候选
  // 3. 用主轴候选的 cross-axis 可见范围过滤另一轴候选（互可见性）
  // 4. 两轴都无结果时回退 FindSnapPositionForMutualSnap：
  //    找同一个 area 在两轴上同时有效
  // 5. 保留旧 target_element_ids（仅当对应 area 仍存在），
  //    避免单轴搜索时另一轴的 snap 状态被误清
  // 6. 填充 result.position + result.target_element_ids
}
```

`FindClosestValidArea` 是单轴搜索核心，对每个 area 依次做：策略有效性检查 → `GetSnapSearchResult` 算对齐 offset → 互可见性检查 → 策略位置有效性 → proximity 范围过滤 → 按距离 + 偏好排序。几个值得记的点：

- **对齐 offset 计算**（`GetSnapSearchResult`）：`start = area.rect.x() - rect.x()`、`center = area.Center - rect.Center`、`end = area.rect.right() - rect.right()`，这就是 `SnapAreaData.rect`（含 scroll-margin）和 `SnapContainerData.rect`（含 scroll-padding）最终被消费的地方。
- **互可见性**（`IsMutualVisible`）：两轴的 snap 点必须落在彼此的可见范围内，避免选出一个 x 偏移导致 y 方向目标不可见（或反之）的组合。
- **优先级**：`must_snap`（`scroll-snap-stop: always`）的点会被特殊处理——一旦选中，不能跳过它去更远的点；`has_focus_within` 和 `targeted_area_id_`（CSS `:target`）的候选享有优先级（见 scroll_snap_data.cc）。
- **`ShouldPrioritizeSnapTargets()` 时沿旧轴**（scroll_snap_data.cc）：若之前只在 x 轴 snap 过（y 为空），程序化滚动优先继续沿 x 轴 snap，而不是重新选轴。

## 谁来调用 FindSnapPosition

`FindSnapPosition` 有两类调用方，都跑同一套算法：

1. **compositor（主路径）**：`cc::InputHandler` 在 `ScrollEnd` / snap fling 结束时，用 `CreateSnapStrategy` 造策略，调 `FindSnapPositionWithViewportAdjustment`，拿到 `SnapPositionData` 后做动画把滚动位置吸过去，并把 `snap.target_element_ids` 写回 `scroll_node->snap_container_data`（见下节同步）。
2. **Blink 自己**：`PaintLayerScrollableArea::GetSnapPositionAndSetTarget`（paint_layer_scrollable_area.cc）在程序化滚动（`scrollTo`/`scrollIntoView`/翻页）路径里直接对本地持有的 `SnapContainerData` 调 `FindSnapPosition`，就地算出 snap 点并把 `target_element_ids` 写回本地数据。所以 cc 的算法不只跑在 compositor 线程，Blink 主线程也会调。

# Blink ↔ cc 的数据同步

snap 是个跨线程、跨阶段的协作：Blink 算好"静态描述"（哪些 area、几何、对齐、`:target`）推给 cc；cc 在滚动时算出"动态结果"（snap 到哪、当前 target 是谁）再回传 Blink 派发事件。两个 target 字段的同步方向正好相反，分清这一点是理解整条链路的关键。

## Blink → cc：SnapContainerData 下发

```
SnapCoordinator::UpdateSnapContainerData                      // snap_coordinator.cc
  → ScrollableArea::SetSnapContainerData(data)                // 存进 PLSA RareData
    → [paint property tree build]                             // paint_property_tree_builder.cc
        state.snap_container_data = *GetSnapContainerData()
        → UpdateScroll → ScrollPaintPropertyNode
          → [commit] → cc::ScrollNode::snap_container_data    // cc/trees/scroll_node.h
```

Blink 端 `SnapContainerData` 存在 `PaintLayerScrollableArea` 的 RareData 上；paint property tree 构建时（`paint_property_tree_builder.cc`）把它读出来挂到 `ScrollPaintPropertyNode`；随 commit 序列化进 `cc::ScrollNode::snap_container_data`。compositor 的 `InputHandler` 就是从 `scroll_node->snap_container_data` 取数据调 `FindSnapPosition` 的（input_handler.cc）。

注意这条链路下发的是**整份容器数据**（含 `snap_area_list_`、`targeted_area_id_`、`target_snap_area_element_ids_` 的当前值）。其中：

- `targeted_area_id_`（`:target`）是 **Blink 单向下发**给 cc 的——cc 只读它做优先级判断，不回写。
- `target_snap_area_element_ids_` 则会 **cc 回写**，见下。

## cc → Blink：snap 结果回传与事件派发

compositor 在滚动/snap 过程中产出两类信息要回传：滚动 offset，以及新的 snap target ids。

```
[compositor 线程]
InputHandler::ScrollEnd / SnapFling 完成
  → data.FindSnapPosition(strategy) → SnapPositionData
  → data.SetTargetSnapAreaElementIds(snap.target_element_ids)   // 写回 cc 侧 ScrollNode
  → updated_snapped_elements_[element_id] = target_ids           // 暂存
  → [commit] ProcessCommitDeltas → CompositorCommitData
    → 携带每 scroller 的 snap_target_ids 回主线程

[主线程]
ScrollingCoordinator::DidCompositorScroll(element_id, offset, snap_target_ids)  // scrolling_coordinator.cc
  → scrollable->DidCompositorScroll(offset)            // 更新 ScrollableArea 的滚动位置
  → scrollable->SetTargetSnapAreaElementIds(ids)       // 把 cc 算出的 target 写回 Blink 侧 SnapContainerData
                                                         // (PLSA) 若变了 → SetNeedsPaintPropertyUpdate
```

回传的 `target_snap_area_element_ids_` 进入 Blink 后，驱动两个事件（`scrollable_area.cc`）：

- **`scrollsnapchanging`**（滚动过程中）：程序化滚动发起时（`SetScrollOffset` 路径，scrollable_area.cc），用当前 `GetTargetSnapAreaElementIds()` 调 `UpdateScrollSnapChangingTargetsAndEnqueueScrollSnapChanging`，提前告知"将要 snap 到谁"。
- **`scrollsnapchange`**（snap 落定）：滚动结束（`OnScrollFinished`，scrollable_area.cc）或程序化 snap 完成（多处 `UpdateSnappedTargetsAndEnqueueScrollSnapChange`）时，比较新旧 target ids，变了就 `EnqueueScrollSnapChangeEvent`。

事件的 target 节点由 `GetSnapEventTargetAlongAxis` 按 block/inline 轴从 `target_snap_area_element_ids_` 解析出 `ElementId`，再交 `Document` 入队（见上文 scrollable_area.cc / document.cc 两节）。

## 两个方向的对称性

| 字段 | Blink → cc | cc → Blink |
|------|------------|------------|
| `snap_area_list_` / 容器几何 / `targeted_area_id_` | 整份下发（commit） | 不回写 |
| `target_snap_area_element_ids_` | 下发当前值作为初值 | compositor snap 后回写新值，驱动事件 |

所以 `SnapCoordinator::UpdateSnapContainerData` 里对 `target_snap_area_element_ids_` 的"保留"逻辑就有了双重意义：Blink 重算容器数据时若把 cc 已算出的当前 target 丢掉，下发后 cc 就会以为"没有当前 snap 目标"，导致 `ShouldPrioritizeSnapTargets` 等逻辑失去基准——所以必须只在 area 仍存在时保留旧 target，area 消失时才允许丢弃。

# 你读代码时的推荐顺序

1. 先看 [element.cc](third_party/blink/renderer/core/dom/element.cc) 里的归属逻辑，理解 snap area 怎么被挂到容器上。
2. 再看 [local_frame_view.cc](third_party/blink/renderer/core/frame/local_frame_view.cc) 里的更新入口，理解什么时候会重算。
3. 接着看 [snap_coordinator.cc](third_party/blink/renderer/core/page/scrolling/snap_coordinator.cc) 的 `UpdateSnapContainerData()` 和 `CalculateSnapAreaData()`。
4. 看 [scrollable_area.cc](third_party/blink/renderer/core/scroll/scrollable_area.cc) 和 [document.cc](third_party/blink/renderer/core/dom/document.cc) 的事件派发。
5. 然后看 cc 层：[snap_selection_strategy.h](cc/input/snap_selection_strategy.h) 的策略类 + [scroll_snap_data.cc](cc/input/scroll_snap_data.cc) 的 `FindSnapPosition` / `FindClosestValidArea`，理解 snap 选择算法本身。
6. 最后看同步链路：[paint_property_tree_builder.cc](third_party/blink/renderer/core/paint/paint_property_tree_builder.cc)（Blink→cc 下发）、[input_handler.cc](cc/input/input_handler.cc) 的 `ScrollEnd` / `ProcessCommitDeltas`（cc 算 snap 并回写）、[scrolling_coordinator.cc](third_party/blink/renderer/core/page/scrolling/scrolling_coordinator.cc) 的 `DidCompositorScroll`（cc→Blink 回传），把整条跨线程闭环串起来。

# 相关测试

如果想验证理解是否正确，优先看 [third_party/blink/renderer/core/page/scrolling/snap_coordinator_test.cc](third_party/blink/renderer/core/page/scrolling/snap_coordinator_test.cc)。里面最有代表性的几类测试是：

- 基础 snap area 收集。
- 容器/嵌套容器的归属关系。
- `scroll-margin`、RTL、纵向 writing mode 的坐标换算。
- 当前 snap target 被移除、或者新 snap area 加入时的保留行为。

这份文档的目的不是替代源码，而是给你一个“先读哪一层、为什么这一层重要”的地图。
