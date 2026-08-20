
> **前置阅读**：建议先看 `blink_scroll_snap_impl.md` 的 "cc 层：snap 选择算法" 一节，了解 `FindSnapPosition` 的主流程。本文档聚焦 **`SnapSelectionStrategy`**——那个决定"哪些 snap 候选算数、最终选哪个"的策略对象。

# 一、为什么需要它

`SnapContainerData::FindSnapPosition` 本身是一套通用的搜索算法：遍历 `snap_area_list_`、算对齐 offset、检查互可见性、按距离排序。但"该把哪个候选筛掉、该偏好哪个"取决于**这次滚动是怎么产生的**：

- 按方向键往下滚：只该考虑更靠下的 snap 点，不该 snap 回头。
- `scrollTo(500)`：该以 500 为基准找最近的 snap 点，方向无所谓。
- 翻页：偏好"滚大约一页就停下"的 snap 点，别一下滚到三页外。
- 布局变化后重新 snap：优先沿用之前 snap 到的目标，别乱跳。

如果把这些判断硬编码进 `FindSnapPosition`，算法会被各种 `if` 撕碎。Chromium 用**策略模式**把这些差异抽出来：`FindSnapPosition` 只调策略对象提供的钩子（"这个 offset 合法吗？""这个位置是偏好吗？""覆盖型和最近型选哪个？"），具体规则由策略子类决定。

# 二、类结构

定义在 [cc/input/snap_selection_strategy.h](cc/input/snap_selection_strategy.h)，实现 in [cc/input/snap_selection_strategy.cc](cc/input/snap_selection_strategy.cc)。

```
SnapSelectionStrategy            (抽象基类，cc/input/snap_selection_strategy.h)
  ├── EndPositionStrategy        (终点位置滚动)
  └── DirectionStrategy          (方向滚动，含 displacement / page scroll)
```

只有两个具体子类。五个工厂方法里，有三个（`CreateForDirection` / `CreateForDisplacement` / `CreateForPageScroll`）都造 `DirectionStrategy`，只是参数不同；`CreateForEndPosition` / `CreateForTargetElement` 造 `EndPositionStrategy`。

两个相关枚举：

```cpp
enum class SnapStopAlwaysFilter { kIgnore, kRequire };      // 是否只考虑 scroll-snap-stop:always 的 area
enum class SnapTargetsPrioritization { kIgnore, kRequire }; // 是否优先沿用已有 snap target
```

# 三、基类的钩子

`FindSnapPosition` / `FindClosestValidArea` 通过这些虚函数与策略交互：

| 钩子 | 作用 | 基类默认 |
|------|------|----------|
| `ShouldSnapOnX/Y()` | 本趟是否在该轴搜索 | 纯虚 |
| `intended_position()` | 预期终点（方向判断用） | 纯虚 |
| `base_position()` | 距离基准点（"离谁最近"） | 纯虚 |
| `IsPreferredSnapPosition(axis, pos)` | 该位置是否是偏好候选 | 纯虚 |
| `IsValidSnapPosition(axis, pos)` | 该 offset 是否合法 | 纯虚 |
| `IsValidSnapArea(axis, area)` | 该 area 是否合法 | 按 alignment 过滤（inline→X, block→Y）|
| `HasIntendedDirection()` | 是否有明确方向 | `true` |
| `ShouldPrioritizeSnapTargets()` | 是否优先沿用当前 snap target | `false` |
| `ShouldRespectSnapStop()` | 是否尊重 `scroll-snap-stop: always` | `false` |
| `PickBestResult(closest, covering)` | 最近候选 vs 覆盖候选选哪个 | 纯虚 |
| `UsingFractionalOffsets()` | 当前 offset 是否小数像素 | `false` |
| `Clone()` | 复制（用于 commit 回传） | 纯虚 |

**"covering" 是什么**：当一个 snap area 比 snapport 还大时，"snap 到它的 start/center/end"反而不如"就停在原本想停的位置让它盖满视口"。`FindSnapPosition` 会单独算一个 `covering` 候选（用 `intended_position` 作为 offset），`PickBestResult` 决定到底用 `closest`（最近 snap 点）还是 `covering`。

`IsValidSnapArea` 的基类实现值得记：

```cpp
bool SnapSelectionStrategy::IsValidSnapArea(SearchAxis axis,
                                            const SnapAreaData& area) const {
  return axis == SearchAxis::kX
             ? area.scroll_snap_align.alignment_inline != SnapAlignment::kNone
             : area.scroll_snap_align.alignment_block != SnapAlignment::kNone;
}
```

即一个 area 只在它**声明了对齐**的轴上才是合法候选——`scroll-snap-align: start none` 的 area 在 Y 轴搜索时直接被过滤。

# 四、EndPositionStrategy：终点位置滚动

适用场景（注释里列的）：

- 不带惯性的拖拽手势松手
- 拖动滚动条 thumb
- `scrollTo()` 等程序化 API
- Tab 切焦点
- 锚点导航（`#fragment`）
- Home/End 键

这类滚动的共同点：**有一个明确的终点位置，方向不重要**。策略要做的就是"在终点附近找最近的 snap 点，如果终点本身就让某个 area 盖满视口，那就停在终点"。

```cpp
gfx::PointF EndPositionStrategy::intended_position() const {
  return current_position_;        // 终点 = 当前位置（调用方已把 end_position 传进来当 current）
}
gfx::PointF EndPositionStrategy::base_position() const {
  return current_position_;        // 距离基准也是终点本身
}
bool EndPositionStrategy::IsPreferredSnapPosition(SearchAxis, float) const {
  return true;                     // 所有合法位置都算"偏好"，不额外加权
}
bool EndPositionStrategy::IsValidSnapPosition(SearchAxis axis, float) const {
  return (scrolled_x_ && axis == SearchAxis::kX) ||
         (scrolled_y_ && axis == SearchAxis::kY);  // 只在真正滚过的轴上有效
}
bool EndPositionStrategy::HasIntendedDirection() const {
  return false;                    // 无方向
}
const std::optional<SnapSearchResult>& EndPositionStrategy::PickBestResult(
    const std::optional<SnapSearchResult>& closest,
    const std::optional<SnapSearchResult>& covering) const {
  return covering.has_value() ? covering : closest;  // 覆盖型优先
}
```

几个要点：

- `intended_position == base_position == current_position`：调用 `CreateForEndPosition(end_position, ...)` 时，调用方（`ScrollableArea::SnapForEndPosition`）已经把 `end_position` 作为 `current_position` 传进来了，所以这里"当前位置"就是终点。
- `ShouldSnapOnX/Y` 由 `scrolled_x_/scrolled_y_` 决定：只有这次手势确实滚过的轴才参与搜索。
- `PickBestResult` 让 `covering` 胜出——如果终点位置已经让某个大 area 盖满视口，就尊重它，不再强行 snap 到边的对齐点。
- `ShouldPrioritizeSnapTargets()` 由 `snap_targets_prioritization_` 决定，默认 `kIgnore`（不沿用旧 target）；但 `CreateForTargetElement` 会传 `kRequire`。

# 五、DirectionStrategy：方向滚动

适用场景：

- 方向键
- 被解释为固定步长的滑动手势
- fling（惯性滚动）
- `scrollBy()` 等程序化 API
- PgUp/PgDn

这类滚动的共同点：**有一个方向（step 向量），snap 点必须在该方向上**。

```cpp
gfx::PointF DirectionStrategy::intended_position() const {
  return current_position_ + step_;        // 预期终点 = 当前 + 步长
}
gfx::PointF DirectionStrategy::base_position() const {
  return preferred_step_ == StepPreference::kDirection
             ? current_position_            // 只偏好方向：以当前位置为基准，停在第 一个 snap 点
             : current_position_ + step_;   // 偏好步长：以终点为基准，找离步长最近的
}
bool DirectionStrategy::IsValidSnapPosition(SearchAxis axis, float position) const {
  // delta 必须与 step 同号（往右滚只接受更靠右的点）
  if (axis == SearchAxis::kX) {
    float delta = position - current_position_.x();
    if (!use_fractional_offsets_) delta = delta > 0 ? std::floor(delta) : std::ceil(delta);
    return (step_.x() > 0 && delta > 0) || (step_.x() < 0 && delta < 0);
  }
  // Y 轴同理
}
bool DirectionStrategy::IsValidSnapArea(SearchAxis axis, const SnapAreaData& area) const {
  return SnapSelectionStrategy::IsValidSnapArea(axis, area) &&
         (snap_stop_always_filter_ == SnapStopAlwaysFilter::kIgnore || area.must_snap);
}
bool DirectionStrategy::ShouldRespectSnapStop() const {
  return true;                            // 方向滚动尊重 scroll-snap-stop: always
}
```

`IsValidSnapPosition` 里那行 `floor/ceil` 截断值得注意：非小数滚动时，当前已 snap area 的 offset（小数）可能不等于当前 scroll offset（整数），不截断会导致"明明已经 snap 在这了，却被当成合法候选再 snap 一次"。截断后 `|delta| < 1` 的位置被忽略，避免抖动。

## StepPreference：方向优先 vs 步长优先

`DirectionStrategy` 内部有个 `StepPreference` 枚举，是理解它行为的关键：

```cpp
enum class StepPreference {
  kDirection,  // 只看方向，停在该方向第一个 snap 点
  kDistance    // 看步长，偏好离步长距离最近的 snap 点（可跳过近处 snap 点）
};
```

- `kDirection`（`CreateForDirection` 用）：`base_position = current_position_`。意思是"从当前位置出发，往 step 方向走，遇到第一个 snap 点就停"。适合方向键——按一下方向键，滚到下一个 snap 点。
- `kDistance`（`CreateForDisplacement` / `CreateForPageScroll` 用）：`base_position = current + step`。意思是"我想滚大概 step 这么远，找离这个距离最近的 snap 点"。适合 fling 和翻页——一次滚很远，不该停在刚出发就遇到的第一个 snap 点。

## preferred_min/max_displacement：翻页的位移约束

`IsPreferredSnapPosition` 用 `preferred_min_displacement_` / `preferred_max_displacement_` 判断一个 snap 点是否"偏好"：

```cpp
bool DirectionStrategy::IsPreferredSnapPosition(SearchAxis axis, float position) const {
  float delta = position - current_position_.y();   // 以 Y 为例
  return std::abs(delta) >= std::abs(preferred_min_displacement_.y()) &&
         std::abs(delta) <= std::abs(preferred_max_displacement_.y());
}
```

只有 `CreateForPageScroll` 会设这两个值（用 `ScrollUtils::CalculateMinPageSnap` / `CalculateMaxPageSnap`），让"滚大约一页、但别超过一页太多"的 snap 点成为偏好候选。`CreateForDirection` 和 `CreateForDisplacement` 设为 0 / MAX（无约束）。

## PickBestResult：方向滚动的覆盖判断

```cpp
const std::optional<SnapSearchResult>& DirectionStrategy::PickBestResult(
    const std::optional<SnapSearchResult>& closest,
    const std::optional<SnapSearchResult>& covering) const {
  if (!closest.has_value()) return covering;
  if (!covering.has_value()) return closest;
  if (covering->element_id() == closest->element_id()) return covering;  // 同一个 area，covering 保终点
  if (preferred_step_ == StepPreference::kDirection) {
    // kDirection：若 closest 比 covering 更靠近当前位置（沿方向更近），选 closest
    if ((step_.x() > 0 || step_.y() > 0) &&
        closest.value().snap_offset() < covering.value().snap_offset())
      return closest;
    if ((step_.x() < 0 || step_.y() < 0) &&
        closest.value().snap_offset() > covering.value().snap_offset())
      return closest;
  }
  return covering;
}
```

逻辑：同 area 时 covering 胜（保住预期终点）；`kDirection` 模式下若 `closest` 比 `covering` 更近（沿方向没越过终点），选 `closest`（停在该方向第一个 snap 点）；否则 `covering` 胜。

# 六、五个工厂方法与滚动场景映射

| 工厂方法 | 造的策略 | 关键参数 | 谁用 |
|----------|----------|----------|------|
| `CreateForEndPosition` | `EndPositionStrategy`(kIgnore) | end_position, scrolled_x/y | `scrollTo`/`scrollIntoView` 终点、Tab 焦点、锚点、Home/End |
| `CreateForDirection` | `DirectionStrategy`(kDirection) | step，min=max=0 | 方向键、`scrollBy` 小步、compositor 滚动条非翻页交互 |
| `CreateForDisplacement` | `DirectionStrategy`(kDistance) | displacement，min=0/max=MAX | `scrollBy` 大位移、fling、`scrollIntoView` 位移 |
| `CreateForPageScroll` | `DirectionStrategy`(kDistance) | page_size 算 min/max | PgUp/PgDn、滚动条翻页 |
| `CreateForTargetElement` | `EndPositionStrategy`(kRequire) | current_position | 布局后 re-snap（`SnapAfterLayout`） |

`CreateForTargetElement` 比较特殊：它造的是 `EndPositionStrategy`，但 `scrolled_x=scrolled_y=true`（两轴都搜）且 `SnapTargetsPrioritization::kRequire`（`ShouldPrioritizeSnapTargets` 返回 true）。`SnapAfterLayout` 用它——布局变化后优先吸回原来的 snap target，找不到再退而求最近。

## Blink 侧调用点

`ScrollableArea` 把滚动场景分派到不同策略（[scrollable_area.cc](third_party/blink/renderer/core/scroll/scrollable_area.cc)）：

```cpp
// scrollTo / scrollIntoView 终点 → EndPosition
ScrollableArea::SnapForEndPosition → CreateForEndPosition

// 方向键 → Direction(kDirection)
ScrollableArea::SnapForDirection → CreateForDirection

// PgUp/PgDn → Direction(kDistance, page min/max)
ScrollableArea::SnapForPageScroll → CreateForPageScroll

// Home/End → 走 EndPosition（绝对滚动）
ScrollableArea::SnapForDocumentScroll → SnapForEndPosition → CreateForEndPosition

// 布局后 re-snap → TargetElement
ScrollableArea::SnapAfterLayout → CreateForTargetElement
```

`scrollBy` / `scrollIntoView` 的位移滚动走 `CreateForDisplacement`（[local_dom_window.cc](third_party/blink/renderer/core/frame/local_dom_window.cc)、[element.cc](../../../third_party/blink/renderer/core/dom/element.cc.md)）。

## cc compositor 侧调用点

`InputHandler` 在 compositor 线程也会造策略（[input_handler.cc](cc/input/input_handler.cc)）：

- `AdjustScrollDeltaForScrollbarSnap`（滚动条交互）：`kScrollByPage` → `CreateForPageScroll`；其余 → `CreateForDirection`。
- `GetSnapFlingInfoAndSetAnimatingSnapTarget`（fling）：`CreateForDisplacement`。
- `ScrollEnd`：`CreateSnapStrategy` 根据 `last_scroll_state` 重建策略，并保留在 `snap_strategy_` 里，随 commit `Clone()` 回传主线程（见 `blink_scroll_snap_impl.md` 同步一节）。

# 七、intended_position 与 base_position

这两个是策略对象里最容易混、但也最关键的钩子。理解了它们，`FindSnapPosition` 里"为什么方向键停在第一个 snap 点、fling 停在远处"就一目了然。

## 三个位置的概念

- **`current_position`**：容器**当前的**滚动位置（基类成员，调用时传入）。
- **`intended_position`**："如果没有 snap 干预，这次滚动**会停在哪儿**"——即自然落点。
- **`base_position`**："算'最近'时**以谁为基准**"——snap 算法在 `FindClosestValidAreaInternal` 里最小化 `|candidate.snap_offset - base_position|`，它就是这把尺子的起点。

关键区别：`intended_position` 是**自然落点**，`base_position` 是**量距离的基准**。两者不一定相同——这正是 `DirectionStrategy` 的 `StepPreference` 玩的花样。

## 各自的用途

`base_position` 决定"参赛的候选里谁最近"——即在通过其他筛子（alignment、方向、互可见、proximity、focus、`:target` 等）的候选中，按距离选出胜者。

`intended_position` 不直接参与"最近"比较，它有两个用途：

1. **covering 候选的 offset**：当某 area 比 snapport 大时，"covering"候选就用 `intended_position` 当 offset——"就停在你本来要停的地方，让大 area 盖满视口"。
2. **方向判断**：`HasIntendedDirection()` 为 true 时，`direction = intended_position - current_position`（即 step），用于判断是否滚向首/末 snap 点（`is_extremity`）。

## 两个策略各取什么值

```cpp
// EndPositionStrategy（scrollTo / Tab / 锚点 / Home/End）
intended_position() = current_position_;   // 调用方已把 end_position 当 current 传入
base_position()    = current_position_;    // 两者相等：在终点附近找最近

// DirectionStrategy（方向键 / fling / 翻页 / scrollBy）
intended_position() = current_position_ + step_;                 // 自然落点 = 当前 + 步长
base_position() = (preferred_step_ == kDirection)
                  ? current_position_                            // kDirection: 从起点量
                  : current_position_ + step_;                   // kDistance:  从落点量
```

- `kDirection`（方向键、`CreateForDirection`）：`base = current`，从**起点**量 → 停在该方向**第一个** snap 点。
- `kDistance`（fling、翻页、`CreateForDisplacement/PageScroll`）：`base = current + step`，从**自然落点**量 → 停在**离落点最近**的 snap 点（可越过中途的 snap 点）。

## 例子：方向键 vs fling

设一个纵向容器，`scroll-snap-type: y mandatory`，snapport 高 200px，snap area（start 对齐）在 offset：

```
A=0   B=200   C=400   D=600   E=800
```

当前滚动位置 = 0（顶部显示 A）。

**场景一：按↓方向键（kDirection，step=+100）**

```
current_position  = 0
step              = +100
intended_position = 0 + 100 = 100      // 不 snap 的话滚到 100
base_position     = 0                   // kDirection → base = current
```

`IsValidSnapPosition` 只接受 delta > 0（同向），候选是 B/C/D/E（A 的 delta=0 被排除）。量到 `base=0` 的距离：B=200, C=400, D=600, E=800 → **最近是 B(200)**。→ snap 到 B。一次方向键滚到下一个 snap 点。

**场景二：往下 fling（kDistance，displacement=+550）**

```
current_position  = 0
step              = +550
intended_position = 0 + 550 = 550      // 不 snap 的话滚到 550
base_position     = 550                 // kDistance → base = intended
```

同样候选 B/C/D/E。量到 `base=550` 的距离：B=350, C=150, D=50, E=250 → **最近是 D(600)**。→ snap 到 D。

**对比的意义**：同一个起点、同样往下滚，方向键落在 **B**，fling 落在 **D**——区别全在 `base_position`：方向键 `base=current=0`（从起点量，最近的就是刚出发遇到的第一个→B）；fling `base=intended=550`（从落点量，最近的是离惯性终点最近的→D，哪怕中途越过了 B、C）。`StepPreference` 通过改 `base_position` 一行就实现了这两种行为的切换。

## 一句话记忆

**`base_position` 量距离选 area，`intended_position` 定落点和方向。**

# 八、SnapSearchResult 数据结构

`FindSnapPosition` 的搜索过程本质上是在造、比较、筛选一堆 `SnapSearchResult` 对象——每个 area 在某个轴上对应一个候选结果。理解它的字段是读算法的前提。定义在 [cc/input/scroll_snap_data.h](cc/input/scroll_snap_data.h)。

## 字段一览

| 字段 | 类型 | 含义 |
|------|------|------|
| `snap_offset_` | `float` | 该候选的 snap 滚动 offset（本结果所在轴）。由对齐方式算出，已 clip 到 `[0, max]`。若 `covered_range_` 有值，则它是该范围内的一个点（covering 候选时等于 `intended_position`）|
| `area_` | `const SnapAreaData*` | 这个结果对应的 snap area。`element_id()` / `has_focus_within()` 都委托给它 |
| `covered_range_` | `optional<RangeF>` | **覆盖区间**。当 area 比 snapport 大时设置，表示"在这个 offset 范围内 area 都盖满视口"。`snap_offset_` 落在其中。区分"对齐型"与"覆盖型"（spec 的 [Snapping Boxes that Overflow the Scrollport](https://drafts.csswg.org/css-scroll-snap-1/#snap-overflow)）|
| `axis_` | `SearchAxis` | 本结果是 X 还是 Y 轴的 |
| `snapport_visible_range_` | `RangeF` | snapport 在**cross 轴**的范围（用于算 `visible_range()`）|
| `snapport_max_visible_` | `float` | cross 轴的最大滚动 offset（用于 clamp `visible_range()`）|
| `rect_` | `optional<RectF>` | area 相对容器的 rect。算 `visible_range()` 和嵌套最内层判断都要用 |
| `alternative_` | `optional<SnapSearchResultAlternative>` | 备选 area：和本 area 在主轴对齐、且两轴都是 snap target。可能在 cross 轴也对齐时成为更好选择 |

## snap_offset：候选的"落点"

`GetSnapSearchResult(axis, area)` 按对齐方式算出来（scroll_snap_data.cc 起）：

```
X 轴（inline）:                       Y 轴（block）:
  kStart:  area.rect.x - rect.x         kStart:  area.rect.y - rect.y
  kCenter: area.Center.x - rect.Center.x  kCenter: area.Center.y - rect.Center.y
  kEnd:    area.rect.right - rect.right   kEnd:    area.rect.bottom - rect.bottom
```

其中 `rect` 是 snapport（已含 scroll-padding）。算完 `Clip(max)` 限制到合法范围。这就是算法里 `candidate.snap_offset()` 被拿来和 `base_position` 比距离的那个值。

## visible_range：互可见性判断的依据

`visible_range()` 是个**计算属性**，返回该 area 在 **cross 轴**仍可见的滚动 offset 范围：

```cpp
// 本结果在 X 轴时，rect_start/end 取 area 的 Y 跨度（cross 轴）
float rect_start = axis_ == kX ? rect_.y() : rect_.x();
float rect_end   = axis_ == kX ? rect_.bottom() : rect_.right();
return RangeF(
    clamp(rect_start - snapport_visible_range_.end(), 0, snapport_max_visible_),
    clamp(rect_end   - snapport_visible_range_.start(), 0, snapport_max_visible_));
```

直觉：视口的 cross 轴范围随滚动平移，area 的 cross 轴跨度固定；两者重叠时 area 可见。`visible_range()` 算出的就是"cross 轴 offset 落在这个范围里时，area 还在视口里"。

它被 `IsMutualVisible` 用：

```cpp
bool IsMutualVisible(const SnapSearchResult& a, const SnapSearchResult& b) {
  return RangeF(b.snap_offset()).IsBoundedBy(a.visible_range()) &&
         RangeF(a.snap_offset()).IsBoundedBy(b.visible_range());
}
```

a 是 X 轴结果、b 是 Y 轴结果时：`a.visible_range()` 是 Y 范围，`b.snap_offset()` 是 Y offset——检查"若 X snap 到 a，b 的 Y 落点是否还在 a 的可见范围内"；反向同理。只有两边都成立，这对 (x, y) snap 点才互可见，避免选出一个让某轴目标滚出视口的组合。

## 例子：visible_range 与 IsMutualVisible

设一个 2D 容器：`scroll-snap-type: both mandatory`，snapport 是 `RectF(0, 0, 200, 200)`（即 `RectF(x, y, w, h)`，x∈[0,200]、y∈[0,200]，无 scroll-padding/边框），内容 1000×1000，两轴 `max_position` 都是 800。三个 area 都 200×200、start 对齐：

| area | x 跨度 | y 跨度 | right() | bottom() | X snap_offset | Y snap_offset |
|------|--------|--------|---------|----------|---------------|---------------|
| A | [100,300] | [100,300] | 300 | 300 | 100 | 100 |
| B | [500,700] | [500,700] | 700 | 700 | 500 | 500 |
| D | [150,350] | [150,350] | 350 | 350 | 150 | 150 |

（snap_offset = `area.x - snapport.x` 等；snapport.x=snapport.y=0。）

**先算 visible_range**。公式：

```
visible_range = [ clamp(rect_start - snapport_visible_range.end, 0, max),
                  clamp(rect_end   - snapport_visible_range.start, 0, max) ]
```

其中 X 结果的 `rect_start/rect_end` 取 area 的 **y 跨度**（`rect_.y()` / `rect_.bottom()`，cross 轴），`snapport_visible_range = [0, 200]`（snapport 的 y 跨度），`max = 800`：

| 结果 | rect_start | rect_end | 计算 | visible_range |
|------|------------|----------|------|---------------|
| A 作 X 结果 | A.y=100 | A.bottom=300 | `[100-200, 300-0]`=`[-100,300]`→clamp | `[0, 300]` |
| A 作 Y 结果 | A.x=100 | A.right=300 | 同上（A 是正方形）| `[0, 300]` |
| B 作 X 结果 | B.y=500 | B.bottom=700 | `[500-200, 700-0]`=`[300,700]` | `[300, 700]` |
| D 作 Y 结果 | D.x=150 | D.right=350 | `[150-200, 350-0]`=`[-50,350]`→clamp | `[0, 350]` |

表格里 `visible_range` 的两项不是凭空凑的，它就是"视口和 area 求交"的上下界。以 A 作 X 结果为例：视口高 200（snapport y 跨度 `[0,200]`），滚到 Y offset=y 时视口 y 跨度为 `[y, y+200]`，A 的 y 跨度是 `[100,300]`。两者相交的两个条件，正好对应公式里的两项：

- 视口底 ≥ A 顶：`y+200 ≥ 100` → `y ≥ -100`，即公式的 `start = rect_start - snapport.end = 100 - 200`；
- 视口顶 ≤ A 底：`y ≤ 300`，即公式的 `end = rect_end - snapport.start = 300 - 0`。

合起来 `y ∈ [-100, 300]`，clamp 到 `[0, 300]`。所以 `visible_range` 的语义就是"**A 还在视口里**"的 Y offset 区间，公式两项分别对应相交的下界和上界。

**情形一：X→A、Y→D（两个 area 重叠，应互可见）**

```
IsMutualVisible(A_x, D_y):
  D.y_offset=150 ∈ A_x.visible_range=[0,300]?  ✓   (Y snap 到 150 时 A 仍可见)
  A.x_offset=100 ∈ D_y.visible_range=[0,350]?  ✓   (X snap 到 100 时 D 仍可见)
→ 互可见
```

几何验证：snap 到 (100,150) 后视口 = x∈[100,300]、y∈[150,350]。A（x∈[100,300], y∈[100,300]）与视口交 x∈[100,300]∩[100,300]、y∈[150,300]，可见；D（x∈[150,350], y∈[150,350]）与视口交 x∈[150,300]、y∈[150,350]，可见。两个 area 都在视口里，pair 成立。✓

**情形二：X→A、Y→B（对角分布，不应互可见）**

```
IsMutualVisible(A_x, B_y):
  B.y_offset=500 ∈ A_x.visible_range=[0,300]?  ✗   (Y snap 到 500 时 A 已滚出视口)
→ 不互可见（短路，反向不用查）
```

几何验证：snap 到 (100,500) 后视口 = x∈[100,300]、y∈[500,700]。A 的 y 跨度 [100,300] 与视口 y 跨度 [500,700] 完全不交——A 被滚出视口了。所以这对 (x=100, y=500) 不能同时 snap，算法拒绝。✓

**对算法的影响**：`FindSnapPosition` 阶段 3 每轴独立搜出 `selected_x=A`、`selected_y=B` 后，阶段 4 的 `IsMutualVisible` 判定不通过，于是 `SelectAxisToFollowForMutualVisibility` 选一个轴（比如沿 X 轴的 A），用 A 作为 cross 轴基准重搜 Y：`FindClosestValidArea(kY, strategy, A_x)`。这次搜 Y 时，候选必须在 A 的 Y 可见范围 `[0,300]` 内才通过互可见性，于是会选到一个和 A 共存的 Y 候选（如 A 自己的 Y=100 或 D 的 Y=150），而不是远处的 B=500。

这就是互可见性的全部意义：**防止两轴各自选了"最近"的 snap 点、合起来却把其中一个 area 滚出视口**。它强制两轴的 snap 目标在几何上能共存。

> 记法注意：`gfx::RectF` 的构造函数是 `RectF(x, y, width, height)`，所以 `rect_.bottom() = y + height`、`rect_.right() = x + width`。本例为避免歧义，area 一律用"x 跨度 / y 跨度"显式标注，而不是写四元组。

## covered_range：覆盖型 vs 对齐型

普通 area：`covered_range_` 为空，`snap_offset_` 是对齐点（start/center/end），算法比较距离。

大 area（比 snapport 大）：`CanCoverSnapportOnAxis` 为 true，`FindCoveringCandidate` 算出一个 offset 范围，area 在这范围内任何位置都盖满视口。此时：

- `covered_range_` 设为该范围；
- `snap_offset_` 取范围内的一个点——covering 候选时就是 `intended_position`（"停在你本来要停的地方"）。

`PickBestResult(closest, covering)` 在两者间决策；最终结果若 `covered_range` 有值，`FindSnapPosition` 把 `result.type` 标成 `kCovered`（区别于 `kAligned`）。

## alternative：双轴对齐的备选

`SnapSearchResultAlternative` 含 `area` / `area_rect` / `cross_axis_snap_offset`。当 `scroll-snap-type: both` 且某 area 在两轴都对齐、和当前 closest 同等优先时，`FindClosestValidAreaInternal` 会把它记进 `alternative_`（scroll_snap_data.cc 的 `UpdateSearchAlternative`）。

到 `FindSnapPosition` 阶段 6，`SelectAlternativeIdForSearchResult` 检查：若这个备选在 cross 轴也对齐，就改用它——因为它能让两轴同时 snap 到同一个 area，比"两轴各 snap 各的"更优。

## 一句话串联

`GetSnapSearchResult` 造候选（`snap_offset` + cross 轴信息）→ `FindClosestValidAreaInternal` 用 `visible_range` 做互可见、用 `snap_offset` 比距离、可能记 `alternative`、大 area 时算 `covered_range` → `FindSnapPosition` 用 `covered_range` 判类型、用 `alternative` 做双轴优化、把 `snap_offset` 和 `element_id` 写进最终 `SnapPositionData`。

# 九、FindSnapPosition 实现详解

前面各节讲的是策略类本身；这一节看策略的**消费方**——`SnapContainerData::FindSnapPosition`（[cc/input/scroll_snap_data.cc](cc/input/scroll_snap_data.cc)）。关键认识是：`FindSnapPosition` 本身不关心是方向滚动还是终点滚动——它只在搜索的各个阶段调策略对象的钩子（`IsValidSnapArea` / `IsValidSnapPosition` / `base_position` / `IsPreferredSnapPosition` / `PickBestResult` 等），具体规则全靠策略子类差异化实现。下面把 `FindSnapPosition` 及其调用的子函数拆开讲。

## 入口：FindSnapPositionWithViewportAdjustment

```cpp
SnapPositionData FindSnapPositionWithViewportAdjustment(
    const SnapSelectionStrategy& strategy, double snapport_height_adjustment) {
  base::AutoReset<double> resetter{&snapport_height_adjustment_, snapport_height_adjustment};
  return FindSnapPosition(strategy);
}
```

这个包装存在的意义是给 `FindSnapPosition` 注入一个**一次性的 snapport 高度调整量**，而不污染持久状态。三个关键点：

**1. `base::AutoReset` 是 RAII**：把 `snapport_height_adjustment_` 这个成员临时设成传入值，函数返回时自动恢复成原值（0）。所以这个调整只对本次 `FindSnapPosition` 调用生效，不影响后续调用或并发场景。

**2. 调整量怎么生效**：`FindSnapPosition` 内部所有 snapport 几何都通过 `snapport()` 取，而 `snapport()` 会读这个成员（scroll_snap_data.cc）：

```cpp
gfx::RectF SnapContainerData::snapport() const {
  if (!snapport_height_adjustment_) return rect_;        // 默认无调整
  gfx::RectF adjusted = rect_;
  // top 不变，只是从 top 锚点向下扩展高度
  adjusted.set_height(adjusted.height() + snapport_height_adjustment_);
  return adjusted;
}
```

即调整量非零时，snapport 的**高度变大**（顶部锚点不动，向下扩展）。snap_offset 计算（`area.edge - snapport.edge`）、covering 判断、互可见性范围全都走 `snapport()`，所以这一个入口自然把调整传导到所有几何计算。

**3. 为什么需要它**：浏览器顶栏（top controls）/ 工具栏显示/隐藏会改变**外层 viewport** 的可见高度——顶栏藏起来时视口变高。如果 snap 还按旧视口高度算，会算错对齐位置。`InputHandler` 在外层 viewport 翻页/fling 时（input_handler.cc）调这个带调整量的入口，把"当前真实可见高度"传进来。内层普通 scroller 的视口不受顶栏影响，直接调无调整的 `FindSnapPosition` 即可。

所以分两个入口：`FindSnapPosition` 是通用算法本体（`const`，纯函数式）；`FindSnapPositionWithViewportAdjustment` 是给外层 viewport 用的薄包装，靠 `AutoReset` + `snapport()` 把一次性高度调整注入进去。

## FindSnapPosition 主流程（7 个阶段）

**阶段 1：早返回**

```cpp
if (scroll_snap_type_.is_none) return result;  // result.type 默认 kNone
```

**阶段 2：决定搜索哪些轴**

```cpp
SnapAxis axis = scroll_snap_type_.axis;
bool should_snap_on_x = strategy.ShouldSnapOnX() && (axis == kX || axis == kBoth);
bool should_snap_on_y = strategy.ShouldSnapOnY() && (axis == kY || axis == kBoth);
```

策略说"这次滚了 X 轴"还要 AND 上容器的 snap-type 也允许 X 轴。两者都没命中时，**保留旧 target id**（仅当对应 area 仍存在）后返回——避免单轴滚动误清掉另一轴的 snap 状态：

```cpp
if (!should_snap_on_x && !should_snap_on_y) {
  if (axis == SnapAxis::kY) result.target_element_ids.y = target_snap_area_element_ids_.y;
  else result.target_element_ids.x = target_snap_area_element_ids_.x;
  return result;
}
```

**阶段 3：每轴独立搜索（带"优先沿用旧 target"）**

```cpp
bool should_prioritize_x_target =
    strategy.ShouldPrioritizeSnapTargets() && target_snap_area_element_ids_.x != ElementId();
...
if (should_snap_on_x) {
  // cross 轴基准: clamp(base.y, 0, max)，互可见性判断要用
  SnapSearchResult initial_snap_position_y = { std::clamp(base_position.y(), 0.f, max_position_.y()), ... };
  if (should_prioritize_x_target)
    selected_x = GetTargetSnapAreaSearchResult(strategy, kX, initial_snap_position_y);
  if (!selected_x)
    selected_x = FindClosestValidArea(kX, strategy, initial_snap_position_y);
}
// Y 轴同理
```

两个要点：

- **cross 轴基准**：搜 X 轴时需要一个 Y 轴位置做互可见性判断。传入值不一定在界内，先 `clamp(base.y, 0, max)` 作为 cross 轴起点。
- **优先旧 target**：`ShouldPrioritizeSnapTargets()` 为 true（仅布局后 re-snap，即 `SnapAfterLayout` → `CreateForTargetElement` 传 `kRequire`）且该轴有旧 target 时，先调 `GetTargetSnapAreaSearchResult` 尝试直接吸回旧 target；失败再退到通用 `FindClosestValidArea`。注意 scrollTo/scrollIntoView 用的 `CreateForEndPosition` 是 `kIgnore`，**不**走这条路——它们直接进 `FindClosestValidArea` 找离终点最近的 snap 点。

`GetTargetSnapAreaSearchResult`（scroll_snap_data.cc）：找到旧 target 的 area → 算对齐 offset → 若该 area 比 snapport 大，还要在它的范围内 `FindClosestValidAreaInternal` 找嵌套的覆盖候选；否则直接返回对齐结果。

**阶段 4：两轴都没搜到 → 找"同元素双轴 snap"回退（带 early return）**

```cpp
if (!selected_x && !selected_y) {
  if (should_snap_on_x && should_snap_on_y &&
      !strategy.ShouldRespectSnapStop() &&
      FindSnapPositionForMutualSnap(strategy, &result.position)) {
    result.type = SnapPositionData::Type::kAligned;
  }
  return result;   // ← 注意：无论 mutual snap 成没成功都直接 return
}
```

分轴搜索可能在两轴各自看不到对方候选时双双落空。`FindSnapPositionForMutualSnap`（scroll_snap_data.cc）换思路：找**同一个 area** 在 X、Y 上都有效的 snap 点，按 block 距离优先、inline 距离次之排序。注意 `ShouldRespectSnapStop()` 时不走这条路（snap-stop:always 场景不妥协）。

**关键**：这个分支里有 `return result;`，无论 mutual snap 是否成功都会直接返回——所以一旦"两轴都空"，后面的阶段 5/6/7 都不会执行。只有至少一个轴搜到了结果时，才会继续往下走到阶段 5。

**阶段 5：双轴都搜到但不互可见 → 选一个轴重搜另一轴**

```cpp
if (selected_x && selected_y && !IsMutualVisible(selected_x, selected_y)) {
  SnapAxis axis_to_follow = SelectAxisToFollowForMutualVisibility(strategy, selected_x, selected_y);
  if (axis_to_follow == kX)
    selected_y = FindClosestValidArea(kY, strategy, selected_x);  // 沿 X 的结果重搜 Y
  else
    selected_x = FindClosestValidArea(kX, strategy, selected_y);
}
```

`SelectAxisToFollowForMutualVisibility`（scroll_snap_data.cc）决定沿哪轴：若某轴之前没 snap 过（target 为空）就沿另一轴；否则优先 focus / `:target` / block 轴。重搜时用被沿的轴的结果作 cross 轴基准（替代阶段 3 的占位 `initial_snap_position_*`），从而找到一个和它互可见的另一轴候选。

**阶段 6：alternative 选择**

```cpp
if (selected_y && selected_y->alternative())
  SelectAlternativeIdForSearchResult(*selected_y, selected_x, ...);
if (selected_x && selected_x->alternative())
  SelectAlternativeIdForSearchResult(*selected_x, selected_y, ...);
```

`FindClosestValidAreaInternal` 里会记一个"alternative"候选——某个 area 在两轴都对齐、可作为备选。这里如果某轴选中的结果带 alternative，就尝试用它在另一轴也对齐的版本做更好选择。

**阶段 7：组装结果**

```cpp
result.position = strategy.current_position();  // 从当前位置开始
// 保留仍存在的旧 target（避免单轴无结果时误清）
for (const auto& area : snap_area_list_) {
  if (area.element_id == target_snap_area_element_ids_.x) result.target_element_ids.x = ...;
  if (area.element_id == target_snap_area_element_ids_.y) result.target_element_ids.y = ...;
}
gfx::Vector2dF direction = strategy.HasIntendedDirection()
    ? strategy.intended_position() - strategy.current_position()
    : gfx::Vector2dF();
if (selected_x) {
  result.position.set_x(selected_x->snap_offset());
  result.target_element_ids.x = selected_x->element_id();
  result.covered_range_x = selected_x->covered_range();
  // is_extremity 判断（见下）
}
if (selected_y) { /* 同理 */ }
if ((!selected_x || result.covered_range_x) && (!selected_y || result.covered_range_y))
  result.type = SnapPositionData::Type::kCovered;
return result;
```

这一阶段做三件事：填位置/target/covered_range、判 `is_extremity`、定 `type`。后两个容易 overlooked，展开讲。

**`is_extremity`：是否 snap 到了该方向的尽头**

`SnapContainerData` 维护了每轴 snap offset 的极值表 `min_snap_offset_x_ / max_snap_offset_x_`（Y 同理），在 `AddSnapAreaData` 时维护（scroll_snap_data.cc）：每加入一个 area，算它的 `snap_offset`（覆盖型则用 `covered_range` 的 start/end），`UpdateMinMax` 进极值表。所以在 mandatory scroller 里，`[min_snap_offset_, max_snap_offset_]` 就是"可达 snap 范围"。

`is_extremity` 的判断（以 X 轴为例）：

```cpp
if (!selected_x->area()->must_snap &&
    ((direction.x() > 0.f && max_snap_offset_x_ &&
      (selected_x->snap_offset() == *max_snap_offset_x_ ||
       (selected_x->covered_range() &&
        selected_x->covered_range()->end() == *max_snap_offset_x_))) ||
     (direction.x() < 0.f && min_snap_offset_x_ &&
      (selected_x->snap_offset() == *min_snap_offset_x_ ||
       (selected_x->covered_range() &&
        selected_x->covered_range()->start() == *min_snap_offset_x_))))) {
  result.is_extremity = true;
}
```

即：**方向滚动**（`direction.x` 非零）+ **area 非 `must_snap`** + **选中点的 offset（或 covered_range 端点）等于该方向的极值**。语义是"这次 snap 落在该方向的第一个或最后一个 snap 点上"。

消费方是 fling：`InputHandler::GetSnapFlingInfoAndSetAnimatingSnapTarget`（input_handler.cc）在 `kSnapFlingNearExtremes` 开启时，用 `is_extremity` 判断 fling 是否有足够速度到达尽头那个 snap 点——不够快就不强 snap 到尽头，让 fling 自然衰减（避免一次轻甩被硬吸到很远的最末 snap 点）。

**`type`：kAligned vs kCovered**

```cpp
if ((!selected_x || result.covered_range_x) && (!selected_y || result.covered_range_y))
  result.type = SnapPositionData::Type::kCovered;
```

逻辑：对每个**有选中结果**的轴，若它都是覆盖型（`covered_range` 有值），则整体 `kCovered`；任一轴是对齐型（`covered_range` 为空）则 `kAligned`。某轴无选中结果（`!selected_x`）则跳过该轴检查——只看有结果的轴。

这区分了两种 snap：`kAligned` = 滚到对齐点（area 边缘贴 snapport 边缘）；`kCovered` = 停在让大 area 盖满视口的位置（不强行对齐到边）。调用方（动画/事件）据此决定如何呈现滚动过渡。

## FindClosestValidArea：单轴搜索 + 两次补救

`FindSnapPosition` 每轴调一次 `FindClosestValidArea`（scroll_snap_data.cc），它在 `FindClosestValidAreaInternal` 之外加了两次补救搜索：

```cpp
std::optional<SnapSearchResult> FindClosestValidArea(...) {
  auto result = FindClosestValidAreaInternal(axis, strategy, cross_axis_snap_result);

  // 补救 1: scroll-snap-stop:always
  if (result && strategy.ShouldRespectSnapStop()) {
    // 用 SnapStopAlwaysFilter::kRequire 再搜一次，只看 must_snap 的 area
    auto must_only_strategy = CreateForDirection(..., SnapStopAlwaysFilter::kRequire);
    auto must_only_result = FindClosestValidAreaInternal(axis, *must_only_strategy, ...);
    result = ClosestSearchResult(current_position, axis, result, must_only_result);
  }

  // 补救 2: 方向策略太严时放松
  if (result || strictness == kProximity || !strategy.HasIntendedDirection())
    return result;
  // mandatory + 方向策略 + 没找到候选 → 退成 EndPosition 再搜一次
  auto relaxed_strategy = CreateForEndPosition(current_position, ShouldSnapOnX, ShouldSnapOnY);
  return FindClosestValidAreaInternal(axis, *relaxed_strategy, ...);
}
```

- **补救 1**：方向滚动尊重 snap-stop:always——若最近候选和起点之间夹着一个 `must_snap` 的 area，不能越过它，得选那个 always 点。
- **补救 2**：mandatory 严格度下，方向策略可能因"只接受同向"而找不到候选（比如已 snap 在最后一个点还往下滚），这时退成终点策略放松方向约束再搜，避免 mandatory 容器 snap 不到任何点。

## FindClosestValidAreaInternal：核心循环

真正遍历 `snap_area_list_` 的地方（scroll_snap_data.cc）。签名：

```cpp
std::optional<SnapSearchResult> FindClosestValidAreaInternal(
    SearchAxis axis,
    const SnapSelectionStrategy& strategy,
    const SnapSearchResult& cross_axis_snap_result,   // cross 轴基准，互可见性要用
    bool should_consider_covering = false,             // 是否算 covering 候选
    std::optional<gfx::RangeF> active_element_range = std::nullopt) const;  // 限定 area 范围（嵌套覆盖搜索用）
```

### 初始化

```cpp
std::optional<SnapSearchResult> closest;              // 当前最近的"对齐型"候选
std::optional<SnapSearchResult> covering_intended;    // 落在 intended_position 的"覆盖型"候选
float intended_position = horiz ? strategy.intended_position().x() : ...y();
float base_position    = horiz ? strategy.base_position().x() : ...y();
bool preferred_candidate = false;                     // closest 是否是 preferred
float smallest_distance = proximity_range;            // 注意：初值就是 proximity_range，不是无穷大
float proximity_distance = proximity_range;
```

`smallest_distance` 初值设成 `proximity_range` 而非无穷大，这样 proximity 范围自然成为"准入门槛"——超出 `proximity_range` 的候选既过不了前面的显式 `distance > proximity_distance` 检查，也不会被纳入 closest。

### evaluate lambda：候选打分（核心）

每个候选（对齐型或 displaced 覆盖型）都进这个 lambda。逻辑分三段：**过滤 → 门槛 → 取代/平局**。

```cpp
auto evaluate = [&](const SnapSearchResult& candidate, const SnapAreaData& area) {
  // --- 过滤 ---
  if (!IsMutualVisible(candidate, cross_axis_snap_result)) return;   // cross 轴看不见 → 弃
  if (!strategy.IsValidSnapPosition(axis, candidate.snap_offset())) return;  // 方向/轴不合法 → 弃
  float distance = std::abs(candidate.snap_offset() - base_position);
  if (distance > proximity_distance) return;                          // 超 proximity 范围 → 弃

  bool is_preferred_candidate = strategy.IsPreferredSnapPosition(axis, candidate.snap_offset());

  // --- preferred 压制 ---
  if (preferred_candidate && !is_preferred_candidate) return;         // 已有 preferred，非 preferred 直接弃
  if (distance > smallest_distance &&
      (preferred_candidate || !is_preferred_candidate)) return;       // 更远且(已有 preferred 或本候选非 preferred) → 弃
```

两道 preferred 门要连起来理解：

- 第一道：已经有 preferred 候选时，非 preferred 的直接弃——preferred（如翻页位移约束内的点）压倒一切非 preferred。
- 第二道：本候选比当前 best 更远时，若"已有 preferred"或"本候选非 preferred"则弃。反过来：一个**更远的 preferred 候选**在"还没 preferred"时**不会被弃**——它会进下面的取代逻辑把当前非 preferred 的 best 顶掉。即 preferred 即便更远也胜过非 preferred。

```cpp
  // --- 取代规则 ---
  if (distance < smallest_distance ||
      (is_preferred_candidate && (!preferred_candidate || candidate.has_focus_within()))) {
    smallest_distance = distance;
    closest = candidate;
    preferred_candidate = is_preferred_candidate;
  }
```

取代条件：**严格更近**，或 **preferred 且（还没 preferred 或带 focus）**。后半句让"preferred 候选替换非 preferred"以及"preferred 之间带 focus 的胜出"成立。

```cpp
  // --- 平局打破（not 更近、not preferred 更优时）---
  else if (closest && !closest->has_focus_within()) {
    if (closest->element_id() == targeted_area_id_) return;           // closest 是 :target → 不让位
    if (candidate.element_id() == targeted_area_id_) {                // 候选是 :target → 让位给它
      closest = candidate; preferred_candidate = is_preferred_candidate; return;
    }
    // 嵌套：closest 包含 candidate（candidate 更内/更小）→ 选内层
    if (closest_rect->Contains(candidate_rect) && closest_rect != candidate_rect) {
      smallest_distance = distance; closest = candidate; ...
    }
    // 否则：两轴都对齐且同优先 → 记为 alternative
    else if (axis == kBoth && area 两轴都对齐 && is_preferred == preferred_candidate) {
      UpdateSearchAlternative(*closest, candidate, area, strategy);
    }
  }
};
```

平局段的几条规则按优先级：

- **`:target` 优先**：closest 若是 CSS `:target` 对应的 area，不让位；候选若是 `:target`，直接顶替。这给 `:target` area 特权（scroll_snap_data.cc）。
- **嵌套选最内层**：两个 area 互相包含时选更小的内层（`RectF::Contains` + 不等）——嵌套 snap area 时优先吸到最内层。
- **alternative 记录**：`scroll-snap-type: both` 且候选在两轴都对齐、和 closest 同等优先时，调 `UpdateSearchAlternative` 把它记到 `closest->alternative_`。这个 alternative 可能在 `FindSnapPosition` 阶段 6 被选中（若它在 cross 轴也对齐）。

### 主循环

```cpp
for (const SnapAreaData& area : snap_area_list_) {
  if (!strategy.IsValidSnapArea(axis, area)) continue;          // alignment / must_snap 过滤
  if (active_element_range && !active_element_range->Intersects(area_range)) continue;  // 范围限定

  SnapSearchResult candidate = GetSnapSearchResult(axis, area);  // 算对齐 offset
  evaluate(candidate, area);

  if (should_consider_covering && CanCoverSnapportOnAxis(axis, snapport(), area.rect)) {
    if (auto covering = FindCoveringCandidate(area, axis, candidate, intended_position)) {
      covering->set_rect(area.rect);
      if (covering->snap_offset() == intended_position) {
        SetOrUpdateResult(*covering, &covering_intended);        // 落在 intended → 进 covering_intended
      } else {
        evaluate(*covering, area);                               // 偏离 intended → 当普通候选比距离
      }
    }
  }
}
```

循环对每个 area 做四件事：

1. **`IsValidSnapArea` 过滤**：alignment 不匹配该轴、或 `snap_stop_always_filter_==kRequire` 但 area 非 `must_snap` → 跳过。
2. **`active_element_range` 过滤**：限定搜索范围（`GetTargetSnapAreaSearchResult` 处理嵌套大 area 时传入，只在目标 area 的范围内找）。
3. **算对齐候选并 `evaluate`**：`GetSnapSearchResult` 按 start/center/end 算 `snap_offset`，进 lambda 打分。
4. **覆盖候选**：area 在该轴比 snapport 大时（`CanCoverSnapportOnAxis`），额外调 `FindCoveringCandidate` 算一个"让大 area 盖满视口"的 offset。

覆盖候选分两种去向：

- `snap_offset == intended_position`：正好停在自然落点就能盖满视口 → 进 `covering_intended`（用 `SetOrUpdateResult` 合并多个覆盖候选，带 focus 的优先）。这类**不参与距离比较**，单独存。
- `snap_offset != intended_position`：偏离落点的覆盖候选 → 当普通候选进 `evaluate`，和对齐型一起比距离。

注释强调：即便 area 覆盖 snapport，循环也**继续**——为了找前/后 snap 点、保留 alternative、以及这个大 area 自己的对齐点可能反过来胜过覆盖候选。

### 终选

```cpp
return strategy.PickBestResult(closest, covering_intended);
```

把"最近对齐候选"和"覆盖候选"交给策略决策：`EndPositionStrategy` 偏好 covering（停落点），`DirectionStrategy` 视方向和 `StepPreference` 在 closest/covering 间权衡（见第五节 `PickBestResult`）。

### 关键辅助函数

**`CanCoverSnapportOnAxis`**（scroll_snap_data.cc）：area 在该轴的尺寸 ≥ snapport 尺寸 → 可能覆盖。是进入 covering 分支的门槛。

**`FindCoveringCandidate`**（scroll_snap_data.cc）：给一个比 snapport 大的 area，算一个"盖满视口又尽量不被其他 area 侵入"的 offset。这个函数对应 spec 的 [Snapping Boxes that Overflow the Scrollport](https://drafts.csswg.org/css-scroll-snap-1/#snap-overflow)——area 比 snapport 大时，snap 到 start/center/end 反而错（会让 area 偏到一边），正确做法是"停在能让 area 盖满视口的位置"。但又要躲开其他 area（侵入者），否则视口里会同时出现两个 area。展开讲。

**核心思路：dodging（躲避）**。在 area 的可覆盖区间里找一个子区间，让 snapport 落进去时既被这个 area 盖满、又不被其他 area 侵入。算三个躲避区间，优先级 middle > 离 intended 更近的 backward/forward。

**setup**：

```cpp
bool horiz = axis == kX;
float scroll_padding = horiz ? rect.x() : rect.y();        // snapport 在本轴的起点
float snapport_size  = horiz ? rect.width() : rect.height();
gfx::RangeF area_range = ...;                               // area 在本轴的跨度
// 若滚到 intended_position，snapport 占据的范围：
gfx::RangeF preferred_snapport(intended_position + scroll_padding,
                               intended_position + scroll_padding + snapport_size);
// 三个躲避区间，初值都是 area_range，后续被侵入者逐步收紧
gfx::RangeF backward_dodging_range = area_range;
gfx::RangeF middle_dodging_range   = area_range;
gfx::RangeF forward_dodging_range  = area_range;
```

**遍历侵入者**：对每个其他 area（`intruder`），跳过不重叠的、以及是 `area_range` 超集的（大 area 包小 area 不算侵入）。剩下的侵入者按位置收紧三个区间：

- **`backward_dodging_range`**（躲到侵入者**上方/前方**）：侵入者在 `preferred_snapport` 之前 → 把区间起点顶到侵入者之后；否则把终点收到侵入者之前。
- **`forward_dodging_range`**（躲到侵入者**下方/后方**）：侵入者在 `preferred_snapport` 之后 → 把终点收到侵入者之前；否则把起点顶到侵入者之后。
- **`middle_dodging_range`**（躲到 `preferred_snapport` 内侵入者之间的**缝隙**）：侵入者包含或被包含于 `preferred_snapport` → 区间清空；侵入者从上方侵入 → 顶起点；从下方侵入 → 压终点。

直觉：`backward`/`forward` 是"整个躲到侵入者外面（前或后）"，`middle` 是"留在侵入者之间但视口内"。`middle` 优先因为它最贴近 `intended_position`。

**`SearchResultForDodgingRange`**（scroll_snap_data.cc）：把一个躲避区间变成 `SnapSearchResult`。

```cpp
float min_offset = dodging_range.start() - scroll_padding;
float max_offset = dodging_range.end() - scroll_padding - snapport_size;
if (max_offset > min_offset) {                              // snapport 放得下
  result.set_snap_offset(std::clamp(preferred_offset, min_offset, max_offset));
  result.set_covered_range(gfx::RangeF(min_offset, max_offset));
  return result;
}
// 放不下：退回按 alignment 选 offset，保证内容仍可达（不设 covered_range → 对齐型）
```

关键：放得下时 `snap_offset = clamp(intended_position, min, max)`——把预期落点夹进合法覆盖区间。**若 `intended_position` 本来就在区间内，`snap_offset` 就等于 `intended_position`**，这正是 `FindClosestValidAreaInternal` 里 `covering->snap_offset() == intended_position` 判断的来源——命中则进 `covering_intended`（不比距离），否则当普通候选进 `evaluate` 比距离。放不下时退回 start/center/end 对齐，且不设 `covered_range`（变成对齐型，避免内容不可达）。

**选择顺序**：

```cpp
if (middle_candidate) return middle_candidate;          // 1. 优先 middle
// 2. 否则 backward / forward 都算，取离 intended_position 更近的
if (!backward_candidate) return forward_candidate;
if (!forward_candidate) return backward_candidate;
return |backward.offset - intended| < |forward.offset - intended| ? backward : forward;
```

**整体流程回顾**：`FindClosestValidAreaInternal` 主循环里，area 在本轴 ≥ snapport 尺寸（`CanCoverSnapportOnAxis`）时调 `FindCoveringCandidate`。它返回的候选要么落点 == `intended_position`（进 `covering_intended`，最终交 `PickBestResult` 决策），要么偏离落点（当普通候选进 `evaluate` 比距离）。这样大 area 既能"盖满视口停在原地"，又不会和侵入它的其他 area 撞在一起。



**`SetOrUpdateResult`**（scroll_snap_data.cc）：合并多个 covering 候选——两个都盖满视口就都合法，`Union` 它们的 rect（`element_id` 任选其一）；带 focus 的 area 优先作为代表。

**`UpdateSearchAlternative`**（scroll_snap_data.cc）：这是 `FindClosestValidAreaInternal` 里最绕的一个辅助函数，单独展开讲。

**它解决什么问题**：`scroll-snap-type: both` 时，X、Y 两轴各自搜出 `closest_x`、`closest_y`，可能指向两个不同的 area。但如果存在一个 area **在两轴都对齐**，且它的 X 对齐点和 `closest_x` 同等优先，那么"两轴都 snap 到这一个 area"比"两轴各 snap 各的"更优（语义上是一个整体目标）。`UpdateSearchAlternative` 就是在 X 轴搜索过程中，把这样一个 area 作为 `closest_x` 的**备选**（`alternative_`）记下来，留给阶段 6 判断是否启用。

**调用时机**（evaluate 的平局段末尾，scroll_snap_data.cc）：

```cpp
else if ((scroll_snap_type_.axis == SnapAxis::kBoth) &&
         (area.scroll_snap_align.alignment_block != kNone) &&
         (area.scroll_snap_align.alignment_inline != kNone) &&
         is_preferred_candidate == preferred_candidate) {
  UpdateSearchAlternative(*closest, candidate, area, strategy);
}
```

即：仅当 `both` 轴 snap-type、候选 area 在两轴都有 alignment、且和当前 closest 同 preferred 状态时才考虑。能走到这里，说明候选既没更近、也没触发嵌套替换——是个"同等优先但不更优"的候选，正好适合当备选。

**函数逻辑**：

```cpp
void UpdateSearchAlternative(SnapSearchResult& current_result,   // = closest
                             const SnapSearchResult& candidate_result,
                             const SnapAreaData& candidate_area,
                             const SnapSelectionStrategy& strategy) const {
  bool horiz = current_result.axis() == SearchAxis::kX;
  // ① 算候选在 cross 轴的对齐 offset（这是备选的关键数据）
  const auto candidate_cross_axis_aligned_result =
      GetSnapSearchResult(horiz ? kY : kX, candidate_area);
  ...
  // ② 候选包含 closest（候选是外层）→ 不当备选（我们偏好内层）
  if (candidate_rect->Contains(*current_result_rect)) return;

  if (auto alt = current_result.alternative()) {
    // ③ 已有备选：比 cross 轴距离
    float cross_axis_base_position =
        horiz ? strategy.base_position().y() : strategy.base_position().x();
    float candidate_cross_axis_distance =
        std::abs(cross_axis_base_position -
                 candidate_cross_axis_aligned_result.snap_offset());
    float alt_cross_axis_distance =
        std::abs(cross_axis_base_position - alt->cross_axis_snap_offset);
    if (candidate_cross_axis_distance > alt_cross_axis_distance) return;  // 候选更远 → 留住旧备选
    // 候选更近，或距离相等且候选嵌套在旧备选内（内层优先）→ 换成候选
    if (candidate_cross_axis_distance < alt_cross_axis_distance ||
        (alt_rect != *candidate_rect && alt_rect.Contains(*candidate_rect))) {
      current_result.set_alternative(&candidate_area, *candidate_rect,
                                     candidate_cross_axis_aligned_result.snap_offset());
    }
  } else {
    // ④ 还没备选 → 直接把候选设为备选
    current_result.set_alternative(...);
  }
}
```

几个要点：

- **备选存什么**：`area`（备选 area）、`area_rect`、`cross_axis_snap_offset`（备选在 cross 轴的对齐 offset）。前两个用于阶段 6 切换 area，第三个用于判断"cross 轴是否真的 snap 到了它"。
- **② 的包含检查**：调用前已保证候选不在 closest 内（否则走嵌套替换分支了）；这里再排除"候选包含 closest"（候选是外层）。即备选只考虑和 closest **互不包含**的 area——既不是 closest 的外层、也不是内层，而是同等层级的另一个 area。
- **③ 的择优标准**：备选之间按 **cross 轴距离**（到 cross 轴 base 的距离）择优。直觉：cross 轴搜索时最可能选中的是离 cross base 近的 area，所以备选也挑离 cross base 近的，这样阶段 6 命中的概率最高。距离相等时再按"内层优先"打破平局。
- **`cross_axis_base_position`** 用的是 `strategy.base_position()` 的 cross 分量——和主轴比距离用的是同一个 base，保持基准一致。

**消费方**（阶段 6 的 `SelectAlternativeIdForSearchResult`，scroll_snap_data.cc）：两轴都搜完后，检查 cross 轴的实际结果是否落在了备选的 cross offset 上：

```cpp
if (cross_selection) {
  if (|cross_selection.snap_offset - alt.cross_axis_snap_offset| <= kSnappedToTolerance)
    selection.set_area(alt.area);   // cross 轴真的 snap 到了备选 → 本轴也改用备选 area
} else {
  // cross 轴无结果时，用 cross 轴当前位置判断
  if (|clamp(cross_current_position, 0, max) - alt.cross_axis_snap_offset| <= kSnappedToTolerance)
    selection.set_area(alt.area);
}
```

命中后把 `selection`（本轴 closest）的 area 换成备选 area，于是 X、Y 两轴的 `element_id` 指向同一个 area——达成"两轴 snap 同一元素"。

**和 `FindSnapPositionForMutualSnap` 的区别**：两者都追求"两轴 snap 同一 area"，但路径不同。`FindSnapPositionForMutualSnap` 是**两轴都搜空时的兜底**（阶段 4），从头按"同元素双轴"搜；`UpdateSearchAlternative` + `SelectAlternativeIdForSearchResult` 是**两轴都搜到了之后的优化**（阶段 5→6），在已有结果上尝试把两轴合并到同一 area。一个事前兜底，一个事后优化。



## 调用层次总览

```
FindSnapPositionWithViewportAdjustment
  └─ FindSnapPosition
       ├─ [每轴] GetTargetSnapAreaSearchResult            (优先旧 target)
       ├─ [每轴] FindClosestValidArea
       │    ├─ FindClosestValidAreaInternal               (核心循环 + evaluate)
       │    │    └─ GetSnapSearchResult / FindCoveringCandidate
       │    ├─ [ShouldRespectSnapStop] 再搜 must_only     (snap-stop 补救)
       │    └─ [无结果+mandatory+方向] 退成 EndPosition 重搜 (放松补救)
       ├─ [两轴都空] FindSnapPositionForMutualSnap         (同元素双轴回退)
       ├─ [双轴不互可见] SelectAxisToFollowForMutualVisibility → 重搜 cross 轴
       ├─ SelectAlternativeIdForSearchResult               (alternative 选择)
       └─ 组装 SnapPositionData { position, target_element_ids, covered_range, type, is_extremity }
```

核心设计思想：**主路径是"每轴独立搜最近合法候选 + 互可见性约束"**，失败时层层回退（沿一轴重搜另一轴 → 同元素双轴 → 放松方向约束），并通过策略对象的钩子把"方向/终点/翻页/snap-stop/优先旧 target"这些场景差异注入到统一的搜索循环里。

# 十、测试

算法的正确性主要靠 [cc/input/scroll_snap_data_unittest.cc](cc/input/scroll_snap_data_unittest.cc) 验证。测试通常的写法是：构造一个 `SnapContainerData`（往里塞若干 `SnapAreaData`），用某个 `CreateFor*` 造策略，调 `FindSnapPosition`，断言落点。看测试时重点关注：

- 同一组 area，换不同策略（`CreateForDirection` vs `CreateForEndPosition`）落点不同的用例——能直观体现策略差异。
- `scroll-snap-stop: always` 不能被跳过的用例。
- covering（area 大于 snapport）场景下 `PickBestResult` 的选择。
- 翻页策略下 `preferred_min/max_displacement` 的约束效果。

# 十一、速查表

| 想知道 | 看这里 |
|--------|--------|
| 策略基类与钩子 | `snap_selection_strategy.h` |
| 终点位置滚动行为 | `EndPositionStrategy`（`snap_selection_strategy.h`、`.cc` 对应实现）|
| 方向滚动行为 | `DirectionStrategy`（`snap_selection_strategy.h`、`.cc` 对应实现）|
| 工厂方法 → 子类映射 | `snap_selection_strategy.cc` 顶部五个 `CreateFor*` |
| 翻页的位移约束 | `CreateForPageScroll` + `ScrollUtils::Calculate*PageSnap` |
| `scroll-snap-stop: always` 处理 | `ShouldRespectSnapStop` / `SnapStopAlwaysFilter` / `IsValidSnapArea` |
| 沿用已有 snap target | `ShouldPrioritizeSnapTargets` / `SnapTargetsPrioritization`（`CreateForTargetElement` 开启）|
| Blink 侧分派 | `ScrollableArea::SnapFor*`（`scrollable_area.cc` 起）|
| compositor 侧分派 | `InputHandler::AdjustScrollDeltaForScrollbarSnap` / `GetSnapFlingInfoAndSetAnimatingSnapTarget` / `ScrollEnd` |
