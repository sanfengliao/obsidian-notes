# cc/animation 模块分析：以滚动为例

---

# 一、类层次结构总览

```
MutatorHost (抽象接口)
└── AnimationHost                          [cc/animation/animation_host.h]
    ├── owns (1:N) AnimationTimeline       [cc/animation/animation_timeline.h]
    │   └── ScrollTimeline                 [cc/animation/scroll_timeline.h]  (子类)
    ├── owns (1:N) AnimationTrigger        [cc/animation/animation_trigger.h]
    │   └── TimelineTrigger               [cc/animation/timeline_trigger.h]
    ├── map: ElementId → ElementAnimations [cc/animation/element_animations.h]
    ├── list: ticking Animations (正在运行的动画)
    ├── ScrollOffsetAnimations             (仅主线程)
    └── ScrollOffsetAnimationsImpl         (仅 impl 线程)

AnimationTimeline (base)
└── owns (1:N) Animation                  [cc/animation/animation.h]
    └── WorkletAnimation                  (子类，由 AnimationWorklet 驱动)
    └── 1:1 → KeyframeEffect              [cc/animation/keyframe_effect.h]
              └── owns (1:N) KeyframeModel [cc/animation/keyframe_model.h]
                            └── owns 1 → gfx::AnimationCurve (来自 ui/gfx/)
                                  具体子类:
                                  ├── ScrollOffsetAnimationCurve     ← 滚动专用
                                  ├── FilterAnimationCurve
                                  ├── gfx::KeyframedFloatAnimationCurve
                                  ├── gfx::KeyframedTransformAnimationCurve
                                  └── gfx::KeyframedColorAnimationCurve

ElementAnimations                         [cc/animation/element_animations.h]
    ├── observed by (0:N) KeyframeEffect  (通过 ObserverList)
    ├── implements: ScrollOffsetAnimationCurve::Target
    ├── implements: gfx::TransformAnimationCurve::Target
    ├── implements: gfx::FloatAnimationCurve::Target
    ├── implements: FilterAnimationCurve::Target
    └── implements: gfx::ColorAnimationCurve::Target
```

---

# 二、文件清单

| 文件 | 职责 |
|------|------|
| `animation_host.h/.cc` | 顶层入口，动画系统的单一拥有者 |
| `animation_timeline.h/.cc` | 分组管理 Animation，是 ScrollTimeline 的基类 |
| `animation.h/.cc` | 核心动画分组类，1:1 拥有 KeyframeEffect |
| `keyframe_effect.h/.cc` | 拥有某元素的所有 KeyframeModel，驱动其状态机 |
| `keyframe_model.h/.cc` | 持有 AnimationCurve + 播放状态（run/pause/finish）|
| `element_animations.h/.cc` | 聚合同一元素的所有 KeyframeEffect，分发计算结果到属性树 |
| `scroll_offset_animation_curve.h/.cc` | 插值滚动偏移的曲线（ease-in-out / linear）|
| `scroll_offset_animation_curve_factory.h/.cc` | 创建 ScrollOffsetAnimationCurve 的工厂 |
| `scroll_offset_animations.h/.cc` | 主线程侧，批量处理滚动动画更新（anchor adjust / takeover）|
| `scroll_offset_animations_impl.h/.cc` | impl 线程侧，管理 impl-only 滚动动画 |
| `scroll_timeline.h/.cc` | AnimationTimeline 子类，由滚动位置驱动，用于 CSS 滚动联动动画 |
| `animation_events.h/.cc` | 动画事件容器（START / FINISH / ABORT / TAKEOVER）|
| `animation_delegate.h` | 动画生命周期回调接口 |
| `animation_trigger.h/.cc` | 基于条件控制动画播放 |
| `timeline_trigger.h/.cc` | 基于 timeline range 的 AnimationTrigger 子类 |
| `worklet_animation.h/.cc` | AnimationWorklet JS API 驱动的 Animation 子类 |
| `filter_animation_curve.h/.cc` | CSS filter 属性动画曲线 |

---

# 三、各类职责详解

## `AnimationHost`

**文件：** `cc/animation/animation_host.h`

动画系统的**唯一顶层拥有者**。每个 `LayerTreeHost`（主线程）和 `LayerTreeHostImpl`（impl/compositor 线程）各持有一个实例，通过 `PushPropertiesTo()` 同步。

关键成员：
```cpp
// 全部 timeline
ProtectedSequenceReadable<IdToTimelineMap> id_to_timeline_map_;
// 当前正在 tick 的动画列表（subset of all animations）
ProtectedSequenceReadable<AnimationsList> ticking_animations_;
// ElementId → ElementAnimations 的映射
ProtectedSequenceReadable<ElementToAnimationsMap> element_to_animations_map_;
// 两者互斥，各线程有其一：
std::unique_ptr<ScrollOffsetAnimations>     scroll_offset_animations_;     // 主线程
std::unique_ptr<ScrollOffsetAnimationsImpl> scroll_offset_animations_impl_; // impl 线程
```

关键方法：
```cpp
bool TickAnimations(base::TimeTicks, const ScrollTree&, bool is_active_tree, MutatorEvents*);
void TickScrollAnimations(base::TimeTicks, const ScrollTree&);
void TickWorkletAnimations();
bool ActivateAnimations(MutatorEvents*);

// impl-only 滚动动画接口（鼠标滚轮 / programmatic）
void ImplOnlyScrollAnimationCreate(...);
std::optional<gfx::PointF> ImplOnlyScrollAnimationUpdateTarget(...);
void ScrollAnimationAbort(ElementId);
```

通过 `MutatorHostClient` 接口（由 `LayerTreeHostImpl` 实现）将结果写回 layer tree 的属性树节点。

---

## `AnimationTimeline`

**文件：** `cc/animation/animation_timeline.h`

拥有一组 `Animation` 对象的容器，通过 `id_to_animation_map_` 以 animation id 为键管理。每个 `AnimationTimeline` 在主线程和 impl 线程各有一个实例，通过 id 匹配同步。

关键成员：
```cpp
ProtectedSequenceWritable<IdToAnimationMap> id_to_animation_map_;
const int id_;
const bool is_impl_only_;  // true 表示此 timeline 仅存在于 impl 线程，不随 commit 同步
```

**`is_impl_only`** 标志区分两类 timeline：
- `is_impl_only = false`（默认）：主线程创建，commit 时通过 `PushPropertiesTo` 镜像到 impl 线程
- `is_impl_only = true`：仅在 impl 线程上创建和销毁，`ScrollOffsetAnimationImpl` 内部使用此类型，保证平滑滚动动画不经过 commit

**生命周期操作：**
```cpp
void AttachAnimation(scoped_refptr<Animation> animation);  // 加入 id_to_animation_map_，设置 animation 的 host 和 timeline 指针
void DetachAnimation(scoped_refptr<Animation> animation);  // 从 map 中移除，断开 animation 的 element 绑定
void ClearAnimations();                                     // 清空全部 animation
```

**commit 时同步（`PushPropertiesTo`）：**
```cpp
void PushPropertiesTo(AnimationTimeline* timeline_impl) {
    PushAttachedAnimationsToImplThread(timeline_impl);    // 主线程有但 impl 没有的 → 在 impl 创建
    RemoveDetachedAnimationsFromImplThread(timeline_impl);// impl 有但主线程没有的 → 从 impl 删除
    PushPropertiesToImplThread(timeline_impl);            // 逐个 Animation::PushPropertiesTo
}
```

**Tick 分发：**
```cpp
// 时间联动（墙上时钟驱动），基类实现；跳过 WorkletAnimation 和 ScrollLinkedAnimation
bool TickTimeLinkedAnimations(ticking_animations, monotonic_time, tick_finished);

// 滚动联动，基类返回 false；由 ScrollTimeline 覆盖实现
virtual bool TickScrollLinkedAnimations(ticking_animations, scroll_tree, is_active_tree);
```

---

## `ScrollTimeline`

**文件：** `cc/animation/scroll_timeline.h`

`AnimationTimeline` 的子类。"当前时间"由 scroller 的位置（而非墙上时钟）决定，实现 [CSS Scroll Animations 规范](https://wicg.github.io/scroll-animations/)。

时间换算公式：
```
current_time = (scroll_offset - start_offset) / (end_offset - start_offset)
             × duration_in_microseconds
// 换算系数：16 μs/pixel（来源于 LayoutUnit 精度 1/64px）
static constexpr double kScrollTimelineMicrosecondsPerPixel = 16;
virtual std::optional<base::TimeTicks> CurrentTime(
    const ScrollTree& scroll_tree, bool is_active_tree) const;

bool TickScrollLinkedAnimations(
    const std::vector<scoped_refptr<Animation>>& ticking_animations,
    const ScrollTree& scroll_tree, bool is_active_tree) override;
```

关键状态：
```cpp
// active_id_ 仅 impl 线程持有
ProtectedSequenceForbidden<std::optional<ElementId>> active_id_;
// pending_id_ 主线程设置，push 到 impl
ProtectedSequenceWritable<std::optional<ElementId>>  pending_id_;
ProtectedSequenceReadable<ScrollDirection>           direction_;
// scroll 起止位置
ProtectedSequenceWritable<std::optional<ScrollOffsets>> pending_offsets_;
```

---

## `Animation`

**文件：** `cc/animation/animation.h`

对单一目标元素 上一组相关 `KeyframeModel` 的分组容器，与 `KeyframeEffect` **1:1** 对应。其本身几乎不包含业务逻辑，大多数操作直接转发给 `KeyframeEffect`。

注释原文的定义：
> An Animation is responsible for managing animating properties for a set of targets. It is only a grouping mechanism for related effects, and the grouping relationship is defined by the client.
> 动画（Animation）负责管理一组目标对象的动画属性。它仅作为相关效果的一种分组机制，且该分组关系由客户端定义。

关键成员：
```cpp
raw_ptr<AnimationHost>     animation_host_;
raw_ptr<AnimationTimeline> animation_timeline_;     // 所属 timeline
std::unique_ptr<KeyframeEffect> keyframe_effect_;   // 1:1 拥有

raw_ptr<AnimationDelegate> animation_delegate_;     // 生命周期回调
bool is_replacement_;  // 是否为替换型动画（start_time 需从 impl 侧取回）
```

**元素绑定流程：**

```cpp
void AttachElement(ElementId element_id)
  → keyframe_effect()->AttachElement(element_id)   // KeyframeEffect 记录 element_id
  → RegisterAnimation()                            // 在 AnimationHost 注册，创建/复用 ElementAnimations
void DetachElement()
  → UnregisterAnimation()                          // 从 AnimationHost 注销，可能销毁 ElementAnimations
```

**AddKeyframeModel 是纯转发：**
```cpp
void Animation::AddKeyframeModel(std::unique_ptr<KeyframeModel> keyframe_model) {
  keyframe_effect()->AddKeyframeModel(std::move(keyframe_model));  // 所有逻辑在 KeyframeEffect
}
```

**ticking 管理：**
```cpp
void AddToTicking()     → animation_host()->AddToTicking(this)    // 加入全局 ticking_animations_
void RemoveFromTicking() → animation_host()->RemoveFromTicking(this)
```

**主线程 → impl 线程同步（`PushPropertiesTo`）：**
```cpp
void Animation::PushPropertiesTo(Animation* animation_impl) {
  keyframe_effect()->PushPropertiesTo(animation_impl->keyframe_effect(), ...);
  // 将 KeyframeModel 推送到 impl 侧，同步 start_time 等状态
}
```

**关键方法：**
```cpp
virtual bool Tick(base::TimeTicks tick_time);                          // 转发给 KeyframeEffect::Tick
virtual void UpdateState(bool start_ready_keyframe_models, ...);       // 转发给 KeyframeEffect::UpdateState
bool IsScrollLinkedAnimation() const;  // 判断所属 timeline 是否为 ScrollTimeline
virtual bool IsWorkletAnimation() const;  // WorkletAnimation 子类返回 true
```

---

## `KeyframeEffect`

**文件：** `cc/animation/keyframe_effect.h`

注释原文的定义：
> A KeyframeEffect owns a group of KeyframeModels for a single target (identified by an ElementId). It is responsible for managing the KeyframeModels' running states (starting, running, paused, etc), as well as ticking the KeyframeModels when it is requested to produce new outputs for a given time.

是动画系统中**状态机驱动层**的核心。每个 `KeyframeEffect` 对应一个 `ElementId`，持有该元素上某一个 `Animation` 关联的所有 `KeyframeModel`。多个 `KeyframeEffect`（来自多个 `Animation`）可以指向同一元素，它们通过共享的 `ElementAnimations` 汇聚。

关键成员：
```cpp
raw_ptr<Animation>               animation_;               // 反向指针（不拥有）
ElementId                        element_id_;              // 目标元素
scoped_refptr<ElementAnimations> element_animations_;      // 与同元素其他 effect 共享
bool needs_to_start_keyframe_models_;                      // 有新 model 需要推进状态
bool scroll_offset_animation_was_interrupted_;             // 滚动动画被中断标记
bool is_ticking_;                                          // 是否在 ticking_animations_ 中
bool awaiting_deletion_;                                   // 所有 model 已完成，等待删除
```

**与 ElementAnimations 的绑定（双向）：**
```cpp
// ElementAnimations 调用此方法建立绑定
void BindElementAnimations(ElementAnimations* element_animations) {
  element_animations_ = element_animations;
  if (has_any_keyframe_model())
    KeyframeModelAdded();  // 若 model 已存在，立即完成 curve→target 绑定
}

void UnbindElementAnimations() { element_animations_ = nullptr; }
```

**AddKeyframeModel 完整逻辑：**
```cpp
void KeyframeEffect::AddKeyframeModel(std::unique_ptr<gfx::KeyframeModel> model) {
  // 1. DCHECK：SCROLL_OFFSET 不能同时存在两个非 finished 的 model
  // 2. 调用基类 gfx::KeyframeEffect::AddKeyframeModel，追加到 keyframe_models_ 列表
  // 3. 若 ElementAnimations 已绑定：
  if (has_bound_element_animations()) {
    KeyframeModelAdded();       // AttachToCurve + UpdateTickingState
    SetNeedsPushProperties();
  }
}
```

**`KeyframeModelAdded` 负责两件事：**
```cpp
void KeyframeEffect::KeyframeModelAdded() {
  needs_to_start_keyframe_models_ = true;
  UpdateTickingState();   // → AddToTicking()，将 Animation 加入全局 ticking 列表
  for (auto& model : keyframe_models())
    element_animations_->AttachToCurve(model->curve());  // curve.target_ = ElementAnimations*
  element_animations_->UpdateClientAnimationState();
}
```

**每帧 Tick 内部流程：**
```cpp
bool KeyframeEffect::Tick(base::TimeTicks monotonic_time) {
  if (needs_to_start_keyframe_models_)
    StartKeyframeModels(monotonic_time);   // WAITING_FOR_TARGET_AVAILABILITY → STARTING

  for (auto& model : keyframe_models()) {
    if (model 处于 RUNNING 或 STARTING)
      model->Tick(monotonic_time);         // 调用曲线插值 → OnXxxAnimated 回调
  }
  element_animations_->UpdateClientAnimationState();
  MarkFinishedKeyframeModels(monotonic_time);  // 超时 → FINISHED
}
```

**状态机推进（`UpdateState`，与 Tick 分离调用）：**
```cpp
void KeyframeEffect::UpdateState(bool start_ready, AnimationEvents* events) {
  PromoteStartedKeyframeModels(events);   // STARTING → RUNNING，发出 START 事件
  MarkKeyframeModelsForDeletion(events);  // FINISHED/ABORTED → WAITING_FOR_DELETION，发出 FINISH 事件
  PurgeDeletedKeyframeModels();           // WAITING_FOR_DELETION → 实际删除
}
```

**跨线程同步（`PushPropertiesTo`）：**
```cpp
void PushPropertiesTo(KeyframeEffect* impl, std::optional<base::TimeTicks> replaced_start_time) {
  PushNewKeyframeModelsToImplThread(impl);          // 主线程有、impl 没有的 model → 在 impl 创建
  RemoveKeyframeModelsCompletedOnMainThread(impl);  // 主线程已完成的 → 标记 impl 侧 affects_pending=false
  MarkAbortedKeyframeModelsForDeletion(impl);       // 中止的 → 标记 impl 侧删除
  PurgeKeyframeModelsMarkedForDeletion(impl_only=false);
}
```

---

## `KeyframeModel`

**文件：** `cc/animation/keyframe_model.h`

封装单个 `gfx::AnimationCurve` 和完整的播放状态，继承自 `ui/gfx/` 的 `gfx::KeyframeModel`。是动画系统中最小的可执行单元。

注释原文：
> A KeyframeModel contains all the state required to play an AnimationCurve. Specifically, the affected property, the run state (paused, finished, etc.), loop count, last pause time, and the total time spent paused.

关键成员：
```cpp
int group_;                    // 同组 model（相同 group_id）保证同时启动，同时结束
TargetPropertyId target_property_id_;   // 目标属性（SCROLL_OFFSET / TRANSFORM / OPACITY 等）
ElementId element_id_;         // 可选，若设置则覆盖 KeyframeEffect 的 element_id（用于 BGPT）

bool needs_synchronized_start_time_;
// 主线程创建的 model 为 true：必须等 impl 线程的 start_time 回传后才能开始 tick
// impl-only model 为 false：可在第一帧直接设置 start_time，无需等待

bool is_controlling_instance_;
// impl 线程上的实例为 true：实际驱动属性值的变化
// 主线程上的实例为 false：只跟踪状态，不输出值

bool is_impl_only_;
// true：仅在 impl 线程上存在，不推送到主线程（impl-only 滚动动画使用）

bool affects_active_elements_;   // 是否写入 active tree（渲染用）
bool affects_pending_elements_;  // 是否写入 pending tree（commit 用）
// commit 路径中，新 push 的 model 初始只有 affects_pending=true；
// activation 后两者均为 true
```

**生命周期状态机（来自基类 gfx::KeyframeModel）：**
```
WAITING_FOR_TARGET_AVAILABILITY  →  STARTING  →  RUNNING
                                                      ↓
                                      PAUSED  ←→  RUNNING
                                                      ↓
                                               FINISHED / ABORTED / ABORTED_BUT_NEEDS_COMPLETION
                                                      ↓
                                          WAITING_FOR_DELETION  →  (实际删除)
```

状态转换时机：
| 转换 | 触发位置 |
|---|---|
| `WAITING → STARTING` | `KeyframeEffect::StartKeyframeModels`，每帧 Tick 开始时 |
| `STARTING → RUNNING` | `KeyframeEffect::PromoteStartedKeyframeModels`，`UpdateState` 时 |
| `RUNNING → FINISHED` | `KeyframeEffect::MarkFinishedKeyframeModels`，超过 duration |
| `FINISHED → WAITING_FOR_DELETION` | `KeyframeEffect::MarkKeyframeModelsForDeletion`，`UpdateState` 时 |

**`group_` 的语义：** 同一 group 的多个 model（如同时动画 X 和 Y 方向）必须原子启动——`StartKeyframeModels` 中只有当 group 内所有 model 的目标属性均不被其他正在运行的 model 占用时，才会整组推进到 `STARTING`。

---

## `ElementAnimations`

**文件：** `cc/animation/element_animations.h`

注释原文：
> An ElementAnimations owns a list of all KeyframeEffects attached to a single target (represented by an ElementId).

同一 `ElementId` 上所有 `KeyframeEffect`（来自不同 `Animation`）共用一个 `ElementAnimations` 实例，它是动画值**写入属性树的分发中心**，也是各类 `AnimationCurve` 回调的 Target 实现。

关键成员：
```cpp
base::ObserverList<KeyframeEffect>::Unchecked keyframe_effects_list_;
// 同元素上所有 Animation 的 KeyframeEffect 都注册在此，用于聚合查询
// 不拥有 KeyframeEffect（生命周期由各自的 Animation 管理）

raw_ptr<AnimationHost> animation_host_;
ElementId element_id_;

PropertyAnimationState active_state_;    // active tree 上正在/可能动画的属性集合
PropertyAnimationState pending_state_;   // pending tree 上正在/可能动画的属性集合
```

**单个元素存在多个 KeyframeEffect 的场景：**

`KeyframeEffect` 与 `Animation` 一一对应，同一元素上有几个 `Animation` 就有几个 `KeyframeEffect` 注册进来。典型场景：

```css
/* 场景1：多个 CSS animation 同时作用于同一元素 */
.box { animation: move 1s, fade 0.5s; }
/* → Blink 为 move 和 fade 各创建一个 Animation，两个 KeyframeEffect 注册到同一 ElementAnimations */

/* 场景2：CSS animation + CSS transition 并存 */
/* transition 和 animation 是不同的 Animation 实例 */

/* 场景3：JS Web Animations API 多次调用 animate() */
el.animate({ transform: ['translateX(0)', 'translateX(100px)'] }, 1000);
el.animate({ opacity: [1, 0] }, 500);
// 两次 animate() → 两个 Animation → 两个 KeyframeEffect → 同一 ElementAnimations
```
**单个 KeyframeEffect 存在多个 KeyframeModel 的场景：**

一个 CSS 动画可以同时对多个属性插值，每个属性对应一个 `KeyframeModel`，但共享同一 `group_id`，保证原子启动：

```css
/* 一个 @keyframes 同时动画多个属性 */
@keyframes combo {
  from { transform: translateX(0); opacity: 1; }
  to   { transform: translateX(100px); opacity: 0; }
}
/* → 同一 KeyframeEffect 里：
     model_1: target_property=TRANSFORM, group=5
     model_2: target_property=OPACITY,   group=5
     group_id 相同，StartKeyframeModels 保证同帧进入 STARTING */
```

另一个场景是 pending/active 双树过渡期：主线程 push 一个新 model 到 impl 后，activation 前同一 `KeyframeEffect` 里短暂共存：
- 旧 model：`affects_active=true, affects_pending=false`（已在 active tree 运行）
- 新 model：`affects_active=false, affects_pending=true`（等待激活）

**完整结构示例：**

```
ElementAnimations (element: div#box)
├── KeyframeEffect_A  ←  Animation "move"（CSS animation）
│     ├── KeyframeModel: TRANSFORM, group=1
│     └── KeyframeModel: OPACITY,   group=1   ← 同 @keyframes 的多属性
│
└── KeyframeEffect_B  ←  Animation "fade"（另一个 CSS animation）
      └── KeyframeModel: FILTER, group=2
```

---

**`keyframe_effects_list_` 的作用：**

聚合同一元素上所有 effect 的状态，供 `ElementAnimations` 统一查询：
```cpp
// 判断是否有正在 tick 的 effect
bool HasTickingKeyframeEffect() const {
  for (auto& effect : keyframe_effects_list_)
    if (effect.HasTickingKeyframeModel()) return true;
  return false;
}

// 聚合所有 effect 的属性动画状态（OR 合并）
void UpdateClientAnimationState() {
  for (auto& effect : keyframe_effects_list_) {
    effect.GetPropertyAnimationState(&pending_state, &active_state);
    pending_state_ |= pending_state;
    active_state_  |= active_state;
  }
  // 通知 LayerTreeHostImpl 属性树哪些属性正在动画
  mutator_host_client()->ElementIsAnimatingChanged(...);
}
```

**AnimationCurve 的回调分发（Target 接口实现）：**

各类 AnimationCurve 在 `Tick` 时调用 `target_->OnXxxAnimated(...)`，由 `ElementAnimations` 接收后根据 `affects_active/pending` 标志分别写入两棵树：

```cpp
void OnScrollOffsetAnimated(const gfx::PointF& scroll_offset,
                             int target_property_id,
                             gfx::KeyframeModel* keyframe_model) {
  if (KeyframeModelAffectsActiveElements(keyframe_model))
    → MutatorHostClient::SetElementScrollOffsetMutated(element_id, ACTIVE, scroll_offset)
  if (KeyframeModelAffectsPendingElements(keyframe_model))
    → MutatorHostClient::SetElementScrollOffsetMutated(element_id, PENDING, scroll_offset)
}
// Transform / Opacity / Filter 同理
```

**`AttachToCurve`：将 ElementAnimations 注册为曲线的 Target：**
```cpp
void AttachToCurve(gfx::AnimationCurve* c) {
  switch (c->Type()) {
    case SCROLL_OFFSET:
      ScrollOffsetAnimationCurve::ToScrollOffsetAnimationCurve(c)->set_target(this);
    case TRANSFORM:
      gfx::TransformAnimationCurve::ToTransformAnimationCurve(c)->set_target(this);
    // ...
  }
}
```
此方法由 `KeyframeEffect::KeyframeModelAdded` 调用，时机是 `AddKeyframeModel` 且 `ElementAnimations` 已绑定之后。

**`active_state_` / `pending_state_` 的用途：**

这两个 `PropertyAnimationState` 记录当前哪些 CSS 属性正在被动画驱动。每次 Tick 结束后调用 `UpdateClientAnimationState`，若状态发生变化，通知 `LayerTreeHostImpl::ElementIsAnimatingChanged`，后者据此决定是否需要禁用某些合成器优化（如 will-change 提升、光栅化缓存失效等）。

---

## `ScrollOffsetAnimationCurve`

**文件：** `cc/animation/scroll_offset_animation_curve.h`

专用于滚动偏移插值的曲线，输出 `gfx::PointF`（滚动位置）。支持三种动画类型和时长行为：

```cpp
enum class AnimationType { kLinear, kEaseInOut };
enum class DurationBehavior {
    kDeltaBased,    // 与位移成正比 — programmatic scroll
    kConstant,      // 固定时长       — 键盘滚动
    kInverseDelta   // 与位移成反比   — 鼠标滚轮（体感更"弹性"）
};
enum class ScrollType { kProgrammatic, kKeyboard, kMouseWheel, kAutoScroll };
```

支持平滑"重定向"（retargeting）——在动画进行中更新目标位置，同时保留当前速度：
```cpp
void UpdateTarget(base::TimeTicks time, const gfx::PointF& new_target);
```

---

## `ScrollOffsetAnimationsImpl`

**文件：** `cc/animation/scroll_offset_animations_impl.h`

impl 线程专用滚动动画管理器，为鼠标滚轮、键盘、programmatic 等滚动创建**完全在 impl 线程上的动画**，无需主线程参与：

```cpp
void MouseWheelScrollAnimationCreate(ElementId, target, current, timing_function, duration);
bool ScrollAnimationUpdateTarget(scroll_delta, max_offset, frame_time, adjustment);
void ScrollAnimationApplyAdjustment(ElementId, adjustment);  // 锚点调整
void ScrollAnimationAbort(ElementId, needs_completion=false);
```

---

## `ScrollOffsetAnimations`

**文件：** `cc/animation/scroll_offset_animations.h`

主线程专用，批量管理从 Blink 发出的滚动动画更新，在 commit 时推送到 impl 侧：

```cpp
struct ScrollOffsetAnimationUpdate {
    ElementId element_id_;
    gfx::Vector2dF adjustment_;  // scroll anchor 调整量
    bool takeover_;              // 终止 impl 动画，交还 Blink 控制
};
void AddAdjustmentUpdate(ElementId, gfx::Vector2dF adjustment);
void AddTakeoverUpdate(ElementId);
void PushPropertiesTo(ScrollOffsetAnimationsImpl*);  // commit 时调用
```

---

# 四、三类滚动动画路径

## 4.1 Impl-Only 滚动动画（鼠标滚轮 / 键盘 / programmatic）
完全在 impl 线程上创建，无主线程对应实体：

```
AnimationHost::ImplOnlyScrollAnimationCreate(ElementId, target, current, ...)
  └── ScrollOffsetAnimationsImpl::MouseWheelScrollAnimationCreate(...)
        └── ScrollOffsetAnimationsImpl::ScrollAnimationCreateInternal(...)
              ├── ScrollOffsetAnimationCurveFactory::CreateAnimation(target, scroll_type)
              │     └── new ScrollOffsetAnimationCurve(target, EaseInOut, DurationBehavior)
              ├── KeyframeModel::Create(std::move(curve), id, group, SCROLL_OFFSET)
              ├── Animation::Create(id)  [impl-only，不同步到主线程]
              └── KeyframeEffect::AddKeyframeModel(std::move(keyframe_model))
```

**滚动目标更新（retargeting）：**
```
ImplOnlyScrollAnimationUpdateTarget(scroll_delta, max_offset, frame_time, ...)
  └── ScrollOffsetAnimationsImpl::ScrollAnimationUpdateTarget(...)
        └── ScrollOffsetAnimationCurve::UpdateTarget(t, new_target)
              // 平滑地将曲线重定向到新目标，保留当前速度
```

## 4.2 主线程发起的滚动动画（scroll-snap / Blink 主动发起）

在 Blink 中创建，通过 commit 推送，可被 `ScrollOffsetAnimations` 拦截或终止：

```
Blink 创建 scroll animation
  → commit 时 AnimationTimeline::PushPropertiesTo()
        → PushAttachedAnimationsToImplThread()   在 impl 侧创建 Animation
        → Animation::PushPropertiesTo()
              → KeyframeEffect::PushPropertiesTo()
                    → PushNewKeyframeModelsToImplThread()
                          → 注册 element → 创建 ElementAnimations（impl 侧）
                          → 将 Animation 加入 ticking_animations_

// Adjustment（锚点调整）或 Takeover（交还控制权）：
ScrollOffsetAnimations::PushPropertiesTo(ScrollOffsetAnimationsImpl*)
  → AdjustmentUpdate → ScrollAnimationApplyAdjustment()
  → TakeoverUpdate   → ScrollAnimationAbort(needs_completion=true)
                         + NotifyAnimationTakeover() → 通知 Blink 接管
```

## 4.3 CSS 滚动联动动画（`animation-timeline: scroll()`）

由 `ScrollTimeline` 提供时间源，以滚动位置驱动任意 CSS 属性动画：

```
ScrollTimeline::TickScrollLinkedAnimations(ticking_animations, scroll_tree, ...)
  ├── ScrollTimeline::CurrentTime(scroll_tree, is_active_tree)
  │     // 读取 ScrollTree 中的当前偏移量
  │     // 将 pixels → microseconds（16 μs/pixel）
  └── for each ticking Animation on this timeline:
        Animation::Tick(scroll_derived_time)
          └── 后续与时间联动动画路径相同
```

---

# 五、每帧 Tick 流程

impl 线程每帧调用（来自 `LayerTreeHostImpl::AnimateLayers`）：

```
AnimationHost::TickAnimations(monotonic_time, scroll_tree, is_active_tree, events)
  │
  ├── [时间联动动画]
  │     AnimationTimeline::TickTimeLinkedAnimations(ticking_animations, monotonic_time)
  │       └── for each ticking Animation:
  │             Animation::Tick(monotonic_time)
  │               └── KeyframeEffect::Tick(monotonic_time)
  │                     ├── StartKeyframeModels()      // STARTING → RUNNING
  │                     ├── for each KeyframeModel (RUNNING):
  │                     │     gfx::KeyframeModel::Tick(t)
  │                     │       └── AnimationCurve::Tick(t, property_id, model)
  │                     │             └── ElementAnimations::OnXxxAnimated(value, ...)
  │                     │                   └── MutatorHostClient::SetXxx(element_id, value)
  │                     │                         └── 更新属性树节点
  │                     └── MarkFinishedKeyframeModels()
  │
  ├── [滚动联动动画]
  │     AnimationHost::TickScrollAnimations(monotonic_time, scroll_tree)
  │       └── for each timeline (IsScrollTimeline()):
  │             ScrollTimeline::TickScrollLinkedAnimations(ticking_animations, scroll_tree, ...)
  │               ├── CurrentTime(scroll_tree)  // pixel → time
  │               └── Animation::Tick(scroll_derived_time)  // 同上
  │
  └── [WorkletAnimation]
        AnimationHost::TickWorkletAnimations()
          └── CollectWorkletAnimationsState() → MutatorInputState
                → 分发到 AnimationWorklet global scope（离线程）
                → SetMutationUpdate(MutatorOutputState) 写回结果
```

**状态更新（与 tick 分离）：**
```
AnimationHost::UpdateAnimationState(start_ready_animations, events)
  └── for each ticking Animation:
        Animation::UpdateState(start_ready_animations, events)
          └── KeyframeEffect::UpdateState(...)
                └── PromoteStartedKeyframeModels() / MarkKeyframeModelsForDeletion()
```

---

# 六、动画结果如何写入属性树

从曲线 tick 到实际属性树更新的完整路径：

## 滚动偏移动画路径：

```
ScrollOffsetAnimationCurve::Tick(t)
  └── target_->OnScrollOffsetAnimated(value, property_id, model)
        // target_ 是 ElementAnimations，由 AttachToCurve() 设置
        └── ElementAnimations::OnScrollOffsetAnimated(list_type, value, model)
              └── animation_host_->mutator_host_client()
                    ->ElementScrollOffsetAnimated(element_id_, list_type, value)
                      (MutatorHostClient = LayerTreeHostImpl)
                        └── PropertyTrees::scroll_tree()
                              .SetScrollOffset(element_id, value)
                              // 更新 ScrollNode 的 current offset（pending/active tree）
```

## Transform 动画路径（类比）：

```
ElementAnimations::OnTransformAnimated(list_type, transform, model)
  └── MutatorHostClient::ElementTransformIsAnimatingChanged()
        └── PropertyTrees::transform_tree()
              .node_at(node_id)->local = transform
```

---

# 七、主线程 ↔ impl 线程同步

```
主线程                                     impl 线程（Compositor）
─────────────────────────────────────────────────────────────────
AnimationHost (main)                       AnimationHost (impl)
  AnimationTimeline::PushPropertiesTo()
    → PushAttachedAnimationsToImplThread()   在 impl 创建 Animation
    → Animation::PushPropertiesTo()
        → KeyframeEffect::PushPropertiesTo()
            → PushNewKeyframeModelsToImplThread()
                → 注册 element → 创建 ElementAnimations（impl 侧）
                → 将 Animation 加入 ticking_animations_

ScrollOffsetAnimations::PushPropertiesTo(ScrollOffsetAnimationsImpl*)
  → AdjustmentUpdate → ScrollAnimationApplyAdjustment()  (动画中途重新锚定)
  → TakeoverUpdate   → ScrollAnimationAbort(needs_completion=true)
                         + NotifyAnimationTakeover() → 通知 Blink 接管
```

---

# 八、完整所有权关系图

```
AnimationHost
├── id_to_timeline_map_: {int → AnimationTimeline}
│     └── AnimationTimeline
│           └── id_to_animation_map_: {int → Animation}
│                 └── Animation
│                       └── keyframe_effect_: KeyframeEffect (1:1)
│                             ├── element_id_
│                             ├── element_animations_: 指向共享的 ElementAnimations
│                             └── keyframe_models_: [KeyframeModel, ...]
│                                   └── curve_: AnimationCurve
│                                         (ScrollOffsetAnimationCurve / Filter / Float / Transform ...)
│
├── element_to_animations_map_: {ElementId → ElementAnimations}
│     └── ElementAnimations
│           ├── keyframe_effects_list_: ObserverList<KeyframeEffect>
│           ├── active_state_ / pending_state_: PropertyAnimationState
│           └── → MutatorHostClient (LayerTreeHostImpl)
│                     └── → ScrollTree / TransformTree / ... 属性树节点
│
├── ticking_animations_: [Animation, ...]        (正在运行的动画子集)
│
├── scroll_offset_animations_: ScrollOffsetAnimations          [主线程]
│     └── element_to_update_map_: {ElementId → ScrollOffsetAnimationUpdate}
│
└── scroll_offset_animations_impl_: ScrollOffsetAnimationsImpl [impl 线程]
      └── element_to_animation_map_: {ElementId → ScrollOffsetAnimationImpl}
            └── ScrollOffsetAnimationImpl
                  ├── scroll_offset_timeline_: AnimationTimeline (impl-only)
                  └── scroll_offset_animation_: Animation (impl-only)
```

---

# 九、关键设计要点

| 要点 | 说明 |
|------|------|
| **双线程分工** | 每个线程独立持有一套 `AnimationHost`，通过 `PushPropertiesTo()` 在 commit 时同步 |
| **Impl-only 动画** | 鼠标滚轮等高频交互完全在 impl 线程创建动画，避免往返主线程的延迟 |
| **ElementAnimations 的聚合作用** | 同一元素的所有动画效果汇聚到一个 `ElementAnimations`，统一分发到属性树 |
| **ScrollTimeline 的时间替换** | 用滚动位置替代墙上时钟作为"时间"，实现 CSS 滚动联动动画，对下层 Tick 路径透明 |
| **三级策略区分** | `DurationBehavior` 区分不同滚动触发类型的动画时长感：kDeltaBased（线性）/ kConstant / kInverseDelta（反比，滚轮触感更弹性）|
| **Takeover 机制** | 当主线程需要接管 impl 正在执行的滚动动画时，通过 `AddTakeoverUpdate` → `ScrollAnimationAbort` + `NotifyAnimationTakeover` 完成交接 |
| **WorkletAnimation** | 完全由 JS AnimationWorklet 控制时序，支持高级自定义滚动动画效果 |