# ScrollAnimation 从创建到执行的详细流程

> 本文是 `cc-animation-analysis.md` 的补充，聚焦于 **Impl-Only ScrollAnimation**（鼠标滚轮/键盘/programmatic 触发的平滑滚动）从创建到每帧执行的完整代码路径。

---

# 一、总览：两个独立阶段

```
[创建阶段]  输入事件 → 构造对象链 → 挂载元素 → 加入 ticking 列表
[执行阶段]  每帧 TickAnimations → 曲线插值 → 写入 ScrollTree
```

两个阶段完全在 **impl 线程**上发生，无需主线程 commit 介入。

---

# 二、创建阶段：对象链的构造

## 2.1 调用入口

```
// 来自 LayerTreeHostImpl，响应鼠标滚轮事件
AnimationHost::ImplOnlyScrollAnimationCreate(
    ElementId element_id,
    const gfx::PointF& target_offset,   // 目标滚动位置
    const gfx::PointF& current_offset,  // 当前滚动位置
    base::TimeDelta delayed_by,
    base::TimeDelta animation_start_offset)
```

`AnimationHost` 直接委托给 `scroll_offset_animations_impl_`（impl 线程专属）：

```cpp
// animation_host.cc
void AnimationHost::ImplOnlyScrollAnimationCreate(...) {
  scroll_offset_animations_impl_.Write(*this)->MouseWheelScrollAnimationCreate(
      element_id, target_offset, current_offset, delayed_by,
      animation_start_offset);
}
```

## 2.2 ScrollOffsetAnimationsImpl：查找或创建 per-element 动画

```cpp
// scroll_offset_animations_impl.cc
void ScrollOffsetAnimationsImpl::MouseWheelScrollAnimationCreate(
    ElementId element_id, ...) {
  // 每个 element_id 对应一个 ScrollOffsetAnimationImpl 实例
  // 若已存在，覆盖（旧动画会被 Abort）
  element_to_animation_map_.insert(
      std::pair(element_id,
                std::make_unique<ScrollOffsetAnimationImpl>(animation_host_)));
  element_to_animation_map_.at(element_id)->MouseWheelScrollAnimationCreate(
      element_id, target_offset, current_offset, delayed_by,
      animation_start_offset);
}
```

`element_to_animation_map_` 结构：
```
ScrollOffsetAnimationsImpl
  └── element_to_animation_map_: {ElementId → ScrollOffsetAnimationImpl}
```

## 2.3 ScrollOffsetAnimationImpl 构造函数：预建 Timeline + Animation

```cpp
// scroll_offset_animations_impl.cc
ScrollOffsetAnimationImpl::ScrollOffsetAnimationImpl(AnimationHost* host)
    : animation_host_(host),
      scroll_offset_timeline_(
          AnimationTimeline::Create(AnimationIdProvider::NextTimelineId(),
                                    /* is_impl_only */ true)),
      scroll_offset_animation_(
          Animation::Create(AnimationIdProvider::NextAnimationId())) {
  scroll_offset_animation_->set_animation_delegate(this);  // 注册完成回调
  // 添加到animation_host
  animation_host_->AddAnimationTimeline(scroll_offset_timeline_.get());
  scroll_offset_timeline_->AttachAnimation(scroll_offset_animation_.get());
}
```

**关键点：** `AnimationTimeline` 和 `Animation` 在构造时已创建并关联，是 **impl-only** 对象（不同步到主线程）。

## 2.4 曲线工厂：创建 ScrollOffsetAnimationCurve

```cpp
// scroll_offset_animations_impl.cc
void ScrollOffsetAnimationImpl::MouseWheelScrollAnimationCreate(
    ElementId element_id, const gfx::PointF& target_offset,
    const gfx::PointF& current_offset,
    base::TimeDelta delayed_by, ...) {

  // 步骤1：工厂创建曲线
  std::unique_ptr<ScrollOffsetAnimationCurve> curve =
      ScrollOffsetAnimationCurveFactory::CreateAnimation(
          target_offset, ScrollOffsetAnimationCurve::ScrollType::kMouseWheel);

  // 步骤2：设置起始值，计算动画时长
  curve->SetInitialValue(current_offset, delayed_by);

  ScrollAnimationCreateInternal(element_id, std::move(curve), ...);
}
```

**工厂内部逻辑（`ScrollOffsetAnimationCurveFactory::CreateAnimation`）：**

```cpp
// scroll_offset_animation_curve_factory.cc
std::unique_ptr<ScrollOffsetAnimationCurve>
ScrollOffsetAnimationCurveFactory::CreateAnimation(
    const gfx::PointF& target_value,
    ScrollOffsetAnimationCurve::ScrollType scroll_type) {
  // kMouseWheel → kInverseDelta（与位移反比，短距离动画更长，体感更"弹性"）
  // kProgrammatic → kDeltaBased（与位移正比）
  // kKeyboard → kConstant（固定时长 9/60 秒）
  // kAutoScroll → Linear（匀速）
  if (scroll_type == ScrollType::kAutoScroll)
    return CreateLinearAnimation(target_value);
  return CreateEaseInOutAnimation(
      target_value, scroll_type,
      GetDurationBehaviorFromScrollType(scroll_type));
}
```

**曲线时长计算（`SetInitialValue` 内部）：**

```cpp
// scroll_offset_animation_curve.cc
void ScrollOffsetAnimationCurve::SetInitialValue(
    const gfx::PointF& initial_value,
    base::TimeDelta delayed_by, float velocity) {
  initial_value_ = initial_value;
  has_set_initial_value_ = true;
  gfx::Vector2dF delta = target_value_ - initial_value;
  // 计算 total_animation_duration_，根据 DurationBehavior：
  // kInverseDelta: duration = clamp(offset + |delta| * slope, 6/60s, 12/60s)
  // kDeltaBased:   duration = sqrt(|delta|) / 60（上限受 feature flag 控制）
  // kConstant:     duration = 9/60s
  total_animation_duration_ = SegmentDuration(delta, delayed_by, velocity);
}
```

## 2.5 ScrollAnimationCreateInternal：组装 KeyframeModel

```cpp
// scroll_offset_animations_impl.cc
void ScrollOffsetAnimationImpl::ScrollAnimationCreateInternal(
    ElementId element_id,
    std::unique_ptr<gfx::AnimationCurve> curve,
    base::TimeDelta animation_start_offset) {

  // 步骤1：包装成 KeyframeModel，目标属性 = SCROLL_OFFSET
  std::unique_ptr<KeyframeModel> keyframe_model = KeyframeModel::Create(
      std::move(curve),
      AnimationIdProvider::NextKeyframeModelId(),
      AnimationIdProvider::NextGroupId(),
      KeyframeModel::TargetPropertyId(TargetProperty::SCROLL_OFFSET));
  keyframe_model->set_time_offset(animation_start_offset);
  keyframe_model->SetIsImplOnly();  // 标记为 impl-only，不推送到主线程

  // 步骤2：若 element_id 变了，重新绑定元素（会触发 ElementAnimations 创建）
  ReattachScrollOffsetAnimationIfNeeded(element_id);

  // 步骤3：将 KeyframeModel 加入 Animation
  scroll_offset_animation_->AddKeyframeModel(std::move(keyframe_model));
}
```

## 2.6 ReattachScrollOffsetAnimationIfNeeded：绑定元素，创建 ElementAnimations

```cpp
void ScrollOffsetAnimationImpl::ReattachScrollOffsetAnimationIfNeeded(
    ElementId element_id) {
  if (scroll_offset_animation_->element_id() != element_id) {
    if (scroll_offset_animation_->element_id())
      scroll_offset_animation_->DetachElement();  // 解绑旧元素
    if (element_id)
      scroll_offset_animation_->AttachElement(element_id);  // 绑定新元素
  }
}
```

`Animation::AttachElement` 内部：

```cpp
// animation.cc
void Animation::AttachElementInternal(ElementId element_id) {
  keyframe_effect()->AttachElement(element_id);  // KeyframeEffect 记录 element_id
  if (animation_host())
    RegisterAnimation();  // 触发 ElementAnimations 的创建
}

void Animation::RegisterAnimation() {
  animation_host()->RegisterAnimationForElement(
      keyframe_effect()->element_id(), this);
}
```

`AnimationHost::RegisterAnimationForElement`：

```cpp
// animation_host.cc
void AnimationHost::RegisterAnimationForElement(ElementId element_id,
                                                Animation* animation) {
  // 查找或创建 ElementAnimations（同一元素共享一个实例）
  scoped_refptr<ElementAnimations> element_animations =
      GetElementAnimationsForElementId(element_id);
  if (!element_animations) {
    element_animations = ElementAnimations::Create(this, element_id);
    element_to_animations_map_.Write(*this)[element_id] = element_animations;
  }
  // 将此 Animation 的 KeyframeEffect 注册进去（加入 ObserverList）
  element_animations->AddKeyframeEffect(animation->keyframe_effect());
}
```

## 2.7 ElementAnimations::AddKeyframeEffect → BindElementAnimations

`RegisterAnimationForElement` 最后调用：

```cpp
element_animations->AddKeyframeEffect(animation->keyframe_effect());
```

内部：

```cpp
// element_animations.cc
void ElementAnimations::AddKeyframeEffect(KeyframeEffect* keyframe_effect) {
  keyframe_effects_list_.AddObserver(keyframe_effect);   // 加入 ObserverList
  keyframe_effect->BindElementAnimations(this);          // 反向绑定
}
```

`KeyframeEffect::BindElementAnimations`：

```cpp
// keyframe_effect.cc
void KeyframeEffect::BindElementAnimations(
    ElementAnimations* element_animations) {
  DCHECK(!element_animations_);
  element_animations_ = element_animations;  // KeyframeEffect 持有 ElementAnimations 指针

  // 若 KeyframeModel 已经存在（先加 model 再绑定元素的情况），立即触发 AttachToCurve
  if (has_any_keyframe_model())
    KeyframeModelAdded();
  SetNeedsPushProperties();
}
```

**在 impl-only 滚动动画流程中**，`BindElementAnimations` 执行时 KeyframeModel **尚未**添加（`has_any_keyframe_model() == false`），所以 `KeyframeModelAdded()` 不会在这里调用，绑定曲线的动作推迟到下一步。

---

## 2.8 Animation::AddKeyframeModel → KeyframeModelAdded → AttachToCurve（核心绑定）

这是 `ScrollOffsetAnimationCurve` 与 `ElementAnimations` 完成关联的**实际时机**。

`scroll_offset_animation_->AddKeyframeModel(...)` 调用的是 `cc::Animation::AddKeyframeModel`，它是一个单行转发，不含任何逻辑：

```cpp
// cc/animation/animation.cc
void Animation::AddKeyframeModel(
    std::unique_ptr<KeyframeModel> keyframe_model) {
  keyframe_effect()->AddKeyframeModel(std::move(keyframe_model));  // 直接转发给 KeyframeEffect
}
```

`Animation` 在架构上是分组容器和生命周期入口，真正持有并驱动 `KeyframeModel` 的是其 1:1 拥有的 `KeyframeEffect`。实际逻辑在：

```cpp
// cc/animation/keyframe_effect.cc
void KeyframeEffect::AddKeyframeModel(
    std::unique_ptr<gfx::KeyframeModel> keyframe_model) {
  // ... DCHECK 省略 ...

  // 步骤1：调用基类，将 model 追加到 keyframe_models_ 列表
  gfx::KeyframeEffect::AddKeyframeModel(std::move(keyframe_model));

  // 步骤2：若 ElementAnimations 已绑定，触发后续关联
  if (has_bound_element_animations()) {
    KeyframeModelAdded();         // ← 核心：完成 curve → target 绑定
    SetNeedsPushProperties();
  }
}
```

`KeyframeEffect::KeyframeModelAdded`：

```cpp
void KeyframeEffect::KeyframeModelAdded() {
  DCHECK(has_bound_element_animations());

  animation_->SetNeedsCommit();
  needs_to_start_keyframe_models_ = true;

  UpdateTickingState();  // 将 Animation 加入 ticking_animations_

  // 遍历所有 keyframe_model，为每条曲线设置 target
  for (auto& keyframe_model : keyframe_models()) {
    element_animations_->AttachToCurve(keyframe_model->curve());
  }
  element_animations_->UpdateClientAnimationState();
}
```

`ElementAnimations::AttachToCurve`：

```cpp
// element_animations.cc
void ElementAnimations::AttachToCurve(gfx::AnimationCurve* c) {
  switch (c->Type()) {
    case gfx::AnimationCurve::SCROLL_OFFSET:
      ScrollOffsetAnimationCurve::ToScrollOffsetAnimationCurve(c)
          ->set_target(this);   // ← curve.target_ = ElementAnimations*
      break;
    case gfx::AnimationCurve::TRANSFORM:
      gfx::TransformAnimationCurve::ToTransformAnimationCurve(c)
          ->set_target(this);
      break;
    // ... 其他类型 ...
  }
}
```

至此，`ScrollOffsetAnimationCurve::target_` 指向 `ElementAnimations`，每帧 `Tick` 时曲线可以直接回调：

```
curve->Tick(t)
  └── target_->OnScrollOffsetAnimated(GetValue(t), ...)
        └── MutatorHostClient::SetElementScrollOffsetMutated(...)
```

---

### 关联建立的两种时序

`BindElementAnimations` 与 `AddKeyframeModel` 的调用顺序在不同场景下不同，但最终都会触发 `KeyframeModelAdded → AttachToCurve`：

| 场景 | 顺序 | 触发点 |
|---|---|---|
| **impl-only 滚动动画**（本文主路径）| 先 `AttachElement`→`BindElementAnimations`，后 `AddKeyframeModel` | `AddKeyframeModel` 内，`has_bound_element_animations()==true` 时 |
| **主线程动画 push 到 impl**（commit 路径）| 先 `AddKeyframeModel`（push 时），后 `BindElementAnimations`（注册元素时）| `BindElementAnimations` 内，`has_any_keyframe_model()==true` 时 |

两路都通过 `KeyframeModelAdded` 汇合，保证曲线必然被绑定。

---

## 2.9 Animation::AddKeyframeModel 后：加入 ticking 列表

`KeyframeModelAdded` 内部的 `UpdateTickingState` 检测到有 `WAITING_FOR_TARGET_AVAILABILITY` 状态的 model，调用：

```
KeyframeEffect::UpdateTickingState()
  └── animation_->AddToTicking()
        └── AnimationHost::AddToTicking(animation)
              └── ticking_animations_.push_back(animation)
```

新加入的 KeyframeModel 初始状态为 `WAITING_FOR_TARGET_AVAILABILITY`，Animation 就此进入每帧被 Tick 的列表。

---

# 三、创建阶段完整对象关系图

```
AnimationHost (impl)
├── scroll_offset_animations_impl_: ScrollOffsetAnimationsImpl
│     └── element_to_animation_map_: {ElementId → ScrollOffsetAnimationImpl}
│           └── ScrollOffsetAnimationImpl
│                 ├── scroll_offset_timeline_: AnimationTimeline (impl-only, is_impl_only=true)
│                 │     └── scroll_offset_animation_: Animation
│                 │           └── keyframe_effect_: KeyframeEffect
│                 │                 └── keyframe_models_[0]: KeyframeModel
│                 │                       ├── run_state: WAITING_FOR_TARGET_AVAILABILITY
│                 │                       └── curve_: ScrollOffsetAnimationCurve
│                 │                             ├── initial_value_: {x0, y0}
│                 │                             ├── target_value_: {xt, yt}
│                 │                             ├── total_animation_duration_: Δt
│                 │                             └── timing_function_: CubicBezierTimingFunction
│                 └── animation_delegate: this (ScrollOffsetAnimationImpl)
│
├── ticking_animations_: [... scroll_offset_animation_ ...]  ← 已加入
│
└── element_to_animations_map_: {ElementId → ElementAnimations}
      └── ElementAnimations
            ├── element_id_: <target ElementId>
            ├── keyframe_effects_list_: [KeyframeEffect*]  ← 指向上面的 KeyframeEffect
            └── animation_host_: AnimationHost*
```

---

# 四、执行阶段：每帧 Tick

## 4.1 入口：LayerTreeHostImpl 每帧调用

```
LayerTreeHostImpl::AnimateLayers(monotonic_time)
  └── AnimationHost::TickAnimations(monotonic_time, scroll_tree,
                                    is_active_tree, mutator_events)
```

## 4.2 AnimationHost::TickAnimations：分类 tick

```cpp
// animation_host.cc
bool AnimationHost::TickAnimations(base::TimeTicks monotonic_time,
                                   const ScrollTree& scroll_tree,
                                   bool is_active_tree,
                                   MutatorEvents* mutator_events) {
  // 快速路径：无需 tick
  if (is_active_tree && !NeedsTickAnimations())
    return false;

  // 先更新 AnimationTrigger（影响动画是否生效）
  if (is_active_tree)
    UpdateTriggers(scroll_tree, animation_events);

  // 遍历所有 timeline，分两类处理：
  std::vector<AnimationTimeline*> scroll_timelines;
  for (auto& kv : id_to_timeline_map_) {
    AnimationTimeline* timeline = kv.second.get();
    if (timeline->IsScrollTimeline()) {
      scroll_timelines.push_back(timeline);  // 滚动联动动画，延后处理
    } else {
      // 时间联动动画（含 impl-only 滚动动画）
      animated |= timeline->TickTimeLinkedAnimations(
          ticking_animations_, monotonic_time, !is_active_tree);
    }
  }

  // 滚动联动动画最后处理（平滑滚动可能先更新 scroll offset）
  for (auto* timeline : scroll_timelines) {
    animated |= timeline->TickScrollLinkedAnimations(
        ticking_animations_, scroll_tree, is_active_tree);
  }

  TickMutator(monotonic_time, scroll_tree, is_active_tree);  // WorkletAnimation
  return animated;
}
```

**impl-only 滚动动画走"时间联动"路径**（非 ScrollTimeline），因为它的 timeline `is_impl_only=true` 但不是 `ScrollTimeline`。

## 4.3 AnimationTimeline::TickTimeLinkedAnimations

```cpp
// animation_timeline.cc
bool AnimationTimeline::TickTimeLinkedAnimations(
    const AnimationsList& ticking_animations,
    base::TimeTicks monotonic_time, bool tick_finished) {
  bool animated = false;
  for (auto& animation : ticking_animations) {
    // 只处理属于本 timeline 的动画
    if (animation->animation_timeline() != this)
      continue;
    if (!animation->IsScrollLinkedAnimation()) {
      animated |= animation->Tick(monotonic_time);
    }
  }
  return animated;
}
```

## 4.4 Animation::Tick → KeyframeEffect::Tick

```cpp
// animation.cc
bool Animation::Tick(base::TimeTicks monotonic_time) {
  return keyframe_effect()->Tick(monotonic_time);
}
// keyframe_effect.cc
bool KeyframeEffect::Tick(base::TimeTicks monotonic_time) {
  // 阶段1：将 WAITING_FOR_TARGET_AVAILABILITY → STARTING
  if (needs_to_start_keyframe_models_)
    StartKeyframeModels(monotonic_time);

  bool animated = false;
  for (auto& keyframe_model : keyframe_models()) {
    // 只 tick RUNNING 或 STARTING 状态的 model
    if (keyframe_model->run_state() != KeyframeModel::RUNNING &&
        keyframe_model->run_state() != KeyframeModel::STARTING)
      continue;

    // 阶段2：调用底层 gfx::KeyframeModel::Tick，触发曲线计算
    animated = true;
    keyframe_model->Tick(monotonic_time);
    // 内部：curve_->Tick(trimmed_time, property_id, this)
    // → ScrollOffsetAnimationCurve::Tick(t, ...)
    //   → target_->OnScrollOffsetAnimated(GetValue(t), ...)
  }

  // 阶段3：标记已完成的 model
  last_tick_time_ = monotonic_time;
  MarkFinishedKeyframeModels(monotonic_time);
  return animated;
}
```

## 4.5 KeyframeEffect::StartKeyframeModels：状态机推进

```cpp
// keyframe_effect.cc
void KeyframeEffect::StartKeyframeModels(base::TimeTicks monotonic_time) {
  // 检查是否有 WAITING_FOR_TARGET_AVAILABILITY 状态的 model
  for (auto& keyframe_model : keyframe_models()) {
    auto* cc_model = KeyframeModel::ToCcKeyframeModel(keyframe_model.get());
    if (cc_model->run_state() ==
        gfx::KeyframeModel::WAITING_FOR_TARGET_AVAILABILITY) {
      // 检查没有其他 model 正在占用同一属性（SCROLL_OFFSET 在此场景只有一个）
      // 将状态推进到 STARTING，记录 start_time
      cc_model->SetRunState(gfx::KeyframeModel::STARTING, monotonic_time);
    }
  }
}
```

**状态流转时序：**
```
创建时:            WAITING_FOR_TARGET_AVAILABILITY
第1帧 StartKeyframeModels: → STARTING  (start_time 设置为本帧时间)
第1帧 UpdateState/PromoteStarted: → RUNNING
后续帧 Tick: 持续 RUNNING，每帧调用曲线插值
动画结束: → FINISHED → WAITING_FOR_DELETION
```

## 4.6 ScrollOffsetAnimationCurve::Tick 与插值

```cpp
// scroll_offset_animation_curve.cc
void ScrollOffsetAnimationCurve::Tick(
    base::TimeDelta t,           // 已由 KeyframeModel 裁剪为 [0, duration]
    int property_id,
    gfx::KeyframeModel* keyframe_model,
    ...) const {
  if (target_) {
    target_->OnScrollOffsetAnimated(GetValue(t), property_id, keyframe_model);
  }
}

gfx::PointF ScrollOffsetAnimationCurve::GetValue(base::TimeDelta t) const {
  const base::TimeDelta duration = total_animation_duration_ - last_retarget_;
  t -= last_retarget_;

  if (t >= duration) return target_value_;   // 动画结束
  if (t <= base::TimeDelta()) return initial_value_;  // 动画未开始

  // 核心插值：timing_function_->GetValue(t/duration) → [0,1]
  const double progress = timing_function_->GetValue(t / duration, ...);
  return gfx::PointF(
      gfx::Tween::FloatValueBetween(progress, initial_value_.x(), target_value_.x()),
      gfx::Tween::FloatValueBetween(progress, initial_value_.y(), target_value_.y()));
}
```

`timing_function_` 对于 `kMouseWheel` 是 `CubicBezierTimingFunction(0.42, 0, 0.58, 1)` (ease-in-out)。

## 4.7 ElementAnimations::OnScrollOffsetAnimated：写入属性树

```cpp
// element_animations.cc
void ElementAnimations::OnScrollOffsetAnimated(
    const gfx::PointF& scroll_offset,
    int target_property_id,
    gfx::KeyframeModel* keyframe_model) {
  // 根据 KeyframeModel 的 affects_active/pending 标志，分别写入两棵树
  if (KeyframeModelAffectsActiveElements(keyframe_model))
    OnScrollOffsetAnimated(ElementListType::ACTIVE, scroll_offset, keyframe_model);
  if (KeyframeModelAffectsPendingElements(keyframe_model))
    OnScrollOffsetAnimated(ElementListType::PENDING, scroll_offset, keyframe_model);
}

void ElementAnimations::OnScrollOffsetAnimated(
    ElementListType list_type,
    const gfx::PointF& scroll_offset,
    gfx::KeyframeModel* keyframe_model) {
  ElementId target_element_id = CalculateTargetElementId(this, keyframe_model);
  // 通过 MutatorHostClient（= LayerTreeHostImpl）写入 ScrollTree
  animation_host_->mutator_host_client()->SetElementScrollOffsetMutated(
      target_element_id, list_type, scroll_offset);
  // 最终: PropertyTrees::scroll_tree().SetScrollOffset(element_id, scroll_offset)
  //       → 更新 ScrollNode 的 current_offset（pending/active tree）
}
```

## 4.8 UpdateAnimationState：状态推进（与 Tick 分离）

在 Tick 之后，`LayerTreeHostImpl` 还会调用：

```
AnimationHost::UpdateAnimationState(start_ready_animations=true, events)
  └── for each ticking Animation:
        Animation::UpdateState(start_ready_animations, animation_events)
          └── KeyframeEffect::UpdateState(...)
                ├── PromoteStartedKeyframeModels(events)  // STARTING → RUNNING + 发出 START 事件
                └── MarkKeyframeModelsForDeletion(...)    // FINISHED → WAITING_FOR_DELETION
```

`PromoteStartedKeyframeModels` 将第一帧 `STARTING` 的 model 推进到 `RUNNING`，同时生成 `AnimationPlaybackEvent::START` 事件。

---

# 五、Retargeting：动画进行中更新目标

当用户在动画进行中继续滚动，需要平滑重定向目标：

```
AnimationHost::ImplOnlyScrollAnimationUpdateTarget(
    scroll_delta, max_offset, frame_time, delayed_by, element_id)
  └── ScrollOffsetAnimationsImpl::ScrollAnimationUpdateTarget(...)
        └── ScrollOffsetAnimationImpl::ScrollAnimationUpdateTarget(...)
              ├── 获取 KeyframeModel 和 ScrollOffsetAnimationCurve
              ├── new_target = curve->target_value() + scroll_delta (clamp to max)
              ├── trimmed = keyframe_model->TrimTimeToCurrentIteration(frame_time)
              │    // 计算当前已过去的时间 t
              └── curve->UpdateTarget(trimmed - delayed_by, new_target)
```

`ScrollOffsetAnimationCurve::UpdateTarget` 的核心逻辑：

```cpp
void ScrollOffsetAnimationCurve::UpdateTarget(base::TimeDelta t,
                                               const gfx::PointF& new_target) {
  gfx::PointF current_position = GetValue(t);    // 当前插值位置
  gfx::Vector2dF new_delta = new_target - current_position;

  // 计算新时长：取 EaseInOut 时长 和 速度推算时长 的较小值
  // 避免"橡皮筋"效果：当速度大、new_delta 小时不要生成过长动画
  const base::TimeDelta new_duration =
      EaseInOutBoundedSegmentDuration(new_delta, t, delayed_by);

  // 关键：保留当前速度，调整新曲线斜率
  double velocity = CalculateVelocity(t);
  double new_slope = velocity * (new_duration / MaximumDimension(new_delta));
  timing_function_ = GetEasingFunction(new_slope);  // 新的 CubicBezier

  // 更新曲线参数
  initial_value_ = current_position;
  target_value_ = new_target;
  total_animation_duration_ = t + new_duration;
  last_retarget_ = t;   // GetValue() 中用于区分分段
}
```

---

# 六、动画结束

```
KeyframeEffect::MarkFinishedKeyframeModels(monotonic_time)
  └── 若 monotonic_time ≥ start_time + total_animation_duration_:
        keyframe_model->SetRunState(FINISHED, monotonic_time)

AnimationHost::UpdateAnimationState(...)
  └── KeyframeEffect::UpdateState(...)
        └── MarkKeyframeModelsForDeletion()
              └── FINISHED → WAITING_FOR_DELETION
              └── 触发 AnimationDelegate::NotifyAnimationFinished()

ScrollOffsetAnimationImpl::NotifyAnimationFinished(...)
  └── animation_host_->mutator_host_client()->ScrollOffsetAnimationFinished(
          scroll_offset_animation_->element_id())
      // LayerTreeHostImpl 响应：通知 Blink 滚动结束，处理 snap 等后续逻辑
```

---

# 七、完整时序图（单次鼠标滚轮事件）

```
impl 线程
│
│ [事件到达]
├─ LayerTreeHostImpl::ScrollBegin / ScrollUpdate
│    └─ AnimationHost::ImplOnlyScrollAnimationCreate(element, target, current)
│         └─ ScrollOffsetAnimationsImpl::MouseWheelScrollAnimationCreate()
│               └─ new ScrollOffsetAnimationImpl(host)          // 若首次
│                    ├─ new AnimationTimeline (impl-only)
│                    ├─ new Animation
│                    └─ AttachAnimation + AddAnimationTimeline
│               └─ ScrollOffsetAnimationImpl::MouseWheelScrollAnimationCreate()
│                    ├─ ScrollOffsetAnimationCurveFactory::CreateAnimation()
│                    │    └─ new ScrollOffsetAnimationCurve(target, EaseInOut, kInverseDelta)
│                    ├─ curve->SetInitialValue(current, delayed_by)
│                    │    └─ total_animation_duration_ = f(delta)   // 6~12帧
│                    └─ ScrollAnimationCreateInternal()
│                         ├─ KeyframeModel::Create(curve, id, SCROLL_OFFSET)
│                         │    └─ run_state = WAITING_FOR_TARGET_AVAILABILITY
│                         ├─ ReattachScrollOffsetAnimationIfNeeded(element)
│                         │    └─ Animation::AttachElement()
│                         │         └─ RegisterAnimation()
│                         │              └─ ElementAnimations::Create (若首次)
│                         └─ Animation::AddKeyframeModel()
│                              └─ AddToTicking() → ticking_animations_.push_back()
│
│ [帧N，第1帧]
├─ AnimationHost::TickAnimations(t₀)
│    └─ AnimationTimeline::TickTimeLinkedAnimations()
│         └─ Animation::Tick(t₀)
│              └─ KeyframeEffect::Tick(t₀)
│                   ├─ StartKeyframeModels(t₀)
│                   │    └─ WAITING_FOR_TARGET_AVAILABILITY → STARTING (start_time=t₀)
│                   └─ [无 RUNNING 的 model，本帧不输出值]
├─ AnimationHost::UpdateAnimationState()
│    └─ KeyframeEffect::UpdateState()
│         └─ PromoteStartedKeyframeModels()
│              └─ STARTING → RUNNING + 发出 START 事件
│
│ [帧N+1 ~ 帧N+k，执行中]
├─ AnimationHost::TickAnimations(tₙ)
│    └─ KeyframeEffect::Tick(tₙ)
│         └─ KeyframeModel::Tick(tₙ)   [run_state=RUNNING]
│              └─ ScrollOffsetAnimationCurve::Tick(trimmed_t)
│                   ├─ GetValue(trimmed_t)
│                   │    └─ progress = timing_function_->GetValue(trimmed_t / duration)
│                   │    └─ return lerp(initial_value_, target_value_, progress)
│                   └─ target_->OnScrollOffsetAnimated(value, ...)
│                        └─ ElementAnimations::OnScrollOffsetAnimated()
│                             └─ MutatorHostClient::SetElementScrollOffsetMutated()
│                                  └─ ScrollTree::SetScrollOffset(element_id, value)
│                                       // 页面滚动位置更新
│
│ [若中途有新滚动事件]
├─ AnimationHost::ImplOnlyScrollAnimationUpdateTarget(delta, max, tₙ)
│    └─ ScrollOffsetAnimationCurve::UpdateTarget(trimmed_t, new_target)
│         ├─ current_pos = GetValue(trimmed_t)
│         ├─ new_duration = min(EaseInOut时长, 速度推算时长)
│         ├─ 保留当前速度，重新计算曲线斜率
│         └─ 更新 initial/target/duration/last_retarget_
│
│ [帧N+k，最后一帧]
├─ KeyframeEffect::MarkFinishedKeyframeModels(tₙ)
│    └─ tₙ ≥ start_time + duration → RUNNING → FINISHED
├─ AnimationHost::UpdateAnimationState()
│    └─ FINISHED → WAITING_FOR_DELETION
│    └─ AnimationDelegate::NotifyAnimationFinished()
│         └─ ScrollOffsetAnimationImpl::NotifyAnimationFinished()
│              └─ MutatorHostClient::ScrollOffsetAnimationFinished(element_id)
│                   // LayerTreeHostImpl 通知 Blink，处理 snap/afterscroll 等
│
```

---

# 八、关键细节补充

## 8.1 `last_retarget_` 的作用

`ScrollOffsetAnimationCurve` 通过 `last_retarget_` 支持多段连续 Retargeting，`GetValue(t)` 每次只计算**从最后一次重定向时刻到现在的进度**：

```cpp
gfx::PointF GetValue(base::TimeDelta t) const {
  const base::TimeDelta duration = total_animation_duration_ - last_retarget_;
  t -= last_retarget_;
  // ...
}
```

每次 `UpdateTarget` 都会推进 `last_retarget_`，因此可以无限次重定向，每次都从当前速度平滑过渡。

## 8.2 `affects_active_elements` vs `affects_pending_elements`

`KeyframeModel` 有两个标志：
- `affects_active_elements_`（默认 true）：写入 active tree 的 ScrollNode
- `affects_pending_elements_`（默认 true）：写入 pending tree 的 ScrollNode

`ScrollAnimationApplyAdjustment`（锚点调整）创建新 model 时会将 `affects_active_elements=false`，因为调整只应用于 pending tree，等 activation 后再生效。

## 8.3 `needs_synchronized_start_time_` 与 impl-only 动画

impl-only 动画通过 `KeyframeModel::SetIsImplOnly()` 标记，此时 `needs_synchronized_start_time_=false`，不需要等待主线程的 `start_time` 同步。`StartKeyframeModels` 可以立即在第一帧设置 `start_time`，无需等待 commit 返回。

## 8.4 ScrollOffsetAnimationsImpl 的单例动画复用

每个 `ScrollOffsetAnimationImpl` 内部只持有**一个** `Animation` 和**一个** `AnimationTimeline`。新的 `AddKeyframeModel` 会追加到同一个 `KeyframeEffect`。
旧动画通过 `AbortKeyframeModelsWithProperty(SCROLL_OFFSET)` 中止后，新 model 立即加入，实现无缝的 retargeting 过渡（与 `UpdateTarget` 是两种不同粒度的重定向机制：前者重建 model，后者就地修改曲线参数）。