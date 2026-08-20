# 概述

**CAP (Compositing After Paint)** 是 Chromium/Blink 中连接 Paint 和 Compositor 的关键架构。它的核心作用是将 Blink Paint 阶段生成的 `PaintArtifact` 转换为 Compositor 线程能理解的 `cc::Layer` 树和 `cc::DisplayItemList`。

## 为什么需要 CAP？

在 CAP 架构之前，Blink 使用的是 **Pre-Composite** 模型：

- **问题**：在 Paint 之前就决定分层策略，很多情况下过度保守
- **限制**：属性变化（如 transform、opacity）需要重新 Paint
- **复杂性**：Paint 和 Composite 逻辑耦合

**CAP 的优势**：

1. **延迟合成决策**：在 Paint 完成后再决定如何分层，更灵活准确
2. **减少重绘**：属性变化不需要重新 Paint，只需更新 cc 属性树
3. **更好的缓存**：DisplayItem 缓存独立于 layer 结构
4. **统一架构**：所有平台使用相同的 Paint → CAP → Raster 流程
5. **优化机会**：可以根据实际绘制结果做更智能的分层优化

---

# CAP 的核心流程：PaintArtifact → cc::Layer

整个 CAP 流程发生在 `PaintArtifactCompositor::Update` 中，分为三个阶段：

```
Blink 主线程:
  PaintArtifact (Blink 格式)
    ↓
  PaintArtifactCompositor::Update
    ↓
  【阶段 1: Layerize】分析分层策略
    ↓ 输出：PendingLayers (决定哪些 chunks 合并/独立)
    ↓
  【阶段 2: UpdateCompositedLayer】创建 cc::Layer 和转换内容
    ↓ 为每个 PendingLayer 创建 cc::Layer
    ↓ PaintChunksToCcLayer::ConvertInto
    ↓ 输出：cc::DisplayItemList (包含 cc::DrawRecordOp)
    ↓
  【阶段 3: PropertyTreeManager】创建属性树并关联
    ↓ 创建 cc 的 Transform/Clip/Effect 属性树节点
    ↓ 将节点 ID 设置到 cc::Layer
    ↓
  输出：cc::LayerTreeHost (包含 layer 树和属性树)
```

---

# 阶段 1：Layerize - 分层决策

## 概述

Layerizer 负责分析 `PaintArtifact` 的所有 `PaintChunk`，决定：

- 哪些 chunks 可以合并到同一个 composited layer
- 哪些需要独立成层（因为 overlap、transform、will-change 等）
- 如何处理 effect 和 isolation

## 核心代码

```cpp
void PaintArtifactCompositor::Update(
    const PaintArtifact& artifact,
    const ViewportProperties& viewport_properties,
    ...) {

  // 1. 【Layerize】分析 PaintArtifact，生成 PendingLayers
  pending_layers_ = Layerizer(*this, artifact, old_size).Layerize();
  PendingLayer::DecompositeTransforms(pending_layers_);

  // Layerizer 内部逻辑：
  // - 遍历所有 PaintChunks
  // - 对于每个 chunk，检查是否能合并到当前 PendingLayer
  // - 检查合并条件：PropertyTreeState 兼容、无 overlap、无强制分层原因
  // - 如果不能合并，创建新的 PendingLayer
  // - 考虑 decompositing：某些 effect 可以被"拆解"避免单独分层
}
```

## Layerizer::Layerize 的工作流程

```cpp
class PaintArtifactCompositor::Layerizer {
 public:
  PendingLayers Layerize() {
    // 从 Root effect 开始递归处理
    LayerizeGroup(EffectPaintPropertyNode::Root(), false);
    return std::move(pending_layers_);
  }

 private:
  void LayerizeGroup(const EffectPaintPropertyNode& current_group,
                     bool force_draws_content) {
    // 遍历属于当前 effect group 的 chunks
    while (chunk_cursor_ != artifact_.GetPaintChunks().end()) {
      const auto& chunk_effect = chunk_cursor_->properties.Effect();

      if (&chunk_effect == &current_group) {
        // Case A: chunk 属于当前 group
        pending_layers_.emplace_back(artifact_, *chunk_cursor_, ...);
        ++chunk_cursor_;

        // 尝试与之前的 layer 合并
        PendingLayer& new_layer = pending_layers_.back();
        MergeOrOverlapCheck(new_layer);

      } else if (IsChildOf(chunk_effect, current_group)) {
        // Case C: chunk 属于子 group，递归处理
        LayerizeGroup(chunk_effect, ...);

        // 尝试 decomposite 子 group
        DecompositeEffect(current_group, ...);

      } else {
        // Case B: chunk 不属于当前 group，退出循环
        break;
      }
    }
  }

  void MergeOrOverlapCheck(PendingLayer& new_layer) {
    // 反向遍历最近的 layer_merge_distance_limit_ 个 layers
    for (candidate in recent_layers) {
      if (candidate.Merge(new_layer, ...)) {
        // 成功合并！删除 new_layer
        pending_layers_.pop_back();
        return;
      }
      if (new_layer.MightOverlap(candidate)) {
        // 有 overlap，不能继续尝试合并
        new_layer.SetCompositingTypeToOverlap();
        return;
      }
    }
  }
};
```

## 合并条件：PendingLayer::Merge

```cpp
bool PendingLayer::Merge(const PendingLayer& guest,
                         LCDTextPreference lcd_text_preference,
                         float device_pixel_ratio,
                         CompositorScrollFn is_composited_scroll) {
  // 1. 检查是否需要独立层
  if (ChunkRequiresOwnLayer() || guest.ChunkRequiresOwnLayer())
    return false;

  // 2. 检查 PropertyTreeState 是否兼容
  std::optional<PropertyTreeState> upcast_state =
      guest.GetPropertyTreeState().CanUpcastWith(
          this->GetPropertyTreeState(), is_composited_scroll);
  if (!upcast_state)
    return false;

  // 3. 检查 LCD text 兼容性
  if (lcd_text_preference == LCDTextPreference::kStronglyPreferred) {
    if (HasText() != guest.HasText())
      return false;
    // ... more text rendering checks
  }

  // 4. 执行合并
  chunks_.Merge(guest.Chunks());
  bounds_.Union(guest.bounds_);
  drawable_bounds_.Union(guest.drawable_bounds_);
  return true;
}
```

## Layerizer 输出：PendingLayers

```cpp
struct PendingLayer {
  PaintChunkSubset chunks_;               // 包含的 PaintChunks
  PropertyTreeState property_tree_state_; // Transform/Clip/Effect
  CompositingType compositing_type_;      // kOther/kForeignLayer/kScrollbar/...
  gfx::Rect bounds_;                      // layer 边界
  gfx::Rect drawable_bounds_;             // 实际绘制区域

  // 根据 compositing_type_ 持有不同的 cc::Layer:
  scoped_refptr<cc::Layer> cc_layer_;                      // ForeignLayer/SolidColor/Scrollbar
  Member<ContentLayerClientImpl> content_layer_client_;   // 普通内容 (cc::PictureLayer)
};
```

**对于红色 div 示例**：

- 假设它是页面上唯一的元素
- Layerizer 创建 1 个 PendingLayer
- CompositingType = `kOther` (普通内容)
- PropertyTreeState = Root Transform/Clip/Effect
- chunks_ 包含 1 个 PaintChunk (包含红色 div 的 DrawingDisplayItem)

---

# 阶段 2：UpdateCompositedLayer - 创建 cc::Layer 和转换内容

## 概述

对于每个 `PendingLayer`，这个阶段：

1. 决定使用哪种 cc::Layer 类型
2. 创建或复用 cc::Layer 对象
3. 将 Blink 的 PaintChunks 转换为 cc::DisplayItemList

## 核心代码

```cpp
void PaintArtifactCompositor::Update(...) {
  // 创建属性树管理器
  PropertyTreeManager property_tree_manager(...);

  // 遍历每个 PendingLayer
  for (auto& pending_layer : pending_layers_) {
    // 【关键！】更新 layer 内容和 cc::Layer 对象
    pending_layer.UpdateCompositedLayer(
        old_pending_layer_matcher.Find(pending_layer),
        layer_selection,
        tracks_raster_invalidations_,
        root_layer_->layer_tree_host());

    cc::Layer& layer = pending_layer.CcLayer();

    // ... 后面会创建属性树节点并关联
  }
}
```

## PendingLayer::UpdateCompositedLayer

```cpp
void PendingLayer::UpdateCompositedLayer(
    PendingLayer* old_pending_layer,
    cc::LayerSelection& layer_selection,
    bool tracks_raster_invalidations,
    cc::LayerTreeHost* layer_tree_host) {

  // 根据 compositing_type_ 分发到不同的更新逻辑
  switch (compositing_type_) {
    case kForeignLayer:
      UpdateForeignLayer();
      // ForeignLayer (如 <video>, <canvas>) 已经是 cc::Layer
      // 只需从 ForeignLayerDisplayItem 获取并复用
      break;

    case kScrollHitTestLayer:
      UpdateScrollHitTestLayer(old_pending_layer);
      // 创建 cc::Layer 用于滚动命中测试
      // 不需要绘制内容，只需要设置 hit_testable 和 bounds
      break;

    case kScrollbarLayer:
      UpdateScrollbarLayer(old_pending_layer);
      // 创建 cc::SolidColorScrollbarLayer 或 cc::PaintedScrollbarLayer
      break;

    default:
      // 普通绘制内容 (红色 div 走这里！)
      if (UsesSolidColorLayer()) {
        UpdateSolidColorLayer(old_pending_layer);
        // 纯色优化：创建 cc::SolidColorLayer
      } else {
        UpdateContentLayer(old_pending_layer, tracks_raster_invalidations);
        // 通用内容：创建 cc::PictureLayer + DisplayItemList
      }
      break;
  }

  // 更新 layer 的通用属性
  cc::Layer& layer = CcLayer();
  layer.SetLayerTreeHost(layer_tree_host);
  UpdateLayerProperties(layer_selection, /*selection_only=*/false);
}
```

## 核心路径：UpdateContentLayer

对于普通的绘制内容（包括我们的红色 div）：

```cpp
void PendingLayer::UpdateContentLayer(
    PendingLayer* old_pending_layer,
    bool tracks_raster_invalidations) {

  // 1. 创建或复用 ContentLayerClientImpl
  if (old_pending_layer) {
    content_layer_client_ = std::move(old_pending_layer->content_layer_client_);
  }
  if (!content_layer_client_) {
    content_layer_client_ = MakeGarbageCollected<ContentLayerClientImpl>();
    content_layer_client_->GetRasterInvalidator().SetTracksRasterInvalidations(
        tracks_raster_invalidations);
  }

  // 2. 【核心！】更新 cc::PictureLayer 和转换内容
  content_layer_client_->UpdateCcPictureLayer(*this);
}
```

## ContentLayerClientImpl::UpdateCcPictureLayer

这是 **PaintChunks → cc::DisplayItemList 转换的核心**：

```cpp
void ContentLayerClientImpl::UpdateCcPictureLayer(
    const PendingLayer& pending_layer) {
  const auto& paint_chunks = pending_layer.Chunks();
  auto layer_state = pending_layer.GetPropertyTreeState();
  auto [layer_offset, layer_bounds] = pending_layer.Bounds();

  // 1. 光栅失效检测
  //    RasterInvalidator 比对新旧 PaintChunks，生成失效区域
  raster_invalidator_->Generate(paint_chunks, layer_offset, layer_bounds, layer_state);

  // 2. 快速路径：如果内容未变，复用旧的 DisplayItemList
  bool may_be_unchanged =
      cc_display_item_list_ &&
      layer_bounds == old_layer_bounds &&
      cc_picture_layer_->draws_content() == pending_layer.DrawsContent();
  if (may_be_unchanged) {
    // 跳过转换，直接复用！
    return;
  }

  // 3. 创建新的 cc::DisplayItemList
  cc_display_item_list_ = base::MakeRefCounted<cc::DisplayItemList>();

  // 4. 【核心转换】PaintChunks → cc::DisplayItemList
  //    这是 Blink 到 cc 的实际桥梁！
  PaintChunksToCcLayer::ConvertInto(
      paint_chunks,              // Blink 的 PaintChunks
      layer_state,               // PropertyTreeState (layer 的属性状态)
      layer_offset,              // layer 偏移 (用于坐标转换)
      nullptr,                   // under-invalidation checking params
      *cc_display_item_list_);  // 输出：cc::DisplayItemList

  // 5. 特殊处理：mask layer 或需要填充滚动内容
  if (is_mask_layer || DrawingShouldFillScrollingContentsLayer(...)) {
    cc_display_item_list_->StartPaint();
    cc_display_item_list_->push<cc::NoopOp>();
    cc_display_item_list_->EndPaintOfUnpaired(gfx::Rect(layer_bounds));
  }

  // 6. Finalize DisplayItemList
  cc_display_item_list_->Finalize();

  // 7. 更新 cc::PictureLayer 的属性
  cc_picture_layer_->SetOffsetToTransformParent(layer_offset);
  cc_picture_layer_->SetBounds(layer_bounds);
  cc_picture_layer_->SetIsDrawable(pending_layer.DrawsContent());
  cc_picture_layer_->SetBackgroundColor(pending_layer.ComputeBackgroundColor());

  // 注意：cc::PictureLayer 通过 cc::ContentLayerClient 接口获取 DisplayItemList
  //       在 Display() 时调用 PaintContentsToDisplayList() → 返回 cc_display_item_list_
}
```

## PaintChunksToCcLayer::ConvertInto - 转换核心

```cpp
void PaintChunksToCcLayer::ConvertInto(
    const PaintChunkSubset& chunks,
    const PropertyTreeState& layer_state,
    const gfx::Vector2dF& layer_offset,
    RasterUnderInvalidationCheckingParams* under_invalidation_checking_params,
    cc::DisplayItemList& cc_list) {

  // 创建转换上下文并执行转换
  ConversionContext(layer_state, layer_offset, cc_list).Convert(chunks);

  // 如果启用了 under-invalidation checking，额外绘制检测图形
  if (under_invalidation_checking_params) {
    // ... 用于调试光栅失效问题
  }
}
```

## ConversionContext::Convert - 实际转换逻辑

`ConversionContext` 是 PaintChunks → cc::DisplayItemList 转换的核心类。它维护一个状态栈，确保属性状态切换时 Save/Restore 正确配对。

### 类结构和职责

```cpp
class ConversionContext {
  STACK_ALLOCATED();

 public:
  // 构造函数：初始化 layer 状态
  ConversionContext(const PropertyTreeState& layer_state,
                    const gfx::Vector2dF& layer_offset,
                    Result& result);

  // 主转换方法：遍历所有 PaintChunks，转换为 cc::DisplayItemList
  void Convert(const PaintChunkSubset& chunks,
               const gfx::Rect* additional_cull_rect = nullptr);

 private:
  // 状态切换方法
  ScrollTranslationAction SwitchToEffect(const EffectPaintPropertyNode&);
  ScrollTranslationAction SwitchToClip(const ClipPaintPropertyNode&);
  ScrollTranslationAction SwitchToTransform(const TransformPaintPropertyNode&);

  // 状态管理
  ChunkToLayerMapper chunk_to_layer_mapper_;  // 坐标映射器
  HeapVector<StateEntry> state_stack_;        // 状态栈 (用于 Save/Restore 配对)

  // 当前状态
  const TransformPaintPropertyNode* current_transform_;
  const ClipPaintPropertyNode* current_clip_;
  const EffectPaintPropertyNode* current_effect_;

  Result& result_;  // 输出 (cc::DisplayItemList 或 PaintOpBuffer)
};
```

### 核心转换流程：Convert 方法

```cpp
template <typename Result>
void ConversionContext<Result>::Convert(PaintChunkIterator& chunk_it,
                                        PaintChunkIterator end_chunk,
                                        const gfx::Rect* additional_cull_rect) {
  // 遍历每个 PaintChunk
  for (; chunk_it != end_chunk; ++chunk_it) {
    const auto& chunk = *chunk_it;

    // 【优化 1】跳过不可见的 chunk
    if (chunk.effectively_invisible) {
      continue;
    }

    PropertyTreeState chunk_state = chunk.properties.Unalias();

    // 【优化 2】跳过没有绘制内容的 chunk
    if (!HasDrawing(chunk_it, chunk_state)) {
      continue;
    }

    // 【步骤 1】首次绘制时添加 layer offset 转换
    TranslateForLayerOffsetOnce();

    // 更新坐标映射器的状态
    chunk_to_layer_mapper_.SwitchToChunkWithState(chunk, chunk_state);

    // 【优化 3】Cull rect 剔除 (如果提供)
    if (additional_cull_rect) {
      gfx::Rect chunk_visual_rect =
          chunk_to_layer_mapper_.MapVisualRect(chunk.drawable_bounds);
      if (!additional_cull_rect->Intersects(chunk_visual_rect)) {
        continue;
      }
    }

    // 【步骤 2】切换到 chunk 的属性状态
    // 按顺序切换：Effect → Clip → Transform
    // 这个顺序很重要！因为 Effect 可能依赖 Clip，Clip 可能依赖 Transform
    ScrollTranslationAction action = SwitchToEffect(chunk_state.Effect());
    if (!action) {
      action = SwitchToClip(chunk_state.Clip());
    }
    if (!action) {
      action = SwitchToTransform(chunk_state.Transform());
    }

    // 【特殊处理】RasterInducingScroll 机制
    // 如果遇到需要独立光栅化的滚动，创建嵌套的 DisplayItemList
    if (action.type == ScrollTranslationAction::kStart) {
      CHECK(action.scroll_translation_to_start);
      EmitDrawScrollingContentsOp(chunk_it, end_chunk,
                                  *action.scroll_translation_to_start);
      // chunk_it 现在指向滚动内容的最后一个 chunk
      --chunk_it;  // 抵消 for 循环的 ++chunk_it
      continue;
    }
    if (action.type == ScrollTranslationAction::kEnd) {
      // 滚动内容结束，返回到外层 context
      return;
    }

    // 【步骤 3】转换 chunk 内的每个 DisplayItem
    for (const auto& item : chunk_it.DisplayItems()) {
      PaintRecord record;

      // 获取 PaintRecord
      if (auto* scrollbar = DynamicTo<ScrollbarDisplayItem>(item)) {
        record = scrollbar->Paint();
      } else if (auto* drawing = DynamicTo<DrawingDisplayItem>(item)) {
        record = drawing->GetPaintRecord();
      } else {
        continue;  // 跳过其他类型 (如 ForeignLayerDisplayItem)
      }

      // 【优化 4】空 PaintRecord 处理
      // 如果 effect 是 Root，空记录可以跳过
      // 否则需要保留（用于正确计算 effect bounds）
      bool can_ignore_record =
          &chunk_state.Effect() == &EffectPaintPropertyNode::Root();
      if (record.empty() && can_ignore_record) {
        continue;
      }

      // 映射 visual rect 到 layer 坐标系
      gfx::Rect visual_rect =
          chunk_to_layer_mapper_.MapVisualRect(item.VisualRect());

      // 【优化 5】再次检查 cull rect (针对单个 item)
      if (additional_cull_rect && can_ignore_record &&
          !additional_cull_rect->Intersects(visual_rect)) {
        continue;
      }

      // 【核心！】输出 cc::DrawRecordOp
      result_.StartPaint();
      if (!record.empty()) {
        push<cc::DrawRecordOp>(std::move(record));
      }
      result_.EndPaintOfUnpaired(visual_rect);
    }

    // 【步骤 4】更新 effect bounds (用于后续的 effect 优化)
    UpdateEffectBounds(gfx::RectF(chunk.drawable_bounds),
                       chunk_state.Transform());
  }
}
```

### 状态切换详解：SwitchToEffect

Effect 切换是最复杂的，因为它涉及多层嵌套和输出 clip 的管理。

```cpp
template <typename Result>
ScrollTranslationAction ConversionContext<Result>::SwitchToEffect(
    const EffectPaintPropertyNode& target_effect) {
  if (&target_effect == current_effect_) {
    return {};  // 已经在目标状态，无需切换
  }

  // 【阶段 1】退出当前 effect，直到找到最低公共祖先 (LCA)
  const auto& lca_effect =
      target_effect.LowestCommonAncestor(*current_effect_).Unalias();

  while (current_effect_ != &lca_effect) {
    // 退出当前 effect 的 clips (调用 EndClips())
    if (auto action = EndClips()) {
      return action;
    }

    // 检查状态栈是否为空 (异常情况)
    if (!state_stack_.size()) {
      // Effect 层级问题，但如果没有实际效果可以继续
      if (!HasRealEffects(*current_effect_, lca_effect)) {
        break;
      }
      return {};
    }

    // 结束当前 effect (发出 cc::RestoreOp)
    EndEffect();
  }

  // 【阶段 2】收集从 LCA 到目标 effect 的路径
  HeapVector<Member<const EffectPaintPropertyNode>, 8> pending_effects;
  for (const auto* effect = &target_effect; effect != &lca_effect;
       effect = effect->UnaliasedParent()) {
    if (!effect) break;  // 异常情况
    pending_effects.push_back(effect);
  }

  // 【阶段 3】从上到下应用 pending effects
  for (const auto& sub_effect : base::Reversed(pending_effects)) {
    if (auto action = StartEffect(*sub_effect)) {
      return action;
    }
  }

  return {};
}
```

### 启动 Effect：StartEffect

```cpp
template <typename Result>
ScrollTranslationAction ConversionContext<Result>::StartEffect(
    const EffectPaintPropertyNode& effect) {
  // 【步骤 1】进入 effect 的 output clip
  if (effect.OutputClip()) {
    if (auto action = SwitchToClip(effect.OutputClip()->Unalias())) {
      return action;
    }
    // 切换到 effect 的 transform (优化光栅化)
    if (auto action =
            SwitchToTransform(effect.LocalTransformSpace().Unalias())) {
      return action;
    }
  } else {
    // 没有 output clip，退出所有 clips
    if (auto action = EndClips()) {
      return action;
    }
  }

  // 【步骤 2】检查 effect 类型
  bool has_filter = !!effect.Filter();
  bool has_backdrop_filter = !!effect.BackdropFilter();
  bool has_opacity = effect.Opacity() != 1.f;
  bool has_other_effects = effect.BlendMode() != SkBlendMode::kSrcOver;

  // 【步骤 3】发出相应的 cc::SaveLayer*Op
  size_t save_layer_id = kNotFound;
  result_.StartPaint();

  if (!has_filter) {
    if (has_other_effects) {
      // 有 blend mode 或其他效果
      cc::PaintFlags flags;
      flags.setBlendMode(effect.BlendMode());
      flags.setAlphaf(effect.Opacity());

      if (has_backdrop_filter) {
        save_layer_id = push<cc::SaveLayerFiltersOp>(
            effect.BackdropFilterBounds().getBounds(),
            /*foreground_filters=*/{},
            cc::RenderSurfaceFilters::BuildImageFilter(
                effect.BackdropFilter()->AsCcFilterOperations()),
            flags);
      } else {
        save_layer_id = push<cc::SaveLayerOp>(flags);
      }
    } else if (has_backdrop_filter) {
      // 只有 backdrop filter
      cc::PaintFlags flags;
      flags.setAlphaf(effect.Opacity());
      save_layer_id = push<cc::SaveLayerFiltersOp>(
          effect.BackdropFilterBounds().getBounds(),
          /*foreground_filters=*/{},
          cc::RenderSurfaceFilters::BuildImageFilter(
              effect.BackdropFilter()->AsCcFilterOperations()),
          flags);
    } else {
      // 只有 opacity
      save_layer_id = push<cc::SaveLayerAlphaOp>(effect.Opacity());
    }
  } else {
    // Filter effect (单独处理，与其他效果互斥)
    cc::PaintFlags filter_flags;
    filter_flags.setImageFilter(cc::RenderSurfaceFilters::BuildImageFilter(
        effect.Filter()->AsCcFilterOperations()));
    save_layer_id = push<cc::SaveLayerOp>(filter_flags);
  }

  result_.EndPaintOfPairedBegin();
  DCHECK_NE(save_layer_id, kNotFound);

  // 【步骤 4】更新状态栈
  const ClipPaintPropertyNode* input_clip = current_clip_;
  PushState(StateEntry::kEffect);
  effect_bounds_stack_.emplace_back(
      EffectBoundsInfo{save_layer_id, current_transform_});

  current_clip_ = input_clip;
  current_effect_ = &effect;

  // 【步骤 5】处理 reference filter (产生输出即使没有输入)
  if (effect.HasReferenceFilter()) {
    gfx::Rect filtered_bounds = effect.FilterOutputBounds();
    effect_bounds_stack_.back().bounds = gfx::RectF(filtered_bounds);

    // 发出空操作以添加 filtered bounds 到 visual rect
    result_.StartPaint();
    result_.EndPaintOfUnpaired(
        chunk_to_layer_mapper_.MapVisualRect(filtered_bounds));
  }

  return {};
}
```

### 状态切换详解：SwitchToClip

Clip 切换需要处理 clip 的合并（多个相邻的矩形 clip 可以合并为一个）。

```cpp
template <typename Result>
ScrollTranslationAction ConversionContext<Result>::SwitchToClip(
    const ClipPaintPropertyNode& target_clip) {
  if (&target_clip == current_clip_) {
    return {};  // 已在目标状态
  }

  // 【阶段 1】退出 clips 直到 LCA
  const auto* lca_clip =
      &target_clip.LowestCommonAncestor(*current_clip_).Unalias();
  const auto* clip = current_clip_;

  while (clip != lca_clip) {
    // 检查是否遇到外层 context 的状态边界
    if (!state_stack_.size() && outer_state_stack_ &&
        !outer_state_stack_->empty() && outer_state_stack_->back().IsClip()) {
      return {ScrollTranslationAction::kEnd};
    }

    // 检查状态栈完整性
    if (!state_stack_.size() || !state_stack_.back().IsClip()) {
      // Clip 层级问题，尽力恢复
      break;
    }

    DCHECK(clip->Parent());
    clip = &clip->Parent()->Unalias();

    StateEntry& previous_state = state_stack_.back();
    if (clip == lca_clip) {
      // LCA 可能是 combined clips 的中间节点
      clip = lca_clip = previous_state.clip;
    }

    if (clip == previous_state.clip) {
      EndClip();  // 发出 cc::RestoreOp
      DCHECK_EQ(current_clip_, clip);
    }
  }

  if (&target_clip == current_clip_) {
    return {};  // 已完成
  }

  // 【阶段 2】收集从当前 clip 到目标 clip 的路径
  HeapVector<Member<const ClipPaintPropertyNode>, 8> pending_clips;
  for (const auto* clip = &target_clip; clip != current_clip_;
       clip = clip->UnaliasedParent()) {
    if (!clip) break;  // 异常情况
    pending_clips.push_back(clip);
  }

  // 【阶段 3】尝试合并 clips (优化)
  DCHECK(pending_clips.size());
  auto pending_combined_clip_rect = pending_clips.back()->PaintClipRect();
  const auto* lowest_combined_clip_node = pending_clips.back().Get();

  for (auto i = pending_clips.size() - 1; i--;) {
    const auto* sub_clip = pending_clips[i].Get();

    if (CombineClip(*sub_clip, pending_combined_clip_rect)) {
      // 可以合并，继续
      lowest_combined_clip_node = sub_clip;
    } else {
      // 不能合并，输出当前 combined clip
      if (auto action = StartClip(pending_combined_clip_rect,
                                  *lowest_combined_clip_node)) {
        return action;
      }
      // 开始新的合并序列
      pending_combined_clip_rect = sub_clip->PaintClipRect();
      lowest_combined_clip_node = sub_clip;
    }
  }

  // 输出最后的 combined clip
  if (auto action =
          StartClip(pending_combined_clip_rect, *lowest_combined_clip_node)) {
    return action;
  }

  DCHECK_EQ(current_clip_, &target_clip);
  return {};
}
```

### Clip 合并逻辑：CombineClip

```cpp
static bool CombineClip(const ClipPaintPropertyNode& clip,
                        FloatRoundedRect& combined_clip_rect) {
  // 【条件 1】Pixel-moving filter 的 clip 直接合并 (无需裁剪)
  if (clip.PixelMovingFilter())
    return true;

  const auto* parent = clip.UnaliasedParent();
  CHECK(parent);

  // 【条件 2】不能合并有 clip path 的 clip
  if (parent->ClipPath()) {
    return false;
  }

  // 【条件 3】不同 transform space 的 clip 不能合并
  const auto& transform_space = clip.LocalTransformSpace().Unalias();
  const auto& parent_transform_space = parent->LocalTransformSpace().Unalias();
  if (&transform_space != &parent_transform_space) {
    if (transform_space.Parent() != &parent_transform_space ||
        !transform_space.IsIdentity()) {
      return false;
    }
    // RasterInducingScroll 模式下，不能跨滚动合并
    if (RuntimeEnabledFeatures::RasterInducingScrollEnabled() &&
        transform_space.ScrollNode()) {
      return false;
    }
  }

  // 【条件 4】不能合并两个都是圆角的 clip
  bool clip_is_rounded = clip.PaintClipRect().IsRounded();
  bool combined_is_rounded = combined_clip_rect.IsRounded();
  if (clip_is_rounded && combined_is_rounded)
    return false;

  // 【优化】如果一个圆角包含另一个矩形，使用圆角
  if (combined_is_rounded) {
    return clip.PaintClipRect().Rect().Contains(combined_clip_rect.Rect());
  }
  if (clip_is_rounded) {
    if (combined_clip_rect.Rect().Contains(clip.PaintClipRect().Rect())) {
      combined_clip_rect = clip.PaintClipRect();
      return true;
    }
    return false;
  }

  // 【标准情况】两个矩形 clip，合并为交集
  DCHECK(!combined_is_rounded && !clip_is_rounded);
  combined_clip_rect = FloatRoundedRect(
      IntersectRects(combined_clip_rect.Rect(), clip.PaintClipRect().Rect()));
  return true;
}
```

### 状态切换详解：SwitchToTransform

Transform 切换相对简单，因为不需要像 clip/effect 那样维护嵌套关系。

```cpp
template <typename Result>
ScrollTranslationAction ConversionContext<Result>::SwitchToTransform(
    const TransformPaintPropertyNode& target_transform) {
  // 首先结束之前的 transform (如果有)
  EndTransform();

  if (&target_transform == current_transform_) {
    return {};  // 已在目标状态
  }

  // 检查是否需要特殊的滚动处理 (RasterInducingScroll)
  if (auto action = ComputeScrollTranslationAction(target_transform)) {
    return action;
  }

  // 计算从当前 transform 到目标 transform 的投影矩阵
  gfx::Transform projection = TargetToCurrentProjection(target_transform);

  if (projection.IsIdentity()) {
    return {};  // 无需变换
  }

  // 发出 Save + Transform
  result_.StartPaint();
  push<cc::SaveOp>();

  if (projection.IsIdentityOr2dTranslation()) {
    // 优化：纯平移使用 TranslateOp (比 ConcatOp 更快)
    gfx::Vector2dF translation = projection.To2dTranslation();
    push<cc::TranslateOp>(translation.x(), translation.y());
  } else {
    // 通用情况：使用 ConcatOp
    push<cc::ConcatOp>(gfx::TransformToSkM44(projection));
  }

  result_.EndPaintOfPairedBegin();

  // 记录之前的 transform (用于后续 EndTransform)
  previous_transform_ = current_transform_;
  current_transform_ = &target_transform;

  return {};
}

template <typename Result>
void ConversionContext<Result>::EndTransform() {
  if (!previous_transform_)
    return;

  // 发出 cc::RestoreOp
  result_.StartPaint();
  push<cc::RestoreOp>();
  result_.EndPaintOfPairedEnd();

  current_transform_ = previous_transform_;
  previous_transform_ = nullptr;
}
```

### 关键数据结构：StateEntry

```cpp
struct StateEntry {
  enum Type { kEffect, kClip, kClipOmitted };

  Type type;
  const ClipPaintPropertyNode* clip;       // 对应的 clip node (用于 EndClip)
  const EffectPaintPropertyNode* effect;   // 对应的 effect node (用于 EndEffect)

  bool IsEffect() const { return type == kEffect; }
  bool IsClip() const { return type == kClip || type == kClipOmitted; }
};

// 状态栈示例（从底到顶）：
// state_stack_ = [
//   {kClip, clip: C1},      // 对应 layer 中的 BeginClip(C1)
//   {kClip, clip: C2},      // 对应 BeginClip(C2)
//   {kEffect, effect: E1},  // 对应 SaveLayerAlpha(E1)
//   {kClip, clip: C3},      // 对应 BeginClip(C3)
// ]
// 这意味着 cc::DisplayItemList 包含：BeginClip(C1), BeginClip(C2), SaveLayerAlpha, BeginClip(C3)
// 析构时会自动发出：EndClip(C3), EndEffect(E1), EndClip(C2), EndClip(C1)
```

### 生命周期管理：析构函数

```cpp
template <typename Result>
ConversionContext<Result>::~ConversionContext() {
  // 自动清理所有未关闭的状态 (确保 Save/Restore 配对)
  while (state_stack_.size()) {
    if (state_stack_.back().IsEffect()) {
      EndEffect();  // 发出 cc::RestoreOp
    } else {
      EndClip();    // 发出 cc::RestoreOp
    }
  }

  // 结束 transform (如果有)
  EndTransform();

  // 结束 layer offset translation (如果有)
  if (translated_for_layer_offset_)
    AppendRestore();
}
```

## 关键理解：DrawRecordOp 的作用

```cpp
// Blink 的 DrawingDisplayItem
DrawingDisplayItem {
  client_id: div 的 ID,
  type: kBoxDecorationBackground,
  visual_rect: {0, 0, 100, 100},
  paint_record: PaintRecord {
    ops: [DrawRectOp{rect={0,0,100,100}, color=red}]
  }
}

// 转换后的 cc::DisplayItemList
cc::DisplayItemList {
  ops: [
    DrawRecordOp {
      record: PaintRecord {  // 直接引用 Blink 的 PaintRecord！
        ops: [DrawRectOp{rect, color=red}]
      },
      visual_rect: {0, 0, 100, 100}
    }
  ]
}
```

**为什么用 DrawRecordOp 包装？**

1. **避免重复序列化**：PaintRecord 在 Blink 和 cc 之间共享，不需要拷贝 PaintOp
2. **延迟执行**：PaintRecord 的实际 playback 延迟到光栅化阶段
3. **统一接口**：cc::DisplayItemList 可以包含各种 cc::PaintOp，DrawRecordOp 只是其中一种

## ConversionContext 完整实例：复杂属性状态切换

根据 ConversionContext 类注释中的例子（对应单元测试 `PaintChunksToCcLayerTest.InterleavedClipEffect`）：

**输入：PaintChunks 和属性树**

```
Clip 树:
  C0 (Root)
   └─ C1
       └─ C2
           └─ C3
               └─ C4

Effect 树:
  E0 (Root, output_clip=C0)
   └─ E1 (output_clip=C2)
       └─ E2 (output_clip=C4)

Layer 状态: PropertyTreeState(Transform=T0, Clip=C0, Effect=E0)

PaintChunks:
  P0: PropertyTreeState(T0, C3, E0)  // 绘制内容 A
  P1: PropertyTreeState(T0, C4, E2)  // 绘制内容 B (在 E2 effect 内)
  P2: PropertyTreeState(T0, C3, E1)  // 绘制内容 C (在 E1 effect 内)
  P3: PropertyTreeState(T0, C4, E0)  // 绘制内容 D (回到 E0)
```

**ConversionContext 执行过程：**

```cpp
// 初始化
ConversionContext ctx(layer_state={T0, C0, E0}, ...);
// 状态：current_clip=C0, current_effect=E0, state_stack=[]

// ============ 处理 P0(C3, E0) ============
SwitchToEffect(E0):  // 已在 E0，无操作
SwitchToClip(C3):    // 需要从 C0 到 C3
  // LCA = C0
  // 路径: C0 → C1 → C2 → C3
  // 尝试合并 clips (假设都是矩形，可以合并)

  输出: StartClip(C1_C2_C3_combined)
    → SaveOp
    → ClipRectOp(combined_rect)  // C1, C2, C3 的交集

  状态更新:
    state_stack = [{kClip, clip=C3}]
    current_clip = C3

输出: DrawRecordOp(PaintRecord_A)  // P0 的绘制内容

// ============ 处理 P1(C4, E2) ============
SwitchToEffect(E2):  // 需要从 E0 到 E2
  // LCA = E0
  // 路径: E0 → E1 → E2

  // 步骤 1: 进入 E1
  StartEffect(E1):
    // E1.output_clip = C2, 需要从 C3 退到 C2
    SwitchToClip(C2):
      // LCA = C2
      // 需要退出 C3 (因为 C3 是 C2 的子节点)

      输出: EndClip(C3)
        → RestoreOp  // 对应之前的 ClipRectOp

      状态更新:
        state_stack = []
        current_clip = C0  // 回到 layer 状态

      // 现在进入 C1 → C2
      输出: StartClip(C1_C2_combined)
        → SaveOp
        → ClipRectOp(C1_C2_combined)

      状态更新:
        state_stack = [{kClip, clip=C2}]
        current_clip = C2

    // 发出 SaveLayerOp for E1
    输出: SaveLayerAlphaOp(opacity=E1.opacity)

    状态更新:
      state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}]
      current_clip = C2  // E1 的 input clip
      current_effect = E1

  // 步骤 2: 进入 E2
  StartEffect(E2):
    // E2.output_clip = C4, 需要从 C2 到 C4
    SwitchToClip(C4):
      // LCA = C2
      // 路径: C2 → C3 → C4

      输出: StartClip(C3_C4_combined)
        → SaveOp
        → ClipRectOp(C3_C4_combined)

      状态更新:
        state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}, {kClip, clip=C4}]
        current_clip = C4

    // 发出 SaveLayerOp for E2
    输出: SaveLayerAlphaOp(opacity=E2.opacity)

    状态更新:
      state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}, {kClip, clip=C4}, {kEffect, effect=E2}]
      current_clip = C4  // E2 的 input clip
      current_effect = E2

输出: DrawRecordOp(PaintRecord_B)  // P1 的绘制内容

// ============ 处理 P2(C3, E1) ============
SwitchToEffect(E1):  // 需要从 E2 回到 E1
  // LCA = E1

  // 步骤 1: 退出 E2
  EndClips():  // E2 没有额外的 clips 需要退出 (C4 是 E2 的 output clip 一部分)
  EndEffect():
    输出: RestoreOp  // 对应 E2 的 SaveLayerAlphaOp

  // 同时需要退出 C4 (因为我们要去 C3)
  输出: RestoreOp  // 对应 C3_C4_combined 的 ClipRectOp

  状态更新:
    state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}]
    current_clip = C2
    current_effect = E1

SwitchToClip(C3):  // 需要从 C2 到 C3
  // LCA = C2
  // 路径: C2 → C3

  输出: StartClip(C3)
    → SaveOp
    → ClipRectOp(C3_rect)

  状态更新:
    state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}, {kClip, clip=C3}]
    current_clip = C3

输出: DrawRecordOp(PaintRecord_C)  // P2 的绘制内容

// ============ 处理 P3(C4, E0) ============
SwitchToEffect(E0):  // 需要从 E1 回到 E0
  // LCA = E0

  // 步骤 1: 退出 C3
  EndClips():
    输出: RestoreOp  // 对应 C3 的 ClipRectOp

    状态更新:
      state_stack = [{kClip, clip=C2}, {kEffect, effect=E1}]
      current_clip = C2

  // 步骤 2: 退出 E1 (和它的 output clip C2)
  EndEffect():
    输出: RestoreOp  // 对应 E1 的 SaveLayerAlphaOp

    // 同时退出 C2
    输出: RestoreOp  // 对应 C1_C2_combined 的 ClipRectOp

    状态更新:
      state_stack = []
      current_clip = C0
      current_effect = E0

SwitchToClip(C4):  // 需要从 C0 到 C4
  // LCA = C0
  // 路径: C0 → C1 → C2 → C3 → C4

  输出: StartClip(C1_C2_C3_C4_combined)
    → SaveOp
    → ClipRectOp(combined_rect)  // C1 到 C4 的交集

  状态更新:
    state_stack = [{kClip, clip=C4}]
    current_clip = C4

输出: DrawRecordOp(PaintRecord_D)  // P3 的绘制内容

// ============ 析构清理 ============
~ConversionContext():
  // 清理所有未关闭的状态
  EndClip():  // 退出 C4
    输出: RestoreOp
```

**最终生成的 cc::DisplayItemList：**

```cpp
cc::DisplayItemList {
  ops: [
    // P0(C3, E0) 的输出
    SaveOp,                           // StartClip(C1_C2_C3)
    ClipRectOp(C1_C2_C3_combined),
    DrawRecordOp(PaintRecord_A),

    // P1(C4, E2) 的输出
    RestoreOp,                        // EndClip(C3)
    SaveOp,                           // StartClip(C1_C2) for E1
    ClipRectOp(C1_C2_combined),
    SaveLayerAlphaOp(E1.opacity),     // StartEffect(E1)
    SaveOp,                           // StartClip(C3_C4) for E2
    ClipRectOp(C3_C4_combined),
    SaveLayerAlphaOp(E2.opacity),     // StartEffect(E2)
    DrawRecordOp(PaintRecord_B),

    // P2(C3, E1) 的输出
    RestoreOp,                        // EndEffect(E2)
    RestoreOp,                        // EndClip(C4)
    SaveOp,                           // StartClip(C3)
    ClipRectOp(C3_rect),
    DrawRecordOp(PaintRecord_C),

    // P3(C4, E0) 的输出
    RestoreOp,                        // EndClip(C3)
    RestoreOp,                        // EndEffect(E1)
    RestoreOp,                        // EndClip(C2)
    SaveOp,                           // StartClip(C1_C2_C3_C4)
    ClipRectOp(C1_C2_C3_C4_combined),
    DrawRecordOp(PaintRecord_D),

    // 析构清理
    RestoreOp                         // EndClip(C4)
  ]
}
```

**关键观察：**

1. **Save/Restore 严格配对**：每个 SaveOp 都有对应的 RestoreOp
2. **状态栈管理**：`state_stack_` 跟踪所有打开的 clips 和 effects
3. **LCA 算法**：总是找到最低公共祖先，最小化状态切换操作
4. **Clip 合并优化**：相邻的矩形 clips 被合并为一个 ClipRectOp
5. **Effect 的 output clip**：Effect 启动前必须先进入其 output clip
6. **正确的嵌套顺序**：Effect 内部的内容被正确地包裹在 SaveLayer/Restore 之间

这个例子展示了 `ConversionContext` 如何处理复杂的、交错的属性状态切换，确保生成的 `cc::DisplayItemList` 既正确又高效。

## 对于红色 div 的完整转换

```
输入：PaintChunk {
  begin_index: 0,
  end_index: 1,
  properties: PropertyTreeState {
    transform: Root,
    clip: Root,
    effect: Root
  }
}

DisplayItemList[0]: DrawingDisplayItem {
  paint_record: PaintRecord {
    ops: [DrawRectOp{rect={0,0,100,100}, color=red, style=fill}]
  }
}

↓ PaintChunksToCcLayer::ConvertInto

输出：cc::DisplayItemList {
  ops: [
    DrawRecordOp {
      record: PaintRecord {  // 直接引用上面的 PaintRecord
        ops: [DrawRectOp{rect={0,0,100,100}, color=red}]
      }
    }
  ]
}
```

---

# 阶段 3：PropertyTreeManager - 创建属性树并关联

## 概述

Compositor 使用独立的属性树 (Property Trees) 来管理 Transform/Clip/Effect：

- **优势**：属性变化不需要重建 layer 树，只需更新属性树节点
- **实现**：每个 cc::Layer 不直接存储 transform/clip/effect，而是存储节点 ID

PropertyTreeManager 的职责：

1. 为 Blink 的 PaintPropertyNode 创建对应的 cc 属性树节点
2. 将节点 ID 设置到 cc::Layer
3. 维护节点的缓存和复用

### 核心代码

```cpp
void PaintArtifactCompositor::Update(...) {
  // 创建属性树管理器
  PropertyTreeManager property_tree_manager(
      *this,
      *host->property_trees(),           // cc 的属性树
      *root_layer_,
      layer_list_builder,
      g_s_property_tree_sequence_number  // 序列号，用于失效检测
  );

  // 遍历每个 PendingLayer
  for (auto& pending_layer : pending_layers_) {
    pending_layer.UpdateCompositedLayer(...);

    cc::Layer& layer = pending_layer.CcLayer();
    const auto& property_state = pending_layer.GetPropertyTreeState();

    // 【关键】为这个 layer 创建/获取 cc 属性树节点
    int transform_id =
        property_tree_manager.EnsureCompositorTransformNode(
            property_state.Transform());

    int clip_id =
        property_tree_manager.EnsureCompositorClipNode(
            property_state.Clip());

    int effect_id =
        property_tree_manager.SwitchToEffectNodeWithSynthesizedClip(
            property_state.Effect(),
            property_state.Clip(),
            layer.draws_content());

    int scroll_id =
        property_tree_manager.EnsureCompositorScrollAndTransformNode(
            ScrollTranslationStateForLayer(pending_layer));

    // 【关联】将属性树节点 ID 设置到 cc::Layer
    layer.SetTransformTreeIndex(transform_id);
    layer.SetClipTreeIndex(clip_id);
    layer.SetEffectTreeIndex(effect_id);
    layer.SetScrollTreeIndex(scroll_id);

    // 添加到 layer 列表
    layer_list_builder.Add(&layer);
  }

  // 设置最终的 layer 树
  root_layer_->SetChildLayerList(layer_list_builder.Finalize());
}
```

## PropertyTreeManager::EnsureCompositorTransformNode

```cpp
int PropertyTreeManager::EnsureCompositorTransformNode(
    const TransformPaintPropertyNode& transform_node) {

  // 1. 检查缓存：是否已经为这个 Blink node 创建了 cc node
  int cached_node_id = transform_node.CcNodeId(sequence_number_);
  if (cached_node_id != cc::kInvalidPropertyNodeId) {
    return cached_node_id;
  }

  // 2. 递归创建父节点
  const auto* parent = transform_node.UnaliasedParent();
  int parent_id = parent ? EnsureCompositorTransformNode(*parent)
                        : cc::TransformTree::kRootNodeId;

  // 3. 创建新的 cc::TransformNode
  cc::TransformNode cc_node;
  cc_node.id = transform_tree_.Insert(cc_node, parent_id);
  cc_node.local = transform_node.Matrix();
  cc_node.origin = transform_node.Origin();
  cc_node.flattens_inherited_transform = transform_node.FlattensInheritedTransform();
  // ... 更多属性

  // 4. 缓存映射
  transform_node.SetCcNodeId(sequence_number_, cc_node.id);

  return cc_node.id;
}
```

类似的逻辑也应用于 Clip、Effect、Scroll 属性树。

## 属性树的优势

**传统方式（Pre-Composite）**：

```
Transform 变化
  ↓ 失效 LayoutObject
  ↓ Re-layout (可能)
  ↓ Re-paint (必须)
  ↓ 重建 layer 树
  ↓ Commit
  ↓ 可能重新光栅化
```

**CAP + 属性树方式**：

```
Transform 变化
  ↓ 只更新 cc::TransformNode 的 local matrix
  ↓ Compositor 线程直接使用新的 transform
  ↓ 无需 Paint 或 Commit
  ↓ 不需要重新光栅化纹理
```

**性能对比**：

- 传统方式：可能需要 10-50ms (取决于复杂度)
- 属性树方式：< 1ms (只是更新一个矩阵)

---

# 完整的数据流：红色 div 示例

## 输入：PaintArtifact

```cpp
PaintArtifact {
  DisplayItemList: [
    DrawingDisplayItem {
      client_id: div 的 DisplayItemClientId,
      type: kBoxDecorationBackground,
      visual_rect: {0, 0, 100, 100},
      paint_record: PaintRecord {
        ops: [
          DrawRectOp {
            rect: {0, 0, 100, 100},
            flags: {color: red, style: fill}
          }
        ]
      }
    }
  ],

  PaintChunks: [
    PaintChunk {
      begin_index: 0,
      end_index: 1,
      id: {client_id: div, type: kBoxDecorationBackground},
      properties: PropertyTreeState {
        transform: TransformPaintPropertyNode (Root),
        clip: ClipPaintPropertyNode (Root),
        effect: EffectPaintPropertyNode (Root)
      },
      bounds: {0, 0, 100, 100},
      drawable_bounds: {0, 0, 100, 100},
      rect_known_to_be_opaque: {0, 0, 100, 100}
    }
  ]
}
```

## 阶段 1 输出：PendingLayers

```cpp
PendingLayers: [
  PendingLayer {
    chunks_: PaintChunkSubset {
      paint_artifact: &PaintArtifact (上面的),
      indices: [0]  // 只包含 chunk[0]
    },

    compositing_type_: kOther,  // 普通内容

    property_tree_state_: PropertyTreeState {
      transform: Root,
      clip: Root,
      effect: Root
    },

    bounds_: {0, 0, 100, 100},
    drawable_bounds_: {0, 0, 100, 100},

    // 还没有创建 cc::Layer (在阶段 2 创建)
    cc_layer_: nullptr,
    content_layer_client_: nullptr
  }
]
```

## 阶段 2 输出：cc::Layer + DisplayItemList

```cpp
// PendingLayer 内部现在有：
content_layer_client_: ContentLayerClientImpl {
  cc_picture_layer_: cc::PictureLayer {
    id: 1,
    bounds: {100, 100},
    offset_to_transform_parent: {0, 0},
    is_drawable: true,
    background_color: transparent
  },

  cc_display_item_list_: cc::DisplayItemList {
    ops: [
      DrawRecordOp {
        record: PaintRecord {  // 引用 Blink 的 PaintRecord
          ops: [
            DrawRectOp {
              rect: {0, 0, 100, 100},
              flags: {color: red, style: fill}
            }
          ]
        },
        visual_rect: {0, 0, 100, 100}
      }
    ]
  }
}
```

## 阶段 3 输出：完整的 cc::Layer (带属性树索引)

```cpp
cc::PictureLayer {
  id: 1,
  bounds: {100, 100},
  offset_to_transform_parent: {0, 0},
  is_drawable: true,

  // 【关键】属性树索引
  transform_tree_index: 1,  // → cc::TransformNode (Root)
  clip_tree_index: 1,       // → cc::ClipNode (Root)
  effect_tree_index: 1,     // → cc::EffectNode (Root)
  scroll_tree_index: 0,     // → cc::ScrollNode (None)

  property_tree_sequence_number: 1,

  // 通过 ContentLayerClient 接口获取内容
  client: ContentLayerClientImpl (上面的)
}
```

以及在 cc::LayerTreeHost 的属性树中：

```cpp
cc::PropertyTrees {
  TransformTree: {
    nodes: [
      {id: 0, parent_id: -1},  // kInvalidNodeId
      {id: 1, parent_id: 0, local: Identity, ...}  // Root
    ]
  },

  ClipTree: {
    nodes: [
      {id: 0, parent_id: -1},
      {id: 1, parent_id: 0, clip_rect: Infinite, ...}  // Root
    ]
  },

  EffectTree: {
    nodes: [
      {id: 0, parent_id: -1},
      {id: 1, parent_id: 0, opacity: 1.0, blend_mode: kSrcOver, ...}  // Root
    ]
  },

  ScrollTree: {
    nodes: [
      {id: 0, parent_id: -1}  // kInvalidNodeId (没有滚动)
    ]
  }
}
```

---

# CAP 与 Raster 的连接

CAP 的输出（cc::Layer 树 + DisplayItemList）会传递给 Compositor 线程，然后进入 Raster 阶段。

## Commit 阶段

```cpp
// 主线程
LayerTreeHost::Commit() {
  // 1. 同步 layer 树
  LayerTreeImpl* sync_tree = host_impl->sync_tree();
  sync_tree->PushPropertiesTo(active_tree);

  // 2. 同步属性树
  sync_tree->property_trees()->PushPropertiesTo(active_tree->property_trees());

  // 3. cc::PictureLayer → cc::PictureLayerImpl
  for (cc::Layer* layer : layers) {
    cc::PictureLayerImpl* layer_impl =
        static_cast<cc::PictureLayerImpl*>(layer->GetLayerImpl(sync_tree));

    // 同步 DisplayItemList (实际是序列化传输)
    layer_impl->UpdateDisplayItemList(layer->GetDisplayItemList());
  }
}
```

## TileManager 调度

```cpp
// Compositor 线程
void TileManager::PrepareTiles(const GlobalStateThatImpactsTilePriority& state) {
  // 1. 遍历所有 PictureLayerImpl
  for (PictureLayerImpl* layer : all_picture_layers_) {
    // 2. 根据 viewport 和 transform 计算可见 tiles
    std::vector<Tile*> visible_tiles = layer->CalculateVisibleTiles(state);

    // 3. 为需要光栅化的 tiles 创建 RasterTask
    for (Tile* tile : visible_tiles) {
      if (!tile->HasRasterTask() && tile->is_invalidated()) {
        CreateRasterTask(tile, layer->GetDisplayItemList());
      }
    }
  }

  // 4. 调度所有 RasterTask (可能发送到 Viz/GPU 进程)
  ScheduleTasksOnOriginThread(raster_tasks);
}
```

## OOP-R (Out-of-Process Rasterization)

```cpp
// Viz/GPU 进程
void GpuRasterBufferProvider::PlaybackAndCopyOnWorkerThread(...) {
  // 1. 反序列化 DisplayItemList
  cc::DisplayItemList* display_list = DeserializeDisplayItemList(task_data);

  // 2. 创建 GPU-backed SkCanvas
  SkCanvas* canvas = CreateGpuCanvas(tile_size, color_space);

  // 3. Playback DisplayItemList
  //    这会执行所有的 cc::PaintOp，包括我们的 DrawRecordOp
  display_list->Raster(canvas, ...);

  // 4. DrawRecordOp::Raster 内部会调用 PaintRecord::Playback
  //    从而执行 Blink 的 DrawRectOp
  //    最终调用 SkCanvas::drawRect → GPU 命令

  // 5. 完成光栅化，返回纹理句柄给 Compositor 线程
  return gpu_texture_handle;
}
```

详细的 Raster 流程请参考：[blink_raster_analysis.md](https://www.notion.so/blink_raster_analysis.md)

---

# CAP 的性能优化

## 1. 智能分层 (Smart Layerization)

**问题**：过度分层会浪费内存和带宽，过少分层会导致不必要的重绘。

**CAP 的解决方案**：

- **Overlap 检测**：只有真正重叠的内容才分层
- **Decompositing**：某些 effect 可以被"拆解"避免分层
- **距离限制**：只检查最近的 N 个层进行合并（避免 O(n²) 复杂度）

```cpp
// Layerizer 中的优化
wtf_size_t candidate_index = pending_layers_.size() - 1;
while (candidate_index > first_layer_in_current_group &&
       pending_layers_.size() - candidate_index <= layer_merge_distance_limit_) {
  // 只检查最近的 layer_merge_distance_limit_ 个层 (默认 50)
  --candidate_index;
  if (candidate_layer.Merge(new_layer, ...)) {
    pending_layers_.pop_back();
    break;
  }
  if (new_layer.MightOverlap(candidate_layer)) {
    break;  // overlap，停止尝试
  }
}
```

## 2. DisplayItemList 缓存

```cpp
void ContentLayerClientImpl::UpdateCcPictureLayer(...) {
  // 快速路径：如果内容未变，复用旧的 DisplayItemList
  bool may_be_unchanged =
      cc_display_item_list_ &&
      layer_bounds == old_layer_bounds &&
      !raster_invalidator_->HasInvalidations();

  if (may_be_unchanged) {
    return;  // 跳过整个转换！
  }

  // 否则重新生成 DisplayItemList
  cc_display_item_list_ = base::MakeRefCounted<cc::DisplayItemList>();
  PaintChunksToCcLayer::ConvertInto(..., *cc_display_item_list_);
}
```

## 3. 属性树直接更新

对于动画等场景，可以跳过整个 CAP 流程：

```cpp
bool PaintArtifactCompositor::DirectlyUpdateTransform(
    const TransformPaintPropertyNode& transform) {
  // 直接更新 cc::TransformNode，无需重新 Paint 或 Layerize
  cc::TransformNode* cc_node =
      root_layer_->layer_tree_host()
          ->property_trees()
          ->transform_tree_mutable()
          .Node(transform.CcNodeId(sequence_number_));

  if (cc_node) {
    cc_node->local = transform.Matrix();
    cc_node->needs_local_transform_update = true;
    return true;
  }
  return false;
}
```

## 4. PaintRecord 共享

DrawRecordOp 直接引用 Blink 的 PaintRecord，避免拷贝：

```
Blink 主线程:
  DrawingDisplayItem.paint_record (RefPtr)
    ↓ 引用计数 +1
cc 主线程:
  cc::DrawRecordOp.record (SharedPtr, 指向同一个 PaintRecord)
    ↓ 序列化传输 (只传输一次)
Compositor/Viz 线程:
  反序列化后的 PaintRecord (深拷贝)
```

---

# 总结

CAP (Compositing After Paint) 是 Chromium 渲染管线的核心架构，连接了 Blink Paint 和 Compositor：

## 三个阶段

1. **Layerize**：分析 PaintChunks，决定分层策略 → PendingLayers
2. **UpdateCompositedLayer**：创建 cc::Layer，转换内容 → cc::DisplayItemList
3. **PropertyTreeManager**：创建属性树节点，关联到 layers

## 关键数据结构

- **PaintArtifact** (输入)：Blink Paint 的输出，包含 DisplayItems 和 PaintChunks
- **PendingLayer** (中间)：CAP 的核心，表示一个待合成的层
- **cc::Layer** (输出)：Compositor 的 layer，持有 DisplayItemList 和属性树索引
- **cc::DisplayItemList**：cc 格式的绘制指令列表，包含 DrawRecordOp
- **cc::PropertyTrees**：独立的属性树，支持高效的属性动画

## 性能关键点

- **智能分层**：避免过度分层和不必要的重绘
- **DisplayItemList 缓存**：内容未变时跳过转换
- **属性树直接更新**：动画不需要重新 Paint
- **PaintRecord 共享**：避免重复序列化

## 与其他阶段的关系

```
Layout → PrePaint → Paint → 【CAP】 → Commit → Raster → Display
                      ↓        ↓
                PaintArtifact  cc::Layer + DisplayItemList
```

CAP 是理解 Chromium 渲染性能的关键，它的优化直接影响页面的流畅度和响应性。