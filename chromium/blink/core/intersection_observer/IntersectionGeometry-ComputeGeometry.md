`IntersectionObserver` 的核心能力是判断一个元素与另一个元素（或 viewport）的交集比例。听起来简单——两个矩形求交集嘛。但实际实现要处理的问题远比这复杂：元素可能有 CSS transform、可能嵌套在多层滚动容器里、可能跨 iframe、可能是 SVG……
Chromium 把这些计算逻辑封装在 `IntersectionGeometry::ComputeGeometry()` 里。这个方法承担了所有实质性的工作：初始化边界框、应用 margin、执行坐标映射、裁剪交集、判定阈值、计算可见性。从工程角度看，这里是性能瓶颈所在，也是复杂 bug 的温床——坐标空间的转换、缓存的失效条件、各种标志位的组合，任何一个环节出错都会导致难以追踪的视觉问题。
这篇文章按 `ComputeGeometry()` 的执行流程逐步拆解，看看它如何在三个坐标空间之间腾挪，最终算出那个看似简单的交集比例。
# 坐标系统的三层转换
整个流程涉及三个坐标空间：
- 本地坐标系：元素自身的坐标系
- 根容器坐标系：target 几何映射到 root 坐标空间
- 绝对坐标系：相对于 viewport 或顶级文档
代码在这三个空间之间不断转换，搞混任何一个都会导致计算错误。`ComputeGeometry` 通过严格的分步转换来管理这个复杂度。aaaa
# 缓存决策
```cpp
bool pre_margin_target_rect_is_empty;
if (ShouldUseCachedRects()) {
    target_rect_ = cached_rects->local_target_rect;
    pre_margin_target_rect_is_empty =
        cached_rects->pre_margin_target_rect_is_empty;
    
    unclipped_intersection_rect_ =
        cached_rects->unscrolled_unclipped_intersection_rect;
} else {
    target_rect_ = InitializeTargetRect(target, flags_);
    pre_margin_target_rect_is_empty = target_rect_.IsEmpty();
    ApplyMargin(target_rect_, target_margin, root_geometry.zoom,
                root_geometry.pre_margin_local_root_rect.size());
    
    unclipped_intersection_rect_ = target_rect_;
}
```
缓存生效需要三个条件：缓存数据有效、root 和 target 之间没有中间 scroll container、当前帧计算模式是"仅滚动和可见性"。
为什么要检查中间 scroll container？如果路径中有可滚动的中间元素，这些元素的滚动会改变 target 相对于 root 的位置，缓存的矩形就失效了。但如果只有 root 本身可滚动，滚动只改变 root 的 clip 范围，不改变 target 与 root 的相对位置。
`pre_margin_target_rect_is_empty` 这个标志值得注意。target 可能本身就是 empty，也可能因为 margin 变成 empty。后续计算交集比例时需要区分这两种情况——即使 target 矩形经过变换变成非空，如果原始 pre-margin 状态是 empty，阈值计算也要反映这一点。
滚动时，框架会检查累积滚动量是否超过 `ComputeMinScrollDeltaToUpdate()` 返回的最小值，没超过就跳过完整的交集重计算。这个优化在频繁滚动的页面上效果显著。
# 不同元素类型的边界框计算
```cpp
gfx::RectF InitializeTargetRect(const LayoutObject* target, unsigned flags) {
  if (flags & IntersectionGeometry::kForFrameViewportIntersection) {
    return gfx::RectF(To<LayoutEmbeddedContent>(target)->ReplacedContentRect());
  }
  if (target->IsSVGChild()) {
    return target->DecoratedBoundingBox();
  }
  if (auto* layout_box = DynamicTo<LayoutBox>(target)) {
    return GetBoxBounds(layout_box,
                        flags & IntersectionGeometry::kUseOverflowClipEdge);
  }
  if (auto* layout_inline = DynamicTo<LayoutInline>(target)) {
    return layout_inline->LocalBoundingBoxRectF();
  }
  return gfx::RectF(To<LayoutText>(target)->PhysicalLinesBoundingBox());
}
```
不同元素类型有本质不同的布局模型，需要分别处理：
- iframe：`ReplacedContentRect()` 返回 iframe 在父文档中占据的矩形，不是内部文档
- SVG：有独立坐标系统，`DecoratedBoundingBox()` 包含 paint offset
- 普通 Box：HTML block/inline-block 用 border box，`kUseOverflowClipEdge` 标志控制是否包含 overflow 内容（如 box-shadow）
- Inline：不是单一矩形，而是多个 line box 的并集
- Text 节点：用 `PhysicalLinesBoundingBox()` 作为降级方案
`kUseOverflowClipEdge` 标志容易被忽视，但在特殊场景下会导致交集计算不准确。
# 分层裁剪
```cpp
bool does_intersect =
    ClipToRoot(root_and_target, root_rect_, unclipped_intersection_rect_,
               intersection_rect_, scroll_margin, cached_rects);

bool IntersectionGeometry::ClipToRoot(const RootAndTarget& root_and_target,
                                      const gfx::RectF& root_rect,
                                      gfx::RectF& unclipped_intersection_rect,
                                      gfx::RectF& intersection_rect,
                                      const Vector<Length>& scroll_margin,
                                      CachedRects* cached_rects) {
  // ... root 和 target 的类型检查 ...
  
  if (!scroll_margin.empty()) {
    // 对每个中间的 scroll container 应用 clip 和 margin
    for (const LayoutBox* scroller : root_and_target.intermediate_scrollers) {
      gfx::RectF scroller_rect =
          gfx::RectF(scroller->OverflowClipRect(PhysicalOffset()));
      if (std::optional<gfx::RectF> clip_path_box =
              ClipPathClipper::LocalClipPathBoundingBox(*scroller)) {
        scroller_rect.Intersect(*clip_path_box);
      }

      local_ancestor = To<LayoutBox>(scroller);
      if (!ApplyClip(target, local_ancestor, scroller, scroller_rect,
                     unclipped_intersection_rect, intersection_rect,
                     scroll_margin, ignore_local_clip_path,
                     /*root_scrolls_target=*/true, cached_rects)) {
        return false;
      }

      unclipped_intersection_rect = intersection_rect;
      target = scroller;
      ignore_local_clip_path = true;
    }
  }

  // 最后对 root 本身应用 clip
  return ApplyClip(target, local_ancestor, root_and_target.root, root_rect,
                   unclipped_intersection_rect, intersection_rect,
                   scroll_margin, ignore_local_clip_path,
                   root_and_target.root_scrolls_target, cached_rects);
}
```
从 target 开始，逐个向上经过所有中间的 scroll container，最后到达 root。每一步都对 target 矩形进行坐标变换并与 clip 边界相交。
`intermediate_scrollers` 列表在构造函数中填充，包含路径上有 scroll margin 的所有可滚动元素。scroll margin 会扩展 scroll container 的 clip 范围，影响交集计算。
`ignore_local_clip_path` 标志容易遗漏。第一次对 target 本身裁剪时需要考虑 target 的 clip-path，但之后每次迭代，当前 target 变成了上一个中间元素，而上一步已经应用过它的 clip-path，所以设置标志忽略它，避免重复应用。
# 坐标映射与 clip
```cpp
unsigned flags = kDefaultVisualRectFlags | kEdgeInclusive |
                 kDontApplyMainFrameOverflowClip | kUsePreciseClipPath;
if (!ShouldRespectFilters()) {
    flags |= kIgnoreFilters;
}
if (CanUseGeometryMapper(*target)) {
    flags |= kUseGeometryMapper;
}
if (ignore_local_clip_path) {
    flags |= kIgnoreLocalClipPath;
}

bool does_intersect = false;

if (ShouldUseCachedRects()) {
    does_intersect = cached_rects->does_intersect;
} else {
    does_intersect = target->MapToVisualRectInAncestorSpace(
        local_ancestor, unclipped_intersection_rect,
        static_cast<VisualRectFlags>(flags));
    // ... scroll space 转换 ...
}

if (does_intersect) {
    intersection_rect = unclipped_intersection_rect;
    does_intersect &= intersection_rect.InclusiveIntersect(root_clip_rect);
}

return does_intersect;
```
`MapToVisualRectInAncestorSpace()` 处理 transform、filter、clip-path 等视觉效果，然后与 root 的 clip 矩形相交。
标志位组合决定映射行为：`kUseGeometryMapper` 启用高效几何映射器；`kIgnoreFilters` 和 `kIgnoreLocalClipPath` 在某些情况下跳过特定效果，避免重复应用。使用缓存时直接跳过 `MapToVisualRectInAncestorSpace()` 调用。
# 转换到绝对坐标系
```cpp
gfx::Transform target_to_view_transform = ObjectToViewTransform(*target);
target_rect_ = target_to_view_transform.MapRect(target_rect_);

if (does_intersect) {
    gfx::RectF unclipped_intersection_rect;
    if (RootIsImplicit()) {
      TransformState implicit_root_to_target_document_transform(
          TransformState::kUnapplyInverseTransformDirection);
      target->View()->MapAncestorToLocal(
          nullptr, implicit_root_to_target_document_transform,
          kTraverseDocumentBoundaries | kApplyRemoteMainFrameTransform);
      gfx::Transform matrix =
          implicit_root_to_target_document_transform.AccumulatedTransform()
              .InverseOrIdentity();
      intersection_rect_ =
          matrix.ProjectQuad(gfx::QuadF(intersection_rect_)).BoundingBox();
      unclipped_intersection_rect =
          matrix.ProjectQuad(gfx::QuadF(unclipped_intersection_rect_))
              .BoundingBox();
    } else {
      intersection_rect_ =
          root_geometry.root_to_view_transform.MapRect(intersection_rect_);
      unclipped_intersection_rect =
          root_geometry.root_to_view_transform.MapRect(
              unclipped_intersection_rect);
    }
    unclipped_intersection_rect_ = unclipped_intersection_rect;
  } else {
    intersection_rect_ = gfx::RectF();
  }
  
  root_rect_ =
      root_geometry.root_to_view_transform.MapRect(gfx::RectF(root_rect_));
```
将矩形从 root 本地坐标系转换到绝对坐标系（viewport 坐标），为比例计算提供统一基准。
显式 root 直接用 `root_geometry.root_to_view_transform`，这个矩阵在构造函数中预计算好了。
隐式 root 跨 frame 时更复杂。调用 `MapAncestorToLocal(nullptr)` 映射到最顶层，然后取反向变换。用 `ProjectQuad()` 而不是 `MapRect()` 是为了处理透视变换可能产生的四边形，再用 `BoundingBox()` 转回矩形。
target 矩形的变换独立于交集矩形——不相交时就不浪费计算交集变换。
# 阈值判定与交集比例
```cpp
if (does_intersect) {
    const gfx::RectF& comparison_rect =
        ShouldTrackFractionOfRoot() ? root_rect_ : target_rect_;
    bool empty_override =
        !ShouldTrackFractionOfRoot() && pre_margin_target_rect_is_empty;
    if (comparison_rect.IsEmpty() || empty_override) {
      intersection_ratio_ = 1;
    } else {
      const gfx::SizeF& intersection_size = intersection_rect_.size();
      const float intersection_area = intersection_size.GetArea();
      const gfx::SizeF& comparison_size = comparison_rect.size();
      const float area_of_interest = comparison_size.GetArea();
      intersection_ratio_ = std::min(intersection_area / area_of_interest, 1.f);
    }
    threshold_index_ =
        FirstThresholdGreaterThan(intersection_ratio_, thresholds);
  } else {
    intersection_ratio_ = 0;
    threshold_index_ = 0;
}
```
计算交集相对于参考矩形的比例，判定跨越了哪个阈值。`kShouldTrackFractionOfRoot` 标志控制两种模式：
- 默认模式：`ratio = 交集面积 / target 面积`，标准 IntersectionObserver 行为
- 追踪 root 分数模式：`ratio = 交集面积 / root 面积`，观察 root 被覆盖的百分比
参考矩形面积为 0 但两个元素相交（至少边界接触）时，直接设置 ratio = 1。`empty_override` 处理前面提到的 `pre_margin_target_rect_is_empty` 情况。
`std::min(..., 1.f)` 限制结果——理论上交集面积不应超过参考矩形，但浮点精度问题可能导致 ratio > 1。
# 可见性追踪
```cpp
if (IsIntersecting() && ShouldComputeVisibility()) {
    auto visibility_info = ComputeVisibilityInfo(
        target, PhysicalRect::FastAndLossyFromRectF(target_rect_), flags_);
    occluder_node_id_ = visibility_info.occluder_node_id;
    if (visibility_info.is_visible) {
      flags_ |= kIsVisible;
    }
  } else {
    occluder_node_id_ = kInvalidDOMNodeId;
  }
```
判定 target 是否真的"可见"——没有被其他内容遮挡。只在 target 与 root 有交集且用户明确要求追踪可见性时执行。
`ComputeVisibilityInfo()` 执行 hit test，开销很大。追踪 visibility 的 observer 会额外标记 frame 需要 occlusion tracking，整个渲染流程有额外开销。不追踪可见性的用户完全不为这个功能付出代价。
# CSS 像素调整
```cpp
if (flags_ & kShouldConvertToCSSPixels) {
    AdjustForAbsoluteZoom::AdjustRectMaybeExcludingCSSZoom(target_rect_,
                                                           *target);
    AdjustForAbsoluteZoom::AdjustRectMaybeExcludingCSSZoom(intersection_rect_,
                                                           *target);
    AdjustForAbsoluteZoom::AdjustRectMaybeExcludingCSSZoom(root_rect_, *root);
}
```
高分辨率屏幕上设备像素和 CSS 像素比例不是 1:1。内部计算使用设备像素，但用户回调参数应该是 CSS 像素。这个调整必须在最后进行，因为之前所有映射和裁剪都在设备像素空间。
