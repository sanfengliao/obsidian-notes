# 入口

```cpp
void FramePainter::Paint(GraphicsContext& context, PaintFlags paint_flags) {
  Document* document = GetFrameView().GetFrame().GetDocument();

  if (GetFrameView().ShouldThrottleRendering() || !document->IsActive())
    return;

  GetFrameView().NotifyPageThatContentAreaWillPaint();

  LayoutView* layout_view = GetFrameView().GetLayoutView();

  bool is_top_level_painter = !in_paint_contents_;
  in_paint_contents_ = true;

  // 是一个用于防止在绘制过程中清除字体缓存的类。它的主要作用是在绘制操作期间，确保字体缓存不会被清除，从而避免在绘制过程中出现字体缺失或重新加载字体的情况。
  FontCachePurgePreventer font_cache_purge_preventer;
  // ScopedDisplayItemFragment 类用于管理显示项片段。在构造函数中，设置当前片段。在析构函数中，恢复原始片段。
  ScopedDisplayItemFragment display_item_fragment(context, 0u);

  PaintLayer* root_layer = layout_view->Layer();

  PaintLayerPainter layer_painter(*root_layer);

  layer_painter.Paint(context, paint_flags);

  // Regions may have changed as a result of the visibility/z-index of element
  // changing.
  if (document->DraggableRegionsDirty()) {
    GetFrameView().UpdateDocumentDraggableRegions();
  }

  if (is_top_level_painter) {
    // Everything that happens after paintContents completions is considered
    // to be part of the next frame.
    in_paint_contents_ = false;
  }
}
```

