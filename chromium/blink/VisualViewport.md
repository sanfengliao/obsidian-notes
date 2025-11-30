## Method
### VisibleRect
可视区域的大小，双指触控板捏合可以放大缩小可视区域

```cpp
gfx::RectF VisualViewport::VisibleRect(
    IncludeScrollbarsInRect scrollbar_inclusion) const {
  if (!IsActiveViewport())
    return gfx::RectF(gfx::PointF(), gfx::SizeF(size_));

  gfx::SizeF visible_size(size_);

  if (scrollbar_inclusion == kExcludeScrollbars)
    visible_size = gfx::SizeF(ExcludeScrollbars(size_));

  visible_size.Enlarge(0, browser_controls_adjustment_);
  visible_size.Scale(1 / scale_);

  return gfx::RectF(ScrollPosition(), visible_size);
}
```

