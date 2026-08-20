# 概述

`PaintOp::Raster` 是 Chromium Paint 系统中将绘制操作（PaintOp）转换为实际 Skia 绘制调用的核心机制。本文档详细分析其实现原理和调用流程。

# 架构设计

## 1. 核心接口

```cpp
// cc/paint/paint_op.h
class PaintOp {
 public:
  // 统一的光栅化入口
  void Raster(SkCanvas* canvas, const PlaybackParams& params) const;

  uint8_t type;  // PaintOp 的类型（枚举值）
};
```

**关键特点**:

- 所有 PaintOp 子类共享同一个虚函数入口
- 通过 `type` 字段进行运行时类型调度
- 使用函数指针数组实现高效的类型派发

## 2. 类型系统

`PaintOpType` 枚举定义了所有支持的绘制操作类型：

```cpp
enum class PaintOpType : uint8_t {
  // 注释操作
  kAnnotate,

  // 裁剪操作
  kClipPath,
  kClipRect,
  kClipRRect,

  // 变换操作
  kConcat,
  kRotate,
  kScale,
  kTranslate,
  kSetMatrix,

  // 绘制操作
  kDrawColor,
  kDrawRect,
  kDrawRRect,
  kDrawPath,
  kDrawLine,
  kDrawLineLite,
  kDrawArc,
  kDrawArcLite,
  kDrawOval,
  kDrawImage,
  kDrawImageRect,
  kDrawIRect,
  kDrawDRRect,
  kDrawRecord,
  kDrawScrollingContents,
  kDrawSkottie,
  kDrawTextBlob,
  kDrawSlug,
  kDrawVertices,

  // 状态操作
  kSave,
  kSaveLayer,
  kSaveLayerAlpha,
  kSaveLayerFilters,
  kRestore,

  // 其他
  kCustomData,
  kNoop,
  kSetNodeId,

  kLastPaintOpType = kTranslate,
};
```

**总共 33 种操作类型**，涵盖了 2D 图形绘制的所有基础能力。

## 3. 双层继承体系

PaintOp 使用两层继承体系支持不同类型的操作：

```
PaintOp (基类)
  ├─ PaintOpBaseInternal (不带 PaintFlags 的 Ops)
  │   ├─ AnnotateOp
  │   ├─ ClipPathOp
  │   ├─ RestoreOp
  │   ├─ RotateOp
  │   ├─ SaveOp
  │   ├─ DrawRecordOp
  │   └─ ...
  │
  └─ PaintOpWithFlags (带 PaintFlags 的 Ops)
      ├─ PaintOpWithFlagsBaseInternal
      │   ├─ DrawRectOp
      │   ├─ DrawPathOp
      │   ├─ DrawImageOp
      │   ├─ DrawLineOp
      │   ├─ SaveLayerOp
      │   └─ ...
```

**关键区别**:

- **PaintOpBaseInternal**: 不需要 PaintFlags（画笔属性）的操作
  - 例如：变换、裁剪、状态保存/恢复
  - 实现 `static void Raster(const T* op, SkCanvas* canvas, const PlaybackParams& params)`
- **PaintOpWithFlagsBaseInternal**: 需要 PaintFlags 的操作
  - 例如：绘制矩形、路径、图片等
  - 实现 `static void RasterWithFlags(const T* op, const PaintFlags* flags, SkCanvas* canvas, const PlaybackParams& params)`
  - PaintFlags 包含：颜色、样式、混合模式、滤镜等画笔属性

# 实现机制

## 1. 函数指针数组派发

`PaintOp::Raster` 的核心实现使用函数指针数组实现高效的类型派发：

```cpp
// cc/paint/paint_op.cc

// 1. 定义 Rasterizer 模板 - 处理不带 PaintFlags 的 Ops
template <typename T>
struct Rasterizer<T, false> {
  static void Raster(const T* op,
                     SkCanvas* canvas,
                     const PlaybackParams& params) {
    static_assert(!T::kHasPaintFlags,
                  "This function should not be used for a PaintOp that has PaintFlags");
    DCHECK(op->IsValid());
    T::Raster(op, canvas, params);  // 调用子类的静态 Raster 方法
  }
};

// 2. 定义 Rasterizer 模板特化 - 处理带 PaintFlags 的 Ops
template <typename T>
struct Rasterizer<T, true> {
  static void RasterWithFlags(const T* op,
                              const PaintFlags* flags,
                              SkCanvas* canvas,
                              const PlaybackParams& params) {
    static_assert(T::kHasPaintFlags,
                  "This function expects the PaintOp to have PaintFlags");
    DCHECK(op->IsValid());
    T::RasterWithFlags(op, flags, canvas, params);  // 调用子类的 RasterWithFlags
  }

  static void Raster(const T* op,
                     SkCanvas* canvas,
                     const PlaybackParams& params) {
    static_assert(T::kHasPaintFlags,
                  "This function expects the PaintOp to have PaintFlags");
    DCHECK(op->IsValid());
    // 使用 op 自带的 flags
    T::RasterWithFlags(op, &op->flags, canvas, params);
  }
};

// 3. 定义函数指针类型
using RasterFunction = void (*)(const PaintOp* op,
                                SkCanvas* canvas,
                                const PlaybackParams& params);

// 4. 构建函数指针数组 - 每个 PaintOpType 对应一个函数
#define M(T)                                                              \\\\
  [](const PaintOp* op, SkCanvas* canvas, const PlaybackParams& params) { \\\\
    Rasterizer<T, T::kHasPaintFlags>::Raster(static_cast<const T*>(op),   \\\\
                                             canvas, params);             \\\\
  },
constexpr std::array<RasterFunction, kNumOpTypes> g_raster_functions = {
    TYPES(M)  // TYPES 宏展开为所有 PaintOp 类型
};
#undef M

// 5. PaintOp::Raster 实现 - 通过 type 索引函数数组
void PaintOp::Raster(SkCanvas* canvas, const PlaybackParams& params) const {
  g_raster_functions[type](this, canvas, params);
}
```

**工作流程**:

1. `PaintOp::Raster()` 被调用
2. 使用 `type` 索引 `g_raster_functions` 数组
3. 调用对应的函数指针（lambda）
4. Lambda 中调用 `Rasterizer<T, kHasPaintFlags>::Raster`
5. Rasterizer 根据是否有 PaintFlags 调用相应的子类方法
6. 子类的静态方法执行实际的 Skia 绘制

## 2. RasterWithFlags 机制

对于需要 PaintFlags 的操作，还有一个额外的函数指针数组：

```cpp
using RasterWithFlagsFunction = void (*)(const PaintOp* op,
                                         const PaintFlags* flags,
                                         SkCanvas* canvas,
                                         const PlaybackParams& params);

#define M(T)                                                       \\\\
  [](const PaintOp* op, const PaintFlags* flags, SkCanvas* canvas, \\\\
     const PlaybackParams& params) {                              \\\\
    Rasterizer<T, T::kHasPaintFlags>::RasterWithFlags(            \\\\
        static_cast<const T*>(op), flags, canvas, params);        \\\\
  },
constexpr std::array<RasterWithFlagsFunction, kNumOpTypes>
    g_raster_with_flags_functions = {TYPES(M)};
#undef M

// PaintOpWithFlags 的 RasterWithFlags 实现
void PaintOpWithFlags::RasterWithFlags(SkCanvas* canvas,
                                       const PaintFlags* raster_flags,
                                       const PlaybackParams& params) const {
  g_raster_with_flags_functions[type](this, raster_flags, canvas, params);
}
```

**用途**: 允许外部传入自定义的 PaintFlags，而不是使用 op 自带的 flags。这在某些优化场景下很有用，例如批量绘制时复用同一个 PaintFlags。

# 具体实现示例

## 1. 简单操作 - RestoreOp

```cpp
// cc/paint/paint_op.cc

void RestoreOp::Raster(const RestoreOp* op,
                       SkCanvas* canvas,
                       const PlaybackParams& params) {
  canvas->restore();  // 直接调用 Skia API
}
```

**特点**:

- 不需要 PaintFlags
- 直接映射到 Skia 的 canvas->restore()
- 无额外参数

### 2. 变换操作 - RotateOp

```cpp
void RotateOp::Raster(const RotateOp* op,
                      SkCanvas* canvas,
                      const PlaybackParams& params) {
  canvas->rotate(op->degrees);  // 旋转 canvas
}
```

**特点**:

- 携带参数（degrees）
- 直接调用 Skia 变换 API

## 3. 绘制操作 - DrawRectOp

```cpp
void DrawRectOp::RasterWithFlags(const DrawRectOp* op,
                                 const PaintFlags* flags,
                                 SkCanvas* canvas,
                                 const PlaybackParams& params) {
  // 使用 PaintFlags 的 DrawToSk 辅助函数
  flags->DrawToSk(canvas, [op](SkCanvas* c, const SkPaint& p) {
    c->drawRect(op->rect, p);  // 使用转换后的 SkPaint 绘制矩形
  });
}
```

**特点**:

- 需要 PaintFlags
- `PaintFlags::DrawToSk()` 将 PaintFlags 转换为 Skia 的 SkPaint
- 处理特殊效果（如 DrawLooper 用于阴影）

## 4. 复杂绘制 - DrawImageOp

```cpp
void DrawImageOp::RasterWithFlags(const DrawImageOp* op,
                                  const PaintFlags* flags,
                                  SkCanvas* canvas,
                                  const PlaybackParams& params) {
  DCHECK(!op->image.IsPaintWorklet());
  SkPaint paint = flags ? flags->ToSkPaint() : SkPaint();

  // 1. 处理 PaintWorklet (动态生成的图片)
  if (params.image_provider && op->image.IsDeferredPaintRecord()) {
    // 获取 PaintWorklet 的结果
    ImageProvider::ScopedResult result =
        params.image_provider->GetRasterContent(DrawImage(op->image));

    if (result && result.has_paint_record()) {
      // 递归光栅化 PaintRecord
      SkAutoCanvasRestore save_restore(canvas, true);
      canvas->translate(op->left, op->top);
      result.paint_record().Playback(canvas, params);
    }
    return;
  }

  // 2. 获取 SkImage（从 ImageProvider 或直接使用）
  sk_sp<SkImage> sk_image;
  sk_sp<SkImage> gainmap_sk_image;  // HDR 增益图
  SkSamplingOptions sampling = op->sampling;

  if (params.image_provider) {
    // 通过 ImageProvider 获取解码后的图片
    DrawImage draw_image(op->image, false,
                         SkIRect::MakeWH(op->image.width(), op->image.height()),
                         op->GetImageQuality(), canvas->getLocalToDevice());
    auto scoped_result = params.image_provider->GetRasterContent(draw_image);
    if (scoped_result) {
      sk_image = scoped_result.decoded_image().image();
      gainmap_sk_image = scoped_result.decoded_image().gainmap_image();
      // ... 处理缩放调整和采样选项
    }
  } else {
    // 直接使用图片
    sk_image = op->image.GetSwSkImage();
    gainmap_sk_image = op->image.gainmap_sk_image_;
  }

  if (!sk_image) {
    return;  // 无效图片，跳过
  }

  // 3. 处理 HDR 增益图（Gainmap）
  if (op->image.HasGainmapInfo() && gainmap_sk_image) {
    skia::DrawGainmapImage(
        canvas, sk_image, gainmap_sk_image, op->image.gainmap_info_.value(),
        std::exp2(ComputeEffectiveHdrHeadroom(flags, params)),
        op->left, op->top, sampling, paint);
    return;
  }

  // 4. 处理 HDR 色调映射
  if (ToneMapUtil::UseGlobalToneMapFilter(sk_image.get(),
                                          canvas->imageInfo().colorSpace())) {
    ToneMapUtil::AddGlobalToneMapFilterToPaint(
        paint, sk_image.get(), hdr_metadata,
        ComputeEffectiveHdrHeadroom(flags, params));
  }

  // 5. 最终绘制
  SkTiledImageUtils::DrawImage(canvas, sk_image.get(), op->left, op->top,
                               sampling, &paint);
}
```

**关键处理**:

- **PaintWorklet**: 动态生成的图片（CSS Paint API）
- **ImageProvider**: 图片解码和缓存管理
- **HDR 支持**: 增益图（Gainmap）和色调映射
- **采样选项**: 根据缩放比例选择合适的采样算法

## 5. 递归操作 - DrawRecordOp

```cpp
void DrawRecordOp::Raster(const DrawRecordOp* op,
                          SkCanvas* canvas,
                          const PlaybackParams& params) {
  // 不使用 drawPicture，因为它会添加隐式裁剪
  // 直接递归 playback 内部的 PaintRecord
  op->record.Playback(canvas, params, op->local_ctm);
}
```

**特点**:

- DrawRecordOp 包含一个嵌套的 PaintRecord（PaintOpBuffer）
- 递归调用 Playback，执行内部的所有 PaintOps
- 类似于红色 div 示例中的 DrawRecordOp 包含 DrawRectOp

## 6. 特殊操作 - DrawScrollingContentsOp

```cpp
void DrawScrollingContentsOp::Raster(const DrawScrollingContentsOp* op,
                                     SkCanvas* canvas,
                                     const PlaybackParams& params) {
  canvas->save();

  // 应用滚动偏移
  CHECK(params.raster_inducing_scroll_offsets);
  gfx::PointF scroll_offset =
      params.raster_inducing_scroll_offsets->at(op->scroll_element_id);
  canvas->translate(-scroll_offset.x(), -scroll_offset.y());

  // 光栅化 DisplayItemList
  op->display_item_list->Raster(canvas, params);

  canvas->restore();
}
```

**特点**:

- 用于滚动优化（Compositor-driven scrolling）
- 根据 scroll_element_id 应用当前的滚动偏移
- 无需重新 paint，只需重新光栅化

## 7. 动画操作 - DrawSkottieOp

```cpp
void DrawSkottieOp::Raster(const DrawSkottieOp* op,
                           SkCanvas* canvas,
                           const PlaybackParams& params) {
  // 绘制 Lottie/Skottie 动画
  op->skottie->Draw(
      canvas, op->t,  // t: 当前时间/帧
      op->dst,        // 目标矩形
      base::BindRepeating(&DrawSkottieOp::GetImageAssetForRaster,
                          base::Unretained(op), canvas, std::cref(params)),
      op->color_map,  // 颜色覆盖
      op->text_map);  // 文本覆盖
}
```

**特点**:

- 支持 Lottie 动画（After Effects 导出的 JSON 动画）
- 动态图片资源获取（通过 ImageProvider）
- 颜色和文本属性覆盖

# PlaybackParams 参数

`PlaybackParams` 携带光栅化所需的上下文信息：

```cpp
struct PlaybackParams {
  // 图片提供者：解码、缓存、生成图片
  ImageProvider* image_provider = nullptr;

  // 原始 CTM（Current Transform Matrix）
  SkM44 original_ctm;

  // 自定义回调（用于 CustomDataOp）
  struct {
    CustomDataRasterCallback custom_callback;
    ConvertOpCallback convert_op_callback;
  } callbacks;

  // 滚动偏移映射（用于 DrawScrollingContentsOp）
  const base::flat_map<ElementId, gfx::PointF>* raster_inducing_scroll_offsets = nullptr;

  // 目标 HDR headroom（HDR 显示器的亮度范围）
  float destination_hdr_headroom = 1.0f;

  // 是否正在分析（而非实际光栅化）
  bool is_analyzing = false;

  // SaveLayerAlpha 是否保留 LCD 文本
  std::optional<bool> save_layer_alpha_should_preserve_lcd_text;
};
```

# 性能优化

## 1. 函数指针数组 vs 虚函数

**传统虚函数方式**:

```cpp
class PaintOp {
 public:
  virtual void Raster(SkCanvas* canvas) = 0;  // 虚函数
};

// 调用
op->Raster(canvas);  // 1 次虚函数调用（查 vtable）
```

**当前函数指针数组方式**:

```cpp
void PaintOp::Raster(SkCanvas* canvas, const PlaybackParams& params) const {
  g_raster_functions[type](this, canvas, params);  // 直接数组索引
}
```

**优势**:

- **更快**: 数组索引 + 函数指针调用 vs vtable 查找 + 虚函数调用
- **更小**: PaintOp 不需要 vtable 指针（节省 8 字节/对象）
- **更灵活**: 可以轻松添加其他函数数组（如 g_serialize_functions）
- **编译时优化**: constexpr 数组可以在编译时完全优化

### 2. 静态函数 vs 成员函数

```cpp
// 子类实现为静态函数，而不是成员函数
static void DrawRectOp::RasterWithFlags(const DrawRectOp* op, ...);
// 而不是
void DrawRectOp::RasterWithFlags(...);
```

**优势**:

- 可以作为函数指针使用
- 编译器更容易内联
- 避免 this 指针调整

## 3. 避免动态分配

```cpp
// PaintOps 通常分配在 PaintOpBuffer 的内存块中
// 而不是通过 new/delete 单独分配
class PaintOpBuffer {
  char* data_;  // 连续内存块
  // PaintOps 紧密排列在 data_ 中
};
```

**优势**:

- 更好的缓存局部性
- 减少内存碎片
- 批量分配/释放更高效

# 调用链示例

以红色 div 的 DrawRectOp 为例：

```
1. DisplayItemList::Raster(canvas, params)
   └─ PaintOpBuffer::Playback(canvas, params)

2. 遍历 PaintOpBuffer 中的每个 PaintOp:
   for (const PaintOp& op : buffer) {
     op.Raster(canvas, params);
   }

3. PaintOp::Raster (基类方法)
   └─ g_raster_functions[type](this, canvas, params)

4. Lambda 函数 (从 g_raster_functions 数组)
   └─ Rasterizer<DrawRectOp, true>::Raster(op, canvas, params)

5. Rasterizer::Raster (模板方法)
   └─ DrawRectOp::RasterWithFlags(op, &op->flags, canvas, params)

6. DrawRectOp::RasterWithFlags (子类静态方法)
   └─ flags->DrawToSk(canvas, [op](SkCanvas* c, const SkPaint& p) {
        c->drawRect(op->rect, p);
      })

7. PaintFlags::DrawToSk
   ├─ 转换 PaintFlags → SkPaint
   ├─ 处理 DrawLooper (阴影效果)
   └─ 调用 lambda: canvas->drawRect(rect, paint)

8. SkCanvas::drawRect (Skia API)
   └─ GrRenderTargetContext::drawRect (GPU 后端)
      └─ 生成 GPU 绘制命令
```

# 类型检查与验证

## 1. 编译时检查

```cpp
template <typename T>
struct Rasterizer<T, false> {
  static void Raster(const T* op, SkCanvas* canvas, const PlaybackParams& params) {
    // 编译时断言：确保 T 不应该有 PaintFlags
    static_assert(!T::kHasPaintFlags,
                  "This function should not be used for a PaintOp that has PaintFlags");
    T::Raster(op, canvas, params);
  }
};
```

## 2. 运行时检查

```cpp
template <typename T>
struct Rasterizer<T, true> {
  static void Raster(const T* op, SkCanvas* canvas, const PlaybackParams& params) {
    // 运行时检查：确保 op 处于有效状态
    DCHECK(op->IsValid());
    T::RasterWithFlags(op, &op->flags, canvas, params);
  }
};
```

## 3. Op 有效性验证

每个 PaintOp 子类实现 `IsValid()` 方法：

```cpp
// DrawRectOp
bool IsValid() const {
  return rect.isFinite() && flags.IsValid();
}

// ClipPathOp
bool IsValid() const {
  return IsValidSkClipOp(op) && IsValidPath(path);
}

// DrawImageOp
bool IsValid() const {
  return left == left && top == top &&  // 检查 NaN
         flags.IsValid();
}
```

# 扩展性

## 添加新的 PaintOp 类型

1. **定义枚举值**:

```cpp
enum class PaintOpType : uint8_t {
  // ...
  kDrawMyCustomOp,  // 新操作
  // ...
};
```

1. **实现 Op 类**:

```cpp
class DrawMyCustomOp final : public PaintOpWithFlagsBaseInternal {
 public:
  static constexpr PaintOpType kType = PaintOpType::kDrawMyCustomOp;

  // Raster 实现
  static void RasterWithFlags(const DrawMyCustomOp* op,
                              const PaintFlags* flags,
                              SkCanvas* canvas,
                              const PlaybackParams& params) {
    // 实现绘制逻辑
    flags->DrawToSk(canvas, [op](SkCanvas* c, const SkPaint& p) {
      // 调用 Skia API
    });
  }

  // 序列化/反序列化
  HAS_SERIALIZATION_FUNCTIONS();

  bool IsValid() const { /* 验证逻辑 */ }
  bool EqualsForTesting(const DrawMyCustomOp& other) const { /* ... */ }

  // Op 数据成员
  MyCustomData data;
};
```

1. **添加到 TYPES 宏**:

```cpp
#define TYPES(M)       \\\\
  M(AnnotateOp)        \\\\
  M(ClipPathOp)        \\\\
  /* ... */            \\\\
  M(DrawMyCustomOp)    \\\\  // 新增
  /* ... */
```

1. **自动生成**:

- `g_raster_functions` 数组自动包含新 Op
- `g_serialize_functions` 数组自动包含新 Op
- `g_deserialize_functions` 数组自动包含新 Op

# 总结

## 核心设计理念

1. **类型安全**: 编译时和运行时双重检查
2. **高性能**: 函数指针数组 + 静态函数 + 内联优化
3. **可扩展**: 添加新 Op 只需定义类和更新宏
4. **统一接口**: 所有 Ops 通过相同的 Raster() 入口
5. **灵活性**: 支持递归、动态图片、动画等复杂场景

## 关键优化技术

- **函数指针数组**: O(1) 类型派发，无 vtable 开销
- **模板特化**: 根据是否有 PaintFlags 选择不同的代码路径
- **静态函数**: 便于内联和函数指针使用
- **constexpr 数组**: 编译时完全构建，无运行时开销
- **内存连续性**: PaintOps 存储在 PaintOpBuffer 的连续内存中

## 与 Skia 的映射

| PaintOp      | Skia API            |
| ------------ | ------------------- |
| DrawRectOp   | SkCanvas::drawRect  |
| DrawPathOp   | SkCanvas::drawPath  |
| DrawImageOp  | SkCanvas::drawImage |
| ClipRectOp   | SkCanvas::clipRect  |
| TranslateOp  | SkCanvas::translate |
| RotateOp     | SkCanvas::rotate    |
| SaveOp       | SkCanvas::save      |
| RestoreOp    | SkCanvas::restore   |
| DrawRecordOp | 递归 Playback       |

PaintOp 系统本质上是 Skia Canvas API 的高级封装，提供了：

- 序列化支持（用于 IPC）
- 延迟执行（录制-回放模式）
- 优化机会（缓存、合并、跳过）
- 统一的类型系统

这种设计使得 Chromium 能够高效地录制、传输和执行绘制操作，是整个渲染流程的基础。