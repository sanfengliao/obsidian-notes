# Blink Raster 流程深度分析

# 目标：将 Paint 结果变成屏幕像素

本文聚焦 Raster 阶段：从 `cc::DisplayItemList` 开始，经过 Tile 调度、跨进程序列化、GPU 执行和 Viz 合成，最终显示到屏幕。

> Paint 阶段请参考：[blink_paint_analysis.md](blink_paint_analysis.md)


# 第一部分：光sl栅化详解 (Rasterization)

前面我们追踪了从 Paint 到创建 `cc::DisplayItemList` 的过程。现在深入最后一步：**如何将 DisplayItemList 转换为 GPU 纹理上的像素**。

## 光栅化的时机和线程架构

现代 Chromium 使用 **OOP-R (Out-of-Process Rasterization)** 架构，光栅化跨越多个进程和线程：

```
渲染进程 (Renderer Process):
  主线程 (Main Thread):
    DOM → Style → Layout → Paint → PaintArtifact
      ↓ 传递给 Compositor

  合成器线程 (Compositor Thread):
    接收 PaintArtifact
      ↓ CAP 转换 → cc::DisplayItemList
      ↓ TileManager 调度 → 创建 RasterTask

  光栅化工作线程 (Raster Worker Threads):
    执行 RasterTask
      ↓ 序列化 DisplayItemList 中的 PaintOps
      ↓ 写入共享内存
      ↓ 通过 GPU Command Buffer Client 发送命令
      ↓ IPC 调用 (跨进程)

GPU 进程 (GPU Process):
  GPU 主线程:
    RasterDecoderImpl 接收命令
      ↓ 从共享内存读取序列化的 PaintOps
      ↓ 流式反序列化 + 执行
      ↓ 调用 Skia API (canvas->drawRect 等)
      ↓ 生成 GPU 绘制命令 (Vulkan/OpenGL/Metal)
      ↓ Fragment Shader 执行 (GPU 硬件)
      ↓ 写入 SharedImage (GPU 纹理)
```

**为什么要多进程架构？**
- **进程隔离**: GPU 进程运行在沙箱中，提高安全性和稳定性（GPU 驱动崩溃不影响渲染进程）
- **真正的并行**: 渲染进程可以继续处理下一帧，GPU 进程并行执行光栅化
- **主线程不阻塞**: JavaScript、DOM 操作、用户输入响应不受光栅化影响
- **Compositor 不阻塞**: 合成器线程专注于调度和合成，不执行耗时的绘制
- **Worker 线程池**: 多个 tiles 可以并行序列化和发送命令
- **GPU 硬件加速**: Fragment Shader 在 GPU 上并行处理像素，速度远超 CPU

**关键理解**：
- Raster Worker Thread **不直接光栅化**，而是**序列化绘制命令并发送到 GPU 进程**
- 真正的光栅化（像素绘制）**发生在 GPU 进程的 Skia GPU 后端**，最终由 **GPU 硬件执行**

## Tile (瓦片) 的概念

页面不是一次性整体光栅化,而是分成 **Tile** (通常 256x256 或 512x512 像素):

```
页面 (1920x1080):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Tile │Tile │Tile │Tile │Tile │Tile │Tile │ 每个 256x256
│ 0,0 │ 1,0 │ 2,0 │ 3,0 │ 4,0 │ 5,0 │ 6,0 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│Tile │Tile │Tile │Tile │Tile │Tile │Tile │
│ 0,1 │ 1,1 │ 2,1 │ 3,1 │ 4,1 │ 5,1 │ 6,1 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│Tile │Tile │Tile │Tile │Tile │Tile │Tile │
│ 0,2 │ 1,2 │ 2,2 │ 3,2 │ 4,2 │ 5,2 │ 6,2 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│Tile │Tile │Tile │Tile │Tile │Tile │Tile │
│ 0,3 │ 1,3 │ 2,3 │ 3,3 │ 4,3 │ 5,3 │ 6,3 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

**好处**:
1. **按需光栅化**: 只光栅化可见区域 + 周边的 tile
2. **并行处理**: 多个 tile 可以同时在不同线程光栅化
3. **内存优化**: 不需要为整个页面分配纹理
4. **缓存粒度**: Tile 级别缓存,滚动时复用未变化的 tile

**对于红色 div** (假设在 (0, 0) 位置,大小 100x100):
- 它完全在 Tile(0,0) 内
- 只需要光栅化 Tile(0,0)
- 其他 tile 可以延迟光栅化

# 完整光栅化流程

## 1. TileManager 调度

在合成器线程,`TileManager` 负责决定哪些 tile 需要光栅化:

```cpp
// cc/tiles/tile_manager.cc
bool TileManager::PrepareTiles(
    const GlobalStateThatImpactsTilePriority& state) {
  ++prepare_tiles_count_;

  if (!tile_task_manager_)
    return false;

  // 1. 确保处理上次遗留的已完成任务
  if (!did_check_for_completed_tasks_since_last_schedule_tasks_) {
    tile_task_manager_->CheckForCompletedTasks();
    did_check_for_completed_tasks_since_last_schedule_tasks_ = true;
  }

  // 2. 分配 GPU 内存，并对每个需要光栅化的 tile 调用 CreateRasterTask
  //    CreateRasterTask 在此函数内部被调用，返回带优先级的任务列表
  PrioritizedWorkToSchedule prioritized_work = AssignGpuMemoryToTiles();

  // 3. 以 TaskGraph 形式批量提交给 TaskGraphRunner
  ScheduleTasks(std::move(prioritized_work));

  return true;
}
```

`AssignGpuMemoryToTiles` 内部会按优先级遍历所有 tile，对尚未持有 raster task 的 tile 调用 `CreateRasterTask`：

```cpp
// cc/tiles/tile_manager.cc
TileManager::PrioritizedWorkToSchedule TileManager::AssignGpuMemoryToTiles() {
  // 1. 建立优先级队列（按距 viewport 距离与优先级排序）
  std::unique_ptr<RasterTilePriorityQueue> raster_priority_queue(
      client_->BuildRasterQueue(global_state_.tree_priority,
                                RasterTilePriorityQueue::Type::ALL));

  PrioritizedWorkToSchedule work_to_schedule;

  for (; !raster_priority_queue->IsEmpty(); raster_priority_queue->Pop()) {
    const PrioritizedTile& prioritized_tile = raster_priority_queue->Top();
    Tile* tile = prioritized_tile.tile();
    TilePriority priority = prioritized_tile.priority();

    // 2. 超出内存策略的 tile 直接跳过
    if (TilePriorityViolatesMemoryPolicy(priority))
      break;

    // 3. 固色分析（可选）：若整块 tile 为单一颜色，直接标记 solid_color 跳过光栅化
    if (!tile->is_solid_color_analysis_performed() &&
        tile->use_picture_analysis() && kUseColorEstimator) {
      tile->set_solid_color_analysis_performed(true);
      SkColor4f color;
      if (raster_source->PerformSolidColorAnalysis(
              tile->enclosing_layer_rect(), &color, /*max_ops=*/5)) {
        tile->draw_info().set_solid_color(color);
        client_->NotifyTileStateChanged(tile, /*update_damage=*/true,
                                        /*set_needs_redraw=*/true);
        continue;  // 无需光栅化
      }
    }

    // 4. 仅需处理图像解码（checker-image）的 tile 特殊处理
    if (tile->is_prepaint() && prioritized_tile.is_process_for_images_only()) {
      work_to_schedule.tiles_to_process_for_images.push_back(prioritized_tile);
      continue;
    }

    // 5. 判断内存是否充足：先尝试驱逐低优先级 tile，再看是否超限
    MemoryUsage memory_required = MemoryUsage::FromConfig(
        tile->desired_texture_size(), client_->GetTileFormat());
    eviction_priority_queue =
        FreeTileResourcesWithLowerPriorityUntilUsageIsWithinLimit(...);
    if (!memory_usage_is_within_limit) {
      all_tiles_that_need_to_be_rasterized_are_scheduled_ = false;
      break;
    }

    // 6. 若 tile 已有 raster_task（上一帧遗留），只需把 checker-images 加入解码队列
    if (tile->HasRasterTask()) {
      if (tile->raster_task_scheduled_with_checker_images())
        AddCheckeredImagesToDecodeQueue(...);
    } else {
      // 7. 核心：为没有任务的 tile 创建 RasterTask
      auto raster_task = CreateRasterTask(prioritized_tile,
                                          target_color_params,
                                          &work_to_schedule);
      if (!raster_task)
        continue;
      tile->raster_task_ = std::move(raster_task);
      tile->mark_used();
    }

    memory_usage += memory_required;
    work_to_schedule.tiles_to_raster.push_back(prioritized_tile);
  }
  return work_to_schedule;
}
```

`CreateRasterTask` 的关键步骤：

```cpp
scoped_refptr<TileTask> TileManager::CreateRasterTask(
    const PrioritizedTile& prioritized_tile,
    const TargetColorParams& target_color_params,
    PrioritizedWorkToSchedule* work_to_schedule) {
  Tile* tile = prioritized_tile.tile();

  // A. 确定像素格式（HDR 内容升级为 RGBA_F16，否则使用默认格式）
  auto format = client_->GetTileFormat();
  if (target_color_params.color_space.IsHDR() && content_is_hdr)
    format = viz::SinglePlaneFormat::kRGBA_F16;

  // B. 获取 GPU 内存资源
  //    优先尝试部分光栅化（只更新失效区域），否则申请全新资源
  ResourcePool::InUsePoolResource resource;
  gfx::Rect invalidated_rect = tile->invalidated_content_rect();
  if (UsePartialRaster(msaa_sample_count) && tile->invalidated_id()) {
    resource = resource_pool_->TryAcquireResourceForPartialRaster(
        tile->id(), tile->invalidated_content_rect(), tile->invalidated_id(),
        &invalidated_rect, target_color_params.color_space, debug_name);
  }
  if (!resource)
    resource = resource_pool_->AcquireResource(
        tile->desired_texture_size(), format,
        target_color_params.color_space, debug_name);

  // C. 配置 PlaybackSettings
  RasterSource::PlaybackSettings playback_settings;
  playback_settings.use_lcd_text       = tile->can_use_lcd_text();
  playback_settings.msaa_sample_count  = msaa_sample_count;
  playback_settings.visible            = tile->required_for_activation() ||
                                         tile->required_for_draw();
  playback_settings.hdr_headroom       = target_color_params.GetHdrHeadroom();

  // D. 图像分类处理
  //    sync_decoded_images  → 需同步解码后才能开始光栅化
  //    checkered_images     → 先用棋盘格占位，后台解码完成后再触发重光栅化
  std::vector<DrawImage> sync_decoded_images;
  std::vector<PaintImage> checkered_images;
  PartitionImagesForCheckering(prioritized_tile, target_color_params,
                               &sync_decoded_images, &checkered_images, ...);

  // D.1 为同步图像创建解码依赖任务（RasterTask 完成前必须先完成解码）
  TileTask::Vector decode_tasks;
  bool has_at_raster_images = false;
  image_controller_.ConvertImagesToTasks(&sync_decoded_images, &decode_tasks,
                                         &has_at_raster_images, tracing_info);

  // D.2 at-raster 图像不允许在预绘制（prepaint）tile 上阻塞，直接跳过
  if (has_at_raster_images && tile->is_prepaint()) {
    // 把图像放入 extra_prepaint_images，稍后单独调度解码
    work_to_schedule->extra_prepaint_images.insert(...);
    OnRasterTaskCompleted(tile->id(), std::move(resource), /*was_canceled=*/true);
    return nullptr;
  }

  // D.3 收集需要跳过的棋盘格图像 id，将其加入 checker decode 队列
  PaintImageIdFlatSet images_to_skip;
  for (const auto& image : checkered_images) {
    images_to_skip.insert(image.stable_id());
    if (prioritized_tile.should_decode_checkered_images_for_tile())
      work_to_schedule->checker_image_decode_queue.emplace_back(
          image, CheckerImageTracker::DecodeType::kRaster);
  }

  // E. 获取 RasterBuffer（封装 GPU 共享内存 / GpuMemoryBuffer）
  std::unique_ptr<RasterBuffer> raster_buffer =
      raster_buffer_provider_->AcquireBufferForRaster(
          resource, resource_content_id, tile->invalidated_id());
  // → 返回 GpuRasterBufferProvider::RasterBufferImpl

  // F. 构造图像解码 Provider（GPU 模式下走 OOP-R 路径）
  PlaybackImageProvider image_provider(
      image_controller_.cache(), target_color_params,
      {.images_to_skip = std::move(images_to_skip),
       .raster_mode = use_gpu_rasterization_
                          ? PlaybackImageProvider::RasterMode::kGpu
                          : PlaybackImageProvider::RasterMode::kSoftware});
  DispatchingImageProvider dispatching_image_provider(
      std::move(image_provider),
      PaintWorkletImageProvider(prioritized_tile.GetPaintWorkletRecords()));

  // G. 创建 RasterTaskImpl 并返回
  return base::MakeRefCounted<RasterTaskImpl>(
      this, tile, std::move(resource),
      prioritized_tile.raster_source(),   // ← 含 DisplayItemList
      playback_settings,
      prioritized_tile.priority().resolution,
      invalidated_rect,
      prepare_tiles_count_,
      std::move(raster_buffer),           // ← GpuRasterBufferProvider::RasterBufferImpl
      &decode_tasks,                      // ← 图像解码依赖任务
      use_gpu_rasterization_,
      std::move(dispatching_image_provider),
      active_url_,
      prioritized_tile.GetRasterInducingScrollOffsets());
}
```

**关键设计点**：

| 步骤 | 说明 |
|------|------|
| 部分光栅化 | `TryAcquireResourceForPartialRaster` 复用旧资源，只重绘 `invalidated_rect` |
| 固色优化 | 纯色 tile 直接标记 `set_solid_color`，绕过整个光栅化流程 |
| Checker 图像 | 棋盘格图像不阻塞光栅化，decode 完成后再触发 re-raster |
| 解码依赖 | `decode_tasks` 与 `RasterTaskImpl` 形成 DAG，调度器保证先解码后光栅化 |
| RasterBuffer | 封装了 GpuMemoryBuffer，负责持有 GPU 内存直到光栅化完成 |

**对于红色 div**:
- Tile(0,0) 包含红色 div,在 viewport 内,优先级最高
- 为 Tile(0,0) 创建 `RasterTaskImpl`
- 调度到 raster worker thread 执行

## 2. RasterTask 创建和调度 - 完整调用链

从 TileManager 到实际执行的完整路径:

```cpp
// 步骤 1: TileManager::CreateRasterTask 创建任务
// cc/tiles/tile_manager.cc
scoped_refptr<TileTask> TileManager::CreateRasterTask(
    const PrioritizedTile& prioritized_tile,
    const TargetColorParams& target_color_params,
    PrioritizedWorkToSchedule* work_to_schedule) {

  Tile* tile = prioritized_tile.tile();

  // 1.1 获取 RasterBuffer (GPU 光栅化)
  //     调用 GpuRasterBufferProvider::AcquireBufferForRaster
  std::unique_ptr<RasterBuffer> raster_buffer =
      raster_buffer_provider_->AcquireBufferForRaster(
          resource, resource_content_id, tile->invalidated_id());

  // 返回: GpuRasterBufferProvider::RasterBufferImpl 实例

  // 1.2 创建 RasterTaskImpl,将 raster_buffer 传入
  return base::MakeRefCounted<RasterTaskImpl>(
      this, tile, std::move(resource), prioritized_tile.raster_source(),
      playback_settings, prioritized_tile.priority().resolution,
      invalidated_rect, prepare_tiles_count_,
      std::move(raster_buffer),  // ← 这里!GpuRasterBufferProvider::RasterBufferImpl
      &decode_tasks, use_gpu_rasterization_,
      std::move(dispatching_image_provider), active_url_,
      prioritized_tile.GetRasterInducingScrollOffsets());
}

// 步骤 2: TileManager::ScheduleTasks 将任务提交
void TileManager::ScheduleTasks(PrioritizedWorkToSchedule work_to_schedule) {
  // 构建任务图 (TaskGraph)
  // ... (前面已分析)

  // 提交任务图到 TaskGraphRunner
  tile_task_manager_->ScheduleTasks(&graph_);
  // → TaskGraphRunner 将任务放入就绪队列
  // → 通知工作线程
}

// 步骤 3: Worker Thread 取出并执行任务
// (在 cc/raster/categorized_worker_pool.cc 的工作线程中)
void WorkerThread::Run() {
  while (true) {
    // 3.1 从就绪队列中取出最高优先级任务
    PrioritizedTask task = work_queue_->GetNextTaskToRun(category);
    if (!task.task) break;

    // 3.2 执行任务的 RunOnWorkerThread 方法
    task.task->RunOnWorkerThread();  // ← 这里调用 RasterTaskImpl::RunOnWorkerThread

    // 3.3 标记完成
    task.task->state().DidFinish();
  }
}

// 步骤 4: RasterTaskImpl::RunOnWorkerThread 被调用
// cc/tiles/tile_manager.cc
class RasterTaskImpl : public TileTask {
 private:
  std::unique_ptr<RasterBuffer> raster_buffer_;  // GpuRasterBufferProvider::RasterBufferImpl
  scoped_refptr<RasterSource> raster_source_;    // 包含 DisplayItemList

 public:
  void RunOnWorkerThread() override {
    TRACE_EVENT1("cc", "RasterizerTaskImpl::RunOnWorkerThread",
                 "source_prepare_tiles_id", source_prepare_tiles_id_);

    DCHECK(raster_source_.get());
    DCHECK(raster_buffer_);

    // 4.1 调用 RasterBuffer 的 Playback 方法
    //     raster_buffer_ 是 GpuRasterBufferProvider::RasterBufferImpl
    raster_buffer_->Playback(raster_source_.get(), content_rect_,
                             invalid_content_rect_, new_content_id_,
                             raster_transform_, playback_settings_, url_);
    // ← 这里就到达了 GpuRasterBufferProvider::RasterBufferImpl::Playback!
  }
};
```

**完整调用栈**:

```
TileManager::PrepareTiles()
  └─> TileManager::AssignGpuMemoryToTiles()
        └─> TileManager::CreateRasterTask()
              ├─> raster_buffer_provider_->AcquireBufferForRaster()
              │     └─> 返回 GpuRasterBufferProvider::RasterBufferImpl
              └─> 创建 RasterTaskImpl(raster_buffer)
  └─> TileManager::ScheduleTasks()
        └─> tile_task_manager_->ScheduleTasks(&graph_)
              └─> TaskGraphRunner::ScheduleTasks()
                    └─> work_queue_.ScheduleTasks() - 构建就绪队列
                          └─> NotifyConcurrencyIncrease() - 唤醒工作线程

[切换到 Worker Thread]
WorkerThread::Run()
  └─> work_queue_->GetNextTaskToRun()
        └─> 返回 RasterTaskImpl
              └─> RasterTaskImpl::RunOnWorkerThread()
                    └─> raster_buffer_->Playback()  ← 这里!
                          │
                          └─> GpuRasterBufferProvider::RasterBufferImpl::Playback()
```

**关键点**:

1. **RasterBuffer 的多态性**: `raster_buffer_` 是 `RasterBuffer` 基类指针,实际类型是 `GpuRasterBufferProvider::RasterBufferImpl`
2. **延迟执行**: `CreateRasterTask` 只是准备任务,不执行;真正执行在 Worker Thread
3. **线程切换**: 从主线程(TileManager)切换到工作线程(WorkerThread)
4. **虚函数调用**: `raster_buffer_->Playback()` 通过虚函数表调用到具体实现

## 3. GpuRasterBufferProvider::RasterBufferImpl::Playback - GPU 光栅化入口

当 Worker Thread 调用 `raster_buffer_->Playback()` 时:

```cpp
// cc/raster/gpu_raster_buffer_provider.cc
void GpuRasterBufferProvider::RasterBufferImpl::Playback(
    const RasterSource* raster_source,
    const gfx::Rect& raster_full_rect,
    const gfx::Rect& raster_dirty_rect,
    uint64_t new_content_id,
    const gfx::AxisTransform2d& transform,
    const RasterSource::PlaybackSettings& playback_settings,
    const GURL& url) {
  TRACE_EVENT0("cc", "GpuRasterBuffer::Playback");
  // 委托给 PlaybackOnWorkerThread（含性能采样逻辑）
  PlaybackOnWorkerThread(raster_source, raster_full_rect, raster_dirty_rect,
                         new_content_id, transform, playback_settings, url);
  backing_->returned_sync_token = gpu::SyncToken();
}

// Playback → PlaybackOnWorkerThread → PlaybackOnWorkerThreadInternal
// 中间层负责可选的 GPU 时间戳查询（raster metric sampling）
void GpuRasterBufferProvider::RasterBufferImpl::PlaybackOnWorkerThreadInternal(
    ..., RasterQuery* query) {
  // 获取 worker thread 的 raster context lock
  viz::RasterContextProvider::ScopedRasterContextLock scoped_context(
      client_->worker_context_provider_, url.possibly_invalid_spec().c_str());
  gpu::raster::RasterInterface* ri =
      client_->worker_context_provider_->RasterInterface();

  // 计算实际需要光栅化的区域
  // （部分光栅化时取 full_rect 与 dirty_rect 的交集）
  gfx::Rect playback_rect = raster_full_rect;
  if (resource_has_previous_content_)
    playback_rect.Intersect(raster_dirty_rect);

  // 可选：插入 GPU 时间戳查询用于性能统计
  if (measure_raster_metric)
    ri->BeginQueryEXT(GL_COMMANDS_ISSUED_CHROMIUM, query->raster_duration_query_id);

  RasterizeSource(raster_source, raster_full_rect, playback_rect,
                  transform, playback_settings);

  if (measure_raster_metric)
    ri->EndQueryEXT(GL_COMMANDS_ISSUED_CHROMIUM);
}
```

## 4. RasterizeSource - 核心光栅化

```cpp
void GpuRasterBufferProvider::RasterBufferImpl::RasterizeSource(
    const RasterSource* raster_source,
    const gfx::Rect& raster_full_rect,
    const gfx::Rect& playback_rect,
    const gfx::AxisTransform2d& transform,
    const RasterSource::PlaybackSettings& playback_settings) {

  gpu::raster::RasterInterface* ri =
      client_->worker_context_provider_->RasterInterface();

  // 1. 创建或复用 SharedImage，并开始 raster access
  bool mailbox_needs_clear = false;
  std::unique_ptr<gpu::RasterScopedAccess> ri_access;
  if (!backing_->shared_image()) {
    // 首次：创建 SharedImage（GPU 纹理抽象）并开始独占写入
    gpu::SharedImageUsageSet flags =
        gpu::SHARED_IMAGE_USAGE_DISPLAY_READ |
        gpu::SHARED_IMAGE_USAGE_RASTER_WRITE;
    backing_->CreateSharedImage(sii, flags, "GpuRasterTile");
    mailbox_needs_clear = true;
    ri_access = backing_->shared_image()->BeginRasterAccess(
        ri, sii->GenUnverifiedSyncToken(), /*readonly=*/false);
  } else {
    // 复用：等待上次读取完成后开始写入
    ri_access = backing_->shared_image()->BeginRasterAccess(
        ri, backing_->returned_sync_token, /*readonly=*/false);
  }

  // 2. 告知 GPU：开始往 SharedImage 写入光栅数据
  ri->BeginRasterCHROMIUM(
      raster_source->background_color(), mailbox_needs_clear,
      playback_settings.msaa_sample_count, msaa_mode, use_lcd_text,
      playback_settings.visible, backing_->color_space(),
      playback_settings.hdr_headroom,
      backing_->shared_image()->mailbox().name);

  // 3. 序列化 DisplayItemList 的 PaintOps 并发送到 GPU 进程执行
  gfx::Vector2dF recording_to_raster_scale = transform.scale();
  recording_to_raster_scale.InvScale(raster_source->recording_scale_factor());
  gfx::Size content_size = raster_source->GetContentSize(transform.scale());

  ri->RasterCHROMIUM(
      raster_source->GetDisplayItemList().get(),
      playback_settings.image_provider,
      content_size,
      raster_full_rect,
      playback_rect,
      transform.translation(),
      recording_to_raster_scale,
      raster_source->requires_clear(),
      playback_settings.raster_inducing_scroll_offsets,
      const_cast<RasterSource*>(raster_source)->max_op_size_hint());

  // 4. 结束光栅化，提交同步 token 供后续读取方等待
  ri->EndRasterCHROMIUM();
  backing_->mailbox_sync_token =
      gpu::RasterScopedAccess::EndAccess(std::move(ri_access));
}
```

**对于红色 div**：
1. 首次光栅化：创建 256x256 的 SharedImage（GPU 纹理），`mailbox_needs_clear = true`
2. `BeginRasterAccess` → `BeginRasterCHROMIUM`：开始写入
3. `RasterCHROMIUM`：序列化 DisplayItemList 并发送到 GPU 进程，GPU 反序列化后执行 DrawRectOp
4. `EndRasterCHROMIUM` + `EndAccess`：完成并记录 sync token 供 Display Compositor 等待

## 5. RasterCHROMIUM - 序列化与跨进程传输

**重要架构说明**: `RasterCHROMIUM` 涉及跨进程通信（IPC），DisplayItemList **不会直接调用** `DisplayItemList::Raster()`。相反，这是一个序列化→传输→反序列化→执行的流程。

### 5.1 客户端 (Compositor 进程)

```cpp
// gpu/command_buffer/client/raster_implementation.cc
void RasterImplementation::RasterCHROMIUM(
    const cc::DisplayItemList* list,
    cc::ImageProvider* image_provider,
    const gfx::Size& content_size,
    const gfx::Rect& full_raster_rect,
    const gfx::Rect& playback_rect,
    const gfx::Vector2dF& post_translate,
    const gfx::Vector2dF& post_scale,
    bool requires_clear,
    const ScrollOffsetMap* raster_inducing_scroll_offsets,
    size_t* max_op_size_hint) {

  // 1. 使用 R-Tree 查找需要绘制的 PaintOps
  //    先将 playback_rect 反变换到 recording space 再查询
  gfx::Rect query_rect = gfx::ScaleToEnclosingRect(
      playback_rect, 1.f / post_scale.x(), 1.f / post_scale.y());
  list->SearchOpsByRect(query_rect, &temp_raster_offsets_);

  // 2. 构建 preamble（前导转换：清屏、缩放、裁剪等）
  cc::PaintOpBufferSerializer::Preamble preamble;
  preamble.content_size = content_size;
  preamble.full_raster_rect = full_raster_rect;
  preamble.playback_rect = playback_rect;
  preamble.post_translation = post_translate;
  preamble.post_scale = post_scale;
  preamble.requires_clear = requires_clear;
  preamble.background_color = raster_properties_->background_color;

  // 3. 创建序列化辅助对象
  //    TransferCacheSerializeHelperImpl: 处理图片/纹理缓存的跨进程传输
  //    PaintOpSerializer: 管理向 transfer buffer 写数据，满后自动 flush
  TransferCacheSerializeHelperImpl transfer_cache_serialize_helper(this);
  PaintOpSerializer op_serializer(free_size, this, &stashing_image_provider,
                                  &transfer_cache_serialize_helper,
                                  &font_manager_, max_op_size_hint);

  // 4. 序列化 PaintOpBuffer
  //    PaintOpBufferSerializer 遍历 temp_raster_offsets_ 指定的 ops，
  //    依次调用 cc::PaintOp::Serialize 写入 transfer buffer（共享内存）
  cc::PaintOpBufferSerializer serializer(
      PaintOpSerializer::Serialize, &op_serializer,
      cc::PaintOp::SerializeOptions(
          &stashing_image_provider, &transfer_cache_serialize_helper, ...));
  serializer.Serialize(list->paint_op_buffer(), &temp_raster_offsets_,
                       preamble);

  // 5. flush 剩余数据，触发 UnmapRasterCHROMIUM，让 GPU 进程执行
  op_serializer.SendSerializedData();
}
```

**对于红色 div**:
1. `SearchOpsByRect` 将 `playback_rect` 反缩放到 recording space 后，在 R-Tree 中找到 DrawRecordOp
2. `PaintOpBufferSerializer::Serialize` 将 DrawRecordOp（及其嵌套的 DrawRectOp）序列化写入 transfer buffer
3. 数据布局示例：`[preamble: SaveLayer+Scale+Translate][DrawRecordOp头][DrawRectOp头][rect][flags]...`
4. `op_serializer.SendSerializedData()` 触发 `UnmapRasterCHROMIUM`，命令发往 GPU 进程

### 5.2 服务端 (GPU 进程)

```cpp
// gpu/command_buffer/service/raster_decoder.cc
error::Error RasterDecoderImpl::DoRasterCHROMIUM(
    GLuint raster_shm_id,
    GLuint raster_shm_offset,
    GLuint raster_shm_size,
    GLuint font_shm_id,    // 字体数据（Skia 文字光栅化）
    GLuint font_shm_offset,
    GLuint font_shm_size) {

  // 1. 从共享内存映射 PaintOp 数据
  char* paint_buffer_memory =
      GetSharedMemoryAs<char*>(raster_shm_id, raster_shm_offset, raster_shm_size);
  size_t paint_buffer_size = raster_shm_size;

  // 2. 构建反序列化选项（transfer cache、paint cache、strike client 等）
  TransferCacheDeserializeHelperImpl impl(raster_decoder_id_, transfer_cache());
  cc::PaintOp::DeserializeOptions options{
      .transfer_cache = &impl,
      .paint_cache = paint_cache_.get(),
      .strike_client = font_manager_->strike_client(),
      // scratch_buffer 来自 shared_context_state_，跨调用复用
      .scratch_buffer = *shared_context_state_->scratch_deserialization_buffer(),
      .crash_dump_on_failure = !gpu_preferences_.disable_oopr_debug_crash_dump,
      ...};

  // 3. 在栈上分配 PaintOp 临时缓冲区（对齐到 kPaintOpAlign）
  //    每个 op 反序列化后就地执行，无需整体重建
  alignas(cc::PaintOpBuffer::kPaintOpAlign) char
      data[cc::kLargestPaintOpAlignedSize];

  // 4. Raw Draw 特殊路径：将 ops 收集到 PaintOpBuffer，由 GPU 延迟批量执行
  if (scoped_shared_image_raster_write_) {
    auto* paint_op_buffer = scoped_shared_image_raster_write_->paint_op_buffer();
    paint_op_buffer->Deserialize(paint_buffer_memory, raster_shm_size, options);
    return error::kNoError;
  }

  // 5. 普通路径：先反序列化字体（若有）
  if (font_shm_size > 0) {
    volatile uint8_t* font_buffer_memory =
        GetSharedMemoryAs<uint8_t*>(font_shm_id, font_shm_offset, font_shm_size);
    font_manager_->Deserialize(
        base::span(font_buffer_memory, font_shm_size), &new_locked_handles);
  }

  // 6. 流式反序列化 + 立即执行每个 PaintOp
  while (paint_buffer_size > 0) {
    // 6.1 将下一个 op 反序列化到栈上 data 缓冲区
    //     skip = 该 op 在共享内存中占用的对齐后字节数
    size_t skip = 0;
    cc::PaintOp* deserialized_op =
        cc::PaintOp::Deserialize(paint_buffer_memory, paint_buffer_size,
                                 data, std::size(data), &skip, options);
    if (!deserialized_op) {
      // 通过 GL 错误机制上报，不用 kOutOfBounds
      LOCAL_SET_GL_ERROR(GL_INVALID_OPERATION, "glRasterCHROMIUM",
                         "RasterCHROMIUM: serialization failure");
      return error::kNoError;
    }

    // 6.2 执行（多态分发到 DrawRectOp::Raster、DrawRecordOp::Raster 等）
    deserialized_op->Raster(raster_canvas_, playback_params);

    // 6.3 释放 placement new 在 data 上构造的 op
    deserialized_op->DestroyThis();

    // 6.4 前进：注意使用 skip 而非 op 的 size 成员
    paint_buffer_size -= skip;
    paint_buffer_memory += skip;
  }

  return error::kNoError;
}
```

**对于红色 div**：
1. 第一次循环：
   - 反序列化 DrawRecordOp 到 `data` 缓冲区
   - 调用 `DrawRecordOp::Raster(raster_canvas_, playback_params)`
2. DrawRecordOp::Raster 内部直接遍历其 `record`（PaintOpBuffer）中的 ops 并调用各自的 `Raster`
3. 第二次（嵌套）：
   - 对 DrawRectOp 调用 `DrawRectOp::Raster`
   - 最终调用 `canvas->drawRect({0,0,100,100}, red_paint)`

### 5.3 为什么不直接调用 DisplayItemList::Raster?

**架构原因**:
- **进程隔离**: Compositor 进程和 GPU 进程是独立的
- **安全性**: GPU 进程运行在沙箱中,不能直接访问 Compositor 进程的内存
- **效率**: 共享内存 IPC 速度快,但需要序列化
- **流式处理**: 可以边反序列化边执行,不需要完整重建 DisplayItemList

**DisplayItemList::Raster 的实际用途**:
- 软件光栅化路径 (Software Rasterization)
- 单进程模式 (如某些测试环境)
- 其他不跨进程的场景

在 GPU 光栅化 (OOP-R, Out-of-Process Rasterization) 中,使用的是序列化机制。

每个 PaintOp 子类都实现了自己的 Raster 方法:

## 6. PaintOp::Raster 逐类分析

### 6.1 DrawRecordOp::Raster - 递归绘制

DrawRecordOp 包含一个嵌套的 PaintRecord (本质上是另一个 PaintOpBuffer):

```cpp
// cc/paint/paint_op.cc
// 注意：静态函数签名（通过函数指针表分派，不是虚函数）
void DrawRecordOp::Raster(const DrawRecordOp* op,
                          SkCanvas* canvas,
                          const PlaybackParams& params) {
  // 不用 drawPicture（它会加隐式 clip），直接 Playback
  // op->record 在反序列化时已完整还原，这里直接遍历其中的 ops 执行
  op->record.Playback(canvas, params, op->local_ctm);
}
```

**对于红色 div**：
- `op->record` 在 `PaintOp::Deserialize` 时已被完整反序列化（record 内部的 ops 也一并还原）
- `record.Playback` 遍历内部的 ops，找到 DrawRectOp 并调用其 `Raster`

### 6.2 DrawRectOp::RasterWithFlags - 最终绘制到 Canvas

```cpp
// cc/paint/paint_op.cc
// 带 flags 的 op 使用 RasterWithFlags 静态方法
void DrawRectOp::RasterWithFlags(const DrawRectOp* op,
                                 const PaintFlags* flags,
                                 SkCanvas* canvas,
                                 const PlaybackParams& params) {
  // PaintFlags::DrawToSk 将 PaintFlags 转换为 SkPaint，然后回调绘制
  flags->DrawToSk(canvas, [op](SkCanvas* c, const SkPaint& p) {
    c->drawRect(op->rect, p);  // 最终调用 Skia drawRect
  });
}
```

**这里就是魔法发生的地方！**
- `canvas->drawRect` 是 Skia 提供的 API
- 如果是 GPU 光栅化,这个 canvas 对应一个 GPU surface (GrRecordingContext)
- `drawRect` 会生成 GPU 绘制操作,最终转换为 Vulkan/OpenGL 命令

### 6.3 Skia → GPU

Skia 将绘制操作转换为 GPU 命令:

```
DrawRectOp::Raster
  ↓
SkCanvas::drawRect
  ↓
GrRenderTargetContext::drawRect  (Skia GPU 后端)
  ↓
GrOpsTask::addDrawOp  (添加 GPU 绘制操作)
  ↓
GrOp::execute  (执行时)
  ↓
生成 Vulkan / OpenGL 命令:
  - vkCmdDraw (Vulkan)
  - 或 glDrawArrays (OpenGL)
  ↓
GPU 执行
  ↓
Fragment Shader 为每个像素计算颜色
  - 对于红色矩形: 输出 RGB(255, 0, 0)
  ↓
写入 SharedImage (GPU 纹理)
```

**对于红色 div**:
1. DrawRectOp 转换为 GPU 绘制命令
2. GPU Fragment Shader 执行,为矩形内的每个像素输出红色
3. 红色像素写入 Tile(0,0) 的 SharedImage 纹理

# 光栅化完成后的完整流程 - 端到端调用链

GPU 进程光栅化完成后，需要经过一系列回调、状态更新和调度，最终触发新的合成帧：

## 完整调用链概览

```
GPU 进程光栅化完成
  ↓
[1] GPU Service 命令完成回调
  ↓
[2] Compositor 进程 RasterTask 完成回调
  ↓
[3] TileManager::OnRasterTaskCompleted (更新 Tile 状态)
  ↓
[4] LayerTreeHostImpl::NotifyTileStateChanged (标记 damage)
  ↓
[5] LayerImpl::NotifyTileStateChanged (记录需要重绘的区域)
  ↓
[6] SetNeedsRedraw (触发重绘请求)
  ↓
[7] Scheduler 调度下一个 BeginFrame
  ↓
[8] WillBeginImplFrame (开始新帧)
  ↓
[9] PrepareToDraw (准备绘制数据)
  ↓
[10] CalculateRenderPasses → AppendQuads (构建 DrawQuads)
  ↓
[11] DrawLayers → GenerateCompositorFrame (生成合成帧)
  ↓
[12] SubmitCompositorFrame (提交到 Viz)
  ↓
[13] Viz 聚合和绘制
  ↓
[14] SwapBuffers (显示到屏幕)
```

## 阶段 1：GPU 进程完成光栅化并发送回调

```cpp
// gpu/command_buffer/service/raster_decoder.cc
error::Error RasterDecoderImpl::DoRasterCHROMIUM(...) {
  // ... 反序列化并执行所有 PaintOps ...
  // ... canvas->drawRect() 调用 Skia GPU 后端 ...
  // ... GPU Fragment Shader 执行，写入 SharedImage ...

  // 光栅化完成，SharedImage 纹理已经包含渲染结果
  return error::kNoError;  // 返回成功
}

// GPU Service 执行完命令后
// gpu/command_buffer/service/command_buffer_service.cc
void CommandBufferService::SetCommandExecuteCallback(...) {
  // 命令执行完成，通过 IPC 回调通知客户端
  execute_callback_.Run();
}
```

**对于红色 div**:
- SharedImage (Tile 0,0 的 256x256 GPU 纹理) 现在包含红色矩形的像素数据
- GPU 命令队列标记为完成，发送 IPC 通知

## 阶段 2-3：Compositor 进程收到回调并更新 Tile 状态

```cpp
// cc/raster/gpu_raster_buffer_provider.cc
// (在 Raster Worker Thread 上)
void RasterTaskImpl::RunOnWorkerThread() {
  // ... 调用 raster_buffer_->Playback() ...
  // Playback 返回后，任务完成
}

// (任务完成时由 TaskGraphRunner 在 Compositor Thread 上调用)
void RasterTaskImpl::OnTaskCompleted() {
  raster_buffer_ = nullptr;  // 必须先释放 RasterBuffer
  // 直接调用 TileManager，无 callback 间接层
  tile_manager_->OnRasterTaskCompleted(tile_id_, std::move(resource_),
                                       state().IsCanceled());
}

// cc/tiles/tile_manager.cc
// (切换到 Compositor Thread)
void TileManager::OnRasterTaskCompleted(
    Tile::Id tile_id,
    ResourcePool::InUsePoolResource resource,
    bool was_canceled) {

  auto found = tiles_.find(tile_id);
  Tile* tile = found != tiles_.end() ? found->second : nullptr;

  // 1. 清理任务引用 & 释放图片引用
  if (tile) tile->raster_task_ = nullptr;
  image_controller_.UnrefImages(scheduled_draw_images_[tile_id]);
  scheduled_draw_images_.erase(tile_id);

  if (was_canceled) {
    resource_pool_->ReleaseResource(std::move(resource));
    return;
  }

  // 2. 标记内容已更新（供 ResourcePool 的缓存失效机制使用）
  resource_pool_->OnContentReplaced(resource, tile_id);

  if (!tile) {
    resource_pool_->ReleaseResource(std::move(resource));
    return;
  }

  // 3. 通知 RasterBufferProvider GPU 工作已提交（用于 flush 时序）
  raster_buffer_provider_->NotifyWorkSubmitted();

  // 4. 给资源分配 ResourceId，准备导出到 Display Compositor
  bool exported = resource_pool_->PrepareForExport(
      resource, viz::TransferableResource::ResourceSource::kTileRasterTask);

  // 5. 检查 GPU 工作是否已完成（仅 SMOOTHNESS_TAKES_PRIORITY 模式下等待）
  bool is_ready_for_draw = true;
  if (global_state_.tree_priority == SMOOTHNESS_TAKES_PRIORITY)
    is_ready_for_draw = raster_buffer_provider_->IsResourceReadyToDraw(resource);

  // 6. 更新 Tile 的 DrawInfo
  TileDrawInfo& draw_info = tile->draw_info();
  if (exported)
    draw_info.SetResource(std::move(resource),
                          raster_task_was_scheduled_with_checker_images);
  else
    draw_info.set_oom();

  // 7. 就绪则立即触发重绘；否则加入 pending 队列等待 GPU 完成
  if (is_ready_for_draw) {
    draw_info.set_resource_ready_for_draw();
    client_->NotifyTileStateChanged(tile, /*update_damage=*/true,
                                    /*set_needs_redraw=*/true);
  } else {
    // GPU 尚未完成（异步模式），等待 sync token 后再标记就绪
    pending_gpu_work_tiles_.insert(tile);
  }
}
```

**对于红色 div**:
- Tile(0,0) 的 DrawInfo 更新：
  - `mode = RESOURCE_MODE`
  - `resource_id = 12345` (SharedImage 的 ID)
  - `resource_ready_for_draw = true`
- 触发 `NotifyTileStateChanged`

## 阶段 4-5：标记 Damage 并触发重绘请求

```cpp
// cc/trees/layer_tree_host_impl.cc
// (Compositor Thread)
void LayerTreeHostImpl::NotifyTileStateChanged(
    const Tile* tile,
    bool update_damage,
    bool set_needs_redraw) {

  TRACE_EVENT0("cc", "LayerTreeHostImpl::NotifyTileStateChanged");

  // 1. 找到 tile 所属的 LayerImpl
  LayerImpl* layer_impl = nullptr;
  const bool is_pending_tree =
      tile->tiling()->tree() == WhichTree::PENDING_TREE;

  if (is_pending_tree) {
    layer_impl = pending_tree_->FindPendingTreeLayerById(tile->layer_id());
  } else {
    layer_impl = active_tree_->FindActiveTreeLayerById(tile->layer_id());
  }

  // 2. 通知 Layer 标记 damage 区域
  layer_impl->NotifyTileStateChanged(tile, update_damage);

  // 3. 如果是必需的 tile 完成了，触发重绘
  if (set_needs_redraw &&
      !client_->IsInsideDraw() &&
      tile->required_for_draw()) {
    // ← 关键！请求重绘
    SetNeedsRedraw(/*animation_only=*/false,
                   /*skip_if_inside_draw=*/false);
  }
}

// cc/layers/picture_layer_impl.cc
void PictureLayerImpl::NotifyTileStateChanged(
    const Tile* tile,
    bool update_damage) {

  if (update_damage) {
    // 标记 tile 所在区域为 damaged
    gfx::Rect tile_damage_rect = tile->content_rect();
    SetNeedsPushProperties();

    // 添加到渲染表面的 damage tracker
    if (layer_tree_impl()->RootRenderSurface()) {
      layer_tree_impl()->RootRenderSurface()
          ->damage_tracker()->AddDamageNextUpdate(tile_damage_rect);
    }
  }
}

// cc/trees/layer_tree_host_impl.cc
void LayerTreeHostImpl::SetNeedsRedraw(
    bool animation_only,
    bool skip_if_inside_draw) {

  if (skip_if_inside_draw && client_->IsInsideDraw())
    return;

  // ← 关键！通知 Scheduler 需要重绘
  client_->SetNeedsRedrawOnImplThread();
}
```

**对于红色 div**:
- Tile(0,0) 的矩形区域 `{0,0,256,256}` 被标记为 damaged
- 重绘请求发送给 Scheduler

### 阶段 6-7：Scheduler 调度新的 BeginFrame

```cpp
// cc/scheduler/scheduler.cc
void Scheduler::SetNeedsRedraw() {
  state_machine_.SetNeedsRedraw();
  ProcessScheduledActions();
}

void Scheduler::ProcessScheduledActions() {
  // 根据状态机决定下一步动作
  SchedulerStateMachine::Action action =
      state_machine_.NextAction();

  if (action == SchedulerStateMachine::Action::SEND_BEGIN_MAIN_FRAME) {
    // 如果主线程有更新，先发送 BeginMainFrame
    SendBeginMainFrame();
  }

  // 等待 VSync 信号触发 BeginImplFrame
  // (由 viz::BeginFrameSource 提供)
}

// 当 VSync 信号到达时
void Scheduler::OnBeginFrameFromBeginFrameSource(
    const viz::BeginFrameArgs& args) {

  // ← 关键！开始新的 Impl 帧
  BeginImplFrame(...);
}

void Scheduler::BeginImplFrame(...) {
  // 通知 LayerTreeHostImpl
  client_->WillBeginImplFrame(args);
}
```

## 阶段 8-9：开始新帧并准备绘制数据

```cpp
// cc/trees/layer_tree_host_impl.cc
bool LayerTreeHostImpl::WillBeginImplFrame(
    const viz::BeginFrameArgs& args) {

  TRACE_EVENT1("cc", "LayerTreeHostImpl::WillBeginImplFrame",
               "frame_time", args.frame_time);

  // 1. 更新动画
  mutator_host_->TickAnimations(args.frame_time);

  // 2. 更新图片动画
  image_animation_controller_.WillBeginImplFrame(args);

  // 3. 其他准备工作...

  return true;
}

// 随后 Scheduler 调用 DrawIfPossible
void Scheduler::DrawIfPossible() {
  client_->ScheduledActionDrawIfPossible();
}

// cc/trees/layer_tree_host_impl.cc
DrawResult LayerTreeHostImpl::PrepareToDraw(FrameData* frame) {
  TRACE_EVENT1("cc", "LayerTreeHostImpl::PrepareToDraw",
               "SourceFrameNumber",
               active_tree_->source_frame_number());

  // 1. Tick 动画工作线程
  mutator_host_->TickWorkletAnimations();

  // 2. 更新绘制属性（重要：计算可见区域、变换等）
  bool ok = active_tree_->UpdateDrawProperties(
      /*update_tiles=*/true,
      /*update_image_animation_controller=*/true);
  DCHECK(ok);

  // 3. 准备 tiles（通知所有完成的 tiles 添加 damage）
  if (!settings_.trees_in_viz_in_viz_process) {
    tile_manager_.PrepareToDraw();
  }

  // 4. 计算需要绘制的 RenderPasses
  //    这里会遍历所有 layer，调用 AppendQuads
  frame->render_surface_list = &active_tree_->GetRenderSurfaceList();
  frame->render_passes.clear();
  frame->will_draw_layers.clear();

  // 添加 viewport damage
  if (active_tree_->RootRenderSurface()) {
    active_tree_->RootRenderSurface()->damage_tracker()
        ->AddDamageNextUpdate(viewport_damage_rect_);
  }

  // ← 关键！计算所有 RenderPasses 和 DrawQuads
  DrawResult draw_result = CalculateRenderPasses(frame);

  return draw_result;
}
```

## 阶段 10：构建 DrawQuads - CalculateRenderPasses 和 AppendQuads

```cpp
// cc/trees/layer_tree_host_impl.cc
DrawResult LayerTreeHostImpl::CalculateRenderPasses(FrameData* frame) {
  TRACE_EVENT0("cc", "LayerTreeHostImpl::CalculateRenderPasses");

  // 1. 遍历所有 RenderSurface (从后到前)
  for (auto* render_surface : *frame->render_surface_list) {
    // 2. 为每个 RenderSurface 创建 CompositorRenderPass
    auto render_pass = viz::CompositorRenderPass::Create();
    render_pass->SetNew(
        render_surface->render_pass_id(),
        render_surface->content_rect(),
        render_surface->damage_tracker()->GetDamageRectIfValid(),
        render_surface->screen_space_transform());

    // 3. 收集这个 surface 上所有 layer 的 quads
    //    遍历贡献到该 surface 的所有 layers
    for (LayerImpl* layer : render_surface->layer_list()) {
      // ← 关键！每个 layer 添加自己的 DrawQuads
      layer->AppendQuads(render_pass.get(), &frame->append_quads_data);
    }

    frame->render_passes.push_back(std::move(render_pass));
  }

  return DrawResult::kSuccess;
}

// cc/layers/picture_layer_impl.cc
void PictureLayerImpl::AppendQuads(
    viz::CompositorRenderPass* render_pass,
    AppendQuadsData* append_quads_data) {

  // 1. 创建 SharedQuadState（变换、裁剪等共享状态）
  viz::SharedQuadState* shared_quad_state =
      render_pass->CreateAndAppendSharedQuadState();

  PopulateSharedQuadState(shared_quad_state, ...);

  // 2. 使用 Cover 迭代器遍历可见区域的 tiles
  //    Cover() 返回智能迭代器，只访问与 visible_rect 相交的 tiles
  for (auto iter = Cover(shared_quad_state->visible_quad_layer_rect,
                         max_contents_scale,
                         GetIdealContentsScaleKey());
       iter; ++iter) {

    // 3. 检查 tile 是否就绪
    if (!*iter || !iter->draw_info().IsReadyToDraw()) {
      // Tile 不存在或未就绪 → 生成 checkerboard quad
      auto* quad = render_pass->CreateAndAppendDrawQuad<
          viz::SolidColorDrawQuad>();
      quad->SetNew(shared_quad_state,
                   geometry_rect,
                   visible_geometry_rect,
                   DebugColors::DefaultCheckerboardColor(),
                   false);
      continue;
    }

    gfx::Rect geometry_rect = iter.geometry_rect();
    // ... 处理遮挡和裁剪 ...

    const TileDrawInfo& draw_info = iter->draw_info();

    // 4. 根据 DrawInfo mode 创建对应的 DrawQuad
    switch (draw_info.mode()) {
      case TileDrawInfo::RESOURCE_MODE: {
        // ← 对于红色 div，走这里！
        // 创建 TileDrawQuad，引用 GPU 纹理资源
        auto* quad = render_pass->CreateAndAppendDrawQuad<
            viz::TileDrawQuad>();
        quad->SetNew(
            shared_quad_state,                   // 共享状态
            geometry_rect,                       // Tile 在屏幕上的位置
            visible_geometry_rect,               // 可见区域
            needs_blending,                      // 是否需要混合
            draw_info.resource_id_for_export(),  // ← SharedImage 资源 ID
            iter.texture_rect(),                 // 纹理坐标
            nearest_neighbor_,                   // 纹理过滤
            !settings().enable_edge_anti_aliasing);
        break;
      }

      case TileDrawInfo::SOLID_COLOR_MODE: {
        // 纯色 tile（优化）
        auto* quad = render_pass->CreateAndAppendDrawQuad<
            viz::SolidColorDrawQuad>();
        quad->SetNew(shared_quad_state,
                     geometry_rect,
                     visible_geometry_rect,
                     draw_info.solid_color(),
                     false);
        break;
      }

      case TileDrawInfo::OOM_MODE:
        // 内存不足，生成 checkerboard
        break;
    }
  }
}
```

**对于红色 div**:
- Cover 迭代器找到 Tile(0,0)
- `iter->draw_info().IsReadyToDraw()` 返回 `true` ✓
- 生成 **TileDrawQuad**:
  ```
  TileDrawQuad {
    rect: {0, 0, 256, 256}           // 在页面中的位置
    visible_rect: {0, 0, 256, 256}   // 完全可见
    resource_id: 12345               // SharedImage 资源 ID
    texture_rect: {0, 0, 1, 1}       // 归一化纹理坐标
    needs_blending: false            // 不透明
  }
  ```

## 阶段 11：生成 CompositorFrame

```cpp
// cc/trees/layer_tree_host_impl.cc
std::optional<SubmitInfo> LayerTreeHostImpl::DrawLayers(FrameData* frame) {
  DCHECK(CanDraw());
  DCHECK_EQ(frame->has_no_damage, frame->render_passes.empty());

  if (frame->has_no_damage) {
    // 没有 damage，跳过绘制
    return std::nullopt;
  }

  // ← 关键！生成 CompositorFrame
  auto compositor_frame = GenerateCompositorFrame(frame);

  // 提交到 FrameSink
  const auto frame_token = compositor_frame.metadata.frame_token;
  frame->frame_token = frame_token;

  // ... 处理 top controls, scroll elasticity 等 ...

  // ← 关键！提交到 Viz
  SubmitInfo submit_info;
  layer_tree_frame_sink_->SubmitCompositorFrame(
      std::move(compositor_frame),
      std::move(hit_test_region_list));

  return submit_info;
}

viz::CompositorFrame LayerTreeHostImpl::GenerateCompositorFrame(
    FrameData* frame) {

  TRACE_EVENT0("cc", "LayerTreeHostImpl::GenerateCompositorFrame");

  // 1. 创建 CompositorFrame
  viz::CompositorFrame compositor_frame;

  // 2. 设置元数据（设备像素比、页面缩放、滚动偏移等）
  compositor_frame.metadata = MakeCompositorFrameMetadata();
  compositor_frame.metadata.frame_token = next_frame_token_++;
  compositor_frame.metadata.begin_frame_ack =
      viz::BeginFrameAck::CreateManualAckWithDamage();

  // 3. 构建资源列表
  //    将 frame 中所有 tiles 使用的 SharedImage 转为 TransferableResource
  std::vector<viz::TransferableResource> resource_list;

  for (const auto& render_pass : frame->render_passes) {
    for (auto* quad : render_pass->quad_list) {
      if (quad->material == viz::DrawQuad::Material::kTileDrawQuad) {
        const auto* tile_quad =
            static_cast<const viz::TileDrawQuad*>(quad);

        // 从 ResourceProvider 获取资源并导出
        viz::TransferableResource resource;
        resource_provider_->PrepareSendToParent(
            {tile_quad->resource_id()},
            &resource_list);
      }
    }
  }

  compositor_frame.resource_list = std::move(resource_list);

  // 4. 移动 RenderPasses 到 frame
  compositor_frame.render_pass_list = std::move(frame->render_passes);

  return compositor_frame;
}
```

**对于红色 div，CompositorFrame 包含**:

```cpp
viz::CompositorFrame {
  metadata: {
    frame_token: 42,
    device_scale_factor: 1.0,
    page_scale_factor: 1.0,
    root_scroll_offset: {0, 0},
    // ...
  },

  resource_list: [
    viz::TransferableResource {
      id: 12345,                          // 资源 ID
      size: {256, 256},                   // 纹理大小
      format: RGBA_8888,                  // 像素格式
      mailbox: gpu::Mailbox(...),         // SharedImage 的 mailbox
      sync_token: gpu::SyncToken(...),    // GPU 同步令牌
      is_software: false,                 // GPU 资源
      // ...
    }
  ],

  render_pass_list: [
    viz::CompositorRenderPass {
      id: viz::CompositorRenderPassId(1),
      output_rect: {0, 0, 1920, 1080},    // 输出区域
      damage_rect: {0, 0, 256, 256},      // 损坏区域
      quad_list: [
        viz::TileDrawQuad {                 // ← 红色 div 的 quad
          rect: {0, 0, 256, 256},
          visible_rect: {0, 0, 256, 256},
          resource_id: 12345,               // 引用上面的资源
          // ...
        },
        // ... 其他 quads ...
      ],
      // ...
    }
  ]
}
```

## 阶段 12：提交到 Viz 进程

```cpp
// components/viz/service/display_embedder/compositor_frame_sink_impl.cc
void CompositorFrameSinkImpl::SubmitCompositorFrame(
    const viz::LocalSurfaceId& local_surface_id,
    viz::CompositorFrame frame,
    std::optional<viz::HitTestRegionList> hit_test_region_list) {

  TRACE_EVENT0("viz", "CompositorFrameSinkImpl::SubmitCompositorFrame");

  // 1. 验证 frame
  if (!frame.metadata.begin_frame_ack.has_damage) {
    // 没有变化，跳过
    return;
  }

  // 2. 提交到 CompositorFrameSinkSupport
  //    Support 管理 Surface 的生命周期
  support_->SubmitCompositorFrame(local_surface_id, std::move(frame));
}

// components/viz/service/frame_sinks/compositor_frame_sink_support.cc
void CompositorFrameSinkSupport::SubmitCompositorFrame(
    const LocalSurfaceId& local_surface_id,
    CompositorFrame frame) {

  // 1. 创建或更新 Surface
  Surface* surface = surface_manager_->GetSurfaceForId(
      SurfaceId(frame_sink_id_, local_surface_id));

  if (!surface) {
    surface = surface_manager_->CreateSurface(
        weak_factory_.GetWeakPtr(),
        SurfaceInfo(SurfaceId(frame_sink_id_, local_surface_id), ...));
  }

  // 2. 激活新的 CompositorFrame
  surface->QueueFrame(std::move(frame), ...);
  surface->ActivatePendingFrameForDeadline();

  // 3. 通知 Display 有新的 Surface 可以合成
  if (display_) {
    display_->SetNeedsRedraw();
  }
}
```

## 阶段 13-14：Viz 聚合、绘制和交换缓冲区

```cpp
// components/viz/service/display/display.cc
void Display::DrawAndSwap() {
  TRACE_EVENT0("viz", "Display::DrawAndSwap");

  // 1. 聚合所有 Surfaces
  //    处理嵌入的 iframes、videos 等子 surfaces
  auto render_pass_list = aggregator_->Aggregate(
      current_surface_id_,
      current_frame_time_);

  // 2. 准备绘制
  renderer_->BeginDrawingFrame();

  // 3. 遍历所有 RenderPass 并绘制
  for (auto& render_pass : render_pass_list) {
    renderer_->DrawRenderPassAndExecuteCopyRequests(
        render_pass.get());
  }

  // 4. 完成绘制
  renderer_->FinishDrawingFrame();

  // 5. 交换缓冲区，显示到屏幕
  SwapBuffers(std::move(render_pass_list));
}

// components/viz/service/display/surface_aggregator.cc
viz::CompositorRenderPassList SurfaceAggregator::Aggregate(
    const viz::SurfaceId& surface_id,
    base::TimeTicks expected_display_time) {

  // 1. 从 SurfaceManager 获取 Surface
  Surface* surface = manager_->GetSurfaceForId(surface_id);
  const CompositorFrame& frame = surface->GetActiveFrame();

  // 2. 聚合所有 RenderPass
  CompositorRenderPassList aggregated_passes;

  for (const auto& pass : frame.render_pass_list()) {
    // 2.1 创建聚合后的 RenderPass
    auto aggregated_pass = viz::CompositorRenderPass::Create();
    aggregated_pass->CopyFrom(*pass);

    // 2.2 处理嵌入的子 Surfaces (如 iframes)
    for (auto* quad : pass->quad_list) {
      if (quad->material == DrawQuad::Material::kSurfaceContent) {
        // 递归聚合子 surface
        const auto* surface_quad =
            static_cast<const SurfaceDrawQuad*>(quad);
        HandleSurfaceQuad(surface_quad, ...);
      } else {
        // 普通 quad 直接添加
        aggregated_pass->CopyQuadsFromAndAppendTo(
            pass, ..., quad);
      }
    }

    aggregated_passes.push_back(std::move(aggregated_pass));
  }

  return aggregated_passes;
}

// components/viz/service/display/gl_renderer.cc
void GLRenderer::DrawRenderPassAndExecuteCopyRequests(
    const CompositorRenderPass* render_pass) {

  // 1. 绑定目标 framebuffer
  //    对于根 RenderPass，绑定到 root framebuffer (显示缓冲区)
  BindFramebufferToOutputSurface(render_pass);

  // 2. 清空背景（如果需要）
  if (render_pass->should_clear_output_surface) {
    ClearFramebuffer();
  }

  // 3. 遍历 quad_list，绘制每个 DrawQuad
  for (auto it = render_pass->quad_list.BackToFrontBegin();
       it != render_pass->quad_list.BackToFrontEnd();
       ++it) {
    const DrawQuad* quad = *it;

    // 根据 quad 类型调度到具体绘制函数
    switch (quad->material) {
      case DrawQuad::Material::kTileDrawQuad:
        // ← 红色 div 走这里！
        DrawTileQuad(TileDrawQuad::MaterialCast(quad), ...);
        break;

      case DrawQuad::Material::kSolidColor:
        DrawSolidColorQuad(SolidColorDrawQuad::MaterialCast(quad), ...);
        break;

      case DrawQuad::Material::kCompositorRenderPass:
        DrawRenderPassQuad(RenderPassDrawQuad::MaterialCast(quad), ...);
        break;

      // ... 其他类型 ...
    }
  }
}

void GLRenderer::DrawTileQuad(
    const TileDrawQuad* quad,
    const gfx::QuadF* clip_region) {

  TRACE_EVENT0("viz", "GLRenderer::DrawTileQuad");

  // 1. 获取 GPU 纹理
  //    通过 resource_id 从 DisplayResourceProvider 获取 GL 纹理 ID
  DisplayResourceProviderGL::ScopedReadLockGL lock(
      resource_provider_, quad->resource_id());
  GLuint texture_id = lock.texture_id();
  GLenum texture_target = lock.target();

  // 2. 绑定纹理
  gl_->ActiveTexture(GL_TEXTURE0);
  gl_->BindTexture(texture_target, texture_id);

  // 3. 设置纹理过滤模式
  GLenum filter = quad->nearest_neighbor ? GL_NEAREST : GL_LINEAR;
  gl_->TexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, filter);
  gl_->TexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, filter);

  // 4. 设置 shader 程序
  //    使用纹理采样 shader
  const Program* program = GetProgramIfInitialized(ProgramKey::Tile());
  SetUseProgram(program, ...);

  // 5. 设置 uniform 变量
  gl_->Uniform1i(program->sampler_location(), 0);  // 纹理单元 0
  gl_->UniformMatrix4fv(program->matrix_location(), 1, GL_FALSE,
                        quad_rect_matrix.data());
  gl_->Uniform4f(program->tex_transform_location(),
                 quad->tex_coord_rect.x(),
                 quad->tex_coord_rect.y(),
                 quad->tex_coord_rect.width(),
                 quad->tex_coord_rect.height());

  // 6. 设置顶点数据
  //    Tile 被绘制为两个三角形（quad）
  SetupQuadForDrawing(quad->rect, ...);

  // 7. 执行绘制调用
  gl_->DrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
  // 绘制 6 个顶点 = 2 个三角形 = 1 个矩形

  num_triangles_drawn_ += 2;
}
```

**Shader 执行 (GPU)**:

```glsl
// Vertex Shader
attribute vec2 a_position;  // 顶点位置
attribute vec2 a_texCoord;  // 纹理坐标
uniform mat4 u_matrix;      // 变换矩阵

varying vec2 v_texCoord;

void main() {
  gl_Position = u_matrix * vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}

// Fragment Shader
precision mediump float;
varying vec2 v_texCoord;
uniform sampler2D u_texture;  // SharedImage 纹理

void main() {
  // 从纹理采样
  // 对于红色 div 的像素，纹理中存储的是 RGB(255, 0, 0)
  gl_FragColor = texture2D(u_texture, v_texCoord);
  // 输出: (1.0, 0.0, 0.0, 1.0) - 红色！
}
```

**最终交换缓冲区**:

```cpp
// components/viz/service/display/output_surface.cc
void OutputSurface::SwapBuffers(viz::OutputSurfaceFrame frame) {
  TRACE_EVENT0("viz", "OutputSurface::SwapBuffers");

  // 1. 确保所有 GL 命令完成
  gl_->Flush();

  // 2. 交换前后缓冲区
  //    前缓冲区（显示中）← 后缓冲区（刚绘制完成）
  context_provider_->ContextSupport()->Swap(
      frame.latency_info,
      base::DoNothing());

  // 3. 等待 VSync 信号
  //    操作系统在下一个 VSync 时将前缓冲区内容显示到屏幕
}

// 平台相关的实现 (例如 X11)
void GLSurfaceGLX::SwapBuffers() {
  glXSwapBuffers(display_, window_);
  // 后缓冲区变为前缓冲区
  // 前缓冲区变为后缓冲区
  // 屏幕将在下一个 VSync 显示新的前缓冲区内容
}
```

**最终结果**:
- 用户屏幕上看到**红色矩形** 🎉
- 位置: (100, 100)
- 大小: 200x200 像素
- 颜色: RGB(255, 0, 0)

## 完整时间线总结 (60 FPS 场景)

**对于红色 div 从光栅化完成到显示的完整耗时**:

| 阶段 | 位置 | 耗时 | 累计 | 说明 |
|------|------|------|------|------|
| **GPU 光栅化** | GPU 进程 | ~2ms | 2ms | Fragment Shader 执行 |
| **IPC 回调** | 跨进程 | ~0.1ms | 2.1ms | 通知 Compositor |
| **状态更新** | Compositor | ~0.2ms | 2.3ms | OnRasterTaskCompleted |
| **Damage 标记** | Compositor | ~0.1ms | 2.4ms | NotifyTileStateChanged |
| **等待 VSync** | 系统 | ~0-16ms | ~8ms | 等待下一个 BeginFrame |
| **BeginImplFrame** | Compositor | ~0.1ms | 8.1ms | WillBeginImplFrame |
| **PrepareToDraw** | Compositor | ~0.5ms | 8.6ms | UpdateDrawProperties |
| **AppendQuads** | Compositor | ~0.3ms | 8.9ms | 构建 DrawQuads |
| **GenerateFrame** | Compositor | ~0.4ms | 9.3ms | 生成 CompositorFrame |
| **SubmitFrame** | IPC | ~0.2ms | 9.5ms | 提交到 Viz |
| **Aggregate** | Viz | ~0.5ms | 10ms | 聚合 Surfaces |
| **DrawQuads** | Viz GPU | ~1.5ms | 11.5ms | GL 绘制命令 |
| **SwapBuffers** | Viz | ~0.1ms | 11.6ms | 交换缓冲区 |
| **等待 VSync** | 系统 | ~0-16ms | ~16.6ms | 显示到屏幕 |
| **总计** | - | **~16.6ms** | - | **60 FPS** ✓ |

**关键观察**:
1. **光栅化本身很快** (~2ms)，但需要等待 VSync 对齐
2. **Compositor 到 Viz 的管道** (~7ms) 占用大部分时间
3. **两次 VSync 等待** 是主要延迟来源（但保证流畅）
4. **异步架构** 允许同时处理多帧：
   - 帧 N 正在显示
   - 帧 N+1 正在 Viz 绘制
   - 帧 N+2 正在 Compositor 准备
   - 帧 N+3 正在 GPU 光栅化

### 核心调用链路图示

```
[GPU 进程]
  DoRasterCHROMIUM (反序列化 PaintOps)
    → PaintOp::Raster (Skia API)
      → SkCanvas::drawRect
        → GPU Command Buffer
          → Fragment Shader
            → 写入 SharedImage 纹理

[IPC 通知]
  GPU Service → Compositor Client

[Compositor 进程 - Raster Worker Thread]
  RasterTaskImpl::RunOnWorkerThread
    → raster_buffer_->Playback() 完成

[切换回 Compositor Thread - by TaskGraphRunner]
  RasterTaskImpl::OnTaskCompleted
    → TileManager::OnRasterTaskCompleted
      → resource_pool_->OnContentReplaced
      → raster_buffer_provider_->NotifyWorkSubmitted
      → TileDrawInfo::SetResource (resource_id=12345)
      → TileDrawInfo::set_resource_ready_for_draw()
      → client_->NotifyTileStateChanged
      → PictureLayerImpl::NotifyTileStateChanged
        → DamageTracker::AddDamageNextUpdate ({0,0,256,256})
      → LayerTreeHostImpl::SetNeedsRedraw
        → Scheduler::SetNeedsRedraw

[VSync 信号触发]
  viz::BeginFrameSource
    → Scheduler::OnBeginFrame
      → LayerTreeHostImpl::WillBeginImplFrame
        → MutatorHost::TickAnimations
      → LayerTreeHostImpl::PrepareToDraw
        → LayerTreeImpl::UpdateDrawProperties
        → TileManager::PrepareToDraw
        → LayerTreeHostImpl::CalculateRenderPasses
          → PictureLayerImpl::AppendQuads
            → Cover 迭代器 → 找到 Tile(0,0)
            → CreateAndAppendDrawQuad<TileDrawQuad>
              → resource_id: 12345
              → rect: {0,0,256,256}
      → LayerTreeHostImpl::DrawLayers
        → LayerTreeHostImpl::GenerateCompositorFrame
          → 构建 resource_list (TransferableResource)
          → 移动 render_pass_list
        → LayerTreeFrameSink::SubmitCompositorFrame

[Viz 进程]
  CompositorFrameSinkImpl::SubmitCompositorFrame
    → Surface::QueueFrame
    → Surface::ActivatePendingFrame
    → Display::SetNeedsRedraw
      → Display::DrawAndSwap
        → SurfaceAggregator::Aggregate
          → 聚合所有 Surfaces
        → GLRenderer::DrawFrame
          → GLRenderer::DrawRenderPassAndExecuteCopyRequests
            → GLRenderer::DrawTileQuad
              → gl_->BindTexture(texture_id=12345)
              → gl_->DrawElements(GL_TRIANGLES, 6)
                → Vertex Shader
                → Fragment Shader (纹理采样)
                  → 输出: RGB(255,0,0) 红色！
        → OutputSurface::SwapBuffers
          → glXSwapBuffers() / eglSwapBuffers()

[系统]
  VSync 信号
    → 屏幕显示前缓冲区
      → 用户看到红色矩形 🎉
```

## 调试建议

使用 `chrome://tracing` 追踪完整流程：

**关键事件**:
1. **`RasterTaskImpl::RunOnWorkerThread`** - 光栅化执行
2. **`TileManager::OnRasterTaskCompleted`** - Tile 就绪
3. **`LayerTreeHostImpl::NotifyTileStateChanged`** - 触发重绘
4. **`LayerTreeHostImpl::WillBeginImplFrame`** - 新帧开始
5. **`PictureLayerImpl::AppendQuads`** - 构建 DrawQuads
6. **`LayerTreeHostImpl::GenerateCompositorFrame`** - 生成合成帧
7. **`Display::DrawAndSwap`** - Viz 绘制
8. **`GLRenderer::DrawTileQuad`** - GPU 绘制 tile

**查看数据**:
- 搜索 tile ID 追踪特定 tile 的生命周期
- 查看 `resource_id` 追踪 SharedImage 从创建到使用
- 观察 `frame_token` 追踪帧的提交和显示

## Paint 到 Display 完整耗时 (60 FPS 场景)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| **Paint** | ~1ms | Blink 生成 DisplayItems |
| **Commit** | ~0.5ms | 传输到 Compositor |
| **Raster** | ~6ms | 光栅化到 GPU 纹理 |
| - 序列化 | ~0.5ms | PaintOps → 二进制 |
| - IPC | ~0.5ms | 共享内存传输 |
| - 反序列化+执行 | ~3ms | 反序列化并调 Skia |
| - GPU 绘制 | ~2ms | Fragment Shader |
| **Composite** | ~3ms | 生成并提交 CompositorFrame |
| **Display** | ~2ms | Viz 绘制到屏幕 |
| **VSync 等待** | ~1.6ms | 等待下一个 VSync |
| **总计** | **~16.6ms** | **60 FPS** ✓ |

### 关键要点

1. **Tile-based**: 页面分成小块 (通常 256x256),按需光栅化,支持并行处理
2. **异步光栅化**: 在独立 worker threads,不阻塞主线程和合成器线程
3. **GPU 加速**: 利用 GPU 的并行能力,快速处理像素
4. **跨进程安全**: Compositor/GPU 完全隔离,共享内存传输 PaintOps
5. **序列化机制** (OOP-R): DisplayItemList 不跨进程传递,PaintOp 流式反序列化执行
6. **多级缓存**: DisplayItem 缓存 + Tile 缓存 + GPU 纹理缓存
7. **R-Tree 优化**: 空间索引快速定位需要绘制的 ops
8. **延迟执行**: Paint 不光栅化,Raster 不合成,Pipeline 并行

---

# 第二部分：TileManager 深度剖析（索引）

TileManager 深度内容保持在独立文档：
- [tile_manager_analysis.md](tile_manager_analysis.md)

---

# 关联阅读

- Paint 篇：[blink_paint_analysis.md](blink_paint_analysis.md)
- PaintOp::Raster 专题：[PAINTOP_RASTER_ANALYSIS.md](PAINTOP_RASTER_ANALYSIS.md)
