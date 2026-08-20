在光栅化流程中, `TileManager::PrepareTiles` 是 Tile 调度的核心入口。本文档深入分析这个关键函数如何协调整个 tile 的光栅化流程。

# TileManager::PrepareTiles - 入口函数

这个函数在每次合成器需要准备 tiles 时被调用,通常是在每一帧开始时:

```cpp
bool TileManager::PrepareTiles(
    const GlobalStateThatImpactsTilePriority& state) {

  // 1. 更新计数和活跃时间
  ++prepare_tiles_count_;              // 调用次数统计
  last_active_time_ = NowWithOverride(); // 记录最后活跃时间

  // 2. 调度内存回收任务
  //    当idle时释放不需要的 tile 资源
  ScheduleReduceTileMemoryWhenIdle(base::TimeDelta());
  ScheduleTrimPrepaintTiles();

  // 3. 检查 TileTaskManager 是否就绪
  if (!tile_task_manager_) {
    return false;  // 如果未初始化,直接返回
  }

  // 4. 重置信号状态
  //    这些信号用于通知客户端:
  //    - activate_tile_tasks_completed: 激活所需的 tiles 完成
  //    - draw_tile_tasks_completed: 绘制所需的 tiles 完成
  //    - all_tile_tasks_completed: 所有 tiles 完成
  signals_ = Signals();

  // 5. 保存全局状态
  //    包含: 内存限制、树优先级、可见区域等
  global_state_ = state;

  // 6. 暂时禁止 checker images 的解码
  //    确保可见 tiles 的光栅化优先完成
  checker_image_tracker_.SetNoDecodesAllowed();

  // 7. 检查已完成的任务
  //    防止调度已取消的任务
  if (!did_check_for_completed_tasks_since_last_schedule_tasks_) {
    tile_task_manager_->CheckForCompletedTasks();
    did_check_for_completed_tasks_since_last_schedule_tasks_ = true;
  }

  // 8. 释放被遮挡的 tiles 的资源
  //    如果某些 tiles 被完全遮挡,没必要保留其资源
  if (!ShouldRasterOccludedTiles()) {
    FreeResourcesForOccludedTiles();
  }

  // 9. 核心！分配 GPU 内存给 tiles
  //    决定哪些 tiles 需要光栅化,优先级是什么
  PrioritizedWorkToSchedule prioritized_work = AssignGpuMemoryToTiles();

  // 10. 通知客户端是否需要绘制
  //     如果最高优先级的 tile 需要绘制,提前通知
  client_->SetIsLikelyToRequireADraw(
      !prioritized_work.tiles_to_raster.empty() &&
      prioritized_work.tiles_to_raster.front().tile()->required_for_draw());

  // 11. 调度任务！
  //     创建 RasterTask 并提交到任务队列
  ScheduleTasks(std::move(prioritized_work));

  return true;
}
```

# 关键步骤详解

## 步骤 8: ShouldRasterOccludedTiles & FreeResourcesForOccludedTiles

```cpp
bool TileManager::ShouldRasterOccludedTiles() const {
  // 通常返回 false,除非特殊配置
  return tile_manager_settings_.raster_occluded_tiles;
}

void TileManager::FreeResourcesForOccludedTiles() {
  // 遍历所有有资源的 tiles
  std::unique_ptr<TilesWithResourceIterator> iterator =
      client_->CreateTilesWithResourceIterator();

  for (; !iterator->AtEnd(); iterator->Next()) {
    // 如果这个 tile 被完全遮挡
    if (iterator->IsCurrentTileOccluded()) {
      Tile* tile = iterator->GetCurrent();

      // 释放它的 GPU 资源
      FreeResourcesForTile(tile);

      // 通知客户端 tile 状态改变
      // update_damage=false: 不需要更新损伤区域(因为不可见)
      client_->NotifyTileStateChanged(tile,
                                      /*update_damage=*/false,
                                      /*set_needs_redraw=*/true);
    }
  }
}
```

**优化原理**:

- 被其他层完全遮挡的 tiles 不需要光栅化
- 释放它们的资源可以节省宝贵的 GPU 内存
- 例如:全屏视频播放时,下层的页面内容 tiles 可以释放

**对于红色 div**:

- 如果 div 在最上层,不会被遮挡,保留资源
- 如果被弹窗覆盖,可能会释放资源

## 步骤 9: AssignGpuMemoryToTiles - 最核心的函数

这个函数决定:

1. 哪些 tiles 需要光栅化
2. 光栅化的优先级顺序
3. 如何在内存限制下分配资源

```cpp
TileManager::PrioritizedWorkToSchedule
TileManager::AssignGpuMemoryToTiles() {

  DCHECK(resource_pool_);
  DCHECK(tile_task_manager_);

  // 初始化优先级计数器
  unsigned schedule_priority = 1u;

  // 标记:是否所有需要光栅化的 tiles 都被调度了
  all_tiles_that_need_to_be_rasterized_are_scheduled_ = true;

  // 标记:是否有足够内存调度 NOW bin 的 tiles
  bool had_enough_memory_to_schedule_tiles_needed_now = true;

  // 内存限制
  MemoryUsage hard_memory_limit(
      global_state_.hard_memory_limit_in_bytes,  // 硬限制
      global_state_.num_resources_limit);        // 资源数量限制

  MemoryUsage soft_memory_limit(
      global_state_.soft_memory_limit_in_bytes,  // 软限制
      global_state_.num_resources_limit);

  // 当前内存使用
  MemoryUsage memory_usage(
      resource_pool_->memory_usage_bytes(),
      resource_pool_->resource_count());

  // 创建光栅化优先级队列
  //   tree_priority: ACTIVE_TREE 或 PENDING_TREE
  //   Type::ALL: 包含所有需要光栅化的 tiles
  std::unique_ptr<RasterTilePriorityQueue> raster_priority_queue(
      client_->BuildRasterQueue(global_state_.tree_priority,
                                RasterTilePriorityQueue::Type::ALL));

  // 驱逐优先级队列(用于释放内存)
  std::unique_ptr<EvictionTilePriorityQueue> eviction_priority_queue;

  // 结果:待调度的工作
  PrioritizedWorkToSchedule work_to_schedule;

  const bool raster_occluded_tiles = ShouldRasterOccludedTiles();

  // 遍历所有需要光栅化的 tiles (按优先级从高到低)
  for (; !raster_priority_queue->IsEmpty(); raster_priority_queue->Pop()) {
    const PrioritizedTile& prioritized_tile = raster_priority_queue->Top();
    Tile* tile = prioritized_tile.tile();
    TilePriority priority = prioritized_tile.priority();

    // 检查:这个 tile 的优先级是否违反内存策略
    if (TilePriorityViolatesMemoryPolicy(priority)) {
      break;  // 优先级太低,停止处理
    }

    DCHECK(!prioritized_tile.is_occluded() || raster_occluded_tiles);

    // 固定颜色分析优化
    //   如果 tile 内容是纯色,不需要光栅化
    if (!tile->is_solid_color_analysis_performed() &&
        tile->use_picture_analysis() && kUseColorEstimator) {

      tile->set_solid_color_analysis_performed(true);
      SkColor4f color = SkColors::kTransparent;

      // 分析 tile 的内容是否是纯色
      bool is_solid_color =
          prioritized_tile.raster_source()->PerformSolidColorAnalysis(
              tile->enclosing_layer_rect(), &color, kMaxOpsToAnalyze);

      if (is_solid_color) {
        // 是纯色!不需要光栅化,直接设置颜色
        tile->draw_info().set_solid_color(color);
        client_->NotifyTileStateChanged(tile, /*update_damage=*/true,
                                        /*set_needs_redraw=*/true);
        continue;  // 跳过这个 tile
      }
    }

    // Prepaint tiles 且仅处理图片
    if (tile->is_prepaint() && prioritized_tile.is_process_for_images_only()) {
      work_to_schedule.tiles_to_process_for_images.push_back(prioritized_tile);
      continue;
    }

    // 获取目标色彩参数
    auto content_color_usage =
        GetContentColorUsageForPrioritizedTile(prioritized_tile);
    const auto target_color_params =
        client_->GetTargetColorParams(content_color_usage);

    // 如果 tile 不需要光栅化,只处理 checker images
    if (!tile->draw_info().NeedsRaster()) {
      DCHECK(tile->draw_info().is_checker_imaged());
      DCHECK(prioritized_tile.should_decode_checkered_images_for_tile());

      AddCheckeredImagesToDecodeQueue(
          prioritized_tile, target_color_params,
          CheckerImageTracker::DecodeType::kRaster,
          &work_to_schedule.checker_image_decode_queue);
      continue;
    }

    // 检查:是否达到调度任务数量上限
    if (work_to_schedule.tiles_to_raster.size() >=
        scheduled_raster_task_limit_) {
      all_tiles_that_need_to_be_rasterized_are_scheduled_ = false;
      break;  // 达到上限,停止调度
    }

    DCHECK(tile->draw_info().mode() == TileDrawInfo::OOM_MODE ||
           !tile->draw_info().IsReadyToDraw());

    // 计算这个 tile 需要的内存
    MemoryUsage memory_required_by_tile_to_be_scheduled;
    if (!tile->raster_task_.get()) {
      // 如果还没有 raster task,需要分配新内存
      memory_required_by_tile_to_be_scheduled = MemoryUsage::FromConfig(
          tile->desired_texture_size(), client_->GetTileFormat());
    }

    // 判断这个 tile 是否是立即需要的
    bool tile_is_needed_now = priority.priority_bin == TilePriority::NOW;

    // 选择内存限制:NOW 用硬限制,其他用软限制
    MemoryUsage& tile_memory_limit =
        tile_is_needed_now ? hard_memory_limit : soft_memory_limit;

    const MemoryUsage& scheduled_tile_memory_limit =
        tile_memory_limit - memory_required_by_tile_to_be_scheduled;

    // 如果内存不够,驱逐低优先级的 tiles
    eviction_priority_queue =
        FreeTileResourcesWithLowerPriorityUntilUsageIsWithinLimit(
            std::move(eviction_priority_queue),
            scheduled_tile_memory_limit,
            priority,
            &memory_usage);

    bool memory_usage_is_within_limit =
        !memory_usage.Exceeds(scheduled_tile_memory_limit);

    // 如果内存还是不够,停止调度
    if (!memory_usage_is_within_limit) {
      if (tile_is_needed_now) {
        LOG(ERROR) << "WARNING: tile memory limits exceeded, some content may "
                      "not draw";
        had_enough_memory_to_schedule_tiles_needed_now = false;
      }
      all_tiles_that_need_to_be_rasterized_are_scheduled_ = false;
      break;
    }

    // 如果 tile 已经有调度的任务,处理 checker images
    if (tile->HasRasterTask()) {
      if (tile->raster_task_scheduled_with_checker_images() &&
          prioritized_tile.should_decode_checkered_images_for_tile()) {
        AddCheckeredImagesToDecodeQueue(
            prioritized_tile, target_color_params,
            CheckerImageTracker::DecodeType::kRaster,
            &work_to_schedule.checker_image_decode_queue);
      }
    } else {
      // 创建新的 RasterTask
      auto raster_task = CreateRasterTask(prioritized_tile,
                                         target_color_params,
                                         &work_to_schedule);
      if (!raster_task) {
        continue;
      }

      tile->raster_task_ = std::move(raster_task);
      tile->mark_used();  // 标记为已使用
    }

    // 设置调度优先级
    tile->scheduled_priority_ = schedule_priority++;

    // 更新内存使用
    memory_usage += memory_required_by_tile_to_be_scheduled;

    // 添加到待光栅化列表
    work_to_schedule.tiles_to_raster.push_back(prioritized_tile);
  }

  // 进一步释放内存,确保在硬限制内
  eviction_priority_queue = FreeTileResourcesUntilUsageIsWithinLimit(
      std::move(eviction_priority_queue), hard_memory_limit, &memory_usage);

  // 如果内存不足且有 checker images,为它们安排解码
  if (!had_enough_memory_to_schedule_tiles_needed_now &&
      num_of_tiles_with_checker_images_ > 0) {
    // 遍历剩余的 NOW bin tiles
    for (; !raster_priority_queue->IsEmpty(); raster_priority_queue->Pop()) {
      const PrioritizedTile& prioritized_tile = raster_priority_queue->Top();

      if (prioritized_tile.priority().priority_bin > TilePriority::NOW)
        break;

      if (!prioritized_tile.should_decode_checkered_images_for_tile())
        continue;

      auto content_color_usage =
          GetContentColorUsageForPrioritizedTile(prioritized_tile);
      const auto target_color_params =
          client_->GetTargetColorParams(content_color_usage);

      Tile* tile = prioritized_tile.tile();
      if (tile->draw_info().is_checker_imaged() ||
          tile->raster_task_scheduled_with_checker_images()) {
        AddCheckeredImagesToDecodeQueue(
            prioritized_tile, target_color_params,
            CheckerImageTracker::DecodeType::kRaster,
            &work_to_schedule.checker_image_decode_queue);
      }
    }
  }

  did_oom_on_last_assign_ = !had_enough_memory_to_schedule_tiles_needed_now;

  // 记录内存统计
  memory_stats_from_last_assign_.total_budget_in_bytes =
      global_state_.hard_memory_limit_in_bytes;
  memory_stats_from_last_assign_.total_bytes_used =
      memory_usage.memory_bytes();
  memory_stats_from_last_assign_.had_enough_memory =
      had_enough_memory_to_schedule_tiles_needed_now;

  return work_to_schedule;
}
```

# TilePriority 优先级系统

Tiles 根据优先级决定光栅化顺序:

```cpp
// Priority Bins (从高到低)
enum PriorityBin {
  NOW,        // 当前可见区域,最高优先级
  SOON,       // 即将可见(预光栅化区域)
  EVENTUALLY  // 可能会见到(远距离区域)
};

struct TilePriority {
  PriorityBin priority_bin;
  float distance_to_visible;  // 到可见区域的距离

  bool IsHigherPriorityThan(const TilePriority& other) const {
    if (priority_bin != other.priority_bin)
      return priority_bin < other.priority_bin;
    return distance_to_visible < other.distance_to_visible;
  }
};
```

**内存策略** (`TilePriorityViolatesMemoryPolicy`):

```cpp
bool TileManager::TilePriorityViolatesMemoryPolicy(
    const TilePriority& priority) {
  switch (global_state_.memory_limit_policy) {
    case ALLOW_NOTHING:
      return true;  // 不允许任何光栅化

    case ALLOW_ABSOLUTE_MINIMUM:
      return priority.priority_bin > TilePriority::NOW;  // 只允许 NOW

    case ALLOW_PREPAINT_ONLY:
      return priority.priority_bin > TilePriority::SOON;  // 允许 NOW + SOON

    case ALLOW_ANYTHING:
      return priority.distance_to_visible ==
             std::numeric_limits<float>::infinity();  // 允许几乎所有
  }
}
```

# PrioritizedWorkToSchedule 结构

`AssignGpuMemoryToTiles` 返回的工作清单:

```cpp
struct PrioritizedWorkToSchedule {
  // 需要光栅化的 tiles (按优先级排序)
  std::vector<PrioritizedTile> tiles_to_raster;

  // 只需要处理图片解码的 tiles
  std::vector<PrioritizedTile> tiles_to_process_for_images;

  // 需要解码的 checker images
  CheckerImageTracker::ImageDecodeQueue checker_image_decode_queue;

  // 额外的预绘制图片
  std::vector<DrawImage> extra_prepaint_images;
};
```

# 实际示例:红色 div 的调度

假设页面有:

- 红色 div (Tile 0,0) - 可见
- 蓝色 div (Tile 1,0) - 可见
- 绿色 div (Tile 2,0) - 即将滚动进来
- 黄色 div (Tile 5,5) - 远离可见区域

**AssignGpuMemoryToTiles 执行**:

```
1. 创建 RasterTilePriorityQueue
   ↓ 按优先级排序:
   [Tile(0,0, NOW, dist=0),
    Tile(1,0, NOW, dist=0),
    Tile(2,0, SOON, dist=300),
    Tile(5,5, EVENTUALLY, dist=2000)]

2. 遍历队列:

   Tile(0,0) - 红色 div:
     - priority.bin = NOW ✓
     - 固定颜色分析: 不是纯色 ✗
     - 需要光栅化 ✓
     - 内存充足 ✓
     - 创建 RasterTask
     - 添加到 tiles_to_raster
     - schedule_priority = 1

   Tile(1,0) - 蓝色 div:
     - priority.bin = NOW ✓
     - 需要光栅化 ✓
     - 内存充足 ✓
     - 创建 RasterTask
     - 添加到 tiles_to_raster
     - schedule_priority = 2

   Tile(2,0) - 绿色 div:
     - priority.bin = SOON ✓
     - 内存策略: 如果是 ALLOW_PREPAINT_ONLY,允许 ✓
     - 需要光栅化 ✓
     - 内存充足 ✓
     - 创建 RasterTask
     - 添加到 tiles_to_raster
     - schedule_priority = 3

   Tile(5,5) - 黄色 div:
     - priority.bin = EVENTUALLY
     - 内存策略: 如果是 ALLOW_PREPAINT_ONLY,拒绝 ✗
     - 跳过

3. 返回 PrioritizedWorkToSchedule:
   tiles_to_raster = [Tile(0,0), Tile(1,0), Tile(2,0)]
```

# 内存驱逐机制

当内存不足时,`FreeTileResourcesWithLowerPriorityUntilUsageIsWithinLimit`:

```cpp
std::unique_ptr<EvictionTilePriorityQueue>
TileManager::FreeTileResourcesWithLowerPriorityUntilUsageIsWithinLimit(
    std::unique_ptr<EvictionTilePriorityQueue> eviction_priority_queue,
    const MemoryUsage& limit,
    const TilePriority& other_priority,
    MemoryUsage* usage) {

  while (usage->Exceeds(limit)) {
    if (!eviction_priority_queue) {
      // 创建驱逐队列(按优先级从低到高排序)
      eviction_priority_queue = client_->BuildEvictionQueue();
    }

    if (eviction_priority_queue->IsEmpty())
      break;  // 没有可驱逐的 tiles

    const PrioritizedTile& prioritized_tile = eviction_priority_queue->Top();

    // 如果驱逐 tile 的优先级不低于 other_priority,停止驱逐
    if (!other_priority.IsHigherPriorityThan(prioritized_tile.priority()))
      break;

    Tile* tile = prioritized_tile.tile();
    *usage -= MemoryUsage::FromTile(tile);

    // 释放资源
    FreeResourcesForTileAndNotifyClientIfTileWasReadyToDraw(tile);

    eviction_priority_queue->Pop();
  }

  return eviction_priority_queue;
}
```

**驱逐示例**:

```
假设内存限制: 100MB
当前使用: 90MB
需要为红色 div 分配: 20MB

当前 tiles 内存占用:
  Tile A (EVENTUALLY, 30MB)
  Tile B (SOON, 30MB)
  Tile C (NOW, 30MB)

驱逐过程:
1. 需要释放: 90 + 20 - 100 = 10MB
2. 驱逐队列(从低到高): [Tile A, Tile B, Tile C]
3. 驱逐 Tile A (EVENTUALLY):
   - 优先级低于红色 div (NOW) ✓
   - 释放 30MB
   - 使用量: 90 - 30 = 60MB
4. 检查: 60 + 20 = 80MB < 100MB ✓
5. 停止驱逐
```

# 小结

`PrepareTiles` 的核心工作流程:

```
1. 准备阶段
   - 更新活跃时间
   - 调度内存回收
   - 检查已完成的任务

2. 资源优化
   - 释放被遮挡的 tiles
   - 固定颜色分析(避免不必要的光栅化)

3. 分配内存 (AssignGpuMemoryToTiles)
   - 创建优先级队列
   - 按优先级遍历 tiles
   - 分配内存,必要时驱逐低优先级 tiles
   - 创建 RasterTask

4. 调度任务 (ScheduleTasks)
   - 构建任务图
   - 提交到 TaskGraphRunner
   - 工作线程执行光栅化

5. 结果
   - 高优先级 tiles 被光栅化
   - 低优先级 tiles 等待下一帧
   - 内存使用在限制内
```

**对于红色 div**:

- 如果在可见区域: priority = NOW,第一批被调度
- 分配 GPU 内存(假设 256x256 RGBA8 = 256KB)
- 创建 RasterTask
- 调度到 worker thread 执行
- 光栅化完成后通知合成器

------

# TileManager::ScheduleTasks - 任务调度器

`ScheduleTasks` 负责将 `AssignGpuMemoryToTiles` 准备好的工作转化为实际可执行的任务图,并提交到 `TaskGraphRunner` 在工作线程上执行。

## 函数签名和核心数据结构

```cpp
void TileManager::ScheduleTasks(PrioritizedWorkToSchedule work_to_schedule);
```

**TaskGraph 任务图结构**:

```cpp
struct TaskGraph {
  struct Node {
    scoped_refptr<Task> task;    // 任务对象
    uint16_t category;            // 任务类别(前台/后台)
    uint16_t priority;            // 优先级(0=最高)
    uint32_t dependencies;        // 依赖数量
    bool has_external_dependency; // 是否有外部依赖
  };

  struct Edge {
    raw_ptr<const Task> task;     // 依赖的任务
    raw_ptr<Task> dependent;      // 依赖它的任务
  };

  Node::Vector nodes;  // 所有任务节点
  Edge::Vector edges;  // 任务依赖边
};
```

**TaskCategory 任务类别**:

```cpp
enum TaskCategory {
  TASK_CATEGORY_NONCONCURRENT_FOREGROUND,  // 非并发前台任务(优先级最高)
  TASK_CATEGORY_FOREGROUND,                // 前台任务
  TASK_CATEGORY_BACKGROUND_WITH_NORMAL_THREAD_PRIORITY,  // 普通优先级后台任务
  TASK_CATEGORY_BACKGROUND                 // 低优先级后台任务
};
```

## 完整流程剖析

```cpp
void TileManager::ScheduleTasks(PrioritizedWorkToSchedule work_to_schedule) {

  // ===== 第一阶段: 准备和初始化 =====

  // 记录开始时间(用于性能统计)
  auto start_time = metrics_sub_sampler_.ShouldSample(metrics_sampling_rate_)
                        ? base::TimeTicks::Now()
                        : base::TimeTicks();

  const std::vector<PrioritizedTile>& tiles_that_need_to_be_rasterized =
      work_to_schedule.tiles_to_raster;

  // 取消之前的完成回调(每次调度都创建新的回调)
  task_set_finished_weak_ptr_factory_.InvalidateWeakPtrs();

  // 标记已调度任务
  has_scheduled_tile_tasks_ = true;

  // 初始化计数器:跟踪有多少任务依赖于每个完成通知
  size_t required_for_activate_count = 0;  // 激活所需的任务数
  size_t required_for_draw_count = 0;      // 绘制所需的任务数
  size_t all_count = 0;                    // 所有任务数

  size_t priority = kTileTaskPriorityBase;  // 起始优先级 = 10

  // ===== 第二阶段: 创建三个完成通知任务 =====

  // 1. 激活完成任务:当所有 required_for_activation 的 tiles 完成时触发
  scoped_refptr<TileTask> required_for_activation_done_task =
      CreateTaskSetFinishedTask(
          &TileManager::DidFinishRunningTileTasksRequiredForActivation);

  // 2. 绘制完成任务:当所有 required_for_draw 的 tiles 完成时触发
  scoped_refptr<TileTask> required_for_draw_done_task =
      CreateTaskSetFinishedTask(
          &TileManager::DidFinishRunningTileTasksRequiredForDraw);

  // 3. 全部完成任务:当所有 tiles 完成时触发
  auto all_done_cb = base::BindOnce(
      &TileManager::DidFinishRunningAllTileTasks,
      task_set_finished_weak_ptr_factory_.GetWeakPtr(), start_time);
  scoped_refptr<TileTask> all_done_task =
      base::MakeRefCounted<DidFinishRunningAllTilesTask>(
          task_runner_, pending_raster_queries_, std::move(all_done_cb));

  // ===== 第三阶段: 为每个 tile 构建任务节点和依赖边 =====

  for (auto& prioritized_tile : tiles_that_need_to_be_rasterized) {
    Tile* tile = prioritized_tile.tile();
    TileTask* task = tile->raster_task_.get();  // 获取 RasterTask
    task->set_frame_number(tile->source_frame_number());

    // 建立依赖关系:如果 tile 需要激活,添加边
    if (tile->required_for_activation()) {
      required_for_activate_count++;
      // task -> required_for_activation_done_task 的依赖边
      graph_.edges.emplace_back(task, required_for_activation_done_task.get());
    }

    // 如果 tile 需要绘制,添加边
    if (tile->required_for_draw()) {
      required_for_draw_count++;
      // task -> required_for_draw_done_task 的依赖边
      graph_.edges.emplace_back(task, required_for_draw_done_task.get());
    }

    // 所有任务都依赖于 all_done_task
    all_count++;
    // task -> all_done_task 的依赖边
    graph_.edges.emplace_back(task, all_done_task.get());

    // 决定任务类别:关键任务使用前台类别
    bool use_foreground_category =
        tile->required_for_draw() || tile->required_for_activation() ||
        prioritized_tile.priority().priority_bin == TilePriority::NOW;

    // 插入 RasterTask 节点(包括其图片解码依赖)
    InsertNodesForRasterTask(task, priority++, use_foreground_category);
  }

  // ===== 第四阶段: 处理图片预解码任务 =====

  // 从 tiles_to_process_for_images 提取需要预解码的图片
  const std::vector<PrioritizedTile>& tiles_to_process_for_images =
      work_to_schedule.tiles_to_process_for_images;
  std::vector<DrawImage> new_locked_images;

  for (const PrioritizedTile& prioritized_tile : tiles_to_process_for_images) {
    auto content_color_usage =
        GetContentColorUsageForPrioritizedTile(prioritized_tile);
    const auto target_color_params =
        client_->GetTargetColorParams(content_color_usage);

    std::vector<DrawImage> sync_decoded_images;
    std::vector<PaintImage> checkered_images;

    // 划分图片:同步解码 vs checker 解码
    PartitionImagesForCheckering(prioritized_tile, target_color_params,
                                 &sync_decoded_images, &checkered_images,
                                 nullptr);

    // 同步解码的图片添加到任务图
    new_locked_images.insert(
        new_locked_images.end(),
        std::make_move_iterator(sync_decoded_images.begin()),
        std::make_move_iterator(sync_decoded_images.end()));

    // Checker 图片添加到解码队列
    for (auto& image : checkered_images) {
      work_to_schedule.checker_image_decode_queue.emplace_back(
          std::move(image), CheckerImageTracker::DecodeType::kPreDecode);
    }
  }

  // 添加额外的预绘制图片
  new_locked_images.insert(new_locked_images.end(),
                           work_to_schedule.extra_prepaint_images.begin(),
                           work_to_schedule.extra_prepaint_images.end());

  // 为预解码图片创建任务
  ImageDecodeCache::TracingInfo tracing_info(prepare_tiles_count_,
                                             TilePriority::SOON);
  std::vector<scoped_refptr<TileTask>> new_locked_image_tasks =
      image_controller_.SetPredecodeImages(new_locked_images, tracing_info);
  decoded_image_tracker_.OnImagesUsedInDraw(new_locked_images);

  // 将图片解码任务添加到任务图
  for (auto& task : new_locked_image_tasks) {
    // 检查任务是否已在图中(避免重复)
    auto decode_it =
        std::ranges::find(graph_.nodes, task.get(), &TaskGraph::Node::task);
    if (decode_it != graph_.nodes.end())
      continue;

    // 插入解码任务节点
    InsertNodeForDecodeTask(task.get(), priority++, false);
    all_count++;
    graph_.edges.emplace_back(task.get(), all_done_task.get());
  }

  // 保留图片任务引用(防止被释放)
  locked_image_tasks_.swap(new_locked_image_tasks);

  // ===== 第五阶段: 资源清理 =====

  // 在调度前减少未使用资源(防止超出限制)
  resource_pool_->ReduceResourceUsage();
  image_controller_.ReduceMemoryUsage();

  // ===== 第六阶段: 插入完成通知任务节点 =====

  bool only_completion_tasks = graph_.nodes.empty();

  // 使用最高优先级类别(NONCONCURRENT_FOREGROUND)
  InsertNodeForTask(&graph_, required_for_activation_done_task.get(),
                    TASK_CATEGORY_NONCONCURRENT_FOREGROUND,
                    kRequiredForActivationDoneTaskPriority,  // priority = 1
                    required_for_activate_count);

  InsertNodeForTask(&graph_, required_for_draw_done_task.get(),
                    TASK_CATEGORY_NONCONCURRENT_FOREGROUND,
                    kRequiredForDrawDoneTaskPriority,  // priority = 2
                    required_for_draw_count);

  InsertNodeForTask(&graph_, all_done_task.get(),
                    TASK_CATEGORY_NONCONCURRENT_FOREGROUND,
                    kAllDoneTaskPriority,  // priority = 3
                    all_count);

  // ===== 第七阶段: 快速路径优化 =====

  // 如果没有实际工作(只有完成任务),直接执行完成任务
  if (only_completion_tasks &&
      base::FeatureList::IsEnabled(features::kFastPathNoRaster)) {

    // 快速路径:同步执行完成任务
    for (const auto& task : graph_.nodes) {
      auto* tile_task = static_cast<TileTask*>(task.task.get());
      tile_task->state().DidSchedule();
      tile_task->state().DidStart();
      tile_task->RunOnWorkerThread();  // 直接在当前线程执行
      tile_task->state().DidFinish();
      tile_task->OnTaskCompleted();
      tile_task->DidComplete();
    }
    graph_.Reset();

    // 跳过提交到 TaskGraphRunner
  }

   // ===== 第八阶段: 提交任务图到 TaskGraphRunner =====

    // 将任务图提交到工作线程池执行
    // 这会取消之前未完成的任务,并开始执行新任务
    tile_task_manager_->ScheduleTasks(&graph_);

  // ===== 第九阶段: 调度 checker image 解码 =====

  // Checker images 单独调度(不阻塞主要光栅化)
  checker_image_tracker_.ScheduleImageDecodeQueue(
      std::move(work_to_schedule.checker_image_decode_queue));

  // ===== 第十阶段: 清理 =====

  // 清空任务图结构(TaskGraphRunner 已持有必要状态)
  graph_.Reset();

  did_check_for_completed_tasks_since_last_schedule_tasks_ = false;
}
```

## InsertNodesForRasterTask - 插入光栅化任务节点

这个函数负责将 RasterTask 及其依赖(图片解码任务)插入到任务图中:

```cpp
void TileManager::InsertNodesForRasterTask(TileTask* raster_task,
                                           uint16_t priority,
                                           bool use_foreground_category) {
  size_t dependencies = 0u;

  // 遍历 raster_task 的所有图片解码依赖
  for (auto it = raster_task->dependencies().begin();
       it != raster_task->dependencies().end(); ++it) {
    TileTask* decode_task = it->get();

    // 如果已经解码完成,跳过
    if (decode_task->HasCompleted()) {
      continue;
    }

    dependencies++;  // 增加依赖计数

    // 检查解码任务是否已在图中
    auto decode_it =
        std::ranges::find(graph_.nodes, decode_task, &TaskGraph::Node::task);

    // 升级类别:如果前台任务依赖后台任务,升级为前台
    if (decode_it != graph_.nodes.end() && use_foreground_category &&
        !IsForegroundCategory(decode_it->category)) {
      decode_it->category = TASK_CATEGORY_FOREGROUND;
    }

    // 如果不在图中,插入解码任务节点
    if (decode_it == graph_.nodes.end()) {
      InsertNodeForDecodeTask(decode_task, priority, use_foreground_category);
    }

    // 添加依赖边: decode_task -> raster_task
    graph_.edges.emplace_back(decode_task, raster_task);
  }

  // 插入 raster_task 节点
  InsertNodeForTask(
      &graph_, raster_task,
      TaskCategoryForTileTask(raster_task, use_foreground_category), priority,
      dependencies);
}
```

## 任务图示例:红色 div 的调度

假设场景:

- 红色 div (Tile A): required_for_draw = true, 需要解码一张图片
- 蓝色 div (Tile B): required_for_draw = false, 无图片依赖

**构建的任务图**:

```
节点 (Nodes):
  [1] ImageDecodeTask_A    (priority=10, category=FOREGROUND, deps=0)
  [2] RasterTask_A         (priority=11, category=FOREGROUND, deps=1)
  [3] RasterTask_B         (priority=12, category=BACKGROUND, deps=0)
  [4] ActivationDoneTask   (priority=1,  category=NONCONCURRENT_FOREGROUND, deps=0)
  [5] DrawDoneTask         (priority=2,  category=NONCONCURRENT_FOREGROUND, deps=1)
  [6] AllDoneTask          (priority=3,  category=NONCONCURRENT_FOREGROUND, deps=2)

边 (Edges):
  ImageDecodeTask_A -> RasterTask_A     # 必须先解码图片
  RasterTask_A      -> DrawDoneTask     # Tile A 是 required_for_draw
  RasterTask_A      -> AllDoneTask
  RasterTask_B      -> AllDoneTask

执行顺序(由 TaskGraphRunner 决定):
  1. ImageDecodeTask_A 先执行(前台类别,无依赖)
  2. ImageDecodeTask_A 完成后,RasterTask_A 可以执行
  3. RasterTask_B 并行执行(后台类别)
  4. RasterTask_A 完成后,DrawDoneTask 执行
     -> 触发 DidFinishRunningTileTasksRequiredForDraw()
  5. 所有 RasterTask 完成后,AllDoneTask 执行
     -> 触发 DidFinishRunningAllTileTasks()
```

### 任务类别和优先级

**任务类别决定线程优先级**:

```cpp
TaskCategory TaskCategoryForTileTask(TileTask* task,
                                     bool use_foreground_category) {
  // 不支持并发的任务(必须在主线程)
  if (!task->supports_concurrent_execution())
    return TASK_CATEGORY_NONCONCURRENT_FOREGROUND;

  // 前台任务:关键任务
  if (use_foreground_category)
    return TASK_CATEGORY_FOREGROUND;

  // 支持后台线程的任务
  if (!task->supports_background_thread_priority())
    return TASK_CATEGORY_BACKGROUND_WITH_NORMAL_THREAD_PRIORITY;

  // 低优先级后台任务
  return TASK_CATEGORY_BACKGROUND;
}
```

**优先级分配**:

```cpp
constexpr size_t kTileTaskPriorityBase = 10u;

// 完成通知任务优先级最高(1-3)
const size_t kRequiredForActivationDoneTaskPriority = 1u;
const size_t kRequiredForDrawDoneTaskPriority = 2u;
const size_t kAllDoneTaskPriority = 3u;

// 实际光栅化任务从 10 开始递增
priority = kTileTaskPriorityBase;  // 10, 11, 12, 13...
```

### 完成通知机制

**三个关键回调**:

```cpp
// 1. 激活完成:pending tree 可以激活成 active tree
DidFinishRunningTileTasksRequiredForActivation() {
  signals_.activate_tile_tasks_completed = true;
  signals_check_notifier_.Schedule();  // 通知客户端
}

// 2. 绘制完成:可以开始绘制
DidFinishRunningTileTasksRequiredForDraw() {
  signals_.draw_tile_tasks_completed = true;
  signals_check_notifier_.Schedule();
}

// 3. 全部完成:所有 tiles 就绪
DidFinishRunningAllTileTasks(start_time, has_pending_queries) {
  has_scheduled_tile_tasks_ = false;
  has_pending_queries_ = has_pending_queries;

  if (all_tiles_that_need_to_be_rasterized_are_scheduled_ &&
      !resource_pool_->ResourceUsageTooHigh()) {
    signals_.all_tile_tasks_completed = true;
    signals_check_notifier_.Schedule();
  } else {
    // 还有更多工作,再次调度 PrepareTiles
    more_tiles_need_prepare_check_notifier_.Schedule();
  }
}
```

## 快速路径优化

当页面滚动但内容不变时(所有 tiles 已光栅化),`only_completion_tasks` 为 true:

```cpp
if (only_completion_tasks &&
    base::FeatureList::IsEnabled(features::kFastPathNoRaster)) {

  // 常见场景:滚动但无需重新光栅化
  // 不通过 TaskGraphRunner,直接同步执行完成任务

  for (const auto& task : graph_.nodes) {
    // 只有三个完成通知任务
    DCHECK(task.task == required_for_activation_done_task.get() ||
           task.task == required_for_draw_done_task.get() ||
           task.task == all_done_task.get());

    auto* tile_task = static_cast<TileTask*>(task.task.get());
    tile_task->state().DidSchedule();
    tile_task->state().DidStart();
    tile_task->RunOnWorkerThread();  // 实际执行回调
    tile_task->state().DidFinish();
    tile_task->OnTaskCompleted();
    tile_task->DidComplete();
  }

  graph_.Reset();
  // 跳过 tile_task_manager_->ScheduleTasks(&graph_)
}
```

**性能优势**:

- 避免任务图调度开销
- 避免线程切换
- 立即通知客户端可以绘制
- 对纯滚动场景性能提升显著

## 时序图:从 PrepareTiles 到任务执行

```
[Main Thread]                [Worker Threads]
     |
PrepareTiles()
     ├─> AssignGpuMemoryToTiles()
     │     ├─ 选择 Tile A, B
     │     └─ 创建 RasterTask A, B
     │
     └─> ScheduleTasks()
           ├─ 创建完成通知任务
           ├─ 构建任务图
           │   ├─ Node: RasterTask A
           │   ├─ Node: RasterTask B
           │   ├─ Edge: RasterA -> DrawDone
           │   └─ Edge: RasterB -> AllDone
           │
           └─ tile_task_manager_->ScheduleTasks(&graph_)
                 │
                 └─────────────────────────┐
                                           ├──> [Worker 1]
                                           │      RasterTask A
                                           │        ├─ Playback
                                           │        ├─ Raster to GPU
                                           │        └─ Complete
                                           │           └─> DrawDoneTask
                                           │                  └─> [Main]
                                           │                      DidFinishForDraw()
                                           │
                                           └──> [Worker 2]
                                                  RasterTask B
                                                    ├─ Playback
                                                    ├─ Raster to GPU
                                                    └─ Complete
                                                        └─> AllDoneTask
                                                               └─> [Main]
                                                                   DidFinishAll()
```

## 小结

`ScheduleTasks` 的核心职责:

1. **任务图构建**:将高优先级 tiles 转化为任务节点和依赖边
2. **类别分配**:关键任务使用前台类别,获得更高线程优先级
3. **依赖管理**:确保图片解码在光栅化之前完成
4. **完成通知**:三级通知机制(激活/绘制/全部)
5. **快速路径**:无工作时跳过任务图调度,直接通知完成
6. **资源管理**:调度前清理未使用资源,防止内存超限

**对于红色 div**:

- RasterTask 被添加到任务图,优先级 = 10+
- 如果 required_for_draw,建立 -> DrawDoneTask 的依赖边
- 如果有图片,先插入 ImageDecodeTask,建立依赖边
- 使用 FOREGROUND 类别(因为 NOW bin)
- TaskGraphRunner 分配工作线程执行光栅化
- 完成后触发 DrawDoneTask,通知合成器可以绘制

------

# tile_task_manager_->ScheduleTasks(&graph_) 深度剖析

这是 `TileManager::ScheduleTasks` 中最关键的一行代码,负责将构建好的任务图提交给工作线程池执行。

## 调用链路

```cpp
TileManager::ScheduleTasks()
  └─> tile_task_manager_->ScheduleTasks(&graph_)
        └─> task_graph_runner_->ScheduleTasks(namespace_token_, graph)
              └─> work_queue_.ScheduleTasks(token, graph)
                    ├─ 取消旧任务
                    ├─ 构建就绪队列
                    └─ 通知工作线程
```

## TileTaskManager 封装层

**TileTaskManager** 是对 TaskGraphRunner 的简单封装:

```cpp
// cc/tiles/tile_task_manager.h
class TileTaskManager {
 public:
  // 调度任务图中的所有 tile 任务及其依赖
  // 之前调度但不在新图中的任务会被取消(除非正在运行)
  // 一旦调度且未被取消,任务保证会执行
  virtual void ScheduleTasks(TaskGraph* graph) = 0;

  // 检查已完成的任务并调用 OnTaskCompleted()
  virtual void CheckForCompletedTasks() = 0;

  // 关闭:取消所有之前调度的任务
  virtual void Shutdown() = 0;
};
```

**TileTaskManagerImpl 实现**:

```cpp
void TileTaskManagerImpl::ScheduleTasks(TaskGraph* graph) {
  TRACE_EVENT0("cc", "TileTaskManagerImpl::ScheduleTasks");

  // 直接转发给 TaskGraphRunner
  // namespace_token_: 用于隔离不同组件的任务
  task_graph_runner_->ScheduleTasks(namespace_token_, graph);
}
```

## TaskGraphRunner - 核心调度器

**接口定义** (cc/raster/task_graph_runner.h):

```cpp
class TaskGraphRunner {
 public:
  // 调度任务图中的任务运行
  // - 之前调度但不在新图中的任务会被取消(除非正在运行)
  // - 一旦调度,任务保证最终出现在 completed_tasks 中
  virtual void ScheduleTasks(NamespaceToken token, TaskGraph* graph) = 0;

  // 等待所有调度的任务完成运行
  virtual void WaitForTasksToFinishRunning(NamespaceToken token) = 0;

  // 收集所有已完成的任务
  virtual void CollectCompletedTasks(NamespaceToken token,
                                     Task::Vector* completed_tasks) = 0;
};
```

**实际实现**: `CategorizedWorkerPoolJob` (cc/raster/categorized_worker_pool.cc)

这是 Chrome 使用的工作线程池实现,支持多种任务类别和优先级。

```cpp
void CategorizedWorkerPoolJob::ScheduleTasks(NamespaceToken token,
                                             TaskGraph* graph) {
  TRACE_EVENT2("disabled-by-default-cc.debug",
               "CategorizedWorkerPool::ScheduleTasks",
               "num_nodes", graph->nodes.size(),
               "num_edges", graph->edges.size());

  base::JobHandle* job_handle_to_notify = nullptr;

  {
    // 获取锁,在锁保护下调度任务
    base::AutoLock lock(lock_);
    job_handle_to_notify = ScheduleTasksWithLockAcquired(token, graph);
  }

  // 通知工作线程池:有新任务可以执行了
  if (job_handle_to_notify) {
    job_handle_to_notify->NotifyConcurrencyIncrease();
  }
}

base::JobHandle* CategorizedWorkerPoolJob::ScheduleTasksWithLockAcquired(
    NamespaceToken token,
    TaskGraph* graph) {

  DCHECK(token.IsValid());
  DCHECK(!TaskGraphWorkQueue::DependencyMismatch(graph));

  // 调用核心工作队列的 ScheduleTasks
  work_queue_.ScheduleTasks(token, graph);

  // 返回需要通知的 JobHandle (前台或后台)
  return GetJobHandleToNotifyWithLockAcquired();
}
```

## TaskGraphWorkQueue - 任务队列核心

**ScheduleTasks 完整流程** (cc/raster/task_graph_work_queue.cc):

```cpp
void TaskGraphWorkQueue::ScheduleTasks(NamespaceToken token, TaskGraph* graph) {
  TaskNamespace& task_namespace = namespaces_[token];

  // ===== 第一阶段:更新依赖计数 =====

  // 遍历已完成的任务,减少依赖它们的任务的依赖计数
  for (const scoped_refptr<Task>& task : task_namespace.completed_tasks) {
    for (DependentIterator node_it(graph, task.get()); node_it; ++node_it) {
      TaskGraph::Node& node = *node_it;
      DCHECK_LT(0u, node.dependencies);
      node.dependencies--;  // 依赖数减 1
    }
  }

  // ===== 第二阶段:构建新的就绪队列 =====

  // 清空旧的就绪队列
  for (auto& ready_to_run_tasks_it : task_namespace.ready_to_run_tasks) {
    ready_to_run_tasks_it.second.clear();
  }

  // 遍历新任务图中的所有节点
  for (const TaskGraph::Node& node : graph->nodes) {

    // 从旧图中移除这个任务(如果存在)
    // 剩下的就是需要取消的任务
    auto old_it = std::ranges::find(task_namespace.graph.nodes, node.task,
                                    &TaskGraph::Node::task);
    if (old_it != task_namespace.graph.nodes.end()) {
      std::swap(*old_it, task_namespace.graph.nodes.back());

      // 如果任务已调度但未开始运行,重置状态
      if (node.task->state().IsScheduled())
        node.task->state().Reset();

      task_namespace.graph.nodes.pop_back();
    }

    // 检查任务是否就绪

    // 1. 如果有依赖未满足,跳过
    if (node.dependencies)
      continue;

    // 2. 如果已完成,跳过
    if (node.task->state().IsFinished())
      continue;

    // 3. 如果正在运行,跳过
    if (base::Contains(task_namespace.running_tasks, node.task.get(),
                       &CategorizedTask::second)) {
      continue;
    }

    // 任务就绪!添加到就绪队列
    node.task->state().DidSchedule();
    task_namespace.ready_to_run_tasks[node.category].emplace_back(
        node.task, &task_namespace, node.category, node.priority);
  }

  // ===== 第三阶段:建立优先级堆 =====

  // 对每个类别的就绪队列,按优先级建堆
  for (auto& it : task_namespace.ready_to_run_tasks) {
    auto& ready_to_run_tasks = it.second;
    std::make_heap(ready_to_run_tasks.begin(), ready_to_run_tasks.end(),
                   CompareTaskPriority);  // 最小堆:优先级小的在顶部
  }

  // ===== 第四阶段:交换任务图 =====

  // 用新图替换旧图
  task_namespace.graph.Swap(graph);

  // ===== 第五阶段:取消旧任务 =====

  // graph 现在包含旧图中不在新图中的任务
  for (auto it = graph->nodes.begin(); it != graph->nodes.end(); ++it) {
    TaskGraph::Node& node = *it;

    // 跳过已完成的任务
    if (node.task->state().IsFinished())
      continue;

    // 跳过正在运行的任务(不能取消)
    if (base::Contains(task_namespace.running_tasks, node.task.get(),
                       &CategorizedTask::second)) {
      continue;
    }

    // 取消任务
    DCHECK(!base::Contains(task_namespace.completed_tasks, node.task.get()));
    node.task->state().DidCancel();
    task_namespace.completed_tasks.push_back(node.task);  // 加入完成队列
  }

  // ===== 第六阶段:重建 namespace 队列 =====

  // 清空旧的 namespace 队列
  for (auto& ready_to_run_namespaces_it : ready_to_run_namespaces_) {
    ready_to_run_namespaces_it.second.clear();
  }

  // 为每个类别,收集有就绪任务的 namespace
  for (auto& namespace_it : namespaces_) {
    auto& task_namespace_to_check = namespace_it.second;
    for (auto& ready_to_run_tasks_it :
         task_namespace_to_check.ready_to_run_tasks) {
      auto& ready_to_run_tasks = ready_to_run_tasks_it.second;
      uint16_t category = ready_to_run_tasks_it.first;
      if (!ready_to_run_tasks.empty()) {
        ready_to_run_namespaces_[category].push_back(&task_namespace_to_check);
      }
    }
  }

  // 为每个类别的 namespace 队列建堆
  RebuildNamespaceHeaps(ready_to_run_namespaces_);
}
```

## 工作线程执行流程

**GetNextTaskToRun** - 工作线程取任务:

```cpp
TaskGraphWorkQueue::PrioritizedTask
TaskGraphWorkQueue::GetNextTaskToRun(uint16_t category) {

  TaskNamespace::Vector& ready_to_run_namespaces =
      ready_to_run_namespaces_[category];
  DCHECK(!ready_to_run_namespaces.empty());

  // 1. 从 namespace 堆中取出最高优先级的 namespace
  std::pop_heap(ready_to_run_namespaces.begin(),
                ready_to_run_namespaces.end(),
                CompareTaskNamespacePriority(category));
  TaskNamespace* task_namespace = ready_to_run_namespaces.back();
  ready_to_run_namespaces.pop_back();

  // 2. 从这个 namespace 的就绪队列中取出最高优先级任务
  PrioritizedTask::Vector& ready_to_run_tasks =
      task_namespace->ready_to_run_tasks[category];
  DCHECK(!ready_to_run_tasks.empty());

  std::pop_heap(ready_to_run_tasks.begin(), ready_to_run_tasks.end(),
                CompareTaskPriority);
  PrioritizedTask task = std::move(ready_to_run_tasks.back());
  ready_to_run_tasks.pop_back();

  // 3. 如果 namespace 还有任务,重新加入 namespace 堆
  if (!ready_to_run_tasks.empty()) {
    ready_to_run_namespaces.push_back(task_namespace);
    std::push_heap(ready_to_run_namespaces.begin(),
                   ready_to_run_namespaces.end(),
                   CompareTaskNamespacePriority(category));
  }

  // 4. 标记任务为运行中
  task.task->state().DidStart();
  task_namespace->running_tasks.emplace_back(task.task, std::move(task));

  return task;
}
```

**工作线程循环** (简化):

```cpp
void WorkerThread::Run() {
  while (true) {
    // 1. 等待通知或检查是否有任务
    WaitForWork();

    // 2. 从队列中取任务(根据类别)
    PrioritizedTask task = work_queue_.GetNextTaskToRun(category);
    if (!task.task)
      break;  // 没有任务,退出

    // 3. 执行任务
    task.task->RunOnWorkerThread();

    // 4. 标记任务完成
    task.task->state().DidFinish();
    work_queue_.CompleteTask(task);
  }
}
```

## 数据结构

**TaskNamespace** - 每个 token 对应一个命名空间:

```cpp
struct TaskNamespace {
  // 当前任务图
  TaskGraph graph;

  // 按类别分组的就绪队列(每个是最小堆)
  std::map<uint16_t, PrioritizedTask::Vector> ready_to_run_tasks;

  // 正在运行的任务
  std::vector<CategorizedTask> running_tasks;

  // 已完成的任务(等待收集)
  std::vector<scoped_refptr<Task>> completed_tasks;
};
```

**就绪队列层级**:

```
ready_to_run_namespaces_[FOREGROUND]
  └─> [namespace_1, namespace_2, ...]  (按 namespace 优先级堆排序)
        └─> namespace_1.ready_to_run_tasks[FOREGROUND]
              └─> [task_1, task_2, ...]  (按任务优先级堆排序)
```

## 实际示例:红色 div 的调度

假设 TileManager 构建了如下任务图:

```
图节点:
  [1] ImageDecodeTask (category=FOREGROUND, priority=10, deps=0)
  [2] RasterTask_RedDiv (category=FOREGROUND, priority=11, deps=1)
  [3] DrawDoneTask (category=NONCONCURRENT_FOREGROUND, priority=2, deps=1)

图边:
  ImageDecodeTask -> RasterTask_RedDiv
  RasterTask_RedDiv -> DrawDoneTask
```

**调度过程**:

```cpp
1. TaskGraphWorkQueue::ScheduleTasks() 调用:

   - 更新依赖计数:
     RasterTask_RedDiv.dependencies = 1 (等待 ImageDecodeTask)
     DrawDoneTask.dependencies = 1 (等待 RasterTask_RedDiv)

   - 构建就绪队列:
     ready_to_run_tasks[FOREGROUND] = [ImageDecodeTask]
     // RasterTask_RedDiv 和 DrawDoneTask 有依赖,不就绪

   - 建堆:
     ready_to_run_tasks[FOREGROUND] 按优先级排序

   - 通知工作线程

2. Worker Thread 1 执行:

   - GetNextTaskToRun(FOREGROUND)
     → 返回 ImageDecodeTask

   - ImageDecodeTask->RunOnWorkerThread()
     → 解码图片

   - CompleteTask(ImageDecodeTask)
     → RasterTask_RedDiv.dependencies-- = 0
     → RasterTask_RedDiv 加入就绪队列

3. Worker Thread 2 执行:

   - GetNextTaskToRun(FOREGROUND)
     → 返回 RasterTask_RedDiv

   - RasterTask_RedDiv->RunOnWorkerThread()
     → 执行光栅化 (playback + GPU raster)

   - CompleteTask(RasterTask_RedDiv)
     → DrawDoneTask.dependencies-- = 0
     → DrawDoneTask 加入就绪队列

4. Main Thread (NONCONCURRENT_FOREGROUND):

   - GetNextTaskToRun(NONCONCURRENT_FOREGROUND)
     → 返回 DrawDoneTask

   - DrawDoneTask->RunOnWorkerThread()
     → 执行 DidFinishRunningTileTasksRequiredForDraw()
     → signals_.draw_tile_tasks_completed = true
     → 通知客户端可以绘制

5. TileManager::CheckForCompletedTasks():

   - CollectCompletedTasks()
     → completed_tasks = [ImageDecodeTask, RasterTask_RedDiv, DrawDoneTask]

   - 对每个任务:
     → task->OnTaskCompleted()
     → task->DidComplete()
```

## 任务取消机制

**场景**:用户快速滚动,新的 tile 变为可见,旧的 tile 离开视口

```cpp
帧 N:
  graph = [RasterTask_TileA (NOW), RasterTask_TileB (SOON)]
  → 两个任务都就绪并开始执行

用户滚动:

帧 N+1:
  graph = [RasterTask_TileC (NOW), RasterTask_TileD (SOON)]
  → TileA 和 TileB 不在新图中

  ScheduleTasks() 处理:
    - TileA 正在运行 → 不能取消,让它完成
    - TileB 还在队列中 → 取消,加入 completed_tasks
    - TileC 和 TileD → 加入就绪队列
```

## 性能优化

**1. 分类调度**:

```cpp
NONCONCURRENT_FOREGROUND  → 主线程,最高优先级
FOREGROUND                → 前台工作线程,高优先级
BACKGROUND_NORMAL         → 后台工作线程,普通优先级
BACKGROUND                → 后台工作线程,低优先级
```

**2. 优先级堆**:

- O(log N) 插入和删除
- 总是取出最高优先级任务

**3. Namespace 隔离**:

- Tile rasterization, image decoding 等不同组件使用独立 namespace
- 避免相互干扰

**4. 依赖追踪**:

- 自动管理任务依赖
- 依赖满足后立即就绪

## 小结

`tile_task_manager_->ScheduleTasks(&graph_)` 的完整流程:

```
1. TileManager 构建任务图
   ├─ 节点:RasterTask, DecodeTask, DoneTask
   └─ 边:依赖关系

2. TileTaskManager 转发
   └─ 添加 namespace_token

3. TaskGraphRunner 调度
   └─ CategorizedWorkerPoolJob

4. TaskGraphWorkQueue 处理
   ├─ 更新依赖计数
   ├─ 构建就绪队列(按类别)
   ├─ 建立优先级堆
   ├─ 取消旧任务
   └─ 通知工作线程

5. Worker threads 执行
   ├─ GetNextTaskToRun() 取任务
   ├─ RunOnWorkerThread() 执行
   ├─ CompleteTask() 更新依赖
   └─ 通知后继任务

6. 主线程收集结果
   └─ CheckForCompletedTasks()
       ├─ OnTaskCompleted()
       └─ DidComplete()
```

**关键点**:

- **异步执行**:ScheduleTasks 不会阻塞,立即返回
- **并行处理**:多个工作线程并行执行无依赖的任务
- **依赖管理**:自动追踪依赖,满足后自动就绪
- **取消支持**:旧任务自动取消(除非正在运行)
- **优先级保证**:高优先级任务优先执行

这就是为什么 Chrome 能够流畅渲染 —— 智能的任务调度系统确保关键任务(如可见 tiles)优先完成,同时充分利用多核 CPU 并行处理。