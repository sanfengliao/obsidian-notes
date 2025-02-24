
###FragmentData

`FragmentData` 类表示一个布局对象的特定片段的数据。它包含了与该片段相关的各种绘制属性和状态。以下是 `FragmentData` 类的一些关键功能：

- **PaintOffset**: 获取和设置片段的绘制偏移量。
- **UniqueId**: 获取该对象的唯一标识符。
- **Layer**: 获取和设置与布局对象关联的绘制层。
```c++
// 初始化layer
void LayoutBoxModelObject::CreateLayerAfterStyleChange() {
  NOT_DESTROYED();
  DCHECK(!HasLayer() && !Layer());
  FragmentData& first_fragment = GetMutableForPainting().FirstFragment();
  first_fragment.EnsureId();
  first_fragment.SetLayer(MakeGarbageCollected<PaintLayer>(this));
  SetHasLayer(true);
  Layer()->InsertOnlyThisLayerAfterStyleChange();
  // Creating a layer may affect existence of the LocalBorderBoxProperties, so
  // we need to ensure that we update paint properties.
  SetNeedsPaintPropertyUpdate();
}

```
- **StickyConstraints**: 获取和设置粘性位置滚动约束。
- **FragmentID**: 获取和设置片段的唯一 ID。
- **PaintProperties**: 获取、确保和清除与对象相关的绘制属性节点。
- **LocalBorderBoxProperties**: 获取、设置和清除局部边框框属性。
- **CullRect**: 获取和设置裁剪矩形。
- **ContentsProperties**: 获取用于绘制片段内容的完整属性节点集。

`FragmentData` 类还包含一个嵌套的 `RareData` 结构，用于存储不常用的数据字段，如绘制层、粘性位置滚动约束和附加片段等。

### FragmentDataList

`FragmentDataList` 类是 `FragmentData` 类的一个派生类，提供了一些列表功能，用于操作与 `LayoutObject` 关联的 `FragmentData` 条目列表。该类的设计保证了至少存在一个 `FragmentData` 条目，因此例如 `Shrink(0)` 这样的操作是被禁止的。通常情况下，一个 `LayoutObject` 只有一个 `FragmentData` 条目，因此第一个条目直接存储在 `FragmentData` 中，任何额外的条目则存储在第一个 `FragmentData` 的 `rare_data_.additional_fragments` 中。

`FragmentDataList` 类提供了一些方法来操作 `FragmentData` 条目：

- **AppendNewFragment**: 添加一个新的 `FragmentData` 条目。
- **Shrink**: 缩小条目列表的大小，但不能小于 1。
- **front**: 返回第一个 `FragmentData` 条目，并确保当前对象是第一个条目。
- **back**: 返回最后一个 `FragmentData` 条目。
- **at**: 根据索引返回指定的 `FragmentData` 条目。
- **size**: 返回条目列表的大小。

总的来说，`FragmentData` 和 `FragmentDataList` 类通过提供一系列方法来管理和操作与布局对象相关的绘制属性和状态，确保每个 `LayoutObject` 至少有一个 `FragmentData` 条目，并提供了灵活的条目管理功能。
