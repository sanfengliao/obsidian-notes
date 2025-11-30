

```c++
// ScrollableArea for the root frame's viewport. This class ties together the
// concepts of layout and visual viewports, used in pinch-to-zoom. This class
// takes two ScrollableAreas, one for the visual viewport and one for the
// layout viewport, and delegates and composes the ScrollableArea API as needed
// between them. For most scrolling APIs, this class will split the scroll up
// between the two viewports in accord with the pinch-zoom semantics. For other
// APIs that don't make sense on the combined viewport, the call is delegated to
// the layout viewport. Thus, we could say this class is a decorator on the
// LocalFrameView scrollable area that adds pinch-zoom semantics to scrolling.
// 根框架视口的可滚动区域。这个类将布局视口和视觉视口的概念联系在一起，用于捏合缩放操作。
// 该类接受两个可滚动区域，一个用于视觉视口(VistualViewPort)，一个用于布局视口(LayoutViewPort)，并根据需要在它们之间委托和组合可滚动区域的 API。
// 对于大多数滚动 API，这个类会根据捏合缩放的语义在两个视口之间分配滚动操作。
// 对于其他在组合视口上没有意义的 API 调用，则会委托给布局视口。
// 因此，我们可以说这个类是对 LocalFrameView 可滚动区域的一个装饰器，它为滚动操作添加了捏合缩放的语义。
class CORE_EXPORT RootFrameViewport final
    : public GarbageCollected<RootFrameViewport>,
      public ScrollableArea {
  private:
     Member<ScrollableArea> visual_viewport_;
     Member<ScrollableArea> layout_viewport_;
}
```
从注释可以看出，这个类负责整合`VisualViewPort` 和`LayoutViewPort` 滚动操作。当通过触控板用双指捏合和，会放大缩小`VistualViewPort`。当`VistualViewPort`被放大时，视图窗口也变成了可滚动区域，因此当进行滚动时，要先滚动视图窗口，视图窗口的已经滚动到底了，才可以执行`LayoutViewPort` 的滚动
* `VisualViewPort`： 视图窗口，也就是可视窗口，当通过触控板用双指放大时，"内容"就被放大了，就形成了`ScrollableArea`
* `LayoutViewPort`： LayoutView和它的ScrollableArea，每个html文档对应一个`#document` 元素(对应js的document对象)，`#document` 的唯一子元素就是html元素。 LayoutView是`#document` 元素的`LayoutNode` ，当html的宽高超出了`#document` 的宽高，`LayoutView` 就形成了`ScrollableArea` 

# RootFrameViewport/Page之间的关系

![](https://cdn.nlark.com/yuque/0/2024/jpeg/1438873/1733061722489-c2167dec-c4be-44de-bc21-642520fa01d4.jpeg)

**layout_viewport一般是Document所对应的PaintLayerScrollableArea**

# 初始化

共有两个路径初始化RouteFrameViewPort

## 路径一

初始化Document的时候会初始化RootFrameViewport

```cpp
void LocalFrameView::InitializeRootScroller() {
  Page* page = frame_->GetPage();
  VisualViewport& visual_viewport = frame_->GetPage()->GetVisualViewport();


  ScrollableArea* layout_viewport = LayoutViewport();

  // This method may be called multiple times during loading. If the root
  // scroller is already initialized this call will be a no-op.
  if (viewport_scrollable_area_)
    return;

  auto* root_frame_viewport = MakeGarbageCollected<RootFrameViewport>(
      visual_viewport, *layout_viewport);
  viewport_scrollable_area_ = root_frame_viewport;

  page->GlobalRootScrollerController().Initialize(*root_frame_viewport,
                                                  *frame_->GetDocument());
}

void LocalFrameView::DidAttachDocument() {
  Page* page = frame_->GetPage();

  VisualViewport& visual_viewport = page->GetVisualViewport();

  if (frame_->IsMainFrame() && visual_viewport.IsActiveViewport()) {
    // If this frame is provisional it's not yet the Page's main frame. In that
    // case avoid creating a root scroller as it has Page-global effects; it
    // will be initialized when the frame becomes the Page's main frame.
    if (!frame_->IsProvisional())
      // 初始化RootFrame
      InitializeRootScroller();
  }

  if (frame_->IsMainFrame()) {
    // Allow for commits to be deferred because this is a new document.
    have_deferred_main_frame_commits_ = false;
  }
}

void Document::Initialize() {
  UpdateForcedColors();
  const ComputedStyle* style = GetStyleResolver().StyleForViewport();
  layout_view_ = MakeGarbageCollected<LayoutView>(this);
  SetLayoutObject(layout_view_);

  layout_view_->SetStyle(style);

  AttachContext context;
  AttachLayoutTree(context);

  // The TextAutosizer can't update layout view info while the Document is
  // detached, so update now in case anything changed.
  if (TextAutosizer* autosizer = GetTextAutosizer())
    autosizer->UpdatePageInfo();

  GetFrame()->DidAttachDocument();
  lifecycle_.AdvanceTo(DocumentLifecycle::kStyleClean);

  if (View())
    // 初始化RootFrameView
    View()->DidAttachDocument();
}
```

### 调用栈

![](https://cdn.nlark.com/yuque/0/2024/png/1438873/1723385708523-2e8e7933-3863-4771-958c-6c0639131af1.png)

![](https://cdn.nlark.com/yuque/0/2024/png/1438873/1723385846814-104a15e8-0299-4fe4-81c1-eeaed420afd3.png)

## 路径二

createMainFrame之后创建

```cpp
void LocalFrameView::InitializeRootScroller() {
  Page* page = frame_->GetPage();
  VisualViewport& visual_viewport = frame_->GetPage()->GetVisualViewport();


  ScrollableArea* layout_viewport = LayoutViewport();

  // This method may be called multiple times during loading. If the root
  // scroller is already initialized this call will be a no-op.
  if (viewport_scrollable_area_)
    return;

  auto* root_frame_viewport = MakeGarbageCollected<RootFrameViewport>(
      visual_viewport, *layout_viewport);
  viewport_scrollable_area_ = root_frame_viewport;

  DCHECK(frame_->GetDocument());
  page->GlobalRootScrollerController().Initialize(*root_frame_viewport,
                                                  *frame_->GetDocument());
}

RenderFrameImpl* RenderFrameImpl::CreateMainFrame(
    AgentSchedulingGroup& agent_scheduling_group,
    blink::WebView* web_view,
    blink::WebFrame* opener,
    bool is_for_nested_main_frame,
    bool is_for_scalable_page,
    blink::mojom::FrameReplicationStatePtr replication_state,
    const base::UnguessableToken& devtools_frame_token,
    mojom::CreateLocalMainFrameParamsPtr params,
    const blink::WebURL& base_url) {
  // A main frame RenderFrame must have a RenderWidget.
  DCHECK_NE(MSG_ROUTING_NONE, params->widget_params->routing_id);

  RenderFrameImpl* render_frame = RenderFrameImpl::Create(
      agent_scheduling_group, params->frame_token, params->routing_id,
      std::move(params->frame),
      std::move(params->associated_interface_provider_remote),
      devtools_frame_token, is_for_nested_main_frame);

  // The WebFrame created here was already attached to the Page as its main
  // frame, and the WebFrameWidget has been initialized, so we can call
  // WebView's DidAttachLocalMainFrame().
  render_frame->GetWebView()->DidAttachLocalMainFrame();

  // The WebFrameWidget should start with valid VisualProperties, including a
  // non-zero size. While WebFrameWidget would not normally receive IPCs and
  // thus would not get VisualProperty updates while the frame is provisional,
  // we need at least one update to them in order to meet expectations in the
  // renderer, and that update comes as part of the CreateFrame message.
  // TODO(crbug.com/40387047): This could become part of WebFrameWidget Init.
  web_frame_widget->ApplyVisualProperties(
      params->widget_params->visual_properties);

  render_frame->in_frame_tree_ = true;
  render_frame->Initialize(nullptr);


}

void WebViewImpl::DidAttachLocalMainFrame() {
  DCHECK(MainFrameImpl());
  DCHECK(!remote_main_frame_host_remote_);

  LocalFrame* local_frame = MainFrameImpl()->GetFrame();

  // ... 

    
  // It's possible that at the time that `local_frame` attached its document it
  // was provisional so it couldn't initialize the root scroller. Try again now
  // that the frame has been attached; this is a no-op if the root scroller is
  // already initialized.
  if (viewport.IsActiveViewport()) {
    DCHECK(local_frame->GetDocument());
    // DidAttachLocalMainFrame can be called before a new document is attached
    // so ensure we don't try to initialize the root scroller on a stopped
    // document.
    if (local_frame->GetDocument()->IsActive())
      // 初始化RootFrame
      local_frame->View()->InitializeRootScroller();
  }
}


```

### 调用栈

![](https://cdn.nlark.com/yuque/0/2024/png/1438873/1723565307117-8e6e772e-af2b-41b6-94ae-46a3dd36694e.png)

# VisualViewport的初始化

在初始化Page的时候就初始化了`VisualViewport`

```cpp
Page::Page(base::PassKey<Page>,
           ChromeClient& chrome_client,
           AgentGroupScheduler& agent_group_scheduler,
           const BrowsingContextGroupInfo& browsing_context_group_info,
           const ColorProviderColorMaps* color_provider_colors,
           bool is_ordinary)
    : SettingsDelegate(std::make_unique<Settings>()),
      main_frame_(nullptr),
      agent_group_scheduler_(agent_group_scheduler),
      animator_(MakeGarbageCollected<PageAnimator>(*this)),
      autoscroll_controller_(MakeGarbageCollected<AutoscrollController>(*this)),
      chrome_client_(&chrome_client),
      drag_caret_(MakeGarbageCollected<DragCaret>()),
      drag_controller_(MakeGarbageCollected<DragController>(this)),
      focus_controller_(MakeGarbageCollected<FocusController>(this)),
      context_menu_controller_(
          MakeGarbageCollected<ContextMenuController>(this)),
      page_scale_constraints_set_(
          MakeGarbageCollected<PageScaleConstraintsSet>(this)),
      pointer_lock_controller_(
          MakeGarbageCollected<PointerLockController>(this)),
      browser_controls_(MakeGarbageCollected<BrowserControls>(*this)),
      console_message_storage_(MakeGarbageCollected<ConsoleMessageStorage>()),
      global_root_scroller_controller_(
          MakeGarbageCollected<TopDocumentRootScrollerController>(*this)),
      // 初始化visual_viewport_
      visual_viewport_(MakeGarbageCollected<VisualViewport>(*this)),
...{....}
```

# LayoutViewPort的初始化

layoutViewport是Document的`PaintLayerScrollableArea`

```cpp
PaintLayerScrollableArea* LocalFrameView::LayoutViewport() const {
  auto* layout_view = GetLayoutView();
  return layout_view ? layout_view->GetScrollableArea() : nullptr;
}

void Document::Initialize() {

  UpdateForcedColors();
  const ComputedStyle* style = GetStyleResolver().StyleForViewport();
  // 初始化layoutViewport
  layout_view_ = MakeGarbageCollected<LayoutView>(this);
  SetLayoutObject(layout_view_);

  layout_view_->SetStyle(style);

  AttachContext context;
  AttachLayoutTree(context);

  // The TextAutosizer can't update layout view info while the Document is
  // detached, so update now in case anything changed.
  if (TextAutosizer* autosizer = GetTextAutosizer())
    autosizer->UpdatePageInfo();

  GetFrame()->DidAttachDocument();
  lifecycle_.AdvanceTo(DocumentLifecycle::kStyleClean);

  if (View())
    View()->DidAttachDocument();
}
```