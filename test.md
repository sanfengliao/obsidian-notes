`IntersectionObserver` 的核心能力是判断一个元素与另一个元素（或 viewport）的交集比例。听起来简单——两个矩形求交集嘛。但实际实现要处理的问题远比这复杂：元素可能有 CSS transform、可能嵌套在多层滚动容器里、可能跨 iframe、可能是 SVG……

Chromium 把这些计算逻辑封装在 `IntersectionGeometry::ComputeGeometry()` 里。这个方法承担了所有实质性的工作：初始化边界框、应用 margin、执行坐标映射、裁剪交集、判定阈值、计算可见性。从工程角度看，这里是性能瓶颈所在，也是复杂 bug 的温床——坐标空间的转换、缓存的失效条件、各种标志位的组合，任何一个环节出错都会导致难以追踪的视觉问题。

这篇文章按 `ComputeGeometry()` 的执行流程逐步拆解，看看它如何在三个坐标空间之间腾挪，最终算出那个看似简单的交集比例。坐标系统的三层转换

整个流程涉及三个坐标空间：

- 本地坐标系：元素自身的坐标系
- 根容器坐标系：target 几何映射到 root 坐标空间
- 绝对坐标系：相对于 viewport 或顶级文档

![image.png](attachment:5e192295-1734-40cb-93aa-c7decd9a88c1:image.png)