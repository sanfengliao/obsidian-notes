
## ChunkLoadingRspackPluginImpl 概述

`ChunkLoadingRspackPluginImpl` 是 `ChunkLoadingRspackPlugin` 的内部实现类，主要负责：

1. 重写 Rspack 默认的 chunk 加载运行时代码
2. 添加 Lynx 特定的 chunk 加载逻辑
3. 处理 JavaScript 和 CSS 资源的异步加载
4. 支持热模块替换（HMR）功能

```typescript
export class ChunkLoadingRspackPluginImpl {
  name = 'ChunkLoadingRspackPlugin';

  constructor(compiler: Compiler, _options: ChunkLoadingRspackPluginOptions) {
    // 实现逻辑...
  }
  
  // 私有方法...
}
```

## 工作原理详解

### 1. 初始化与钩子注册

在构造函数中，插件通过 Rspack 的钩子系统注册了多个回调函数：

```typescript
constructor(compiler: Compiler, _options: ChunkLoadingRspackPluginOptions) {
  compiler.hooks.thisCompilation.tap(this.name, (compilation) => {
    // 注册各种钩子...
  });
}
```

这里使用了 `thisCompilation` 钩子，它在每次编译开始时触发，允许插件访问和修改当前编译过程。

### 2. 添加运行时需求

```typescript
compilation.hooks.additionalTreeRuntimeRequirements.tap(
  this.name,
  (_, set) => {
    set.add(compiler.webpack.RuntimeGlobals.exports);
    set.add(compiler.webpack.RuntimeGlobals.publicPath);
  },
);
```

这段代码向运行时需求集合中添加了两个全局变量：
- `exports`：用于模块导出
- `publicPath`：用于资源路径解析

这确保了这些全局变量在运行时可用，是异步加载资源的基础。

### 3. 注册 chunk 处理器依赖

```typescript
compilation.hooks.runtimeRequirementInTree.for(
  compiler.webpack.RuntimeGlobals.ensureChunkHandlers
).tap(this.name, (_, runtimeRequirements) => {
  runtimeRequirements.add(RuntimeGlobals.lynxAsyncChunkIds);
});
```

这段代码在 `ensureChunkHandlers`（确保 chunk 处理器）的运行时需求中添加了 `lynxAsyncChunkIds` 依赖，这是 Lynx 框架中用于处理异步 chunk 的特殊全局变量。

### 4. 重写运行时模块

插件的核心功能是重写默认的运行时模块，这通过 `runtimeModule` 钩子实现：

```typescript
compilation.hooks.runtimeModule.tap(this.name, (runtimeModule, chunk) => {
  if (runtimeModule.name === 'require_chunk_loading') {
    this.#overrideChunkLoadingRuntimeModule(runtimeModule);
    if (compiler.options.mode === 'development') {
      this.#overrideHMRChunkLoadingRuntimeModule(runtimeModule);
    }
  } else if (
    runtimeModule.name === 'css loading'
    && compiler.options.mode === 'development'
  ) {
    this.#overrideCSSChunkLoadingRuntimeModule(
      compilation,
      runtimeModule,
      chunk,
    );
  }
});
```

这里根据运行时模块的名称进行不同处理：
- 对于 `require_chunk_loading` 模块，重写 JavaScript chunk 加载逻辑
- 在开发模式下，额外添加 HMR 支持
- 对于 `css loading` 模块，重写 CSS chunk 加载逻辑（目前仅在开发模式下）

### 5. JavaScript Chunk 加载实现

```typescript
#overrideChunkLoadingRuntimeModule(runtimeModule: RuntimeModule) {
  runtimeModule.source!.source = Buffer.concat([
    Buffer.from(runtimeModule.source!.source),
    Buffer.from('\n'),
    Buffer.from('\n'),
    // withLoading
    Buffer.from(JavaScriptRuntimeModule.generateChunkLoadingRuntime('true')),
    Buffer.from('\n'),
    // withOnChunkLoad
    Buffer.from(JavaScriptRuntimeModule.generateChunkOnloadRuntime()),
  ]);
}
```

这个方法通过以下步骤重写 JavaScript chunk 加载逻辑：
1. 保留原始运行时代码
2. 添加 Lynx 特定的 chunk 加载运行时代码
3. 添加 chunk 加载完成后的回调处理

### 6. HMR 支持实现

```typescript
#overrideHMRChunkLoadingRuntimeModule(runtimeModule: RuntimeModule) {
  runtimeModule.source!.source = Buffer.concat([
    Buffer.from(runtimeModule.source!.source),
    // withHmr
    Buffer.from(JavaScriptRuntimeModule.generateHMRRuntime()),
    // withHmrManifest
    Buffer.from(JavaScriptRuntimeModule.generateHMRManifestRuntime()),
  ]);
}
```

在开发模式下，这个方法添加了 HMR 相关的运行时代码：
1. HMR 基础运行时，处理模块热替换逻辑
2. HMR 清单运行时，处理模块依赖关系和更新传播

### 7. CSS Chunk 加载实现

```typescript
#overrideCSSChunkLoadingRuntimeModule(
  compilation: Compilation,
  runtimeModule: RuntimeModule,
  chunk: Chunk,
) {
  const chunkMap = getCssChunkObject(chunk, compilation);
  runtimeModule.source!.source = Buffer.concat([
    Buffer.from('\n'),
    Buffer.from(CssRuntimeModule.generateLoadStyleRuntime()),
    Buffer.from('\n'),
    // withLoading
    Buffer.from(CssRuntimeModule.generateChunkLoadingRuntime(
      chunkMap,
      chunk.ids ?? [],
    )),
    Buffer.from('\n'),
    // withHmr
    Buffer.from(CssRuntimeModule.generateHMRLoadChunkRuntime()),
    Buffer.from('\n'),
  ]);
}
```

这个方法实现了 CSS chunk 的加载逻辑：
1. 生成 CSS 样式加载函数
2. 根据当前 chunk 信息生成 CSS chunk 加载运行时
3. 添加 CSS 的 HMR 支持



