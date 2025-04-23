
## 引言

在现代前端开发中，构建工具扮演着至关重要的角色。今天，我们将深入探讨 `@lynx-js/rspeedy` 包中的 `createRspeedy` 函数，这是一个强大的构建工具实例创建函数，它为开发者提供了灵活且高效的前端应用构建能力。
## createRspeedy 函数概述
`createRspeedy` 函数是 Rspeedy 构建工具的入口点，它允许开发者创建一个 Rspeedy 实例，并通过该实例来自定义构建或开发过程。这个函数设计简洁而强大，为开发者提供了丰富的配置选项。
```typescript
export async function createRspeedy(
  { cwd = process.cwd(), rspeedyConfig = {} }: CreateRspeedyOptions,
): Promise<RspeedyInstance> {
  // 函数实现...
}
```
## 函数参数解析
`createRspeedy` 接受一个配置对象，包含以下参数：
1. **cwd**（可选）：指定当前构建的根路径，默认为 `process.cwd()`，即当前工作目录。
2. **rspeedyConfig**（可选）：Rspeedy 的配置对象，允许开发者自定义构建行为。
## 函数实现流程

让我们逐步分析 `createRspeedy` 的实现流程：
### 1. 应用默认配置

```typescript
const config = applyDefaultRspeedyConfig(rspeedyConfig)
```

首先，函数会调用 `applyDefaultRspeedyConfig` 来处理用户提供的配置，确保所有必要的默认值都被正确设置。这是一种常见的配置处理模式，可以减轻用户的配置负担，同时保证系统的稳定性。

### 2. 创建底层 Rsbuild 实例

```typescript
const [rspeedy, { applyDefaultPlugins }] = await Promise.all([
  createRsbuild({
    cwd,
    rsbuildConfig: toRsbuildConfig(config) as RsbuildConfig,
  }),
  import('./plugins/index.js'),
])
```

这里使用了 `Promise.all` 来并行执行两个异步操作：
- 创建底层的 Rsbuild 实例，并将转换后的配置传递给它
- 动态导入插件系统

### 3. 应用默认插件

```typescript
await applyDefaultPlugins(rspeedy, config)
```

创建实例后，函数会应用一系列默认插件。这些插件为 Rspeedy 提供了核心功能，如资源处理、代码转换等。
### 4. 扩展实例功能
```typescript
const inspectConfig = rspeedy.inspectConfig.bind(rspeedy)

return Object.assign(rspeedy, {
  getRspeedyConfig: () => config,
  async inspectConfig(options: InspectConfigOptions): Promise<InspectConfigResult> {
    // 扩展的 inspectConfig 实现...
  },
})
```

最后，函数扩展了原始的 Rsbuild 实例，添加了两个新方法：
- `getRspeedyConfig`：返回当前的 Rspeedy 配置
- `inspectConfig`：增强版的配置检查方法，除了执行原始的 inspectConfig 外，还会生成 Rspeedy 特定的配置文件

## 使用示例

```typescript
import { createRspeedy } from '@lynx-js/rspeedy'

void async function () {
  // 创建 Rspeedy 实例
  const rspeedy = await createRspeedy({
    // 自定义配置
    rspeedyConfig: {
      // 配置选项...
    }
  })
  
  // 使用实例进行构建
  await rspeedy.build()
}()
```

## applyDefaultPlugins
### 核心默认插件

函数首先定义了一组核心默认插件，这些插件提供了构建过程中的基础功能：
```typescript
const defaultPlugins = Object.freeze<Promise<RsbuildPlugin>[]>([
    import('./api.plugin.js').then(({ pluginAPI }) => pluginAPI(config)),

    import('./chunkLoading.plugin.js').then(({ pluginChunkLoading }) =>
      pluginChunkLoading()
    ),

    import('./minify.plugin.js').then(({ pluginMinify }) =>
      pluginMinify(config.output?.minify)
    ),

    import('./optimization.plugin.js').then(({ pluginOptimization }) =>
      pluginOptimization()
    ),

    import('./output.plugin.js').then(({ pluginOutput }) =>
      pluginOutput(config.output)
    ),

    import('./resolve.plugin.js').then(({ pluginResolve }) => pluginResolve()),

    import('./rsdoctor.plugin.js').then(({ pluginRsdoctor }) =>
      pluginRsdoctor(config.tools?.rsdoctor)
    ),

    import('./sourcemap.plugin.js').then(({ pluginSourcemap }) =>
      pluginSourcemap()
    ),

    import('./swc.plugin.js').then(({ pluginSwc }) => pluginSwc()),

    import('./target.plugin.js').then(({ pluginTarget }) => pluginTarget()),
  ])
```
#### api
主要是expose了一些方法
```typescript
export function pluginAPI(config: Config): RsbuildPlugin {
  return {
    name: 'lynx:rsbuild:plugin-api',
    setup(api) {
      api.expose<ExposedAPI>(sAPI, {
        config,
        debug,
        async exit(code) {
          const { exit } = await import('../cli/exit.js')
          return exit(code)
        },
        logger,
        version,
      })
    },
  }
}
```

#### chunkloading
添加`ChunkLoadingWebpackPlugin`插件，并修改output的`chunkLoading` 和`chunkFormat` 配置
```typescript
export function pluginChunkLoading(): RsbuildPlugin {
  return {
    name: 'lynx:rsbuild:chunk-loading',
    setup(api) {
      api.modifyBundlerChain(chain => {
        // dprint-ignore
        chain
          .plugin('lynx:chunk-loading')
            .use(ChunkLoadingWebpackPlugin)
          .end()
          .output
            // Rspack needs `chunkLoading: 'require'` since we use runtimeModule hook
            // to override the chunk loading runtime.
            .chunkLoading('require')
            .chunkFormat('commonjs')
            .iife(false)
          .end()
      })

      api.modifyWebpackChain(chain => {
        chain
          .output
          // For webpack, we directly use `chunkLoading: 'lynx'`.
          .chunkLoading('lynx')
          .end()
      })
    },
  }
}
```
[ChunkLoadingWebpackPlugin实现](./ChunkLoadingRspackPluginImpl)


#### 剩余的插件
剩余的插件都是修改rsbuild或者rspack相关的配置，不多赘述