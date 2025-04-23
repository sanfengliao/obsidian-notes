# pluginReactLynx 插件详解：Lynx 框架的 React 集成方案

`pluginReactLynx` 是 Lynx 框架中用于集成 React 的 Rsbuild 插件，它为开发者提供了一套完整的工具链，使 React 应用能够在 Lynx 环境中高效运行。本文将详细解析这个插件的作用及其内部实现。

## 插件概述

`pluginReactLynx` 函数接收用户配置选项，返回一个 Rsbuild 插件对象，该插件负责配置构建环境，使 React 应用能够被编译为 Lynx 可执行的格式。

```typescript
export function pluginReactLynx(
  userOptions?: PluginReactLynxOptions,
): RsbuildPlugin {
  // 函数实现...
}
```

## 配置验证与合并

函数首先验证用户配置，然后将用户配置与默认配置合并：

```typescript
validateConfig(userOptions)

const engineVersion = userOptions?.engineVersion
  ?? userOptions?.targetSdkVersion ?? '3.2'

const defaultOptions: Required<PluginReactLynxOptions> = {
  // 默认配置...
}

const resolvedOptions = Object.assign(defaultOptions, userOptions, {
  // 使用 engineVersion 覆盖默认值
  targetSdkVersion: engineVersion,
  engineVersion,
})
```

这里的 `validateConfig` 函数用于检查用户配置的有效性，确保不会出现配置错误。

## 插件结构

插件对象包含名称、前置依赖和设置函数：

```typescript
return {
  name: 'lynx:react',
  pre: ['lynx:rsbuild:plugin-api'],
  async setup(api) {
    // 设置函数实现...
  },
}
```

`pre` 字段指定了该插件依赖的其他插件，确保在正确的顺序执行。

## 内部功能模块详解

`setup` 函数内部调用了多个 `applyXxx` 函数，每个函数负责配置构建过程的特定方面：

### 1. applyAlias

```typescript
await applyAlias(api, resolvedOptions.experimental_isLazyBundle)
```

**作用**：
- 配置模块解析别名（alias）
- 简化导入路径
- 根据是否为懒加载包设置不同的别名

**具体实现**：
- 为 React、ReactDOM 等库设置别名，指向 Lynx 特定版本
- 配置 `@lynx-js/react` 相关包的别名
- 处理mainthread和background不同环境的别名
```typescript
export function pluginReactAlias(options: Options): RsbuildPlugin {
  const { LAYERS, lazy, rootPath } = options ?? {}

  return {
    name: 'lynx:react-alias',
    setup(api) {
      const hasAlias = api.useExposed<boolean>(S_PLUGIN_REACT_ALIAS)
      if (hasAlias) {
        // We make sure that only make aliased once
        return
      }
      api.expose(S_PLUGIN_REACT_ALIAS, true)

      const require = createRequire(import.meta.url)

      const reactLynxDir = path.dirname(
        require.resolve('@lynx-js/react/package.json', {
          paths: [rootPath ?? api.context.rootPath],
        }),
      )
      const resolve = createLazyResolver(
        reactLynxDir,
        lazy ? ['lazy', 'import'] : ['import'],
      )
      api.modifyBundlerChain(async chain => {
        // FIXME(colinaaa): use `Promise.all`
        const jsxRuntime = {
          background: await resolve('@lynx-js/react/jsx-runtime'),
          mainThread: await resolve('@lynx-js/react/lepus/jsx-runtime'),
        }
        const jsxDevRuntime = {
          background: await resolve('@lynx-js/react/jsx-dev-runtime'),
          mainThread: await resolve('@lynx-js/react/lepus/jsx-dev-runtime'),
        }
        const reactLepus = {
          background: await resolve('@lynx-js/react'),
          mainThread: await resolve('@lynx-js/react/lepus'),
        }

        // dprint-ignore
        chain
          .module
            .rule('react:jsx-runtime:main-thread')
              .issuerLayer(LAYERS.MAIN_THREAD)
              .resolve
                .alias
                  .set('react/jsx-runtime', jsxRuntime.mainThread)
                  .set('react/jsx-dev-runtime', jsxDevRuntime.mainThread)
                  .set('@lynx-js/react/jsx-runtime', jsxRuntime.mainThread)
                  .set('@lynx-js/react/jsx-dev-runtime', jsxDevRuntime.mainThread)
                  .set('@lynx-js/react/lepus$', reactLepus.mainThread)
                  .set('@lynx-js/react/lepus/jsx-runtime', jsxRuntime.mainThread)
                  .set('@lynx-js/react/lepus/jsx-dev-runtime', jsxDevRuntime.mainThread)
                .end()
              .end()
            .end()
            .rule('react:jsx-runtime:background')
            .issuerLayer(LAYERS.BACKGROUND)
              .resolve
                .alias
                  .set('react/jsx-runtime', jsxRuntime.background)
                  .set('react/jsx-dev-runtime', jsxDevRuntime.background)
                  .set('@lynx-js/react/jsx-runtime', jsxRuntime.background)
                  .set('@lynx-js/react/jsx-dev-runtime', jsxDevRuntime.background)
                  .set('@lynx-js/react/lepus$', reactLepus.background)
                .end()
              .end()
            .end()
          .end()

        // react-transform may add imports of the following entries
        // We need to add aliases for that
        const transformedEntries = [
          'internal',
          'legacy-react-runtime',
          'runtime-components',
          'worklet-runtime/bindings',
        ]

        await Promise.all(
          transformedEntries
            .map(entry => `@lynx-js/react/${entry}`)
            .map(entry =>
              resolve(entry).then(value => {
                chain
                  .resolve
                  .alias
                  .set(`${entry}$`, value)
              })
            ),
        )

        chain
          .resolve
          .alias
          .set(
            'react$',
            reactLepus.background,
          )
          .set(
            '@lynx-js/react$',
            reactLepus.background,
          )

        const preactEntries = [
          'preact',
          'preact/compat',
          'preact/debug',
          'preact/devtools',
          'preact/hooks',
          'preact/test-utils',
          'preact/jsx-runtime',
          'preact/jsx-dev-runtime',
          'preact/compat',
          'preact/compat/client',
          'preact/compat/server',
          'preact/compat/jsx-runtime',
          'preact/compat/jsx-dev-runtime',
          'preact/compat/scheduler',
        ]
        await Promise.all(
          preactEntries.map(entry =>
            resolve(entry).then(value => {
              chain
                .resolve
                .alias
                .set(`${entry}$`, value)
            })
          ),
        )
      })
    },
  }
}
```

### 2. [applyCSS](./applyCss)

```typescript
applyCSS(api, resolvedOptions)
```
`applyCSS` 函数是 Lynx 框架中处理 CSS 的核心组件，它通过精细的配置和优化，实现了 Lynx 环境中的 CSS 模块化、作用域控制和选择器增强等特性。函数采用分层处理策略，为主线程和后台线程提供不同的 CSS 处理逻辑，并通过自定义插件和加载器，确保 CSS 在 Lynx 环境中高效运行。


### 3.[ applyEntry](./applyEntry)

```typescript
applyEntry(api, resolvedOptions)
```
applyEntry是 Lynx Stack 项目中 RSpeedy 构建工具的 React 插件核心入口文件。该文件主要负责配置 React 应用在 Lynx 环境下的构建过程，处理双线程架构（主线程和后台线程）的入口点设置。

### 4. applyBackgroundOnly

```typescript
applyBackgroundOnly(api)
```

 applyBackgroundOnly函数是 Lynx 框架中实现线程隔离的关键组件，它通过巧妙的别名配置和特殊加载器，确保 background-only 模块只能在后台线程中使用。这种机制有效地防止了开发者在主线程中误用后台线程专用的功能，从而提高了应用的稳定性和性能。
```
导入 'background-only'
        |
        v
检查导入位置的层级
        |
        +-----------------+
        |                 |
        v                 v
    主线程           后台线程
        |                 |
        v                 v
重定向到 error.js    重定向到 empty.js
        |                 |
        v                 v
触发错误加载器       正常加载模块
        |                 |
        v                 v
构建失败并显示错误   构建成功
```


```typescript
export function applyBackgroundOnly(
  api: RsbuildPluginAPI,
): void {
  const __dirname = path.dirname(fileURLToPath(import.meta.url))

  api.modifyBundlerChain(chain => {
    chain
      .module
      .rule(ALIAS_BACKGROUND_ONLY_MAIN)
      .issuerLayer(LAYERS.MAIN_THREAD)
      .resolve
      .alias
      .set(
        'background-only$',
        path.resolve(__dirname, 'background-only', 'error.js'),
      )

    chain
      .module
      .rule(ALIAS_BACKGROUND_ONLY_BACKGROUND)
      .issuerLayer(LAYERS.BACKGROUND)
      .resolve
      .alias
      .set(
        'background-only$',
        path.resolve(__dirname, 'background-only', 'empty.js'),
      )

    chain
      .module
      .rule(DETECT_IMPORT_ERROR)
      .test(path.resolve(__dirname, 'background-only', 'error.js'))
      .issuerLayer(LAYERS.MAIN_THREAD)
      .use(DETECT_IMPORT_ERROR)
      .loader(path.resolve(__dirname, 'loaders/invalid-import-error-loader'))
      .options({
        message:
          '\'background-only\' cannot be imported from a main-thread module.',
      })
  })
}

```

### 5. [applyLoaders](./applyLoaders)

```typescript
applyLoaders(api, resolvedOptions)
```
`applyLoaders` 函数是 Lynx 框架中处理代码转换的核心组件，它通过配置专用的加载器和优化选项，确保 React 代码能够正确地转换为 Lynx 环境可执行的格式。函数采用线程分离的策略，为主线程和后台线程提供不同的处理逻辑，并支持多种代码优化技术，如代码摇树和死代码消除。

### 6. applyRefresh

```typescript
applyRefresh(api)
```

**作用**：
- 配置 React 热更新（Hot Module Replacement）
- 提升开发体验

**具体实现**：
- 设置 React Refresh 插件
- 配置热更新相关的 webpack 插件
- 处理热更新边界和状态保持

### 7. applySplitChunksRule

```typescript
applySplitChunksRule(api)
```
applySplitChunksRule 函数通过精细的代码分割配置，优化了 Lynx 应用的包体积和加载性能。它默认采用 all-in-one 策略以简化构建，同时支持更复杂的 split-by-experience 策略以优化大型应用。最重要的是，它通过防止主线程代码被分割，保证了 Lynx 应用的性能和稳定性。
#### 主要功能

1. **设置默认分割策略**：
   ```typescript
   api.modifyRsbuildConfig((config, { mergeRsbuildConfig }) => {
     const userConfig = api.getRsbuildConfig('original')
     if (!userConfig.performance?.chunkSplit?.strategy) {
       return mergeRsbuildConfig(config, {
         performance: {
           chunkSplit: {
             strategy: 'all-in-one',
           },
         },
       })
     }
     return config
   })
   ```
   如果用户没有指定分割策略，默认使用 `all-in-one` 策略，将所有代码打包到一个文件中。

2. **配置体验优化分割**：
   ```typescript
   if (config.performance.chunkSplit.strategy !== 'split-by-experience') {
     return
   }
   
   // 配置额外的缓存组...
   extraGroups['preact'] = {
     name: 'lib-preact',
     test: /node_modules[\\/](.*?[\\/])?(?:preact|preact[\\/]compat|preact[\\/]hooks|preact[\\/]jsx-runtime)[\\/]/,
     priority: 0,
   }
   ```
   当使用 `split-by-experience` 策略时，添加特定的缓存组，如将 Preact 相关库分割到单独的 chunk 中。

3. **保护主线程代码**：
   ```typescript
   rspackConfig.optimization.splitChunks.chunks = function chunks(chunk) {
     // 不对主线程代码进行分割
     return !chunk.name?.includes('__main-thread')
   }
   ```
通过自定义 `chunks` 函数，确保主线程代码不会被分割，这对于 Lynx 的性能至关重要。

### 8. applySWC

```typescript
applySWC(api)
export function applySWC(api: RsbuildPluginAPI): void {
  api.modifyRsbuildConfig((config, { mergeRsbuildConfig }) => {
    return mergeRsbuildConfig(config, {
      tools: {
        swc(config) {
          config.jsc ??= {}
          config.jsc.transform ??= {}
          config.jsc.transform.useDefineForClassFields = false
          config.jsc.transform.optimizer ??= {}
          config.jsc.transform.optimizer.simplify = true

          config.jsc.parser ??= {
            syntax: 'typescript',
          }
          if (config.jsc.parser.syntax === 'typescript') {
            config.jsc.parser.tsx = false
            config.jsc.parser.decorators = true
          }

          return config
        },
      },
    })
  })
}

```

applySWC 函数调整 SWC 编译器的配置，优化 Lynx 应用的 JavaScript/TypeScript 编译过程。
### 9. 其他配置

```typescript
api.modifyRsbuildConfig((config, { mergeRsbuildConfig }) => {
  // 修改 Rsbuild 配置...
})

if (resolvedOptions.experimental_isLazyBundle) {
  applyLazy(api)
}
```

**作用**：
- 修改 Rsbuild 基础配置
- 处理特殊的构建场景（如懒加载包）

**具体实现**：
- 将 `@lynx-js/react` 包添加到 include 列表
- 配置源文件包含规则
- 应用懒加载特定的配置（如果启用）

### 10. 调试信息

```typescript
const rspeedyAPIs = api.useExposed<ExposedAPI>(Symbol.for('rspeedy.api'))!
const require = createRequire(import.meta.url)
const { version } = require('../package.json') as { version: string }

rspeedyAPIs.debug(() => {
  const webpackPluginPath = require.resolve('@lynx-js/react-webpack-plugin')
  return `Using @lynx-js/react-webpack-plugin v${version} at ${webpackPluginPath}`
})
```

**作用**：
- 提供调试信息
- 记录使用的插件版本和路径

**具体实现**：
- 获取插件版本信息
- 记录 webpack 插件路径
- 通过 debug API 输出信息

## 配置选项详解

`pluginReactLynx` 支持丰富的配置选项，包括：

1. **基础配置**：
   - `engineVersion`/`targetSdkVersion`：指定所需的 Lynx 引擎版本
   - `experimental_isLazyBundle`：生成独立的懒加载包

2. **CSS 相关配置**：
   - `enableCSSInheritance`：启用 CSS 继承
   - `customCSSInheritanceList`：自定义可继承的 CSS 属性
   - `enableCSSSelector`：启用新的 CSS 选择器实现
   - `enableCSSInvalidation`：启用 CSS 失效机制
   - `enableRemoveCSSScope`：控制 CSS 作用域

3. **渲染相关配置**：
   - `pipelineSchedulerConfig`：渲染管道调度配置
   - `enableParallelElement`：启用并行元素解析
   - `firstScreenSyncTiming`：首屏同步时机

4. **代码转换配置**：
   - `jsx`：JSX 转换配置
   - `shake`：代码摇树配置
   - `defineDCE`：编译时定义和死代码消除

5. **兼容性配置**：
   - `compat`：与 ReactLynx 2.0 的兼容性选项
   - `defaultDisplayLinear`：默认显示模式
