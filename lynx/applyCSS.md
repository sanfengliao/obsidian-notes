`applyCSS` 函数是 `pluginReactLynx` 插件中的核心组件之一，负责配置和优化 Lynx 应用中的 CSS 处理流程。这个函数实现了多种 CSS 相关特性，包括 CSS 模块化、作用域控制、选择器优化等。下面将详细解析其工作原理和实现细节。
## 函数概述

```typescript
export function applyCSS(
  api: RsbuildPluginAPI,
  options: Required<PluginReactLynxOptions>,
): void {
  // 函数实现...
}
```
`applyCSS` 接收两个参数：
- `api`：Rsbuild 插件 API，用于修改构建配置
- `options`：插件选项，包含 CSS 相关的配置项
## 关键配置提取

函数首先从选项中提取关键的 CSS 配置：

```typescript
const {
  enableRemoveCSSScope,
  enableCSSSelector,
  targetSdkVersion,
} = options
```

这些配置项决定了 CSS 处理的行为：
- `enableRemoveCSSScope`：控制 CSS 作用域行为
- `enableCSSSelector`：启用新的 CSS 选择器实现
- `targetSdkVersion`：目标 Lynx 引擎版本

## 禁用样式注入

```typescript
api.modifyRsbuildConfig((config, { mergeRsbuildConfig }) => {
  return mergeRsbuildConfig(config, {
    output: { injectStyles: false },
  })
})
```

这段代码修改 Rsbuild 配置，设置 `output.injectStyles` 为 `false`，这会产生以下效果：
1. 禁用 `style-loader`（不会将 CSS 注入到 DOM）
2. 启用 `CssExtractRspackPlugin`（将 CSS 提取为独立文件）
3. 禁用 `experiment.css`（Rspack 的实验性 CSS 处理）

这是因为 Lynx 环境中，CSS 需要以特定方式处理，而不是像普通 Web 应用那样注入到 DOM。

## 修改打包链配置

函数的核心是通过 `modifyBundlerChain` 方法修改 webpack/rspack 的配置链：

```typescript
api.modifyBundlerChain(
  async function handler(chain, { CHAIN_ID, environment }) {
    // 实现...
  }
)
```

### 1. 导入和准备 CSS 提取插件

```typescript
const { CssExtractRspackPlugin, CssExtractWebpackPlugin } = await import(
  '@lynx-js/css-extract-webpack-plugin'
)
const CssExtractPlugin = api.context.bundlerType === 'rspack'
  ? CssExtractRspackPlugin
  : CssExtractWebpackPlugin
```

这段代码动态导入 CSS 提取插件，并根据当前使用的打包工具（rspack 或 webpack）选择对应的插件实现。

### 2. 处理 CSS 规则

```typescript
const cssRules = [
  CHAIN_ID.RULE.CSS,
  CHAIN_ID.RULE.SASS,
  CHAIN_ID.RULE.LESS,
  CHAIN_ID.RULE.STYLUS,
] as const

cssRules
  .filter(rule => chain.module.rules.has(rule))
  .forEach(ruleName => {
    // 处理每种 CSS 规则...
  })
```

这段代码遍历所有 CSS 相关规则（包括普通 CSS、SASS、LESS 和 Stylus），并对每种规则进行处理。

### 3. 禁用 LightningCSS（针对 Lynx 环境）

```typescript
if (
  rule.uses.has(CHAIN_ID.USE.LIGHTNINGCSS)
  && environment.name === 'lynx'
) {
  rule.uses.delete(CHAIN_ID.USE.LIGHTNINGCSS)
}
```

在 Lynx 环境中，函数禁用了 LightningCSS 加载器，因为 Lynx 有自己的 CSS 处理机制。

### 4. 替换 CSS 提取加载器

```typescript
rule
  .issuerLayer(LAYERS.BACKGROUND)
  .use(CHAIN_ID.USE.MINI_CSS_EXTRACT)
  .loader(CssExtractPlugin.loader)
  .end()
```

这段代码为后台线程层（BACKGROUND）配置 CSS 提取加载器，使用 Lynx 特定的 CSS 提取插件。

### 5. 为主线程添加特殊规则

```typescript
chain
  .module
    .rule(`${ruleName}:${LAYERS.MAIN_THREAD}`)
    .merge(ruleEntries)
    .issuerLayer(LAYERS.MAIN_THREAD)
    .use(CHAIN_ID.USE.IGNORE_CSS)
      .loader(path.resolve(__dirname, './loaders/ignore-css-loader'))
    .end()
    // ... 更多配置 ...
```

这段代码为主线程（MAIN_THREAD）添加了特殊的 CSS 处理规则：
1. 使用 `ignore-css-loader` 处理 CSS
2. 移除不必要的加载器（如 CSS 提取加载器）
3. 配置 CSS 加载器以仅导出本地标识符映射

这是因为在 Lynx 的主线程中，CSS 内容本身不需要被处理，只需要提取类名映射关系。

### 6. 配置 CSS 提取插件

```typescript
chain
  .plugin(CHAIN_ID.PLUGIN.MINI_CSS_EXTRACT)
  .tap(([options]) => {
    return [
      {
        ...options,
        enableRemoveCSSScope: enableRemoveCSSScope ?? true,
        enableCSSSelector,
        targetSdkVersion,
        cssPlugins: [
          CSSPlugins.parserPlugins.removeFunctionWhiteSpace(),
        ],
      } as CssExtractWebpackPluginOptions | CssExtractRspackPluginOptions,
    ]
  })
  .init((_, args: unknown[]) => {
    return new CssExtractPlugin(...args)
  })
```

这段代码配置 CSS 提取插件，传入 Lynx 特定的选项：
1. `enableRemoveCSSScope`：控制 CSS 作用域行为
2. `enableCSSSelector`：启用新的 CSS 选择器实现
3. `targetSdkVersion`：目标 Lynx 引擎版本
4. `cssPlugins`：添加 CSS 解析插件，如移除函数空白

### 7. 处理 CSS 最小化

```typescript
if (enableRemoveCSSScope !== true) {
  chain.optimization.minimizers.delete(CHAIN_ID.MINIMIZER.CSS)
}
```

当 `enableRemoveCSSScope` 不为 `true` 时，禁用 CSS 最小化器，因为当前的 CSS 最小化器可能不支持 Lynx 的自定义解析选项。

### 8. 优化 CSS 模块副作用

```typescript
chain
  .module
  .when(
    enableRemoveCSSScope === undefined,
    module =>
      module
        .rule('lynx.css.scoped')
        .test(/\.css$/)
        .resourceQuery({
          and: [
            /cssId/,
          ],
        })
        .sideEffects(false),
  )
```

当 `enableRemoveCSSScope` 为 `undefined` 时，为带有 `?cssId=<hash>` 查询参数的 CSS 文件（CSS 模块）添加 `sideEffects: false` 标记，这样当 CSS 模块未被使用时，可以被 tree-shaking 移除。

## CSS 加载器选项规范化

函数末尾定义了一个辅助函数 `normalizeCssLoaderOptions`，用于规范化 CSS 加载器选项：

```typescript
export const normalizeCssLoaderOptions = (
  options: CSSLoaderOptions,
  exportOnlyLocals: boolean,
): CSSLoaderOptions => {
  // 实现...
}
```

这个函数主要处理 CSS 模块的 `exportOnlyLocals` 选项，确保在主线程中 CSS 加载器只导出本地标识符映射，而不生成实际的 CSS 代码。
# 总结

`applyCSS` 函数是 Lynx 框架中处理 CSS 的核心组件，它通过精细的配置和优化，实现了 Lynx 环境中的 CSS 模块化、作用域控制和选择器增强等特性。函数采用分层处理策略，为主线程和后台线程提供不同的 CSS 处理逻辑，并通过自定义插件和加载器，确保 CSS 在 Lynx 环境中高效运行。

