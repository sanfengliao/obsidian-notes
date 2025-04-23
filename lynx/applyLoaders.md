
`applyLoaders` 函数是 Lynx 框架中 `pluginReactLynx` 插件的关键组件，负责配置 JavaScript/JSX 代码的转换和处理规则。这个函数为主线程和后台线程设置不同的加载器，确保代码能够正确地转换为 Lynx 环境可执行的格式。

## 主要功能

1. **启用层级支持**：
   ```typescript
   chain.experiments({
     ...experiments,
     layers: true,
   })
   ```
   开启 webpack/rspack 的层级（layers）功能，这是实现线程分离的基础。

2. **配置后台线程加载器**：
   ```typescript
   backgroundRule
     .issuerLayer(LAYERS.BACKGROUND)
     .uses
       .merge(uses)
     .end()
     .use(LAYERS.BACKGROUND)
       .loader(ReactWebpackPlugin.loaders.BACKGROUND)
       .options({
         compat,
         enableRemoveCSSScope,
         jsx,
         isDynamicComponent: experimental_isLazyBundle,
         inlineSourcesContent,
         defineDCE,
       })
   ```
   为后台线程（BACKGROUND）代码配置专用的加载器，传入相关选项。

3. **配置主线程加载器**：
   ```typescript
   mainThreadRule
     .issuerLayer(LAYERS.MAIN_THREAD)
     .uses
       .merge(uses)
     .end()
     // 其他配置...
     .use(LAYERS.MAIN_THREAD)
       .loader(ReactWebpackPlugin.loaders.MAIN_THREAD)
       .options({
         compat,
         enableRemoveCSSScope,
         jsx,
         inlineSourcesContent,
         isDynamicComponent: experimental_isLazyBundle,
         shake,
         defineDCE,
       })
   ```
   为主线程（MAIN_THREAD）代码配置专用的加载器，传入相关选项。

4. **优化 SWC 配置**：
   ```typescript
   .when(uses[CHAIN_ID.USE.SWC] !== undefined, rule => {
     rule.uses.delete(CHAIN_ID.USE.SWC)
     // 重新配置 SWC 加载器...
     rule.use(CHAIN_ID.USE.SWC)
       .merge(swcLoaderRule)
       .options({
         ...swcLoaderOptions,
         jsc: {
           ...swcLoaderOptions.jsc,
           target: 'es2019',
         },
       })
   })
   ```
   如果存在 SWC 加载器，则优化其配置，特别是将目标设置为 ES2019。

5. **清除默认加载器**：
   ```typescript
   rule.uses.clear()
   ```
   清除 Rsbuild 的默认加载器，避免 JSX 被默认的 `builtin:swc-loader` 转换。






## 总结

`applyLoaders` 函数是 Lynx 框架中处理代码转换的核心组件，它通过配置专用的加载器和优化选项，确保 React 代码能够正确地转换为 Lynx 环境可执行的格式。函数采用线程分离的策略，为主线程和后台线程提供不同的处理逻辑，并支持多种代码优化技术，如代码摇树和死代码消除。

