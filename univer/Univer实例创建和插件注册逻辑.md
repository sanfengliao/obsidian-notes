![]()![]()
# 实例创建
univer的初始化非常简单，new一下构造函数即可
```typescript
 // univer
const univer = new Univer({
    theme: defaultTheme,
    locale: LocaleType.ZH_CN,
    locales: {
        [LocaleType.ZH_CN]: zhCN,
        [LocaleType.EN_US]: enUS,
        [LocaleType.FR_FR]: frFR,
        [LocaleType.RU_RU]: ruRU,
        [LocaleType.ZH_TW]: zhTW,
        [LocaleType.VI_VN]: viVN,
        [LocaleType.FA_IR]: faIR,
    },
    logLevel: LogLevel.VERBOSE,
});

```
传给`Univer` 构造函数的几个参数的含义也很明显，分别是 `theme: 主题` ， `locale: 语言`, `locales: 支持的语言`和`logLevel:日志级别` 
接下来看一下构造函数的内部实现：
```typescript
 /**
     * Create a Univer instance.
     * @param config Configuration data for Univer
     * @param parentInjector An optional parent injector of the Univer injector. For more information, see https://redi.wendell.fun/docs/hierarchy.
     */
    constructor(config: Partial<IUniverConfig> = {}, parentInjector?: Injector) {
        const injector = this._injector = createUniverInjector(parentInjector, config?.override);

        const { theme, locale, locales, logLevel } = config;
        if (theme) this._injector.get(ThemeService).setTheme(theme);
        if (locales) this._injector.get(LocaleService).load(locales);
        if (locale) this._injector.get(LocaleService).setLocale(locale);
        if (logLevel) this._injector.get(ILogService).setLogLevel(logLevel);

        this._init(injector);
    }
```
构造函数主要做了几个事情:
1. 创建`Injector`, injector负责实现`Univer`的依赖注入和控制反转
2. 设置`theme`, `locale` ,`logLevel`
3. 调用`_init` 函数执行一些列初始化逻辑
## `_init`函数
看一下`_init` 函数的实现
```typescript
private _init(injector: Injector): void {
    this._univerInstanceService.registerCtorForType(UniverInstanceType.UNIVER_SHEET, Workbook);
    this._univerInstanceService.registerCtorForType(UniverInstanceType.UNIVER_DOC, DocumentDataModel);
    this._univerInstanceService.registerCtorForType(UniverInstanceType.UNIVER_SLIDE, SlideDataModel);

    const univerInstanceService = injector.get(IUniverInstanceService) as UniverInstanceService;
    univerInstanceService.__setCreateHandler(
        (type: UnitType, data, ctor, options) => {
            if (!this._startedTypes.has(type)) {
                this._pluginService.startPluginsForType(type);
                this._startedTypes.add(type);

                const model = injector.createInstance(ctor, data);
                univerInstanceService.__addUnit(model, options);

                this._tryProgressToReady();

                return model;
            }

            const model = injector.createInstance(ctor, data);
            univerInstanceService.__addUnit(model, options);
            return model;
        }
    );
}      
```

1. 首先调用`_univerInstanceService` 的方法注册Ctor，等待后续使用
2. 设置`univerInstanceService` 的`createHandler` 方法

# 插件的注册和加载
Univer通过调用`registerPlugin` 方法来注册插件，来给`Univer` 注入更多的能力。`Univer` 会调用`_pluginService` 的`registerPlugin`方法 来完成插件的注册
```typescript
/** Register a plugin into univer. */
registerPlugin<T extends PluginCtor<Plugin>>(plugin: T, config?: ConstructorParameters<T>[0]): void {
    this._pluginService.registerPlugin(plugin, config);
}
```

## PluginService#registerPlugin
看一下`PluginService#registerPlugin` 的实现

```typescript
/**
 * Register a plugin into univer.
 * @param {PluginCtor} ctor The plugin's constructor.
 * @param {ConstructorParameters} [config] The configuration for the plugin.
 */
registerPlugin<T extends PluginCtor>(ctor: T, config?: ConstructorParameters<T>[0]): void {
    this._assertPluginValid(ctor);

    const item = { plugin: ctor, options: config };
    this._pluginRegistry.set(ctor.pluginName, item);

    this._logService.debug('[PluginService]', `Plugin "${ctor.pluginName}" registered.`);

    const { type } = ctor;
    if (this._loadedPluginTypes.has(type)) {
        if (type === UniverInstanceType.UNIVER_UNKNOWN) {
            this._loadFromPlugins([item]);
        } else {
            this._flushType(type);
        }
    }
}
```
该函数做了几件事
1. 判断插件是否合法
2. 在`_pluginRegistry` 存储要注册的插件
3. 根据plugin的type 调用不同的方式来注册插件
    1. 如果插件的类型`UNIVER_UNKNOWN`, 调用 `_loadFromPlugins` 立刻加载
    2. 否则调用`_flushType` 加载，`_flushType` 会批量加载类型为`type` 的插件，提升性能

## `PluginService#_loadFromPlugins` 

来看一下`PluginService#_loadFromPlugins` 的实现
```typescript
private _loadFromPlugins(plugins: IPluginRegistryItem[]): void {
    const finalPlugins: IPluginRegistryItem[] = [];

    // We do a topological sort here to make sure that plugins with dependencies are registered first.
    const visited = new Set<string>();
    const dfs = (item: IPluginRegistryItem) => {
        const { plugin } = item;
        const { pluginName } = plugin;

        // See if the plugin has already been loaded or visited.
        if (this._loadedPlugins.has(pluginName) || visited.has(pluginName)) {
            return;
        }

        // Mark it self as visited.
        visited.add(pluginName);

        // We do not need to load it again because it will be loaded in this `_loadFromPlugins`.
        this._pluginRegistry.delete(pluginName);

        const dependents = plugin[DependentOnSymbol];
        if (dependents) {
            // Loop over its dependencies.
            dependents.forEach((d) => {
                // If the dependency is among those who are already registered, we should push it to the queue.
                const dItem = this._pluginRegistry.get(d.pluginName);
                if (dItem) {
                    dfs(dItem);
                } else if (!this._seenPlugins.has(d.pluginName) && !visited.has(d.pluginName)) {
                    // Otherwise, it maybe a plugin that is not registered yet.
                    if (plugin.type === UniverInstanceType.UNIVER_UNKNOWN && d.type !== UniverInstanceType.UNIVER_UNKNOWN) {
                        throw new Error('[PluginService]: cannot register a plugin with Univer type that depends on a plugin with other type. '
                            + `The dependent is ${plugin.pluginName} and the dependency is ${d.pluginName}.`
                        );
                    }

                    if (plugin.type !== d.type && d.type !== UniverInstanceType.UNIVER_UNKNOWN) {
                        this._logService.debug(
                            '[PluginService]',
                            `Plugin "${pluginName}" depends on "${d.pluginName}" which has different type.`
                        );
                    }

                    this._logService.debug(
                        '[PluginService]',
                        `Plugin "${pluginName}" depends on "${d.pluginName}" which is not registered. Univer will automatically register it with default configuration.`
                    );

                    this._assertPluginValid(d);
                    dfs({ plugin: d, options: undefined });
                }
            });
        }

        finalPlugins.push(item);
    };

    plugins.forEach((p) => dfs(p));

    const pluginInstances = finalPlugins.map((p) => this._initPlugin(p.plugin, p.options));
    this._pluginsRunLifecycle(pluginInstances);
}
```
这个函数做了几件事: 
1. 遍历每个插件，调用dfs确保每个插件的依赖项都已经注册，因为一个插件可能依赖别的插件，因此需要确保被插件依赖的插件也已经注册了
2. 调用`_initPlugin` 初始化插件
3. `_pluginsRunLifecycle` 执行`plugin` 的生命周期函数

###  `PluginService#_initPlugin`
`PluginService#_initPlugin` 的实现很简单，使用`inject` 来创建插件实例，然后插件信息存储`_pluginStore` 和`_loadedPlugins` ，最后返回

```typescript
 private _initPlugin<T extends Plugin>(plugin: PluginCtor<T>, options: any): Plugin {
        // eslint-disable-next-line ts/no-explicit-any
        const pluginInstance: Plugin = this._injector.createInstance(plugin as unknown as Ctor<any>, options);
        this._pluginStore.addPlugin(pluginInstance);
        this._loadedPlugins.add(plugin.pluginName);

        this._logService.debug('[PluginService]', `Plugin "${pluginInstance.getPluginName()}" loaded.`);
        return pluginInstance;
    }
```

###  `PluginService#_pluginsRunLifecycle` 

```typescript
protected _pluginsRunLifecycle(plugins: Plugin[]): void {
    // Let plugins go through already reached lifecycle stages.
    const currentStage = this._lifecycleService.stage;
    getLifecycleStagesAndBefore(currentStage).subscribe((stage) => this._runStage(plugins, stage));

    if (currentStage !== LifecycleStages.Steady) {
        const subscription = this._lifecycleService.lifecycle$.pipe(
            skip(1)
        ).subscribe((stage) => {
            this._runStage(plugins, stage);
            if (stage === LifecycleStages.Steady) {
                subscription.unsubscribe();
            }
        });
    }
}
```
`_pluginsRunLifecycle`做了两件事
1. 首先获取univer当前所在的生命周期，然后触发插件当前生命周期之前所有生命周期的回调函数，确保新注册的插件能够"追赶"上系统当前的生命周期状态，执行所有已经经过的阶段。
2. 然后订阅后续生命周期事件
    1. 这里使用`skip(1)` 操作符跳过第一个事件（因为第一个事件是当前阶段，已经在第一部分处理过了）
    2. 对于后续的每个生命周期事件，调用 `_runStage` 方法执行对应阶段的回调
    3. 如果到达了 `Steady` 阶段，则取消订阅，因为不会有更多生命周期事件了


## `PluginService#_flushType` 

`_flushType` 会负责延迟加载类型`type` 的的插件, 通过setTimeout将它们合并为一次批量加载操作,  然后调用`_loadPluginsForType` 完成插件的注册。
`_loadPluginsForType`获取所有类型为`type` 的插件，存储到`allPluginsOfThisType` 中，然后在调用`_loadFromPlugins`加载插件

```typescript
 private _flushType(type: UnitType): void {
    if (this._flushTimerByType.get(type) === undefined) {
        this._flushTimerByType.set(type, setTimeout(() => {
            this._loadPluginsForType(type);
            this._flushTimerByType.delete(type);
        }, INIT_LAZY_PLUGINS_TIMEOUT) as unknown as number);
    }
}

private _loadPluginsForType(type: UnitType): void {
    const keys = Array.from(this._pluginRegistry.keys());
    const allPluginsOfThisType: IPluginRegistryItem[] = [];
    keys.forEach((key) => {
        const item = this._pluginRegistry.get(key)!;
        if (item.plugin.type === type) {
            allPluginsOfThisType.push(item);
        }
    });

    this._loadFromPlugins(allPluginsOfThisType);
    this._loadedPluginTypes.add(type);
}
```
