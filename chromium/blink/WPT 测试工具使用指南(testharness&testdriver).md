本文档只讲**怎么用**。先给最小可用模板，再讲断言、异步测试、模拟输入的常用 API，最后用仓库里的真实例子串起来。

官方文档（推荐配合阅读）：
- testharness API：<https://web-platform-tests.org/writing-tests/testharness-api.html>
- testdriver：<https://web-platform-tests.org/writing-tests/testdriver.html>
- testdriver-actions：<https://web-platform-tests.org/writing-tests/testdriver-actions.html>
- 写测试总览：<https://web-platform-tests.org/writing-tests/>

---

# 0. 一句话区分

- **testharness.js** → 写测试、写断言（`test()` / `assert_*`）。
- **testharnessreport.js** → 必须一起引入，让测试结果能被运行器接收（不用直接调用）。
- **testdriver.js** → 模拟用户输入（点击、按键、滚动等，纯 JS 做不到的事）。
- **testdriver-vendor.js** → 必须跟 testdriver.js 一起引入（不用直接调用）。
- **testdriver-actions.js** → 需要用 `Actions` 链式动作（拖拽、连续输入）时才引入。

---

# 1. 最小模板

```html
<!DOCTYPE html>
<script src="/resources/testharness.js"></script>
<script src="/resources/testharnessreport.js"></script>
<script>
  test(() => {
    assert_equals(1 + 1, 2, "1+1 应该等于 2");
  }, "基本加法测试");
</script>
```

需要模拟输入时，再加三行：

```html
<script src="/resources/testdriver.js"></script>
<script src="/resources/testdriver-vendor.js"></script>
<script src="/resources/testdriver-actions.js"></script> <!-- 仅当用 Actions 时 -->
```

> 这些 `/resources/...` 路径是 WPT 的虚拟根，在 Chromium 里对应 [third_party/blink/web_tests/resources/](third_party/blink/web_tests/resources/)。

---

# 2. testharness.js：写测试与断言

## 三种测试写法

```js
// (a) 同步测试
test(() => {
  assert_equals(document.title, "");
}, "title 默认为空");

// (b) promise_test —— 推荐，处理异步最方便
promise_test(async () => {
  const res = await fetch("/resources/blank.html");
  assert_true(res.ok);
}, "fetch 能拿到资源");

// (c) async_test —— 需要手动控制完成（事件回调场景）
async_test(t => {
  const el = document.createElement("div");
  el.addEventListener("click", t.step_func_done(() => {
    assert_equals(el.dataset.clicked, "yes");
  }));
  document.body.appendChild(el);
  el.click();   // 注意：真实输入要用 test_driver，见第 3 节
}, "点击后设置标记");
```

要点：
- `promise_test` 里 `await` 抛出的 rejection 会自动算 FAIL，不用 try/catch。
- `async_test` 中要把回调包进 `t.step_func(...)` 或 `t.step_func_done(...)`，否则断言失败不会被框架捕获。
- 默认超时 10 秒；需要更长用 `setup({timeout: 60000})` 或单测 `test(fn, name, {timeout: 30000})`。

## 常用断言

| 断言 | 用途 |
|------|------|
| `assert_equals(actual, expected, msg)` | 严格相等（`===`） |
| `assert_not_equals(a, b, msg)` | 不相等 |
| `assert_true(x, msg)` / `assert_false(x, msg)` | 布尔判断 |
| `assert_approx_equals(a, b, epsilon, msg)` | 浮点近似（动画/滚动常用） |
| `assert_array_equals(a, b, msg)` | 数组逐项相等 |
| `assert_throws_dom(name, fn, msg)` | 断言 fn 抛出指定 DOMException |
| `assert_greater_than(a, b, msg)` / `assert_less_than` / `assert_between` | 大小比较 |
| `assert_unreached(msg)` | 不该执行到这里 |

## 文件级配置

```js
setup({
  explicit_done: true,   // 自己调 done() 才结束（多 iframe 时用）
  timeout: 30000,        // 整体超时
});
```

---

# 3. testdriver.js：模拟用户输入

纯 JS 受安全限制不能合成「可信」输入。testdriver 通过测试运行器从外部注入真实事件。

## 点击与按键

```js
// 点击元素（会先移动指针到元素中心）
await test_driver.click(myButton);

// 向元素发送按键（需要先聚焦/可交互）
// 注意：特殊键用 WebDriver 规范的码点，见下表
await test_driver.send_keys(inputEl, "hello");
await test_driver.send_keys(scroller, "\uE014"); // ArrowRight
```

> 仓库里 [support/common.js](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/support/common.js) 封装了一个 `keyPress(target, key)` 帮助函数，把 `ArrowRight` 等友好名称映射成 `\uE014` 等码点，优先用它。

常用特殊键码点（来自 [support/common.js](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/support/common.js)）：

| 名称 | 码点 |
|------|------|
| ArrowLeft | `\uE012` |
| ArrowUp | `\uE013` |
| ArrowRight | `\uE014` |
| ArrowDown | `\uE015` |
| PageUp / PageDown | `\uE00E` / `\uE00F` |
| Home / End | `\uE011` / `\uE010` |

## 动作序列 Actions（连续/复杂输入）

需要「按下 → 移动 → 松开」这种时序，或拖拽、多指时用：

```js
let actions = new test_driver.Actions()
  .pointerMove(0, 0, {origin: dragHandle})
  .pointerDown()
  .pointerMove(100, 0, {origin: dragHandle})  // 拖动 100px
  .pointerUp();

await actions.send();
```

方法链：

| 方法 | 含义 |
|------|------|
| `.pointerMove(x, y, {origin})` | 移动指针，`origin` 可为元素或 `"viewport"`/`"pointer"` |
| `.pointerDown({button=0})` / `.pointerUp()` | 按下/松开（0=左键） |
| `.keyDown(key)` / `.keyUp(key)` | 按下/松开按键 |
| `.addTick()` | 插入一个时间帧（默认每帧 16ms） |
| `.scroll(x, y, deltaX, deltaY, {origin})` | 滚动 |
| `.send()` | 执行整条序列，返回 Promise |

## 其它常用能力

```js
await test_driver.set_permission({name: "geolocation"}, "granted"); // 授权
await test_driver.bless("user activation", () => doSomething());    // 赋予用户激活
await test_driver.minimize_window();                                 // 最小化窗口
await test_driver.set_window_rect({x, y, width, height});            // 改窗口
```

---

# 4. 真实例子：键盘打断滚动吸附

来自 [keyboard-snap-interruption.html](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/input/keyboard-snap-interruption.html)，两条线一起用：

```html
<script src="/resources/testharness.js"></script>
<script src="/resources/testharnessreport.js"></script>
<script src="/resources/testdriver.js"></script>
<script src="/resources/testdriver-vendor.js"></script>
<script src="/resources/testdriver-actions.js"></script>
<script src="/dom/events/scrolling/scroll_support.js"></script>
<script src="../support/common.js"></script>   <!-- 提供 keyPress() -->

<script>
const scroller = document.getElementById("scroller");

promise_test(async t => {
  scroller.focus();
  assert_equals(scroller.scrollLeft, 0, "前置条件");

  const scrollEndPromise = waitForScrollEndFallbackToDelayWithoutScrollEvent(scroller);

  // 第 1 次按键：开始平滑吸附动画
  await keyPress(scroller, "ArrowRight");          // → test_driver.send_keys

  // 等动画开始但未结束
  await waitFor(() => scroller.scrollLeft > 0, "等待滚动开始");
  assert_less_than(scroller.scrollLeft, 600, "还没滚完");

  // 第 2 次按键：打断进行中的动画
  await keyPress(scroller, "ArrowRight");

  await scrollEndPromise;

  // 断言：即使被打断，仍吸附到目标
  assert_equals(scroller.scrollLeft, 600, "应成功吸附到目标");
}, "按键打断吸附动画后仍保持目标");
</script>
```

这个例子展示了典型模式：
1. `promise_test` 包裹异步流程；
2. `test_driver.send_keys`（经 `keyPress` 封装）产生可信键盘输入；
3. `assert_*` 在每个关键时间点断言状态；
4. 用 `waitFor` / `waitForScrollEnd...` 等帮助函数（在 [scroll_support.js](third_party/blink/web_tests/external/wpt/dom/events/scrolling/scroll_support.js)）等动画/滚动完成。

---

# 5. 在 Chromium 里运行/调试

WPT 测试在 Chromium 里通常用 `content_shell` 跑（不是 ChromeDriver），此时：
- `testharnessreport.js` 调 `testRunner.dumpAsText()` + `waitUntilDone()`，结束时 `notifyDone()`；
- `testdriver-vendor.js` 把 `test_driver.*` 转发给 `testRunner`/`eventSender`（C++ 侧合成真实事件）。

常用命令（仓库根目录）：

```bash
# 跑单个 WPT 测试（需要先编译 out/Release/content_shell）
python third_party/blink/tools/run_web_tests.py \
  external/wpt/css/css-scroll-snap/input/keyboard-snap-interruption.html

# 指定构建目录
python third_party/blink/tools/run_web_tests.py -t Release <test_path>
```

输出里会有 `PASS` / `FAIL` / `TIMEOUT` 行；失败时可加 `--debug` 或直接在浏览器里打开该 HTML 手动观察。

## 常用运行参数（已核对实际存在）

```bash
python third_party/blink/tools/run_web_tests.py \
  --child-processes=1 \                 # 单进程，便于复现/调试
  --no-retry-failures \                 # 失败不重试，看真实结果
  --additional-driver-flag=--disable-smooth-scrolling \  # 给 content_shell 传 flag
  --iterations=5 \                      # 重复跑 N 轮，找偶发失败
  --order=natural \                     # 按文件顺序（默认 random）
  --verbose \                           # 详细输出
  external/wpt/css/css-scroll-snap/input/keyboard-snap-interruption.html
```

- **更新/重置基线**：`--reset-results <test>` —— 用本次实际输出覆盖 `*-expected.txt`。
- **不弹结果页**：`--no-show-results`。
- 想看渲染：直接在 Chrome 里打开该 HTML（但 testdriver 输入在普通 Chrome 下不工作，只能观察静态行为）。

---

# 6. 期望值与结果基线（Chromium 专属，重要）

WPT 测试在 Chromium 里**不是「跑过即 Pass」**，而是把实际输出和期望输出比对。

## testharness 测试的基线

每个 testharness 测试有对应的 `*-expected.txt`，记录各子测的预期结果：

```
# external/wpt/css/css-scroll-snap/input/keyboard-snap-interruption-expected.txt
This is a testharness.js-based test.
PASS Interrupting a snap animation with another key press targeting the same node preserves snapping destination
Harness: the test ran to completion.
```

- 实际输出 == 期望 → PASS；不一致 → FAIL。
- 新增/修改测试后，跑 `run_web_tests.py --reset-results <test>` 生成基线，并把 `*-expected.txt` 一起提交。

## 失败结果登记（期望文件）

如果一个测试在当前实现下**确实**会失败（已知 bug、平台差异、未实现特性），必须登记到期望文件，否则 CI 报红：

| 文件 | 用途 |
|------|------|
| [TestExpectations](third_party/blink/web_tests/TestExpectations) | 通用期望（FAIL/CRASH/TIMEOUT/SLOW/MISSING 等） |
| [NeverFixTests](third_party/blink/web_tests/NeverFixTests) | 永久不跑（Skip）的测试 |
| [SlowTests](third_party/blink/web_tests/SlowTests) | 标记 SLOW（给更长超时） |
| [StaleTestExpectations](third_party/blink/web_tests/StaleTestExpectations) | 待清理的过时期望 |

语法示例：

```
# TestExpectations
crbug.com/12345 external/wpt/css/css-scroll-snap/input/foo.html [ Failure ]
external/wpt/css/css-scroll-snap/input/bar.html [ Skip ]   # 整文件跳过
```

## Virtual 套件

同一测试在不同 flag 组合下重跑（如启用某实验特性），配置在 [third_party/blink/web_tests/virtual/](third_party/blink/web_tests/virtual/)，每个 virtual 目录有自己的 `TestExpectations`。

---

# 7. 测试元数据（写测试必备）

在 `<head>` 里声明，WPT 工具链和 CI 都会读取：

```html
<link rel="help" href="https://drafts.csswg.org/css-scroll-snap-1/#scroll-snap-type">
<link rel="author" title="Sanfeng Liao" href="mailto:sanfeng@chromium.org">
<link rel="match" href="my-ref.html">              <!-- reftest：参考页 -->
<link rel="mismatch" href="my-notref.html">        <!-- reftest：应不同 -->
<link rel="assert" href="...">                      <!-- 关联 issue -->
<meta name="assert" content="按键打断吸附后仍到达目标">  <!-- 测试意图，给 reviewer/失败排查用 -->
<meta name="timeout" content="long">                <!-- 60s（默认 10s）-->
<meta name="variant" content="?feature=bidi">       <!-- 变体：以不同 URL 参数重跑 -->
```

给你的 scroll-snap 测试的建议：带平滑动画的容易踩 10s 默认超时，加 `<meta name="timeout" content="long">`；并补 `<meta name="assert">` 说明意图。

---

# 8. `t` 对象：异步测试的进阶用法

`test(fn, name)` 和 `promise_test(async t => {...})` 收到的 `t` 是 `Test` 实例，有几个常用方法（见 [testharness.js](third_party/blink/web_tests/external/wpt/resources/testharness.js)）：

| 方法 | 用途 |
|------|------|
| `t.add_cleanup(fn)` | 注册清理函数，测试结束（无论成败）都会跑 |
| `t.step_func(fn)` | 包裹回调，内部异常会被框架捕获算 FAIL |
| `t.step_timeout(fn, ms)` | 框架感知的延时（异常被捕获） |
| `t.step_wait(cond, description, timeout=3000, interval=100)` | **轮询等待条件成立**，推荐替代手写 rAF 循环 |

## 用 `t.step_wait` 替代手写轮询

你当前测试里：

```js
await waitFor(() => scroller.scrollLeft > 0, "Timeout waiting for scroll to start");
```

可改用标准的 `t.step_wait`（注意 `t` 来自 `promise_test` 的参数）：

```js
promise_test(async t => {
  ...
  await t.step_wait(
    () => scroller.scrollLeft > 0,
    "等待滚动开始"
  );
  ...
});
```

好处：内置超时、异常处理、与框架集成，不用自己维护 `waitFor` 帮助函数。

## `t.add_cleanup` 示例

```js
promise_test(async t => {
  scroller.scrollTop = 0;
  t.add_cleanup(() => { scroller.scrollTop = 0; });  // 恢复状态

  const orig = scroller.style['scroll-snap-type'];
  scroller.style['scroll-snap-type'] = 'y mandatory';
  t.add_cleanup(() => { scroller.style['scroll-snap-type'] = orig; });

  ...
});
```

> 注意：不存在 `t.step_wait_cond`（我之前口误），标准方法是 `t.step_wait`。全局也有 `step_wait(...)` 和 `step_timeout(...)` 可在测试外用。

---

# 9. Reftest：参考页像素对比测试

## 它解决什么问题

testharness 只能断言「数值/字符串」状态（`assert_equals(scrollLeft, 600)`）。但有些 bug 表现为**视觉异常**——比如吸附位置偏了几个像素、边框画错、文字渲染崩了——很难用断言表达。reftest 就是干这个的：**不写断言，而是把测试页和一个「参考页」并排渲染，比对像素**。

## 工作原理

1. 写两个 HTML：**测试页**（test）和**参考页**（reference）。
2. 在测试页里用 `<link rel="match" href="ref.html">` 声明参考页。
3. 运行器分别渲染两者，**逐像素比对**（允许通过 fuzzy 容差）。
4. 像素一致 → PASS；不一致 → FAIL，并产出 `-actual.png` / `-expected.png` 差分图。

关键点：**参考页必须用「与被测特性无关」的方式画出相同结果**。比如要测 scroll-snap 把元素吸到某个位置，参考页就直接用 `position:absolute` 把元素摆在那个位置，不依赖 snap 逻辑。这样如果两者像素一致，就说明 snap 工作正常。

## 真实例子

[large-scroll-margin-001.html](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/large-scroll-margin-001.html)：

```html
<link rel="match" href="../reference/ref-filled-green-100px-square.xht">
<p>Test passes if there is a filled green square and <strong>no red</strong>.</p>
<div style="overflow:hidden; width:100px; height:100px; scroll-snap-type: y mandatory;">
  <div style="height:100px; background:green;"></div>
  <div style="scroll-snap-align: start end; height:12345674890px;
              scroll-margin-top:9223372036854775767px; background:red;"></div>
</div>
```

参考页 [ref-filled-green-100px-square.xht](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/reference/ref-filled-green-100px-square.xht) 就是一个纯绿色的 100×100 方块。这里测的是「超大 scroll-margin 数值不能让红色块露出来/不能崩溃」，只要最终视觉是满绿方块即通过。

## `match` vs `mismatch`

```html
<link rel="match" href="ref.html">       <!-- 测试页应与 ref.html 像素一致 -->
<link rel="mismatch" href="notref.html">  <!-- 测试页应与 notref.html 像素不同 -->
```

`mismatch` 用于「两个本应不同的页面确实不同」——例如测「禁用某特性后确实不渲染了」。

## 多个参考页

```html
<link rel="match" href="ref-a.html">
<link rel="match" href="ref-b.html">
```

任一参考页匹配即 PASS（用于允许实现差异的场景）。

## fuzzy 容差

渲染难免有亚像素/抗锯齿差异，可用 `<meta name="fuzzy">` 声明容差：

```html
<meta name="fuzzy" content="maxDifference=0-1; totalPixels=0-5">
<!-- 或针对特定参考页 -->
<meta name="fuzzy" content="ref.html: maxDifference=2; totalPixels=10">
```

- `maxDifference`：单像素最大 RGB 差异范围。
- `totalPixels`：允许不同的像素总数范围。

## reftest vs testharness 怎么选

| 场景 | 选 |
|------|----|
| 断言数值/状态（scrollLeft、属性、抛异常） | testharness |
| 验证视觉渲染（布局、吸附后位置、颜色） | reftest |
| 需要复杂交互后再看视觉 | testharness + reftest 都可，或 testharness 内截图（Chromium 专属） |

> 顺带：reftest 不需要引入 testharness.js / testdriver.js，它只看 `<link rel="match/mismatch">` 和最终像素。但可以和 testdriver 结合（在测试页里先用 testdriver 触发交互，再让运行器截图比对）。

---

# 10. 常见坑

1. **忘引 `testharnessreport.js`** → 测试不会等待，结果输出异常。两者成对引入。
2. **忘引 `testdriver-vendor.js`** → `test_driver.click` 调用时 `test_driver_internal` 未定义，报错。
3. **用 `el.click()` 代替 `test_driver.click(el)`** → 前者不是「可信」输入，某些行为（如全屏、权限、焦点）表现不同。需要真实用户语义就用 testdriver。
4. **`async_test` 回调里直接 `assert_*`** → 异常逃逸。用 `t.step_func()` 包裹。
5. **滚动/动画测试不等待结束就断言** → 状态还没稳定。用 `waitFor` / `waitForAnimationEnd` 等帮助函数。
6. **特殊键直接传字符串** → `send_keys(el, "ArrowRight")` 不会触发方向键，要用码点 `\uE014` 或 [support/common.js](third_party/blink/web_tests/external/wpt/css/css-scroll-snap/support/common.js) 的 `keyPress()`。

---

## 来源
- [WPT testharness API 文档](https://web-platform-tests.org/writing-tests/testharness-api.html)
- [WPT testdriver 文档](https://web-platform-tests.org/writing-tests/testdriver.html)
- [WPT testdriver-actions 文档](https://web-platform-tests.org/writing-tests/testdriver-actions.html)
- [WPT 写测试总览](https://web-platform-tests.org/writing-tests/)
