# 02 · JavaScript 与 TypeScript 速查

React Native 用 **JavaScript（简称 JS）** 写，我们这次还额外用 **TypeScript（简称 TS）**。TS 就是 JS 加了「类型标注」，让 AI 更不容易写错。

这一篇不教你语法，只告诉你**代码里那些奇怪符号是啥意思**，看到不至于懵。

---

## let / const / var —— 声明变量

```js
let age = 20        // 以后会改的，用 let
const name = "小明"  // 以后不会改的，用 const（最常用）
var oldWay = "旧的"  // 老写法，现在基本不用，看到也当 let 看就行
```

- 经验：**默认用 `const`**，需要改的时候才用 `let`。AI 一般也这么写。
- `const` 不是说盒子里的东西完全不能动，而是「这个标签不能再贴到别的盒子上」。对象/数组内部还是能改的（见 [[03-编程基础概念]]）。

---

## 箭头函数（Arrow Function）

```js
// 普通函数
function add(a, b) {
  return a + b
}

// 箭头函数（等价的简写）
const add = (a, b) => {
  return a + b
}

// 更简：函数体只有一行 return 时，可以省略 return 和花括号
const add = (a, b) => a + b
```

- 看到 `=>` 就是函数，读作「输入 … 输出 …」。
- RN 代码里**几乎全是箭头函数**，AI 默认就用它。别被 `=>` 吓到。

---

## 模板字符串（Template Literal）

```js
let name = "小明"
let age = 20

// 旧写法：用 + 拼接，麻烦
"我叫" + name + "，今年" + age + "岁"

// 模板字符串：用反引号 ` ` 包起来，${} 里放变量
`我叫${name}，今年${age}岁`
```

- 反引号 `` ` `` 在键盘左上角（数字 1 左边那个键）。
- `${...}` 里面可以放任何表达式，AI 会大量用这个拼文字。

---

## 解构（Destructuring）

**大白话**：把盒子里的东西**一次性拆出来**几个，省得一个一个点。

```js
let person = { name: "小明", age: 20, city: "北京" }

// 对象解构：按名字取
let { name, age } = person
// 等于 name = "小明"，age = 20

let arr = ["苹果", "香蕉"]
// 数组解构：按位置取
let [first, second] = arr
// first = "苹果"，second = "香蕉"
```

- 看到 `const { name, age } = props` 这种写法，意思就是「从 props 这个对象里取出 name 和 age 两个」。
- 在 React 里**极其常见**，因为组件经常收到一大坨 props。

---

## 展开运算符 `...`（Spread Operator）

**大白话**：把一个盒子里的东西**全部倒出来**，铺平。

```js
let defaults = { color: "blue", size: 10 }
let custom = { ...defaults, color: "red" }
// 结果：{ color: "red", size: 10 }
//   先把 defaults 全倒进来，再用 color: "red" 覆盖

let a = [1, 2]
let b = [...a, 3]   // [1, 2, 3]
```

- `...` 读作「展开」或「spread」。
- AI 改对象/数组时常用它做「在原基础上改一点」：**先展开旧的，再覆盖要改的字段**，这样不会动到原数据。

---

## 可选链 `?.` 和空值合并 `??`

```js
let user = { name: "小明" }   // 没有 address

user.address           // undefined —— 直接取没有的字段，不报错
user.address.city      // 报错！因为 address 是 undefined
user.address?.city     // 不报错，返回 undefined —— ?. 表示「有就取，没有就给 undefined」

let nick = null
let show = nick ?? "匿名"   // "匿名" —— ?? 表示「左边是空(null/undefined)就用右边」
```

- `?.` = 「**稳妥地往下一层取**，半路断了也不报错」。
- `??` = 「**给个默认值**，前面没东西就用这个」。
- AI 写「从复杂对象里取某个深层字段」时常用，是防报错的好习惯。

---

## import / export —— 模块导入导出

**大白话**：代码拆成多个文件，一个文件**导出**东西，另一个文件**导入**来用。像把工具分门别类放进不同抽屉。

```js
// utils.js 导出
export function add(a, b) { return a + b }
export const PI = 3.14

// App.tsx 导入
import { add, PI } from "./utils"
```

- `import { xxx } from "路径"` = 从某个文件拿东西过来用。
- 路径以 `./` 开头表示「相对当前文件」，`@/` 是某些项目配的别名（指代 src 目录）。
- 不以 `./` 开头的（如 `import { View } from "react-native"`）是从**装好的第三方包**里拿，见 [[01-最基础的工具与环境术语]]。

---

## Promise 和 async / await —— 处理「要等一会儿」的事

**大白话**：有些事不是立刻有结果（比如从网上请求数据、存文件）。`async/await` 让你**用看起来像同步的写法等它**。

```js
// 拿数据：等服务器返回，可能要几秒
async function loadData() {
  const data = await fetch("https://api.example.com/users")  // 等 fetch 完成
  const json = await data.json()                              // 等解析完成
  console.log(json)
}
```

- `async` 标在函数前，表示「这个函数里有要等的事」。
- `await` 标在前面，表示「**在这停一下，等它出结果再继续**」。
- AI 说「这是个异步操作」「接口返回的是 Promise」，就是这回事。请求网络、读本地存储、延时，都是异步。
- 常配合 `try / catch` 捕获出错（详见下文 AI 写法）。

---

## TypeScript 类型标注（你会看到但不用太纠结）

TS 在 JS 基础上**给变量标上类型**，AI 用它防错。你看不懂类型标注不影响理解逻辑。

```ts
let age: number = 20                          // age 是数字
let name: string = "小明"                     // name 是字符串

function add(a: number, b: number): number {  // 参数是数字，返回也是数字
  return a + b
}

interface User {          // 定义一种「用户」的形状
  name: string
  age: number
  city?: string           // ? 表示「这个字段可有可无」
}

let u: User = { name: "小明", age: 20 }
```

- `: 类型` 紧跟在变量/参数后面，是**类型标注**，不是 JS 的一部分，TS 编译时会检查。
- `interface` / `type` 用来**描述一个对象的形状**（有哪些字段、什么类型）。
- `?` 表示「可选」。
- 看到这些**直接跳过类型部分**看核心逻辑就行，类型错了 AI 和编译器会提醒。

---

## null 和 undefined 的区别

- `undefined`：盒子**压根没被赋过值**（声明了但没放东西）。
- `null`：**主动放进去一个「空」**，表示「这里故意没有」。

实际用的时候，把它们都当「没有东西」处理即可，用 `??` 给默认值。

---

## 常见「看不懂符号」速查表

| 符号 | 意思 |
|------|------|
| `=` | 赋值，把右边放进左边 |
| `===` | 全等于（值和类型都一样），判断相等**用这个**不用 `==` |
| `!==` | 不等于 |
| `=>` | 箭头函数 |
| `` ` `` | 反引号，包模板字符串 |
| `${ }` | 模板字符串里放变量 |
| `...` | 展开运算符，把东西倒出来 |
| `?.` | 可选链，稳妥地往下取，断了不报错 |
| `??` | 空值合并，左边空就用右边 |
| `?:` | 三目运算符，`条件 ? A : B` |
| `!` | TS 里表示「我保证它不是 null/undefined」 |
| `//` | 单行注释 |
| `/* */` | 多行注释 |

---

上一篇：[[03-编程基础概念]] ｜ 下一篇：[[05-React核心概念]]
