# 03 · React 核心概念

React Native 用的是 **React** 这套思想，只是把网页的「按钮、文字」换成了手机上的「按钮、文字」。所以搞懂 React 的几个核心概念，RN 你就看懂一大半了。

---

## 组件（Component）

**大白话**：组件是一块**可以反复用的「乐高积木」**。一个页面就是一块块积木拼起来的。

```tsx
// 定义一个组件：本质上就是一个返回界面的函数
function Greeting({ name }) {
  return <Text>你好，{name}</Text>
}

// 用它（像用自定义标签一样）
<Greeting name="小明" />
<Greeting name="小红" />
```

- 组件名**首字母必须大写**（`Greeting` 而不是 `greeting`），这是 React 的硬规矩，AI 一定会遵守。
- 一个 App 就是一棵「组件树」：最外面 `App` 里面套着各个页面，页面里套着各种小组件。
- AI 说「把这个抽成一个组件」「写个 `UserCard` 组件」，意思是把一块界面**封装成可复用的积木**。

---

## JSX

**大白话**：在 JS 里**直接写类 HTML 的标签**。让你像写界面结构一样写代码。

```tsx
function Profile() {
  return (
    <View>
      <Text>名字：小明</Text>
      <Text>年龄：20</Text>
    </View>
  )
}
```

- 长得像 HTML，但其实是在写 JS。React Native 里用的是 RN 的组件（`View` `Text`），不是 HTML 的 `div` `p`（详见 [[06-React Native核心组件]]）。
- 标签里可以**嵌入 JS 表达式**，用 `{}` 包起来：`<Text>你好，{name}</Text>`。
- 一个组件**只能返回一个根元素**，所以外面常套一层 `<View>` 或「空片段 `<>...</>`」包裹。

---

## Props（属性）

**大白话**：**父组件传给子组件的数据**。像你给一块积木贴上不同的贴纸，它就显示不同的样子。

```tsx
// 父传子
function App() {
  return <UserCard name="小明" age={20} />
}

function UserCard({ name, age }) {   // 子组件接收 props
  return <Text>{name}，{age}岁</Text>
}
```

- `name="小明"` 这种写在标签上的，就是 **props**。
- 子组件**只读**这些值，不能改（要改的话用下面的 state，或通知父组件改）。
- 类比：props 像**函数的参数**——父组件调用子组件时「喂」进去什么，子组件就显示什么。
- AI 说「通过 props 传」「这个组件接收一个 `onPress` prop」，就是这个。

---

## State（状态）

**大白话**：组件**自己的、会变的记忆**。变了，界面就跟着变。

```tsx
function Counter() {
  const [count, setCount] = useState(0)   // 记忆：count，初始 0

  return (
    <View>
      <Text>点了 {count} 次</Text>
      <Button title="点我" onPress={() => setCount(count + 1)} />
    </View>
  )
}
```

- `count` 是当前值，`setCount` 是「改它的函数」。改了之后，React 自动**重新画一遍界面**（叫**重渲染 re-render**）。
- 和 props 的区别：props 是**外面给的**（只读），state 是**自己管的**（能改）。
- AI 说「用 state 存一下」「加个 loading 状态」，就是让你组件记住一个会变的值。
- 类比：state 是组件的**私房钱抽屉**，只有自己能开能关；props 是别人塞给你的便条。

---

## 函数组件 vs 类组件

- **函数组件**：就是一个函数返回 JSX（上面所有例子都是），**现在都用这个**。
- **类组件**：老写法，用 `class` 关键字，里面有 `this.state`、`componentDidMount` 之类。**基本不用了**，看到也不用学，AI 默认写函数组件。

AI 写给你的全是函数组件。除非你在抄很老的教程，否则不用管类组件。

---

## Hooks（钩子）

**大白话**：给函数组件「**加超能力**」的函数，名字都以 `use` 开头。最常用的就下面几个。

### useState —— 给组件加个「记忆」

```tsx
const [count, setCount] = useState(0)
//      ↑当前值  ↑改它的函数          ↑初始值
```

上面 state 那节讲过了。**最常用的 Hook，没有之一。**

### useEffect —— 副作用（干点「分外之事」）

**大白话**：让组件在**特定时机**做点事，比如「**一进来就**去请求数据」「**某个值变了**就重新执行」。

```tsx
useEffect(() => {
  // 这里面的代码在组件显示后执行
  fetchUsers()
}, [])   // 末尾 [] 表示只在「第一次」执行一次

useEffect(() => {
  // count 变了才执行
  console.log("count 变成", count)
}, [count])   // 依赖数组：监听 count
```

- 第二个参数 `[]` 叫**依赖数组**：写哪些变量，那些变量变了才重新执行；写空数组表示只执行一次。
- 「副作用（Side Effect）」= 改组件外部世界的事：请求数据、存本地、设定时器、订阅事件。
- AI 说「在 useEffect 里调接口」「加个依赖」，就是这个。

### useRef —— 记一个「不触发刷新」的值

```tsx
const timerRef = useRef(null)
timerRef.current = 123   // 改它不会让界面刷新
```

- 用来存「需要记住、但变了不用刷新界面」的东西，比如定时器 id、某个 DOM 引用。
- 比较少直接碰到，AI 用到再问。

### useMemo / useCallback —— 性能优化用

- `useMemo`：**记住一个计算结果**，依赖没变就不重算。
- `useCallback`：**记住一个函数**，依赖没变就不重建。
- 这俩是**性能优化**用的，小 App 根本用不上。AI 提了你就说「先别优化，能跑就行」。

---

## 渲染（Render）与重渲染（Re-render）

- **渲染**：React 把组件变成界面的过程。
- **重渲染**：state 或 props 变了，React 自动**重新跑一遍这个组件**，更新界面。
- 关键认知：**state 一变，界面就变**，这是 React 的核心机制，你不用手动去改界面，只管改数据。
- 性能坑：如果一个组件重渲染太频繁/太重，会卡。AI 说「这个列表卡，优化一下重渲染」，就是这问题。

---

## key（列表的身份证）

```tsx
{users.map(user => (
  <UserCard key={user.id} name={user.name} />
))}
```

- 用 `map` 渲染一列表时，每个元素要给个 **`key`**，是个**唯一标识**（通常是 id）。
- 不给会报警告。React 靠它分辨「哪一项是哪一项」，变了能高效更新。
- AI 一定会在列表里加 `key`，看到不奇怪。

---

## 受控组件（Controlled Component）

主要指**输入框**：输入框的值绑在 state 上，用户敲字 → state 变 → 输入框显示 state 的值。

```tsx
const [text, setText] = useState("")
<TextInput value={text} onChangeText={setText} />
```

- AI 说「做个受控输入框」，就是这个意思。好处是值在 state 里，随时能拿到/校验。

---

## 小结：React 的核心心智模型

1. **界面 = 数据的镜像**。你不去「操纵界面」，而是**改数据（state）**，React 自动更新界面。
2. **组件是积木**，**props 是外面给的便条**，**state 是自己的私房钱**。
3. **数据变了 → 重渲染**。useEffect 负责在「时机」上做分外之事。
4. 组件首字母大写，列表要给 key。

记住这几条，AI 写的 React 代码你就能看个七七八八。

上一篇：[[04-JavaScript与TypeScript速查]] ｜ 下一篇：[[06-React Native核心组件]]
