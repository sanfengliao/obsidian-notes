# LangGraph Quickstart 快速入门

## 概览

本快速入门演示如何使用 LangGraph 构建一个计算器 Agent。LangGraph 提供两种 API：

- **Graph API**: 将 Agent 定义为节点和边的图
- **Functional API**: 将 Agent 定义为单个函数

本指南使用 **Graph API** 方法。

**前置要求**:
- 设置 Claude (Anthropic) 账户并获取 API 密钥
- 在终端中设置 `ANTHROPIC_API_KEY` 环境变量

---

## 核心概念

### LangGraph 是什么？

LangGraph 是一个用于构建状态化、多步骤应用的框架。它将应用建模为**图**：
- **节点（Nodes）**: 执行操作的函数
- **边（Edges）**: 连接节点的路径
- **状态（State）**: 在整个执行过程中持久化的数据

### 为什么使用 LangGraph？

- **清晰的控制流**: 显式定义执行路径
- **状态管理**: 自动处理状态更新和持久化
- **灵活性**: 支持循环、条件分支和并行执行
- **可观察性**: 易于调试和追踪

---

## 构建计算器 Agent

我们将构建一个能执行基本算术运算的 Agent。Agent 循环如下：

1. 用户提出数学问题
2. LLM 决定调用哪个工具
3. 执行工具并返回结果
4. LLM 使用结果生成最终答案

---

## 步骤 1：定义工具和模型

首先定义工具（加法、乘法、除法）和模型。

```javascript
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import * as z from "zod";

// 初始化模型
const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  temperature: 0,
});

// 定义加法工具
const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// 定义乘法工具
const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// 定义除法工具
const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// 创建工具集合
const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
};
const tools = Object.values(toolsByName);

// 将工具绑定到模型
const modelWithTools = model.bindTools(tools);
```

**关键点**:
- 每个工具都有清晰的名称、描述和参数定义
- 使用 Zod schema 定义参数类型
- `bindTools` 让模型知道可用的工具

---

## 步骤 2：定义状态

图的状态用于存储消息和 LLM 调用次数。

```javascript
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  ReducedValue,
  GraphNode,
  ConditionalEdgeRouter,
  START,
  END,
} from "@langchain/langgraph";
import { z } from "zod/v4";

const MessagesState = new StateSchema({
  // messages: 使用内置 reducer 追加消息
  messages: MessagesValue,
  
  // llmCalls: 使用自定义 reducer 累积计数
  llmCalls: new ReducedValue(
    z.number().default(0),
    { reducer: (x, y) => x + y }
  ),
});
```

**状态特性**:
- **持久化**: 状态在 Agent 执行过程中保持
- **Reducer**: 定义如何合并状态更新
  - `MessagesValue`: 内置 reducer，追加新消息
  - `ReducedValue`: 自定义 reducer，这里用于累加计数

---

## 步骤 3：定义模型节点

模型节点调用 LLM 并决定是否调用工具。

```javascript
import { SystemMessage } from "@langchain/core/messages";

const llmCall: GraphNode<typeof MessagesState> = async (state) => {
  // 调用模型
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    ),
    ...state.messages,
  ]);
  
  // 返回状态更新
  return {
    messages: [response],
    llmCalls: 1,  // 增加 LLM 调用计数
  };
};
```

**节点职责**:
- 接收当前状态
- 调用模型生成响应
- 返回状态更新（新消息 + 调用计数）

---

## 步骤 4：定义工具节点

工具节点执行 LLM 请求的工具。

```javascript
import { AIMessage, ToolMessage } from "@langchain/core/messages";

const toolNode: GraphNode<typeof MessagesState> = async (state) => {
  // 获取最后一条消息
  const lastMessage = state.messages.at(-1);

  // 检查是否为 AI 消息
  if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
    return { messages: [] };
  }

  // 执行所有工具调用
  const result: ToolMessage[] = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  // 返回工具结果
  return { messages: result };
};
```

**工具执行流程**:
1. 从最后一条 AI 消息中提取工具调用
2. 逐个执行工具
3. 将结果包装为 `ToolMessage`
4. 返回所有工具结果

---

## 步骤 5：定义结束逻辑

条件边函数决定是继续调用工具还是结束。

```javascript
const shouldContinue: ConditionalEdgeRouter<typeof MessagesState, "toolNode"> = (state) => {
  const lastMessage = state.messages.at(-1);

  // 检查是否为 AIMessage
  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  // 如果 LLM 进行了工具调用，则执行工具
  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  // 否则结束（回复用户）
  return END;
};
```

**决策逻辑**:
- 有工具调用 → 路由到 `toolNode`
- 无工具调用 → 结束执行（`END`）

---

## 步骤 6：构建和编译 Agent

使用 `StateGraph` 构建图并编译。

```javascript
const agent = new StateGraph(MessagesState)
  // 添加节点
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  
  // 添加边
  .addEdge(START, "llmCall")                          // 开始 → LLM 调用
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])  // LLM → 工具/结束
  .addEdge("toolNode", "llmCall")                     // 工具 → LLM 调用
  
  // 编译图
  .compile();
```

**图结构**:
```
START → llmCall → [工具调用?] → toolNode → llmCall
                      ↓ [无工具]
                     END
```

### 执行 Agent

```javascript
import { HumanMessage } from "@langchain/core/messages";

const result = await agent.invoke({
  messages: [new HumanMessage("Add 3 and 4.")],
});

// 打印所有消息
for (const message of result.messages) {
  console.log(`[${message.type}]: ${message.text}`);
}
```

**输出示例**:
```
[human]: Add 3 and 4.
[ai]: I'll add those numbers for you.
[tool]: 7
[ai]: The sum of 3 and 4 is 7.
```

---

## 完整代码示例

```javascript
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import {
  StateGraph,
  StateSchema,
  MessagesValue,
  ReducedValue,
  START,
  END,
} from "@langchain/langgraph";
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";
import * as z from "zod";

// 1. 定义工具和模型
const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  temperature: 0,
});

const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
};
const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools);

// 2. 定义状态
const MessagesState = new StateSchema({
  messages: MessagesValue,
  llmCalls: new ReducedValue(
    z.number().default(0),
    { reducer: (x, y) => x + y }
  ),
});

// 3. 定义模型节点
const llmCall = async (state) => {
  const response = await modelWithTools.invoke([
    new SystemMessage(
      "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    ),
    ...state.messages,
  ]);
  return {
    messages: [response],
    llmCalls: 1,
  };
};

// 4. 定义工具节点
const toolNode = async (state) => {
  const lastMessage = state.messages.at(-1);

  if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
    return { messages: [] };
  }

  const result = [];
  for (const toolCall of lastMessage.tool_calls ?? []) {
    const tool = toolsByName[toolCall.name];
    const observation = await tool.invoke(toolCall);
    result.push(observation);
  }

  return { messages: result };
};

// 5. 定义结束逻辑
const shouldContinue = (state) => {
  const lastMessage = state.messages.at(-1);

  if (!lastMessage || !AIMessage.isInstance(lastMessage)) {
    return END;
  }

  if (lastMessage.tool_calls?.length) {
    return "toolNode";
  }

  return END;
};

// 6. 构建和编译
const agent = new StateGraph(MessagesState)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge(START, "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
  .addEdge("toolNode", "llmCall")
  .compile();

// 执行
const result = await agent.invoke({
  messages: [new HumanMessage("Add 3 and 4.")],
});

for (const message of result.messages) {
  console.log(`[${message.type}]: ${message.text}`);
}
```

---

## 核心概念总结

### StateGraph

`StateGraph` 是构建图的主要类：

```javascript
new StateGraph(stateSchema)
  .addNode(name, function)           // 添加节点
  .addEdge(from, to)                 // 添加固定边
  .addConditionalEdges(from, router, targets)  // 添加条件边
  .compile()                         // 编译图
```

### 节点类型

- **函数节点**: 执行逻辑的异步函数
- **参数**: 接收当前状态
- **返回值**: 状态更新对象

### 边类型

- **固定边**: 始终从一个节点到另一个节点
- **条件边**: 基于状态动态路由

### 特殊节点

- **START**: 图的入口点
- **END**: 图的出口点

---

## 调试和追踪

### 使用 LangSmith

LangSmith 提供可视化追踪和调试：

```javascript
// 设置 LangSmith 环境变量
process.env.LANGCHAIN_TRACING_V2 = "true";
process.env.LANGCHAIN_API_KEY = "your-api-key";

// 执行会自动追踪
const result = await agent.invoke({
  messages: [new HumanMessage("Add 3 and 4.")],
});
```

访问 LangSmith 查看：
- 每个节点的执行
- 消息流动
- 工具调用
- 状态变化

---

## 常见问题

**Q: Graph API 和 Functional API 有什么区别？**

A: 
- **Graph API**: 显式定义节点和边，适合复杂流程
- **Functional API**: 定义单个函数，LangGraph 自动处理循环，适合简单用例

**Q: 为什么需要 StateSchema？**

A: StateSchema 定义状态结构和更新规则（reducer），确保状态正确合并。

**Q: Reducer 是什么？**

A: Reducer 定义如何合并状态更新。例如：
- 消息：追加新消息
- 计数：累加数值

**Q: 如何添加更多工具？**

A: 定义新工具并添加到 `toolsByName`：
```javascript
const subtract = tool(({ a, b }) => a - b, { ... });
const toolsByName = { add, multiply, divide, subtract };
```

**Q: 能否有多个条件边？**

A: 可以。每个节点可以有多个出边，包括多个条件边。

**Q: 如何处理错误？**

A: 在节点中使用 try-catch：
```javascript
const llmCall = async (state) => {
  try {
    const response = await modelWithTools.invoke(...);
    return { messages: [response] };
  } catch (error) {
    return { 
      messages: [new AIMessage("Error occurred")],
      error: error.message 
    };
  }
};
```

---

## 下一步

- 学习 [Graph API 详解](https://docs.langchain.com/oss/javascript/langgraph/graph-api)
- 了解 [状态管理](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- 探索 [高级模式](https://docs.langchain.com/oss/javascript/langgraph/patterns)
- 查看 [完整示例](https://docs.langchain.com/oss/javascript/langgraph/examples)

---

## 相关资源

- [LangGraph 概述](https://docs.langchain.com/oss/javascript/langgraph/overview)
- [Graph API 参考](https://reference.langchain.com/javascript/classes/_langchain_langgraph.index.StateGraph.html)
- [LangSmith 追踪](https://docs.langchain.com/langsmith/trace-with-langgraph)
- [工具定义](https://docs.langchain.com/oss/javascript/langchain/tools)
