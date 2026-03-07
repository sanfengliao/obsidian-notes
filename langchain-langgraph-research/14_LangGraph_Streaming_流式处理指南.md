# LangGraph Streaming 流式处理指南

## 概览

LangGraph 实现了强大的流式处理系统，提供实时更新。流式处理对于提升基于 LLM 应用的响应能力至关重要。通过在完整响应就绪之前逐步显示输出，流式处理显著改善了用户体验，特别是在处理 LLM 延迟时。

**LangGraph 流式处理能力**:
- ✅ **流式图状态**: 使用 `updates` 和 `values` 模式获取状态更新
- ✅ **流式子图输出**: 包含父图和嵌套子图的输出
- ✅ **流式 LLM tokens**: 从任何地方捕获 token 流（节点内、子图、工具）
- ✅ **流式自定义数据**: 直接从工具函数发送自定义更新或进度信号
- ✅ **多种流式模式**: `values`（完整状态）、`updates`（状态增量）、`messages`（LLM tokens + 元数据）、`custom`（任意用户数据）、`debug`（详细追踪）

---

## 支持的流式模式

将以下一个或多个流式模式作为参数传递给 `.stream()` 方法：

| 模式 | 说明 |
|------|------|
| **values** | 在图的每一步后流式传输状态的完整值 |
| **updates** | 在图的每一步后流式传输状态的更新。如果同一步中有多个更新（例如，运行多个节点），这些更新会分别流式传输 |
| **custom** | 从图节点内部流式传输自定义数据 |
| **messages** | 从任何调用 LLM 的图节点流式传输 2 元组（LLM token、元数据） |
| **debug** | 在图执行过程中流式传输尽可能多的信息 |

---

## 基本用法

LangGraph 图通过 `.stream()` 方法公开流式输出作为迭代器。

```javascript
for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  console.log(chunk);
}
```

---

## 流式多种模式

可以将数组作为 `streamMode` 参数传递，一次流式传输多种模式。

流式输出将是 `[mode, chunk]` 的元组，其中：
- `mode`: 流式模式的名称
- `chunk`: 该模式流式传输的数据

```javascript
for await (const [mode, chunk] of await graph.stream(inputs, {
  streamMode: ["updates", "custom"],
})) {
  console.log(`Mode: ${mode}`, chunk);
}
```

---

## 流式图状态

使用 `updates` 和 `values` 流式模式在图执行时流式传输图的状态。

### 模式对比

- **`updates`**: 流式传输图每一步后的状态更新
- **`values`**: 流式传输图每一步后的完整状态值

### 示例：基础状态流式处理

```javascript
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import { z } from "zod/v4";

const State = new StateSchema({
  topic: z.string(),
  joke: z.string(),
});

const graph = new StateGraph(State)
  .addNode("refineTopic", (state) => {
    return { topic: state.topic + " and cats" };
  })
  .addNode("generateJoke", (state) => {
    return { joke: `This is a joke about ${state.topic}` };
  })
  .addEdge(START, "refineTopic")
  .addEdge("refineTopic", "generateJoke")
  .addEdge("generateJoke", END)
  .compile();
```

### 使用 `updates` 模式

仅流式传输节点在每一步后返回的状态更新。流式输出包括节点名称以及更新内容。

```javascript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "updates" }
)) {
  console.log(chunk);
}
```

**输出示例**:
```javascript
{ refineTopic: { topic: 'ice cream and cats' } }
{ generateJoke: { joke: 'This is a joke about ice cream and cats' } }
```

### 使用 `values` 模式

流式传输每一步后的完整状态值。

```javascript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "values" }
)) {
  console.log(chunk);
}
```

**输出示例**:
```javascript
{ topic: 'ice cream', joke: '' }
{ topic: 'ice cream and cats', joke: '' }
{ topic: 'ice cream and cats', joke: 'This is a joke about ice cream and cats' }
```

---

## 流式子图输出

要在流式输出中包含子图的输出，可以在父图的 `.stream()` 方法中设置 `subgraphs: true`。这将流式传输来自父图和任何子图的输出。

输出将作为元组 `[namespace, data]` 流式传输，其中：
- `namespace`: 包含调用子图的节点路径的元组，例如 `["parent_node:<task_id>", "child_node:<task_id>"]`
- `data`: 流式传输的数据

```javascript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    // 设置 subgraphs: true 以流式传输来自子图的输出
    subgraphs: true,
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

**输出示例**:
```javascript
// 父图的输出
{ parent_node: { ... } }

// 子图的输出（带命名空间）
[ ["parent_node:123"], { child_node: { ... } } ]
```

---

## 调试模式

使用 `debug` 流式模式在图执行过程中流式传输尽可能多的信息。流式输出包括节点名称以及完整状态。

```javascript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "debug" }
)) {
  console.log(chunk);
}
```

**用途**:
- 深度调试
- 理解执行流程
- 检查中间状态

---

## 流式 LLM Tokens

使用 `messages` 流式模式从图的任何部分（包括节点、工具、子图或任务）逐 token 流式传输大型语言模型（LLM）的输出。

**流式输出格式**: `[message_chunk, metadata]` 元组

其中：
- `message_chunk`: LLM 的 token 或消息片段
- `metadata`: 包含图节点和 LLM 调用详细信息的字典

### 基本示例

```javascript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, StateSchema, GraphNode, START } from "@langchain/langgraph";
import * as z from "zod";

const MyState = new StateSchema({
  topic: z.string(),
  joke: z.string().default(""),
});

const model = new ChatOpenAI({ model: "gpt-4o-mini" });

const callModel: GraphNode<typeof MyState> = async (state) => {
  // 调用 LLM 生成关于主题的笑话
  // 注意：即使 LLM 使用 .invoke 而不是 .stream 运行，也会发出消息事件
  const modelResponse = await model.invoke([
    { role: "user", content: `Generate a joke about ${state.topic}` },
  ]);
  return { joke: modelResponse.content };
};

const graph = new StateGraph(MyState)
  .addNode("callModel", callModel)
  .addEdge(START, "callModel")
  .compile();

// "messages" 流式模式返回元组 [messageChunk, metadata] 的迭代器
for await (const [messageChunk, metadata] of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "messages" }
)) {
  if (messageChunk.content) {
    console.log(messageChunk.content + "|");
  }
}
```

**输出示例**:
```
Why|did|the|ice|cream|go|to|therapy|?|Because|it|had|too|many|sprinkles|of|anxiety|!|
```

> **注意**: 即使使用 `.invoke()` 而不是 `.stream()` 调用 LLM，也会流式传输 tokens。

---

## 过滤流式输出

### 按节点过滤

要仅从特定节点流式传输 tokens，使用 `streamMode: "messages"` 并通过流式元数据中的 `langgraph_node` 字段过滤输出：

```javascript
// "messages" 流式模式返回元组 [messageChunk, metadata]
for await (const [msg, metadata] of await graph.stream(
  inputs,
  { streamMode: "messages" }
)) {
  // 通过元数据中的 langgraph_node 字段过滤流式 tokens
  // 仅包含来自指定节点的 tokens
  if (msg.content && metadata.langgraph_node === "some_node_name") {
    console.log(msg.content);
  }
}
```

### 按 LLM 调用过滤

可以将 `tags` 与 LLM 调用关联，以通过 LLM 调用过滤流式 tokens。

```javascript
import { ChatOpenAI } from "@langchain/openai";

// model1 标记为 "joke"
const model1 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['joke']
});

// model2 标记为 "poem"
const model2 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['poem']
});

const graph = // ... 定义使用这些 LLMs 的图

// streamMode 设置为 "messages" 以流式传输 LLM tokens
// metadata 包含有关 LLM 调用的信息，包括 tags
for await (const [msg, metadata] of await graph.stream(
  { topic: "cats" },
  { streamMode: "messages" }
)) {
  // 通过元数据中的 tags 字段过滤流式 tokens
  // 仅包含来自带有 "joke" 标签的 LLM 调用的 tokens
  if (metadata.tags?.includes("joke")) {
    console.log(msg.content + "|");
  }
}
```

**用途**:
- 区分多个 LLM 调用的输出
- 在复杂图中路由特定 LLM 的输出
- 实现选择性输出显示

---

## 流式自定义数据

要从 LangGraph 节点或工具内部发送自定义用户定义的数据，按以下步骤操作：

1. 使用 `LangGraphRunnableConfig` 中的 `writer` 参数发出自定义数据
2. 调用 `.stream()` 时设置 `streamMode: "custom"` 以在流中获取自定义数据
3. 可以组合多种模式（例如，`["updates", "custom"]`），但至少一个必须是 `"custom"`

### 从节点流式传输

```javascript
import { StateGraph, StateSchema, GraphNode, START, LangGraphRunnableConfig } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({
  query: z.string(),
  answer: z.string(),
});

const node: GraphNode<typeof State> = async (state, config) => {
  // 使用 writer 发出自定义键值对（例如，进度更新）
  config.writer({ custom_key: "Generating custom data inside node" });
  return { answer: "some data" };
};

const graph = new StateGraph(State)
  .addNode("node", node)
  .addEdge(START, "node")
  .compile();

const inputs = { query: "example" };

// 设置 streamMode: "custom" 以在流中接收自定义数据
for await (const chunk of await graph.stream(inputs, { streamMode: "custom" })) {
  console.log(chunk);
}
```

**输出示例**:
```javascript
{ custom_key: "Generating custom data inside node" }
```

### 从工具流式传输

同样的方式适用于工具：

```javascript
import { tool } from "@langchain/core/tools";
import * as z from "zod";

const myTool = tool(
  async ({ query }, config) => {
    // 使用 writer 发出自定义数据
    config.writer({ tool_progress: "50% complete" });
    
    // 执行工具逻辑
    const result = await someOperation(query);
    
    config.writer({ tool_progress: "100% complete" });
    return result;
  },
  {
    name: "myTool",
    description: "A tool that reports progress",
    schema: z.object({
      query: z.string(),
    }),
  }
);
```

**用途**:
- 进度更新
- 中间结果
- 调试信息
- 自定义指标

---

## 与任意 LLM 一起使用

可以使用 `streamMode: "custom"` 从任何 LLM API 流式传输数据——即使该 API 没有实现 LangChain 聊天模型接口。

这允许你集成原始 LLM 客户端或提供自己流式接口的外部服务，使 LangGraph 对于自定义设置非常灵活。

```javascript
import { StateGraph, GraphNode, StateSchema } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({ result: z.string() });

const callArbitraryModel: GraphNode<typeof State> = async (state, config) => {
  // 调用任意模型并流式传输输出的示例节点
  // 假设你有一个产生 chunks 的流式客户端
  
  // 使用自定义流式客户端生成 LLM tokens
  for await (const chunk of yourCustomStreamingClient(state.topic)) {
    // 使用 writer 将自定义数据发送到流
    config.writer({ custom_llm_chunk: chunk });
  }
  
  return { result: "completed" };
};

const graph = new StateGraph(State)
  .addNode("callArbitraryModel", callArbitraryModel)
  // 根据需要添加其他节点和边
  .compile();

// 设置 streamMode: "custom" 以在流中接收自定义数据
for await (const chunk of await graph.stream(
  { topic: "cats" },
  { streamMode: "custom" }
)) {
  // chunk 将包含从 llm 流式传输的自定义数据
  console.log(chunk);
}
```

**适用场景**:
- 非 LangChain 集成的 LLM
- 自定义 API 客户端
- 专有模型服务
- 需要特殊处理的流式输出

---

## 禁用特定模型的流式处理

如果应用程序混合使用支持流式处理的模型和不支持的模型，可能需要显式禁用不支持流式处理的模型的流式处理。

在初始化模型时设置 `streaming: false`。

```javascript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "o1-preview",
  // 设置 streaming: false 以禁用聊天模型的流式处理
  streaming: false,
});
```

> **注意**: 并非所有聊天模型集成都支持 `streaming` 参数。如果你的模型不支持它，请改用 `disableStreaming: true`。此参数通过基类在所有聊天模型上可用。

---

## 流式处理模式对比

| 模式 | 用途 | 输出格式 | 适用场景 |
|------|------|---------|---------|
| **values** | 完整状态 | 每步的完整状态对象 | 需要完整状态快照 |
| **updates** | 状态增量 | 节点名称 + 状态更新 | 跟踪状态变化 |
| **messages** | LLM tokens | `[token, metadata]` | 实时显示 LLM 输出 |
| **custom** | 自定义数据 | 用户定义的对象 | 进度更新、自定义指标 |
| **debug** | 调试信息 | 详细执行信息 | 深度调试和分析 |

---

## 最佳实践

### 1. 选择合适的模式

- **实时 UI 更新**: 使用 `messages` 模式
- **状态追踪**: 使用 `updates` 或 `values` 模式
- **进度报告**: 使用 `custom` 模式
- **调试**: 使用 `debug` 模式

### 2. 组合多种模式

```javascript
for await (const [mode, chunk] of await graph.stream(inputs, {
  streamMode: ["updates", "messages", "custom"],
})) {
  switch (mode) {
    case "updates":
      // 处理状态更新
      break;
    case "messages":
      // 处理 LLM tokens
      break;
    case "custom":
      // 处理自定义数据
      break;
  }
}
```

### 3. 过滤输出

根据需要使用 `metadata` 中的 `langgraph_node` 或 `tags` 字段过滤：

```javascript
for await (const [msg, metadata] of await graph.stream(
  inputs,
  { streamMode: "messages" }
)) {
  // 只处理特定节点的输出
  if (metadata.langgraph_node === "important_node") {
    processToken(msg.content);
  }
}
```

### 4. 处理子图

处理复杂图时，启用子图流式处理：

```javascript
for await (const chunk of await graph.stream(inputs, {
  subgraphs: true,
  streamMode: "updates",
})) {
  // 检查是否为子图输出
  if (Array.isArray(chunk)) {
    const [namespace, data] = chunk;
    console.log("Subgraph output:", namespace, data);
  } else {
    console.log("Parent graph output:", chunk);
  }
}
```

### 5. 错误处理

始终处理流式处理中的错误：

```javascript
try {
  for await (const chunk of await graph.stream(inputs, {
    streamMode: "updates",
  })) {
    console.log(chunk);
  }
} catch (error) {
  console.error("Streaming error:", error);
}
```

---

## 实战场景

### 场景 1：聊天界面实时响应

```javascript
// 实时显示聊天机器人响应
for await (const [msg, metadata] of await chatGraph.stream(
  { message: userInput },
  { streamMode: "messages" }
)) {
  if (msg.content) {
    // 逐 token 更新 UI
    updateChatUI(msg.content);
  }
}
```

### 场景 2：长时间任务进度追踪

```javascript
// 报告复杂处理流程的进度
const processNode: GraphNode<typeof State> = async (state, config) => {
  config.writer({ progress: 0, status: "Starting..." });
  
  // 步骤 1
  await step1();
  config.writer({ progress: 33, status: "Step 1 complete" });
  
  // 步骤 2
  await step2();
  config.writer({ progress: 66, status: "Step 2 complete" });
  
  // 步骤 3
  await step3();
  config.writer({ progress: 100, status: "Complete" });
  
  return { result: "done" };
};

// 显示进度
for await (const chunk of await graph.stream(inputs, {
  streamMode: "custom",
})) {
  updateProgressBar(chunk.progress);
  updateStatusText(chunk.status);
}
```

### 场景 3：多 LLM 输出区分

```javascript
// 区分不同 LLM 的输出
const writer = new ChatOpenAI({ model: "gpt-4o", tags: ["writer"] });
const editor = new ChatOpenAI({ model: "gpt-4o", tags: ["editor"] });

for await (const [msg, metadata] of await graph.stream(
  inputs,
  { streamMode: "messages" }
)) {
  if (metadata.tags?.includes("writer")) {
    displayInWriterPane(msg.content);
  } else if (metadata.tags?.includes("editor")) {
    displayInEditorPane(msg.content);
  }
}
```

### 场景 4：调试复杂图

```javascript
// 使用 debug 模式追踪执行
for await (const chunk of await graph.stream(
  inputs,
  { streamMode: "debug" }
)) {
  logToDebugger(chunk);
  if (chunk.type === "error") {
    console.error("Error detected:", chunk.error);
  }
}
```

---

## 性能考虑

### 1. 流式模式开销

不同模式的性能影响：
- **values**: 中等开销（需要序列化完整状态）
- **updates**: 低开销（仅序列化增量）
- **messages**: 低开销（按 token 流式）
- **custom**: 最低开销（完全由用户控制）
- **debug**: 高开销（大量数据）

### 2. 网络考虑

流式处理会增加网络往返次数：
- 在本地开发中影响较小
- 在高延迟网络中可能影响性能
- 考虑批量处理 tokens 以减少请求

### 3. 内存管理

- `values` 模式会保留完整状态历史
- 对于大状态对象，优先使用 `updates`
- 及时处理流式数据，避免累积

---

## 常见问题

**Q: 能否同时使用所有流式模式？**

A: 可以，但会增加开销。建议只启用需要的模式。

**Q: 流式处理会影响图执行速度吗？**

A: 对执行本身影响很小，主要影响数据传输。

**Q: 如何处理流式中的错误？**

A: 使用 try-catch 包裹流式循环，错误会在迭代器中抛出。

**Q: 子图流式输出的命名空间格式是什么？**

A: `["parent_node:<task_id>", "child_node:<task_id>"]`，表示调用路径。

**Q: `messages` 模式支持所有 LLM 吗？**

A: 仅支持实现 LangChain 聊天模型接口的 LLM。对于其他 LLM，使用 `custom` 模式。

**Q: 如何在流式处理中实现超时？**

A: 使用 JavaScript 的 `Promise.race` 或 `AbortController`:

```javascript
const controller = new AbortController();
setTimeout(() => controller.abort(), 30000); // 30 秒超时

try {
  for await (const chunk of await graph.stream(inputs, {
    streamMode: "updates",
    signal: controller.signal,
  })) {
    console.log(chunk);
  }
} catch (error) {
  if (error.name === 'AbortError') {
    console.log("Stream timed out");
  }
}
```

---

## 相关资源

- [LangGraph Quickstart](https://docs.langchain.com/oss/javascript/langgraph/quickstart)
- [Workflows 和 Agents](https://docs.langchain.com/oss/javascript/langgraph/workflows-agents)
- [Persistence 持久化](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- [LangSmith 追踪](https://smith.langchain.com/)
