# LangGraph Subgraphs 子图使用指南

## 概览

本指南介绍子图的使用机制。**子图**是作为另一个图中的节点使用的图。

**子图的用途**:
- ✅ 构建多 Agent 系统
- ✅ 在多个图中重用一组节点
- ✅ 分布式开发：不同团队独立开发图的不同部分

![子图概念示意图](示意：Parent Graph → Subgraph Node → Child Graph)

---

## 核心概念

添加子图时，需要定义父图和子图如何通信：

### 两种集成方式

| 方式 | 状态共享 | 适用场景 |
|------|---------|---------|
| **从节点调用图** | 不需要共享状态 | 独立的子任务、不同状态结构 |
| **添加图作为节点** | 需要共享状态键 | 多 Agent 系统、共享消息历史 |

---

## 环境设置

```bash
npm install @langchain/langgraph
```

> **提示**: 设置 LangSmith 用于 LangGraph 开发。注册 [LangSmith](https://smith.langchain.com/) 以快速发现问题并提高 LangGraph 项目的性能。

---

## 方式 1：从节点调用图

这是实现子图的简单方式，通过在父图的节点内部调用子图。子图可以拥有与父图完全不同的状态结构（无共享键）。

**适用场景**:
- 多 Agent 系统中为每个 Agent 保留私有消息历史
- 子任务使用不同的状态结构
- 需要显式转换输入输出

### 基本实现

```javascript
import { StateGraph, StateSchema, START } from "@langchain/langgraph";
import * as z from "zod";

// 子图状态（不同的状态结构）
const SubgraphState = new StateSchema({
  bar: z.string(),
});

// 子图
const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "hi! " + state.bar };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// 父图状态
const State = new StateSchema({
  foo: z.string(),
});

// 父图：在节点中转换状态并调用子图
const builder = new StateGraph(State)
  .addNode("node1", async (state) => {
    // 1. 转换父状态到子图状态
    const subgraphOutput = await subgraph.invoke({ bar: state.foo });
    
    // 2. 转换子图输出回父状态
    return { foo: subgraphOutput.bar };
  })
  .addEdge(START, "node1");

const graph = builder.compile();
```

**执行流程**:
```
Parent State { foo: "world" }
    ↓ 转换
Subgraph State { bar: "world" }
    ↓ 子图处理
Subgraph Output { bar: "hi! world" }
    ↓ 转换
Parent State { foo: "hi! world" }
```

### 多层子图嵌套

子图本身也可以包含子图：

```javascript
// 第二层子图
const nestedSubgraphBuilder = new StateGraph(NestedState)
  .addNode("nestedNode", (state) => {
    return { baz: state.baz + "!!!" };
  })
  .addEdge(START, "nestedNode");

const nestedSubgraph = nestedSubgraphBuilder.compile();

// 第一层子图（包含嵌套子图）
const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("callNested", async (state) => {
    // 调用嵌套子图
    const result = await nestedSubgraph.invoke({ baz: state.bar });
    return { bar: result.baz };
  })
  .addEdge(START, "callNested");

const subgraph = subgraphBuilder.compile();

// 父图
const graph = new StateGraph(State)
  .addNode("callSubgraph", async (state) => {
    const result = await subgraph.invoke({ bar: state.foo });
    return { foo: result.bar };
  })
  .compile();
```

**嵌套层次**:
```
Parent Graph
  └─ Subgraph Level 1
       └─ Subgraph Level 2
```

---

## 方式 2：添加图作为节点

当父图和子图可以通过状态结构（schema）中的共享状态键（channel）通信时，可以将图作为节点添加到另一个图中。

**适用场景**:
- 多 Agent 系统中的 Agent 通过共享 `messages` 键通信
- 子图和父图使用相同或兼容的状态结构
- 不需要显式转换状态

### 基本实现

```javascript
import { StateGraph, StateSchema, START } from "@langchain/langgraph";
import * as z from "zod";

// 共享状态结构
const State = new StateSchema({
  foo: z.string(),
});

// 子图
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// 父图：直接将子图作为节点
const builder = new StateGraph(State)
  .addNode("node1", subgraph)  // 直接添加编译后的子图
  .addEdge(START, "node1");

const graph = builder.compile();
```

**关键步骤**:
1. 定义子图工作流并编译
2. 将编译后的子图传递给父图的 `.addNode()` 方法

**执行流程**:
```
Parent State { foo: "world" }
    ↓ 直接传递（共享键）
Subgraph State { foo: "world" }
    ↓ 子图处理
Subgraph Output { foo: "hi! world" }
    ↓ 直接传递
Parent State { foo: "hi! world" }
```

### 多 Agent 系统示例

```javascript
import { StateSchema, MessagesValue } from "@langchain/langgraph";

// 共享的 messages 状态
const AgentState = new StateSchema({
  messages: MessagesValue,
  currentAgent: z.string().optional(),
});

// SQL Agent 子图
const sqlAgentBuilder = new StateGraph(AgentState)
  .addNode("processSQLQuery", async (state) => {
    const response = await sqlModel.invoke(state.messages);
    return { messages: [response] };
  })
  .addEdge(START, "processSQLQuery");

const sqlAgent = sqlAgentBuilder.compile();

// Research Agent 子图
const researchAgentBuilder = new StateGraph(AgentState)
  .addNode("doResearch", async (state) => {
    const response = await researchModel.invoke(state.messages);
    return { messages: [response] };
  })
  .addEdge(START, "doResearch");

const researchAgent = researchAgentBuilder.compile();

// 主协调器图
const mainGraph = new StateGraph(AgentState)
  .addNode("sqlAgent", sqlAgent)
  .addNode("researchAgent", researchAgent)
  .addNode("router", (state) => {
    // 路由逻辑
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage.content.includes("SQL")) {
      return { currentAgent: "sqlAgent" };
    }
    return { currentAgent: "researchAgent" };
  })
  .addEdge(START, "router")
  .addConditionalEdges("router", (state) => state.currentAgent)
  .compile();
```

---

## 添加持久化

只需在编译父图时提供检查点器。LangGraph 会自动将检查点器传播到子图。

```javascript
import { StateGraph, StateSchema, START, MemorySaver } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({
  foo: z.string(),
});

// 子图
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: state.foo + "bar" };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// 父图
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });  // 自动传播到子图
```

### 子图独立记忆

如果希望子图拥有自己的记忆，可以在编译子图时使用相应的检查点器选项。这在多 Agent 系统中很有用，可以让 Agent 跟踪各自的内部消息历史。

```javascript
const subgraphBuilder = new StateGraph(...)
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```

**使用场景**:
- 每个 Agent 维护独立的对话历史
- 子任务需要独立的状态追踪
- 隔离不同子图的执行状态

---

## 查看子图状态

启用持久化后，可以通过相应方法检查图状态（检查点）。要查看子图状态，可以使用 `subgraphs` 选项。

### 基本用法

```javascript
// 查看父图状态
const state = await graph.getState(config);

// 查看子图状态
const stateWithSubgraphs = await graph.getState(config, { subgraphs: true });
```

> **警告**: 仅在中断时可用。子图状态仅在子图中断时可以查看。一旦恢复图的执行，将无法访问子图状态。

### 实战示例

```javascript
import { StateGraph, StateSchema, START, MemorySaver } from "@langchain/langgraph";

const State = new StateSchema({
  foo: z.string(),
});

// 包含中断的子图
const subgraphBuilder = new StateGraph(State)
  .addNode("processData", (state) => {
    return { foo: state.foo + " processed" };
  })
  .addNode("waitForApproval", (state) => {
    // 这个节点会中断
    return { foo: state.foo };
  })
  .addEdge(START, "processData")
  .addEdge("processData", "waitForApproval");

const subgraph = subgraphBuilder.compile({
  checkpointer: true,
  interruptBefore: ["waitForApproval"],
});

const graph = new StateGraph(State)
  .addNode("subgraphNode", subgraph)
  .addEdge(START, "subgraphNode")
  .compile({ checkpointer: new MemorySaver() });

// 执行（会在子图中断）
const config = { configurable: { thread_id: "1" } };
await graph.invoke({ foo: "test" }, config);

// 查看子图状态
const state = await graph.getState(config, { subgraphs: true });
console.log(state);
```

**输出结构**:
```javascript
{
  values: { foo: "test processed" },
  next: ["subgraphNode"],
  subgraphs: {
    subgraphNode: {
      values: { foo: "test processed" },
      next: ["waitForApproval"],
      // ... 子图的详细状态
    }
  }
}
```

---

## 流式子图输出

要在流式输出中包含来自子图的输出，可以在父图的 `.stream()` 方法中设置 `subgraphs` 选项。这将流式传输来自父图和任何子图的输出。

```javascript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    subgraphs: true,      // 启用子图输出流式传输
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

**输出格式**:

```javascript
// 父图输出
{ node1: { foo: "processed" } }

// 子图输出（带命名空间）
[
  ["node1:task_123"],                    // 命名空间：父节点路径
  { subgraphNode1: { foo: "result" } }   // 子图节点输出
]
```

### 多层嵌套流式处理

```javascript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    subgraphs: true,
    streamMode: "updates",
  }
)) {
  if (Array.isArray(chunk)) {
    const [namespace, data] = chunk;
    console.log("Subgraph output from:", namespace);
    console.log("Data:", data);
  } else {
    console.log("Parent graph output:", chunk);
  }
}
```

**命名空间层次**:
```javascript
// 父图输出
{ parentNode: { ... } }

// 第一层子图
[["parentNode:123"], { subNode: { ... } }]

// 第二层子图
[["parentNode:123", "subNode:456"], { nestedNode: { ... } }]
```

---

## 两种方式对比

| 特性 | 从节点调用图 | 添加图作为节点 |
|------|------------|--------------|
| **状态共享** | 不需要 | 需要共享键 |
| **状态转换** | 手动转换 | 自动传递 |
| **代码复杂度** | 高（需要转换逻辑） | 低（直接集成） |
| **灵活性** | 高（完全控制转换） | 中等（依赖共享状态） |
| **适用场景** | 独立子任务 | 协同工作流 |
| **持久化** | 需要显式配置 | 自动传播 |
| **调试** | 较难（状态转换） | 较易（统一状态） |

---

## 最佳实践

### 1. 选择合适的集成方式

**从节点调用图**：
```javascript
// 适用于：不同状态结构、需要转换
const node = async (state) => {
  const subResult = await subgraph.invoke({ 
    differentKey: transform(state.myKey) 
  });
  return { myKey: transformBack(subResult.differentKey) };
};
```

**添加图作为节点**：
```javascript
// 适用于：共享状态、简单集成
const graph = new StateGraph(SharedState)
  .addNode("subgraph", compiledSubgraph)
  .compile();
```

### 2. 状态设计原则

**共享键设计**：
```javascript
// 好的做法：使用通用的共享键
const State = new StateSchema({
  messages: MessagesValue,  // 多个子图可以共享
  context: z.object({...}), // 通用上下文
});

// 避免：过于具体的状态键
const State = new StateSchema({
  sqlAgentSpecificData: z.object({...}),  // 只有一个子图使用
});
```

### 3. 持久化策略

```javascript
// 父图使用数据库持久化
const parentCheckpointer = PostgresSaver.fromConnString(DB_URI);

// 子图使用独立记忆
const subgraph = subgraphBuilder.compile({ 
  checkpointer: true  // 使用父图的检查点器，但独立追踪
});

const graph = builder.compile({ 
  checkpointer: parentCheckpointer 
});
```

### 4. 错误处理

```javascript
const nodeWithSubgraph = async (state) => {
  try {
    const result = await subgraph.invoke({ 
      input: state.input 
    });
    return { output: result.output };
  } catch (error) {
    console.error("Subgraph error:", error);
    // 返回错误状态或重试
    return { error: error.message };
  }
};
```

### 5. 调试技巧

```javascript
// 使用 LangSmith 追踪
import { traceable } from "langsmith/traceable";

const nodeWithSubgraph = traceable(
  async (state) => {
    const result = await subgraph.invoke({ input: state.input });
    return { output: result.output };
  },
  { name: "SubgraphNode", tags: ["subgraph"] }
);
```

---

## 实战场景

### 场景 1：多 Agent 协作系统

```javascript
import { StateGraph, StateSchema, MessagesValue } from "@langchain/langgraph";

const AgentState = new StateSchema({
  messages: MessagesValue,
  currentTask: z.string().optional(),
});

// 代码生成 Agent
const codeAgentBuilder = new StateGraph(AgentState)
  .addNode("generateCode", async (state) => {
    const code = await codeModel.invoke(state.messages);
    return { messages: [{ role: "assistant", content: code }] };
  })
  .addNode("reviewCode", async (state) => {
    const review = await reviewModel.invoke(state.messages);
    return { messages: [{ role: "assistant", content: review }] };
  })
  .addEdge(START, "generateCode")
  .addEdge("generateCode", "reviewCode");

const codeAgent = codeAgentBuilder.compile({ checkpointer: true });

// 文档编写 Agent
const docAgentBuilder = new StateGraph(AgentState)
  .addNode("writeDoc", async (state) => {
    const doc = await docModel.invoke(state.messages);
    return { messages: [{ role: "assistant", content: doc }] };
  })
  .addEdge(START, "writeDoc");

const docAgent = docAgentBuilder.compile({ checkpointer: true });

// 主协调器
const mainGraph = new StateGraph(AgentState)
  .addNode("codeAgent", codeAgent)
  .addNode("docAgent", docAgent)
  .addNode("coordinator", (state) => {
    const lastMsg = state.messages[state.messages.length - 1];
    if (lastMsg.content.includes("code")) {
      return { currentTask: "codeAgent" };
    }
    return { currentTask: "docAgent" };
  })
  .addEdge(START, "coordinator")
  .addConditionalEdges("coordinator", (state) => state.currentTask)
  .compile({ checkpointer: new MemorySaver() });
```

### 场景 2：数据处理管道

```javascript
// ETL 子图：提取、转换、加载
const ETLState = new StateSchema({
  rawData: z.string(),
  processedData: z.object({...}),
});

const etlSubgraph = new StateGraph(ETLState)
  .addNode("extract", (state) => {
    const data = parseRawData(state.rawData);
    return { processedData: data };
  })
  .addNode("transform", (state) => {
    const transformed = transformData(state.processedData);
    return { processedData: transformed };
  })
  .addNode("load", async (state) => {
    await saveToDatabase(state.processedData);
    return {};
  })
  .addEdge(START, "extract")
  .addEdge("extract", "transform")
  .addEdge("transform", "load")
  .compile();

// 主数据处理图
const MainState = new StateSchema({
  inputFiles: z.array(z.string()),
  results: z.array(z.object({...})),
});

const dataProcessingGraph = new StateGraph(MainState)
  .addNode("processFile", async (state) => {
    // 为每个文件调用 ETL 子图
    const results = [];
    for (const file of state.inputFiles) {
      const result = await etlSubgraph.invoke({ 
        rawData: await readFile(file) 
      });
      results.push(result.processedData);
    }
    return { results };
  })
  .addEdge(START, "processFile")
  .compile();
```

### 场景 3：条件子图执行

```javascript
const State = new StateSchema({
  userQuery: z.string(),
  requiresComplexProcessing: z.boolean(),
  result: z.string(),
});

// 简单处理子图
const simpleProcessing = new StateGraph(State)
  .addNode("quickResponse", (state) => {
    return { result: `Quick answer to: ${state.userQuery}` };
  })
  .addEdge(START, "quickResponse")
  .compile();

// 复杂处理子图
const complexProcessing = new StateGraph(State)
  .addNode("deepAnalysis", async (state) => {
    const analysis = await deepModel.invoke(state.userQuery);
    return { result: analysis };
  })
  .addNode("verification", async (state) => {
    const verified = await verifyResult(state.result);
    return { result: verified };
  })
  .addEdge(START, "deepAnalysis")
  .addEdge("deepAnalysis", "verification")
  .compile();

// 主图：根据条件选择子图
const mainGraph = new StateGraph(State)
  .addNode("analyze", (state) => {
    const isComplex = state.userQuery.length > 100 || 
                      state.userQuery.includes("detailed");
    return { requiresComplexProcessing: isComplex };
  })
  .addNode("simpleProcessing", simpleProcessing)
  .addNode("complexProcessing", complexProcessing)
  .addEdge(START, "analyze")
  .addConditionalEdges("analyze", (state) => 
    state.requiresComplexProcessing ? "complexProcessing" : "simpleProcessing"
  )
  .compile();
```

---

## 性能考虑

### 1. 子图调用开销

```javascript
// 避免：频繁调用小子图
for (const item of items) {
  await subgraph.invoke({ item });  // 每次都有序列化/反序列化开销
}

// 优化：批量处理
const results = await subgraph.invoke({ items });
```

### 2. 状态传递效率

```javascript
// 使用共享键（更高效）
const graph = new StateGraph(SharedState)
  .addNode("subgraph", compiledSubgraph)  // 直接传递，无需序列化转换
  .compile();

// vs 手动转换（较慢）
.addNode("subgraph", async (state) => {
  const transformed = transformState(state);  // 额外开销
  return await subgraph.invoke(transformed);
});
```

### 3. 持久化开销

```javascript
// 每个子图独立持久化会增加数据库操作
const subgraph1 = builder1.compile({ checkpointer: true });
const subgraph2 = builder2.compile({ checkpointer: true });

// 如果不需要独立追踪，共享父图的检查点器
const subgraph1 = builder1.compile();
const subgraph2 = builder2.compile();
const mainGraph = mainBuilder.compile({ checkpointer });
```

---

## 常见问题

**Q: 子图可以有几层嵌套？**

A: 理论上无限制，但实际建议不超过 3-4 层，否则调试和维护会变得困难。

**Q: 父图和子图能否使用不同的检查点器？**

A: 可以，但不推荐。通常使用相同的检查点器类型，子图通过 `checkpointer: true` 使用父图的检查点器。

**Q: 如何在子图中访问父图的状态？**

A: 
- 方式 1：使用共享键（添加图作为节点）
- 方式 2：在调用节点中手动传递

```javascript
.addNode("callSubgraph", async (state, config) => {
  // 可以通过 config 传递额外信息
  return await subgraph.invoke(
    { subState: state.parentKey },
    { ...config, configurable: { parentInfo: state.metadata } }
  );
});
```

**Q: 子图可以修改父图的状态吗？**

A: 
- 添加图作为节点：是的，通过共享键自动合并
- 从节点调用图：取决于节点的返回值转换

**Q: 如何调试子图？**

A: 
1. 使用 LangSmith 追踪
2. 启用 `debug` 流式模式
3. 查看子图状态（需要中断）
4. 单独测试子图

```javascript
// 单独测试子图
const result = await subgraph.invoke(testInput);
console.log(result);
```

**Q: 子图中的错误如何处理？**

A: 错误会向上传播到父图。建议在调用节点中捕获：

```javascript
.addNode("callSubgraph", async (state) => {
  try {
    return await subgraph.invoke({ input: state.input });
  } catch (error) {
    return { error: error.message, fallbackResult: "default" };
  }
});
```

---

## 相关资源

- [Graph API 图 API](https://docs.langchain.com/oss/javascript/langgraph/graph-api)
- [Multi-Agent Systems 多 Agent 系统](https://docs.langchain.com/oss/javascript/langchain/multi-agent)
- [Persistence 持久化](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- [Streaming 流式处理](https://docs.langchain.com/oss/javascript/langgraph/streaming)
- [Memory 记忆管理](https://docs.langchain.com/oss/javascript/langgraph/add-memory)
