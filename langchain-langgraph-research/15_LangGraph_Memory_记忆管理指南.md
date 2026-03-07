# LangGraph Memory 记忆管理指南

## 概览

AI 应用需要**记忆**来在多次交互中共享上下文。在 LangGraph 中，可以添加两种类型的记忆：

- **短期记忆（Short-term Memory）**: 作为 Agent 状态的一部分，实现多轮对话
- **长期记忆（Long-term Memory）**: 跨会话存储用户特定或应用级数据

---

## 添加短期记忆

短期记忆（线程级持久化）使 Agent 能够跟踪多轮对话。

### 基本用法

```javascript
import { MemorySaver, StateGraph } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const builder = new StateGraph(...);
const graph = builder.compile({ checkpointer });

await graph.invoke(
  { messages: [{ role: "user", content: "hi! i am Bob" }] },
  { configurable: { thread_id: "1" } }
);
```

**关键点**:
- 使用 `MemorySaver` 创建检查点器（checkpointer）
- 编译图时传入 `checkpointer`
- 调用时通过 `thread_id` 隔离不同对话线程

### 在生产环境使用

在生产环境中，使用数据库支持的检查点器：

```javascript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
const checkpointer = PostgresSaver.fromConnString(DB_URI);

const builder = new StateGraph(...);
const graph = builder.compile({ checkpointer });
```

**支持的数据库**:
- PostgreSQL
- Redis
- 其他自定义实现

### 在子图中使用

如果图包含子图，只需在编译父图时提供检查点器。LangGraph 会自动将检查点器传播到子图。

```javascript
import { StateGraph, StateSchema, START, MemorySaver } from "@langchain/langgraph";
import { z } from "zod/v4";

const State = new StateSchema({ foo: z.string() });

const subgraphBuilder = new StateGraph(State)
  .addNode("subgraph_node_1", (state) => {
    return { foo: state.foo + "bar" };
  })
  .addEdge(START, "subgraph_node_1");
const subgraph = subgraphBuilder.compile();

const builder = new StateGraph(State)
  .addNode("node_1", subgraph)
  .addEdge(START, "node_1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });
```

**子图独立记忆**:

如果希望子图拥有自己的记忆（例如在多 Agent 系统中，Agent 跟踪各自的内部消息历史），可以为子图单独编译时传入检查点器：

```javascript
const subgraphBuilder = new StateGraph(...);
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```

---

## 添加长期记忆

使用长期记忆跨对话存储用户特定或应用特定数据。

### 基本用法

```javascript
import { InMemoryStore, StateGraph } from "@langchain/langgraph";

const store = new InMemoryStore();

const builder = new StateGraph(...);
const graph = builder.compile({ store });
```

**用途**:
- 用户偏好设置
- 应用配置
- 跨会话的上下文信息
- 知识库

### 在生产环境使用

在生产环境中，使用数据库支持的存储：

```javascript
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";

const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
const store = PostgresStore.fromConnString(DB_URI);

const builder = new StateGraph(...);
const graph = builder.compile({ store });
```

### 使用语义搜索

在图的记忆存储中启用语义搜索，让 Agent 通过语义相似性搜索存储中的项目。

```javascript
import { OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryStore } from "@langchain/langgraph";

// 创建启用语义搜索的存储
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
const store = new InMemoryStore({
  index: {
    embeddings,
    dims: 1536,
  },
});

// 存储记忆
await store.put(["user_123", "memories"], "1", { text: "I love pizza" });
await store.put(["user_123", "memories"], "2", { text: "I am a plumber" });

// 语义搜索
const items = await store.search(["user_123", "memories"], {
  query: "I'm hungry",
  limit: 1,
});
```

**输出示例**:
```javascript
[
  {
    value: { text: "I love pizza" },
    key: "1",
    namespace: ["user_123", "memories"],
    score: 0.85  // 相似度分数
  }
]
```

**应用场景**:
- 个性化推荐
- 上下文感知响应
- 知识检索
- 用户行为分析

---

## 管理短期记忆

启用短期记忆后，长对话可能超过 LLM 的上下文窗口。常见解决方案：

1. **修剪消息**: 删除前 N 或后 N 条消息（调用 LLM 前）
2. **删除消息**: 从 LangGraph 状态永久删除消息
3. **总结消息**: 总结早期消息并用摘要替换
4. **管理检查点**: 存储和检索消息历史
5. **自定义策略**: 消息过滤等

---

## 修剪消息

大多数 LLM 都有最大支持的上下文窗口（以 tokens 为单位）。决定何时截断消息的一种方法是计算消息历史中的 tokens 数量，并在接近该限制时截断。

使用 `trimMessages` 函数修剪消息历史：

```javascript
import { trimMessages } from "@langchain/core/messages";
import { StateSchema, MessagesValue, GraphNode } from "@langchain/langgraph";

const State = new StateSchema({
  messages: MessagesValue,
});

const callModel: GraphNode<typeof State> = async (state) => {
  const messages = trimMessages(state.messages, {
    strategy: "last",
    maxTokens: 128,
    startOn: "human",
    endOn: ["human", "tool"],
  });
  const response = await model.invoke(messages);
  return { messages: [response] };
};

const builder = new StateGraph(State)
  .addNode("call_model", callModel);
  // ...
```

**参数说明**:
- `strategy`: 保留策略，`"last"` 表示保留最后的消息
- `maxTokens`: 保留的最大 token 数量
- `startOn`: 起始消息类型
- `endOn`: 结束消息类型数组

**修剪策略**:
- `"last"`: 保留最后的消息
- `"first"`: 保留最前面的消息
- 自定义逻辑

---

## 删除消息

可以从图状态中删除消息来管理消息历史。这在需要删除特定消息或清除整个消息历史时很有用。

要从图状态删除消息，使用 `RemoveMessage`。为使 `RemoveMessage` 工作，需要使用带有 `messagesStateReducer` reducer 的状态键，如 `MessagesValue`。

### 删除特定消息

```javascript
import { RemoveMessage } from "@langchain/core/messages";

const deleteMessages = (state) => {
  const messages = state.messages;
  if (messages.length > 2) {
    // 删除最早的两条消息
    return {
      messages: messages
        .slice(0, 2)
        .map((m) => new RemoveMessage({ id: m.id })),
    };
  }
};
```

> **警告**: 删除消息时，确保生成的消息历史有效。检查你使用的 LLM 提供商的限制。例如：
> - 某些提供商期望消息历史以 `user` 消息开始
> - 大多数提供商要求带有工具调用的 `assistant` 消息后跟相应的 `tool` 结果消息

**常见删除场景**:
- 删除最早的消息以节省 tokens
- 删除敏感信息
- 删除错误或无效消息
- 清空对话历史

---

## 总结消息

修剪或删除消息的问题在于可能会丢失信息。因此，某些应用受益于使用聊天模型总结消息历史的更复杂方法。

![总结流程示意图](概念示意：messages → summarize → summary + recent messages)

### 实现方式

在状态中包含 `summary` 键以及 `messages` 键：

```javascript
import { StateSchema, MessagesValue, GraphNode } from "@langchain/langgraph";
import { z } from "zod/v4";

const State = new StateSchema({
  messages: MessagesValue,
  summary: z.string().optional(),
});
```

### 生成总结

生成聊天历史的摘要，使用任何现有摘要作为下一个摘要的上下文。此 `summarizeConversation` 节点可以在 `messages` 状态键中累积了一定数量的消息后调用。

```javascript
import { RemoveMessage, HumanMessage } from "@langchain/core/messages";

const summarizeConversation: GraphNode<typeof State> = async (state) => {
  // 首先，获取任何现有摘要
  const summary = state.summary || "";

  // 创建总结提示
  let summaryMessage: string;
  if (summary) {
    // 已存在摘要
    summaryMessage =
      `This is a summary of the conversation to date: ${summary}\n\n` +
      "Extend the summary by taking into account the new messages above:";
  } else {
    summaryMessage = "Create a summary of the conversation above:";
  }

  // 将提示添加到历史中
  const messages = [
    ...state.messages,
    new HumanMessage({ content: summaryMessage })
  ];
  const response = await model.invoke(messages);

  // 删除除最近 2 条消息外的所有消息
  const deleteMessages = state.messages
    .slice(0, -2)
    .map(m => new RemoveMessage({ id: m.id }));

  return {
    summary: response.content,
    messages: deleteMessages
  };
};
```

**工作流程**:
1. 获取现有摘要（如果有）
2. 创建总结提示（基于是否已有摘要）
3. 调用 LLM 生成新摘要
4. 删除旧消息，仅保留最近的消息
5. 返回更新的摘要和消息列表

**优势**:
- 保留重要信息
- 节省 tokens
- 保持上下文连续性
- 适合长对话

---

## 管理检查点

可以查看和删除检查点器存储的信息。

### 查看线程状态

```javascript
const config = {
  configurable: {
    thread_id: "1",
    // 可选：提供特定检查点的 ID
    // 否则显示最新检查点
    // checkpoint_id: "1f029ca3-1f5b-6704-8004-820c16b69a5a"
  },
};
await graph.getState(config);
```

**输出示例**:
```javascript
{
  values: { 
    messages: [
      HumanMessage(...), 
      AIMessage(...), 
      HumanMessage(...), 
      AIMessage(...)
    ] 
  },
  next: [],
  config: { 
    configurable: { 
      thread_id: '1', 
      checkpoint_ns: '', 
      checkpoint_id: '1f029ca3-1f5b-6704-8004-820c16b69a5a' 
    } 
  },
  metadata: {
    source: 'loop',
    writes: { call_model: { messages: AIMessage(...) } },
    step: 4,
    parents: {},
    thread_id: '1'
  },
  createdAt: '2025-05-05T16:01:24.680462+00:00',
  parentConfig: { 
    configurable: { 
      thread_id: '1', 
      checkpoint_ns: '', 
      checkpoint_id: '1f029ca3-1790-6b0a-8003-baf965b6a38f' 
    } 
  },
  tasks: [],
  interrupts: []
}
```

### 查看线程历史

```javascript
const config = {
  configurable: {
    thread_id: "1",
  },
};

const history = [];
for await (const state of graph.getStateHistory(config)) {
  history.push(state);
}
```

**用途**:
- 查看对话演变
- 调试问题
- 审计对话历史
- 回溯到特定检查点

### 删除线程的所有检查点

```javascript
const threadId = "1";
await checkpointer.deleteThread(threadId);
```

**用途**:
- 清理测试数据
- 响应用户删除请求
- 管理存储空间
- 符合数据保留政策

---

## 数据库管理

如果使用任何数据库支持的持久化实现（如 Postgres 或 Redis）来存储短期和/或长期记忆，需要在使用前运行迁移以设置所需的模式。

### 运行迁移

按照惯例，大多数数据库特定的库在检查点器或存储实例上定义 `setup()` 方法来运行所需的迁移。

```javascript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const checkpointer = PostgresSaver.fromConnString(DB_URI);
await checkpointer.setup();
```

**建议**:
- 将迁移作为专用部署步骤运行
- 或确保它们作为服务器启动的一部分运行

**注意**: 应查看你的特定 `BaseCheckpointSaver` 或 `BaseStore` 实现，以确认确切的方法名称和用法。

---

## 记忆类型对比

| 类型 | 持久化级别 | 生命周期 | 典型用途 |
|------|----------|---------|---------|
| **短期记忆** | 线程级 | 单个对话会话 | 多轮对话、上下文维护 |
| **长期记忆** | 跨会话 | 多个会话 | 用户偏好、知识库 |

---

## 最佳实践

### 1. 选择合适的记忆类型

**短期记忆**:
- 用于对话上下文
- 需要跟踪当前会话状态
- 临时信息

**长期记忆**:
- 用户个性化数据
- 应用配置
- 跨会话知识

### 2. 生产环境使用数据库

开发环境：
```javascript
const checkpointer = new MemorySaver();  // 内存中
const store = new InMemoryStore();
```

生产环境：
```javascript
const checkpointer = PostgresSaver.fromConnString(DB_URI);
const store = PostgresStore.fromConnString(DB_URI);
```

### 3. 管理上下文窗口

```javascript
// 策略组合
const manageContext: GraphNode<typeof State> = async (state) => {
  let messages = state.messages;
  
  // 1. 如果消息过多，先总结
  if (messages.length > 10) {
    messages = await summarizeOldMessages(messages);
  }
  
  // 2. 修剪到合适的 token 数
  messages = trimMessages(messages, {
    maxTokens: 4000,
    strategy: "last",
  });
  
  // 3. 调用模型
  const response = await model.invoke(messages);
  
  return { messages: [response] };
};
```

### 4. 线程隔离

为不同用户或会话使用不同的 `thread_id`：

```javascript
// 用户 A 的对话
await graph.invoke(input, { 
  configurable: { thread_id: "user_a_session_1" } 
});

// 用户 B 的对话
await graph.invoke(input, { 
  configurable: { thread_id: "user_b_session_1" } 
});
```

### 5. 语义搜索优化

```javascript
// 使用合适的嵌入模型
const embeddings = new OpenAIEmbeddings({ 
  model: "text-embedding-3-small"  // 性能和成本平衡
});

// 限制搜索结果数量
const items = await store.search(namespace, {
  query: userQuery,
  limit: 5,  // 仅获取最相关的
});
```

### 6. 定期清理

```javascript
// 定期删除旧线程
const OLD_THREAD_THRESHOLD = 30 * 24 * 60 * 60 * 1000; // 30 天

async function cleanupOldThreads() {
  const allThreads = await checkpointer.list();
  
  for (const thread of allThreads) {
    const lastUpdated = new Date(thread.createdAt);
    if (Date.now() - lastUpdated.getTime() > OLD_THREAD_THRESHOLD) {
      await checkpointer.deleteThread(thread.thread_id);
    }
  }
}
```

---

## 实战场景

### 场景 1：客服聊天机器人

```javascript
import { StateGraph, StateSchema, MessagesValue } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { z } from "zod/v4";

const State = new StateSchema({
  messages: MessagesValue,
  userId: z.string(),
});

// 设置持久化
const checkpointer = PostgresSaver.fromConnString(DB_URI);
const store = PostgresStore.fromConnString(DB_URI);

// 构建图
const graph = new StateGraph(State)
  .addNode("handleQuery", async (state) => {
    // 从长期记忆检索用户信息
    const userPrefs = await store.get(["users", state.userId], "preferences");
    
    // 使用短期记忆（messages）和长期记忆（userPrefs）
    const response = await model.invoke([
      { role: "system", content: `User preferences: ${JSON.stringify(userPrefs)}` },
      ...state.messages,
    ]);
    
    return { messages: [response] };
  })
  .compile({ checkpointer, store });

// 使用
await graph.invoke(
  { messages: [{ role: "user", content: "What's my order status?" }], userId: "user123" },
  { configurable: { thread_id: "user123_session_456" } }
);
```

### 场景 2：文档问答系统

```javascript
import { OpenAIEmbeddings } from "@langchain/openai";
import { InMemoryStore } from "@langchain/langgraph";

const State = new StateSchema({
  messages: MessagesValue,
  query: z.string(),
});

// 启用语义搜索的知识库
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
const knowledgeBase = new InMemoryStore({
  index: { embeddings, dims: 1536 },
});

// 预先加载文档
await knowledgeBase.put(["docs"], "1", { text: "产品使用说明..." });
await knowledgeBase.put(["docs"], "2", { text: "故障排除指南..." });

const graph = new StateGraph(State)
  .addNode("search", async (state) => {
    // 语义搜索相关文档
    const relevantDocs = await knowledgeBase.search(["docs"], {
      query: state.query,
      limit: 3,
    });
    
    // 构建上下文
    const context = relevantDocs.map(doc => doc.value.text).join("\n\n");
    
    // 生成答案
    const response = await model.invoke([
      { role: "system", content: `Context: ${context}` },
      ...state.messages,
    ]);
    
    return { messages: [response] };
  })
  .compile({ store: knowledgeBase });
```

### 场景 3：长对话管理

```javascript
const State = new StateSchema({
  messages: MessagesValue,
  summary: z.string().optional(),
  messageCount: z.number().default(0),
});

const graph = new StateGraph(State)
  .addNode("chat", async (state) => {
    let messages = state.messages;
    
    // 如果消息超过阈值，触发总结
    if (state.messageCount > 10 && state.messageCount % 5 === 0) {
      const summary = await summarizeConversation(state);
      
      // 删除旧消息，保留最近的
      const deleteOld = state.messages
        .slice(0, -3)
        .map(m => new RemoveMessage({ id: m.id }));
      
      messages = [
        new HumanMessage({ content: `Previous summary: ${summary}` }),
        ...state.messages.slice(-3),
      ];
      
      return { 
        summary, 
        messages: [...deleteOld, response],
        messageCount: state.messageCount + 1 
      };
    }
    
    const response = await model.invoke(messages);
    return { 
      messages: [response],
      messageCount: state.messageCount + 1 
    };
  })
  .compile({ checkpointer: new MemorySaver() });
```

---

## 性能考虑

### 1. 检查点开销

每次调用都会保存检查点：
- 使用数据库索引优化查询
- 考虑异步写入
- 定期清理旧检查点

### 2. 语义搜索成本

- 嵌入生成有成本
- 缓存常见查询的嵌入
- 批量生成嵌入

### 3. 内存使用

```javascript
// 好的做法：及时修剪
const messages = trimMessages(state.messages, { maxTokens: 4000 });

// 不好的做法：无限累积
// messages 会持续增长，消耗内存和 tokens
```

---

## 常见问题

**Q: 短期记忆和长期记忆能同时使用吗？**

A: 可以！实际上很常见：

```javascript
const graph = builder.compile({ 
  checkpointer,  // 短期记忆
  store          // 长期记忆
});
```

**Q: 如何在多个图之间共享记忆？**

A: 使用相同的检查点器和存储实例：

```javascript
const sharedCheckpointer = PostgresSaver.fromConnString(DB_URI);
const graph1 = builder1.compile({ checkpointer: sharedCheckpointer });
const graph2 = builder2.compile({ checkpointer: sharedCheckpointer });
```

**Q: thread_id 应该如何设置？**

A: 常见模式：
- 用户会话：`user_${userId}_session_${sessionId}`
- 对话主题：`conversation_${conversationId}`
- 临时任务：`task_${taskId}`

**Q: 如何迁移现有对话到新版本？**

A: 
1. 使用 `getStateHistory` 导出旧数据
2. 转换为新格式
3. 使用新检查点器重新导入

**Q: 语义搜索支持哪些嵌入模型？**

A: 支持所有 LangChain 嵌入模型：
- OpenAI Embeddings
- Cohere Embeddings
- HuggingFace Embeddings
- 自定义嵌入模型

**Q: 如何处理 GDPR 等数据删除要求？**

A: 
```javascript
// 删除用户的所有数据
await checkpointer.deleteThread(userId);
await store.delete(["users", userId]);
```

---

## 相关资源

- [Persistence 持久化](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- [State Management 状态管理](https://docs.langchain.com/oss/javascript/langgraph/graph-api#state)
- [Multi-Agent 多 Agent 系统](https://docs.langchain.com/oss/javascript/langchain/multi-agent)
- [Streaming 流式处理](https://docs.langchain.com/oss/javascript/langgraph/streaming)
