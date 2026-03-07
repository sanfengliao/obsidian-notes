# LangChain Short-Term Memory

## 概述

短期内存让 Agent 记住同一个对话或线程内的交互历史。这在长对话中很重要，因为模型的上下文窗口有限制。

想象一下，用户和 Agent 聊了几十轮，消息列表会变得非常长。超过模型的上下文限制就会出问题。短期内存系统提供多种方式来智能地管理这个历史，既不丢失重要信息，又不超过上下文限制。

## 基础：启用短期内存

### 添加 Checkpointer

短期内存的核心是 checkpointer，它负责保存和恢复对话状态。最简单的是用 `MemorySaver`：

```javascript
import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools: [],
  checkpointer,
});

// 每次调用时传入 thread_id 来维持对话
await agent.invoke(
  { messages: [{ role: "user", content: "hi! i am Bob" }] },
  { configurable: { thread_id: "1" } }
);
```

同一个 `thread_id` 的调用会共享对话历史。不同 thread_id 是独立的。

### 生产环境：用数据库

开发时 MemorySaver 直接存在内存就够了。生产环境需要数据库持久化：

```javascript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable";
const checkpointer = PostgresSaver.fromConnString(DB_URI);

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  checkpointer,
});
```

## 扩展短期内存

### 自定义状态

除了消息，还可以在状态中存储其他信息（用户 ID、用户偏好等）：

```javascript
import * as z from "zod";
import { createAgent, createMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const customStateSchema = z.object({
  userId: z.string(),
  preferences: z.record(z.string(), z.any()),
});

const stateExtensionMiddleware = createMiddleware({
  name: "StateExtension",
  stateSchema: customStateSchema,
});

const checkpointer = new MemorySaver();
const agent = createAgent({
  model: "gpt-5",
  tools: [],
  middleware: [stateExtensionMiddleware],
  checkpointer,
});

// 调用时传入自定义状态
const result = await agent.invoke({
  messages: [{ role: "user", content: "Hello" }],
  userId: "user_123",
  preferences: { theme: "dark" },
});
```

## 管理长对话

对话太长会超过上下文限制。有三种常见策略：

### 1. 裁剪（Trim）- 保留最新的消息

只保留最近的几条消息或最近的 N 个 token：

```javascript
import { RemoveMessage } from "@langchain/core/messages";
import { createAgent, createMiddleware } from "langchain";
import { MemorySaver, REMOVE_ALL_MESSAGES } from "@langchain/langgraph";

const trimMessages = createMiddleware({
  name: "TrimMessages",
  beforeModel: (state) => {
    const messages = state.messages;
    if (messages.length <= 3) {
      return;  // 消息不多，无需处理
    }

    // 保留第一条消息（系统提示或上下文）和最后几条
    const firstMsg = messages[0];
    const recentMessages =
      messages.length % 2 === 0 ? messages.slice(-3) : messages.slice(-4);
    const newMessages = [firstMsg, ...recentMessages];

    return {
      messages: [
        new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
        ...newMessages,
      ],
    };
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [trimMessages],
  checkpointer: new MemorySaver(),
});
```

或者用内置的 `trimMessages` 工具按 token 计数：

```javascript
import { createAgent, createMiddleware, trimMessages } from "langchain";
import { RemoveMessage } from "@langchain/core/messages";
import { MemorySaver, REMOVE_ALL_MESSAGES } from "@langchain/langgraph";

const trimMessageHistory = createMiddleware({
  name: "TrimMessages",
  beforeModel: async (state) => {
    const trimmed = await trimMessages(state.messages, {
      maxTokens: 384,        // 最多保留 384 个 token
      strategy: "last",      // 保留最后的消息
      startOn: "human",
      endOn: ["human", "tool"],
      tokenCounter: (msgs) => msgs.length,  // 简单的计数方式
    });
    return {
      messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...trimmed],
    };
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [trimMessageHistory],
  checkpointer: new MemorySaver(),
});
```

### 2. 删除（Delete）- 永久移除旧消息

从状态中彻底删除某些消息：

```javascript
import { RemoveMessage } from "@langchain/core/messages";
import { createAgent, createMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const deleteOldMessages = createMiddleware({
  name: "DeleteOldMessages",
  afterModel: (state) => {
    const messages = state.messages;
    if (messages.length > 2) {
      // 删除前两条消息
      return {
        messages: messages
          .slice(0, 2)
          .map((m) => new RemoveMessage({ id: m.id! })),
      };
    }
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [deleteOldMessages],
  checkpointer: new MemorySaver(),
});
```

**注意**：删除消息时要确保剩下的历史对模型仍然有效。比如 tool 消息必须跟在 tool_call 后面。

### 3. 汇总（Summarize）- 用模型总结旧消息

不是简单地丢弃，而是用另一个模型把早期消息汇总成摘要，这样保留了信息：

```javascript
import { createAgent, summarizationMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    summarizationMiddleware({
      model: "gpt-4o-mini",           // 用来汇总的模型
      trigger: { tokens: 4000 },      // 当消息超过 4000 token 时触发
      keep: { messages: 20 },         // 保留最近的 20 条消息
    }),
  ],
  checkpointer,
});

const config = { configurable: { thread_id: "1" } };
await agent.invoke({ messages: "hi, my name is bob" }, config);
await agent.invoke({ messages: "write a short poem about cats" }, config);
await agent.invoke({ messages: "now do the same but for dogs" }, config);
const finalResponse = await agent.invoke({ messages: "what's my name?" }, config);

console.log(finalResponse.messages.at(-1)?.content);
// Output: "Your name is Bob!"
```

## 在工具中读取短期内存

工具可以通过 `config.state` 访问当前对话的状态：

```javascript
import * as z from "zod";
import { createAgent, tool, type ToolRuntime } from "langchain";

const stateSchema = z.object({
  userId: z.string(),
});

const getUserInfo = tool(
  async (_, config: ToolRuntime<z.infer<typeof stateSchema>>) => {
    const userId = config.state.userId;
    return userId === "user_123" ? "John Doe" : "Unknown User";
  },
  {
    name: "get_user_info",
    description: "Get user info",
    schema: z.object({}),
  }
);

const agent = createAgent({
  model: "gpt-5-nano",
  tools: [getUserInfo],
  stateSchema,
});

const result = await agent.invoke(
  {
    messages: [{ role: "user", content: "what's my name?" }],
    userId: "user_123",
  },
  {
    context: {},
  }
);

console.log(result.messages.at(-1)?.content);
// Output: "Your name is John Doe."
```

## 在工具中修改短期内存

工具可以直接更新状态，供后续的工具或对话使用：

```javascript
import * as z from "zod";
import { tool, createAgent, ToolMessage, type ToolRuntime } from "langchain";
import { Command } from "@langchain/langgraph";

const CustomState = z.object({
  userId: z.string().optional(),
  userName: z.string().optional(),
});

const updateUserInfo = tool(
  async (_, config: ToolRuntime<typeof CustomState>) => {
    const userId = config.state.userId;
    const name = userId === "user_123" ? "John Smith" : "Unknown user";
    
    return new Command({
      update: {
        userName: name,
        messages: [
          new ToolMessage({
            content: "Successfully looked up user information",
            tool_call_id: config.toolCall?.id ?? "",
          }),
        ],
      },
    });
  },
  {
    name: "update_user_info",
    description: "Look up and update user info.",
    schema: z.object({}),
  }
);

const greet = tool(
  async (_, config) => {
    const userName = config.context?.userName;
    return `Hello ${userName}!`;
  },
  {
    name: "greet",
    description: "Greet the user using their info.",
    schema: z.object({}),
  }
);

const agent = createAgent({
  model: "openai:gpt-5-mini",
  tools: [updateUserInfo, greet],
  stateSchema: CustomState,
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "greet the user" }],
  userId: "user_123",
});

console.log(result.messages.at(-1)?.content);
// Output: "Hello! I'm here to help — what would you like to do today?"
```

## 在中间件中访问状态

### beforeModel Hook - 在调用模型前处理

在发送消息给模型之前，可以修改状态（比如裁剪消息）：

```javascript
import { createAgent, createMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const processBeforeModel = createMiddleware({
  name: "ProcessBeforeModel",
  beforeModel: (state) => {
    // 在这里可以修改消息、检查状态等
    const messages = state.messages;
    console.log(`About to call model with ${messages.length} messages`);
    return;  // 无需修改时返回 undefined 或空对象
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [processBeforeModel],
  checkpointer: new MemorySaver(),
});
```

### afterModel Hook - 在模型返回后处理

在模型返回响应后处理（比如验证响应、更新状态）：

```javascript
import { RemoveMessage } from "@langchain/core/messages";
import { createAgent, createMiddleware } from "langchain";
import { REMOVE_ALL_MESSAGES } from "@langchain/langgraph";

const validateResponse = createMiddleware({
  name: "ValidateResponse",
  afterModel: (state) => {
    const lastMessage = state.messages.at(-1)?.content;
    if (
      typeof lastMessage === "string" &&
      lastMessage.toLowerCase().includes("confidential")
    ) {
      // 如果包含敏感内容，清除所有消息
      return {
        messages: [
          new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
          ...state.messages,
        ],
      };
    }
    return;
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [validateResponse],
});
```

## 动态系统提示

根据当前状态动态生成系统提示：

```javascript
import * as z from "zod";
import { createAgent, tool, dynamicSystemPromptMiddleware } from "langchain";

const contextSchema = z.object({
  userName: z.string(),
});
type ContextSchema = z.infer<typeof contextSchema>;

const getWeather = tool(
  async ({ city }) => {
    return `The weather in ${city} is always sunny!`;
  },
  {
    name: "get_weather",
    description: "Get weather for a location",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const agent = createAgent({
  model: "gpt-5-nano",
  tools: [getWeather],
  contextSchema,
  middleware: [
    dynamicSystemPromptMiddleware<ContextSchema>((_, config) => {
      // 根据当前用户动态生成提示词
      return `You are a helpful assistant. Address the user as ${config.context?.userName}.`;
    }),
  ],
});

const result = await agent.invoke(
  {
    messages: [{ role: "user", content: "What is the weather in SF?" }],
  },
  {
    context: {
      userName: "John Smith",
    },
  }
);
```

## 常见问题

**Q：Thread 是什么？**

A：Thread 是一个独立的对话线程。同一个 `thread_id` 的调用会共享内存。不同 thread 彼此隔离，适合支持多个独立对话。

**Q：什么时候用裁剪、什么时候用汇总？**

A：简单的应用用裁剪就够了，快速高效。需要保留历史细节的应用用汇总，成本稍高但信息丢失少。

**Q：能同时用多个策略吗？**

A：可以。比如先裁剪最新消息，旧消息用汇总。多个 middleware 可以叠加。

**Q：怎样持久化自定义状态？**

A：Checkpointer 会自动保存所有状态字段。无论是消息、用户 ID 还是自定义数据，都会被保存到数据库。

**Q：能在 Tool 中创建新的 Thread 吗？**

A：可以。工具可以返回新的 thread_id，让 Agent 在后续的调用中使用。
