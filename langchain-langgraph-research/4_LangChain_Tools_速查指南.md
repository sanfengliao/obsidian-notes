# LangChain Tools

## 概述

工具让 Agent 做更多事——访问实时数据、执行代码、查询数据库等。

本质上，工具就是有明确输入和输出的函数。模型根据对话决定何时调用、传什么参数。

关键强大之处：工具能访问 Agent 的状态、运行时数据和长期内存。这样工具可以做上下文感知的决策、提供个性化响应，甚至在不同对话之间维护信息。

## 创建工具

### 基础写法

最简单的是用 `tool()` 函数，配合 Zod 定义输入的 schema：

```javascript
import * as z from "zod";
import { tool } from "langchain";

const searchDatabase = tool(
  ({ query, limit }) => `Found ${limit} results for '${query}'`,
  {
    name: "search_database",
    description: "Search the customer database for records matching the query.",
    schema: z.object({
      query: z.string().describe("Search terms to look for"),
      limit: z.number().describe("Maximum number of results to return"),
    }),
  }
);
```

tool函数需要四个关键参数：
1. **执行函数** - 接收参数，返回结果
2. **name** - 工具的标识符，LLM 用这个决定调用哪个工具
3. **description** - 解释工具的用途，帮助 LLM 理解何时调用它
4. **schema** - Zod 对象，定义输入参数及其验证规则

### 异步工具

工具可以是异步的（async）：

```javascript
import * as z from "zod";
import { tool } from "langchain";

const getUserInfo = tool(
  async ({ user_id }) => {
    const response = await fetch(`/api/users/${user_id}`);
    return response.json();
  },
  {
    name: "get_user_info",
    description: "Fetch user information from the database.",
    schema: z.object({
      user_id: z.string(),
    }),
  }
);
```

## 工具能访问什么数据

工具函数的第二个参数是 `config` 对象，提供运行时信息。

### 从 Context 中读取数据

Agent 调用时可以传入 context，工具可以访问它：

```javascript
import * as z from "zod";
import { tool } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";

const getUserName = tool(
  (_, config) => {
    return config.context.user_name;  // 读取传入的 context
  },
  {
    name: "get_user_name",
    description: "Get the user's name.",
    schema: z.object({}),  // 没有输入参数
  }
);

const contextSchema = z.object({
  user_name: z.string(),
});

const agent = createAgent({
  model: new ChatOpenAI({ model: "gpt-4o" }),
  tools: [getUserName],
  contextSchema,
});

// 调用时传入 context 数据
const result = await agent.invoke(
  { messages: [{ role: "user", content: "What is my name?" }] },
  { context: { user_name: "John Smith" } }
);
```

用 context 的好处：
- 传入用户 ID、数据库连接、API 密钥等
- 避免全局状态，工具更容易测试和复用
- 根据用户或会话提供个性化的响应

### 跨对话存储数据

可以用 Store 来持久化存储，这样数据可以跨对话访问。通过 `config.store` 来操作：

```javascript
import * as z from "zod";
import { createAgent, tool } from "langchain";
import { InMemoryStore } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const store = new InMemoryStore();

// 读取数据
const getUserInfo = tool(
  async ({ user_id }) => {
    const value = await store.get(["users"], user_id);
    return value;
  },
  {
    name: "get_user_info",
    description: "Look up user info.",
    schema: z.object({
      user_id: z.string(),
    }),
  }
);

// 保存数据
const saveUserInfo = tool(
  async ({ user_id, name, age, email }) => {
    await store.put(["users"], user_id, { name, age, email });
    return "Successfully saved user info.";
  },
  {
    name: "save_user_info",
    description: "Save user info.",
    schema: z.object({
      user_id: z.string(),
      name: z.string(),
      age: z.number(),
      email: z.string(),
    }),
  }
);

const agent = createAgent({
  model: new ChatOpenAI({ model: "gpt-4o" }),
  tools: [getUserInfo, saveUserInfo],
  store,
});

// 第一次会话：保存数据
await agent.invoke({
  messages: [{
    role: "user",
    content: "Save: userid abc123, name Foo, age 25, email foo@example.com"
  }],
});

// 第二次会话：读取数据（跨对话保留）
const result = await agent.invoke({
  messages: [{ role: "user", content: "Get user info for abc123" }],
});
```

Store 提供的操作：
- `await store.get(namespace, key)` - 读取数据
- `await store.put(namespace, key, value)` - 保存数据
- `await store.delete(namespace, key)` - 删除数据

namespace 是一个数组，用来组织数据，比如 `["users"]`、`["conversations"]` 等。

### 实时推送工具的进度

有些操作可能很耗时。可以用流式写入来向用户实时推送进度：

```javascript
import * as z from "zod";
import { tool, ToolRuntime } from "langchain";

const getWeather = tool(
  ({ city }, config: ToolRuntime) => {
    const writer = config.writer;

    // 实时推送更新
    if (writer) {
      writer(`Looking up data for city: ${city}`);
      // ... 执行操作 ...
      writer(`Acquired data for city: ${city}`);
    }

    return `It's always sunny in ${city}!`;
  },
  {
    name: "get_weather",
    description: "Get weather for a given city.",
    schema: z.object({
      city: z.string(),
    }),
  }
);
```

用途：
- 向用户展示工具正在做什么
- 长时间操作时提供进度反馈
- 提升用户体验

## 服务端工具

某些模型（OpenAI、Anthropic、Gemini）支持服务端工具，比如 Web 搜索、代码执行、文件操作等。

这些工具在服务端执行，不需要你手动编写循环。查看相应 provider 的 integration 文档了解如何启用。

## 如何设计好的工具

### 1. 清晰的名字和描述

```javascript
// ❌ 不好
const tool1 = tool(
  (input) => "result",
  {
    name: "t1",
    description: "do stuff",
    schema: z.object({}),
  }
);

// ✅ 好
const searchCustomers = tool(
  (input) => "result",
  {
    name: "search_customers",
    description: "Search customer database by name, email, or phone number.",
    schema: z.object({
      query: z.string().describe("Name, email, or phone to search for"),
    }),
  }
);
```

### 2. 参数描述要详细
```javascript
// ✅ 参数描述清楚，LLM 才能正确调用
const book = tool(
  (input) => "booked",
  {
    name: "book_flight",
    description: "Book a flight for a passenger.",
    schema: z.object({
      departure_city: z.string()
        .describe("IATA code of departure city, e.g. 'SFO' for San Francisco"),
      destination_city: z.string()
        .describe("IATA code of destination city, e.g. 'LAX' for Los Angeles"),
      departure_date: z.string()
        .describe("Flight date in ISO 8601 format (YYYY-MM-DD)"),
      passenger_count: z.number()
        .describe("Number of passengers (1-9)"),
    }),
  }
);
```

### 3. 用 Zod 做输入验证

Zod schema 会自动验证输入。充分利用这点确保数据有效：

```javascript
const withdraw = tool(
  ({ amount }) => {
    // amount 已被验证为 number 且在 1-5000 之间
    return `Withdrew $${amount}`;
  },
  {
    name: "withdraw_money",
    description: "Withdraw money from account.",
    schema: z.object({
      amount: z.number()
        .min(1, "Amount must be at least $1")
        .max(5000, "Daily limit is $5000")
        .describe("Amount to withdraw in dollars"),
    }),
  }
);
```

### 4. 处理错误

工具中的错误会被捕获并传给模型，所以要让模型看到有用的错误消息：

```javascript
const fetchUser = tool(
  async ({ user_id }) => {
    try {
      const response = await fetch(`/api/users/${user_id}`);
      if (!response.ok) {
        throw new Error(`User not found (HTTP ${response.status})`);
      }
      return response.json();
    } catch (error) {
      // 返回有用的错误消息，让模型能理解并做出反应
      throw new Error(`Failed to fetch user: ${error.message}`);
    }
  },
  {
    name: "fetch_user",
    description: "Fetch user data by ID.",
    schema: z.object({
      user_id: z.string(),
    }),
  }
);
```

## 如何使用工具的完整流程

### 绑定工具到模型

```javascript
const modelWithTools = model.bindTools([
  searchDatabase,
  fetchUser,
  saveUserInfo,
]);
```

### 处理工具调用的循环

```javascript
const messages = [];

while (true) {
  messages.push(new HumanMessage("User input"));
  const response = await modelWithTools.invoke(messages);
  messages.push(response);

  if (!response.tool_calls || response.tool_calls.length === 0) {
    break;  // 模型不再请求调用工具，完成
  }

  // 执行模型要求的工具
  for (const toolCall of response.tool_calls) {
    console.log(`Calling: ${toolCall.name}`);
    console.log(`Args: ${JSON.stringify(toolCall.args)}`);
    
    // 在这儿执行实际的工具
    const result = await executeTool(toolCall);
    
    // 把结果返回给模型
    messages.push(new ToolMessage({
      content: result,
      tool_call_id: toolCall.id,
    }));
  }
}
```

**提示**：用 Agent 时，它会自动处理工具调用循环。单独用模型时，需要手动实现上面的循环。

## 常见问题

**Q：工具函数应该返回什么？**

A：通常返回字符串。也可以返回 JSON、对象等，最终会被序列化成字符串传给模型。

**Q：工具可以有副作用吗？**

A：可以。工具可以执行任何操作——查询数据库、发邮件、调用 API 等。重点是返回结果给模型。

**Q：怎样在工具之间共享状态？**

A：用 Store（跨对话保留）或 context（单次调用）。避免全局变量，保持工具独立和可测试。

**Q：工具出错了会怎样？**

A：错误会被捕获，错误消息发给模型。模型可以看到错误，尝试修正参数或调用不同的工具。

**Q：能并行调用多个工具吗？**

A：可以。模型可以在一个响应中请求多个工具调用。Agent 会并行执行它们。

**Q：怎样限制工具的访问权限？**

A：用 context 和 Store。只传给工具它需要的数据，不要全局暴露。工具函数内部也可以检查权限。
