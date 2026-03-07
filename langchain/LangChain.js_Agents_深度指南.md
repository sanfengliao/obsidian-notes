# LangChain.js Agents 深度指南

## 核心概念

**Agents（智能体）** 是 LangChain 中的核心抽象，将 LLM 与工具结合，创建能够推理、选择工具并迭代解决问题的系统。

### 工作原理

Agent 通过 **ReAct 模式**（Reasoning + Acting）循环运行：

```
推理 → 行动(工具调用) → 观察 → 推理 → ... → 最终答案
```

Agent 运行直到满足停止条件：
- 模型给出最终答案（不调用工具）
- 达到最大迭代次数
- 遇到错误

### `createAgent` 与 LangGraph 的关系 ⭐ 重要

**`createAgent` 是 LangGraph 的高度抽象**，本质上是一个预构建的 LangGraph 工作流：

| 层级 | 技术 | 特点 | 使用场景 |
|------|------|------|---------|
| **高度抽象** | `createAgent` | 一行代码创建完整智能体 | 快速原型、标准 Agent 场景 |
| **中度抽象** | `StateGraph + Runnable` | 自定义图结构、灵活控制 | 复杂工作流、多 Agent 协作 |
| **底层实现** | LangGraph Runtime | 低级 API、完全控制 | 极端自定义场景 |

#### `createAgent` 内部做了什么？

当你调用 `createAgent()` 时，LangChain 在后台自动创建了一个 LangGraph：

```typescript
// 你写的代码
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, weather],
});

// LangChain 内部等价于（简化版）
const agentGraph = new StateGraph({
  channels: {
    messages: MessagesValue,
  },
});

// 添加节点：模型调用节点
agentGraph.addNode("model", async (state) => {
  // 调用 LLM
  const response = await model.invoke(state.messages);
  return { messages: [response] };
});

// 添加节点：工具执行节点
agentGraph.addNode("tools", async (state) => {
  // 执行工具调用
  const lastMessage = state.messages[-1];
  for (const toolCall of lastMessage.tool_calls) {
    const result = await executeTool(toolCall);
    // 返回工具结果...
  }
  return { messages: toolResults };
});

// 添加边：连接节点
agentGraph.addConditionalEdges("model", shouldContinue, {
  continue: "tools",
  end: END,
});

agentGraph.addEdge("tools", "model");

// 编译成可运行的图
const agent = agentGraph.compile();
```

#### 何时使用 `createAgent` vs 手写 LangGraph？

**使用 `createAgent`：** ✅
- 标准的 ReAct Agent 模式
- 需要快速开发
- 工具调用和推理足够
- 90% 的用例都适用

**使用 LangGraph StateGraph：** 🎯
- 需要多个独立的决策点（不是简单的工具调用）
- 需要多 Agent 协作
- 需要复杂的状态转移逻辑
- 需要条件分支和并行处理
- 需要完全的自定义控制

#### 具体示例对比

**简单场景：使用 `createAgent`**

```typescript
// ✅ 推荐：一行代码解决
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, weather, calculator],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in SF?" }],
});
```

**复杂场景：使用 LangGraph StateGraph**

```typescript
// 场景：需要研究员和写手两个 Agent 协作
import { StateGraph, START, END } from "@langchain/langgraph";

const workflow = new StateGraph({
  channels: {
    topic: State,
    research: State,
    draft: State,
  },
});

// 添加研究员节点
workflow.addNode("researcher", async (state) => {
  const researcher = createAgent({
    model: "openai:gpt-4o",
    tools: [search, getWeather],
  });
  const research = await researcher.invoke(...);
  return { research: research.messages };
});

// 添加写手节点
workflow.addNode("writer", async (state) => {
  const writer = createAgent({
    model: "openai:gpt-4o",
    tools: [formatDocument, checkGrammar],
  });
  const draft = await writer.invoke({
    messages: [{ role: "user", content: state.research }],
  });
  return { draft: draft.messages };
});

// 工作流
workflow.addEdge(START, "researcher");
workflow.addEdge("researcher", "writer");
workflow.addEdge("writer", END);

const graph = workflow.compile();
```

#### 关键差异总结

| 特性 | `createAgent` | LangGraph StateGraph |
|------|---------------|---------------------|
| **学习曲线** | 平缓（易上手） | 陡峭（需要理解图概念） |
| **代码量** | 最小（5-10 行） | 中等（50+ 行） |
| **功能范围** | Agent + 工具调用 | 任意工作流和状态转移 |
| **性能** | 自动优化 | 需要手动优化 |
| **调试** | 简单清晰 | 可视化图更清楚 |
| **扩展性** | 中等（中间件） | 高（完全自定义） |

**总结**：`createAgent` 是 LangGraph 为 Agent 模式优化过的快捷方式，让 99% 的用户无需触及 LangGraph 底层细节。

---

## 1. 创建 Agent

### 基础创建方式

```typescript
import { createAgent } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "langchain/tools";
import * as z from "zod";

// 第一步：定义工具
const search = tool(
  ({ query }) => `Results for: ${query}`,
  {
    name: "search",
    description: "Search for information",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
);

const weather = tool(
  ({ location }) => `Weather in ${location}: Sunny, 72°F`,
  {
    name: "get_weather",
    description: "Get weather for a location",
    schema: z.object({
      location: z.string().describe("City name"),
    }),
  }
);

// 第二步：创建 Agent
const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,
});

const agent = createAgent({
  model,
  tools: [search, weather],
  systemPrompt: "You are a helpful assistant. Be concise.",
});

// 第三步：调用 Agent
const result = await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in SF?" }],
});

console.log(result.messages[result.messages.length - 1].content);
```

### 使用字符串标识符

简化的创建方式，直接使用 `provider:model` 格式：

```typescript
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, weather],
});
```

**优点**：快速、简洁
**缺点**：无法配置模型参数（如 temperature、timeout）

---

## 2. 模型配置

### 2.1 静态模型

在创建时固定选择，整个生命周期保持不变。

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";

// 方式一：OpenAI
const openaiAgent = createAgent({
  model: new ChatOpenAI({
    model: "gpt-4o",
    temperature: 0.1,
    maxTokens: 2000,
    timeout: 60000,
  }),
  tools,
});

// 方式二：Anthropic
const anthropicAgent = createAgent({
  model: new ChatAnthropic({
    model: "claude-3-5-sonnet-20241022",
    temperature: 0.2,
  }),
  tools,
});
```

### 2.2 动态模型选择 ⭐ 重点

根据对话复杂度、成本考量或其他运行时条件动态选择模型。

**场景 1：基于消息数量的成本优化**

```typescript
import { createAgent, createMiddleware } from "langchain";
import { ChatOpenAI } from "@langchain/openai";

const basicModel = new ChatOpenAI({ model: "gpt-4o-mini" }); // 便宜
const advancedModel = new ChatOpenAI({ model: "gpt-4o" }); // 强大

const dynamicModelMiddleware = createMiddleware({
  name: "DynamicModelSelection",
  wrapModelCall: (request, handler) => {
    const messageCount = request.messages.length;

    // 如果对话较长，使用更强大的模型
    const selectedModel = messageCount > 10 ? advancedModel : basicModel;

    return handler({
      ...request,
      model: selectedModel,
    });
  },
});

const agent = createAgent({
  model: basicModel, // 默认使用低成本模型
  tools,
  middleware: [dynamicModelMiddleware],
});
```

**场景 2：基于任务难度的动态选择**

```typescript
import { HumanMessage } from "@langchain/core/messages";

const taskComplexityMiddleware = createMiddleware({
  name: "TaskComplexityAware",
  wrapModelCall: (request, handler) => {
    // 检查是否包含复杂关键词
    const lastMessage = request.messages[request.messages.length - 1];
    const content = 
      lastMessage instanceof HumanMessage 
        ? lastMessage.content 
        : "";

    const complexKeywords = ["analyze", "compare", "optimize", "architecture"];
    const isComplex = complexKeywords.some(kw => 
      String(content).toLowerCase().includes(kw)
    );

    const model = isComplex ? advancedModel : basicModel;

    return handler({
      ...request,
      model,
    });
  },
});
```

---

## 3. 工具系统

### 3.1 工具定义

工具赋予 Agent 执行真实操作的能力。LangChain.js 支持 3 种定义方式。

**方式 1：使用 `tool()` 辅助函数（推荐）**

```typescript
import { tool } from "langchain/tools";
import * as z from "zod";

const calculateTotal = tool(
  ({ items, quantities }) => {
    const total = items.reduce((sum, item, i) => 
      sum + item.price * quantities[i], 0
    );
    return `Total: $${total.toFixed(2)}`;
  },
  {
    name: "calculate_total",
    description: "Calculate the total price of items",
    schema: z.object({
      items: z.array(z.object({
        name: z.string(),
        price: z.number(),
      })),
      quantities: z.array(z.number()),
    }),
  }
);

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [calculateTotal],
});
```

**方式 2：使用 `StructuredTool` 类**

```typescript
import { StructuredTool } from "@langchain/core/tools";

class DatabaseQuery extends StructuredTool {
  name = "query_database";
  description = "Query the user database";
  
  schema = z.object({
    query: z.string().describe("SQL query to execute"),
  });

  async _call(input: z.infer<typeof this.schema>) {
    // 实现实际的数据库查询
    return `Query result: ${input.query}`;
  }
}

const dbTool = new DatabaseQuery();
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [dbTool],
});
```

**方式 3：使用 `DynamicTool`（动态创建）**

```typescript
import { DynamicTool } from "@langchain/core/tools";

const dynamicTool = new DynamicTool({
  name: "get_current_time",
  description: "Get the current time",
  func: async () => new Date().toISOString(),
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [dynamicTool],
});
```

### 3.2 工具调用流程

Agent 遵循 ReAct 循环：

```typescript
// 完整示例：多步骤推理
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, getWeather, calculateTotal],
  systemPrompt: "You are a helpful shopping assistant.",
});

const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "What's the weather in NYC and help me calculate total for 3 laptops at $1000 each"
  }],
});

// Agent 的执行过程（自动）：
// 1. 推理："需要先检查纽约天气，然后计算费用"
// 2. 行动：调用 getWeather({ location: "NYC" })
// 3. 观察：返回天气信息
// 4. 行动：调用 calculateTotal({ items: [...], quantities: [3] })
// 5. 观察：返回总价 $3000
// 6. 推理："有足够的信息回答问题"
// 7. 最终答案：综合的回复
```

### 3.3 工具错误处理 ⭐ 重点

通过中间件在工具调用时处理错误：

```typescript
import { createAgent, createMiddleware, ToolMessage } from "langchain";

const toolErrorHandlingMiddleware = createMiddleware({
  name: "ToolErrorHandling",
  wrapToolCall: async (request, handler) => {
    try {
      return await handler(request);
    } catch (error) {
      console.error(`Tool ${request.toolCall.name} failed:`, error);

      // 返回错误消息给 Agent，让它重试或选择其他工具
      return new ToolMessage({
        content: `Tool error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again or use a different approach.`,
        tool_call_id: request.toolCall.id!,
      });
    }
  },
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, weather],
  middleware: [toolErrorHandlingMiddleware],
});
```

### 3.4 并行工具调用

LangChain.js 1.0 支持智能体同时调用多个工具：

```typescript
// Agent 会自动识别可并行的工具调用
// 示例：获取多个城市的天气
const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "Get weather for NYC, LA, and Chicago"
  }],
});

// Agent 的并行执行：
// 1. 识别：3 个独立的工具调用
// 2. 并行调用：getWeather(NYC), getWeather(LA), getWeather(Chicago) - 同时发生
// 3. 汇聚结果：等待所有工具返回
// 4. 生成答案
```

---

## 4. 系统提示词

### 4.1 静态系统提示词

在创建时固定，整个生命周期保持不变。

```typescript
const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  systemPrompt: `You are a helpful customer service assistant.
- Be professional and courteous
- Provide accurate information
- Escalate complex issues to human support
- Keep responses concise`,
});
```

### 4.2 使用 SystemMessage 实现高级功能

包括 Anthropic 的提示词缓存（性能优化）：

```typescript
import { SystemMessage } from "@langchain/core/messages";
import { createAgent } from "langchain";

// 假设有一个很长的系统文档
const largeDocumentation = "...很长的文档内容...";

const literaryAgent = createAgent({
  model: "anthropic:claude-3-5-sonnet-20241022",
  tools,
  systemPrompt: new SystemMessage({
    content: [
      {
        type: "text",
        text: "You are an AI assistant that analyzes literary works.",
      },
      {
        type: "text",
        text: largeDocumentation,
        cache_control: { type: "ephemeral" }, // 启用 Anthropic 提示词缓存
      },
    ],
  }),
});

// 优势：
// - 第一次调用：完整的 documentation 被缓存
// - 后续调用：只需发送新的用户消息，缓存的文档可直接复用
// - 节省成本和延迟（特别是大文档场景）
```

### 4.3 动态系统提示词 ⭐ 重点

根据上下文和用户角色动态改变系统提示词。

```typescript
import * as z from "zod";
import { createAgent, dynamicSystemPromptMiddleware } from "langchain";

// 定义上下文 schema
const contextSchema = z.object({
  userRole: z.enum(["expert", "beginner", "manager"]),
  language: z.enum(["en", "zh", "es"]),
  domain: z.enum(["technical", "business", "general"]),
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  contextSchema,
  middleware: [
    dynamicSystemPromptMiddleware<z.infer<typeof contextSchema>>(
      (state, runtime) => {
        const context = runtime.context;
        const userRole = context?.userRole || "user";
        const domain = context?.domain || "general";
        const language = context?.language || "en";

        // 基础提示词
        let prompt = "You are a helpful assistant.";

        // 根据用户角色调整
        if (userRole === "expert") {
          prompt += " Provide detailed technical responses with best practices.";
        } else if (userRole === "beginner") {
          prompt += " Explain concepts simply, avoid jargon, provide examples.";
        } else if (userRole === "manager") {
          prompt += " Focus on business impact and ROI.";
        }

        // 根据域名调整
        if (domain === "technical") {
          prompt += " Include code examples when relevant.";
        } else if (domain === "business") {
          prompt += " Focus on strategic implications.";
        }

        // 多语言支持
        if (language === "zh") {
          prompt += " Respond in Simplified Chinese.";
        } else if (language === "es") {
          prompt += " Respond in Spanish.";
        }

        return prompt;
      }
    ),
  ],
});

// 调用时传入上下文
const expertResult = await agent.invoke(
  {
    messages: [{
      role: "user",
      content: "Explain machine learning architectures",
    }],
  },
  { context: { userRole: "expert", domain: "technical", language: "en" } }
);

const beginnerResult = await agent.invoke(
  {
    messages: [{
      role: "user",
      content: "Explain machine learning",
    }],
  },
  { context: { userRole: "beginner", domain: "general", language: "zh" } }
);
```

---

## 5. 高级功能

### 5.1 结构化输出

强制 Agent 返回特定格式的结构化数据，确保类型安全：

```typescript
import * as z from "zod";
import { createAgent } from "langchain";

// 定义输出 schema
const ContactInfo = z.object({
  name: z.string().describe("Full name"),
  email: z.string().email().describe("Email address"),
  phone: z.string().describe("Phone number"),
  company: z.string().optional().describe("Company name"),
});

const agent = createAgent({
  model: "openai:gpt-4o",
  responseFormat: ContactInfo,
});

const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "Extract contact info: John Doe, john@example.com, +1-555-123-4567, Acme Corp",
  }],
});

// 结果类型安全
console.log(result.structuredResponse);
// {
//   name: 'John Doe',
//   email: 'john@example.com',
//   phone: '+1-555-123-4567',
//   company: 'Acme Corp'
// }

// 在 TypeScript 中类型完全推导
type ContactInfo = z.infer<typeof ContactInfo>;
```

**实战应用**：从自由文本中提取结构化数据

```typescript
const OrderExtractor = z.object({
  items: z.array(z.object({
    productName: z.string(),
    quantity: z.number(),
    unitPrice: z.number(),
  })),
  totalAmount: z.number(),
  shippingAddress: z.string(),
  deliveryDate: z.string().datetime().optional(),
});

const agent = createAgent({
  model: "openai:gpt-4o",
  responseFormat: OrderExtractor,
});

const orderData = await agent.invoke({
  messages: [{
    role: "user",
    content: "Process this order: 3 laptops at $1200 each, 2 keyboards at $80 each, ship to 123 Main St, delivery by 2025-01-25",
  }],
});

// 自动解析为结构化对象
const orders = orderData.structuredResponse;
const total = orders.items.reduce((sum, item) => 
  sum + item.quantity * item.unitPrice, 0
);
```

### 5.2 对话记忆

Agent 自动维护消息历史。可通过自定义 State Schema 扩展记忆功能。

**基础记忆（自动）**

```typescript
// Agent 自动保存所有消息
const result1 = await agent.invoke({
  messages: [{ role: "user", content: "My name is Alice" }],
});

// 第二次对话时保持上下文
const result2 = await agent.invoke({
  messages: [
    ...result1.messages, // 保留之前的消息
    { role: "user", content: "What's my name?" }, // Agent 会回答 "Alice"
  ],
});
```

**扩展记忆（自定义 State）**

```typescript
import { z } from "zod";
import { StateSchema, MessagesValue } from "@langchain/langgraph";

const CustomAgentState = new StateSchema({
  messages: MessagesValue,
  userPreferences: z.record(z.string(), z.string()), // 用户偏好
  conversationSummary: z.string().optional(), // 对话摘要
  userId: z.string(), // 用户 ID
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  stateSchema: CustomAgentState,
});

// 初始化自定义状态
const result = await agent.invoke({
  messages: [{ role: "user", content: "I prefer technical explanations" }],
  userPreferences: { style: "technical", language: "en" },
  userId: "user-123",
  conversationSummary: "",
});

// 后续对话可以访问这些信息
const followUp = await agent.invoke({
  ...result, // 保留所有状态
  messages: [
    ...result.messages,
    { role: "user", content: "Explain databases" },
  ],
});
```

### 5.3 流式处理 ⭐ 重点

实时获取 Agent 的中间步骤，提升用户体验：

```typescript
const stream = await agent.stream(
  {
    messages: [{
      role: "user",
      content: "Search for AI news and summarize the findings",
    }],
  },
  { streamMode: "values" }
);

// 实时处理每一步
for await (const chunk of stream) {
  const latestMessage = chunk.messages.at(-1);

  if (latestMessage?.content) {
    // 输出文本回复
    console.log(`Assistant: ${latestMessage.content}`);
  } else if (latestMessage?.tool_calls) {
    // 显示正在调用的工具
    const toolNames = latestMessage.tool_calls.map((tc) => tc.name);
    console.log(`Calling tools: ${toolNames.join(", ")}`);
  }
}
```

**完整的流式交互示例**

```typescript
const stream = await agent.stream(
  {
    messages: [{
      role: "user",
      content: "Get weather for NYC and LA, then compare them",
    }],
  },
  { streamMode: "values" }
);

console.log("Starting Agent Stream...\n");

for await (const chunk of stream) {
  const lastMsg = chunk.messages.at(-1);

  if (lastMsg?.type === "ai") {
    if (lastMsg.content) {
      // 流式输出 AI 文本
      process.stdout.write(`AI: ${lastMsg.content}\n`);
    }
    if (lastMsg.tool_calls && lastMsg.tool_calls.length > 0) {
      // 显示工具调用
      console.log("🔧 Tools invoked:");
      for (const call of lastMsg.tool_calls) {
        console.log(`  - ${call.name}(${JSON.stringify(call.args)})`);
      }
    }
  } else if (lastMsg?.type === "tool") {
    // 显示工具返回结果
    console.log(`✅ Tool result: ${lastMsg.content}\n`);
  }
}
```

### 5.4 中间件系统 ⭐ 核心创新

中间件是 LangChain 1.0 的最大创新，提供强大的可扩展性钩子。

**中间件的 6 大钩子**

```typescript
import { createAgent, createMiddleware } from "langchain";

// 1. beforeAgent - Agent 执行前
// 2. afterAgent - Agent 执行后
// 3. beforeModel - 模型调用前（处理消息、添加上下文等）
// 4. afterModel - 模型调用后（验证、过滤响应等）
// 5. wrapToolCall - 工具调用时（错误处理、追踪等）
// 6. wrapAgentStep - 每个 Agent 步骤的包装
```

**示例 1：成本追踪中间件**

```typescript
import { createAgent, createMiddleware } from "langchain";

const costTrackingMiddleware = createMiddleware({
  name: "CostTracking",
  afterModel: (response, handler) => {
    // 模型返回后，计算成本
    const promptTokens = response.usage?.prompt_tokens || 0;
    const completionTokens = response.usage?.completion_tokens || 0;

    // OpenAI 价格（以最新模型为例）
    const promptCost = (promptTokens / 1000) * 0.003; // GPT-4o
    const completionCost = (completionTokens / 1000) * 0.006;
    const totalCost = promptCost + completionCost;

    console.log(`Cost for this call: $${totalCost.toFixed(4)}`);

    return handler(response);
  },
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  middleware: [costTrackingMiddleware],
});
```

**示例 2：人机协同中间件（HITL）**

```typescript
import { createAgent, createMiddleware } from "langchain";

const hitlMiddleware = createMiddleware({
  name: "HumanInTheLoop",
  afterModel: async (response, handler) => {
    // 检查是否是危险操作
    const hasDeleteOperation = response.choices[0]?.message?.tool_calls?.some(
      (tc) => tc.name.includes("delete")
    );

    if (hasDeleteOperation) {
      // 需要人工审批
      console.log("⚠️  Dangerous operation detected. Awaiting human approval...");
      
      // 这里可以调用真实的审批系统
      // const approved = await askHumanApproval(response);
      
      // 模拟：
      const approved = true;

      if (!approved) {
        throw new Error("Operation rejected by human");
      }
    }

    return handler(response);
  },
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [deleteUserData, modifySettings],
  middleware: [hitlMiddleware],
});
```

**示例 3：消息修剪中间件（优化成本）**

```typescript
import { createAgent, createMiddleware } from "langchain";

const messagePruningMiddleware = createMiddleware({
  name: "MessagePruning",
  beforeModel: (request, handler) => {
    // 只保留最近的 10 条消息，节省成本
    const maxMessages = 10;
    const messages = request.messages;

    if (messages.length > maxMessages) {
      const prunedMessages = messages.slice(-maxMessages);
      return handler({
        ...request,
        messages: prunedMessages,
      });
    }

    return handler(request);
  },
});

const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  middleware: [messagePruningMiddleware],
});
```

**示例 4：内容审核中间件**

```typescript
const contentModerationMiddleware = createMiddleware({
  name: "ContentModeration",
  beforeModel: (request, handler) => {
    // 在发送给模型前检查用户输入
    const lastMessage = request.messages[request.messages.length - 1];
    const content = String(lastMessage.content);

    // 检查禁用内容
    const bannedKeywords = ["violence", "hate"];
    const hasIllegalContent = bannedKeywords.some(kw =>
      content.toLowerCase().includes(kw)
    );

    if (hasIllegalContent) {
      throw new Error("Content violates policy");
    }

    return handler(request);
  },
});
```

**组合多个中间件**

```typescript
const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  middleware: [
    contentModerationMiddleware,  // 第一步：内容审核
    messagePruningMiddleware,      // 第二步：消息修剪
    dynamicModelMiddleware,        // 第三步：动态模型选择
    costTrackingMiddleware,        // 第四步：成本追踪
  ],
});

// 中间件按顺序执行：1 → 2 → 3 → 4
```

---

## 6. 完整实战示例

### 6.1 智能采购助手

```typescript
import { createAgent } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "langchain/tools";
import * as z from "zod";
import { createMiddleware } from "langchain";

// 定义工具
const searchProducts = tool(
  async ({ query, category }) => {
    // 模拟产品搜索
    return `Found products: Laptop Pro ($1200), Dell XPS ($1500) in ${category}`;
  },
  {
    name: "search_products",
    description: "Search for products",
    schema: z.object({
      query: z.string(),
      category: z.string(),
    }),
  }
);

const checkInventory = tool(
  async ({ productId }) => {
    return `Product ${productId}: In stock, 50 units available`;
  },
  {
    name: "check_inventory",
    description: "Check product inventory",
    schema: z.object({
      productId: z.string(),
    }),
  }
);

const getPrice = tool(
  async ({ productId }) => {
    return `Product ${productId}: $1200 (regular), $1100 (bulk discount available)`;
  },
  {
    name: "get_price",
    description: "Get product pricing",
    schema: z.object({
      productId: z.string(),
    }),
  }
);

// 成本优化中间件
const costOptimization = createMiddleware({
  name: "CostOptimization",
  wrapToolCall: async (request, handler) => {
    const startTime = Date.now();
    const result = await handler(request);
    const duration = Date.now() - startTime;

    console.log(`Tool ${request.toolCall.name} took ${duration}ms`);
    return result;
  },
});

// 创建 Agent
const agent = createAgent({
  model: new ChatOpenAI({ model: "gpt-4o", temperature: 0 }),
  tools: [searchProducts, checkInventory, getPrice],
  systemPrompt: `You are a procurement specialist.
- Help find the best products
- Consider price, availability, and bulk discounts
- Provide clear recommendations`,
  middleware: [costOptimization],
});

// 使用 Agent
const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "I need to buy 50 laptops. Find the best options with bulk pricing.",
  }],
});

console.log(result.messages[result.messages.length - 1].content);
```

### 6.2 带流式处理的研究助手

```typescript
const researchAgent = createAgent({
  model: "openai:gpt-4o",
  tools: [search, getWeather, analyzeData],
  systemPrompt: "You are a research analyst. Provide thorough analysis.",
});

// 实时流式处理
const stream = await researchAgent.stream(
  {
    messages: [{
      role: "user",
      content: "Research and compare AI frameworks for our use case",
    }],
  },
  { streamMode: "values" }
);

let stepCount = 0;

for await (const chunk of stream) {
  const lastMsg = chunk.messages.at(-1);

  if (lastMsg?.type === "ai") {
    if (lastMsg.tool_calls?.length) {
      stepCount++;
      console.log(`\n[Step ${stepCount}] Calling tools: ${lastMsg.tool_calls.map(t => t.name).join(", ")}`);
    }
    if (lastMsg.content && !lastMsg.tool_calls?.length) {
      console.log(`\n[Final] Assistant: ${lastMsg.content}`);
    }
  } else if (lastMsg?.type === "tool") {
    console.log(`✅ Result: ${lastMsg.content?.substring(0, 100)}...`);
  }
}
```

---

## 7. 性能和成本优化

### 优化策略

| 策略 | 实现 | 效果 |
|------|------|------|
| **动态模型选择** | 简单任务用 mini，复杂用完整版 | 成本 ↓ 40-60% |
| **消息修剪** | 仅保留最近 N 条 | Token 用量 ↓ 50% |
| **缓存** | Anthropic 提示词缓存 | 延迟 ↓ 70% |
| **并行工具调用** | 独立操作同时执行 | 时间 ↓ 60-80% |
| **结构化输出** | 避免不必要的重试 | Token 用量 ↓ 30% |

### 完整的优化配置

```typescript
import { createAgent, createMiddleware } from "langchain";

const optimizedAgent = createAgent({
  model: "openai:gpt-4o-mini", // 默认使用轻量级模型
  tools,
  middleware: [
    // 1. 内容审核
    contentModerationMiddleware,

    // 2. 消息修剪（保留最近 15 条）
    createMiddleware({
      name: "MessagePruning",
      beforeModel: (request, handler) => {
        const pruned = request.messages.slice(-15);
        return handler({ ...request, messages: pruned });
      },
    }),

    // 3. 动态模型选择
    createMiddleware({
      name: "DynamicModel",
      wrapModelCall: (request, handler) => {
        const model = request.messages.length > 20
          ? advancedModel
          : basicModel;
        return handler({ ...request, model });
      },
    }),

    // 4. 成本追踪
    createMiddleware({
      name: "CostTracking",
      afterModel: (response, handler) => {
        const cost = calculateCost(response.usage);
        console.log(`Cost: $${cost.toFixed(4)}`);
        return handler(response);
      },
    }),
  ],
});
```

---

## 8. 常见问题

### Q: 如何限制 Agent 的最大迭代次数？

```typescript
const agent = createAgent({
  model: "openai:gpt-4o",
  tools,
  maxIterations: 10, // 最多执行 10 步
});
```

### Q: 如何处理工具调用失败？

使用 `wrapToolCall` 中间件捕获异常并返回用户友好的错误消息。

### Q: Agent 可以调用另一个 Agent 吗？

可以，将另一个 Agent 包装为工具：

```typescript
const subAgent = createAgent({
  model: "openai:gpt-4o",
  tools: [...],
});

const metaAgentTool = tool(
  async ({ query }) => {
    const result = await subAgent.invoke({
      messages: [{ role: "user", content: query }],
    });
    return String(result.messages[result.messages.length - 1].content);
  },
  {
    name: "delegate_to_expert",
    description: "Delegate complex queries to expert agent",
    schema: z.object({
      query: z.string(),
    }),
  }
);

const mainAgent = createAgent({
  model: "openai:gpt-4o",
  tools: [metaAgentTool, ...otherTools],
});
```

---

## 总结

LangChain.js Agents 1.0 的核心优势：

✅ **统一的 `createAgent` API** — 简化创建流程  
✅ **强大的中间件系统** — 灵活控制执行  
✅ **动态配置支持** — 模型、提示词、工具的运行时选择  
✅ **生产级功能** — 流式处理、错误处理、成本控制  
✅ **完整的工具生态** — 3 种工具定义方式、并行执行  

建议快速启动项目时：
1. 使用字符串标识符创建基础 Agent
2. 通过中间件逐步添加功能（成本控制、审核等）
3. 使用流式处理提升用户体验
4. 通过结构化输出确保数据安全