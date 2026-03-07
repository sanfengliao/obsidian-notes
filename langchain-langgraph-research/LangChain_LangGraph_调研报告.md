# 概述

**LangChain** 和 **LangGraph** 是 LangChain 生态中的两个核心框架，定位不同但可以组合使用。

**LangChain** 是一个 Agent 构建框架，提供了构建 AI Agent 所需的基础组件：

- **Models** - 统一的 LLM 调用接口，支持 OpenAI、Anthropic、Google 等多家提供商
- **Messages** - 标准化的消息系统，支持多模态内容
- **Tools** - 让 Agent 能够执行实际操作的工具系统
- **Memory** - 短期记忆管理，支持多轮对话
- **Middleware** - 可插拔的中间件，用于日志、重试、限流等
- **MCP** - Model Context Protocol 协议集成
- **Multi-Agent** - 多 Agent 协作模式
- **Guardrails** - 安全防护，包括 PII 检测、内容过滤
- **Human-in-the-Loop** - 人工干预机制

**LangGraph** 是一个工作流编排引擎，提供了图结构的状态管理和流程控制能力：

- **StateGraph** - 基于图的状态机定义
- **Nodes/Edges** - 节点和边的灵活编排，支持条件分支和循环
- **Checkpointer** - 状态持久化，支持断点续跑
- **Store** - 长期存储，支持语义搜索
- **Streaming** - 5 种流式处理模式
- **Subgraphs** - 子图嵌套，支持模块化设计

简单来说，LangChain 提供 Agent 的"大脑"（模型、工具、消息）, 适合创建单一agent。LangGraph 提供"骨架"（状态图、流程控制），适合创建复杂的多agent系统结构。两者可以独立使用，也可以组合使用。

# LangChain 核心架构

## 架构概览

LangChain 提供构建 AI Agent 的完整组件栈：

[](https://iwiki.woa.com/tencent/api/attachments/s3/url?attachmentid=39971862)

## 快速开始：createAgent

createAgent 是创建 Agent 的核心函数，一行代码就能创建一个功能完整的 Agent：

```
import { createAgent } from "langchain";

const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [searchTool, calculatorTool],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "北京今天天气怎么样？" }],
});

```

### createAgent 参数详解

```
const agent = createAgent({
  // 必填：模型配置
  model: "openai:gpt-4o",           // 字符串格式，或传入模型实例

  // 必填：工具列表
  tools: [tool1, tool2],            // 空数组 [] 则退化为纯 LLM

  // 可选：系统提示
  systemPrompt: "You are a helpful assistant.",

  // 可选：中间件
  middleware: [loggingMiddleware, retryMiddleware],

  // 可选：状态持久化
  checkpointer: new MemorySaver(),

  // 可选：自定义状态 Schema
  stateSchema: customStateSchema,

  // 可选：运行时上下文 Schema
  contextSchema: contextSchema,
});

```

### 两种模型配置方式

1. 字符串指定（简单快捷）

```
const agent = createAgent({
  model: "openai:gpt-4o",  // 格式: provider:model
  tools: [],
});

```

1. 模型实例（完全控制）

```
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,        // 确定性强，适合任务导向
  maxTokens: 1000,         // 限制输出长度
  timeout: 30,             // 30秒超时
});

const agent = createAgent({ model, tools: [] });

```

### 调用方式

1. invoke - 等待完成

```
const result = await agent.invoke({
  messages: [{ role: "user", content: "帮我搜索 AI 新闻" }],
});
console.log(result.messages.at(-1)?.content);

```

1. stream - 实时查看进度

```
const stream = await agent.stream(
  { messages: [{ role: "user", content: "搜索 AI 新闻并总结" }] },
  { streamMode: "values" }
);

for await (const chunk of stream) {
  const latestMessage = chunk.messages.at(-1);

  if (latestMessage?.content) {
    console.log(`Agent: ${latestMessage.content}`);
  } else if (latestMessage?.tool_calls) {
    const toolNames = latestMessage.tool_calls.map((tc) => tc.name);
    console.log(`正在调用工具: ${toolNames.join(", ")}`);
  }
}

```

### Agent 的 ReAct 循环

Agent 内部使用 ReAct（Reasoning + Acting）模式自动完成任务：

[](https://iwiki.woa.com/tencent/api/attachments/s3/url?attachmentid=39972733)

与直接调用 LLM 相比，Agent 的优势：

- **工具调用自动化**：自动执行工具并处理结果，无需手动循环
- **容错能力**：工具失败时可以自动重试或调整策略
- **状态记忆**：自动保存对话历史
- **并行执行**：没有依赖关系的工具可以同时执行

## 核心组件详解

### Models（模型层）

模型是 Agent 的"大脑"，LangChain 提供统一的接口来调用不同厂商的 LLM。

### 支持的提供商

| 提供商          | 模型示例                 | 初始化格式                         |
| ------------ | -------------------- | ----------------------------- |
| OpenAI       | GPT-4, GPT-4o, o1    | "openai:gpt-4o"               |
| Anthropic    | Claude 3.5, Claude 4 | "anthropic:claude-sonnet-4-5" |
| Google       | Gemini 2.0           | "google:gemini-2.0-flash"     |
| Azure OpenAI | GPT-4 (Azure)        | "azure:gpt-4"                 |
| 本地模型         | Ollama, LlamaCpp     | "ollama:llama2"               |

### 模型初始化

1. 自动初始化

```
import { initChatModel } from "langchain";

// 自动处理 API 密钥和配置
const model = await initChatModel("gpt-4o");

// 指定提供商
const claudeModel = await initChatModel("anthropic:claude-sonnet-4-5");

```

1. 使用实例初始化，更加灵活，可以自定义 baseURL 和 apiKey

```
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,
  maxTokens: 1000,
  timeout: 30,
  maxRetries: 3,
  configuration: {
    baseURL: "",
    apiKey: ""
  }
});

```

### 模型的三种调用方式

| 方式 | 用途 | 特点 |
| --- | --- | --- |
| invoke | 单次调用 | 等待完整响应，最常用 |
| stream | 流式调用 | 逐 token 返回，提升体验 |
| batch | 批量调用 | 并行处理多个请求 |

```
// invoke - 最基础的调用
const response = await model.invoke("Hello!");
console.log(response.content);

// stream - 实时看到生成过程
const stream = await model.stream("Why do parrots talk?");
for await (const chunk of stream) {
  process.stdout.write(chunk.content);  // 逐步输出
}

// batch - 并行处理多个请求
const responses = await model.batch([
  "What is AI?",
  "What is ML?",
  "What is DL?",
], { maxConcurrency: 5 });  // 最多同时 5 个请求

```

### 工具绑定（Tool Binding）

工具绑定让模型能够"知道"有哪些工具可用，并决定何时调用它们。

```
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// 定义工具
const weatherTool = tool(
  async ({ city }) => {
    const data = await fetchWeather(city);
    return JSON.stringify(data);
  },
  {
    name: "get_weather",
    description: "获取指定城市的天气信息",
    schema: z.object({
      city: z.string().describe("城市名称，如 'Beijing'")
    })
  }
);

// 绑定工具到模型
const modelWithTools = model.bindTools([weatherTool]);

```

**两步调用流程**

工具调用是一个**两步过程**，模型本身不会直接执行工具，而是告诉你"我需要调用这个工具"，然后由你的代码执行工具并把结果反馈给模型：

[](https://iwiki.woa.com/tencent/api/attachments/s3/url?attachmentid=39973064)

为什么是两步？

- 模型只负责"思考"：分析用户意图，决定需要什么信息
- 工具执行由你的代码控制：可以添加权限检查、日志记录、错误处理等
- 更安全：你可以在执行前审核工具调用参数

完整代码示例：

```
import { ToolMessage } from "@langchain/core/messages";

// 绑定工具到模型
const modelWithTools = model.bindTools([weatherTool]);

// 步骤 1: 模型分析用户请求
const messages = [{ role: "user", content: "北京今天天气怎么样？" }];
const aiResponse = await modelWithTools.invoke(messages);

// aiResponse.tool_calls 包含模型想要调用的工具信息
// [{ id: "call_abc123", name: "get_weather", args: { city: "Beijing" } }]

// 将 AI 响应加入消息历史
messages.push(aiResponse);

//  步骤 2: 执行工具并返回结果
if (aiResponse.tool_calls && aiResponse.tool_calls.length > 0) {
  for (const toolCall of aiResponse.tool_calls) {
    console.log(`模型请求调用工具: ${toolCall.name}`);
    console.log(`参数: ${JSON.stringify(toolCall.args)}`);

    // 执行工具（这里可以添加权限检查、日志等）
    const toolResult = await weatherTool.invoke(toolCall.args);

    // 构建 ToolMessage 返回给模型
    // 注意：tool_call_id 必须与 toolCall.id 匹配
    const toolMessage = new ToolMessage({
      tool_call_id: toolCall.id,
      content: toolResult,
    });

    messages.push(toolMessage);
  }

  // 步骤 3: 模型根据工具结果生成最终回答
  const finalResponse = await modelWithTools.invoke(messages);
  console.log(finalResponse.content);
  // "北京今天天气晴朗，气温 25°C，适合外出活动。"
}

```

消息流转过程：

| 步骤 | 消息类型 | 内容 |
| --- | --- | --- |
| 1 | HumanMessage | "北京今天天气怎么样？" |
| 2 | AIMessage | { tool_calls: [{ name: "get_weather", args: { city: "Beijing" } }] } |
| 3 | ToolMessage | "{"city": "Beijing", "temperature": 25, "condition": "sunny"}" |
| 4 | AIMessage | "北京今天天气晴朗，气温 25°C..." |

> 直接使用模型调用时需要手动执行这个循环。如果使用 createAgent 创建 Agent，它会自动完成整个流程，包括多轮工具调用。
> 

**支持绑定多个工具**

```
const modelWithTools = model.bindTools([
  weatherTool,
  calculatorTool,
  searchTool,
  databaseTool

]);
// 模型会根据用户输入自动选择合适的工具
const response = await modelWithTools.invoke("帮我计算 123 * 456");
// tool_calls: [{ name: "calculator", args: { expression: "123 * 456" } }]
```

### 结构化输出（Structured Output）

结构化输出确保模型返回符合特定 Schema 的数据，而不是自由文本。

1. 基本用法

```
import { z } from "zod";

// 定义输出结构
const ResponseSchema = z.object({
  answer: z.string().describe("问题的答案"),
  confidence: z.number().min(0).max(1).describe("置信度，0-1之间"),
  sources: z.array(z.string()).describe("信息来源列表"),
  reasoning: z.string().optional().describe("推理过程")
});

// 创建结构化输出模型
const structuredModel = model.withStructuredOutput(ResponseSchema);

// 调用 - 直接返回结构化对象
const result = await structuredModel.invoke("法国的首都是哪里？");
console.log(result);
// {
//   answer: "Paris",
//   confidence: 0.99,
//   sources: ["Wikipedia", "Geography textbook"],
//   reasoning: "Paris is the capital and largest city of France..."
// }

```

1. 复杂嵌套结构

```
const ProductReviewSchema = z.object({
  product: z.object({
    name: z.string(),
    category: z.string(),
    price: z.number()
  }),
  review: z.object({
    rating: z.number().min(1).max(5),
    pros: z.array(z.string()),
    cons: z.array(z.string()),
    summary: z.string()
  }),
  recommendation: z.enum(["highly_recommend", "recommend", "neutral", "not_recommend"])
});

const reviewModel = model.withStructuredOutput(ProductReviewSchema);
const review = await reviewModel.invoke("分析这款 iPhone 15 Pro 的用户评价...");

```

1. 带验证的结构化输出

```
const structuredModel = model.withStructuredOutput(ResponseSchema, {
  // 包含原始响应（用于调试）
  includeRaw: true,

  // 输出方法
  method: "json_schema",  // 或 "function_calling"

  // 严格模式（确保完全符合 schema）
  strict: true
});

const { raw, parsed } = await structuredModel.invoke("...");
console.log("原始响应:", raw);
console.log("解析结果:", parsed);

```

**实际应用场景**

| 场景 | Schema 示例 | 用途 |
| --- | --- | --- |
| **信息抽取** | { name, email, phone } | 从文本中提取联系人信息 |
| **情感分析** | { sentiment, score, keywords } | 分析评论情感倾向 |
| **分类任务** | { category, confidence, explanation } | 文本分类 |
| **数据转换** | { fields: [...] } | 非结构化→结构化数据 |
| **决策输出** | { decision, reasoning, alternatives } | AI 决策记录 |

```
// 信息抽取示例
const ContactSchema = z.object({
  contacts: z.array(z.object({
    name: z.string(),
    email: z.string().email().optional(),
    phone: z.string().optional(),
    company: z.string().optional()
  }))
});

const extractModel = model.withStructuredOutput(ContactSchema);
const result = await extractModel.invoke(`
  请从以下邮件中提取联系人信息：

  Hi, I'm John Smith from Acme Corp.
  You can reach me at john@acme.com or call 555-1234.
  Also CC my colleague Jane Doe (jane@acme.com).
`);
// { contacts: [
//   { name: "John Smith", email: "john@acme.com", phone: "555-1234", company: "Acme Corp" },
//   { name: "Jane Doe", email: "jane@acme.com", company: "Acme Corp" }
// ]}

```

### Messages（消息系统）

消息是 Agent 与 LLM 通信的基本单位，LangChain 定义了四种标准消息类型。

### 四种消息类型

| 类型 | 角色 | 用途 | 示例 |
| --- | --- | --- | --- |
| SystemMessage | system | 设定 AI 行为和规则 | "你是一个专业的客服助手" |
| HumanMessage | user | 用户的输入 | "帮我查询订单状态" |
| AIMessage | assistant | AI 的响应 | "好的，请提供订单号" |
| ToolMessage | tool | 工具执行结果 | { temperature: 25 } |

### 消息创建方式

1. 对象字面量（简洁）

```
const messages = [
  { role: "system", content: "You are a translator." },
  { role: "user", content: "Translate: Hello" },
  { role: "assistant", content: "Bonjour" },
  { role: "user", content: "Translate: Goodbye" },
];

```

1. 消息类（类型安全**）**

```
import { SystemMessage, HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";

const messages = [
  new SystemMessage("You are a translator."),
  new HumanMessage("Translate: Hello"),
  new AIMessage("Bonjour"),
  new HumanMessage("Translate: Goodbye"),
];

```

### 多模态消息

支持文本、图片、音频等多种内容类型：

```
// 图片理解
new HumanMessage({
  content: [
    { type: "text", text: "这张图片里有什么？" },
    {
      type: "image_url",
      image_url: {
        url: "data:image/jpeg;base64,/9j/4AAQ...",
        detail: "high"  // low/high/auto
      }
    }
  ]
});

// 多图片对比
new HumanMessage({
  content: [
    { type: "text", text: "比较这两张图片的区别" },
    { type: "image_url", image_url: { url: "image1.jpg" } },
    { type: "image_url", image_url: { url: "image2.jpg" } },
  ]
});

```

### 

### Tools（工具系统）

工具让 Agent 能够执行实际操作——访问 API、查询数据库、执行代码等。

### 创建工具

```
import { tool } from "@langchain/core/tools";
import * as z from "zod";

const searchDatabase = tool(
  async ({ query, limit }) => {
    // 执行实际搜索逻辑
    const results = await db.search(query, limit);
    return JSON.stringify(results);
  },
  {
    name: "search_database",
    description: "Search the customer database. Use when user asks about customer info.",
    schema: z.object({
      query: z.string().describe("Search terms to look for"),
      limit: z.number().default(10).describe("Max results to return"),
    }),
  }
);

```

关键参数：

1. **执行函数**: 接收参数，返回结果（可以是异步）
2. **name**: 工具标识符，LLM 用此决定调用哪个工具
3. **description**: 说明何时使用此工具，影响 LLM 决策
4. **schema**: Zod 对象，定义输入参数和验证规则

### 工具设计最佳实践

| 原则 | 说明 | 示例 |
| --- | --- | --- |
| **单一职责** | 每个工具只做一件事 | 分离 search 和 filter |
| **清晰描述** | 说明何时使用 | "Use when user asks about weather" |
| **参数说明** | 每个参数都有 describe | z.string().describe("城市名称") |
| **错误处理** | 返回有意义的错误信息 | "City not found: Beijing" |
| **返回格式** | 统一的返回结构 | JSON 字符串或结构化对象 |

### MCP（Model Context Protocol）

### 集成

```
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// 配置多个 MCP 服务器
const client = new MultiServerMCPClient({
  // 本地文件系统服务器
  filesystem: {
    transport: "stdio",
    command: "npx",
    args: ["-y", "@anthropic/mcp-server-filesystem"],
  },
  // 远程天气服务
  weather: {
    transport: "sse",
    url: "https://mcp.example.com/weather",
    headers: { "Authorization": "Bearer xxx" },
  },
  // 数据库服务
  database: {
    transport: "stdio",
    command: "python",
    args: ["mcp_postgres_server.py"],
    env: { "DATABASE_URL": process.env.DATABASE_URL },
  },
});

// 获取所有工具
const tools = await client.getTools();

// 创建 Agent
const agent = createReactAgent({
  llm: model,
  tools,
});

```

### 

### Middleware（中间件）

Middleware 让你在 Agent 执行的各个阶段插入自定义逻辑，无需修改核心代码。

### Agent 执行流程

[](https://iwiki.woa.com/tencent/api/attachments/s3/url?attachmentid=39974044)

### 两种钩子类型

Node-Style Hooks（顺序执行）

```
import { createMiddleware } from "langchain";

const loggingMiddleware = createMiddleware({
  name: "LoggingMiddleware",
  beforeAgent: (state) => {
    console.log("🚀 Agent starting...");
  },
  beforeModel: (state) => {
    console.log(`📤 Calling model with ${state.messages.length} messages`);
  },
  afterModel: (state) => {
    const lastMsg = state.messages.at(-1);
    console.log(`📥 Model response: ${lastMsg?.content?.slice(0, 100)}...`);
  },
  beforeTool: (state, toolCall) => {
    console.log(`🔧 Calling tool: ${toolCall.name}`);
  },
  afterTool: (state, toolResult) => {
    console.log(`✅ Tool result: ${toolResult.slice(0, 100)}...`);
  },
  afterAgent: (state) => {
    console.log("🏁 Agent completed");
  },
});

```

Wrap-Style Hooks（拦截执行）

```
const retryMiddleware = createMiddleware({
  name: "RetryMiddleware",
  wrapModelCall: async (request, handler) => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await handler(request);
      } catch (e) {
        if (attempt === 2) throw e;
        console.log(`Retry ${attempt + 1}/3...`);
        await sleep(1000 * Math.pow(2, attempt));  // 指数退避
      }
    }
  },
  wrapToolCall: async (request, handler) => {
    const startTime = Date.now();
    try {
      const result = await handler(request);
      console.log(`Tool ${request.toolCall.name} took ${Date.now() - startTime}ms`);
      return result;
    } catch (e) {
      console.error(`Tool ${request.toolCall.name} failed:`, e);
      throw e;
    }
  },
});

```

### 内置 Middleware

| Middleware | 功能 | 使用场景 |
| --- | --- | --- |
| summarizationMiddleware | 自动总结长对话 | 长对话管理 |
| modelCallLimitMiddleware | 限制模型调用次数 | 防止无限循环 |
| toolCallLimitMiddleware | 限制工具调用次数 | 控制 API 成本 |
| toolRetryMiddleware | 工具失败自动重试 | 提高可靠性 |
| toolFallbackMiddleware | 工具失败时降级 | 容错处理 |
| piiDetectionMiddleware | PII 信息检测 | 隐私保护 |
| humanInTheLoopMiddleware | 人工审批 | 高风险操作 |

```
import {
  createReactAgent,
  summarizationMiddleware,
  modelCallLimitMiddleware,
  toolRetryMiddleware
} from "langchain";

const agent = createReactAgent({
  llm: model,
  tools,
  middleware: [
    // 消息管理
    summarizationMiddleware({
      model: "gpt-4o-mini",
      trigger: { tokens: 4000 },
      keep: { messages: 20 },
    }),
    // 执行控制
    modelCallLimitMiddleware({
      runLimit: 10,
      exitBehavior: "end",
    }),
    // 重试机制
    toolRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2.0,
      initialDelayMs: 1000,
    }),
  ],
});

```

### 多 Middleware 执行顺序

```
const agent = createAgent({
  middleware: [middleware1, middleware2, middleware3],
});

```

| 钩子类型 | 执行顺序 |
| --- | --- |
| before_* | 1 → 2 → 3（顺序） |
| after_* | 3 → 2 → 1（反向） |
| wrap_* | 1 wrap 2，2 wrap3（嵌套） |

### Multi-Agent（多 Agent 系统）

Multi-Agent 系统通过协调专业化组件来解决复杂工作流。但并非所有复杂任务都需要这种方法——一个具有正确工具和提示的单一 Agent 往往能达到类似效果。

### 何时需要 Multi-Agent

| 需求 | 说明 | 示例 |
| --- | --- | --- |
| **上下文管理** | 提供专业知识而不使模型上下文超载 | 每个 Agent 专注一个领域 |
| **分布式开发** | 不同团队独立开发和维护 | 独立部署和测试 |
| **并行化** | 生成专业化 worker 并发执行 | 同时处理多个子任务 |

**核心要点**: Multi-Agent 设计的核心是 **Context Engineering**——决定每个 Agent 看到什么信息。

### 五种核心模式

| 模式 | 描述 | 最佳场景 |
| --- | --- | --- |
| **Subagents** | 主 Agent 将子 Agent 作为工具协调 | 多个独立领域，集中控制 |
| **Handoffs** | 基于状态动态改变行为 | 顺序流程，需要与用户直接交互 |
| **Skills** | 按需加载专业提示和知识 | 许多专业化，轻量级组合 |
| **Router** | 路由步骤分类输入并分发到专业 Agent | 明确的垂直领域，并行查询 |
| **Custom Workflow** | 用 LangGraph 构建定制执行流 | 需要完全控制 |

### Subagents（子 Agent）

主 Agent 将子 Agent 作为工具调用来协调它们。

```
import { createAgent, tool } from "langchain";
import { z } from "zod";

// 创建子 Agent
const researchAgent = createAgent({
  model: "anthropic:claude-sonnet-4-20250514",
  tools: [searchTool, scrapeTool],
  systemPrompt: "You are a research specialist..."
});

const codeAgent = createAgent({
  model: "anthropic:claude-sonnet-4-20250514",
  tools: [runCodeTool, lintTool],
  systemPrompt: "You are a coding expert..."
});

// 将子 Agent 包装为工具
const callResearchAgent = tool(
  async ({ query }) => {
    const result = await researchAgent.invoke({
      messages: [{ role: "user", content: query }]
    });
    return result.messages.at(-1)?.content;
  },
  {
    name: "research",
    description: "Research a topic thoroughly. Use for factual questions.",
    schema: z.object({ query: z.string() })
  }
);

const callCodeAgent = tool(
  async ({ task }) => {
    const result = await codeAgent.invoke({
      messages: [{ role: "user", content: task }]
    });
    return result.messages.at(-1)?.content;
  },
  {
    name: "code",
    description: "Write or analyze code. Use for programming tasks.",
    schema: z.object({ task: z.string() })
  }
);

// 主 Agent 协调子 Agent
const mainAgent = createAgent({
  model: "anthropic:claude-sonnet-4-20250514",
  tools: [callResearchAgent, callCodeAgent],
  systemPrompt: `You are a supervisor that coordinates specialized agents.

Available agents:
- research: For factual research and information gathering
- code: For programming and code analysis

Decide which agent(s) to use based on the user's request.`
});

```

### Handoffs（交接）

TODO

### Skills（技能）

TODO

### Router（路由器）

TODO

### **Custom Workflow**

TODO

### Short-Term Memory（短期记忆）

短期记忆让 Agent 记住同一个对话线程内的交互历史。这在长对话中很重要，因为模型的上下文窗口有限制。

### 启用短期记忆

短期记忆的核心是 Checkpointer，它负责保存和恢复对话状态：

```
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

// 同一 thread_id 的调用会共享对话历史
await agent.invoke(
  { messages: [{ role: "user", content: "What's my name?" }] },
  { configurable: { thread_id: "1" } }
);
// Agent 会回答 "Your name is Bob"

```

### 生产环境持久化

开发时 MemorySaver 直接存在内存就够了。生产环境需要数据库持久化：

```
import { createAgent } from "langchain";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const DB_URI = "postgresql://user:pass@localhost:5432/db";
const checkpointer = PostgresSaver.fromConnString(DB_URI);

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  checkpointer,
});

```

### 消息管理策略

对话太长会超过上下文限制。有三种常见策略：

1. 裁剪（Trim）- 保留最新的消息

通过中间件在调用模型前裁剪消息：

```
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

    // 保留第一条消息（系统提示）和最后几条
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

1. 删除（Delete）- 永久移除旧消息

从状态中彻底删除某些消息：

```
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

1. 汇总（Summarize）- 用模型总结旧消息

使用内置的 summarizationMiddleware，不是简单丢弃，而是用另一个模型把早期消息汇总成摘要：

```
import { createAgent, summarizationMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

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
  checkpointer: new MemorySaver(),
});

const config = { configurable: { thread_id: "1" } };
await agent.invoke({ messages: "hi, my name is bob" }, config);
await agent.invoke({ messages: "write a short poem about cats" }, config);
await agent.invoke({ messages: "now do the same but for dogs" }, config);
const finalResponse = await agent.invoke({ messages: "what's my name?" }, config);

console.log(finalResponse.messages.at(-1)?.content);
// Output: "Your name is Bob!"

```

---

# LangGraph 核心架构和使用指南

LangGraph 是一个用于构建状态化、多步骤应用的框架。它将应用建模为**图**：

- **节点（Nodes）**: 执行操作的函数
- **边（Edges）**: 连接节点的路径
- **状态（State）**: 在整个执行过程中持久化的数据

## 快速开始：构建一个计算器 Agent

下面通过一个完整的示例展示如何使用 LangGraph 构建多 Agent 协作系统。我们将创建一个**研究助手系统**，包含三个专业 Agent：

- **Researcher Agent**: 负责信息搜索和收集
- **Analyst Agent**: 负责数据分析和总结
- **Writer Agent**: 负责生成最终报告

### 1. 定义状态

```
import { StateGraph, StateSchema, MessagesValue, ReducedValue, START, END } from "@langchain/langgraph";
import { z } from "zod/v4";

// 定义共享状态
const TeamState = new StateSchema({
  // 消息历史
  messages: MessagesValue,

  // 当前活跃的 Agent
  currentAgent: z.string().default("supervisor"),

  // 研究结果（使用 reducer 累积）
  researchResults: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (a, b) => a.concat(b) }
  ),

  // 分析结果
  analysis: z.string().optional(),

  // 最终报告
  finalReport: z.string().optional(),

  // 任务状态
  taskComplete: z.boolean().default(false),
});

```

### 2. 创建专业 Agent 节点

```
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  temperature: 0,
});

// 定义工具
const searchTool = tool(
  async ({ query }) => {
    // 模拟搜索
    return `Search results for "${query}": [相关信息...]`;
  },
  {
    name: "search",
    description: "Search for information on a topic",
    schema: z.object({ query: z.string() }),
  }
);

const analyzeTool = tool(
  async ({ data }) => {
    return `Analysis of data: [分析结果...]`;
  },
  {
    name: "analyze",
    description: "Analyze data and extract insights",
    schema: z.object({ data: z.string() }),
  }
);

// Researcher Agent 节点
const researcherNode = async (state) => {
  const researcherModel = model.bindTools([searchTool]);

  const response = await researcherModel.invoke([
    new SystemMessage(`You are a research specialist.
Your job is to gather information on the given topic.
Use the search tool to find relevant information.`),
    ...state.messages,
  ]);

  // 如果有工具调用，执行工具
  if (response.tool_calls?.length) {
    const results = [];
    for (const toolCall of response.tool_calls) {
      const result = await searchTool.invoke(toolCall);
      results.push(result.content);
    }
    return {
      messages: [response],
      researchResults: results,
    };
  }

  return { messages: [response] };
};

// Analyst Agent 节点
const analystNode = async (state) => {
  const analystModel = model.bindTools([analyzeTool]);

  const researchContext = state.researchResults.join("\n");

  const response = await analystModel.invoke([
    new SystemMessage(`You are a data analyst.
Analyze the research results and provide insights.

Research Results:
${researchContext}`),
    ...state.messages,
  ]);

  return {
    messages: [response],
    analysis: response.content,
  };
};

// Writer Agent 节点
const writerNode = async (state) => {
  const response = await model.invoke([
    new SystemMessage(`You are a professional writer.
Based on the research and analysis, write a comprehensive report.

Research Results:
${state.researchResults.join("\n")}

Analysis:
${state.analysis || "No analysis available"}`),
    ...state.messages,
  ]);

  return {
    messages: [response],
    finalReport: response.content,
    taskComplete: true,
  };
};

```

### 3. 创建 Supervisor Agent

Supervisor 负责协调各个 Agent 的工作流程。

```
// Supervisor 决策 schema
const routeSchema = z.object({
  next: z.enum(["researcher", "analyst", "writer", "FINISH"]).describe(
    "The next agent to call, or FINISH if the task is complete"
  ),
  reason: z.string().describe("Why this agent was chosen"),
});

const supervisorModel = model.withStructuredOutput(routeSchema);

// Supervisor 节点
const supervisorNode = async (state) => {
  const systemPrompt = `You are a supervisor managing a team of agents:
- researcher: Gathers information on topics
- analyst: Analyzes data and provides insights
- writer: Writes final reports

Based on the current state, decide which agent should act next.
If research is needed, call researcher.
If research is done but analysis is needed, call analyst.
If analysis is done and we need the final report, call writer.
If the task is complete, respond with FINISH.

Current Research Results: ${state.researchResults.length} items
Analysis Done: ${state.analysis ? "Yes" : "No"}
Final Report Done: ${state.finalReport ? "Yes" : "No"}`;

  const decision = await supervisorModel.invoke([
    new SystemMessage(systemPrompt),
    ...state.messages,
  ]);

  return {
    currentAgent: decision.next,
    messages: [new AIMessage(`Supervisor: Routing to ${decision.next}. Reason: ${decision.reason}`)],
  };
};

```

### 4. 定义路由逻辑

```
// 根据 supervisor 的决策路由到对应的 Agent
const routeToAgent = (state) => {
  const next = state.currentAgent;

  if (next === "FINISH" || state.taskComplete) {
    return "__end__";
  }

  return next;  // "researcher" | "analyst" | "writer"
};

```

### 5. 构建和编译多 Agent 图

```
const multiAgentGraph = new StateGraph(TeamState)
  // 添加所有节点
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherNode)
  .addNode("analyst", analystNode)
  .addNode("writer", writerNode)

  // Supervisor 是入口点
  .addEdge(START, "supervisor")

  // Supervisor 路由到各个 Agent
  .addConditionalEdges(
    "supervisor",
    routeToAgent,
    ["researcher", "analyst", "writer", "__end__"]
  )

  // 各个 Agent 完成后返回 Supervisor
  .addEdge("researcher", "supervisor")
  .addEdge("analyst", "supervisor")
  .addEdge("writer", "supervisor")

  // 编译图
  .compile();

```

执行流程图如下

[](https://iwiki.woa.com/tencent/api/attachments/s3/url?attachmentid=39976586)

### 6. 执行多 Agent 系统

```
const result = await multiAgentGraph.invoke({
  messages: [new HumanMessage("Research the impact of AI on healthcare and write a report")],
});

console.log("=== Final Report ===");
console.log(result.finalReport);

console.log("\n=== Execution Trace ===");
for (const msg of result.messages) {
  console.log(`[${msg.type}]: ${msg.content?.slice(0, 100)}...`);
}

```

# LangChain vs LangGraph 对比分析

## 定位对比

| 维度 | LangChain | LangGraph |
| --- | --- | --- |
| **核心定位** | Agent 构建框架 | 工作流编排引擎 |
| **抽象层级** | 高层抽象 | 中层抽象 |
| **控制粒度** | 粗粒度（Agent 级） | 细粒度（节点级） |
| **学习曲线** | 较低 | 中等 |
| **灵活性** | 中等 | 高 |

## 功能对比

| 功能 | LangChain | LangGraph |
| --- | --- | --- |
| Agent 构建 | 原生支持 | 需手动实现 |
| 工作流定义 | 有限支持 | 原生支持 |
| 状态管理 | 基础 | 高级 |
| 条件分支 | 有限 | 完整支持 |
| 循环/迭代 | 有限 | 完整支持 |
| 子图嵌套 | 不支持 | 支持 |

### 使用场景对比

| 场景 | LangChain | LangGraph | 推荐选择 |
| --- | --- | --- | --- |
| 简单对话机器人 | 适合 | 过度设计 | LangChain |
| 基础 RAG 问答 | 适合 | 过度设计 | LangChain |
| 单一工具调用 | 适合 | 过度设计 | LangChain |
| 复杂工作流 | 有限 | 适合 | LangGraph |
| 多 Agent 协作 | 有限 | 适合 | LangGraph |
| 条件分支/循环 | 有限 | 适合 | LangGraph |

### 协同使用

最佳实践是**组合使用**两者：

```
// LangChain 提供组件
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";

// LangGraph 提供编排
import { StateGraph, StateSchema } from "@langchain/langgraph";

// 组合使用
const model = new ChatOpenAI({ model: "gpt-4o" });
const tools = [searchTool, calculatorTool];
const modelWithTools = model.bindTools(tools);

const graph = new StateGraph(State)
  .addNode("agent", async (state) => {
    const response = await modelWithTools.invoke(state.messages);
    return { messages: [response] };
  })
  .addNode("tools", new ToolNode(tools))
  // ... 编排逻辑
  .compile();

```

##