# LangChain Context Engineering 速查指南

## 概览

Context Engineering 是为 LLM 提供恰当的信息和工具，以正确的格式来完成任务。这是 AI 工程师的核心工作，也是提升 Agent 可靠性的首要任务。

**为什么 Agent 会失败？**

大多数情况下，Agent 失败不是因为 LLM 能力不足，而是因为 **没有提供正确的上下文**。Context Engineering 的目的就是解决这个问题。

**核心思想**: Agent 的可靠性取决于我们在三个关键环节能否提供正确的上下文：
1. **模型上下文** - 模型接收什么（指令、工具、消息、输出格式）
2. **工具上下文** - 工具能访问什么和返回什么
3. **生命周期上下文** - Agent 执行过程中发生了什么（总结、守护、日志）

---

## Agent 循环基础

一个典型的 Agent 循环包含两个主要步骤：

1. **模型调用** - 调用 LLM，传入提示和可用工具，返回响应或工具执行请求
2. **工具执行** - 执行 LLM 请求的工具，返回工具结果

循环继续进行，直到 LLM 决定完成任务。

你可以在 Agent 循环的各个步骤中控制上下文，以及步骤之间发生的事情。

---

## 三类上下文对比

| 类型 | 说明 | 作用域 | 生命周期 |
|------|------|--------|--------|
| **模型上下文** | 模型调用中的信息（指令、消息历史、工具、输出格式） | 单次调用 | 临时性 |
| **工具上下文** | 工具能访问和产生的内容（读写 State、Store、运行时上下文） | Agent 范围 | 持久性 |
| **生命周期上下文** | 模型和工具调用之间发生的事情（总结、守护、日志等） | Agent 范围 | 持久性 |

---

## 数据源三层结构

在 Agent 执行过程中，你会使用到三种不同的数据源：

| 数据源 | 类型 | 作用域 | 典型用途 |
|--------|------|--------|---------|
| **运行时上下文** | 静态配置 | 对话范围 | 用户 ID、API 密钥、数据库连接、权限、环境变量 |
| **State** | 短期内存 | 对话范围 | 当前消息、上传文件、认证状态、工具结果 |
| **Store** | 长期内存 | 跨对话 | 用户偏好、提取的洞察、记忆、历史数据 |

---

## 模型上下文

控制每个模型调用中输入什么 - 指令、可用工具、模型选择、输出格式。这些决策直接影响可靠性和成本。

### 1. 系统提示（System Prompt）

系统提示设定 LLM 的行为和能力。不同用户、上下文或对话阶段需要不同的指令。

#### 动态系统提示示例

基于对话长度从 State 中读取，动态调整提示：

```javascript
import { createAgent } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [...],
  middleware: [
    dynamicSystemPromptMiddleware((state) => {
      // 从 State 读取：检查对话长度
      const messageCount = state.messages.length;

      let base = "You are a helpful assistant.";

      // 长对话时更简洁
      if (messageCount > 10) {
        base += "\nThis is a long conversation - be extra concise.";
      }

      return base;
    }),
  ],
});
```

**最佳实践**:
- 根据对话深度调整指令（早期阶段简单，深入阶段详细）
- 根据用户角色或权限提供不同指令
- 利用 Store 中的用户偏好个性化提示

### 2. 消息（Messages）

消息组成发送给 LLM 的提示词。管理消息内容是确保 LLM 有正确信息的关键。

#### 注入文件上下文示例

从 State 中获取上传的文件，并在相关时动态注入：

```javascript
import { createMiddleware } from "langchain";

const injectFileContext = createMiddleware({
  name: "InjectFileContext",
  wrapModelCall: (request, handler) => {
    // 获取上传的文件（从 State）
    const uploadedFiles = request.state.uploadedFiles || [];

    if (uploadedFiles.length > 0) {
      // 构建文件描述
      const fileDescriptions = uploadedFiles.map(file =>
        `- ${file.name} (${file.type}): ${file.summary}`
      );

      const fileContext = `Files you have access to in this conversation:
${fileDescriptions.join("\n")}

Reference these files when answering questions.`;

      // 在消息中注入文件上下文（临时修改，不影响 State）
      const messages = [
        ...request.messages,
        { role: "user", content: fileContext }
      ];
      request = request.override({ messages });
    }

    return handler(request);
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [...],
  middleware: [injectFileContext],
});
```

**重要区别**:
- **临时修改** (`wrapModelCall`): 只影响当前模型调用，不改变 State
- **持久修改** (`beforeModel`/`afterModel`): 永久更新对话历史，影响未来的转向

### 3. 工具（Tools）

工具让模型与数据库、API 和外部系统交互。如何定义和选择工具直接影响模型能否有效完成任务。

#### 定义清晰的工具

每个工具需要清晰的名称、描述、参数名称和参数描述。这些不仅是元数据，还指导模型的推理：

```javascript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchOrders = tool(
  async ({ userId, status, limit }) => {
    // 实现代码
  },
  {
    name: "search_orders",
    description: `Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.`,
    schema: z.object({
      userId: z.string().describe("Unique identifier for the user"),
      status: z.enum(["pending", "shipped", "delivered"]).describe("Order status to filter by"),
      limit: z.number().default(10).describe("Maximum number of results to return"),
    }),
  }
);
```

**工具设计原则**:
- 名称明确表达功能（`search_orders` vs `search`）
- 描述清晰说明何时使用
- 参数描述准确指导正确调用
- 枚举有限的预设值而不是自由文本

#### 动态工具选择

不是每个工具都适合每种情况。太多工具会让模型感到压力，太少会限制能力。

基于认证状态和对话阶段动态选择工具：

```javascript
import { createMiddleware } from "langchain";

const stateBasedTools = createMiddleware({
  name: "StateBasedTools",
  wrapModelCall: (request, handler) => {
    // 从 State 读取：检查认证和对话长度
    const state = request.state;
    const isAuthenticated = state.authenticated || false;
    const messageCount = state.messages.length;

    let filteredTools = request.tools;

    // 仅在认证后启用敏感工具
    if (!isAuthenticated) {
      filteredTools = request.tools.filter(t => t.name.startsWith("public_"));
    } else if (messageCount < 5) {
      // 早期对话不启用高级工具
      filteredTools = request.tools.filter(t => t.name !== "advanced_search");
    }

    return handler({ ...request, tools: filteredTools });
  },
});
```

**动态工具选择场景**:
- 基于认证状态（公开 vs 敏感工具）
- 基于对话进展（初级 vs 高级工具）
- 基于用户权限（受限工具）
- 基于功能标志（A/B 测试）

### 4. 模型选择

不同模型有不同的优点、成本和上下文窗口。选择合适的模型对当前任务很重要，可能在运行过程中改变。

#### 基于对话长度选择模型

```javascript
import { createMiddleware, initChatModel } from "langchain";

const largeModel = initChatModel("claude-sonnet-4-5-20250929");
const standardModel = initChatModel("gpt-4o");
const efficientModel = initChatModel("gpt-4o-mini");

const stateBasedModel = createMiddleware({
  name: "StateBasedModel",
  wrapModelCall: (request, handler) => {
    const messageCount = request.messages.length;
    let model;

    // 长对话需要更强大的模型
    if (messageCount > 20) {
      model = largeModel;
    } else if (messageCount > 10) {
      model = standardModel;
    } else {
      model = efficientModel;
    }

    return handler({ ...request, model });
  },
});
```

**模型选择策略**:
- 初期：便宜高效的模型（gpt-4o-mini）
- 中期：平衡模型（gpt-4o）
- 复杂：强力模型（claude-sonnet）

### 5. 响应格式（Response Format）

结构化输出将非结构化文本转换为验证的结构化数据。

#### 定义输出格式

```javascript
import { z } from "zod";

const customerSupportTicket = z.object({
  category: z.enum(["billing", "technical", "account", "product"]).describe(
    "Issue category"
  ),
  priority: z.enum(["low", "medium", "high", "critical"]).describe(
    "Urgency level"
  ),
  summary: z.string().describe(
    "One-sentence summary of the customer's issue"
  ),
  customerSentiment: z.enum(["frustrated", "neutral", "satisfied"]).describe(
    "Customer's emotional tone"
  ),
}).describe("Structured ticket information extracted from customer message");
```

#### 动态响应格式

基于对话状态调整输出复杂度：

```javascript
import { createMiddleware } from "langchain";
import { z } from "zod";

const simpleResponse = z.object({
  answer: z.string().describe("A brief answer"),
});

const detailedResponse = z.object({
  answer: z.string().describe("A detailed answer"),
  reasoning: z.string().describe("Explanation of reasoning"),
  confidence: z.number().describe("Confidence score 0-1"),
});

const stateBasedOutput = createMiddleware({
  name: "StateBasedOutput",
  wrapModelCall: (request, handler) => {
    const messageCount = request.messages.length;

    let responseFormat;
    if (messageCount < 3) {
      // 早期对话 - 简单格式
      responseFormat = simpleResponse;
    } else {
      // 已建立的对话 - 详细格式
      responseFormat = detailedResponse;
    }

    return handler({ ...request, responseFormat });
  },
});
```

---

## 工具上下文

工具既读取也写入上下文，是 Agent 与外部世界交互的方式。

### 从上下文中读取（Reads）

工具通常需要比 LLM 参数更多的信息。

#### 从 State 读取认证状态

```javascript
import * as z from "zod";
import { createAgent, tool, type ToolRuntime } from "langchain";

const checkAuthentication = tool(
  async (_, runtime: ToolRuntime) => {
    // 从 State 读取：检查当前认证状态
    const currentState = runtime.state;
    const isAuthenticated = currentState.authenticated || false;

    if (isAuthenticated) {
      return "User is authenticated";
    } else {
      return "User is not authenticated";
    }
  },
  {
    name: "check_authentication",
    description: "Check if user is authenticated",
    schema: z.object({}),
  }
);
```

**工具可读取的上下文**:
- **State**: 当前消息、认证状态、用户信息
- **Store**: 用户偏好、历史数据、提取的洞察
- **Runtime**: API 密钥、数据库连接、权限

### 写入上下文（Writes）

工具可以更新 State 和 Store，为未来的步骤提供重要信息。

#### 写入 State 更新认证状态

```javascript
import * as z from "zod";
import { tool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { Command } from "@langchain/langgraph";

const authenticateUser = tool(
  async ({ password }) => {
    // 执行认证
    if (password === "correct") {
      // 写入 State：标记为已认证
      return new Command({
        update: { authenticated: true },
      });
    } else {
      return new Command({ update: { authenticated: false } });
    }
  },
  {
    name: "authenticate_user",
    description: "Authenticate user and update State",
    schema: z.object({
      password: z.string(),
    }),
  }
);
```

**工具写入最佳实践**:
- 用 `Command` 对象更新 State
- 只写入重要的上下文信息
- 明确说明写入的用途和影响

---

## 生命周期上下文

控制 Agent 执行步骤之间发生的事情 - 实现跨领域的关注点，如总结、守护和日志。

### 示例：对话总结

当对话变得太长时自动压缩历史记录。与模型上下文中的临时消息修剪不同，总结持久地更新 State。

```javascript
import { createAgent, summarizationMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [...],
  middleware: [
    summarizationMiddleware({
      model: "gpt-4o-mini",
      trigger: { tokens: 4000 },  // 触发阈值
      keep: { messages: 20 },     // 保留最近 20 条消息
    }),
  ],
});
```

**总结工作流程**:
1. 当对话超过令牌限制时触发
2. 使用独立 LLM 调用总结旧消息
3. 将总结替换到 State（永久）
4. 保留最近消息以保留上下文

**生命周期钩子中的其他用途**:
- 记录和审计
- 验证和守护
- 动态提示修改
- 状态清理

---

## Context Engineering 最佳实践

### 1. 渐进式开发

- 从静态提示和工具开始
- 只在需要时才添加动态元素
- 逐个测试每个新特性

### 2. 增量测试

- 一次添加一个 context engineering 特性
- 测试每个步骤的影响

### 3. 性能监控

- 跟踪模型调用次数
- 监控令牌使用量
- 记录延迟指标

### 4. 利用内置中间件

使用已提供的实现而不是重新造轮子：
- `summarizationMiddleware` - 对话总结
- `piiRedactionMiddleware` - PII 保护
- `llmToolSelectorMiddleware` - 智能工具选择

### 5. 文档化 Context 策略

清楚地记录：
- 传递什么上下文
- 为什么要传递
- 在哪个阶段传递

### 6. 临时 vs 持久

理解两者的区别很关键：
- **临时** (模型上下文): 单次调用修改，不影响 State
- **持久** (生命周期上下文): 永久更新 State，影响未来转向

---

## 常见问题

**Q: 应该在哪个阶段添加上下文？**

A: 按优先级排列：
1. 模型上下文（系统提示、消息、工具）
2. 工具上下文（工具如何访问数据）
3. 生命周期上下文（消息总结、日志）

**Q: 太多工具会怎样？**

A: 模型会感到困惑，易出错。使用动态工具选择，只提供相关工具。

**Q: 临时和持久修改何时使用？**

A: 
- 临时：一次性修改（修剪消息、注入上下文）
- 持久：需要记住的修改（总结、状态更新）

**Q: 如何避免上下文溢出？**

A: 
- 使用消息修剪或总结
- 动态选择模型（长对话用更强模型）
- 实现滑动窗口机制

**Q: Store 和 State 何时使用？**

A:
- **State**: 当前对话的信息（消息、文件、认证）
- **Store**: 跨多个对话的信息（用户偏好、历史）

---

## 相关资源

- [Context 概念概览](https://docs.langchain.com/oss/javascript/concepts/context) - 理解上下文类型
- [Middleware 文档](https://docs.langchain.com/oss/javascript/langchain/middleware) - 完整的中间件指南
- [工具](https://docs.langchain.com/oss/javascript/langchain/tools) - 工具创建和上下文访问
- [记忆](https://docs.langchain.com/oss/javascript/concepts/memory) - 短期和长期记忆模式
- [Agents](https://docs.langchain.com/oss/javascript/langchain/agents) - Agent 核心概念
