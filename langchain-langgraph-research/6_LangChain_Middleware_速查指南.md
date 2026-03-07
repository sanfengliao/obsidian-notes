# LangChain Middleware 完全指南

## 什么是 Middleware

Middleware 让你在 Agent 执行的各个步骤上插入自定义逻辑。不需要修改 Agent 核心代码，就能添加日志、重试、验证、安全检查等功能。

想象 Agent 的执行流程是一个管道，middleware 就像在管道的各个关键点放置的拦截器，可以观察、修改甚至停止流程。

## 核心概念

### Agent 执行流程

```
beginAgent (开始)
     ↓
beforeAgent (执行一次)
     ↓
[循环直到结束]
  beforeModel → 模型调用 → afterModel
  (如有工具调用)
  beforeTool → 工具执行 → afterTool
     ↓
afterAgent (执行一次)
     ↓
结束
```

### Middleware 的两种钩子类型

#### 1. Node-Style Hooks（顺序执行）

用于日志、验证、状态更新。按定义的顺序依次执行：

```javascript
import { createMiddleware } from "langchain";

const loggingMiddleware = createMiddleware({
  name: "LoggingMiddleware",
  beforeAgent: (state) => {
    console.log("Agent starting...");
  },
  beforeModel: (state) => {
    console.log(`📤 Model call with ${state.messages.length} messages`);
  },
  afterModel: (state) => {
    console.log(`📥 Model returned: ${state.messages.at(-1)?.content?.slice(0, 50)}...`);
  },
  afterAgent: (state) => {
    console.log("Agent completed");
  },
});
```

可用的钩子：
- `beforeAgent` - Agent 开始前（执行一次）
- `beforeModel` - 每次模型调用前
- `afterModel` - 每次模型调用后
- `afterAgent` - Agent 完成后（执行一次）

#### 2. Wrap-Style Hooks（拦截执行）

围绕执行步骤，你决定是否调用处理器。用于重试、缓存、转换：

```javascript
const retryMiddleware = createMiddleware({
  name: "RetryMiddleware",
  wrapModelCall: (request, handler) => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return handler(request);  // 调用处理器
      } catch (e) {
        if (attempt === 2) throw e;
        console.log(`Retry ${attempt + 1}/3`);
      }
    }
  },
  wrapToolCall: (request, handler) => {
    console.log(`🔧 Calling tool: ${request.toolCall.name}`);
    try {
      const result = handler(request);
      console.log(`✅ Tool succeeded`);
      return result;
    } catch (e) {
      console.log(`❌ Tool failed: ${e}`);
      throw e;
    }
  },
});
```

可用的钩子：
- `wrapModelCall` - 围绕模型调用
- `wrapToolCall` - 围绕工具调用

### 多 Middleware 的执行顺序

```javascript
const agent = createAgent({
  model: "gpt-4o",
  middleware: [middleware1, middleware2, middleware3],
});
```

**执行规则**：
- `before_*` 钩子：1→2→3（顺序）
- `after_*` 钩子：3→2→1（反向）
- `wrap_*` 钩子：1 包装 2，2 包装 3（嵌套）

## 内置 Middleware（开箱即用）

LangChain 提供了一套生产就绪的 middleware，可直接使用。

### 消息管理

#### 消息汇总（Summarization）

当消息超过 token 限制时自动汇总旧消息，保留最新的对话：

```javascript
import { createAgent, summarizationMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    summarizationMiddleware({
      model: "gpt-4o-mini",        // 用来汇总的模型
      trigger: { tokens: 4000 },   // 接近 4000 token 时触发
      keep: { messages: 20 },      // 保留最近 20 条消息
    }),
  ],
});
```

**场景**：长对话需要保留全部信息但不能超过上下文。

### 执行控制

#### 模型调用限制（Model Call Limit）

防止模型被调用过多次，避免无限循环和费用失控：

```javascript
import { createAgent, modelCallLimitMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    modelCallLimitMiddleware({
      threadLimit: 10,      // 同一 thread 最多 10 次
      runLimit: 5,          // 单次 invoke 最多 5 次
      exitBehavior: "end",  // 超限后直接结束
    }),
  ],
});
```

**场景**：防止卡住，控制成本。

#### 工具调用限制（Tool Call Limit）

限制工具调用次数，防止昂贵 API 被过度调用：

```javascript
import { createAgent, toolCallLimitMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, databaseTool],
  middleware: [
    // 全局限制
    toolCallLimitMiddleware({ threadLimit: 20, runLimit: 10 }),
    // 特定工具限制
    toolCallLimitMiddleware({
      toolName: "search",
      threadLimit: 5,
      runLimit: 3,
    }),
  ],
});
```

### 恢复和降级

#### 工具重试（Tool Retry）

工具失败时自动重试，支持指数退避：

```javascript
import { createAgent, toolRetryMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, databaseTool],
  middleware: [
    toolRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2.0,      // 延迟翻倍
      initialDelayMs: 1000,    // 初始 1 秒
    }),
  ],
});
```

**场景**：网络波动、临时 API 故障。

#### 模型重试（Model Retry）

模型调用失败时自动重试：

```javascript
import { createAgent, modelRetryMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    modelRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2.0,
      initialDelayMs: 1000,
    }),
  ],
});
```

#### 模型降级（Model Fallback）

主模型失败时自动切换到备用模型：

```javascript
import { createAgent, modelFallbackMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    modelFallbackMiddleware(
      "gpt-4o-mini",                    // 备选 1
      "claude-3-5-sonnet-20241022"      // 备选 2
    ),
  ],
});
```

**场景**：跨 provider 冗余，成本优化。

### 优化和筛选

#### LLM 工具选择器（LLM Tool Selector）

当工具很多时，用 LLM 先智能筛选出相关工具，减少上下文：

```javascript
import { createAgent, llmToolSelectorMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [...50个工具],
  middleware: [
    llmToolSelectorMiddleware({
      model: "gpt-4o-mini",
      maxTools: 3,               // 最多选 3 个
      alwaysInclude: ["search"], // 搜索总是包括
    }),
  ],
});
```

**优势**：更短的提示词，更高准确性，更低成本。

#### LLM 工具模拟（LLM Tool Emulator）

用 LLM 模拟工具执行，用于开发和测试：

```javascript
import { createAgent, toolEmulatorMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [getWeather, sendEmail, queryDB],
  middleware: [
    toolEmulatorMiddleware(),  // 模拟所有工具
  ],
});
```

**场景**：开发阶段，外部工具还不可用或很贵。

### 安全和合规

#### PII 检测（PII Detection）

检测敏感信息并按策略处理（删除、掩码、哈希等）：

```javascript
import { createAgent, piiMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    piiMiddleware("email", { strategy: "redact" }),
    piiMiddleware("credit_card", { strategy: "mask" }),
    piiMiddleware("ssn", { strategy: "hash" }),
  ],
});
```

处理策略：
- `"redact"` - 完全删除
- `"mask"` - 部分显示（如 `****1234`）
- `"hash"` - 转成哈希
- `"block"` - 拒绝

**场景**：医疗、金融、隐私敏感应用。

#### 人工审核（Human-in-the-Loop）

在关键工具执行前暂停，等待人工批准：

```javascript
import { createAgent, humanInTheLoopMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [sendEmail, deleteData, transferMoney],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        sendEmail: { allowedDecisions: ["approve", "edit", "reject"] },
        deleteData: { allowedDecisions: ["approve", "reject"] },
        transferMoney: true,  // 所有决定都可以
      }
    })
  ]
});
```

**需求**：需要 checkpointer 保存中断状态。

### 任务管理

#### 待办清单（To-do List）

为 Agent 添加任务规划和追踪能力：

```javascript
import { createAgent, todoListMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [readFile, writeFile, runTests],
  middleware: [todoListMiddleware()],
});
```

自动提供：
- `write_todos` 工具
- 系统提示指导任务规划

**场景**：复杂多步任务（代码审查、项目规划）。

#### 上下文编辑（Context Editing）

当 token 接近限制时，清除旧的工具输出，保留最新的：

```javascript
import { createAgent, contextEditingMiddleware, ClearToolUsesEdit } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    contextEditingMiddleware({
      edits: [
        new ClearToolUsesEdit({
          triggerTokens: 100000,
          keep: 3,  // 保留最近 3 个工具结果
        }),
      ],
    }),
  ],
});
```

**场景**：长对话，很多工具调用。

## 自定义 Middleware

### 基础：创建自定义 Middleware

```javascript
import { createMiddleware } from "langchain";

const myMiddleware = createMiddleware({
  name: "MyMiddleware",
  beforeModel: (state) => {
    console.log(`Messages: ${state.messages.length}`);
  },
  afterModel: (state) => {
    console.log(`Model response: ${state.messages.at(-1)?.content}`);
  },
});
```

### 自定义状态

在 middleware 中添加自定义状态字段，跨钩子存储数据：

```javascript
import * as z from "zod";
import { createMiddleware, createAgent, HumanMessage } from "langchain";

const callCounterMiddleware = createMiddleware({
  name: "CallCounterMiddleware",
  stateSchema: z.object({
    modelCallCount: z.number().default(0),
    userId: z.string().optional(),
  }),
  beforeModel: (state) => {
    if (state.modelCallCount > 10) {
      return { jumpTo: "end" };  // 超限就终止
    }
  },
  afterModel: (state) => {
    return { modelCallCount: state.modelCallCount + 1 };
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [callCounterMiddleware],
});

const result = await agent.invoke({
  messages: [new HumanMessage("Hello")],
  modelCallCount: 0,
  userId: "user-123",
});
```

**提示**：以下划线 `_` 开头的字段是私有的，不会出现在结果中。

### 自定义 Context

每次调用传入的只读数据，不在调用间持久化：

```javascript
import { createAgent, createMiddleware } from "langchain";
import * as z from "zod";

const contextSchema = z.object({
  userId: z.string(),
  tenantId: z.string(),
  apiKey: z.string().optional(),
});

const userContextMiddleware = createMiddleware({
  name: "UserContextMiddleware",
  contextSchema,
  wrapModelCall: (request, handler) => {
    const { userId, tenantId } = request.runtime.context;
    const contextText = `User ID: ${userId}, Tenant: ${tenantId}`;
    const newSystemMessage = request.systemMessage.concat(contextText);

    return handler({
      ...request,
      systemMessage: newSystemMessage,
    });
  },
});

const agent = createAgent({
  model: "gpt-4o",
  middleware: [userContextMiddleware],
  tools: [],
  contextSchema,
});

// 必须提供必要的 context
const result = await agent.invoke(
  { messages: [new HumanMessage("Hello")] },
  { context: { userId: "user-123", tenantId: "acme" } }
);
```

### 实用例子

#### 例 1：动态选择模型

```javascript
const dynamicModelMiddleware = createMiddleware({
  name: "DynamicModelMiddleware",
  wrapModelCall: (request, handler) => {
    const modifiedRequest = { ...request };
    
    if (request.messages.length > 10) {
      modifiedRequest.model = initChatModel("gpt-4o");
    } else {
      modifiedRequest.model = initChatModel("gpt-4o-mini");
    }
    
    return handler(modifiedRequest);
  },
});
```

#### 例 2：工具调用监控

```javascript
const toolMonitoringMiddleware = createMiddleware({
  name: "ToolMonitoringMiddleware",
  wrapToolCall: (request, handler) => {
    console.log(`📞 Tool: ${request.toolCall.name}`);
    console.log(`📋 Args: ${JSON.stringify(request.toolCall.args)}`);
    try {
      const result = handler(request);
      console.log(`✅ Success`);
      return result;
    } catch (e) {
      console.log(`❌ Failed: ${e}`);
      throw e;
    }
  },
});
```

#### 例 3：修改系统消息

```javascript
const systemPromptMiddleware = createMiddleware({
  name: "SystemPromptModifier",
  wrapModelCall: (request, handler) => {
    const additionalContext = "You are in production mode.";
    const newSystemMessage = request.systemMessage.concat(additionalContext);
    
    return handler({
      ...request,
      systemMessage: newSystemMessage,
    });
  },
});
```

## 提前终止（Agent Jumps）

用 `jumpTo` 跳过执行步骤：

```javascript
const blockedContentMiddleware = createMiddleware({
  name: "BlockedContentCheck",
  beforeModel: (state) => {
    if (state.messages.at(-1)?.content?.includes("BLOCKED")) {
      return {
        messages: [new AIMessage("I cannot respond to that.")],
        jumpTo: "end",  // 直接结束
      };
    }
  },
});
```

可用的跳转目标：
- `'end'` - 跳到 Agent 结束
- `'model'` - 跳到模型调用
- `'tools'` - 跳到工具执行

## 实际应用场景

### 场景 1：生产环保险方案

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: allTools,
  middleware: [
    // 安全层
    piiMiddleware("email", { strategy: "redact" }),
    
    // 控制层
    modelCallLimitMiddleware({ threadLimit: 10 }),
    toolCallLimitMiddleware({ threadLimit: 50 }),
    
    // 优化层
    llmToolSelectorMiddleware({ maxTools: 5 }),
    
    // 恢复层
    toolRetryMiddleware({ maxRetries: 3 }),
    modelRetryMiddleware({ maxRetries: 2 }),
    
    // 长对话管理
    summarizationMiddleware({ trigger: { tokens: 4000 } }),
    
    // 人工控制（高风险操作）
    humanInTheLoopMiddleware({ interruptOn: { sendEmail: true } }),
  ],
});
```

### 场景 2：开发和测试

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [realTools],
  middleware: [
    toolEmulatorMiddleware(),  // 模拟工具，不调真实 API
    summarizationMiddleware({ trigger: { tokens: 2000 } }),  // 更低的阈值用于开发
  ],
});
```

### 场景 3：成本优化

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: tools,
  middleware: [
    modelFallbackMiddleware("gpt-4o-mini", "claude-3-sonnet"),  // 便宜备选
    llmToolSelectorMiddleware({ maxTools: 3 }),                  // 减少上下文
    toolCallLimitMiddleware({ toolName: "expensiveAPI", threadLimit: 5 }),
  ],
});
```

## 最佳实践

1. **单一职责** - 每个 middleware 只负责一个功能
2. **错误处理** - middleware 错误不要奔溃整个 Agent
3. **选择合适的钩子**：
   - Node-style 用于顺序逻辑（日志、验证）
   - Wrap-style 用于控制流（重试、缓存、转换）
4. **清晰文档** - 说明自定义状态的含义
5. **独立测试** - 先单元测试 middleware，再集成
6. **执行顺序** - 关键 middleware 放在前面
7. **优先内置** - 优先用官方 middleware

## 何时使用 Middleware

✅ **适合用 Middleware**：
- 需要在多处添加相同逻辑
- 想改变 Agent 行为但不改核心代码
- 需要跨对话维持状态
- 实现通用功能（日志、重试、安全）

❌ **不适合用 Middleware**：
- 对单个工具的定制：用工具配置
- 简单一次性修改：直接改 Agent
- 特定工具的业务逻辑：写在工具函数里

## 常见问题

**Q：beforeModel vs wrapModelCall 怎么选？**

A：beforeModel 看消息并修改状态。wrapModelCall 拦截调用，决定是否执行或重试。

**Q：多个 middleware 能共享数据吗？**

A：能。用自定义状态或通过 afterModel/beforeModel 链式修改。

**Q：能从 middleware 中调用工具吗？**

A：不建议。工具应该由 Agent 在适当时刻调用。如果需要在 middleware 中执行逻辑，考虑改成工具。

**Q：Middleware 执行失败会怎样？**

A：错误会被抛出。要稳健的应用，在 middleware 中加 try-catch。

**Q：能调整 middleware 的执行顺序吗？**

A：能。在 `middleware: [...]` 数组中调整顺序。关键 middleware 通常放在前面。
