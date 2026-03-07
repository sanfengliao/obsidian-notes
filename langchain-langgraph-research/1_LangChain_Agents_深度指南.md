# LangChain Agents

## 核心概念
### 什么是Agent?
简单说，Agent是一个能自主做决策的系统。给它一个任务，它会通过ReAct循环（Reasoning + Acting）一步步解决：先分析局面，决定用什么工具，执行工具，看结果，再决定下一步。

具体流程是：接收输入 → LLM分析形势 → 选择合适的工具 → 执行工具 → 观察结果 → 判断是否完成（没完成就继续循环）

### 为什么要用Agent？

相比直接调用LLM，Agent强大在哪儿？核心差异有这么几个：

- **工具能力**：LLM只能用预定义好的工具。Agent能根据情况动态选择用哪个工具，甚至一个请求里调用多个工具
- **容错能力**：工具失败了，Agent有重试机制，而不是直接返回错误
- **记忆和状态**：自动保存对话历史，还能自定义追踪额外信息
- **并行处理**：多个独立的工具可以同时执行

### 技术架构

底层用LangGraph实现，本质是个有向图。图里有节点（执行步骤）和边（流向）：
- 模型节点：调用LLM做决策
- 工具节点：执行实际操作
- 中间件层：在关键点拦截、修改数据流

这样的设计让扩展变得很灵活——想添加新能力，大多时候只需加个中间件就行。

## 核心组件详解

### 模型(Model) 

LangChain支持两种模型配置方式。

#### 简单配置：用字符串指定

最快的方式，格式是 `provider:model`：

```javascript
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: []
});
```

这样做的好处是代码简洁，坏处是没法调整超参数。

#### 完全控制：传入模型实例

如果需要调整`temperature`、`maxTokens`这些参数，直接初始化模型对象：

```javascript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,        // 确定性强，适合任务导向
  maxTokens: 1000,         // 限制输出长度
  timeout: 30              // 30秒超时
});

const agent = createAgent({ model, tools: [] });
```

这样你对模型的行为有完整控制权。API密钥、基础URL这些provider特定的配置也都能在这儿设置。

#### 动态模型切换

有些情况下需要根据问题难度动态选模型——简单问题用便宜的mini模型，复杂问题升级到高端模型。这种场景可以使用中间件的wrapModelCall hook来实现，

```javascript
const basicModel = new ChatOpenAI({ model: "gpt-4o-mini" });  // 便宜
const advancedModel = new ChatOpenAI({ model: "gpt-4o" });    // 强大

const dynamicModelSelection = createMiddleware({
  name: "DynamicModelSelection",
  wrapModelCall: (request, handler) => {
    const messageCount = request.messages.length;
    // 对话超过10条说明是复杂问题，升级模型
    return handler({
      ...request,
      model: messageCount > 10 ? advancedModel : basicModel,
    });
  },
});

const agent = createAgent({
  model: "gpt-4o-mini",
  tools,
  middleware: [dynamicModelSelection],
});
```

这样既能控制成本，又能保证复杂问题的质量。还可以用这个机制做A/B测试。

---

### 工具(Tools) - 行动能力

工具定义了Agent能干什么。一个工具就是一个可以被LLM调用的函数。

#### 定义工具

定义工具需要四个部分：执行函数、名字、说明文档、输入参数规范。

```javascript
import * as z from "zod";
import { tool } from "langchain";

const search = tool(
  ({ query }) => `Results for: ${query}`,
  {
    name: "search",
    description: "Search for information",
    schema: z.object({
      query: z.string().describe("The query to search for"),
    }),
  }
);
```

说明一下各部分的用途：
- **第一个参数（函数）**：实际执行的逻辑
- **name**：LLM根据这个来判断该调用哪个工具，很关键
- **description**：告诉LLM这个工具是干嘛的，什么时候该用
- **schema**：Zod定义的输入参数。注意每个参数都要加`.describe()`，不然LLM不知道这个参数是什么意思

多个工具可以组成一个工具集传给Agent：

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [search, getWeather, sendEmail],
});
```

如果传入空数组`[]`，那Agent就退化成单纯的LLM，没有工具调用能力。

#### 错误处理

工具执行失败默认会直接返回错误, 但是可以让Agent尝试修正输入再试一遍， 可以使用中间件的`wrapToolCall`钩子：

```javascript
const handleToolErrors = createMiddleware({
  name: "HandleToolErrors",
  wrapToolCall: async (request, handler) => {
    try {
      return await handler(request);
    } catch (error) {
      // 不是返回错误并停止，而是返回一条消息让Agent知道出了问题
      return new ToolMessage({
        content: `工具执行失败了。请检查你的输入并重试。错误: ${error}`,
        tool_call_id: request.toolCall.id!,
      });
    }
  },
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [search, /* ... */],
  middleware: [handleToolErrors],
});
```

这样工具失败时不会打断整个流程，而是返回一条error message给模型，让它看到错误信息后自己想办法重试或改变策略。

#### 工具在循环中的作用

Agent执行时，工具可以：
- **串行执行**：一个接一个调用
- **并行执行**：没有依赖关系的工具同时跑
- **动态选择**：根据上一步的结果决定下一步调用什么工具
- **自动重试**：失败后通过中间件自动调整参数再试
- **完整记录**：每次调用和结果都保存在对话历史里，方便调试

---

### 系统提示次(System Prompt)

。有三种方式来设置系统提示次。

#### 直接传字符串

```javascript
const agent = createAgent({
  model,
  tools,
  systemPrompt: "You are a helpful assistant. Be concise and accurate.",
});
```

不提供的话，Agent会从消息内容自动推断它应该扮演什么角色。这适合大多数场景。

#### 需要缓存长文本时：用SystemMessage对象

如果系统提示里需要包含大量文本（比如一本书的内容），用`SystemMessage`对象可以启用某些LLM提供商的缓存功能，减少成本和延迟：

```javascript
import { SystemMessage, HumanMessage } from "@langchain/core/messages";

const literaryAgent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  systemPrompt: new SystemMessage({
    content: [
      {
        type: "text",
        text: "You are analyzing literary works.",
      },
      {
        type: "text",
        text: "<整本傲慢与偏见的内容>",
        cache_control: { type: "ephemeral" }  // Anthropic会缓存这部分
      }
    ]
  })
});
```

这样Anthropic会缓存那个`cache_control`标记的文本块。后续请求就能直接用缓存，快很多，成本也省不少。

#### 根据用户或场景动态调整提示

有时候针对不同用户要用不同的提示。这时用中间件的动态提示：

```javascript
const contextSchema = z.object({
  userRole: z.enum(["expert", "beginner"]),
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [/* ... */],
  contextSchema,
  middleware: [
    dynamicSystemPromptMiddleware((state, runtime) => {
      const userRole = runtime.context.userRole || "user";
      const basePrompt = "You are a helpful assistant.";

      if (userRole === "expert") {
        return `${basePrompt} 给出深度的技术分析。`;
      } else if (userRole === "beginner") {
        return `${basePrompt} 用简单的语言解释，避免术语。`;
      }
      return basePrompt;
    }),
  ],
});

// 调用时传入context
const result = await agent.invoke(
  { messages: [{ role: "user", content: "解释机器学习" }] },
  { context: { userRole: "expert" } }  // 这个值会被动态提示读到
);
```

---

## 运行和调用

### 基本使用

启动Agent最简单的方式就是`invoke`，等待它完成后返回结果：

```javascript
const result = await agent.invoke({
  messages: [{ role: "user", content: "旧金山的天气怎么样？" }],
});
```

Agent会根据需要多次运行LLM和工具，直到得出答案。

### 看中间步骤：流式返回

如果Agent需要多个步骤才能完成，用户得等很久。这时候用`stream`，能实时看到Agent在干什么：

```javascript
const stream = await agent.stream(
  {
    messages: [{
      role: "user",
      content: "搜索AI新闻，然后给我总结"
    }],
  },
  { streamMode: "values" }
);

for await (const chunk of stream) {
  const latestMessage = chunk.messages.at(-1);
  
  if (latestMessage?.content) {
    // Agent说了什么
    console.log(`Agent: ${latestMessage.content}`);
  } else if (latestMessage?.tool_calls) {
    // Agent在调用工具
    const toolNames = latestMessage.tool_calls.map((tc) => tc.name);
    console.log(`正在调用: ${toolNames.join(", ")}`);
  }
}
```

流式返回对UI特别有用——用户能看到进度，而不是对着一个空白的屏幕。如果发现Agent做错了什么，还能中断。

### 自动管理状态

Agent自动维护对话历史。如果你想追踪额外信息（比如用户偏好、任务进度），可以自定义状态：

```javascript
import { StateSchema, MessagesValue } from "@langchain/langgraph";

const CustomAgentState = new StateSchema({
  messages: MessagesValue,  // 对话历史，自动管理
  userPreferences: z.record(z.string(), z.string()),  // 额外信息
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  stateSchema: CustomAgentState,
});
```

简单说，`messages`自动保存，其他你想记的都可以加到state里。

---

## 高级玩法

### 强制输出结构化数据

有时候你要求Agent返回特定格式的数据（比如JSON结构），而不仅仅是文本。这时用`responseFormat`：

```javascript
import * as z from "zod";

const ContactInfo = z.object({
  name: z.string(),
  email: z.string(),
  phone: z.string(),
});

const agent = createAgent({
  model: "gpt-4o",
  responseFormat: ContactInfo,
});

const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "从这里提取联系方式：John Doe, john@example.com, (555) 123-4567",
    },
  ],
});

console.log(result.structuredResponse);
// 得到: { name: 'John Doe', email: 'john@example.com', phone: '(555) 123-4567' }
```

这对数据提取、表单填充、API集成很有用。保证输出符合预期格式。

### 内存：记住上下文

Agent有两种记忆方式。

**短期记忆**：自动保存对话历史
```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  // messages自动保存到state里
});
```

不需要特别配置，Agent会记住之前说过的每句话。

**扩展记忆**：保存额外信息
```javascript
const CustomAgentState = new StateSchema({
  messages: MessagesValue,
  userPreferences: z.record(z.string(), z.string()),
  taskProgress: z.string(),
});

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  stateSchema: CustomAgentState,
});
```

这样你可以存用户偏好、任务进度这些信息，Agent能利用这些上下文做出更好的决策。

**长期记忆**：持久化存储
官方文档里没详细介绍，但通常需要用向量数据库（如Pinecone）存储和检索。这是个高阶用法。

### 中间件：拦截和修改执行流

中间件是Agent最强大的扩展机制。想象成在Agent的执行流里插入检查点，在这些地方拦截、修改或记录数据。

```javascript
const customMiddleware = createMiddleware({
  name: "MyMiddleware",
  beforeModel: async (state, handler) => {
    // LLM调用前处理：比如截断长消息历史
    console.log("About to call LLM");
    return handler(state);
  },
  wrapModelCall: async (request, handler) => {
    // 包装LLM调用：比如动态选模型
    console.log("Calling model:", request.model);
    return handler(request);
  },
  afterModel: async (state, handler) => {
    // LLM调用后处理：比如验证输出
    return handler(state);
  },
  wrapToolCall: async (request, handler) => {
    // 工具调用的包装：比如错误重试
    try {
      return await handler(request);
    } catch (error) {
      console.log("Tool failed:", error);
      throw error;
    }
  },
});
```

常见的中间件用途：
- **动态模型选择**：根据对话复杂度切换模型
- **内容过滤**：把不合适的内容删掉
- **自动重试**：工具失败时自动调整参数重试
- **监控和日志**：记录每步执行情况
- **消息优化**：太长了就截断历史，省tokens

---

## 快速参考

### createAgent()都有哪些参数

| 参数 | 类型 | 必需？ | 说明 |
|------|------|------|------|
| `model` | 字符串或模型实例 | ✅ | "openai:gpt-4o" 或 ChatOpenAI实例 |
| `tools` | 工具数组 | ✅ | 可以传空数组 |
| `systemPrompt` | 字符串或SystemMessage | ❌ | 定制Agent的性格和行为 |
| `responseFormat` | Zod Schema | ❌ | 强制输出格式 |
| `stateSchema` | StateSchema | ❌ | 自定义Agent的记忆结构 |
| `middleware` | 中间件数组 | ❌ | 执行流拦截和修改 |
| `contextSchema` | Zod Schema | ❌ | 验证传入的context参数 |








