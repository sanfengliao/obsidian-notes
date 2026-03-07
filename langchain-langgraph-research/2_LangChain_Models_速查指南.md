# LangChain Models

## 概述

模型是Agent的"大脑"。LangChain支持OpenAI、Anthropic、Google、Azure、AWS等主流LLM厂商。选什么模型直接影响Agent的可靠性和性能。

## 快速开始

### 初始化模型

最简单的方式是用`initChatModel`，会自动处理API密钥和配置：

```javascript
import { initChatModel } from "langchain";

process.env.OPENAI_API_KEY = "your-api-key";
const model = await initChatModel("gpt-4o");
```

格式是`provider:model`，比如：
- `"openai:gpt-4o"`
- `"anthropic:claude-sonnet-4-5"`
- `"google:gemini-2.0-flash"`

查看[integrations page](https://docs.langchain.com/oss/javascript/integrations/providers/overview)了解所有支持的模型。

### 调用模型

最直接的方法：

```javascript
const response = await model.invoke("Why do parrots talk?");
console.log(response);  // AIMessage对象
```

## 配置参数

大多数情况下使用默认参数就够了，但需要调优时这些参数很有用：

| 参数 | 说明 | 常见值 |
|------|------|------|
| `temperature` | 控制输出的随意程度：高=更创意；低=更稳定 | 0.7 |
| `maxTokens` | 限制输出长度 | 1000 |
| `timeout` | 请求超时时间（秒） | 30 |
| `maxRetries` | 失败自动重试的次数 | 3 |
| `apiKey` | API密钥 | (用环境变量更安全) |

### 怎样传递这些参数

```javascript
const model = await initChatModel(
  "claude-sonnet-4-5-20250929",
  { temperature: 0.7, timeout: 30, max_tokens: 1000 }
);
```

或者直接创建模型实例：

```javascript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0.1,
  maxTokens: 1000
});
```

每个模型可能有自己的特殊参数。比如 ChatOpenAI 支持 `use_responses_api`，详细列表可以看官方文档的 integration page。

## 怎样调用模型

### 最基础：invoke

等到模型完全生成，再返回结果。大多数情况用这个就够了：

```javascript
const response = await model.invoke("Hello!");
console.log(response);  // AIMessage
```

可以传单个消息，也可以传消息列表来保持对话历史。

**带上历史**：

```javascript
const conversation = [
  { role: "system", content: "You are a translator." },
  { role: "user", content: "Translate: Hello" },
  { role: "assistant", content: "Bonjour" },
  { role: "user", content: "Translate: Goodbye" },
];

const response = await model.invoke(conversation);
```

或者用消息类(更结构化一些)：

```javascript
import { SystemMessage, HumanMessage, AIMessage } from "langchain";

const conversation = [
  new SystemMessage("You are a translator."),
  new HumanMessage("Translate: Hello"),
  new AIMessage("Bonjour"),
  new HumanMessage("Translate: Goodbye"),
];

const response = await model.invoke(conversation);
```

### stream - 实时看到生成过程

适合长回复或需要即时反馈的场景：

```javascript
const stream = await model.stream("Why do parrots talk?");
for await (const chunk of stream) {
  console.log(chunk.text);  // 一段一段地打印
}
```

每个 chunk 是一个 `AIMessageChunk` 对象，包含部分输出。如果需要完整的最终结果，可以把 chunk 累积起来：

```javascript
let full = null;
for await (const chunk of stream) {
  full = full ? full.concat(chunk) : chunk;
  console.log(full.text);  // 看到持续增长的输出
}

console.log(full.contentBlocks);  // 最终完整的内容块
```

### batch - 并行处理多个请求

有多个独立的请求时，可以用 batch 并行处理，提高吞吐量：

```javascript
const responses = await model.batch([
  "Why do parrots have colorful feathers?",
  "How do airplanes fly?",
  "What is quantum computing?",
]);

for (const response of responses) {
  console.log(response);
}
```

同时控制并发数，避免请求过多导致问题：

```javascript
await model.batch(
  listOfInputs,
  {
    maxConcurrency: 5,  // 最多同时 5 个请求
  }
);
```

## 让模型调用工具

模型可以识别需要调用工具的场景（比如搜索、查数据库等），然后告诉你"我需要调用这个工具"。

### 第一步：绑定工具到模型

```javascript
import { tool } from "langchain";
import * as z from "zod";

const getWeather = tool(
  (input) => `It's sunny in ${input.location}.`,
  {
    name: "get_weather",
    description: "Get the weather at a location.",
    schema: z.object({
      location: z.string(),
    }),
  }
);

const modelWithTools = model.bindTools([getWeather]);
const response = await modelWithTools.invoke("What's the weather in Boston?");
```

### 第二步：处理模型的工具调用请求

模型会在响应中包含一个工具调用请求，你需要执行这个工具并把结果返回给它：

```javascript
// 绑定工具到模型
const modelWithTools = model.bindTools([getWeather])

// 第 1 步：模型分析用户请求并生成工具调用
const messages = [{"role": "user", "content": "What's the weather in Boston?"}]
const ai_msg = await modelWithTools.invoke(messages)
messages.push(ai_msg)

// 第 2 步：执行模型要求的工具
for (const tool_call of ai_msg.tool_calls) {
    const tool_result = await getWeather.invoke(tool_call)
    messages.push(tool_result)
}

// 第 3 步：把工具结果反馈给模型，让它生成最终答案
const final_response = await modelWithTools.invoke(messages)
```

**提示**：直接用模型调用时需要手动执行这个循环。如果用 Agent，它会自动完成整个流程。

## 让模型输出结构化数据

有时候需要模型返回特定格式的数据（而不是自由文本），比如提取信息、生成 JSON 等：

```javascript
import * as z from "zod";

const Movie = z.object({
  title: z.string(),
  year: z.number(),
  director: z.string(),
  rating: z.number(),
});

const modelWithStructure = model.withStructuredOutput(Movie);
const response = await modelWithStructure.invoke(
  "Provide details about Inception"
);

console.log(response);
// { title: "Inception", year: 2010, director: "Christopher Nolan", rating: 8.8 }
```

### 高级选项

```javascript
const modelWithStructure = model.withStructuredOutput(Movie, {
  method: "jsonSchema",  // 或 "functionCalling", "jsonMode"
  includeRaw: true,      // 同时返回原始的 AIMessage
});
```

- `method`：不同模型支持不同的方法，可以选择最兼容的
- `includeRaw: true`：除了解析后的数据，还返回原始 AIMessage

## 好用的小功能

### 查询模型本身支持什么

某些模型可以通过 `.profile` 暴露自身能力信息：

```javascript
model.profile;
// {
//   maxInputTokens: 400000,
//   imageInputs: true,
//   reasoningOutput: true,
//   toolCalling: true,
//   ...
// }
```

这样可以：
- 根据模型实际能力动态调整代码
- 消息总结可根据上下文窗口大小自动决定何时触发
- 结构化输出策略根据原生支持情况自动选择最优方案

### 处理图像、音频等多模态数据

```javascript
const response = await model.invoke("Create a picture of a cat");
console.log(response.contentBlocks);
// [
//   { type: "text", text: "Here's a picture..." },
//   { type: "image", data: "...", mimeType: "image/jpeg" },
// ]
```

支持的格式包括：
- LangChain 的标准格式（详见 messages 指南）
- OpenAI 聊天完成格式
- 各个 provider 的原生格式（比如 Anthropic）

### 多步推理

某些高级模型支持花更多时间进行多步推理：

```javascript
const stream = await model.stream("Why do parrots have colorful feathers?");
for await (const chunk of stream) {
  const reasoning = chunk.contentBlocks.filter(b => b.type === "reasoning");
  console.log(reasoning.length > 0 ? reasoning : chunk.text);
}
```

可以根据模型支持调整推理的深度或 token 预算。

### 用 Ollama 在本地运行模型

如果想在本地跑模型，参考 Ollama integration 文档。

### 提示词缓存

**隐式缓存**（自动启用）：OpenAI、Gemini  
**显式缓存**（需要配置）：
- ChatOpenAI（通过 `prompt_cache_key`）
- Anthropic（AnthropicPromptCachingMiddleware）
- AWS Bedrock

一般来说，当输入 token 超过某个阈值时缓存才会生效。具体限制条件看各个 provider 的文档。

### 服务端工具调用

有些模型可以在服务端完成整个工具调用循环（比如 Web 搜索），无需你手动处理：

```javascript
import { initChatModel } from "langchain";

const model = await initChatModel("gpt-4-mini");
const modelWithTools = model.bindTools([{ type: "web_search" }]);

const message = await modelWithTools.invoke(
  "What was positive news from today?"
);
console.log(message.contentBlocks);
```

这种情况下是一轮对话完成，不需要 ToolMessage 的往返。

### Token 级别的概率信息

```javascript
const model = new ChatOpenAI({
  model: "gpt-4o",
  logprobs: true,
});

const response = await model.invoke("Why do parrots talk?");
console.log(response.response_metadata.logprobs.content.slice(0, 5));
```

### 查看 Token 用量

大多数模型会返回用量统计：

```javascript
const response = await model.invoke("Hello");
// response.usage_metadata 中包含 input_tokens、output_tokens 等
```

某些 provider（OpenAI、Azure）在流式时需要特别配置才能获得统计。见 integration 文档。

### 自定义 API 端点或代理

可以配置自定义的基础 URL（比如用代理）：

```javascript
const model = new ChatOpenAI({
  model: "gpt-4o",
  configuration: {
    baseURL: "https://models.github.ai/inference",
  },
});
```

这对于 OpenAI 兼容 API 或代理场景很有用。

## 消息与角色

模型支持这些消息角色：
- `system` - 系统指令
- `user` - 用户的消息
- `assistant` - 模型的回复

也可以用简单对象格式：`{ role: "user", content: "..." }`

更多细节见 [messages 指南](https://docs.langchain.com/oss/javascript/langchain/messages)。

## 怎样选模型

- **简单任务**：用迷你或小型模型，可以节省成本
- **需要工具调用**：确认模型支持 tool calling 功能
- **需要结构化输出**：选择支持你需要的输出方法的模型
- **处理长文本**：注意 `maxInputTokens` 和 `maxOutputTokens` 的限制
- **多模态需求**：检查是否支持 `imageInputs`、`audioInputs` 等
- **复杂推理**：用支持推理功能的高端模型

## 常见问题

**Q：在流式时怎样获得 token 用量？**

A：OpenAI 和 Azure 需要特别配置。见它们的 integration 文档。

**Q：能用自定义的模型吗？**

A：可以。如果你的模型兼容某个 provider 的 API，就可以通过设置 `baseURL` 使用。

**Q：多模态输入怎么传给模型？**

A：用 content blocks 格式。详见 messages 指南的多模态部分。

**Q：temperature 应该设置多少？**

A：创意任务设 0.7-1.0，需要准确回答的任务设 0-0.3，不确定时用 0.5。
