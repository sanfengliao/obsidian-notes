# LangChain Messages

## 概述

消息是 LangChain 中最基础的数据单位。每条消息都代表了与 LLM 的一次交互，可以是输入也可以是输出。

消息包含三个部分：
- **Role** - 这条消息是谁说的（system、user、assistant 等）
- **Content** - 消息的内容（可以是文本、图像、音频等）
- **Metadata** - 可选的附加信息（如 token 用量、消息 ID 等）

## 基本用法

### 最简单的：直接传一个字符串

```javascript
const response = await model.invoke("Write a haiku about spring");
```

这适合简单的一次性请求，不需要对话历史的场景。

### 保持对话历史：用消息列表

```javascript
import { SystemMessage, HumanMessage, AIMessage } from "langchain";

const messages = [
  new SystemMessage("You are a poetry expert"),
  new HumanMessage("Write a haiku about spring"),
  new AIMessage("Cherry blossoms bloom..."),
];
const response = await model.invoke(messages);
```

这种方式适合：
- 多轮对话
- 包含图像、音频等多模态内容
- 需要系统级指令

### 字典格式（OpenAI 风格）

```javascript
const messages = [
  { role: "system", content: "You are a poetry expert" },
  { role: "user", content: "Write a haiku about spring" },
  { role: "assistant", content: "Cherry blossoms bloom..." },
];
const response = await model.invoke(messages);
```

## 消息类型

### SystemMessage - 系统指令

定义模型的行为、身份和角色。通常放在对话的最开头。

**简单指令**：
```javascript
const systemMsg = new SystemMessage("You are a helpful coding assistant.");

const messages = [
  systemMsg,
  new HumanMessage("How do I create a REST API?"),
];
```

**详细人设**：
```javascript
const systemMsg = new SystemMessage(`
You are a senior TypeScript developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
`);
```

### HumanMessage - 用户消息

代表用户的输入。可以包含文本、图像、音频、文件等内容。

**简单文本**：
```javascript
const humanMsg = new HumanMessage("What is machine learning?");
```

**带元数据**：
```javascript
const humanMsg = new HumanMessage({
  content: "What is machine learning?",
  name: "alice",        // 用户标识（不同 provider 支持不同）
  id: "msg_123",       // 消息 ID
});
```

**字符串快捷方式**：
```javascript
const response = await model.invoke("What is machine learning?");
```

### AIMessage - 模型的回复

这是模型生成的响应。除了文本，还包含工具调用、推理过程等结构化信息。

**从模型获得**：
```javascript
const response = await model.invoke("Explain AI");
console.log(typeof response);  // AIMessage
```

**手动创建**（比如要插入到历史记录中）：
```javascript
const aiMsg = new AIMessage("I'd be happy to help!");

const messages = [
  new SystemMessage("You are helpful"),
  new HumanMessage("Can you help?"),
  aiMsg,  // 模拟来自模型的响应
  new HumanMessage("What's 2+2?")
];
```

### ToolMessage - 工具执行结果

当模型调用工具后，工具的执行结果会包装在 ToolMessage 中，然后传回给模型让它继续思考。

```javascript
import { AIMessage, ToolMessage } from "langchain";

const aiMessage = new AIMessage({
  content: [],
  tool_calls: [{
    name: "get_weather",
    args: { location: "San Francisco" },
    id: "call_123"
  }]
});

const toolMessage = new ToolMessage({
  content: "Sunny, 72°F",
  tool_call_id: "call_123"  // 必须与上面的 id 匹配
});

const messages = [
  new HumanMessage("What's the weather in San Francisco?"),
  aiMessage,      // 模型要求调用工具
  toolMessage,    // 工具返回的结果
];

const response = await model.invoke(messages);  // 模型看到结果，继续推理
```

**artifact 字段**：用来存储一些补充数据，这些数据不会发送给模型，但可以在代码中访问。比如存原始的 API 响应或调试信息。

## AIMessage 中的数据

### 工具调用信息

```javascript
const modelWithTools = model.bindTools([getWeather]);
const response = await modelWithTools.invoke("What's the weather in Paris?");

for (const toolCall of response.tool_calls) {
  console.log(`工具: ${toolCall.name}`);
  console.log(`参数: ${toolCall.args}`);
  console.log(`ID: ${toolCall.id}`);
}
```

### Token 用量信息

```javascript
const response = await model.invoke("Hello!");
console.log(response.usage_metadata);

// 输出样例:
// {
//   "output_tokens": 304,
//   "input_tokens": 8,
//   "total_tokens": 312,
//   "input_token_details": { "cache_read": 0 },
//   "output_token_details": { "reasoning": 256 }
// }
```

### 流式消息的拼接

```javascript
import { AIMessageChunk } from "langchain";

let finalChunk = undefined;
for (const chunk of chunks) {
  finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk;
}
```

流式响应时会收到多个 `AIMessageChunk`，每个包含部分输出。用 `concat()` 可以把它们合并成完整的消息。

## 消息内容格式

消息的 content 部分支持三种写法：

1. **简单字符串** - 快速简便
2. **Provider 原生格式** - 直接用 OpenAI 或 Anthropic 的格式
3. **LangChain 标准格式** - 统一的跨 provider 格式

```javascript
import { HumanMessage } from "langchain";

// 简单字符串
const msg1 = new HumanMessage("Hello");

// Provider 原生格式
const msg2 = new HumanMessage({
  content: [
    { type: "text", text: "Hello, how are you?" },
    { type: "image_url", image_url: { url: "https://example.com/image.jpg" } },
  ],
});

// LangChain 标准格式
const msg3 = new HumanMessage({
  contentBlocks: [
    { type: "text", text: "Hello, how are you?" },
    { type: "image", url: "https://example.com/image.jpg" },
  ],
});
```

### 内容块（Content Blocks）

每条消息都有 `contentBlocks` 属性，它是一个标准化的内容表示，方便统一处理。

支持的内容块类型：

**基础类型**：
- `text` - 文本内容
- `reasoning` - 模型的推理过程
- `toolUse` - 模型请求调用工具
- `toolResult` - 工具的执行结果

**多模态**：
- `image` - 图像
- `audio` - 音频
- `video` - 视频
- `pdf` - PDF 文档

**服务端工具**（特殊）：
- `serverToolCall` - 由服务端完成的工具调用
- `serverToolResult` - 服务端工具的结果

## 多模态内容

### 图像

**从 URL**：
```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "Describe this image" },
    { type: "image", source_type: "url", url: "https://example.com/image.jpg" },
  ],
});
```

**从 Base64 数据**：
```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "Describe this image" },
    { type: "image", source_type: "base64", data: "AAAAIGZ0eXBtcDQyAAAA..." },
  ],
});
```

**从 File ID**（由 provider 管理）：
```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "Describe this image" },
    { type: "image", source_type: "id", id: "file-abc123" },
  ],
});
```

### 音频

```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "Transcribe this audio" },
    { type: "audio", source_type: "url", url: "https://example.com/audio.mp3" },
  ],
});
```

### 视频

```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "What happens in this video?" },
    { type: "video", source_type: "url", url: "https://example.com/video.mp4" },
  ],
});
```

### PDF

```javascript
const message = new HumanMessage({
  content: [
    { type: "text", text: "Summarize this PDF" },
    { type: "document", source_type: "url", url: "https://example.com/doc.pdf" },
  ],
});
```

**重要**：不是所有模型都支持所有类型的文件。查看你使用的 provider 的文档，了解支持的格式和文件大小限制。

### 多模态输出

某些模型可以生成图像等多模态内容：

```javascript
const response = await model.invoke("Create a picture of a cat");
console.log(response.contentBlocks);

// [
//   { type: "text", text: "Here's a picture of a cat" },
//   { type: "image", data: "...", mimeType: "image/jpeg" },
// ]
```

## 标准内容块的类型定义

### 类型定义

```javascript
import { ContentBlock } from "langchain";

// 文本块
const textBlock: ContentBlock.Text = {
  type: "text",
  text: "Hello world",
};

// 图像块
const imageBlock: ContentBlock.Multimodal.Image = {
  type: "image",
  url: "https://example.com/image.png",
  mimeType: "image/png",
};
```

完整的类型定义和更多选项见 [API 文档](https://reference.langchain.com/javascript/modules/_langchain_core.messages.html)。

## 高级特性

### 跨 Provider 的标准化

不同的 provider 有不同的内容格式（Anthropic 的 `thinking` vs OpenAI 的 `reasoning`）。LangChain 把它们都标准化成统一的 `contentBlocks` 表示。

比如，Anthropic 的推理内容和 OpenAI 的推理内容都会被解析为同一个 ReasoningContentBlock：

```javascript
const message = new AIMessage({
  content: [
    { type: "thinking", thinking: "..." },
    { type: "text", text: "..." },
  ],
  response_metadata: { model_provider: "anthropic" },
});

console.log(message.contentBlocks);  // 统一格式
```

### 启用标准内容序列化

当外部应用需要标准化格式时，可以通过环境变量或模型参数启用：

```javascript
// 方法 1：环境变量
process.env.LC_OUTPUT_VERSION = "v1";

// 方法 2：模型初始化时配置
const model = await initChatModel(
  "gpt-5-nano",
  { outputVersion: "v1" }
);
```

## 常见的对话模式

### 多轮对话

```javascript
const messages = [];

while (true) {
  messages.push(new HumanMessage("User input"));
  const response = await model.invoke(messages);
  messages.push(response);  // 保存模型的响应
}
```

### 工具调用循环

```javascript
const messages = [];

while (true) {
  messages.push(new HumanMessage("User input"));
  const response = await modelWithTools.invoke(messages);
  messages.push(response);

  if (!response.tool_calls || response.tool_calls.length === 0) {
    break;  // 模型没有调用工具，完成
  }

  // 执行模型请求的工具
  for (const toolCall of response.tool_calls) {
    const result = await executeTool(toolCall);
    messages.push(new ToolMessage({
      content: result,
      tool_call_id: toolCall.id,
    }));
  }
}
```

### 管理上下文窗口

当消息太多会消耗大量 token。此时需要截断或汇总历史记录来节省成本。详见 [short-term-memory](https://docs.langchain.com/oss/javascript/langchain/short-term-memory) 指南。

## 常见问题

**Q：什么时候用字符串，什么时候用消息列表？**

A：简单的一次性请求用字符串。多轮对话、需要多模态内容或系统指令时用消息列表。

**Q：name 字段有什么用？**

A：用来标识用户。不同 provider 支持不同——有的用于用户识别，有的直接忽略。查看 provider 的文档。

**Q：怎样访问标准化的内容块？**

A：用 `message.contentBlocks`。它会自动把 provider 的原生格式转换为标准格式。

**Q：多模态内容怎样传给模型？**

A：在 HumanMessage 的 content 中加入相应的内容块。不同 provider 支持不同格式，查看文档了解。

**Q：artifact 字段的用途？**

A：存储不需要发送给模型、但代码需要访问的补充数据。比如原始 API 响应或调试信息，可以存在这里避免污染模型的上下文。
