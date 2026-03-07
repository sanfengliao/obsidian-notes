# Claude Agent SDK 完整指南

本文档详细介绍 Claude Agent SDK 的功能和使用方法，所有代码示例使用 TypeScript。

## 目录

1. [快速入门](#快速入门)
   - [安装](#安装)
   - [设置 API 密钥](#设置-api-密钥)
   - [客户端配置选项](#客户端配置选项)
2. [流式输入模式](#流式输入模式)
3. [流式输出](#流式输出)
4. [处理停止原因](#处理停止原因)
5. [权限配置](#权限配置)
6. [用户输入和审批处理](#用户输入和审批处理)
7. [Hooks 机制](#hooks-机制)
8. [会话管理](#会话管理)
9. [文件检查点](#文件检查点)
10. [结构化输出](#结构化输出)
11. [MCP 连接外部工具](#mcp-连接外部工具)
12. [自定义工具](#自定义工具)
13. [子 Agents](#子-agents)
14. [斜杠命令](#斜杠命令)
15. [Agent 技能](#agent-技能)
16. [插件](#插件)

---

## 快速入门

### 安装

```bash
npm install @anthropic-ai/claude-agent-sdk
```

### 设置 API 密钥

在项目目录中创建 `.env` 文件：

```bash
ANTHROPIC_API_KEY=your-api-key
```

SDK 还支持通过第三方 API 提供商进行身份验证：
- **Amazon Bedrock**：设置 `CLAUDE_CODE_USE_BEDROCK=1` 环境变量
- **Google Vertex AI**：设置 `CLAUDE_CODE_USE_VERTEX=1` 环境变量
- **Microsoft Azure**：设置 `CLAUDE_CODE_USE_FOUNDRY=1` 环境变量

### 客户端配置选项

Claude Agent SDK 通过 `options` 参数和环境变量进行配置，支持自定义 API 密钥、基础 URL 等参数。

#### API Key 配置

SDK 支持多种 API Key 配置方式：

**方式一：环境变量（推荐）**

```bash
# 在 .env 文件或 shell 中设置
ANTHROPIC_API_KEY=your-api-key
# 或使用认证令牌（自动添加 Bearer 前缀）
ANTHROPIC_AUTH_TOKEN=your-auth-token
```

**方式二：通过 `options.env` 传递**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Hello, Claude!",
  options: {
    // 通过 env 字段传递环境变量
    env: {
      ANTHROPIC_API_KEY: "your-api-key"
    }
  }
})) {
  // 处理消息
}
```

**方式三：使用 apiKeyHelper 脚本**

在 `~/.claude/settings.json` 或项目的 `.claude/settings.json` 中配置：

```json
{
  "apiKeyHelper": "/path/to/generate_api_key.sh"
}
```

脚本输出的值将作为 `X-Api-Key` 和 `Authorization: Bearer` 请求头发送。

#### 自定义 Base URL

通过环境变量或 `options.env` 配置自定义 API 端点：

**方式一：环境变量**

```bash
# 设置自定义 API 基础地址
ANTHROPIC_BASE_URL=https://your-proxy.example.com/v1
```

**方式二：通过 `options.env` 传递**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Hello, Claude!",
  options: {
    env: {
      ANTHROPIC_API_KEY: "your-api-key",
      ANTHROPIC_BASE_URL: "https://your-proxy.example.com/v1"
    }
  }
})) {
  // 处理消息
}
```

#### 第三方云平台配置

SDK 支持通过第三方云服务提供商访问 Claude API：

| 平台 | 环境变量 | 额外配置 |
|------|---------|---------|
| **Amazon Bedrock** | `CLAUDE_CODE_USE_BEDROCK=1` | 配置 AWS 凭证 |
| **Google Vertex AI** | `CLAUDE_CODE_USE_VERTEX=1` | 配置 Google Cloud 凭证 |
| **Microsoft Azure** | `CLAUDE_CODE_USE_FOUNDRY=1` | 配置 `ANTHROPIC_FOUNDRY_BASE_URL` 和 `ANTHROPIC_FOUNDRY_RESOURCE` |

**Azure AI Foundry 示例：**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Hello, Claude!",
  options: {
    env: {
      CLAUDE_CODE_USE_FOUNDRY: "1",
      ANTHROPIC_FOUNDRY_BASE_URL: "https://my-resource.services.ai.azure.com/anthropic",
      ANTHROPIC_FOUNDRY_RESOURCE: "my-resource"
    }
  }
})) {
  // 处理消息
}
```

#### 代理配置

如需通过代理访问 API，可配置标准代理环境变量：

```bash
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1
```

#### 完整配置选项

`options` 对象支持以下配置项：

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "分析这个项目的架构",
  options: {
    // 模型配置
    model: "claude-sonnet-4-5",                  // 指定模型
    fallbackModel: "claude-3-5-sonnet",          // 备用模型

    // 环境变量
    env: {
      ANTHROPIC_API_KEY: process.env["ANTHROPIC_API_KEY"],
      ANTHROPIC_BASE_URL: process.env["ANTHROPIC_BASE_URL"]
    },

    // 工具和权限
    allowedTools: ["Read", "Edit", "Bash", "Glob"],  // 允许的工具
    permissionMode: "acceptEdits",                    // 权限模式

    // 对话控制
    maxTurns: 10,                                     // 最大轮数
    systemPrompt: "你是一个代码分析专家",              // 系统提示词

    // 工作目录
    cwd: process.cwd(),                               // 工作目录

    // MCP 服务器
    mcpServers: {                                     // MCP 服务器配置
      "my-server": {
        type: "stdio",
        command: "node",
        args: ["server.js"]
      }
    },

    // 设置来源
    settingSources: ["user", "project", "local"],     // 加载设置的来源

    // CLI 配置
    cliPath: "/path/to/claude",                       // 自定义 CLI 路径
    extraArgs: {                                      // 额外 CLI 参数
      "some-flag": "value"
    }
  }
})) {
  // 处理消息
}
```

#### 环境变量汇总

| 环境变量 | 说明 |
|---------|------|
| `ANTHROPIC_API_KEY` | API 密钥（作为 `X-Api-Key` 请求头） |
| `ANTHROPIC_AUTH_TOKEN` | 认证令牌（作为 `Authorization: Bearer` 请求头） |
| `ANTHROPIC_BASE_URL` | API 基础地址 |
| `ANTHROPIC_CUSTOM_HEADERS` | 自定义请求头（格式：`Name: Value`） |
| `CLAUDE_CODE_USE_BEDROCK` | 使用 Amazon Bedrock（设为 `1` 启用） |
| `CLAUDE_CODE_USE_VERTEX` | 使用 Google Vertex AI（设为 `1` 启用） |
| `CLAUDE_CODE_USE_FOUNDRY` | 使用 Microsoft Azure（设为 `1` 启用） |
| `ANTHROPIC_FOUNDRY_BASE_URL` | Azure Foundry 资源的基础 URL |
| `ANTHROPIC_FOUNDRY_RESOURCE` | Azure Foundry 资源名称 |
| `HTTP_PROXY` / `HTTPS_PROXY` | 代理服务器地址 |

### 基本用法

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// Agent 循环：在 Claude 工作时流式传输消息
for await (const message of query({
  prompt: "Review utils.py for bugs that would cause crashes. Fix any issues you find.",
  options: {
    allowedTools: ["Read", "Edit", "Glob"],  // Claude 可以使用的工具
    permissionMode: "acceptEdits"            // 自动批准文件编辑
  }
})) {
  // 打印人类可读的输出
  if (message.type === "assistant" && message.message?.content) {
    for (const block of message.message.content) {
      if ("text" in block) {
        console.log(block.text);             // Claude 的推理
      } else if ("name" in block) {
        console.log(`Tool: ${block.name}`);  // 正在调用的工具
      }
    }
  } else if (message.type === "result") {
    console.log(`Done: ${message.subtype}`); // 最终结果
  }
}
```

### 核心概念

**工具** 控制 agent 可以做什么：

| 工具 | Agent 可以做什么 |
| :--- | :--- |
| `Read`、`Glob`、`Grep` | 只读分析 |
| `Read`、`Edit`、`Glob` | 分析和修改代码 |
| `Read`、`Edit`、`Bash`、`Glob`、`Grep` | 完全自动化 |

**权限模式** 控制需要多少人工监督：

| 模式 | 行为 | 使用场景 |
| :--- | :--- | :--- |
| `acceptEdits` | 自动批准文件编辑 | 受信任的开发工作流 |
| `bypassPermissions` | 无需提示即可运行 | CI/CD 管道、自动化 |
| `default` | 需要 `canUseTool` 回调来处理审批 | 自定义审批流程 |

---

## 流式输入模式

Claude Agent SDK 支持两种输入模式：

### 流式输入模式（推荐）

提供对 agent 功能的完全访问，支持图像上传、消息队列、工具集成和实时反馈。

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import { readFileSync } from "fs";

async function* generateMessages() {
  // 第一条消息
  yield {
    type: "user" as const,
    message: {
      role: "user" as const,
      content: "Analyze this codebase for security issues"
    }
  };
  
  // 等待条件或用户输入
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // 带图像的后续消息
  yield {
    type: "user" as const,
    message: {
      role: "user" as const,
      content: [
        {
          type: "text",
          text: "Review this architecture diagram"
        },
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: readFileSync("diagram.png", "base64")
          }
        }
      ]
    }
  };
}

// 处理流式响应
for await (const message of query({
  prompt: generateMessages(),
  options: {
    maxTurns: 10,
    allowedTools: ["Read", "Grep"]
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}
```

### 单消息输入

适用于一次性查询、无状态环境（如 Lambda 函数）。

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 简单的一次性查询
for await (const message of query({
  prompt: "Explain the authentication flow",
  options: {
    maxTurns: 1,
    allowedTools: ["Read", "Grep"]
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}

// 使用会话管理继续对话
for await (const message of query({
  prompt: "Now explain the authorization process",
  options: {
    continue: true,
    maxTurns: 1
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}
```

**限制**：单消息输入模式不支持图像附件、动态消息队列、实时中断、钩子集成。

---

## 流式输出

启用 `includePartialMessages` 可在文本和工具调用流式传入时接收增量更新。

### 启用流式输出

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "List the files in my project",
  options: {
    includePartialMessages: true,
    allowedTools: ["Bash", "Read"],
  }
})) {
  if (message.type === "stream_event") {
    const event = message.event;
    if (event.type === "content_block_delta") {
      if (event.delta.type === "text_delta") {
        process.stdout.write(event.delta.text);
      }
    }
  }
}
```

### 流式传输工具调用

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 跟踪当前工具并累积其输入 JSON
let currentTool: string | null = null;
let toolInput = "";

for await (const message of query({
  prompt: "Read the README.md file",
  options: {
    includePartialMessages: true,
    allowedTools: ["Read", "Bash"],
  }
})) {
  if (message.type === "stream_event") {
    const event = message.event;

    if (event.type === "content_block_start") {
      if (event.content_block.type === "tool_use") {
        currentTool = event.content_block.name;
        toolInput = "";
        console.log(`Starting tool: ${currentTool}`);
      }
    } else if (event.type === "content_block_delta") {
      if (event.delta.type === "input_json_delta") {
        const chunk = event.delta.partial_json;
        toolInput += chunk;
        console.log(`  Input chunk: ${chunk}`);
      }
    } else if (event.type === "content_block_stop") {
      if (currentTool) {
        console.log(`Tool ${currentTool} called with: ${toolInput}`);
        currentTool = null;
      }
    }
  }
}
```

### 构建流式 UI

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 跟踪当前是否在工具调用中
let inTool = false;

for await (const message of query({
  prompt: "Find all TODO comments in the codebase",
  options: {
    includePartialMessages: true,
    allowedTools: ["Read", "Bash", "Grep"],
  }
})) {
  if (message.type === "stream_event") {
    const event = message.event;

    if (event.type === "content_block_start") {
      if (event.content_block.type === "tool_use") {
        process.stdout.write(`\n[Using ${event.content_block.name}...]`);
        inTool = true;
      }
    } else if (event.type === "content_block_delta") {
      if (event.delta.type === "text_delta" && !inTool) {
        process.stdout.write(event.delta.text);
      }
    } else if (event.type === "content_block_stop") {
      if (inTool) {
        console.log(" done");
        inTool = false;
      }
    }
  } else if (message.type === "result") {
    console.log("\n\n--- Complete ---");
  }
}
```

---

## 处理停止原因

`stop_reason` 字段告诉您模型停止生成的原因。

### 读取 stop_reason

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Write a poem about the ocean",
})) {
  if (message.type === "result") {
    console.log("Stop reason:", message.stop_reason);
    if (message.stop_reason === "refusal") {
      console.log("The model declined this request.");
    }
  }
}
```

### 可用的停止原因

| 停止原因 | 含义 |
| :--- | :--- |
| `end_turn` | 模型正常完成了响应的生成 |
| `max_tokens` | 响应达到了最大输出 token 限制 |
| `stop_sequence` | 模型生成了配置的停止序列 |
| `refusal` | 模型拒绝执行该请求 |
| `tool_use` | 模型的最终输出是工具调用 |
| `null` | 未收到 API 响应 |

### 检测拒绝

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

async function safeQuery(prompt: string): Promise<string | null> {
  for await (const message of query({ prompt })) {
    if (message.type === "result") {
      if (message.stop_reason === "refusal") {
        console.log("Request was declined. Please revise your prompt.");
        return null;
      }
      if (message.subtype === "success") {
        return message.result;
      }
      return null;
    }
  }
  return null;
}
```

---

## 权限配置

### 权限评估流程

当 Claude 请求使用工具时，SDK 按以下顺序检查权限：

1. **Hooks**：首先运行 hooks，可以允许、拒绝或继续到下一步
2. **权限规则**：检查 `settings.json` 中定义的规则
3. **权限模式**：应用当前激活的权限模式
4. **canUseTool 回调**：如果规则或模式未能解决，则调用回调

### 可用权限模式

| 模式 | 描述 | 工具行为 |
| :--- | :--- | :--- |
| `default` | 标准权限行为 | 无自动批准；未匹配的工具会触发 `canUseTool` 回调 |
| `acceptEdits` | 自动接受文件编辑 | 文件编辑和文件系统操作会自动批准 |
| `bypassPermissions` | 绕过所有权限检查 | 所有工具无需权限提示即可运行 |
| `plan` | 规划模式 | 不执行工具；Claude 只进行规划 |

### 设置权限模式

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  for await (const message of query({
    prompt: "Help me refactor this code",
    options: {
      permissionMode: "default",
    },
  })) {
    if ("result" in message) {
      console.log(message.result);
    }
  }
}

main();
```

### 动态更改权限模式

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const q = query({
    prompt: "Help me refactor this code",
    options: {
      permissionMode: "default",
    },
  });

  // 在会话中途动态更改模式
  await q.setPermissionMode("acceptEdits");

  for await (const message of q) {
    if ("result" in message) {
      console.log(message.result);
    }
  }
}

main();
```

---

## 用户输入和审批处理

### 处理工具审批请求

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import * as readline from "readline";

// 帮助提示用户输入
function prompt(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) =>
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer);
    })
  );
}

for await (const message of query({
  prompt: "Create a test file in /tmp and then delete it",
  options: {
    canUseTool: async (toolName, input) => {
      // 显示工具请求
      console.log(`\nTool: ${toolName}`);
      if (toolName === "Bash") {
        console.log(`Command: ${input.command}`);
        if (input.description) console.log(`Description: ${input.description}`);
      } else {
        console.log(`Input: ${JSON.stringify(input, null, 2)}`);
      }

      // 获取用户批准
      const response = await prompt("Allow this action? (y/n): ");

      // 根据用户响应返回允许或拒绝
      if (response.toLowerCase() === "y") {
        return { behavior: "allow", updatedInput: input };
      } else {
        return { behavior: "deny", message: "User denied this action" };
      }
    },
  },
})) {
  if ("result" in message) console.log(message.result);
}
```

### 处理澄清问题

Claude 可以通过 `AskUserQuestion` 工具提出澄清问题。

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";
import * as readline from "readline";

function prompt(question: string): Promise<string> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => rl.question(question, (answer) => { rl.close(); resolve(answer); }));
}

// 解析用户输入
function parseResponse(response: string, options: any[]): string {
  const indices = response.split(",").map((s) => parseInt(s.trim()) - 1);
  const labels = indices
    .filter((i) => !isNaN(i) && i >= 0 && i < options.length)
    .map((i) => options[i].label);
  return labels.length > 0 ? labels.join(", ") : response;
}

// 显示 Claude 的问题并收集用户答案
async function handleAskUserQuestion(input: any) {
  const answers: Record<string, string> = {};

  for (const q of input.questions) {
    console.log(`\n${q.header}: ${q.question}`);

    const options = q.options;
    options.forEach((opt: any, i: number) => {
      console.log(`  ${i + 1}. ${opt.label} - ${opt.description}`);
    });
    if (q.multiSelect) {
      console.log("  (Enter numbers separated by commas, or type your own answer)");
    } else {
      console.log("  (Enter a number, or type your own answer)");
    }

    const response = (await prompt("Your choice: ")).trim();
    answers[q.question] = parseResponse(response, options);
  }

  return {
    behavior: "allow",
    updatedInput: { questions: input.questions, answers },
  };
}

async function main() {
  for await (const message of query({
    prompt: "Help me decide on the tech stack for a new mobile app",
    options: {
      canUseTool: async (toolName, input) => {
        if (toolName === "AskUserQuestion") {
          return handleAskUserQuestion(input);
        }
        return { behavior: "allow", updatedInput: input };
      },
    },
  })) {
    if ("result" in message) console.log(message.result);
  }
}

main();
```

---

## Hooks 机制

Hooks 让您可以在关键点拦截 agent 执行，以添加验证、日志记录、安全控制或自定义逻辑。

### 可用 Hooks

| Hook 事件 | 触发条件 | 示例用例 |
| :--- | :--- | :--- |
| `PreToolUse` | 工具调用请求 | 阻止危险的 shell 命令 |
| `PostToolUse` | 工具执行结果 | 将所有文件更改记录到审计跟踪 |
| `PostToolUseFailure` | 工具执行失败 | 处理或记录工具错误 |
| `UserPromptSubmit` | 用户提示提交 | 向提示中注入额外上下文 |
| `Stop` | Agent 执行停止 | 退出前保存会话状态 |
| `SubagentStart` | 子 Agent 初始化 | 跟踪并行任务生成 |
| `SubagentStop` | 子 Agent 完成 | 聚合并行任务的结果 |
| `PreCompact` | 对话压缩请求 | 在摘要之前归档完整记录 |
| `PermissionRequest` | 将显示权限对话框 | 自定义权限处理 |
| `SessionStart` | 会话初始化 | 初始化日志记录和遥测 |
| `SessionEnd` | 会话终止 | 清理临时资源 |
| `Notification` | Agent 状态消息 | 将状态更新发送到 Slack |

### 配置 Hooks

```typescript
import { query, HookCallback, PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

// 使用 HookCallback 类型定义 hook 回调
const protectEnvFiles: HookCallback = async (input, toolUseID, { signal }) => {
  const preInput = input as PreToolUseHookInput;

  // 从工具的输入参数中提取文件路径
  const filePath = preInput.tool_input?.file_path as string;
  const fileName = filePath?.split('/').pop();

  // 如果目标是 .env 文件则阻止操作
  if (fileName === '.env') {
    return {
      hookSpecificOutput: {
        hookEventName: input.hook_event_name,
        permissionDecision: 'deny',
        permissionDecisionReason: 'Cannot modify .env files'
      }
    };
  }

  return {};
};

for await (const message of query({
  prompt: "Update the database configuration",
  options: {
    hooks: {
      PreToolUse: [{ matcher: 'Write|Edit', hooks: [protectEnvFiles] }]
    }
  }
})) {
  console.log(message);
}
```

### 自动批准特定工具

```typescript
import { query, HookCallback, PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

const autoApproveReadOnly: HookCallback = async (input, toolUseID, { signal }) => {
  if (input.hook_event_name !== 'PreToolUse') return {};

  const preInput = input as PreToolUseHookInput;
  const readOnlyTools = ['Read', 'Glob', 'Grep', 'LS'];
  if (readOnlyTools.includes(preInput.tool_name)) {
    return {
      hookSpecificOutput: {
        hookEventName: input.hook_event_name,
        permissionDecision: 'allow',
        permissionDecisionReason: 'Read-only tool auto-approved'
      }
    };
  }
  return {};
};

const options = {
  hooks: {
    PreToolUse: [{ hooks: [autoApproveReadOnly] }]
  }
};
```

### 链接多个 Hooks

```typescript
const options = {
  hooks: {
    PreToolUse: [
      { hooks: [rateLimiter] },        // 第一：检查速率限制
      { hooks: [authorizationCheck] }, // 第二：验证权限
      { hooks: [inputSanitizer] },     // 第三：清理输入
      { hooks: [auditLogger] }         // 最后：记录操作
    ]
  }
};
```

---

## 会话管理

### 获取会话 ID

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk"

let sessionId: string | undefined

const response = query({
  prompt: "Help me build a web application",
  options: {
    model: "claude-opus-4-6"
  }
})

for await (const message of response) {
  // 第一条消息是包含会话 ID 的系统初始化消息
  if (message.type === 'system' && message.subtype === 'init') {
    sessionId = message.session_id
    console.log(`Session started with ID: ${sessionId}`)
  }
}

// 之后，可以使用保存的 sessionId 来恢复会话
if (sessionId) {
  const resumedResponse = query({
    prompt: "Continue where we left off",
    options: {
      resume: sessionId
    }
  })
}
```

### 恢复会话

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk"

// 使用会话 ID 恢复之前的会话
const response = query({
  prompt: "Continue implementing the authentication system from where we left off",
  options: {
    resume: "session-xyz",
    model: "claude-opus-4-6",
    allowedTools: ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
  }
})

for await (const message of response) {
  console.log(message)
}
```

### 分叉会话

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk"

let sessionId: string | undefined

const response = query({
  prompt: "Help me design a REST API",
  options: { model: "claude-opus-4-6" }
})

for await (const message of response) {
  if (message.type === 'system' && message.subtype === 'init') {
    sessionId = message.session_id
    console.log(`Original session: ${sessionId}`)
  }
}

// 分叉会话以尝试不同的方法
const forkedResponse = query({
  prompt: "Now let's redesign this as a GraphQL API instead",
  options: {
    resume: sessionId,
    forkSession: true,  // 创建新的会话 ID
    model: "claude-opus-4-6"
  }
})

for await (const message of forkedResponse) {
  if (message.type === 'system' && message.subtype === 'init') {
    console.log(`Forked session: ${message.session_id}`)
  }
}
```

---

## 文件检查点

文件检查点功能跟踪 agent 会话期间的文件修改，允许将文件回退到任何先前状态。

### 实现检查点

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  // 步骤 1：启用检查点
  const opts = {
    enableFileCheckpointing: true,
    permissionMode: "acceptEdits" as const,
    extraArgs: { 'replay-user-messages': null },
    env: { ...process.env, CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING: '1' }
  };

  const response = query({
    prompt: "Refactor the authentication module",
    options: opts
  });

  let checkpointId: string | undefined;
  let sessionId: string | undefined;

  // 步骤 2：从第一条用户消息中捕获检查点 UUID
  for await (const message of response) {
    if (message.type === 'user' && message.uuid && !checkpointId) {
      checkpointId = message.uuid;
    }
    if ('session_id' in message && !sessionId) {
      sessionId = message.session_id;
    }
  }

  // 步骤 3：稍后，通过恢复会话并使用空提示来回退
  if (checkpointId && sessionId) {
    const rewindQuery = query({
      prompt: "",
      options: { ...opts, resume: sessionId }
    });

    for await (const msg of rewindQuery) {
      await rewindQuery.rewindFiles(checkpointId);
      break;
    }
    console.log(`Rewound to checkpoint: ${checkpointId}`);
  }
}

main();
```

### 在风险操作前设置检查点

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const response = query({
    prompt: "Refactor the authentication module",
    options: {
      enableFileCheckpointing: true,
      permissionMode: "acceptEdits" as const,
      extraArgs: { 'replay-user-messages': null },
      env: { ...process.env, CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING: '1' }
    }
  });

  let safeCheckpoint: string | undefined;

  for await (const message of response) {
    // 在每个 agent 轮次开始前更新检查点
    if (message.type === 'user' && message.uuid) {
      safeCheckpoint = message.uuid;
    }

    // 根据逻辑决定何时回退
    if (yourRevertCondition && safeCheckpoint) {
      await response.rewindFiles(safeCheckpoint);
      break;
    }
  }
}

main();
```

---

## 结构化输出

使用 JSON Schema 或 Zod 从 agent 工作流中返回经过验证的 JSON。

### 基本用法

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk'

// 定义想要返回的数据形状
const schema = {
  type: 'object',
  properties: {
    company_name: { type: 'string' },
    founded_year: { type: 'number' },
    headquarters: { type: 'string' }
  },
  required: ['company_name']
}

for await (const message of query({
  prompt: 'Research Anthropic and provide key company information',
  options: {
    outputFormat: {
      type: 'json_schema',
      schema: schema
    }
  }
})) {
  if (message.type === 'result' && message.structured_output) {
    console.log(message.structured_output)
    // { company_name: "Anthropic", founded_year: 2021, headquarters: "San Francisco, CA" }
  }
}
```

### 使用 Zod 的类型安全 Schema

```typescript
import { z } from 'zod'
import { query } from '@anthropic-ai/claude-agent-sdk'

// 使用 Zod 定义 schema
const FeaturePlan = z.object({
  feature_name: z.string(),
  summary: z.string(),
  steps: z.array(z.object({
    step_number: z.number(),
    description: z.string(),
    estimated_complexity: z.enum(['low', 'medium', 'high'])
  })),
  risks: z.array(z.string())
})

type FeaturePlan = z.infer<typeof FeaturePlan>

// 转换为 JSON Schema
const schema = z.toJSONSchema(FeaturePlan)

// 在查询中使用
for await (const message of query({
  prompt: 'Plan how to add dark mode support to a React app. Break it into implementation steps.',
  options: {
    outputFormat: {
      type: 'json_schema',
      schema: schema
    }
  }
})) {
  if (message.type === 'result' && message.structured_output) {
    // 验证并获取完全类型化的结果
    const parsed = FeaturePlan.safeParse(message.structured_output)
    if (parsed.success) {
      const plan: FeaturePlan = parsed.data
      console.log(`Feature: ${plan.feature_name}`)
      plan.steps.forEach(step => {
        console.log(`${step.step_number}. [${step.estimated_complexity}] ${step.description}`)
      })
    }
  }
}
```

### 错误处理

```typescript
for await (const msg of query({
  prompt: 'Extract contact info from the document',
  options: {
    outputFormat: {
      type: 'json_schema',
      schema: contactSchema
    }
  }
})) {
  if (msg.type === 'result') {
    if (msg.subtype === 'success' && msg.structured_output) {
      console.log(msg.structured_output)
    } else if (msg.subtype === 'error_max_structured_output_retries') {
      console.error('Could not produce valid output')
    }
  }
}
```

---

## MCP 连接外部工具

Model Context Protocol (MCP) 是一个用于将 AI agent 连接到外部工具和数据源的开放标准。

### HTTP 传输

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Use the docs MCP server to explain what hooks are in Claude Code",
  options: {
    mcpServers: {
      "claude-code-docs": {
        type: "http",
        url: "https://code.claude.com/docs/mcp"
      }
    },
    allowedTools: ["mcp__claude-code-docs__*"]
  }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### stdio 服务器

```typescript
options: {
  mcpServers: {
    "github": {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
      env: {
        GITHUB_TOKEN: process.env.GITHUB_TOKEN
      }
    }
  },
  allowedTools: ["mcp__github__list_issues", "mcp__github__search_issues"]
}
```

### 从配置文件加载

在项目根目录创建 `.mcp.json` 文件：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### 工具命名约定

MCP 工具遵循 `mcp__<server-name>__<tool-name>` 的命名模式。例如，名为 `github` 的服务器中的 `list_issues` 工具变成 `mcp__github__list_issues`。

### 示例：列出 GitHub Issues

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "List the 3 most recent issues in anthropics/claude-code",
  options: {
    mcpServers: {
      "github": {
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-github"],
        env: {
          GITHUB_TOKEN: process.env.GITHUB_TOKEN
        }
      }
    },
    allowedTools: ["mcp__github__list_issues"]
  }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

---

## 自定义工具

使用 `createSdkMcpServer` 和 `tool` 辅助函数定义类型安全的自定义工具。

### 创建自定义工具

```typescript
import { query, tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

// 创建一个带有自定义工具的 SDK MCP 服务器
const customServer = createSdkMcpServer({
  name: "my-custom-tools",
  version: "1.0.0",
  tools: [
    tool(
      "get_weather",
      "使用坐标获取某个位置的当前温度",
      {
        latitude: z.number().describe("纬度坐标"),
        longitude: z.number().describe("经度坐标")
      },
      async (args) => {
        const response = await fetch(
          `https://api.open-meteo.com/v1/forecast?latitude=${args.latitude}&longitude=${args.longitude}&current=temperature_2m&temperature_unit=fahrenheit`
        );
        const data = await response.json();

        return {
          content: [{
            type: "text",
            text: `温度：${data.current.temperature_2m}°F`
          }]
        };
      }
    )
  ]
});
```

### 使用自定义工具

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 在查询中使用自定义工具和流式输入
async function* generateMessages() {
  yield {
    type: "user" as const,
    message: {
      role: "user" as const,
      content: "旧金山的天气怎么样？"
    }
  };
}

for await (const message of query({
  prompt: generateMessages(),
  options: {
    mcpServers: {
      "my-custom-tools": customServer
    },
    allowedTools: ["mcp__my-custom-tools__get_weather"],
    maxTurns: 3
  }
})) {
  if (message.type === "result" && message.subtype === "success") {
    console.log(message.result);
  }
}
```

### 数据库查询工具示例

```typescript
const databaseServer = createSdkMcpServer({
  name: "database-tools",
  version: "1.0.0",
  tools: [
    tool(
      "query_database",
      "执行数据库查询",
      {
        query: z.string().describe("要执行的 SQL 查询"),
        params: z.array(z.any()).optional().describe("查询参数")
      },
      async (args) => {
        const results = await db.query(args.query, args.params || []);
        return {
          content: [{
            type: "text",
            text: `找到 ${results.length} 行：\n${JSON.stringify(results, null, 2)}`
          }]
        };
      }
    )
  ]
});
```

---

## 子 Agents

子 agent 是主 agent 可以生成的独立 agent 实例，用于处理专注的子任务。

### 创建子 Agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({
  prompt: "Review the authentication module for security issues",
  options: {
    // Task tool is required for subagent invocation
    allowedTools: ['Read', 'Grep', 'Glob', 'Task'],
    agents: {
      'code-reviewer': {
        description: 'Expert code review specialist. Use for quality, security, and maintainability reviews.',
        prompt: `You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.`,
        tools: ['Read', 'Grep', 'Glob'],
        model: 'sonnet'
      },
      'test-runner': {
        description: 'Runs and analyzes test suites. Use for test execution and coverage analysis.',
        prompt: `You are a test execution specialist. Run tests and provide clear analysis of results.`,
        tools: ['Bash', 'Read', 'Grep'],
      }
    }
  }
})) {
  if ('result' in message) console.log(message.result);
}
```

### AgentDefinition 配置

| 字段 | 类型 | 必需 | 描述 |
| :--- | :--- | :--- | :--- |
| `description` | `string` | 是 | 描述何时使用此 agent 的自然语言描述 |
| `prompt` | `string` | 是 | Agent 的系统提示词 |
| `tools` | `string[]` | 否 | 允许的工具名称数组 |
| `model` | `'sonnet' \| 'opus' \| 'haiku' \| 'inherit'` | 否 | 此 agent 的模型覆盖 |

### 检测子 Agent 调用

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Use the code-reviewer agent to review this codebase",
  options: {
    allowedTools: ["Read", "Glob", "Grep", "Task"],
    agents: {
      "code-reviewer": {
        description: "Expert code reviewer.",
        prompt: "Analyze code quality and suggest improvements.",
        tools: ["Read", "Glob", "Grep"]
      }
    }
  }
})) {
  const msg = message as any;

  // 检查子 agent 调用
  for (const block of msg.message?.content ?? []) {
    if (block.type === "tool_use" && block.name === "Task") {
      console.log(`Subagent invoked: ${block.input.subagent_type}`);
    }
  }

  // 检查消息是否来自子 agent 上下文
  if (msg.parent_tool_use_id) {
    console.log("  (running inside subagent)");
  }

  if ("result" in message) {
    console.log(message.result);
  }
}
```

### 恢复子 Agent

```typescript
import { query, type SDKMessage } from '@anthropic-ai/claude-agent-sdk';

// 从消息内容中提取 agentId
function extractAgentId(message: SDKMessage): string | undefined {
  if (!('message' in message)) return undefined;
  const content = JSON.stringify(message.message.content);
  const match = content.match(/agentId:\s*([a-f0-9-]+)/);
  return match?.[1];
}

let agentId: string | undefined;
let sessionId: string | undefined;

// 第一次调用
for await (const message of query({
  prompt: "Use the Explore agent to find all API endpoints in this codebase",
  options: { allowedTools: ['Read', 'Grep', 'Glob', 'Task'] }
})) {
  if ('session_id' in message) sessionId = message.session_id;
  const extractedId = extractAgentId(message);
  if (extractedId) agentId = extractedId;
  if ('result' in message) console.log(message.result);
}

// 恢复并提问
if (agentId && sessionId) {
  for await (const message of query({
    prompt: `Resume agent ${agentId} and list the top 3 most complex endpoints`,
    options: { allowedTools: ['Read', 'Grep', 'Glob', 'Task'], resume: sessionId }
  })) {
    if ('result' in message) console.log(message.result);
  }
}
```

---

## 斜杠命令

### 发现可用的斜杠命令

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Hello Claude",
  options: { maxTurns: 1 }
})) {
  if (message.type === "system" && message.subtype === "init") {
    console.log("Available slash commands:", message.slash_commands);
    // 示例: ["/compact", "/clear", "/help"]
  }
}
```

### 发送斜杠命令

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 发送斜杠命令
for await (const message of query({
  prompt: "/compact",
  options: { maxTurns: 1 }
})) {
  if (message.type === "result") {
    console.log("Command executed:", message.result);
  }
}
```

### 常用斜杠命令

**`/compact` - 压缩对话历史：**

```typescript
for await (const message of query({
  prompt: "/compact",
  options: { maxTurns: 1 }
})) {
  if (message.type === "system" && message.subtype === "compact_boundary") {
    console.log("Compaction completed");
    console.log("Pre-compaction tokens:", message.compact_metadata.pre_tokens);
  }
}
```

**`/clear` - 清除对话：**

```typescript
for await (const message of query({
  prompt: "/clear",
  options: { maxTurns: 1 }
})) {
  if (message.type === "system" && message.subtype === "init") {
    console.log("Conversation cleared, new session started");
    console.log("Session ID:", message.session_id);
  }
}
```

### 创建自定义斜杠命令

在 `.claude/commands/` 目录中创建 markdown 文件：

**`.claude/commands/refactor.md`：**

```markdown
Refactor the selected code to improve readability and maintainability.
Focus on clean code principles and best practices.
```

**带 Frontmatter 的示例 `.claude/commands/security-check.md`：**

```markdown
---
allowed-tools: Read, Grep, Glob
description: Run security vulnerability scan
model: claude-opus-4-6
---

Analyze the codebase for security vulnerabilities including:
- SQL injection risks
- XSS vulnerabilities
- Exposed credentials
```

---

## Agent 技能

技能为 Claude 扩展了专业能力，Claude 会在相关时自主调用这些技能。

### 在 SDK 中使用技能

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Help me process this PDF document",
  options: {
    cwd: "/path/to/project",
    settingSources: ["user", "project"],  // 从文件系统加载技能
    allowedTools: ["Skill", "Read", "Write", "Bash"]
  }
})) {
  console.log(message);
}
```

### 技能位置

- **项目技能**：`.claude/skills/`
- **用户技能**：`~/.claude/skills/`
- **插件技能**：与已安装的 Claude Code 插件捆绑

### 发现可用技能

```typescript
for await (const message of query({
  prompt: "What Skills are available?",
  options: {
    settingSources: ["user", "project"],
    allowedTools: ["Skill"]
  }
})) {
  console.log(message);
}
```

---

## 插件

插件允许使用自定义功能扩展 Claude Code，包括命令、agent、技能、hook 和 MCP 服务器。

### 加载插件

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Hello",
  options: {
    plugins: [
      { type: "local", path: "./my-plugin" },
      { type: "local", path: "/absolute/path/to/another-plugin" }
    ]
  }
})) {
  // 插件命令、agent 和其他功能现在可用
}
```

### 验证插件安装

```typescript
for await (const message of query({
  prompt: "Hello",
  options: {
    plugins: [{ type: "local", path: "./my-plugin" }]
  }
})) {
  if (message.type === "system" && message.subtype === "init") {
    console.log("Plugins:", message.plugins);
    console.log("Commands:", message.slash_commands);
  }
}
```

### 使用插件命令

来自插件的命令使用命名空间格式 `plugin-name:command-name`：

```typescript
for await (const message of query({
  prompt: "/my-plugin:greet",
  options: {
    plugins: [{ type: "local", path: "./my-plugin" }]
  }
})) {
  if (message.type === "assistant") {
    console.log(message.content);
  }
}
```

### 插件结构

```
my-plugin/
├── .claude-plugin/
│   └── plugin.json          # 必需：插件清单
├── commands/                 # 自定义斜杠命令
│   └── custom-cmd.md
├── agents/                   # 自定义 agents
│   └── specialist.md
├── skills/                   # Agent Skills
│   └── my-skill/
│       └── SKILL.md
├── hooks/                    # 事件处理程序
│   └── hooks.json
└── .mcp.json                 # MCP 服务器定义
```

---

## 总结

Claude Agent SDK 提供了构建强大 AI Agent 应用程序的完整工具集：

- **核心功能**：通过 `query()` 函数创建 agent 循环
- **流式处理**：支持流式输入和输出
- **权限控制**：灵活的权限模式和审批机制
- **Hooks 机制**：在关键执行点拦截和自定义行为
- **会话管理**：支持会话恢复和分叉
- **文件检查点**：跟踪和回退文件修改
- **结构化输出**：类型安全的 JSON 输出
- **MCP 集成**：连接外部工具和服务
- **自定义工具**：创建自己的工具扩展
- **子 Agents**：并行执行和专门化任务
- **扩展系统**：通过斜杠命令、技能和插件扩展功能


curl https://api.individual.githubcopilot.com/chat/completions \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer tid=aa63859c8039a64d7353d2d62545150f;exp=1772261963;sku=free_educational_quota;proxy-ep=proxy.individual.githubcopilot.com;st=dotcom;chat=1;cit=1;malfil=1;editor_preview_features=1;agent_mode=1;agent_mode_auto_approval=1;mcp=1;ccr=1;8kp=1;ip=43.132.141.15;asn=AS132203:7b6418bc94dc7ae9593902de70f0c800fe7d15825a25a9c21207ab7cfa0ce4cf" \
 -d '{
"model": "gpt-4o-mini-2024-07-18",
"messages": [
{"role": "user", "content": "你是什么模型, 支持多模态吗"}
],
"temperature": 0.7
}'