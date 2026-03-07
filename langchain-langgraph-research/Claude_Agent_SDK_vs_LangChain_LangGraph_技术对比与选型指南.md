# Claude Agent SDK vs LangChain/LangGraph 技术对比与选型指南

## 概述

**Claude Agent SDK** 和 **LangChain/LangGraph** 是两种不同定位的 AI Agent 开发方案：

| 方案 | 定位 | 核心价值 | 维护方 |
|------|------|----------|--------|
| **Claude Agent SDK** | Claude 专属 Agent SDK | 原生集成 Claude，开箱即用 | Anthropic 官方 |
| **LangChain** | 通用 Agent 构建框架 | 跨模型统一接口，组件化设计 | LangChain Inc |
| **LangGraph** | 工作流编排引擎 | 图结构状态管理，复杂流程控制 | LangChain Inc |

**核心差异**：Claude Agent SDK 是 Anthropic 官方解决方案，深度绑定 Claude 模型；LangChain 提供跨模型的通用组件；LangGraph 提供灵活的工作流编排能力。三者可以独立使用，也可以组合使用。

---

## 核心架构对比

### Claude Agent SDK 架构

Claude Agent SDK 以 `query()` 函数为核心，构建完整的 Agent 执行循环：

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Analyze this codebase",
  options: {
    model: "claude-sonnet-4-5",
    allowedTools: ["Read", "Edit", "Bash"],
    permissionMode: "acceptEdits",
    hooks: { PreToolUse: [myHook] },
    mcpServers: { "github": mcpConfig }
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}
```

**核心特点**：
- **流式优先**：输入输出都支持流式处理
- **内置 ReAct 循环**：自动处理工具调用的多轮迭代
- **权限系统**：4 种权限模式，细粒度审批
- **Hooks 机制**：12 种执行节点拦截

### LangChain 架构

LangChain 以 `createAgent()` 为核心，提供跨模型的 Agent 构建能力：

```typescript
import { createAgent } from "langchain";

const agent = createAgent({
  model: "openai:gpt-4o",  // 或 "anthropic:claude-sonnet-4-5"
  tools: [searchTool, calculatorTool],
  middleware: [loggingMiddleware, retryMiddleware],
  checkpointer: new MemorySaver(),
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "分析这个项目" }],
});
```

**核心特点**：
- **模型无关**：统一接口支持 OpenAI、Anthropic、Google 等多家提供商
- **组件化设计**：Models、Messages、Tools、Memory 独立可组合
- **Middleware 模式**：Node-style + Wrap-style 两种扩展方式
- **内置中间件**：summarization、retry、rate limit 等开箱即用

### LangGraph 架构

LangGraph 以 `StateGraph` 为核心，提供图结构的流程编排：

```typescript
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";

const graph = new StateGraph(MyState)
  .addNode("researcher", researcherNode)
  .addNode("analyst", analystNode)
  .addNode("writer", writerNode)
  .addEdge(START, "researcher")
  .addConditionalEdges("researcher", routeToAgent)
  .addEdge("researcher", "supervisor")
  .compile();
```

**核心特点**：
- **图结构编排**：节点（Nodes）和边（Edges）灵活定义流程
- **状态管理**：StateSchema 定义共享状态，支持 Reducer
- **条件分支**：addConditionalEdges 支持动态路由
- **子图嵌套**：模块化设计，支持复杂工作流

---

## Agent 创建对比

### Claude Agent SDK

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// 基础用法
for await (const message of query({
  prompt: "Review utils.py for bugs",
  options: {
    allowedTools: ["Read", "Edit", "Glob"],
    permissionMode: "acceptEdits"
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}

// 带会话管理
for await (const message of query({
  prompt: "Continue where we left off",
  options: {
    resume: "session-xyz",
    model: "claude-opus-4-6"
  }
})) {
  console.log(message);
}
```

### LangChain

```typescript
import { createAgent } from "langchain";

// 基础用法
const agent = createAgent({
  model: "openai:gpt-4o",
  tools: [searchTool, calculatorTool],
  systemPrompt: "You are a helpful assistant.",
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "帮我搜索 AI 新闻" }],
});

// 带状态持久化
import { MemorySaver } from "@langchain/langgraph";

const agent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  tools: [],
  checkpointer: new MemorySaver(),
});

// 使用 thread_id 维持对话
await agent.invoke(
  { messages: [{ role: "user", content: "hi! i am Bob" }] },
  { configurable: { thread_id: "1" } }
);
```

### LangGraph

```typescript
import { StateGraph, StateSchema, MessagesValue, START, END } from "@langchain/langgraph";

// 定义状态
const AgentState = new StateSchema({
  messages: MessagesValue,
  currentStep: z.string().default("start"),
});

// 定义节点
const agentNode = async (state) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

// 构建图
const graph = new StateGraph(AgentState)
  .addNode("agent", agentNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent")
  .compile();
```

**对比总结**：

| 维度 | Claude Agent SDK | LangChain | LangGraph |
|------|------------------|-----------|-----------|
| **Agent 创建** | `query()` | `createAgent()` | 手动构建 `StateGraph` |
| **抽象层级** | 高层（一行代码） | 高层（一行代码） | 中层（手动编排） |
| **调用方式** | invoke / stream | invoke / stream / batch | invoke / stream |
| **模型支持** | 仅 Claude | 多模型支持 | 使用 LangChain 模型 |

---

## 工具系统对比

### Claude Agent SDK 工具系统

**内置工具**：SDK 提供了一组内置工具，Agent 可以直接使用：

| 工具 | Agent 可以做什么 |
| :--- | :--- |
| `Read`、`Glob`、`Grep` | 只读分析 |
| `Read`、`Edit`、`Glob` | 分析和修改代码 |
| `Read`、`Edit`、`Bash`、`Glob`、`Grep` | 完全自动化 |

**自定义工具**：使用 `createSdkMcpServer` 和 `tool` 辅助函数：

```typescript
import { query, tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

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
          `https://api.open-meteo.com/v1/forecast?latitude=${args.latitude}&longitude=${args.longitude}&current=temperature_2m`
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

// 使用自定义工具
for await (const message of query({
  prompt: "旧金山的天气怎么样？",
  options: {
    mcpServers: { "my-custom-tools": customServer },
    allowedTools: ["mcp__my-custom-tools__get_weather"],
  }
})) {
  if (message.type === "result") {
    console.log(message.result);
  }
}
```

**MCP 集成**：

```typescript
// HTTP 传输
for await (const message of query({
  prompt: "Use the docs MCP server",
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
  console.log(message);
}

// stdio 服务器
options: {
  mcpServers: {
    "github": {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-github"],
      env: { GITHUB_TOKEN: process.env.GITHUB_TOKEN }
    }
  }
}
```

### LangChain 工具系统

**工具定义**：

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

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
```

**工具绑定**：

```typescript
// 绑定工具到模型
const modelWithTools = model.bindTools([weatherTool, calculatorTool]);

// 模型请求调用工具
const response = await modelWithTools.invoke("北京今天天气怎么样？");

// 手动执行工具（两步流程）
if (response.tool_calls?.length) {
  for (const toolCall of response.tool_calls) {
    const result = await weatherTool.invoke(toolCall.args);
    messages.push(new ToolMessage({
      tool_call_id: toolCall.id,
      content: result,
    }));
  }
  const finalResponse = await modelWithTools.invoke(messages);
}
```

**MCP 集成**：

```typescript
import { MultiServerMCPClient } from "@langchain/mcp-adapters";

const client = new MultiServerMCPClient({
  filesystem: {
    transport: "stdio",
    command: "npx",
    args: ["-y", "@anthropic/mcp-server-filesystem"],
  },
  weather: {
    transport: "sse",
    url: "https://mcp.example.com/weather",
    headers: { "Authorization": "Bearer xxx" },
  },
});

const tools = await client.getTools();
```

**对比总结**：

| 维度 | Claude Agent SDK | LangChain |
|------|------------------|-----------|
| **工具定义** | `tool()` 辅助函数 | `tool()` 辅助函数 |
| **工具绑定** | `allowedTools` 数组 | `model.bindTools([tools])` |
| **工具执行** | SDK 自动执行 | 需手动执行或使用 Agent |
| **MCP 支持** | 原生支持 HTTP/stdio | 通过 `@langchain/mcp-adapters` |

---

## Hooks vs Middleware 对比

### Claude Agent SDK Hooks

Hooks 让您可以在关键点拦截 agent 执行，提供 12 种执行节点：

```typescript
import { query, HookCallback, PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

const protectEnvFiles: HookCallback = async (input, toolUseID, { signal }) => {
  const preInput = input as PreToolUseHookInput;
  const filePath = preInput.tool_input?.file_path as string;
  const fileName = filePath?.split('/').pop();

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

**可用的 Hooks**：

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

### LangChain Middleware

LangChain 提供两种 Middleware 模式：

**Node-Style Hooks（顺序执行）**：

```typescript
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

**Wrap-Style Hooks（拦截执行）**：

```typescript
const retryMiddleware = createMiddleware({
  name: "RetryMiddleware",
  wrapModelCall: async (request, handler) => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await handler(request);
      } catch (e) {
        if (attempt === 2) throw e;
        console.log(`Retry ${attempt + 1}/3...`);
        await sleep(1000 * Math.pow(2, attempt));
      }
    }
  },
  wrapToolCall: async (request, handler) => {
    const startTime = Date.now();
    const result = await handler(request);
    console.log(`Tool ${request.toolCall.name} took ${Date.now() - startTime}ms`);
    return result;
  },
});
```

**内置 Middleware**：

| Middleware | 功能 | 使用场景 |
| --- | --- | --- |
| summarizationMiddleware | 自动总结长对话 | 长对话管理 |
| modelCallLimitMiddleware | 限制模型调用次数 | 防止无限循环 |
| toolCallLimitMiddleware | 限制工具调用次数 | 控制 API 成本 |
| toolRetryMiddleware | 工具失败自动重试 | 提高可靠性 |
| piiDetectionMiddleware | PII 信息检测 | 隐私保护 |
| humanInTheLoopMiddleware | 人工审批 | 高风险操作 |

**对比总结**：

| 维度 | Claude Agent SDK | LangChain |
|------|------------------|-----------|
| **权限模式** | 4 种内置模式 | 通过 Middleware 实现 |
| **审批回调** | `canUseTool` | `humanInTheLoopMiddleware` |
| **细粒度控制** | Hooks（12 种事件） | Middleware（Node + Wrap） |
| **Guardrails** | 不支持 | PII 检测、内容过滤 |

---

## 流式处理对比

### Claude Agent SDK

**流式输入**（独有功能）：

```typescript
async function* generateMessages() {
  yield {
    type: "user" as const,
    message: { role: "user" as const, content: "Analyze this codebase" }
  };
  
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  yield {
    type: "user" as const,
    message: {
      role: "user" as const,
      content: [
        { type: "text", text: "Review this architecture diagram" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: readFileSync("diagram.png", "base64") } }
      ]
    }
  };
}

for await (const message of query({
  prompt: generateMessages(),
  options: { maxTurns: 10 }
})) {
  console.log(message);
}
```

**流式输出**：

```typescript
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

### LangChain

```typescript
const stream = await agent.stream(
  { messages: [{ role: "user", content: "搜索 AI 新闻并总结" }] },
  { streamMode: "values" }
);

for await (const chunk of stream) {
  const latestMessage = chunk.messages.at(-1);
  if (latestMessage?.content) {
    console.log(`Agent: ${latestMessage.content}`);
  }
}
```

### LangGraph

LangGraph 支持 5 种流式模式：

```typescript
// values - 完整状态值
const stream = await graph.stream(input, { streamMode: "values" });

// updates - 增量更新
const stream = await graph.stream(input, { streamMode: "updates" });

// messages - 消息级别的流式
const stream = await graph.stream(input, { streamMode: "messages" });

// custom - 自定义事件
const stream = await graph.stream(input, { streamMode: "custom" });

// debug - 调试信息
const stream = await graph.stream(input, { streamMode: "debug" });
```

**对比总结**：

| 维度 | Claude Agent SDK | LangChain | LangGraph |
|------|------------------|-----------|-----------|
| **流式输入** | 支持（Generator） | 不支持 | 不支持 |
| **流式输出** | `includePartialMessages` | `stream()` | 5 种 streamMode |
| **流式模式** | 1 种 | 1 种 | 5 种（values/updates/messages/custom/debug） |

---

## 权限控制对比

### Claude Agent SDK 权限系统

**权限评估流程**：

1. **Hooks**：首先运行 hooks，可以允许、拒绝或继续到下一步
2. **权限规则**：检查 `settings.json` 中定义的规则
3. **权限模式**：应用当前激活的权限模式
4. **canUseTool 回调**：如果规则或模式未能解决，则调用回调

**权限模式**：

| 模式 | 描述 | 工具行为 |
| :--- | :--- | :--- |
| `default` | 标准权限行为 | 无自动批准；未匹配的工具会触发 `canUseTool` 回调 |
| `acceptEdits` | 自动接受文件编辑 | 文件编辑和文件系统操作会自动批准 |
| `bypassPermissions` | 绕过所有权限检查 | 所有工具无需权限提示即可运行 |
| `plan` | 规划模式 | 不执行工具；Claude 只进行规划 |

**动态审批**：

```typescript
for await (const message of query({
  prompt: "Create a test file in /tmp",
  options: {
    canUseTool: async (toolName, input) => {
      console.log(`\nTool: ${toolName}`);
      if (toolName === "Bash") {
        console.log(`Command: ${input.command}`);
      }
      const response = await prompt("Allow this action? (y/n): ");
      if (response.toLowerCase() === "y") {
        return { behavior: "allow", updatedInput: input };
      } else {
        return { behavior: "deny", message: "User denied this action" };
      }
    },
  },
})) {
  console.log(message);
}
```

### LangChain 权限控制

LangChain 通过 Middleware 实现权限控制：

```typescript
import { humanInTheLoopMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [bashTool, fileTool],
  middleware: [
    humanInTheLoopMiddleware({
      // 需要审批的工具
      toolsRequiringApproval: ["bash", "file_write"],
      // 审批回调
      approvalCallback: async (toolCall) => {
        console.log(`Tool: ${toolCall.name}`);
        console.log(`Args: ${JSON.stringify(toolCall.args)}`);
        const response = await prompt("Approve? (y/n): ");
        return response === "y";
      },
    }),
  ],
});
```

---

## 会话管理对比

### Claude Agent SDK

**会话恢复**：

```typescript
let sessionId: string | undefined;

for await (const message of query({
  prompt: "Help me build a web application",
})) {
  if (message.type === 'system' && message.subtype === 'init') {
    sessionId = message.session_id;
    console.log(`Session started: ${sessionId}`);
  }
}

// 恢复会话
for await (const message of query({
  prompt: "Continue where we left off",
  options: { resume: sessionId }
})) {
  console.log(message);
}
```

**会话分叉**（独有功能）：

```typescript
// 分叉会话以尝试不同的方法
for await (const message of query({
  prompt: "Now let's redesign this as a GraphQL API",
  options: {
    resume: sessionId,
    forkSession: true,  // 创建新的会话 ID
  }
})) {
  console.log(message);
}
```

**文件检查点**（独有功能）：

```typescript
const response = query({
  prompt: "Refactor the authentication module",
  options: {
    enableFileCheckpointing: true,
    permissionMode: "acceptEdits",
    env: { CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING: '1' }
  }
});

let checkpointId: string | undefined;

for await (const message of response) {
  if (message.type === 'user' && message.uuid) {
    checkpointId = message.uuid;
  }
}

// 回退到检查点
await response.rewindFiles(checkpointId);
```

### LangChain/LangGraph

**短期记忆**：

```typescript
import { MemorySaver } from "@langchain/langgraph";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  checkpointer: new MemorySaver(),
});

await agent.invoke(
  { messages: [{ role: "user", content: "hi! i am Bob" }] },
  { configurable: { thread_id: "1" } }
);

// 同一 thread_id 共享对话历史
await agent.invoke(
  { messages: [{ role: "user", content: "What's my name?" }] },
  { configurable: { thread_id: "1" } }
);
```

**生产环境持久化**：

```typescript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const checkpointer = PostgresSaver.fromConnString(DB_URI);

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  checkpointer,
});
```

**消息管理策略**：

```typescript
import { summarizationMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [],
  middleware: [
    summarizationMiddleware({
      model: "gpt-4o-mini",
      trigger: { tokens: 4000 },
      keep: { messages: 20 },
    }),
  ],
  checkpointer: new MemorySaver(),
});
```

**对比总结**：

| 维度 | Claude Agent SDK | LangChain/LangGraph |
|------|------------------|---------------------|
| **会话恢复** | `resume` 参数 | `checkpointer` |
| **会话分叉** | `forkSession: true` | 不支持 |
| **状态持久化** | 会话文件 | MemorySaver / PostgresSaver |
| **消息管理** | 手动处理 | Middleware 自动处理 |

---

## 多 Agent 协作对比

### Claude Agent SDK 子 Agents

```typescript
for await (const message of query({
  prompt: "Review the authentication module for security issues",
  options: {
    allowedTools: ['Read', 'Grep', 'Glob', 'Task'],
    agents: {
      'code-reviewer': {
        description: 'Expert code review specialist.',
        prompt: 'You are a code review specialist with expertise in security.',
        tools: ['Read', 'Grep', 'Glob'],
        model: 'sonnet'
      },
      'test-runner': {
        description: 'Runs and analyzes test suites.',
        prompt: 'You are a test execution specialist.',
        tools: ['Bash', 'Read', 'Grep'],
      }
    }
  }
})) {
  console.log(message);
}
```

### LangChain Multi-Agent 模式

**Subagents 模式**：

```typescript
import { createAgent, tool } from "langchain";
import { z } from "zod";

// 创建子 Agent
const researchAgent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  tools: [searchTool, scrapeTool],
  systemPrompt: "You are a research specialist..."
});

const codeAgent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
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
    description: "Research a topic thoroughly.",
    schema: z.object({ query: z.string() })
  }
);

// 主 Agent 协调子 Agent
const mainAgent = createAgent({
  model: "anthropic:claude-sonnet-4-5",
  tools: [callResearchAgent, callCodeAgent],
  systemPrompt: "You are a supervisor that coordinates specialized agents."
});
```

### LangGraph 多 Agent 编排

```typescript
// 定义共享状态
const TeamState = new StateSchema({
  messages: MessagesValue,
  currentAgent: z.string().default("supervisor"),
  researchResults: new ReducedValue(z.array(z.string()), { reducer: (a, b) => a.concat(b) }),
  analysis: z.string().optional(),
  finalReport: z.string().optional(),
  taskComplete: z.boolean().default(false),
});

// Supervisor 决策
const routeSchema = z.object({
  next: z.enum(["researcher", "analyst", "writer", "FINISH"]),
  reason: z.string(),
});

const supervisorModel = model.withStructuredOutput(routeSchema);

const supervisorNode = async (state) => {
  const decision = await supervisorModel.invoke([
    new SystemMessage(`You are a supervisor managing a team of agents...`),
    ...state.messages,
  ]);
  return { currentAgent: decision.next };
};

// 路由逻辑
const routeToAgent = (state) => {
  if (state.currentAgent === "FINISH" || state.taskComplete) {
    return "__end__";
  }
  return state.currentAgent;
};

// 构建图
const multiAgentGraph = new StateGraph(TeamState)
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherNode)
  .addNode("analyst", analystNode)
  .addNode("writer", writerNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", routeToAgent, ["researcher", "analyst", "writer", "__end__"])
  .addEdge("researcher", "supervisor")
  .addEdge("analyst", "supervisor")
  .addEdge("writer", "supervisor")
  .compile();
```

**对比总结**：

| 维度 | Claude Agent SDK | LangChain | LangGraph |
|------|------------------|-----------|-----------|
| **子 Agent** | `agents` 配置 + Task 工具 | Subagents 模式 | StateGraph 节点 |
| **协调方式** | 主 Agent 作为 Supervisor | 主 Agent 作为工具协调 | 图结构编排 |
| **路由决策** | 模型决策 | 模型决策 | 条件边路由 |

---

## 结构化输出对比

### Claude Agent SDK

```typescript
import { z } from 'zod';

const FeaturePlan = z.object({
  feature_name: z.string(),
  summary: z.string(),
  steps: z.array(z.object({
    step_number: z.number(),
    description: z.string(),
    estimated_complexity: z.enum(['low', 'medium', 'high'])
  })),
  risks: z.array(z.string())
});

const schema = z.toJSONSchema(FeaturePlan);

for await (const message of query({
  prompt: 'Plan how to add dark mode support to a React app.',
  options: {
    outputFormat: {
      type: 'json_schema',
      schema: schema
    }
  }
})) {
  if (message.type === 'result' && message.structured_output) {
    const parsed = FeaturePlan.safeParse(message.structured_output);
    if (parsed.success) {
      console.log(parsed.data);
    }
  }
}
```

### LangChain

```typescript
import { z } from "zod";

const ResponseSchema = z.object({
  answer: z.string().describe("问题的答案"),
  confidence: z.number().min(0).max(1).describe("置信度"),
  sources: z.array(z.string()).describe("信息来源列表"),
});

const structuredModel = model.withStructuredOutput(ResponseSchema);

const result = await structuredModel.invoke("法国的首都是哪里？");
console.log(result);
// { answer: "Paris", confidence: 0.99, sources: ["Wikipedia"] }
```

---

## 扩展系统对比

### Claude Agent SDK 扩展系统

**斜杠命令**：

```typescript
// 发现可用命令
for await (const message of query({ prompt: "Hello", options: { maxTurns: 1 } })) {
  if (message.type === "system" && message.subtype === "init") {
    console.log("Available commands:", message.slash_commands);
  }
}

// 发送命令
for await (const message of query({ prompt: "/compact", options: { maxTurns: 1 } })) {
  console.log(message);
}
```

**自定义命令**：在 `.claude/commands/` 目录创建 markdown 文件。

**技能（Skills）**：专业能力扩展，Claude 会在相关时自主调用。

**插件（Plugins）**：完整扩展包，包含命令、agent、技能、hook 和 MCP 服务器。

### LangChain 扩展系统

LangChain 主要通过 Middleware 和 Tool 扩展，没有内置的命令系统。

---

## 优缺点分析

### Claude Agent SDK

#### 优点

| 优点 | 说明 |
|------|------|
| **官方支持** | Anthropic 官方维护，与 Claude 模型深度集成，功能最完整 |
| **API 简洁** | 一行代码启动 Agent，学习曲线低 |
| **权限系统完善** | 4 种内置模式 + canUseTool 回调 + 12 种 Hooks，控制粒度细 |
| **流式输入** | 独有的 Generator 输入模式，支持动态消息队列 |
| **文件检查点** | 独有功能，跟踪文件修改，支持回退 |
| **扩展系统** | 斜杠命令、技能、插件三层扩展体系 |
| **Claude Code 集成** | 与 Claude CLI 共享配置、技能和上下文 |

#### 缺点

| 缺点 | 说明 |
|------|------|
| **模型绑定** | 仅支持 Claude 模型，无法切换到其他 LLM |
| **成本高** | Claude API 价格较高，大规模使用成本压力 |
| **生态封闭** | 仅限 Anthropic 生态，无法利用其他平台能力 |
| **批量处理** | 不支持 batch 操作，需要手动并行 |
| **持久化有限** | 依赖会话文件，生产级持久化能力较弱 |

### LangChain

#### 优点

| 优点 | 说明 |
|------|------|
| **模型无关** | 统一接口支持 OpenAI、Anthropic、Google、Azure、本地模型 |
| **成本灵活** | 可根据场景选择不同价格的模型（如 GPT-4o-mini） |
| **组件化设计** | Models、Messages、Tools、Memory 独立可组合 |
| **内置 Middleware** | summarization、retry、rate limit、PII 检测等开箱即用 |
| **批量处理** | 支持 batch 操作，并行处理多个请求 |
| **生态丰富** | 大量第三方集成、社区资源、文档完善 |
| **生产就绪** | PostgresSaver 等生产级持久化方案 |

#### 缺点

| 缺点 | 说明 |
|------|------|
| **学习曲线** | 概念较多，需要理解 Models、Messages、Tools 等组件 |
| **版本更新快** | API 变化频繁，迁移成本高 |
| **调试复杂** | 多层抽象，错误追踪困难 |
| **性能开销** | 抽象层带来一定性能损耗 |

### LangGraph

#### 优点

| 优点 | 说明 |
|------|------|
| **流程可视化** | 图结构直观展示执行流程，便于理解和调试 |
| **状态管理** | StateSchema + Reducer 提供强大的状态管理能力 |
| **灵活性高** | 条件分支、循环、并行执行完全可控 |
| **子图嵌套** | 模块化设计，支持复杂工作流 |
| **5 种流式模式** | values/updates/messages/custom/debug 满足不同场景 |
| **与 LangChain 兼容** | 可复用 LangChain 的 Models、Tools 等组件 |

#### 缺点

| 缺点 | 说明 |
|------|------|
| **学习曲线高** | 需要理解图、状态、节点、边等概念 |
| **代码量大** | 需要手动定义节点、边、路由逻辑 |
| **调试困难** | 复杂图结构的调试需要额外工具支持 |
| **过度设计风险** | 简单场景使用 LangGraph 可能过度复杂 |

---

## 技术方案抉择指南

### 决策树

```
开始
  │
  ├─ 是否需要跨模型支持？
  │   ├─ 是 → LangChain 或 LangChain + LangGraph
  │   └─ 否 ↓
  │
  ├─ 是否使用 Claude 模型？
  │   ├─ 是 → Claude Agent SDK
  │   └─ 否 → LangChain
  │
  ├─ 是否需要复杂工作流？
  │   ├─ 是（多分支、循环、并行）→ LangGraph 或 LangChain + LangGraph
  │   └─ 否 ↓
  │
  ├─ 是否需要多 Agent 协作？
  │   ├─ 是 → LangGraph 或 LangChain Subagents
  │   └─ 否 ↓
  │
  └─ 项目规模？
      ├─ 小型/原型 → Claude Agent SDK 或 LangChain
      ├─ 中型 → LangChain
      └─ 大型/生产 → LangChain + LangGraph
```

### 场景推荐

#### 选择 Claude Agent SDK 的场景

| 场景 | 原因 |
|------|------|
| **Claude 深度用户** | 原生集成，功能最完整，与 Claude Code 无缝衔接 |
| **快速原型开发** | API 简洁，一行代码启动，学习成本低 |
| **需要细粒度权限控制** | 4 种模式 + Hooks，控制粒度最细 |
| **文件操作密集型** | 文件检查点、回退功能独有 |
| **Claude Code 插件开发** | 共享配置、技能、上下文 |

**示例**：开发 Claude Code 插件，需要读取用户项目文件、修改代码、回滚错误操作。

#### 选择 LangChain 的场景

| 场景 | 原因 |
|------|------|
| **多模型切换** | 需要在不同 LLM 之间切换，或做模型对比 |
| **成本敏感** | 根据场景选择不同价格的模型 |
| **组件化开发** | 需要灵活组合 Models、Tools、Memory |
| **生产环境** | 需要数据库级别的状态持久化 |
| **需要内置功能** | 重试、限流、摘要、PII 检测等开箱即用 |

**示例**：开发企业级客服系统，需要在高峰期使用 GPT-4o-mini 降低成本，非高峰期使用 GPT-4 提升质量。

#### 选择 LangGraph 的场景

| 场景 | 原因 |
|------|------|
| **复杂工作流** | 多步骤、多分支、有循环的业务流程 |
| **多 Agent 协作** | 需要精细控制 Agent 间的协作和状态传递 |
| **状态管理复杂** | 需要自定义 StateSchema 和 Reducer |
| **需要可视化** | 图结构便于理解和沟通 |

**示例**：开发研究报告生成系统，包含：研究 → 分析 → 写作 → 审核 → 修改循环，每个步骤可能有分支。

#### 组合使用场景

| 组合 | 场景 |
|------|------|
| **LangChain + LangGraph** | 需要 LangChain 的跨模型能力 + LangGraph 的流程控制 |
| **Claude Agent SDK + LangGraph** | Claude 深度用户，但需要复杂工作流（需自行适配） |

**示例**：开发代码审查系统，使用 LangGraph 编排流程（扫描 → 分析 → 报告），使用 LangChain 支持不同模型。

---

## 迁移成本评估

### 从 Claude Agent SDK 迁移到 LangChain

| 迁移项 | 难度 | 说明 |
|--------|------|------|
| query() → createAgent() | 低 | API 相似，概念映射清晰 |
| Hooks → Middleware | 中 | 需要重新实现拦截逻辑 |
| 权限模式 | 高 | LangChain 无内置权限模式，需自行实现 |
| 文件检查点 | 高 | 需要自行实现回滚机制 |
| 斜杠命令/技能/插件 | 高 | 无对应功能，需要自行设计 |

### 从 LangChain 迁移到 Claude Agent SDK

| 迁移项 | 难度 | 说明 |
|--------|------|------|
| createAgent() → query() | 低 | API 相似 |
| 多模型 → Claude only | 不可能 | 核心差异，无法迁移 |
| Middleware → Hooks | 中 | 需要重新映射事件 |
| Guardrails | 中 | 需要通过 Hooks 实现 |

---

## 总结对比表

| 维度 | Claude Agent SDK | LangChain | LangGraph |
|------|------------------|-----------|-----------|
| **定位** | Claude 专属 SDK | 通用 Agent 框架 | 工作流编排引擎 |
| **学习曲线** | 低 | 中 | 中高 |
| **灵活性** | 中 | 高 | 最高 |
| **开箱即用** | 高 | 高 | 中 |
| **生态集成** | Claude 深度集成 | 多平台支持 | 使用 LangChain 生态 |
| **生产就绪** | 中 | 高 | 高 |
| **成本** | 高（仅 Claude） | 灵活 | 使用 LangChain 模型 |
| **适用规模** | 中小型项目 | 中大型项目 | 大型复杂项目 |

---

## 最终建议

### 技术选型原则

1. **优先考虑模型选择**：如果需要跨模型或成本优化，选择 LangChain；如果深度绑定 Claude，选择 Claude Agent SDK
2. **根据复杂度选择**：简单场景用高层抽象（Claude SDK / LangChain），复杂场景用中层抽象（LangGraph）
3. **考虑团队技能**：团队熟悉 Claude 生态选 Claude SDK，熟悉 Python/JS 生态选 LangChain
4. **评估长期成本**：Claude API 价格较高，大规模使用需评估成本

### 典型项目推荐

| 项目类型 | 推荐方案 | 原因 |
|----------|----------|------|
| Claude Code 插件 | Claude Agent SDK | 无缝集成，共享上下文 |
| 企业级 ChatBot | LangChain | 跨模型、生产持久化、成本灵活 |
| 多 Agent 协作系统 | LangGraph | 流程可控、状态管理强 |
| 个人 AI 助手 | Claude Agent SDK | 快速开发、功能完整 |
| 研究型项目 | LangChain + LangGraph | 灵活组合、可扩展 |