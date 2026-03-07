# LangChain MCP（模型上下文协议）速查指南

## 概览

MCP（Model Context Protocol）是一个开放协议，用来标准化应用程序如何向 LLM 提供工具和上下文。LangChain Agent 可以通过 `@langchain/mcp-adapters` 库使用 MCP 服务器上定义的工具。

**核心优势**:
- **标准化**: 统一的协议让不同的 MCP 服务器互相兼容
- **灵活性**: 支持多种传输方式（stdio、HTTP/SSE）
- **可扩展性**: 轻松集成新的工具源
- **解耦**: Agent 不需要关心工具是如何实现的

---

## 快速开始

### 安装库

```bash
npm install @langchain/mcp-adapters
```

### 连接多个 MCP 服务器

```javascript
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatAnthropic } from "@langchain/anthropic";
import { createAgent } from "langchain";

// 连接多个 MCP 服务器
const client = new MultiServerMCPClient({
  // 本地数学服务器（stdio 传输）
  math: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/math_server.js"],
  },
  // 远程天气服务器（HTTP 传输）
  weather: {
    transport: "http",
    url: "http://localhost:8000/mcp",
  },
});

// 从所有服务器获取工具
const tools = await client.getTools();

// 创建 Agent，使用 MCP 工具
const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools,
});

// 使用数学工具
const mathResponse = await agent.invoke({
  messages: [{ role: "user", content: "what's (3 + 5) x 12?" }],
});

// 使用天气工具
const weatherResponse = await agent.invoke({
  messages: [{ role: "user", content: "what is the weather in nyc?" }],
});
```

**重要笔记**: `MultiServerMCPClient` 默认是无状态的。每次工具调用都会创建一个新的 MCP 会话，执行工具，然后清理。详见[有状态会话](#有状态会话)部分。

---

## 创建自定义 MCP 服务器

### 安装 MCP SDK

```bash
npm install @modelcontextprotocol/sdk
```

### 示例 1：数学服务器（stdio 传输）

使用标准输入/输出进行通信，适合本地工具：

```javascript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// 创建服务器
const server = new Server(
  {
    name: "math-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 定义可用工具
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "add",
        description: "Add two numbers",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "First number",
            },
            b: {
              type: "number",
              description: "Second number",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "First number",
            },
            b: {
              type: "number",
              description: "Second number",
            },
          },
          required: ["a", "b"],
        },
      },
    ],
  };
});

// 实现工具调用
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "add": {
      const { a, b } = request.params.arguments as { a: number; b: number };
      return {
        content: [
          {
            type: "text",
            text: String(a + b),
          },
        ],
      };
    }
    case "multiply": {
      const { a, b } = request.params.arguments as { a: number; b: number };
      return {
        content: [
          {
            type: "text",
            text: String(a * b),
          },
        ],
      };
    }
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

// 启动服务器
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Math MCP server running on stdio");
}

main();
```

### 示例 2：天气服务器（SSE/HTTP 传输）

使用 HTTP 进行通信，适合远程服务：

```javascript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import express from "express";

const app = express();
app.use(express.json());

// 创建服务器
const server = new Server(
  {
    name: "weather-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// 定义天气工具
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_weather",
        description: "Get weather for location",
        inputSchema: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "Location to get weather for",
            },
          },
          required: ["location"],
        },
      },
    ],
  };
});

// 实现天气查询
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "get_weather": {
      const { location } = request.params.arguments as {
        location: string;
      };
      return {
        content: [
          {
            type: "text",
            text: `It's always sunny in ${location}`,
          },
        ],
      };
    }
    default:
      throw new Error(`Unknown tool: ${request.params.name}`);
  }
});

// HTTP 端点处理 MCP 请求
app.post("/mcp", async (req, res) => {
  const transport = new SSEServerTransport("/mcp", res);
  await server.connect(transport);
});

// 启动 HTTP 服务器
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Weather MCP server running on port ${PORT}`);
});
```

### 创建自定义服务器的步骤

1. **导入 SDK**: 使用 `@modelcontextprotocol/sdk`
2. **定义服务器**: 创建 `Server` 实例，声明名称和版本
3. **声明能力**: 指定服务器支持的功能（如 `tools`）
4. **列出工具**: 实现 `ListToolsRequestSchema` 处理器返回可用工具列表
5. **实现工具**: 实现 `CallToolRequestSchema` 处理器执行工具逻辑
6. **连接传输**: 使用 `StdioServerTransport` 或 `SSEServerTransport`
7. **启动服务器**: 调用 `server.connect(transport)`

---

## 传输方式

### stdio（标准输入/输出）

**特点**:
- 客户端以子进程形式启动服务器
- 通过标准输入/输出通信
- 适合本地工具和简单设置
- 每次工具调用都会创建新的会话（默认）

**配置**:

```javascript
const client = new MultiServerMCPClient({
  math: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/math_server.js"],
  },
});
```

**适用场景**:
- 本地数据库查询
- 文件系统操作
- 开发和测试
- 轻量级工具

### HTTP / SSE（服务器发送事件）

**特点**:
- 基于 HTTP 的客户端-服务器通信
- 适合远程服务
- 可以在任何支持 HTTP 的地方运行服务器

**配置**:

```javascript
const client = new MultiServerMCPClient({
  weather: {
    transport: "sse",
    url: "http://localhost:8000/mcp",
  },
});
```

**传递请求头**:

```javascript
const client = new MultiServerMCPClient({
  api: {
    transport: "sse",
    url: "https://api.example.com/mcp",
    headers: {
      "Authorization": "Bearer YOUR_TOKEN",
      "Custom-Header": "value",
    },
  },
});
```

**适用场景**:
- 微服务架构
- 远程 API 集成
- 多个客户端共享服务器
- 云部署

---

## 核心功能

### 工具（Tools）

工具允许 MCP 服务器暴露可执行的函数，LLM 可以调用它们来执行操作。LangChain 将 MCP 工具转换为 LangChain 工具，使其可以直接在任何 LangChain Agent 或工作流中使用。

#### 加载工具

```javascript
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { createAgent } from "langchain";

const client = new MultiServerMCPClient({
  // 配置你的服务器
});

// 从所有服务器获取工具
const tools = await client.getTools();

// 创建 Agent
const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools,
});
```

#### 工具定义最佳实践

```javascript
{
  name: "search_database",
  description: "Search for users in the database by name or email",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search term (name or email)",
      },
      limit: {
        type: "number",
        description: "Maximum number of results (default: 10)",
      },
    },
    required: ["query"],
  },
}
```

**工具定义建议**:
- **名称**: 清晰简洁，用下划线分隔（如 `search_database`）
- **描述**: 明确说明工具做什么和何时使用
- **参数**: 定义每个参数的类型和用途
- **必需参数**: 明确指定哪些参数是必需的

---

## 有状态会话

默认情况下，`MultiServerMCPClient` 是无状态的。有些场景需要持久的服务器连接：

```javascript
import { createStatefulMCPClient } from "@langchain/mcp-adapters";

// 创建有状态客户端（保持持久连接）
const statefulClient = await createStatefulMCPClient({
  math: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/math_server.js"],
  },
});

const tools = await statefulClient.getTools();

// 连接在整个 agent 生命周期中保持活动
const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools,
});

// 确保清理连接
process.on("exit", () => {
  statefulClient.close();
});
```

**何时使用有状态会话**:
- 需要维持服务器状态
- 频繁调用工具时的性能优化
- 服务器初始化成本很高

**何时使用无状态会话**:
- 简单的工具调用
- 避免长期连接开销
- 无需持久状态

---

## 常见集成模式

### 模式 1：多服务器编排

组合来自不同服务器的工具：

```javascript
const client = new MultiServerMCPClient({
  database: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/db_server.js"],
  },
  api: {
    transport: "http",
    url: "http://api.example.com/mcp",
  },
  files: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/file_server.js"],
  },
});

const tools = await client.getTools();
```

### 模式 2：带认证的远程服务

```javascript
const client = new MultiServerMCPClient({
  secure_api: {
    transport: "sse",
    url: "https://secure-api.example.com/mcp",
    headers: {
      "Authorization": `Bearer ${process.env.API_KEY}`,
      "X-Client-ID": "my-agent",
    },
  },
});
```

### 模式 3：动态服务器配置

```javascript
const serverConfigs = {
  dev: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/local_server.js"],
  },
  prod: {
    transport: "sse",
    url: process.env.PROD_MCP_URL,
    headers: {
      "Authorization": `Bearer ${process.env.PROD_TOKEN}`,
    },
  },
};

const env = process.env.NODE_ENV || "dev";
const client = new MultiServerMCPClient({
  services: serverConfigs[env],
});
```

---

## MCP 最佳实践

### 服务器设计

1. **清晰的工具描述**
   - 明确说明工具的用途
   - 提供参数的详细文档
   - 示例用法

2. **错误处理**
   ```javascript
   server.setRequestHandler(CallToolRequestSchema, async (request) => {
     try {
       // 工具实现
     } catch (error) {
       return {
         content: [
           {
             type: "text",
             text: `Error: ${error.message}`,
           },
         ],
         isError: true,
       };
     }
   });
   ```

3. **结果格式**
   - 返回结构化的内容
   - 支持多种内容类型（文本、JSON 等）
   - 考虑响应大小

### 客户端使用

1. **工具选择**
   - 避免暴露太多工具给 Agent
   - 根据场景动态选择工具
   - 为工具分类组织

2. **性能考虑**
   - 使用有状态会话处理频繁的工具调用
   - 缓存工具列表
   - 监控传输延迟

3. **错误恢复**
   ```javascript
   let tools;
   try {
     tools = await client.getTools();
   } catch (error) {
     console.error("Failed to load MCP tools:", error);
     // 使用备用工具或降级模式
   }
   ```

---

## 常见问题

**Q: stdio 和 HTTP 传输应该选哪个？**

A: 
- **stdio**: 本地工具，开发环境，简单设置
- **HTTP**: 远程服务，多客户端共享，云部署

**Q: 如何在多个 MCP 服务器之间共享状态？**

A: 使用 Store 机制存储跨服务器的数据。MCP 服务器可以写入 LangChain Store，其他服务器可以读取。

**Q: 能否同时连接相同服务的多个实例？**

A: 可以，在配置中使用不同的键：
```javascript
{
  db_primary: { ... },
  db_replica: { ... },
}
```

**Q: 如何处理 MCP 服务器的超时？**

A: 根据传输类型配置超时：
```javascript
{
  transport: "sse",
  url: "http://localhost:8000/mcp",
  timeout: 30000,  // 30 秒
}
```

**Q: MCP 服务器可以访问 Agent State 吗？**

A: 目前 MCP 主要关注工具。未来版本可能支持更深的集成。

**Q: 如何测试自定义 MCP 服务器？**

A: 使用 MCP 提供的测试工具或手动测试：
```javascript
// 手动测试
const { MultiServerMCPClient } = require("@langchain/mcp-adapters");

const client = new MultiServerMCPClient({
  test: {
    transport: "stdio",
    command: "node",
    args: ["/path/to/server.js"],
  },
});

const tools = await client.getTools();
console.log(tools);
```

---

## 相关资源

- [MCP 官方文档](https://modelcontextprotocol.io/introduction) - 完整的 MCP 规范
- [MCP 传输文档](https://modelcontextprotocol.io/docs/concepts/transports) - 传输详解
- [@langchain/mcp-adapters](https://github.com/langchain-ai/langchainjs/tree/main/libs/langchain-mcp-adapters/) - LangChain MCP 适配器源代码
- [LangChain 工具](https://docs.langchain.com/oss/javascript/langchain/tools) - 工具创建指南
