# LangChain Multi-Agent 综合指南

## 概览

Multi-Agent 系统通过协调专业化组件来解决复杂工作流。但并非所有复杂任务都需要这种方法——一个具有正确（有时是动态的）工具和提示的单一 Agent 往往能达到类似的效果。

**为什么需要 Multi-Agent？**

当开发者说需要 "multi-agent" 时，通常是在寻找以下一种或多种能力：

1. **上下文管理**: 提供专业知识而不使模型的上下文窗口超载
2. **分布式开发**: 允许不同团队独立开发和维护能力
3. **并行化**: 为子任务生成专业化 worker 并并发执行

**适用场景**:
- 单个 Agent 拥有太多工具，难以做出正确决策
- 任务需要具有广泛上下文的专业知识
- 需要强制执行顺序约束

**核心要点**: Multi-Agent 设计的核心是 **Context Engineering**——决定每个 Agent 看到什么信息。系统质量取决于确保每个 Agent 能访问其任务所需的正确数据。

---

## 五种核心模式

| 模式 | 描述 | 最佳场景 |
|------|------|---------|
| **Subagents** | 主 Agent 将子 Agent 作为工具协调 | 多个独立领域，集中控制 |
| **Handoffs** | 基于状态动态改变行为 | 顺序流程，需要与用户直接交互 |
| **Skills** | 按需加载专业提示和知识 | 许多专业化，轻量级组合 |
| **Router** | 路由步骤分类输入并分发到专业 Agent | 明确的垂直领域，并行查询 |
| **Custom Workflow** | 用 LangGraph 构建定制执行流 | 需要完全控制，混合确定性和智能行为 |

### 模式选择对比

|  | 分布式开发 | 并行化 | 多跳 | 直接用户交互 |
|--|-----------|---------|------|-------------|
| **Subagents** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Handoffs** | — | — | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Skills** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Router** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | — | ⭐⭐⭐ |

**提示**: 可以混合模式！例如，Subagents 架构可以调用使用自定义工作流或 Router Agent 的工具。

---

## 模式 1：Subagents（子 Agent）

### 什么是 Subagents

在 Subagents 架构中，一个中心主 Agent（通常称为 supervisor）通过将子 Agent 作为工具调用来协调它们。主 Agent 决定调用哪个子 Agent、提供什么输入以及如何组合结果。

**关键特征**:
- **集中控制**: 所有路由都通过主 Agent
- **无直接用户交互**: 子 Agent 将结果返回给主 Agent
- **作为工具调用**: 子 Agent 通过工具被调用
- **并行执行**: 主 Agent 可以在单个回合中调用多个子 Agent
- **上下文隔离**: 每个子 Agent 调用在干净的上下文窗口中工作

**何时使用**:
- 有多个独立领域（日历、邮件、CRM、数据库）
- 子 Agent 不需要直接与用户对话
- 想要集中的工作流控制

### 基本实现

```javascript
import { createAgent, tool } from "langchain";
import { z } from "zod";

// 创建子 Agent
const subagent = createAgent({ 
  model: "anthropic:claude-sonnet-4-20250514", 
  tools: [...] 
});

// 将其包装为工具
const callResearchAgent = tool(
  async ({ query }) => {
    const result = await subagent.invoke({
      messages: [{ role: "user", content: query }]
    });
    return result.messages.at(-1)?.content;
  },
  {
    name: "research",
    description: "Research a topic and return findings",
    schema: z.object({ query: z.string() })
  }
);

// 主 Agent，将子 Agent 作为工具
const mainAgent = createAgent({ 
  model: "anthropic:claude-sonnet-4-20250514", 
  tools: [callResearchAgent] 
});
```

### 设计决策

#### 1. 同步 vs 异步

**同步（默认）**:
- 主 Agent 等待每个子 Agent 完成后再继续
- 适用于下一步依赖结果的场景
- 简单实现，但会阻塞对话

**异步**:
- 子 Agent 在后台运行，主 Agent 保持响应
- 使用三工具模式：启动任务、检查状态、获取结果
- 适用于独立任务

```javascript
// 异步模式示例
const startJob = tool(async ({ task }) => {
  const jobId = await jobSystem.start(task);
  return jobId;
}, { name: "start_job", ... });

const checkStatus = tool(async ({ jobId }) => {
  return await jobSystem.getStatus(jobId);
}, { name: "check_status", ... });

const getResult = tool(async ({ jobId }) => {
  return await jobSystem.getResult(jobId);
}, { name: "get_result", ... });
```

#### 2. 工具模式

**每个 Agent 一个工具**:
- 为每个子 Agent 创建单独的工具
- 对每个子 Agent 的输入/输出提供精细控制
- 更多设置，但更可定制

**单一分发工具**:
- 一个参数化工具调用任何注册的子 Agent
- 基于约定的调用：按名称选择 Agent
- 适合分布式团队开发

```javascript
// 单一分发工具示例
const taskTool = tool(
  async ({ agentName, task }) => {
    const agent = agentRegistry.get(agentName);
    const result = await agent.invoke({
      messages: [{ role: "user", content: task }]
    });
    return result.messages.at(-1)?.content;
  },
  {
    name: "task",
    description: "Invoke a specialized agent by name",
    schema: z.object({
      agentName: z.string(),
      task: z.string()
    })
  }
);
```

### Context Engineering

**子 Agent 规格**:
- **名称**: 清晰、面向行动（如 `research_agent`、`code_reviewer`）
- **描述**: 具体说明处理什么任务以及何时使用

**子 Agent 输入**:
- 自定义子 Agent 接收的上下文
- 从 State 中拉取完整消息历史、先前结果或任务元数据

```javascript
const callSubagent = tool(
  async ({query}) => {
    const state = getCurrentTaskInput<AgentState>();
    const subAgentInput = transformMessages(query, state.messages);
    const result = await subagent.invoke({
      messages: subAgentInput,
      exampleStateKey: state.exampleStateKey
    });
    return result.messages.at(-1)?.content;
  },
  { name: "subagent_name", ... }
);
```

**子 Agent 输出**:
1. **提示子 Agent**: 指定确切应返回什么
2. **在代码中格式化**: 使用 `Command` 调整或丰富响应

```javascript
const callSubagent = tool(
  async ({ query }, config) => {
    const result = await subagent.invoke({
      messages: [{ role: "user", content: query }]
    });

    return new Command({
      update: {
        exampleStateKey: result.exampleStateKey,
        messages: [
          new ToolMessage({
            content: result.messages.at(-1)?.text,
            tool_call_id: config.toolCall?.id!
          })
        ]
      }
    });
  },
  { name: "subagent_name", ... }
);
```

---

## 模式 2：Handoffs（交接）

### 什么是 Handoffs

在 Handoffs 架构中，行为基于状态动态改变。核心机制：工具更新一个持久的状态变量（如 `current_step` 或 `active_agent`），系统读取该变量以调整行为。

**关键特征**:
- **状态驱动行为**: 基于状态变量改变行为
- **基于工具的转换**: 工具更新状态变量以在状态之间移动
- **直接用户交互**: 每个状态的配置直接处理用户消息
- **持久状态**: 状态在对话回合间保持

**何时使用**:
- 需要强制顺序约束
- Agent 需要在不同状态下直接与用户对话
- 构建多阶段对话流程
- 客户支持场景（例如，在处理退款前收集保修 ID）

### 基本实现

```javascript
import { tool, ToolMessage, type ToolRuntime } from "langchain";
import { Command } from "@langchain/langgraph";
import { z } from "zod";

const transferToSpecialist = tool(
  async (_, config: ToolRuntime<typeof StateSchema>) => {
    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: "Transferred to specialist",
            tool_call_id: config.toolCallId 
          })
        ],
        currentStep: "specialist"  // 触发行为改变
      }
    });
  },
  {
    name: "transfer_to_specialist",
    description: "Transfer to the specialist agent.",
    schema: z.object({})
  }
);
```

**为什么包含 ToolMessage？** 当 LLM 调用工具时，它期望得到响应。带有匹配 `tool_call_id` 的 `ToolMessage` 完成此请求-响应周期。

### 实现方式

#### 1. 单 Agent + Middleware

单个 Agent 基于状态改变其行为。Middleware 拦截每个模型调用并动态调整系统提示和可用工具。

```javascript
const recordWarrantyStatus = tool(
  async ({ status }, config: ToolRuntime<typeof StateSchema>) => {
    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `Warranty status recorded: ${status}`,
            tool_call_id: config.toolCallId,
          }),
        ],
        warrantyStatus: status,
        currentStep: "specialist", // 更新状态触发转换
      },
    });
  },
  {
    name: "record_warranty_status",
    description: "Record warranty status and transition to next step.",
    schema: z.object({ status: z.string() }),
  }
);
```

**优点**: 简单，消息历史自然流动
**缺点**: 所有 Agent 共享相同的基础实现

#### 2. 多 Agent 子图

多个独立的 Agent 作为图中的单独节点。交接工具使用 `Command.PARENT` 在 Agent 节点间导航。

```javascript
const transferToSales = tool(
  async (_, runtime: ToolRuntime<typeof stateSchema>) => {
    const lastAiMessage = runtime.state.messages 
      .reverse()  
      .find(AIMessage.isInstance);

    const transferMessage = new ToolMessage({
      content: "Transferred to sales agent",
      tool_call_id: runtime.toolCallId,
    });

    return new Command({
      goto: "sales_agent",
      update: {
        activeAgent: "sales_agent",
        messages: [lastAiMessage, transferMessage].filter(Boolean),
      },
      graph: Command.PARENT,
    });
  },
  {
    name: "transfer_to_sales",
    description: "Transfer to the sales agent.",
    schema: z.object({}),
  }
);
```

**Context Engineering 关键点**:
- 必须包含触发交接的 `AIMessage`
- 必须包含确认交接的 `ToolMessage`
- 不要传递所有子 Agent 消息（会造成上下文混乱）

**建议**: 大多数 Handoff 用例使用单 Agent + Middleware——更简单。只有在需要定制 Agent 实现时才使用多 Agent 子图。

### 实现考虑

- **上下文过滤策略**: 每个 Agent 接收完整历史、过滤部分还是摘要？
- **工具语义**: 交接工具是否只更新路由状态还是也执行副作用？
- **Token 效率**: 平衡上下文完整性与成本

---

## 模式 3：Skills（技能）

### 什么是 Skills

在 Skills 架构中，专业能力被打包为可调用的 "技能"，增强 Agent 的行为。技能主要是 Agent 可以按需调用的提示驱动的专业化。

**关键特征**:
- **提示驱动专业化**: 技能主要由专业提示定义
- **渐进式披露**: 技能根据上下文或用户需求变得可用
- **团队分布**: 不同团队可以独立开发和维护技能
- **轻量级组合**: 技能比完整的子 Agent 更简单

**何时使用**:
- 想要具有许多可能专业化的单个 Agent
- 不需要在技能之间强制特定约束
- 不同团队需要独立开发能力

**典型例子**:
- 编码助手（不同语言或任务的技能）
- 知识库（不同领域的技能）
- 创意助手（不同格式的技能）

### 基本实现

```javascript
import { tool, createAgent } from "langchain";
import * as z from "zod";

const loadSkill = tool(
  async ({ skillName }) => {
    // 从文件/数据库加载技能内容
    const skillContent = await loadSkillFromDB(skillName);
    return skillContent;
  },
  {
    name: "load_skill",
    description: `Load a specialized skill.

Available skills:
- write_sql: SQL query writing expert
- review_legal_doc: Legal document reviewer

Returns the skill's prompt and context.`,
    schema: z.object({
      skillName: z.string().describe("Name of skill to load")
    })
  }
);

const agent = createAgent({
  model: "gpt-4o",
  tools: [loadSkill],
  systemPrompt: (
    "You are a helpful assistant. " +
    "You have access to two skills: " +
    "write_sql and review_legal_doc. " +
    "Use load_skill to access them."
  ),
});
```

### 扩展模式

**1. 动态工具注册**:
将渐进式披露与状态管理结合，在技能加载时注册新工具。

```javascript
// 加载 "database_admin" 技能时
// - 添加专业上下文
// - 注册数据库特定工具（备份、恢复、迁移）
```

**2. 层次化技能**:
技能可以在树结构中定义其他技能。

```javascript
// 加载 "data_science" 技能可能使以下子技能可用：
// - pandas_expert
// - visualization
// - statistical_analysis
```

---

## 模式 4：Router（路由器）

### 什么是 Router

在 Router 架构中，路由步骤对输入进行分类并将其定向到专业 Agent。这在有明确垂直领域时很有用。

**关键特征**:
- **路由器分解查询**
- **调用零个或多个专业 Agent（并行）**
- **将结果合成为连贯响应**

**何时使用**:
- 有明确的垂直领域（需要各自 Agent 的独立知识域）
- 需要并行查询多个源
- 想要将结果合成为组合响应

### 基本实现

**单 Agent 路由**:

```javascript
import { z } from "zod";
import { Command } from "@langchain/langgraph";

const ClassificationResult = z.object({
  query: z.string(),
  agent: z.string(),
});

function classifyQuery(query: string): z.infer<typeof ClassificationResult> {
  // 使用 LLM 分类查询并确定合适的 Agent
  // ...
}

function routeQuery(state: z.infer<typeof ClassificationResult>) {
  const classification = classifyQuery(state.query);
  return new Command({ goto: classification.agent });
}
```

**多 Agent 并行路由**:

```javascript
import { Send } from "@langchain/langgraph";

function routeQuery(state) {
  const agents = classifyQuery(state.query); // 返回多个 Agent
  
  // 并行发送到多个 Agent
  return agents.map(agent => 
    new Send(agent, { query: state.query })
  );
}
```

### 无状态 vs 有状态

**无状态**:
- 每个请求独立路由
- 无请求间记忆
- 简单、轻量级

**有状态方式 1 - 工具包装器**:
```javascript
const searchDocs = tool(
  async ({ query }) => {
    const result = await workflow.invoke({ query });
    return result.finalAnswer;
  },
  {
    name: "search_docs",
    description: "Search across multiple documentation sources",
    schema: z.object({ query: z.string() }),
  }
);

// 对话 Agent 使用路由器作为工具
const conversationalAgent = createAgent({
  model,
  tools: [searchDocs],
});
```

**有状态方式 2 - 完整持久化**:
使用持久化存储消息历史，在路由时获取并选择性包含。

**Router vs Subagents**:
- **Router**: 专用路由步骤（分类），通常不维护对话历史
- **Subagents**: 主管 Agent 动态决定调用哪些子 Agent，维护上下文

---

## 模式 5：Custom Workflow（自定义工作流）

### 什么是 Custom Workflow

在自定义工作流架构中，你使用 LangGraph 定义自己的定制执行流。你拥有对图结构的完全控制。

**关键特征**:
- **完全控制图结构**
- **混合确定性逻辑和智能行为**
- **支持顺序步骤、条件分支、循环和并行执行**
- **将其他模式嵌入为工作流中的节点**

**何时使用**:
- 标准模式不符合需求
- 需要混合确定性逻辑和智能行为
- 用例需要复杂路由或多阶段处理

### 基本实现

```javascript
import { z } from "zod";
import { createAgent } from "langchain";
import { StateGraph, START, END, MessagesZodState } from "@langchain/langgraph";

const agent = createAgent({ model: "openai:gpt-4o", tools: [...] });
const State = MessagesZodState.extend({
  query: z.string(),
});

async function agentNode(state: z.infer<typeof State>) {
  // LangGraph 节点调用 LangChain Agent
  const result = await agent.invoke({
    messages: [{ role: "user", content: state.query }]
  });
  return { answer: result.messages.at(-1)?.content };
}

// 构建简单工作流
const workflow = new StateGraph(State)
  .addNode("agent", agentNode)
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile();
```

**关键洞察**: 可以在任何 LangGraph 节点内直接调用 LangChain Agent，将自定义工作流的灵活性与预构建 Agent 的便利性结合。

### 示例：RAG 流程

结合检索和 Agent 的常见用例：

```javascript
const workflow = new StateGraph(State)
  .addNode("retrieve", retrieveNode)      // 从知识库检索
  .addNode("agent", agentNode)            // Agent 处理
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "agent")
  .addEdge("agent", END)
  .compile();
```

---

## 性能对比

### 关键指标

- **模型调用**: LLM 调用次数（更多调用 = 更高延迟和成本）
- **Token 处理**: 所有调用中的总上下文窗口使用

### 场景 1：一次性请求

**任务**: "Buy coffee" - 专业咖啡 Agent/Skill 可以调用 `buy_coffee` 工具

| 模式 | 模型调用 | 最优 |
|------|---------|------|
| Subagents | 4 |  |
| Handoffs | 3 | ✅ |
| Skills | 3 | ✅ |
| Router | 3 | ✅ |

**关键洞察**: Handoffs、Skills 和 Router 对单任务最高效（各 3 次调用）。Subagents 多一次调用因为结果通过主 Agent 流回。

### 场景 2：重复请求

**任务**: 同一对话中重复 "Buy coffee"

| 模式 | 第 1 次 | 第 2 次 | 总计 | 最优 |
|------|--------|--------|------|------|
| Subagents | 4 | 4 | 8 |  |
| Handoffs | 3 | 2 | 5 | ✅ |
| Skills | 3 | 2 | 5 | ✅ |
| Router | 3 | 3 | 6 |  |

**关键洞察**: 有状态模式（Handoffs、Skills）在重复请求上节省 40-50% 的调用。Subagents 保持一致的每请求成本（无状态设计）。

### 场景 3：多领域

**任务**: "Compare Python, JavaScript, and Rust for web development" - 每个语言 Agent/Skill 包含约 2000 token 文档

| 模式 | 调用 | Token | 最优 |
|------|------|-------|------|
| Subagents | 5 | ~9K | ✅ |
| Handoffs | 7+ | ~14K+ |  |
| Skills | 3 | ~15K |  |
| Router | 5 | ~9K | ✅ |

**关键洞察**: 对于多领域任务，支持并行执行的模式（Subagents、Router）最高效。Skills 调用较少但由于上下文积累导致 token 使用高。

### 总结

| 模式 | 单次请求 | 重复请求 | 并行执行 | 大上下文领域 |
|------|---------|---------|---------|------------|
| Subagents |  |  | ✅ | ✅ |
| Handoffs | ✅ | ✅ |  |  |
| Skills | ✅ | ✅ |  |  |
| Router | ✅ |  | ✅ | ✅ |

---

## 模式选择指南

### 按需求选择

**需要集中控制 + 并行执行**:
- 使用 **Subagents**

**需要顺序流程 + 直接用户交互**:
- 使用 **Handoffs**

**需要轻量级专业化 + 按需加载**:
- 使用 **Skills**

**需要明确分类 + 并行查询**:
- 使用 **Router**

**需要完全自定义 + 复杂逻辑**:
- 使用 **Custom Workflow**

### 按团队结构选择

**单团队，紧密集成**:
- **Handoffs** 或 **Skills**

**多团队，独立开发**:
- **Subagents** 或 **Skills**

**需要清晰边界**:
- **Subagents** 或 **Router**

### 按性能需求选择

**优先考虑低延迟**:
- **Skills** 或 **Handoffs**（单次请求）
- **Subagents** 或 **Router**（并行任务）

**优先考虑低 Token 成本**:
- **Subagents** 或 **Router**（多领域）
- **Handoffs** 或 **Skills**（重复请求）

---

## Context Engineering 要点

所有 multi-agent 模式的核心都是 Context Engineering：

### 1. 决定传递什么

- **完整历史**: 简单但可能导致上下文膨胀
- **过滤消息**: 只传递相关部分
- **摘要**: 压缩历史为简洁摘要

### 2. 何时传递

- **每次调用**: Subagents 默认行为
- **状态转换**: Handoffs 在转换时
- **按需**: Skills 在加载时

### 3. 如何格式化

- **直接传递**: 原始消息
- **转换**: 调整格式或结构
- **丰富**: 添加额外上下文

### 4. 返回什么

- **完整响应**: 所有消息和状态
- **最终消息**: 只返回结果
- **结构化数据**: 使用 `Command` 更新多个状态键

---

## 最佳实践

### 1. 从简单开始

从单 Agent 开始，只在需要时添加复杂性。

### 2. 增量测试

一次添加一个模式或组件，彻底测试。

### 3. 监控性能

跟踪：
- 模型调用次数
- Token 使用量
- 端到端延迟
- 成本

### 4. 文档化决策

记录：
- 为什么选择特定模式
- 上下文如何流动
- 每个 Agent/技能的职责

### 5. 混合模式

不要害怕组合模式：
- Subagents 可以使用 Skills
- Router 可以调用 Handoffs
- Custom Workflow 可以嵌入任何模式

### 6. 优先考虑 Context Engineering

花时间设计：
- 每个 Agent 看到什么
- 消息如何过滤和转换
- 状态如何在组件间共享

---

## 常见陷阱

### 1. 过度工程

**问题**: 为简单任务使用复杂的 multi-agent 系统
**解决方案**: 从单 Agent 开始，只在必要时扩展

### 2. 上下文膨胀

**问题**: 在 Agent 间传递过多上下文
**解决方案**: 积极过滤和总结消息

### 3. 状态不一致

**问题**: Agent 间状态不同步
**解决方案**: 使用共享状态模式和清晰的更新语义

### 4. 工具定义不清

**问题**: Agent 不知道何时调用哪个工具/子 Agent
**解决方案**: 编写清晰、具体的工具描述

### 5. 忽略性能

**问题**: 未考虑调用次数和 token 成本
**解决方案**: 提前进行性能分析，选择合适的模式

---

## 常见问题

**Q: Multi-agent 总是比单 Agent 好吗？**

A: 不。单 Agent 通常更简单、更快、更便宜。只有在有明确需求时才使用 multi-agent。

**Q: 可以混合多种模式吗？**

A: 可以！实际上这很常见。例如，Subagents 可以使用 Skills，Router 可以调用 Custom Workflow。

**Q: 哪种模式最快？**

A: 取决于场景。Skills 和 Handoffs 对单次请求快，Subagents 和 Router 对并行任务快。

**Q: 哪种模式成本最低？**

A: 同样取决于场景。查看性能对比部分了解详情。

**Q: 如何调试 multi-agent 系统？**

A: 
- 使用日志跟踪消息流
- 监控每个 Agent 的调用
- 使用 LangSmith 进行可视化追踪

**Q: 如何测试 multi-agent 系统？**

A:
- 单独测试每个 Agent/组件
- 测试集成点
- 使用端到端测试验证完整流程

---

## 相关资源

- [Agents](https://docs.langchain.com/oss/javascript/langchain/agents) - Agent 核心概念
- [Tools](https://docs.langchain.com/oss/javascript/langchain/tools) - 工具创建和使用
- [Context Engineering](https://docs.langchain.com/oss/javascript/langchain/context-engineering) - 上下文管理详解
- [LangGraph](https://docs.langchain.com/oss/javascript/langgraph/overview) - 图编排
- [Short-Term Memory](https://docs.langchain.com/oss/javascript/langchain/short-term-memory) - 状态和持久化
