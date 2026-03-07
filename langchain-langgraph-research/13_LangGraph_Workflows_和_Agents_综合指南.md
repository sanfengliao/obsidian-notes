# LangGraph Workflows 和 Agents 综合指南

## 概览

本指南介绍常见的 Workflow 和 Agent 模式。

**核心区别**:
- **Workflows（工作流）**: 有预定的代码路径，按照特定顺序操作
- **Agents（智能体）**: 动态的，定义自己的流程和工具使用

LangGraph 为构建 Agent 和 Workflow 提供了多项优势：
- **持久化**: 状态在执行间保存
- **流式处理**: 实时获取输出
- **调试支持**: 可视化执行流程
- **部署**: 简化生产环境部署

---

## 环境设置

### 安装依赖

```bash
npm install @langchain/langgraph @langchain/core
```

### 初始化 LLM

```javascript
import { ChatAnthropic } from "@langchain/anthropic";

const llm = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  apiKey: "<your_anthropic_key>"
});
```

---

## LLM 和增强功能

Workflows 和 Agents 都基于 LLM 及其各种增强功能。常见增强包括：
- **工具调用**: 让 LLM 调用外部函数
- **结构化输出**: 确保输出符合特定格式
- **短期记忆**: 维护对话历史

### 结构化输出示例

```javascript
import * as z from "zod";

// 定义输出 schema
const SearchQuery = z.object({
  search_query: z.string().describe("Query that is optimized web search."),
  justification: z
    .string()
    .describe("Why this query is relevant to the user's request."),
});

// 增强 LLM
const structuredLlm = llm.withStructuredOutput(SearchQuery);

// 调用
const output = await structuredLlm.invoke(
  "How does Calcium CT score relate to high cholesterol?"
);
```

### 工具调用示例

```javascript
import { tool } from "@langchain/core/tools";
import * as z from "zod";

const multiply = tool(
  ({ a, b }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

// 绑定工具到 LLM
const llmWithTools = llm.bindTools([multiply]);

// 调用
const msg = await llmWithTools.invoke("What is 2 times 3?");
console.log(msg.tool_calls);
```

---

## Workflow 模式

### 模式 1：提示链（Prompt Chaining）

**概念**: 每个 LLM 调用处理前一个调用的输出。

**适用场景**:
- 可分解为更小、可验证步骤的任务
- 将文档翻译成不同语言
- 验证生成内容的一致性

#### 实现示例：笑话生成器

```javascript
import { StateGraph, StateSchema, GraphNode, ConditionalEdgeRouter } from "@langchain/langgraph";
import { z } from "zod/v4";

// 定义状态
const State = new StateSchema({
  topic: z.string(),
  joke: z.string(),
  improvedJoke: z.string(),
  finalJoke: z.string(),
});

// 节点 1：生成初始笑话
const generateJoke: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(`Write a short joke about ${state.topic}`);
  return { joke: msg.content };
};

// 质量检查：是否有 punchline
const checkPunchline: ConditionalEdgeRouter<typeof State, "improveJoke"> = (state) => {
  if (state.joke?.includes("?") || state.joke?.includes("!")) {
    return "Pass";
  }
  return "Fail";
};

// 节点 2：改进笑话
const improveJoke: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${state.joke}`
  );
  return { improvedJoke: msg.content };
};

// 节点 3：最终润色
const polishJoke: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${state.improvedJoke}`
  );
  return { finalJoke: msg.content };
};

// 构建工作流
const chain = new StateGraph(State)
  .addNode("generateJoke", generateJoke)
  .addNode("improveJoke", improveJoke)
  .addNode("polishJoke", polishJoke)
  .addEdge("__start__", "generateJoke")
  .addConditionalEdges("generateJoke", checkPunchline, {
    Pass: "improveJoke",
    Fail: "__end__"
  })
  .addEdge("improveJoke", "polishJoke")
  .addEdge("polishJoke", "__end__")
  .compile();

// 执行
const state = await chain.invoke({ topic: "cats" });
console.log("Initial joke:", state.joke);
if (state.improvedJoke) {
  console.log("Improved joke:", state.improvedJoke);
  console.log("Final joke:", state.finalJoke);
} else {
  console.log("Joke failed quality gate - no punchline detected!");
}
```

**流程图**:
```
START → generateJoke → [有 punchline?] → improveJoke → polishJoke → END
                            ↓ [无]
                           END
```

---

### 模式 2：并行化（Parallelization）

**概念**: 多个 LLM 同时处理任务。

**适用场景**:
- 拆分子任务并行运行以提高速度
- 多次运行任务以检查不同输出

**典型例子**:
- 一个子任务处理文档关键词，另一个检查格式错误
- 多次运行任务根据不同标准评分

#### 实现示例：内容生成器

```javascript
import { StateGraph, StateSchema, GraphNode } from "@langchain/langgraph";
import * as z from "zod";

// 定义状态
const State = new StateSchema({
  topic: z.string(),
  joke: z.string(),
  story: z.string(),
  poem: z.string(),
  combinedOutput: z.string(),
});

// 并行节点 1：生成笑话
const callLlm1: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(`Write a joke about ${state.topic}`);
  return { joke: msg.content };
};

// 并行节点 2：生成故事
const callLlm2: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(`Write a story about ${state.topic}`);
  return { story: msg.content };
};

// 并行节点 3：生成诗歌
const callLlm3: GraphNode<typeof State> = async (state) => {
  const msg = await llm.invoke(`Write a poem about ${state.topic}`);
  return { poem: msg.content };
};

// 聚合节点：合并所有输出
const aggregator: GraphNode<typeof State> = async (state) => {
  const combined = `Here's a story, joke, and poem about ${state.topic}!\n\n` +
    `STORY:\n${state.story}\n\n` +
    `JOKE:\n${state.joke}\n\n` +
    `POEM:\n${state.poem}`;
  return { combinedOutput: combined };
};

// 构建工作流
const parallelWorkflow = new StateGraph(State)
  .addNode("callLlm1", callLlm1)
  .addNode("callLlm2", callLlm2)
  .addNode("callLlm3", callLlm3)
  .addNode("aggregator", aggregator)
  // 并行执行三个节点
  .addEdge("__start__", "callLlm1")
  .addEdge("__start__", "callLlm2")
  .addEdge("__start__", "callLlm3")
  // 所有节点完成后聚合
  .addEdge("callLlm1", "aggregator")
  .addEdge("callLlm2", "aggregator")
  .addEdge("callLlm3", "aggregator")
  .addEdge("aggregator", "__end__")
  .compile();

// 执行
const result = await parallelWorkflow.invoke({ topic: "cats" });
console.log(result.combinedOutput);
```

**流程图**:
```
           ┌→ callLlm1 →┐
START → ├→ callLlm2 →├→ aggregator → END
           └→ callLlm3 →┘
```

---

### 模式 3：路由（Routing）

**概念**: 处理输入后将其定向到上下文特定任务。

**适用场景**:
- 为复杂任务定义专门流程
- 产品问题路由到定价、退款、退货等不同流程

#### 实现示例：内容类型路由器

```javascript
import { StateGraph, StateSchema, GraphNode, ConditionalEdgeRouter } from "@langchain/langgraph";
import * as z from "zod";

// 路由 schema
const routeSchema = z.object({
  step: z.enum(["poem", "story", "joke"]).describe(
    "The next step in the routing process"
  ),
});

const router = llm.withStructuredOutput(routeSchema);

// 定义状态
const State = new StateSchema({
  input: z.string(),
  decision: z.string(),
  output: z.string(),
});

// 节点 1：写故事
const llmCall1: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke([{
    role: "system",
    content: "You are an expert storyteller.",
  }, {
    role: "user",
    content: state.input
  }]);
  return { output: result.content };
};

// 节点 2：写笑话
const llmCall2: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke([{
    role: "system",
    content: "You are an expert comedian.",
  }, {
    role: "user",
    content: state.input
  }]);
  return { output: result.content };
};

// 节点 3：写诗歌
const llmCall3: GraphNode<typeof State> = async (state) => {
  const result = await llm.invoke([{
    role: "system",
    content: "You are an expert poet.",
  }, {
    role: "user",
    content: state.input
  }]);
  return { output: result.content };
};

// 路由节点：决定去哪里
const llmCallRouter: GraphNode<typeof State> = async (state) => {
  const decision = await router.invoke([
    {
      role: "system",
      content: "Route the input to story, joke, or poem based on the user's request."
    },
    {
      role: "user",
      content: state.input
    },
  ]);

  return { decision: decision.step };
};

// 条件边：根据决策路由
const routeDecision: ConditionalEdgeRouter<typeof State, "llmCall1" | "llmCall2" | "llmCall3"> = (state) => {
  if (state.decision === "story") {
    return "llmCall1";
  } else if (state.decision === "joke") {
    return "llmCall2";
  } else {
    return "llmCall3";
  }
};

// 构建工作流
const routerWorkflow = new StateGraph(State)
  .addNode("llmCall1", llmCall1)
  .addNode("llmCall2", llmCall2)
  .addNode("llmCall3", llmCall3)
  .addNode("llmCallRouter", llmCallRouter)
  .addEdge("__start__", "llmCallRouter")
  .addConditionalEdges(
    "llmCallRouter",
    routeDecision,
    ["llmCall1", "llmCall2", "llmCall3"],
  )
  .addEdge("llmCall1", "__end__")
  .addEdge("llmCall2", "__end__")
  .addEdge("llmCall3", "__end__")
  .compile();

// 执行
const state = await routerWorkflow.invoke({
  input: "Write me a joke about cats"
});
console.log(state.output);
```

**流程图**:
```
START → Router → [分类] → Story/Joke/Poem → END
```

---

### 模式 4：编排器-工作者（Orchestrator-Worker）

**概念**: 编排器分解任务并委派给工作者，然后综合结果。

**适用场景**:
- 子任务无法预定义（如并行化）
- 需要处理未知数量的文件或文档
- 编写代码或更新多个文件

**编排器职责**:
1. 将任务分解为子任务
2. 委派子任务给工作者
3. 综合工作者输出为最终结果

#### 使用 Send API 创建工作者

LangGraph 提供内置的 `Send` API 动态创建工作者节点。

```javascript
import { StateGraph, StateSchema, ReducedValue, GraphNode, Send } from "@langchain/langgraph";
import * as z from "zod";

// 主状态
const State = new StateSchema({
  topic: z.string(),
  sections: z.array(z.custom<SectionsSchema>()),
  completedSections: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (a, b) => a.concat(b) }
  ),
  finalReport: z.string(),
});

// 工作者状态
const WorkerState = new StateSchema({
  section: z.custom<SectionsSchema>(),
  completedSections: new ReducedValue(
    z.array(z.string()).default(() => []),
    { reducer: (a, b) => a.concat(b) }
  ),
});

// 编排器节点：生成计划
const orchestrator: GraphNode<typeof State> = async (state) => {
  const reportSections = await planner.invoke([
    { role: "system", content: "Generate a plan for the report." },
    { role: "user", content: `Here is the report topic: ${state.topic}` },
  ]);

  return { sections: reportSections.sections };
};

// 工作者节点：写章节
const llmCall: GraphNode<typeof WorkerState> = async (state) => {
  const section = await llm.invoke([
    {
      role: "system",
      content: "Write a report section following the provided name and description.",
    },
    {
      role: "user",
      content: `Section: ${state.section.name}, Description: ${state.section.description}`,
    },
  ]);

  return { completedSections: [section.content] };
};

// 综合节点：合并所有章节
const synthesizer: GraphNode<typeof State> = async (state) => {
  const completedReportSections = state.completedSections.join("\n\n---\n\n");
  return { finalReport: completedReportSections };
};

// 使用 Send API 分配工作者
const assignWorkers: ConditionalEdgeRouter<typeof State, "llmCall"> = (state) => {
  return state.sections.map((section) =>
    new Send("llmCall", { section })
  );
};

// 构建工作流
const orchestratorWorker = new StateGraph(State)
  .addNode("orchestrator", orchestrator)
  .addNode("llmCall", llmCall)
  .addNode("synthesizer", synthesizer)
  .addEdge("__start__", "orchestrator")
  .addConditionalEdges(
    "orchestrator",
    assignWorkers,
    ["llmCall"]
  )
  .addEdge("llmCall", "synthesizer")
  .addEdge("synthesizer", "__end__")
  .compile();

// 执行
const state = await orchestratorWorker.invoke({
  topic: "Create a report on LLM scaling laws"
});
console.log(state.finalReport);
```

**关键点**:
- 每个工作者有自己的状态
- 所有工作者输出写入共享状态键
- `Send` API 动态创建工作者并发送特定输入

---

### 模式 5：评估器-优化器（Evaluator-Optimizer）

**概念**: 一个 LLM 创建响应，另一个评估该响应。如果需要改进，提供反馈并重新创建。

**适用场景**:
- 任务有特定成功标准但需要迭代
- 翻译文本以保持含义
- 生成符合特定质量标准的内容

#### 实现示例：笑话评估器

```javascript
import { StateGraph, StateSchema, GraphNode, ConditionalEdgeRouter } from "@langchain/langgraph";
import * as z from "zod";

// 定义状态
const State = new StateSchema({
  joke: z.string(),
  topic: z.string(),
  feedback: z.string(),
  funnyOrNot: z.string(),
});

// 评估 schema
const feedbackSchema = z.object({
  grade: z.enum(["funny", "not funny"]),
  feedback: z.string().describe(
    "If the joke is not funny, provide feedback on how to improve it."
  ),
});

const evaluator = llm.withStructuredOutput(feedbackSchema);

// 生成器节点：创建笑话
const llmCallGenerator: GraphNode<typeof State> = async (state) => {
  let msg;
  if (state.feedback) {
    msg = await llm.invoke(
      `Write a joke about ${state.topic} but take into account the feedback: ${state.feedback}`
    );
  } else {
    msg = await llm.invoke(`Write a joke about ${state.topic}`);
  }
  return { joke: msg.content };
};

// 评估器节点：评估笑话
const llmCallEvaluator: GraphNode<typeof State> = async (state) => {
  const grade = await evaluator.invoke(`Grade the joke ${state.joke}`);
  return { funnyOrNot: grade.grade, feedback: grade.feedback };
};

// 路由：接受或重试
const routeJoke: ConditionalEdgeRouter<typeof State, "llmCallGenerator"> = (state) => {
  if (state.funnyOrNot === "funny") {
    return "Accepted";
  } else {
    return "Rejected + Feedback";
  }
};

// 构建工作流
const optimizerWorkflow = new StateGraph(State)
  .addNode("llmCallGenerator", llmCallGenerator)
  .addNode("llmCallEvaluator", llmCallEvaluator)
  .addEdge("__start__", "llmCallGenerator")
  .addEdge("llmCallGenerator", "llmCallEvaluator")
  .addConditionalEdges(
    "llmCallEvaluator",
    routeJoke,
    {
      "Accepted": "__end__",
      "Rejected + Feedback": "llmCallGenerator",
    }
  )
  .compile();

// 执行
const state = await optimizerWorkflow.invoke({ topic: "Cats" });
console.log(state.joke);
```

**流程图**:
```
START → Generator → Evaluator → [好笑?] → END
                        ↑            ↓ [不好笑]
                        └────────────┘ (带反馈)
```

---

## Agent 模式

**概念**: Agent 使用工具执行操作，在持续反馈循环中运行。

**特点**:
- 比 Workflow 更具自主性
- 可以决定使用哪些工具以及如何解决问题
- 适合问题和解决方案不可预测的情况

### 基本 Agent 实现

```javascript
import { tool } from "@langchain/core/tools";
import { StateGraph, StateSchema, MessagesValue, GraphNode, ConditionalEdgeRouter } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import * as z from "zod";

// 定义工具
const multiply = tool(
  ({ a, b }) => a * b,
  {
    name: "multiply",
    description: "Multiply two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  ({ a, b }) => a + b,
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

const divide = tool(
  ({ a, b }) => a / b,
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);

// 绑定工具
const tools = [add, multiply, divide];
const llmWithTools = llm.bindTools(tools);

// 定义状态
const State = new StateSchema({
  messages: MessagesValue,
});

// LLM 节点：决定是否调用工具
const llmCall: GraphNode<typeof State> = async (state) => {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    },
    ...state.messages
  ]);

  return { messages: [result] };
};

// 工具节点：执行工具
const toolNode = new ToolNode(tools);

// 路由：继续还是结束
const shouldContinue: ConditionalEdgeRouter<typeof State, "toolNode"> = (state) => {
  const lastMessage = state.messages.at(-1);

  if (lastMessage?.tool_calls?.length) {
    return "toolNode";
  }
  return "__end__";
};

// 构建 Agent
const agentBuilder = new StateGraph(State)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges(
    "llmCall",
    shouldContinue,
    ["toolNode", "__end__"]
  )
  .addEdge("toolNode", "llmCall")
  .compile();

// 执行
const messages = [{
  role: "user",
  content: "Add 3 and 4."
}];
const result = await agentBuilder.invoke({ messages });
console.log(result.messages);
```

**Agent 循环**:
```
START → LLM → [工具调用?] → Tools → LLM
                  ↓ [无工具]
                 END
```

---

## 模式对比总结

| 模式 | 控制流 | 适用场景 | 复杂度 |
|------|--------|---------|--------|
| **提示链** | 顺序 | 可分解步骤 | 低 |
| **并行化** | 并行 | 独立子任务 | 中 |
| **路由** | 条件分支 | 明确分类 | 中 |
| **编排器-工作者** | 动态并行 | 未知数量子任务 | 高 |
| **评估器-优化器** | 循环反馈 | 迭代改进 | 中 |
| **Agent** | 自主决策 | 不可预测问题 | 高 |

---

## 最佳实践

### 选择合适的模式

1. **确定性强** → Workflow
2. **需要灵活性** → Agent
3. **可分解任务** → 提示链
4. **独立并行** → 并行化
5. **需要分类** → 路由
6. **动态子任务** → 编排器-工作者
7. **需要迭代** → 评估器-优化器

### 性能优化

- **并行化**: 尽可能并行执行独立任务
- **早期退出**: 使用条件边提前结束不必要的处理
- **状态管理**: 只保存必要的状态信息

### 调试技巧

- 使用 LangSmith 追踪执行
- 记录每个节点的输入输出
- 可视化图结构

---

## 常见问题

**Q: 何时使用 Workflow vs Agent？**

A: 
- **Workflow**: 流程确定、步骤可预测
- **Agent**: 需要根据情况动态决策

**Q: 能否混合使用模式？**

A: 可以！例如，Agent 可以调用 Workflow 作为工具。

**Q: 如何处理循环？**

A: 使用条件边和状态检查来决定何时退出循环。

**Q: 并行化有性能限制吗？**

A: 取决于 API 速率限制和并发设置。

**Q: 如何在 Workflow 中添加人工审核？**

A: 使用 interrupts 功能在特定节点暂停执行。

---

## 相关资源

- [LangGraph Quickstart](https://docs.langchain.com/oss/javascript/langgraph/quickstart)
- [Graph API 参考](https://docs.langchain.com/oss/javascript/langgraph/graph-api)
- [持久化](https://docs.langchain.com/oss/javascript/langgraph/persistence)
- [流式处理](https://docs.langchain.com/oss/javascript/langgraph/streaming)
- [部署](https://docs.langchain.com/oss/javascript/langgraph/deploy)
