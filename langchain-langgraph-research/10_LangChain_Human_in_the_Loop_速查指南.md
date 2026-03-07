# LangChain Human-in-the-Loop 速查指南

## 概览

Human-in-the-Loop (HITL) 中间件让你为 Agent 的工具调用添加人工监督。当模型提议可能需要审核的操作时——比如写入文件或执行 SQL——中间件可以暂停执行并等待决策。

**工作原理**:
1. 检查每个工具调用是否符合可配置的策略
2. 如果需要人工干预，发起中断（interrupt）暂停执行
3. 使用 LangGraph 的持久层保存图状态
4. 等待人工决策（批准、编辑或拒绝）
5. 基于决策恢复执行

**核心优势**:
- 防止高风险操作自动执行
- 提供人工审核和修改机会
- 确保关键决策的准确性
- 维护审计跟踪

---

## 三种决策类型

中间件定义了三种内置的人工响应方式：

| 决策类型 | 图标 | 说明 | 示例 |
|---------|------|------|------|
| **approve** | ✅ | 批准操作并原样执行，不做任何修改 | 完全按照草稿发送邮件 |
| **edit** | ✏️ | 修改后再执行工具调用 | 更改收件人后再发送邮件 |
| **reject** | ❌ | 拒绝工具调用，并向对话添加解释 | 拒绝邮件草稿并解释如何重写 |

**重要提示**:
- 每个工具的可用决策类型取决于你在 `interruptOn` 中的配置
- 当多个工具调用同时暂停时，每个操作都需要单独的决策
- 决策必须按照中断请求中操作的顺序提供

**编辑建议**: 编辑工具参数时要保守。对原始参数的大幅修改可能导致模型重新评估其方法，可能多次执行工具或采取意外操作。

---

## 配置中断

### 基本配置

```javascript
import { createAgent, humanInTheLoopMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph";

const agent = createAgent({
  model: "gpt-4o",
  tools: [writeFileTool, executeSQLTool, readDataTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        // 所有决策类型（approve、edit、reject）都允许
        write_file: true,
        
        // 只允许批准或拒绝，不允许编辑
        execute_sql: {
          allowedDecisions: ["approve", "reject"],
          description: "🚨 SQL execution requires DBA approval",
        },
        
        // 安全操作，不需要批准
        read_data: false,
      },
      
      // 中断消息前缀 - 与工具名称和参数组合形成完整消息
      // 例如："Tool execution pending approval: execute_sql with query='DELETE FROM...'"
      // 单个工具可以通过在中断配置中指定 "description" 来覆盖此设置
      descriptionPrefix: "Tool execution pending approval",
    }),
  ],
  
  // Human-in-the-loop 需要 checkpointer 来处理中断
  // 生产环境使用持久化 checkpointer，如 AsyncPostgresSaver
  checkpointer: new MemorySaver(),
});
```

### 配置选项

#### `interruptOn` 策略配置

每个工具可以配置为：

**1. 布尔值（简单模式）**
```javascript
{
  tool_name: true,  // 允许所有决策（approve、edit、reject）
  tool_name: false, // 不中断，直接执行
}
```

**2. 详细配置对象**
```javascript
{
  tool_name: {
    allowedDecisions: ["approve", "reject"],  // 限制可用的决策类型
    description: "Custom description for this tool", // 自定义描述
  },
}
```

#### `descriptionPrefix`

为所有中断设置默认描述前缀。单个工具可以通过 `description` 字段覆盖。

**必需**: Checkpointer 用于在中断期间持久化图状态。生产环境使用 `AsyncPostgresSaver`，测试使用 `MemorySaver`。

---

## 响应中断

### 基本流程

```javascript
import { HumanMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

// 必须提供 thread ID 来关联执行和对话线程
// 这样对话可以暂停和恢复（人工审核所需）
const config = { configurable: { thread_id: "some_id" } };

// 运行图直到遇到中断
const result = await agent.invoke(
  {
    messages: [new HumanMessage("Delete old records from the database")],
  },
  config
);

// 中断包含完整的 HITL 请求，含 action_requests 和 review_configs
console.log(result.__interrupt__);
// 输出示例：
// [
//   Interrupt(
//     value: {
//       action_requests: [
//         {
//           name: 'execute_sql',
//           arguments: { 
//             query: 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';' 
//           },
//           description: 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
//         }
//       ],
//       review_configs: [
//         {
//           action_name: 'execute_sql',
//           allowed_decisions: ['approve', 'reject']
//         }
//       ]
//     }
//   )
// ]

// 用批准决策恢复执行
await agent.invoke(
  new Command({
    resume: { 
      decisions: [{ type: "approve" }]  // 或 "reject"
    },
  }),
  config  // 使用相同的 thread ID 来恢复暂停的对话
);
```

### 三种决策详解

#### ✅ Approve（批准）

批准工具调用原样执行，不做任何修改：

```javascript
await agent.invoke(
  new Command({
    resume: {
      decisions: [
        {
          type: "approve",
        }
      ]
    }
  }),
  config  // 相同的 thread ID
);
```

**使用场景**:
- 工具调用参数完全正确
- 操作已经过审核确认安全
- 不需要任何修改

#### ✏️ Edit（编辑）

修改工具参数后再执行：

```javascript
await agent.invoke(
  new Command({
    resume: {
      decisions: [
        {
          type: "edit",
          // 提供修改后的参数
          tool_call: {
            name: "execute_sql",
            arguments: {
              query: "DELETE FROM records WHERE created_at < NOW() - INTERVAL '90 days';"
            }
          }
        }
      ]
    }
  }),
  config
);
```

**使用场景**:
- 工具调用方向正确但参数需要调整
- 需要修改查询条件、收件人等
- 微调操作以符合业务规则

**注意**: 保守地编辑参数，大幅修改可能导致意外行为。

#### ❌ Reject（拒绝）

拒绝工具调用，并添加解释给模型：

```javascript
await agent.invoke(
  new Command({
    resume: {
      decisions: [
        {
          type: "reject",
          // 提供拒绝的理由，帮助模型改进
          feedback: "Cannot delete records - retention policy requires 1 year minimum. Please revise the query to only delete records older than 1 year."
        }
      ]
    }
  }),
  config
);
```

**使用场景**:
- 操作违反业务规则或政策
- 参数明显错误或不安全
- 需要模型重新思考方法

**最佳实践**: 提供清晰的反馈说明为什么拒绝以及如何改进。

### 处理多个中断

当多个工具调用同时暂停时：

```javascript
// 假设两个工具都需要审核
const result = await agent.invoke(
  { messages: [{ role: "user", content: "Send email and update database" }] },
  config
);

// result.__interrupt__ 包含两个 action_requests
console.log(result.__interrupt__[0].value.action_requests);
// [
//   { name: "send_email", arguments: {...} },
//   { name: "update_database", arguments: {...} }
// ]

// 必须按顺序为每个操作提供决策
await agent.invoke(
  new Command({
    resume: {
      decisions: [
        { type: "approve" },  // 批准发送邮件
        { 
          type: "edit",       // 编辑数据库更新
          tool_call: {
            name: "update_database",
            arguments: { /* 修改后的参数 */ }
          }
        }
      ]
    }
  }),
  config
);
```

**关键规则**:
- 决策顺序必须与 `action_requests` 中操作的顺序一致
- 每个操作都需要一个决策
- 不能跳过或重排决策

---

## 流式处理与 HITL

使用 `stream()` 代替 `invoke()` 可以实时获取 Agent 运行进度和中断处理：

```javascript
import { Command } from "@langchain/langgraph";

const config = { configurable: { thread_id: "some_id" } };

// 流式传输 Agent 进度和 LLM tokens，直到中断
for await (const [mode, chunk] of await agent.stream(
  { messages: [{ role: "user", content: "Delete old records from the database" }] },
  { ...config, streamMode: ["updates", "messages"] }
)) {
  if (mode === "messages") {
    // LLM token
    const [token, metadata] = chunk;
    if (token.content) {
      process.stdout.write(token.content);
    }
  } else if (mode === "updates") {
    // 检查中断
    if ("__interrupt__" in chunk) {
      console.log(`\n\nInterrupt: ${JSON.stringify(chunk.__interrupt__)}`);
      
      // 这里可以暂停流，等待人工决策
      // 然后在下面的流中恢复
    }
  }
}

// 人工决策后，用流式方式恢复
for await (const [mode, chunk] of await agent.stream(
  new Command({ resume: { decisions: [{ type: "approve" }] } }),
  { ...config, streamMode: ["updates", "messages"] }
)) {
  if (mode === "messages") {
    const [token, metadata] = chunk;
    if (token.content) {
      process.stdout.write(token.content);
    }
  }
}
```

**流式模式**:
- `"messages"`: LLM 生成的 token
- `"updates"`: Agent 状态更新，包括中断

---

## 执行生命周期

理解 HITL 中间件如何集成到 Agent 生命周期：

1. **模型调用**: Agent 调用模型生成响应
2. **中间件检查**: 中间件检查响应中的工具调用（`after_model` 钩子）
3. **触发中断**: 如果任何调用需要人工输入，中间件构建 `HITLRequest` 并调用 `interrupt()`
4. **等待决策**: Agent 等待人工决策，状态被持久化
5. **恢复执行**: 基于 `HITLResponse` 决策：
   - **approve**: 执行批准的调用
   - **edit**: 执行修改后的调用
   - **reject**: 为拒绝的调用合成 `ToolMessage`，继续执行

**关键点**:
- 中断发生在模型生成响应之后、工具执行之前
- 使用 `after_model` 钩子拦截工具调用
- 状态持久化确保可以安全暂停和恢复

---

## 实际应用场景

### 场景 1：数据库操作审核

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [readDBTool, updateDBTool, deleteDBTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        read_db: false,              // 读取无需批准
        update_db: true,              // 更新需要批准
        delete_db: {                  // 删除需要严格控制
          allowedDecisions: ["approve", "reject"],
          description: "⚠️ Delete operation requires manager approval",
        },
      },
    }),
  ],
  checkpointer: new MemorySaver(),
});
```

### 场景 2：邮件和通信审核

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [draftEmailTool, sendEmailTool, sendSlackTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        draft_email: false,           // 草稿无需批准
        send_email: true,              // 发送需要批准和编辑
        send_slack: {
          allowedDecisions: ["approve", "edit", "reject"],
          description: "Review before posting to Slack",
        },
      },
    }),
  ],
  checkpointer: new MemorySaver(),
});
```

### 场景 3：金融交易审核

```javascript
const agent = createAgent({
  model: "gpt-4o",
  tools: [checkBalanceTool, transferFundsTool, investTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        check_balance: false,         // 查询无需批准
        transfer_funds: {              // 转账需要严格审核
          allowedDecisions: ["approve", "reject"],
          description: "💰 Financial transaction - requires approval",
        },
        invest: {
          allowedDecisions: ["approve", "edit", "reject"],
          description: "📈 Investment decision - review recommended",
        },
      },
    }),
  ],
  checkpointer: new AsyncPostgresSaver(...),  // 生产环境用持久化
});
```

---

## 最佳实践

### 1. 决策类型选择

- **只读操作**: `false` - 不需要中断
- **低风险写操作**: `["approve", "edit", "reject"]` - 允许全部决策
- **高风险操作**: `["approve", "reject"]` - 不允许编辑，强制重新生成

### 2. 描述清晰

```javascript
{
  execute_payment: {
    allowedDecisions: ["approve", "reject"],
    description: "💳 Payment of $${amount} to ${recipient}\n⚠️ This action cannot be undone",
  },
}
```

- 使用表情符号增强可视性
- 说明操作的影响
- 提供足够的上下文便于决策

### 3. 反馈质量

拒绝时提供具体的反馈：

```javascript
{
  type: "reject",
  feedback: "Cannot approve: Transaction amount exceeds daily limit of $10,000. Current request: $15,000. Please split into multiple transactions or request manager override."
}
```

### 4. Checkpointer 选择

- **开发/测试**: `MemorySaver` - 简单但不持久
- **生产**: `AsyncPostgresSaver` - 持久化存储，支持分布式
- **需求**: 根据业务需求选择合适的存储后端

### 5. 错误处理

```javascript
try {
  const result = await agent.invoke(input, config);
  
  if (result.__interrupt__) {
    // 处理中断
    const decisions = await getUserDecisions(result.__interrupt__);
    await agent.invoke(
      new Command({ resume: { decisions } }),
      config
    );
  }
} catch (error) {
  console.error("Agent execution failed:", error);
  // 处理错误并可能重试
}
```

---

## 自定义 HITL 逻辑

对于更专业化的工作流，可以使用 `interrupt` 原语和 `middleware` 抽象直接构建自定义 HITL 逻辑。

### 自定义中断示例

```javascript
import { createMiddleware } from "langchain";
import { interrupt } from "@langchain/langgraph";

const customHITLMiddleware = createMiddleware({
  name: "CustomHITL",
  afterModel: {
    hook: (state) => {
      const lastMessage = state.messages[state.messages.length - 1];
      
      // 自定义逻辑：检查特定条件
      if (lastMessage.tool_calls) {
        for (const toolCall of lastMessage.tool_calls) {
          // 自定义条件判断
          if (shouldRequireApproval(toolCall)) {
            // 触发中断
            const decision = interrupt({
              action: toolCall,
              reason: "Custom approval required",
            });
            
            // 处理决策
            if (decision === "reject") {
              // 自定义拒绝处理
            }
          }
        }
      }
    },
  },
});
```

---

## 常见问题

**Q: 忘记提供 thread ID 会怎样？**

A: 会抛出错误。HITL 需要 thread ID 来持久化和恢复状态。

**Q: 可以动态改变中断策略吗？**

A: 可以在运行时条件性地配置中间件，或使用自定义 HITL 逻辑。

**Q: 多个审核者如何协调？**

A: 使用持久化 checkpointer 和外部协调系统。每个审核者可以查询中断并提交决策。

**Q: 中断会超时吗？**

A: 不会自动超时。你需要在应用层实现超时逻辑。

**Q: 如何测试 HITL 工作流？**

A: 使用 `MemorySaver` 进行单元测试，模拟不同的决策场景。

**Q: 能否在中断时修改 State？**

A: 可以，但建议通过决策对象完成修改，保持逻辑清晰。

---

## 相关资源

- [Middleware 文档](https://docs.langchain.com/oss/javascript/langchain/middleware) - 完整的中间件指南
- [LangGraph 中断](https://docs.langchain.com/oss/javascript/langgraph/interrupts) - 中断原语详解
- [持久化层](https://docs.langchain.com/oss/javascript/langgraph/persistence) - Checkpointer 配置
- [流式处理](https://docs.langchain.com/oss/javascript/langchain/streaming) - 流式模式指南
