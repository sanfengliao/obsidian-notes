# LangChain Guardrails 速查指南

## 概览

Guardrails 帮你为 Agent 应用构建安全防护，通过在执行关键点验证和过滤内容来预防问题。可以检测敏感信息、强制实施内容策略、验证输出质量，以及阻止不安全行为。

**常见应用场景**:
- 防止 PII（个人隐私信息）泄露
- 检测和阻止提示词注入攻击
- 阻止不当或有害的内容
- 强制执行业务规则和合规要求
- 验证输出质量和准确性

Guardrails 通过 Middleware 实现，可以在 Agent 启动前、完成后、或在模型/工具调用周围拦截执行。

---

## 两种实现方式

### 确定性 Guardrails（规则型）

使用基于规则的逻辑，如正则表达式、关键词匹配或显式检查。

**优点**:
- 速度快
- 行为可预测
- 成本低

**缺点**:
- 可能遗漏微妙的违规行为

### 模型型 Guardrails（学习型）

使用 LLM 或分类器来评估内容，具有语义理解能力。

**优点**:
- 能捕捉规则容易漏掉的细微问题

**缺点**:
- 速度相对较慢
- 成本较高

---

## 内置 Guardrails

### 1. PII 检测

LangChain 提供了内置中间件来检测和处理对话中的个人身份信息 (PII)。

**适用场景**:
- 医疗保健和金融应用（需要合规）
- 客户服务 Agent（需要清理日志）
- 任何处理敏感用户数据的应用

#### PII 检测策略

| 策略 | 说明 | 示例 |
|------|------|------|
| `redact` | 替换为 [REDACTED_{类型}] | [REDACTED_EMAIL] |
| `mask` | 部分隐藏（如最后 4 位） | ****-****-****-1234 |
| `hash` | 替换为确定性哈希值 | a8f5f167... |
| `block` | 检测到就抛出异常 | 错误被抛出 |

#### 使用示例

```javascript
import { createAgent, piiRedactionMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [customerServiceTool, emailTool],
  middleware: [
    // 在发送给模型前，隐去用户输入中的邮箱
    piiRedactionMiddleware({
      piiType: "email",
      strategy: "redact",
      applyToInput: true,
    }),
    // 隐去用户输入中的信用卡号
    piiRedactionMiddleware({
      piiType: "credit_card",
      strategy: "mask",
      applyToInput: true,
    }),
    // 检测到 API key 就直接阻止
    piiRedactionMiddleware({
      piiType: "api_key",
      detector: /sk-[a-zA-Z0-9]{32}/,
      strategy: "block",
      applyToInput: true,
    }),
  ],
});

// 当用户提供 PII 时，会按照策略进行处理
const result = await agent.invoke({
  messages: [{
    role: "user",
    content: "My email is john.doe@example.com and card is 5105-1051-0510-5100"
  }]
});
```

**支持的 PII 类型**: 邮箱、信用卡、IP 地址等。更多详情见 Middleware 文档。

#### 如何使用

1. 导入 `piiRedactionMiddleware`
2. 配置检测类型（`piiType`）
3. 选择处理策略（`strategy`）
4. 设置应用范围（`applyToInput`/`applyToOutput`）
5. 添加到 Agent 的 middleware 数组

### 2. 人工审核（Human-in-the-Loop）

LangChain 提供了内置中间件，在执行敏感操作之前要求人工批准。这是处理高风险决策最有效的 guardrail。

**适用场景**:
- 金融交易和转账
- 删除或修改生产数据
- 向外部方发送通信
- 任何有重大业务影响的操作

#### 使用示例

```javascript
import { createAgent, humanInTheLoopMiddleware } from "langchain";
import { MemorySaver, Command } from "@langchain/langgraph";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, sendEmailTool, deleteDatabaseTool],
  middleware: [
    humanInTheLoopMiddleware({
      interruptOn: {
        // 需要批准的敏感操作
        send_email: { 
          allowAccept: true, 
          allowEdit: true, 
          allowRespond: true 
        },
        delete_database: { 
          allowAccept: true, 
          allowEdit: true, 
          allowRespond: true 
        },
        // 安全操作自动批准
        search: false,
      }
    }),
  ],
  checkpointer: new MemorySaver(),
});

// Human-in-the-loop 需要用 thread ID 来保证状态持久化
const config = { configurable: { thread_id: "some_id" } };

// Agent 会在执行敏感工具前暂停，等待批准
let result = await agent.invoke(
  { messages: [{ role: "user", content: "Send an email to the team" }] },
  config
);

// 用户批准后恢复执行
result = await agent.invoke(
  new Command({ resume: { decisions: [{ type: "approve" }] } }),
  config  // 使用相同的 thread ID 来恢复对话
);
```

**关键特性**:
- `allowAccept`: 允许直接批准
- `allowEdit`: 允许修改请求后批准
- `allowRespond`: 允许提供自定义响应

---

## 自定义 Guardrails

对于更复杂的需求，可以创建自定义中间件在 Agent 执行前后运行，实现完整的内容过滤和安全检查控制。

### Agent 执行前的 Guardrail

在每个调用开始时验证请求，适合会话级别的检查，如认证、速率限制或在任何处理前阻止不当请求。

```javascript
import { createMiddleware, AIMessage } from "langchain";

const contentFilterMiddleware = (bannedKeywords: string[]) => {
  const keywords = bannedKeywords.map(kw => kw.toLowerCase());

  return createMiddleware({
    name: "ContentFilterMiddleware",
    beforeAgent: {
      hook: (state) => {
        // 获取第一条用户消息
        if (!state.messages || state.messages.length === 0) {
          return;
        }

        const firstMessage = state.messages[0];
        if (firstMessage._getType() !== "human") {
          return;
        }

        const content = firstMessage.content.toString().toLowerCase();

        // 检查禁用关键词
        for (const keyword of keywords) {
          if (content.includes(keyword)) {
            // 在任何处理前阻止执行
            return {
              messages: [
                new AIMessage(
                  "I cannot process requests containing inappropriate content. Please rephrase your request."
                )
              ],
              jumpTo: "end",
            };
          }
        }

        return;
      },
      canJumpTo: ['end']
    }
  });
};

// 使用自定义 guardrail
import { createAgent } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, calculatorTool],
  middleware: [
    contentFilterMiddleware(["hack", "exploit", "malware"]),
  ],
});

// 这个请求会在任何处理前被阻止
const result = await agent.invoke({
  messages: [{ role: "user", content: "How do I hack into a database?" }]
});
```

**使用场景**:
- 输入内容过滤
- 认证和授权检查
- 请求速率限制
- 黑名单/白名单验证

### Agent 执行后的 Guardrail

在返回给用户前验证最终输出，适合基于模型的安全检查、质量验证或对完整 Agent 响应的最终合规扫描。

```javascript
import { createMiddleware, AIMessage, initChatModel } from "langchain";

const safetyGuardrailMiddleware = () => {
  const safetyModel = initChatModel("gpt-4o-mini");

  return createMiddleware({
    name: "SafetyGuardrailMiddleware",
    afterAgent: {
      hook: async (state) => {
        // 获取最终 AI 响应
        if (!state.messages || state.messages.length === 0) {
          return;
        }

        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage._getType() !== "ai") {
          return;
        }

        // 使用模型来评估安全性
        const safetyPrompt = `Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: ${lastMessage.content.toString()}`;

        const result = await safetyModel.invoke([
          { role: "user", content: safetyPrompt }
        ]);

        if (result.content.toString().includes("UNSAFE")) {
          return {
            messages: [
              new AIMessage(
                "I cannot provide that response. Please rephrase your request."
              )
            ],
            jumpTo: "end",
          };
        }

        return;
      },
      canJumpTo: ['end']
    }
  });
};

// 使用安全 guardrail
import { createAgent } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, calculatorTool],
  middleware: [safetyGuardrailMiddleware()],
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "How do I make explosives?" }]
});
```

**使用场景**:
- 模型型安全检查
- 输出质量验证
- 最终合规检查
- 结果准确性验证

### 堆叠多个 Guardrails

可以添加多个 guardrail 到 middleware 数组中，它们会按顺序执行，让你构建分层的安全防护：

```javascript
import { createAgent, piiRedactionMiddleware, humanInTheLoopMiddleware } from "langchain";

const agent = createAgent({
  model: "gpt-4o",
  tools: [searchTool, sendEmailTool],
  middleware: [
    // 第 1 层：确定性输入过滤（Agent 前）
    contentFilterMiddleware(["hack", "exploit"]),

    // 第 2 层：PII 保护（模型前后）
    piiRedactionMiddleware({
      piiType: "email",
      strategy: "redact",
      applyToInput: true,
    }),
    piiRedactionMiddleware({
      piiType: "email",
      strategy: "redact",
      applyToOutput: true,
    }),

    // 第 3 层：敏感工具的人工审核
    humanInTheLoopMiddleware({
      interruptOn: {
        send_email: { allowAccept: true, allowEdit: true, allowRespond: true },
      }
    }),

    // 第 4 层：模型型安全检查（Agent 后）
    safetyGuardrailMiddleware(),
  ],
});
```

**分层防护的优势**:
- 多层次的防线
- 快速的确定性检查在前
- 更智能的模型检查在后
- 充分利用人工判断

---

## Guardrails 设计最佳实践

### 1. 优先级顺序

- **第 1 层**：快速的确定性检查（关键词、黑名单）
- **第 2 层**：数据保护（PII 隐去、掩码）
- **第 3 层**：人工审核（敏感操作）
- **第 4 层**：智能验证（模型型检查）

### 2. 性能考虑

- 将快速的规则检查放在前面，减少后续处理
- 模型型检查成本较高，只用于关键的最终验证
- 缓存检查结果以避免重复计算

### 3. 错误处理

- 使用 `jumpTo: 'end'` 让 guardrail 直接跳过中间步骤
- 提供有意义的错误消息给用户
- 记录所有被阻止的请求以便审计

### 4. 合规性

- 记录所有敏感操作（PII 检测、人工审核）
- 确保 Checkpointer 持久化以满足审计要求
- 定期审查和更新规则和策略

### 5. 用户体验

- 给用户清晰的反馈为什么请求被拒绝
- 人工审核时提供足够的上下文
- 允许用户重新表述请求而不是直接拒绝

---

## 常见问题

**Q: Guardrail 和 Middleware 有什么区别？**

A: Guardrail 是 Middleware 的一个特殊应用，用于安全和合规。Middleware 是更宽泛的概念，可以用于各种执行流程控制，而 Guardrail 专注于安全检查和内容过滤。

**Q: 确定性和模型型 guardrail 应该怎么选？**

A: 大多数情况下两者结合最好。用确定性 guardrail 做快速的初步过滤，用模型型 guardrail 做最终的智能检查。这样既快又准。

**Q: 人工审核会拖累流程吗？**

A: 是的，会增加延迟。所以只在必要的敏感操作上使用，比如金融交易或数据删除。日常操作不需要人工审核。

**Q: 怎样在 PII 检测中自定义规则？**

A: 你可以用 `detector` 参数传入自定义正则表达式：

```javascript
piiRedactionMiddleware({
  piiType: "custom_id",
  detector: /ID-\d{6}/,  // 自定义模式
  strategy: "redact",
})
```

**Q: 多个 guardrail 之间会互相影响吗？**

A: 不会。它们独立执行，但执行顺序很重要。建议把快速检查放前面，复杂检查放后面。

**Q: 怎样测试 guardrail 的有效性？**

A: 可以用测试工具套件测试各种场景，包括应该被阻止和应该通过的请求。记录所有测试结果以确保 guardrail 符合预期。

---

## 相关资源

- [Middleware 文档](https://docs.langchain.com/oss/javascript/langchain/middleware) - 自定义中间件完整指南
- [人工审核](https://docs.langchain.com/oss/javascript/langchain/human-in-the-loop) - 敏感操作的人工审核
- [Agent 测试](https://docs.langchain.com/oss/javascript/langchain/test) - 安全机制的测试策略
