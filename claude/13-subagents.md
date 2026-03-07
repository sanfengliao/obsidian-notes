# SDK 中的子 agent

在 Claude Agent SDK 应用程序中定义和调用子 agent，以隔离上下文、并行运行任务并应用专门的指令。

---

子 agent 是主 agent 可以生成的独立 agent 实例，用于处理专注的子任务。
使用子 agent 可以为专注的子任务隔离上下文、并行运行多个分析，以及应用专门的指令而不会使主 agent 的提示词膨胀。

本指南介绍如何使用 `agents` 参数在 SDK 中定义和使用子 agent。

## 概述

您可以通过三种方式创建子 agent：

- **编程方式**：在 `query()` 选项中使用 `agents` 参数
- **基于文件系统**：在 `.claude/agents/` 目录中将 agent 定义为 markdown 文件
- **内置通用型**：Claude 可以随时通过 Task 工具调用内置的 `general-purpose` 子 agent，无需您进行任何定义

本指南重点介绍编程方式，这是 SDK 应用程序的推荐方法。

当您定义子 agent 时，Claude 会根据每个子 agent 的 `description` 字段决定是否调用它们。编写清晰的描述来说明何时应该使用该子 agent，Claude 将自动委派适当的任务。您也可以在提示词中按名称显式请求子 agent（例如，"使用 code-reviewer agent 来..."）。

## 使用子 agent 的好处

### 上下文管理

子 agent 与主 agent 保持独立的上下文，防止信息过载并保持交互的专注性。这种隔离确保专门的任务不会用无关的细节污染主对话上下文。

**示例**：`research-assistant` 子 agent 可以探索数十个文件和文档页面，而不会用所有中间搜索结果来干扰主对话，只返回相关的发现。

### 并行化

多个子 agent 可以并发运行，显著加速复杂的工作流程。

**示例**：在代码审查期间，您可以同时运行 `style-checker`、`security-scanner` 和 `test-coverage` 子 agent，将审查时间从几分钟缩短到几秒钟。

### 专门的指令和知识

每个子 agent 都可以拥有定制的系统提示词，包含特定的专业知识、最佳实践和约束条件。

**示例**：`database-migration` 子 agent 可以拥有关于 SQL 最佳实践、回滚策略和数据完整性检查的详细知识，这些在主 agent 的指令中会是不必要的噪音。

### 工具限制

子 agent 可以被限制使用特定的工具，降低意外操作的风险。

**示例**：`doc-reviewer` 子 agent 可能只能访问 Read 和 Grep 工具，确保它可以分析但永远不会意外修改您的文档文件。

## 创建子 agent

### 编程定义（推荐）

直接在代码中使用 `agents` 参数定义子 agent。此示例创建了两个子 agent：一个具有只读访问权限的代码审查器和一个可以执行命令的测试运行器。`Task` 工具必须包含在 `allowedTools` 中，因为 Claude 通过 Task 工具调用子 agent。

#### Python

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    async for message in query(
        prompt="Review the authentication module for security issues",
        options=ClaudeAgentOptions(
            # Task tool is required for subagent invocation
            allowed_tools=["Read", "Grep", "Glob", "Task"],
            agents={
                "code-reviewer": AgentDefinition(
                    # description tells Claude when to use this subagent
                    description="Expert code review specialist. Use for quality, security, and maintainability reviews.",
                    # prompt defines the subagent's behavior and expertise
                    prompt="""You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.""",
                    # tools restricts what the subagent can do (read-only here)
                    tools=["Read", "Grep", "Glob"],
                    # model overrides the default model for this subagent
                    model="sonnet"
                ),
                "test-runner": AgentDefinition(
                    description="Runs and analyzes test suites. Use for test execution and coverage analysis.",
                    prompt="""You are a test execution specialist. Run tests and provide clear analysis of results.

Focus on:
- Running test commands
- Analyzing test output
- Identifying failing tests
- Suggesting fixes for failures""",
                    # Bash access lets this subagent run test commands
                    tools=["Bash", "Read", "Grep"]
                )
            }
        )
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

#### TypeScript

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({
  prompt: "Review the authentication module for security issues",
  options: {
    // Task tool is required for subagent invocation
    allowedTools: ['Read', 'Grep', 'Glob', 'Task'],
    agents: {
      'code-reviewer': {
        // description tells Claude when to use this subagent
        description: 'Expert code review specialist. Use for quality, security, and maintainability reviews.',
        // prompt defines the subagent's behavior and expertise
        prompt: `You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.`,
        // tools restricts what the subagent can do (read-only here)
        tools: ['Read', 'Grep', 'Glob'],
        // model overrides the default model for this subagent
        model: 'sonnet'
      },
      'test-runner': {
        description: 'Runs and analyzes test suites. Use for test execution and coverage analysis.',
        prompt: `You are a test execution specialist. Run tests and provide clear analysis of results.

Focus on:
- Running test commands
- Analyzing test output
- Identifying failing tests
- Suggesting fixes for failures`,
        // Bash access lets this subagent run test commands
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
| `prompt` | `string` | 是 | agent 的系统提示词，定义其角色和行为 |
| `tools` | `string[]` | 否 | 允许的工具名称数组。如果省略，则继承所有工具 |
| `model` | `'sonnet' \| 'opus' \| 'haiku' \| 'inherit'` | 否 | 此 agent 的模型覆盖。如果省略，默认使用主模型 |

> **注意**：子 agent 不能生成自己的子 agent。不要在子 agent 的 `tools` 数组中包含 `Task`。

### 基于文件系统的定义（替代方案）

您也可以在 `.claude/agents/` 目录中将子 agent 定义为 markdown 文件。以编程方式定义的 agent 优先于同名的基于文件系统的 agent。

> **提示**：即使不定义自定义子 agent，当 `Task` 在您的 `allowedTools` 中时，Claude 也可以生成内置的 `general-purpose` 子 agent。这对于在不创建专门 agent 的情况下委派研究或探索任务非常有用。

## 调用子 agent

### 自动调用

Claude 会根据任务和每个子 agent 的 `description` 自动决定何时调用子 agent。例如，如果您定义了一个 `performance-optimizer` 子 agent，其描述为"用于查询调优的性能优化专家"，当您的提示词提到优化查询时，Claude 将调用它。

编写清晰、具体的描述，以便 Claude 能够将任务匹配到正确的子 agent。

### 显式调用

要确保 Claude 使用特定的子 agent，请在提示词中按名称提及它：

```
"Use the code-reviewer agent to check the authentication module"
```

这将绕过自动匹配并直接调用指定的子 agent。

### 动态 agent 配置

您可以根据运行时条件动态创建 agent 定义。此示例创建了一个具有不同严格级别的安全审查器，对严格审查使用更强大的模型。

#### Python

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

# Factory function that returns an AgentDefinition
# This pattern lets you customize agents based on runtime conditions
def create_security_agent(security_level: str) -> AgentDefinition:
    is_strict = security_level == "strict"
    return AgentDefinition(
        description="Security code reviewer",
        # Customize the prompt based on strictness level
        prompt=f"You are a {'strict' if is_strict else 'balanced'} security reviewer...",
        tools=["Read", "Grep", "Glob"],
        # Key insight: use a more capable model for high-stakes reviews
        model="opus" if is_strict else "sonnet"
    )

async def main():
    # The agent is created at query time, so each request can use different settings
    async for message in query(
        prompt="Review this PR for security issues",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob", "Task"],
            agents={
                # Call the factory with your desired configuration
                "security-reviewer": create_security_agent("strict")
            }
        )
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

#### TypeScript

```typescript
import { query, type AgentDefinition } from '@anthropic-ai/claude-agent-sdk';

// Factory function that returns an AgentDefinition
// This pattern lets you customize agents based on runtime conditions
function createSecurityAgent(securityLevel: 'basic' | 'strict'): AgentDefinition {
  const isStrict = securityLevel === 'strict';
  return {
    description: 'Security code reviewer',
    // Customize the prompt based on strictness level
    prompt: `You are a ${isStrict ? 'strict' : 'balanced'} security reviewer...`,
    tools: ['Read', 'Grep', 'Glob'],
    // Key insight: use a more capable model for high-stakes reviews
    model: isStrict ? 'opus' : 'sonnet'
  };
}

// The agent is created at query time, so each request can use different settings
for await (const message of query({
  prompt: "Review this PR for security issues",
  options: {
    allowedTools: ['Read', 'Grep', 'Glob', 'Task'],
    agents: {
      // Call the factory with your desired configuration
      'security-reviewer': createSecurityAgent('strict')
    }
  }
})) {
  if ('result' in message) console.log(message.result);
}
```

## 检测子 agent 调用

子 agent 通过 Task 工具调用。要检测子 agent 何时被调用，请检查 `name: "Task"` 的 `tool_use` 块。来自子 agent 上下文内的消息包含 `parent_tool_use_id` 字段。

此示例遍历流式消息，记录子 agent 何时被调用以及后续消息何时来自该子 agent 的执行上下文。

> **注意**：SDK 之间的消息结构有所不同。在 Python 中，内容块通过 `message.content` 直接访问。在 TypeScript 中，`SDKAssistantMessage` 包装了 Claude API 消息，因此内容通过 `message.message.content` 访问。

#### Python

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    async for message in query(
        prompt="Use the code-reviewer agent to review this codebase",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep", "Task"],
            agents={
                "code-reviewer": AgentDefinition(
                    description="Expert code reviewer.",
                    prompt="Analyze code quality and suggest improvements.",
                    tools=["Read", "Glob", "Grep"]
                )
            }
        )
    ):
        # Check for subagent invocation in message content
        if hasattr(message, 'content') and message.content:
            for block in message.content:
                if getattr(block, 'type', None) == 'tool_use' and block.name == 'Task':
                    print(f"Subagent invoked: {block.input.get('subagent_type')}")

        # Check if this message is from within a subagent's context
        if hasattr(message, 'parent_tool_use_id') and message.parent_tool_use_id:
            print("  (running inside subagent)")

        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

#### TypeScript

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

  // Check for subagent invocation in message content
  for (const block of msg.message?.content ?? []) {
    if (block.type === "tool_use" && block.name === "Task") {
      console.log(`Subagent invoked: ${block.input.subagent_type}`);
    }
  }

  // Check if this message is from within a subagent's context
  if (msg.parent_tool_use_id) {
    console.log("  (running inside subagent)");
  }

  if ("result" in message) {
    console.log(message.result);
  }
}
```

## 恢复子 agent

子 agent 可以被恢复以从中断处继续。恢复的子 agent 保留其完整的对话历史，包括所有先前的工具调用、结果和推理。子 agent 会从停止的地方精确恢复，而不是重新开始。

当子 agent 完成时，Claude 会在 Task 工具结果中接收其 agent ID。要以编程方式恢复子 agent：

1. **捕获会话 ID**：在第一次查询期间从消息中提取 `session_id`
2. **提取 agent ID**：从消息内容中解析 `agentId`
3. **恢复会话**：在第二次查询的选项中传递 `resume: sessionId`，并在提示词中包含 agent ID

> **注意**：您必须恢复同一会话才能访问子 agent 的记录。每次 `query()` 调用默认启动一个新会话，因此传递 `resume: sessionId` 以在同一会话中继续。
>
> 如果您使用的是自定义 agent（而非内置 agent），您还需要在两次查询的 `agents` 参数中传递相同的 agent 定义。

下面的示例演示了此流程：第一次查询运行子 agent 并捕获会话 ID 和 agent ID，然后第二次查询恢复会话以提出需要第一次分析上下文的后续问题。

#### TypeScript

```typescript
import { query, type SDKMessage } from '@anthropic-ai/claude-agent-sdk';

// Helper to extract agentId from message content
// Stringify to avoid traversing different block types (TextBlock, ToolResultBlock, etc.)
function extractAgentId(message: SDKMessage): string | undefined {
  if (!('message' in message)) return undefined;
  // Stringify the content so we can search it without traversing nested blocks
  const content = JSON.stringify(message.message.content);
  const match = content.match(/agentId:\s*([a-f0-9-]+)/);
  return match?.[1];
}

let agentId: string | undefined;
let sessionId: string | undefined;

// First invocation - use the Explore agent to find API endpoints
for await (const message of query({
  prompt: "Use the Explore agent to find all API endpoints in this codebase",
  options: { allowedTools: ['Read', 'Grep', 'Glob', 'Task'] }
})) {
  // Capture session_id from ResultMessage (needed to resume this session)
  if ('session_id' in message) sessionId = message.session_id;
  // Search message content for the agentId (appears in Task tool results)
  const extractedId = extractAgentId(message);
  if (extractedId) agentId = extractedId;
  // Print the final result
  if ('result' in message) console.log(message.result);
}

// Second invocation - resume and ask follow-up
if (agentId && sessionId) {
  for await (const message of query({
    prompt: `Resume agent ${agentId} and list the top 3 most complex endpoints`,
    options: { allowedTools: ['Read', 'Grep', 'Glob', 'Task'], resume: sessionId }
  })) {
    if ('result' in message) console.log(message.result);
  }
}
```

#### Python

```python
import asyncio
import json
import re
from claude_agent_sdk import query, ClaudeAgentOptions

def extract_agent_id(text: str) -> str | None:
    """Extract agentId from Task tool result text."""
    match = re.search(r"agentId:\s*([a-f0-9-]+)", text)
    return match.group(1) if match else None

async def main():
    agent_id = None
    session_id = None

    # First invocation - use the Explore agent to find API endpoints
    async for message in query(
        prompt="Use the Explore agent to find all API endpoints in this codebase",
        options=ClaudeAgentOptions(allowed_tools=["Read", "Grep", "Glob", "Task"])
    ):
        # Capture session_id from ResultMessage (needed to resume this session)
        if hasattr(message, "session_id"):
            session_id = message.session_id
        # Search message content for the agentId (appears in Task tool results)
        if hasattr(message, "content"):
            # Stringify the content so we can search it without traversing nested blocks
            content_str = json.dumps(message.content, default=str)
            extracted = extract_agent_id(content_str)
            if extracted:
                agent_id = extracted
        # Print the final result
        if hasattr(message, "result"):
            print(message.result)

    # Second invocation - resume and ask follow-up
    if agent_id and session_id:
        async for message in query(
            prompt=f"Resume agent {agent_id} and list the top 3 most complex endpoints",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Grep", "Glob", "Task"],
                resume=session_id
            )
        ):
            if hasattr(message, "result"):
                print(message.result)

asyncio.run(main())
```

子 agent 记录独立于主对话持久存在：

- **主对话压缩**：当主对话压缩时，子 agent 记录不受影响。它们存储在单独的文件中。
- **会话持久性**：子 agent 记录在其会话内持久存在。您可以通过恢复同一会话在重启 Claude Code 后恢复子 agent。
- **自动清理**：记录根据 `cleanupPeriodDays` 设置进行清理（默认：30 天）。

## 工具限制

子 agent 可以通过 `tools` 字段限制工具访问：

- **省略该字段**：agent 继承所有可用工具（默认）
- **指定工具**：agent 只能使用列出的工具

此示例创建了一个只读分析 agent，可以检查代码但不能修改文件或运行命令。

#### Python

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    async for message in query(
        prompt="Analyze the architecture of this codebase",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob", "Task"],
            agents={
                "code-analyzer": AgentDefinition(
                    description="Static code analysis and architecture review",
                    prompt="""You are a code architecture analyst. Analyze code structure,
identify patterns, and suggest improvements without making changes.""",
                    # Read-only tools: no Edit, Write, or Bash access
                    tools=["Read", "Grep", "Glob"]
                )
            }
        )
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

#### TypeScript

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({
  prompt: "Analyze the architecture of this codebase",
  options: {
    allowedTools: ['Read', 'Grep', 'Glob', 'Task'],
    agents: {
      'code-analyzer': {
        description: 'Static code analysis and architecture review',
        prompt: `You are a code architecture analyst. Analyze code structure,
identify patterns, and suggest improvements without making changes.`,
        // Read-only tools: no Edit, Write, or Bash access
        tools: ['Read', 'Grep', 'Glob']
      }
    }
  }
})) {
  if ('result' in message) console.log(message.result);
}
```

### 常见工具组合

| 用例 | 工具 | 描述 |
| :--- | :--- | :--- |
| 只读分析 | `Read`、`Grep`、`Glob` | 可以检查代码但不能修改或执行 |
| 测试执行 | `Bash`、`Read`、`Grep` | 可以运行命令和分析输出 |
| 代码修改 | `Read`、`Edit`、`Write`、`Grep`、`Glob` | 完全读写访问，无命令执行 |
| 完全访问 | 所有工具 | 从父 agent 继承所有工具（省略 `tools` 字段） |

## 故障排除

### Claude 未委派给子 agent

如果 Claude 直接完成任务而不是委派给您的子 agent：

1. **包含 Task 工具**：子 agent 通过 Task 工具调用，因此它必须在 `allowedTools` 中
2. **使用显式提示**：在提示词中按名称提及子 agent（例如，"使用 code-reviewer agent 来..."）
3. **编写清晰的描述**：准确说明何时应该使用该子 agent，以便 Claude 能够适当地匹配任务

### 基于文件系统的 agent 未加载

在 `.claude/agents/` 中定义的 agent 仅在启动时加载。如果您在 Claude Code 运行时创建了新的 agent 文件，请重启会话以加载它。

### Windows：长提示词失败

在 Windows 上，具有非常长提示词的子 agent 可能由于命令行长度限制（8191 个字符）而失败。保持提示词简洁或对复杂指令使用基于文件系统的 agent。

## 相关文档

- **[会话管理](8-sessions.md)**：了解如何恢复会话，这是恢复子 agent 所必需的
- **[权限](5-configuring-permissions.md)**：配置 `allowedTools` 控制 agent 可以使用哪些工具
- **[通过 MCP 连接外部工具](11-mcp.md)**：为子 agent 提供外部工具访问
