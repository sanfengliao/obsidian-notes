# 会话管理

了解 Claude Agent SDK 如何处理会话和会话恢复

---

Claude Agent SDK 提供了会话管理功能，用于处理对话状态和恢复。会话允许您在多次交互中继续对话，同时保持完整的上下文。

## 会话的工作原理

当您启动新查询时，SDK 会自动创建一个会话，并在初始系统消息中返回会话 ID。您可以捕获此 ID 以便稍后恢复会话。

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
    // 可以保存此 ID 以便稍后恢复
  }

  // 处理其他消息...
  console.log(message)
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

## 恢复会话

SDK 支持从之前的对话状态恢复会话，实现持续的开发工作流。使用 `resume` 选项配合会话 ID 来继续之前的对话。

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk"

// 使用会话 ID 恢复之前的会话
const response = query({
  prompt: "Continue implementing the authentication system from where we left off",
  options: {
    resume: "session-xyz", // 来自上次对话的会话 ID
    model: "claude-opus-4-6",
    allowedTools: ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
  }
})

// 对话将从上次会话的完整上下文继续
for await (const message of response) {
  console.log(message)
}
```

当您恢复会话时，SDK 会自动处理加载对话历史和上下文，使 Claude 能够从上次中断的地方继续。

要跟踪和回滚跨会话的文件更改，请参阅[文件检查点](https://platform.claude.com/docs/zh-CN/agent-sdk/file-checkpointing)。

## 分叉会话

恢复会话时，您可以选择继续原始会话或将其分叉为新分支。默认情况下，恢复会继续原始会话。使用 `forkSession` 选项（TypeScript）或 `fork_session` 选项（Python）来创建一个从恢复状态开始的新会话 ID。

### 何时分叉会话

分叉在以下场景中非常有用：

- 从同一起点探索不同的方法
- 创建多个对话分支而不修改原始会话
- 在不影响原始会话历史的情况下测试更改
- 为不同的实验维护独立的对话路径

### 分叉与继续的对比

| | 继续（continue） | 分叉（fork） |
| :--- | :--- | :--- |
| 会话 ID | 与原始相同 | 生成新的会话 ID |
| 历史记录 | 追加到原始会话 | 从恢复点创建新分支 |
| 原始会话 | 被修改 | 保持不变 |
| 使用场景 | 继续线性对话 | 分支以探索替代方案 |

### 示例：分叉会话

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk"

// 首先，捕获会话 ID
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
    // 这将是一个不同的会话 ID
  }
}

// 原始会话保持不变，仍然可以恢复
const originalContinued = query({
  prompt: "Add authentication to the REST API",
  options: {
    resume: sessionId,
    forkSession: false,  // 继续原始会话（默认值）
    model: "claude-opus-4-6"
  }
})
```
