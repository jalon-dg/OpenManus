# OpenManus 技术架构说明文档

## 项目概述

OpenManus 是一个开源的通用 AI Agent 框架，由 MetaGPT 团队开发。它能够在极短时间内复现 Manus 的核心功能，无需邀请码即可使用。项目支持多种 LLM 提供商，具备浏览器自动化、代码执行、文件操作等能力。

**GitHub**: https://github.com/MetaGPT-DeepResearch/OpenManus

---

## 目录结构

```
OpenManus/
├── main.py                      # 标准模式入口
├── run_flow.py                  # 多 Agent 流程模式入口
├── run_mcp.py                   # MCP Agent 入口
├── run_mcp_server.py            # MCP 服务器入口
├── sandbox_main.py              # Sandbox 模式入口
├── setup.py                     # 包配置
├── requirements.txt             # 依赖清单
├── config/
│   ├── config.toml              # 主配置文件
│   ├── config.example.toml      # 配置示例
│   ├── mcp.json                 # MCP 服务器配置
│   └── config.example-model-*.toml  # 各种模型配置示例
├── app/                         # 核心应用代码
│   ├── agent/                   # Agent 模块
│   ├── tool/                    # 工具模块
│   ├── flow/                    # 流程模块
│   ├── prompt/                  # 提示词模块
│   ├── sandbox/                 # 沙箱模块
│   ├── mcp/                     # MCP 相关模块
│   ├── llm.py                   # LLM 接口
│   ├── config.py                # 配置管理
│   ├── schema.py                # 数据模式定义
│   └── bedrock.py               # AWS Bedrock 集成
├── protocol/a2a/                # Agent-to-Agent 协议
├── workspace/                   # 工作目录
└── tests/                       # 测试文件
```

---

## 五种运行方式详解

### 1. 标准模式 (main.py)

**命令**:

```bash
python main.py
python main.py --prompt "你的任务描述"
```

**特点**:

- 最基础的运行方式，启动单一的 Manus Agent
- 支持 `--prompt` 命令行参数直接传入任务
- 无参数时进入交互式输入模式
- 支持 Ctrl+C 优雅中断处理

**适用场景**: 简单任务、快速上手测试

---

### 2. 流程模式 (run_flow.py)

**命令**:

```bash
python run_flow.py
python run_flow.py --prompt "你的任务描述"
```

**特点**:

- 使用 `PlanningFlow` 协调多个 Agent 协作
- 默认包含 Manus Agent，可选启用 DataAnalysis Agent
- 具备任务规划能力，将复杂任务分解为步骤
- 1 小时超时机制防止无限运行
- 可通过 `use_data_analysis_agent` 配置启用数据分析 Agent

**适用场景**: 复杂任务、需要多步骤规划的场景

**流程工作流**:

```
用户请求 → LLM创建初始计划 → 逐步执行每个步骤 → 使用合适Agent完成 → 标记完成状态 → 最终总结
```

---

### 3. MCP Agent 模式 (run_mcp.py)

**命令**:

```bash
python run_mcp.py                           # 默认模式（交互式）
python run_mcp.py --interactive             # 交互模式
python run_mcp.py --prompt "任务"           # 单次执行模式
python run_mcp.py --connection stdio        # stdio 连接模式
python run_mcp.py --connection sse          # SSE 连接模式
```

**特点**:

- 支持连接外部 MCP (Model Context Protocol) 服务器
- 支持 `stdio` 和 `SSE` 两种连接方式
- 支持交互模式和单次执行模式
- 可动态加载 MCP 服务器提供的工具

**适用场景**: 集成第三方 MCP 服务、扩展工具能力

---

### 4. MCP 服务器模式 (run_mcp_server.py)

**命令**:

```bash
python run_mcp_server.py
```

**特点**:

- 将 OpenManus 作为 MCP 服务器运行
- 使用 FastMCP 框架构建
- 支持 stdio 传输
- 注册标准工具供外部调用：
  - bash: Shell 命令执行
  - browser: 浏览器操作
  - editor: 文件编辑
  - terminate: 终止任务

**适用场景**: 被其他 MCP 客户端系统调用、构建工具服务

---

### 5. Sandbox 模式 (sandbox_main.py)

**命令**:

```bash
python sandbox_main.py
```

**特点**:

- 使用 Docker 容器隔离执行命令
- 防止危险操作影响宿主系统
- 包含隔离的浏览器、文件、Shell 和视觉工具

**适用场景**: 安全敏感环境、不可信代码执行

---

## 运行方式对比总览

| 方式             | 入口文件            | 特点                       | 适用场景           |
| ---------------- | ------------------- | -------------------------- | ------------------ |
| **标准模式**     | `main.py`           | 单一 Manus Agent，简单直接 | 简单任务、快速测试 |
| **流程模式**     | `run_flow.py`       | 多 Agent 协作，有规划能力  | 复杂任务、分解执行 |
| **MCP 模式**     | `run_mcp.py`        | 支持 MCP 协议连接外部服务  | 集成第三方工具     |
| **MCP 服务器**   | `run_mcp_server.py` | 作为 MCP 服务器运行        | 被其他系统调用     |
| **Sandbox 模式** | `sandbox_main.py`   | Docker 隔离执行，安全      | 安全敏感环境       |

---

## 核心架构设计

### Agent 继承层次

```
BaseAgent (抽象基类)
    │
    └── ReActAgent (Think-Act 模式)
            │
            ├── ToolCallAgent (工具调用)
            │       └── Manus (主 Agent)
            │
            └── 其他 Specialized Agent
```

### 主要 Agent 列表

| Agent             | 文件                         | 功能描述                        |
| ----------------- | ---------------------------- | ------------------------------- |
| **Manus**         | `app/agent/manus.py`         | 通用主 Agent，支持 MCP 工具     |
| **ToolCallAgent** | `app/agent/toolcall.py`      | 工具调用基础 Agent              |
| **ReActAgent**    | `app/agent/react.py`         | ReAct (Reasoning + Acting) 模式 |
| **BaseAgent**     | `app/agent/base.py`          | 所有 Agent 的基类               |
| **DataAnalysis**  | `app/agent/data_analysis.py` | 数据分析专用 Agent              |
| **MCPAgent**      | `app/agent/mcp.py`           | MCP 专用 Agent                  |
| **BrowserAgent**  | `app/agent/browser.py`       | 浏览器操作 Agent                |
| **SWEBenchAgent** | `app/agent/swe.py`           | 软件工程 Benchmark Agent        |
| **SandboxAgent**  | `app/agent/sandbox_agent.py` | 沙箱执行 Agent                  |

---

## 模块协作关系

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         入口层 (main.py)                          │
│                    run_flow.py / run_mcp.py 等                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Flow 层                                   │
│    BaseFlow / PlanningFlow (编排多个 Agent)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Manus       │   │  DataAnalysis │   │   Other       │
│   Agent       │   │    Agent      │   │    Agents     │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └─────────────────────┬──────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Agent 层                                    │
│              BaseAgent / ReActAgent / ToolCallAgent               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Core: LLM + Memory + State + Tools                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│      LLM      │   │    Memory     │   │ToolCollection │
│  (OpenAI/     │   │   (消息历史)  │   │  (工具集合)    │
│   Azure/      │   │               │   │               │
│   Bedrock)    │   │               │   │               │
└───────┬───────┘   └───────────────┘   └───────┬───────┘
        │                                       │
        ▼                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tool 层 (BaseTool 的子类)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Browser │  │  Bash    │  │  Editor  │  │  MCPClientTool   │ │
│  │  UseTool │  │          │  │          │  │                  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     配置层 (Config)                               │
│           config.toml + mcp.json + environment                   │
└─────────────────────────────────────────────────────────────────┘
```

### 模块职责说明

| 模块 | 职责 | 核心依赖 |
|------|------|----------|
| **Config** | 提供所有配置（LLM、浏览器、工具等） | TOML 配置文件 |
| **LLM** | 封装大语言模型调用（OpenAI/Azure/Bedrock） | Config + OpenAI 客户端 |
| **BaseAgent** | Agent 基类：状态管理、消息内存、执行循环 | LLM + Memory |
| **ReActAgent** | 实现 Think-Act 模式 | BaseAgent |
| **ToolCallAgent** | 工具调用核心逻辑 | ReActAgent + ToolCollection |
| **ToolCollection** | 管理工具集合、执行工具 | BaseTool |
| **BaseTool** | 所有工具的基类 | - |
| **BaseFlow** | 编排多个 Agent | BaseAgent |
| **PlanningFlow** | 任务规划与分步执行 | BaseFlow + LLM + PlanningTool |
| **Memory** | 消息历史管理 | Message |

### 核心协作流程

#### 1. Agent 执行流程 (Think-Act 循环)

```
用户请求
    │
    ▼
BaseAgent.run() ──▶ 添加用户消息到 Memory
    │
    ▼
循环执行 (max_steps 次)
    │
    ├─▶ think() ──▶ 调用 LLM.ask_tool() ──▶ 获取工具调用决策
    │          │
    │          ▼
    │       解析 tool_calls
    │
    ├─▶ act() ──▶ execute_tool() ──▶ ToolCollection.execute()
    │          │
    │          ▼
    │       执行 BaseTool.execute()
    │
    ├─▶ 添加工具结果到 Memory
    │
    ▼
完成 / 达到最大步数 / 收到 Terminate
```

#### 2. ToolCollection 执行工具流程

```
ToolCollection.execute(name, tool_input)
        │
        ▼
查找工具 (tool_map)
        │
        ▼
调用 tool.execute(**tool_input)
        │
        ▼
返回 ToolResult (output/error/base64_image)
```

#### 3. Flow 编排 Agent 流程

```
PlanningFlow.execute(input_text)
        │
        ▼
_create_initial_plan() ──▶ 调用 LLM 生成计划
        │
        ▼
循环执行计划步骤
        │
        ├─▶ _get_current_step_info() ──▶ 获取下一步骤
        │
        ├─▶ get_executor() ──▶ 选择合适的 Agent
        │
        ├─▶ _execute_step() ──▶ Agent.run(step_prompt)
        │
        ├─▶ _mark_step_completed() ──▶ 更新计划状态
        │
        ▼
_finalize_plan() ──▶ 生成最终总结
```

### 关键数据流

#### 消息流转

```
User Message → Memory.add_message()
       ↓
LLM.ask_tool() 接收 messages
       ↓
LLM 返回 response (content + tool_calls)
       ↓
Assistant Message → Memory.add_message()
       ↓
Tool Result → Memory.add_message()
       ↓
循环...
```

#### 工具调用

```
LLM 返回 tool_calls
       ↓
ToolCallAgent.act() 解析参数
       ↓
ToolCollection.execute(name, args)
       ↓
BaseTool.execute(**args)
       ↓
ToolResult (output/error)
       ↓
Message.tool_message() → Memory
```

### 依赖注入关系

#### 1. Config → LLM
```
Config.toml
    │
    ▼
LLM.__init__(config_name) ──▶ 读取配置 ──▶ 创建 OpenAI Client
```

#### 2. Config → Agent
```
Config.toml
    │
    ▼
Manus.create() ──▶ 读取 config.mcp_config ──▶ 初始化 MCP 服务器
```

#### 3. Agent → LLM
```
Agent.__init__()
    │
    ▼
self.llm = LLM(config_name=self.name) ──▶ 单例模式获取 LLM
```

#### 4. Agent → ToolCollection
```
Agent.available_tools: ToolCollection
    │
    ▼
ToolCollection.add_tools() ──▶ 动态扩展工具
```

### 状态机

#### AgentState 枚举
```
IDLE (空闲) ──▶ RUNNING (运行中) ──▶ FINISHED (完成)
                         │
                         │ 错误
                         ▼
                       ERROR (错误)
```

#### PlanStepStatus 枚举
```
NOT_STARTED ──▶ IN_PROGRESS ──▶ COMPLETED
                    │
                    │ 阻塞
                    ▼
                 BLOCKED
```

### 消息类型 (Message)

| 类型 | 角色 (role) | 用途 |
|------|-------------|------|
| system | system | 系统提示词 |
| user | user | 用户输入 |
| assistant | assistant | LLM 回复 |
| tool | tool | 工具执行结果 |

### 执行流程时序图

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────────┐   ┌────────┐
│  User   │   │ManusAgent│   │  LLM    │   │ToolCollection│   │  Tool  │
└────┬────┘   └────┬─────┘   └────┬────┘   └──────┬──────┘   └────┬───┘
     │             │              │               │               │
     │  run()      │              │               │               │
     ├────────────▶│              │               │               │
     │             │  ask_tool()  │               │               │
     │             ├─────────────▶│               │               │
     │             │              │  tool_calls   │               │
     │             │◀─────────────┤               │               │
     │             │              │               │               │
     │             │ execute()    │               │               │
     │             │──────────────┼──────────────▶│               │
     │             │              │               │  execute()    │
     │             │              │               ├──────────────▶│
     │             │              │               │               │ 实际执行
     │             │              │               │◀──────────────┤
     │             │              │               │  ToolResult   │
     │             │◀─────────────┼───────────────┤               │
     │             │   result     │               │               │
     │             │              │               │               │
     │◀────────────┤              │               │               │
     │   output    │              │               │               │
     │             │              │               │               │
```

### 模块通信方式

| 通信场景 | 方式 | 说明 |
|----------|------|------|
| Agent → LLM | 同步调用 | `await llm.ask_tool()` |
| Agent → Tool | 同步调用 | `await tool.execute()` |
| Flow → Agent | 异步调用 | `await agent.run()` |
| Memory → Agent | 属性访问 | `memory.add_message()` |
| ToolCollection → Tool | 方法调用 | `tool_map[name]` |

### 配置传递路径

```
config/config.toml
        │
        ▼
┌────────────────┐
│    Config      │  ──▶ LLMSettings (llm_config)
│    (app/config)│  ──▶ BrowserSettings
└───────┬────────┘  ──▶ SandboxSettings
        │          ──▶ MCPSettings
        ▼
┌────────────────┐
│     LLM        │  ──▶ api_key, base_url, model
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    Agent       │  ──▶ self.llm = LLM()
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ToolCallAgent   │  ──▶ self.available_tools
└────────────────┘
```

---

## Tool 模块详解

### 基础工具类

- **BaseTool** (`app/tool/base.py`): 所有工具的抽象基类
- **ToolCollection** (`app/tool/tool_collection.py`): 工具集合管理

### 核心工具

| 工具                     | 文件                                 | 功能            |
| ------------------------ | ------------------------------------ | --------------- |
| **PythonExecute**        | `app/tool/python_execute.py`         | Python 代码执行 |
| **BrowserUseTool**       | `app/tool/browser_use_tool.py`       | 浏览器自动化    |
| **StrReplaceEditor**     | `app/tool/str_replace_editor.py`     | 文件编辑        |
| **Bash**                 | `app/tool/bash.py`                   | Shell 命令执行  |
| **WebSearch**            | `app/tool/web_search.py`             | 网络搜索        |
| **CreateChatCompletion** | `app/tool/create_chat_completion.py` | LLM 调用        |
| **PlanningTool**         | `app/tool/planning.py`               | 任务规划        |
| **Terminate**            | `app/tool/terminate.py`              | 终止任务        |
| **AskHuman**             | `app/tool/ask_human.py`              | 请求人类输入    |
| **MCPClientTool**        | `app/tool/mcp.py`                    | MCP 客户端工具  |

### 工具分类

#### 搜索工具 (`app/tool/search/`)

- 百度搜索
- Bing 搜索
- DuckDuckGo 搜索
- Google 搜索

#### 沙箱工具 (`app/tool/sandbox/`)

- 隔离浏览器
- 隔离文件系统
- 隔离 Shell
- 隔离视觉处理

#### 专用工具

- **chart_visualization**: 图表可视化
- **file_operators.py**: 文件操作

---

## Flow 模块

| 组件             | 文件                       | 功能       |
| ---------------- | -------------------------- | ---------- |
| **BaseFlow**     | `app/flow/base.py`         | 流程基类   |
| **PlanningFlow** | `app/flow/planning.py`     | 规划型流程 |
| **FlowFactory**  | `app/flow/flow_factory.py` | 流程工厂   |

---

## 配置管理

### 配置来源

1. `config/config.toml` - 主配置文件
2. `config/mcp.json` - MCP 服务器配置
3. `config/config.example.toml` - 配置模板

### 主要配置类

| 配置类              | 功能                                        |
| ------------------- | ------------------------------------------- |
| **LLMSettings**     | LLM 配置（模型、API Key、base_url、温度等） |
| **BrowserSettings** | 浏览器设置                                  |
| **SandboxSettings** | 沙箱设置                                    |
| **SearchSettings**  | 搜索引擎设置                                |
| **MCPSettings**     | MCP 服务器配置                              |
| **RunflowSettings** | 运行流程配置                                |
| **DaytonaSettings** | Daytona 云沙箱配置                          |

---

## 数据模式 (Schema)

| 模式类         | 功能                                             |
| -------------- | ------------------------------------------------ |
| **Message**    | 聊天消息，支持多模态内容                         |
| **Memory**     | 消息历史管理                                     |
| **ToolCall**   | 工具调用定义                                     |
| **ToolResult** | 工具执行结果                                     |
| **AgentState** | Agent 状态枚举（IDLE, RUNNING, FINISHED, ERROR） |
| **ToolChoice** | 工具选择策略（NONE, AUTO, REQUIRED）             |

---

## LLM 模块

**文件**: `app/llm.py`

### 主要功能

- 支持多种 API 类型（OpenAI、Azure、AWS Bedrock）
- 多模态模型支持（图像、文本）
- Token 计数和管理
- 流式响应支持
- 重试机制（使用 tenacity 库）

### 支持的模型类型

- **推理模型**: o1, o3-mini
- **多模态模型**: GPT-4o 系列, Claude-3 系列

---

## MCP 集成

MCP (Model Context Protocol) 是一个标准化协议，允许 AI Agent 与外部工具和服务进行交互。OpenManus 提供了完整的 MCP 客户端和服务器支持。

### MCP 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP 生态系统                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐           ┌─────────────────┐            │
│   │   MCP Client    │           │   MCP Server    │            │
│   │  (MCPAgent/     │◀─────────▶│  (run_mcp_server│            │
│   │   Manus)        │   stdio   │   .py)          │            │
│   └────────┬────────┘   /sse    └────────┬────────┘            │
│            │                             │                     │
│            │                             │                     │
│            ▼                             ▼                     │
│   ┌─────────────────────────────────────────────────┐          │
│   │              工具调用流程                         │          │
│   │  LLM → think() → act() → execute_tool()         │          │
│   │       ↓                                          │          │
│   │  ToolCollection.execute()                        │          │
│   │       ↓                                          │          │
│   │  MCPClientTool.execute() → MCP Session           │          │
│   │                      ↓                           │          │
│   │              远程 MCP Server                      │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MCP 组件

| 组件 | 文件 | 功能 |
|------|------|------|
| **MCPAgent** | `app/agent/mcp.py` | 连接 MCP 服务器并使用其工具 |
| **MCPClients** | `app/tool/mcp.py` | 管理多个 MCP 连接和工具 |
| **MCPClientTool** | `app/tool/mcp.py` | 代理工具，调用远程 MCP 服务 |
| **MCPServer** | `app/mcp/server.py` | 将 OpenManus 作为 MCP 服务器运行 |

### 1. 使用 MCP Agent 连接外部服务器

MCPAgent 可以连接外部 MCP 服务器并使用其工具。

#### 命令行方式

```bash
# stdio 连接（默认）
python run_mcp.py

# SSE 连接
python run_mcp.py --connection sse --server-url http://localhost:8080/sse

# 交互模式
python run_mcp.py --interactive

# 单次执行
python run_mcp.py --prompt "使用 filesystem 工具列出文件"
```

#### 代码方式

```python
import asyncio
from app.agent.mcp import MCPAgent

async def main():
    # 创建 Agent
    agent = MCPAgent()

    # 方式1: stdio 连接
    await agent.initialize(
        connection_type="stdio",
        command="python",  # 或其他 MCP 服务器命令
        args=["-m", "my_mcp_server"]
    )

    # 方式2: SSE 连接
    # await agent.initialize(
    #     connection_type="sse",
    #     server_url="http://localhost:8080/sse"
    # )

    # 运行任务
    result = await agent.run("请列出当前目录的文件")
    print(result)

    # 清理
    await agent.cleanup()

asyncio.run(main())
```

### 2. 在 Manus 中使用 MCP

Manus Agent 支持通过配置文件连接多个 MCP 服务器。

#### 配置文件 (`config/mcp.json`)

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextplugin/server-filesystem", "/path/to/dir"]
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextplugin/server-github"]
    },
    "remote-server": {
      "type": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

#### 启动使用 MCP 的 Manus

```bash
python main.py --prompt "使用 filesystem 工具读取 config.toml 文件"
```

### 3. 将 OpenManus 作为 MCP 服务器

将 OpenManus 的工具暴露为 MCP 服务，供其他客户端使用。

#### 启动 MCP 服务器

```bash
# stdio 模式（默认）
python run_mcp_server.py

# 指定传输方式
python run_mcp_server.py --transport stdio
```

#### 可用工具

MCP 服务器暴露以下工具：

| 工具名 | 功能 |
|--------|------|
| `bash` | 执行 Shell 命令 |
| `browser` | 浏览器自动化操作 |
| `editor` | 文件编辑 |
| `terminate` | 终止当前任务 |

#### 在其他应用中连接

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def main():
    # 连接到 OpenManus MCP 服务器
    server_params = StdioServerParameters(
        command="python",
        args=["run_mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()

            # 列出可用工具
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools])

            # 调用工具
            result = await session.call_tool("bash", {"command": "ls -la"})
            print(result.content[0].text)

asyncio.run(main())
```

### 4. 创建自定义 MCP 服务器

#### 使用 FastMCP 创建

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP
from app.tool.base import BaseTool, ToolResult

app = FastMCP("my-server")

# 注册函数
@app.tool()
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 的天气是晴天"

@app.tool()
def calculate(a: int, b: int) -> int:
    """执行计算"""
    return a + b

if __name__ == "__main__":
    app.run()
```

#### 连接到自定义服务器

```python
from app.agent.mcp import MCPAgent
import asyncio

async def main():
    agent = MCPAgent()
    await agent.initialize(
        connection_type="stdio",
        command="python",
        args=["my_mcp_server.py"]
    )

    result = await agent.run("北京天气怎么样？")
    print(result)

asyncio.run(main())
```

### 5. MCPClients 工具管理

MCPClients 继承自 ToolCollection，管理多个 MCP 服务器连接。

```python
from app.tool.mcp import MCPClients

# 创建客户端集合
clients = MCPClients()

# 连接多个服务器
await clients.connect_stdio(
    command="python",
    args=["-m", "server_a"],
    server_id="server_a"
)

await clients.connect_sse(
    server_url="http://localhost:8080/sse",
    server_id="server_b"
)

# 访问工具
tool = clients.tool_map["mcp_server_a_my_tool"]
result = await tool.execute(param="value")

# 断开连接
await clients.disconnect("server_a")
await clients.disconnect()  # 断开所有
```

### 6. 工具名称转换

MCP 工具名会自动转换为 OpenManus 格式：

```
原始工具名: get_weather
转换后:     mcp_{server_id}_get_weather

示例:
  server_id = "filesystem"
  原始:      read_file
  转换后:    mcp_filesystem_read_file
```

### 7. 配置管理

#### MCPSettings 配置类

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `server_reference` | str | MCP 服务器模块引用 |
| `servers` | Dict[str, MCPServerConfig] | 服务器配置字典 |

#### MCPServerConfig 配置类

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `type` | str | 连接类型 (`stdio` 或 `sse`) |
| `url` | str | SSE 连接的 URL |
| `command` | str | stdio 连接的命令 |
| `args` | List[str] | 命令参数 |

### 8. 最佳实践

1. **连接管理**: 使用 `async/await` 确保正确清理连接
2. **工具刷新**: MCPAgent 会定期刷新工具列表
3. **错误处理**: MCPClientTool 会捕获并返回远程错误
4. **名称转换**: 注意工具名的前缀转换
5. **超时设置**: SSE 连接可能需要配置超时

### 常见问题

**Q: 工具调用失败**
A: 检查 MCP 服务器是否正常运行，确认工具名称前缀正确

**Q: SSE 连接超时**
A: 检查网络配置，确保服务器 URL 可访问

**Q: 工具不存在**
A: 确认 mcp.json 配置正确，重新启动 Agent

---

## 技术特点总结

1. **模块化设计**: Agent、Tool、Flow 分离，便于扩展
2. **工具集合管理**: 动态添加/移除工具
3. **MCP 集成**: 完全支持 Model Context Protocol
4. **多模型支持**: OpenAI、Azure、AWS Bedrock、各种兼容 API
5. **流式响应**: 支持流式输出
6. **Token 管理**: 输入/输出 token 计数和限制
7. **错误处理**: 重试机制、状态管理
8. **浏览器自动化**: 基于 Playwright
9. **沙箱安全**: Docker 隔离执行环境

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
cp config/config.example.toml config/config.toml

# 编辑配置，填入你的 API Key
```

### 3. 运行

```bash
# 标准模式
python main.py --prompt "写一个 Python 快速排序算法"

# 流程模式
python run_flow.py --prompt "分析这份数据并生成图表"

# MCP 服务器模式
python run_mcp_server.py
```

---

## 扩展开发

详细的扩展开发指南请参考以下独立文档：

### 1. 添加新工具
**文档**: [DEVELOP_TOOL.md](./DEVELOP_TOOL.md)

包含：
- 工具继承结构
- BaseTool 类的使用
- 在 Agent 中注册工具的三种方式
- 完整示例（计算器工具）
- ToolResult 响应处理
- 最佳实践

### 2. 添加新 Agent
**文档**: [DEVELOP_AGENT.md](./DEVELOP_AGENT.md)

包含：
- Agent 继承层次
- 创建简单 Agent
- 创建 ToolCallAgent（完整模板）
- 继承现有 Agent（Manus 等）
- Agent 生命周期方法
- 专用 Agent 示例（数据分析、浏览器操作）

### 3. 添加新 Flow
**文档**: [DEVELOP_FLOW.md](./DEVELOP_FLOW.md)

包含：
- Flow 基础实现
- 使用 FlowFactory 注册
- 多步骤处理流程示例
- 继承 PlanningFlow
- Agent 管理与资源清理
- 与 Tool 和外部服务集成

### 快速示例

**添加工具**:
```python
from app.tool.base import BaseTool, ToolResult

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "我的工具"
    parameters: dict = {...}

    async def execute(self, **kwargs) -> ToolResult:
        return self.success_response("结果")
```

**创建 Agent**:
```python
from app.agent.toolcall import ToolCallAgent

class MyAgent(ToolCallAgent):
    name: str = "my_agent"
    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(MyTool()))
```

**创建 Flow**:
```python
from app.flow.base import BaseFlow

class MyFlow(BaseFlow):
    async def execute(self, task: str) -> str:
        # 流程逻辑
        return "结果"
```

---

## 许可证

MIT License

---

## 贡献者

感谢所有为 OpenManus 项目做出贡献的开发者！
