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

### MCP Server (`app/mcp/server.py`)

- 使用 FastMCP 框架构建
- 支持工具注册和传输管理
- 支持 stdio 传输方式

### MCP Agent (`app/agent/mcp.py`)

- 支持连接外部 MCP 服务器
- 动态加载 MCP 服务器提供的工具

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

### 添加新工具

1. 继承 `BaseTool` 类
2. 实现 `tool_call` 方法
3. 在 Agent 中注册工具

### 添加新 Agent

1. 继承 `BaseAgent` 或 `ToolCallAgent`
2. 实现 `run` 方法
3. 配置工具集合

### 添加新 Flow

1. 继承 `BaseFlow` 类
2. 实现 `execute` 方法
3. 使用 FlowFactory 注册

---

## 许可证

MIT License

---

## 贡献者

感谢所有为 OpenManus 项目做出贡献的开发者！
