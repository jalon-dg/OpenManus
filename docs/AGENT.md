# OpenManus Agent 开发指南

本文档详细介绍 OpenManus 中 Agent 模块的架构设计和开发指南。

## 目录

- [概述](#概述)
- [Agent 继承层次](#agent-继承层次)
- [架构设计](#架构设计)
  - [BaseAgent](#baseagent)
  - [ReActAgent](#reactagent)
  - [ToolCallAgent](#toolcallagent)
  - [Manus Agent](#manus-agent)
- [数据模式](#数据模式)
- [创建自定义 Agent](#创建自定义-agent)
- [最佳实践](#最佳实践)

---

## 概述

OpenManus 是一个通用的 Agent 框架，采用分层架构设计，支持灵活的 Agent 创建和扩展。Agent 通过调用大语言模型（LLM）来执行任务，并可以集成各种工具（Tools）来扩展能力。

## Agent 继承层次

```
BaseAgent (抽象基类)
    │
    ├── 核心属性:
    │   ├── name: str - Agent 名称
    │   ├── description: str - Agent 描述
    │   ├── system_prompt: str - 系统提示词
    │   ├── llm: LLM - 大语言模型实例
    │   ├── memory: Memory - 消息历史
    │   ├── state: AgentState - 当前状态
    │   ├── max_steps: int - 最大执行步数
    │   └── current_step: int - 当前步数
    │
    ├── 核心方法:
    │   ├── run(request) - 运行 Agent
    │   ├── step() - 执行单步（抽象方法）
    │   ├── think() - 思考下一步（抽象方法）
    │   └── update_memory() - 更新消息历史
    │
    └── ReActAgent (Think-Act 模式)
            │
            ├── think() - 调用 LLM 思考
            └── act() - 执行动作（抽象方法）
                    │
                    └── ToolCallAgent (工具调用)
                            │
                            ├── available_tools: ToolCollection - 可用工具
                            ├── tool_choices: ToolChoice - 工具选择策略
                            ├── special_tool_names: List[str] - 特殊工具名
                            │
                            ├── think() - 调用 LLM 获取工具调用决策
                            ├── act() - 执行工具调用
                            ├── execute_tool() - 执行单个工具
                            ├── _handle_special_tool() - 处理特殊工具
                            └── cleanup() - 清理资源
                                    │
                                    └── Manus (主 Agent)
                                    ├── mcp_clients: MCPClients - MCP 客户端
                                    ├── browser_context_helper - 浏览器上下文
                                    ├── initialize_mcp_servers() - 初始化 MCP
                                    └── connect_mcp_server() - 连接 MCP 服务器
```

### 主要 Agent 列表

| Agent | 文件 | 功能描述 |
| ----- | ---- | -------- |
| **Manus** | `app/agent/manus.py` | 通用主 Agent，支持 MCP 工具 |
| **ToolCallAgent** | `app/agent/toolcall.py` | 工具调用基础 Agent |
| **ReActAgent** | `app/agent/react.py` | ReAct (Reasoning + Acting) 模式 |
| **BaseAgent** | `app/agent/base.py` | 所有 Agent 的基类 |
| **MCPAgent** | `app/agent/mcp.py` | MCP 专用 Agent |
| **DataAnalysis** | `app/agent/data_analysis.py` | 数据分析专用 Agent |
| **BrowserAgent** | `app/agent/browser.py` | 浏览器操作 Agent |
| **SWEBenchAgent** | `app/agent/swe.py` | 软件工程 Benchmark Agent |
| **SandboxAgent** | `app/agent/sandbox_agent.py` | 沙箱执行 Agent |

---

## 架构设计

### BaseAgent

`BaseAgent` 是所有 Agent 的抽象基类，提供核心状态管理和执行循环功能。

#### 核心属性

```python
class BaseAgent(BaseModel, ABC):
    # 基本信息
    name: str                           # 唯一名称
    description: Optional[str]          # 描述

    # 提示词
    system_prompt: Optional[str]        # 系统级指令
    next_step_prompt: Optional[str]     # 下一步提示

    # 依赖组件
    llm: LLM = Field(default_factory=LLM)   # 语言模型
    memory: Memory = Field(default_factory=Memory)  # 内存

    # 状态管理
    state: AgentState = Field(default=AgentState.IDLE)
    max_steps: int = 10                 # 最大步数
    current_step: int = 0               # 当前步数
    duplicate_threshold: int = 2        # 重复检测阈值
```

#### AgentState 状态机

```
IDLE (空闲) ──────────────▶ RUNNING (运行中)
       │                          │
       │                          │
       │  完成/终止                │  错误
       │                          │
       ▼                          ▼
   FINISHED                    ERROR (错误)
   (完成)
```

#### 核心方法

##### run(request)

```python
async def run(self, request: Optional[str] = None) -> str:
    """执行 Agent 主循环"""
    if request:
        self.update_memory("user", request)

    results = []
    async with self.state_context(AgentState.RUNNING):
        while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
            self.current_step += 1
            step_result = await self.step()
            results.append(f"Step {self.current_step}: {step_result}")

    return "\n".join(results)
```

##### update_memory()

```python
def update_memory(
    self,
    role: str,           # "user", "system", "assistant", "tool"
    content: str,
    base64_image: Optional[str] = None,
    **kwargs
) -> None:
    """添加消息到历史记录"""
```

##### state_context()

```python
@asynccontextmanager
async def state_context(self, new_state: AgentState):
    """状态切换的上下文管理器"""
    previous_state = self.state
    self.state = new_state
    try:
        yield
    except Exception:
        self.state = AgentState.ERROR
        raise
    finally:
        self.state = previous_state
```

#### 死循环检测

`BaseAgent` 包含自动检测死循环的机制：

```python
def is_stuck(self) -> bool:
    """通过检测重复内容检查 Agent 是否陷入循环"""

def handle_stuck_state(self):
    """通过添加提示来改变策略，处理陷入循环的状态"""
```

---

### ReActAgent

`ReActAgent` 实现 Think-Act 模式：先思考（Reasoning），再行动（Acting）。

```python
class ReActAgent(BaseAgent, ABC):
    """Reasoning + Acting Agent"""

    # 提示词配置
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_plan_tries: int = 3
    allow_delegation: bool = False

    async def think(self) -> bool:
        """思考下一步做什么"""
        # 调用 LLM 获取思考结果
        response = await self.llm.ask(
            messages=self.messages,
            system_msgs=[Message.system_message(self.system_prompt)]
        )
        return bool(response)

    async def act(self) -> str:
        """执行思考结果"""
        raise NotImplementedError

    async def step(self) -> str:
        """执行单步：思考 + 行动"""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
```

---

### ToolCallAgent

`ToolCallAgent` 是工具调用的基础 Agent，继承自 `ReActAgent`。

#### 核心属性

```python
class ToolCallAgent(ReActAgent):
    """工具调用 Agent"""

    # 工具集合
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )

    # 工具选择策略
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO

    # 特殊工具（会触发终止）
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    # 当前工具调用
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # 最大观察数
    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
```

#### ToolChoice 策略

| 策略 | 说明 |
| ---- | ---- |
| **NONE** | 不允许使用工具 |
| **AUTO** | 自动决定是否使用工具 |
| **REQUIRED** | 必须使用工具 |

#### think() 流程

```
1. 准备消息（添加 next_step_prompt）
2. 调用 LLM.ask_tool() 获取响应
3. 解析 tool_calls
4. 添加 assistant message 到 memory
5. 返回是否需要执行工具
```

#### act() 流程

```
1. 遍历所有 tool_calls
2. 调用 execute_tool() 执行每个工具
3. 添加 tool result 到 memory
4. 返回所有结果
```

#### execute_tool() 流程

```
1. 解析工具参数 (JSON)
2. 在 available_tools 中查找工具
3. 调用 ToolCollection.execute()
4. 处理特殊工具（Terminate）
5. 返回观察结果
```

#### _handle_special_tool()

处理特殊工具（如 `terminate`）：

```python
async def _handle_special_tool(self, name: str, result: Any, **kwargs):
    """处理需要特殊处理的工具"""
    if self._is_special_tool(name):
        if self._should_finish_execution(name=name, result=result, **kwargs):
            self.state = AgentState.FINISHED

def _is_special_tool(self, name: str) -> bool:
    """检查是否是特殊工具"""
    return name.lower() in [n.lower() for n in self.special_tool_names]

def _should_finish_execution(self, **kwargs) -> bool:
    """确定是否应该结束执行"""
    return name.lower() == "terminate"
```

---

### Manus Agent

`Manus` 是通用主 Agent，支持 MCP 工具集成。

#### 核心属性

```python
class Manus(ToolCallAgent):
    """通用 Agent，支持 MCP 工具"""

    name: str = "Manus"
    description: str = "通用问题解决 Agent"

    max_observe: int = 10000
    max_steps: int = 20

    # MCP 客户端
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 可用工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    # 浏览器上下文助手
    browser_context_helper: Optional[BrowserContextHelper] = None

    # 已连接的 MCP 服务器
    connected_servers: Dict[str, str] = Field(default_factory=dict)
    _initialized: bool = False
```

#### MCP 集成

```python
class Manus(ToolCallAgent):
    async def initialize_mcp_servers(self) -> None:
        """初始化配置的 MCP 服务器"""
        for server_id, server_config in config.mcp_config.servers.items():
            if server_config.type == "sse":
                await self.connect_mcp_server(server_config.url, server_id)
            elif server_config.type == "stdio":
                await self.connect_mcp_server(
                    server_config.command,
                    server_id,
                    use_stdio=True,
                    stdio_args=server_config.args,
                )

    async def connect_mcp_server(self, server_url: str, server_id: str = "",
                                  use_stdio: bool = False, stdio_args: List[str] = None):
        """连接到 MCP 服务器"""
        if use_stdio:
            await self.mcp_clients.connect_stdio(server_url, stdio_args or [], server_id)
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)

        # 添加工具
        new_tools = [tool for tool in self.mcp_clients.tools if tool.server_id == server_id]
        self.available_tools.add_tools(*new_tools)
```

---

## 数据模式

### Message

```python
class Message(BaseModel):
    """聊天消息"""

    role: str              # "system", "user", "assistant", "tool"
    content: str           # 消息内容
    base64_image: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    @classmethod
    def system_message(cls, content: str) -> "Message"
    @classmethod
    def user_message(cls, content: str, **kwargs) -> "Message"
    @classmethod
    def assistant_message(cls, content: str, **kwargs) -> "Message"
    @classmethod
    def tool_message(cls, content: str, tool_call_id: str, name: str, **kwargs) -> "Message"
```

### Memory

```python
class Memory(BaseModel):
    """消息历史管理"""

    messages: List[Message] = Field(default_factory=list)

    def add_message(self, message: Message) -> None
    def get_messages(self) -> List[Message]
    def clear(self) -> None
```

### AgentState

```python
class AgentState(Enum):
    """Agent 状态枚举"""

    IDLE = "idle"          # 空闲
    RUNNING = "running"    # 运行中
    FINISHED = "finished"  # 完成
    ERROR = "error"        # 错误
```

### ToolCall & ToolResult

```python
class ToolCall(BaseModel):
    """工具调用"""

    id: str
    type: str = "function"
    function: FunctionCall

class FunctionCall(BaseModel):
    """函数调用"""

    name: str
    arguments: str

class ToolResult(BaseModel):
    """工具执行结果"""

    output: Any = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None
```

---

## 创建自定义 Agent

### 创建简单 Agent

继承 `BaseAgent` 实现最小化的 Agent：

```python
from app.agent.base import BaseAgent


class SimpleAgent(BaseAgent):
    """简单的 Agent 示例"""

    name: str = "simple_agent"
    description: str = "处理简单任务的 Agent"

    async def run(self, task: str) -> str:
        """执行任务"""
        # 实现核心逻辑
        return f"完成了任务: {task}"
```

### 创建 ToolCallAgent

#### 完整模板

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.base import BaseTool, ToolResult
from app.logger import logger
from pydantic import Field


class MyAgent(ToolCallAgent):
    """我的自定义 Agent"""

    # 基本信息
    name: str = "my_agent"
    description: str = "我的自定义 Agent 描述"

    # 系统提示词
    system_prompt: str = """你是一个有用的助手。
    请根据用户的需求选择合适的工具来完成任务。"""

    # 行为限制
    max_observe: int = 1000    # 最大观察消息数
    max_steps: int = 10        # 最大执行步数
    max_plan_tries: int = 3    # 最大计划尝试次数
    allow_delegation: bool = False  # 是否允许委派任务

    # 工具集合
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # 在这里添加你的工具
        )
    )

    # 特殊工具名称列表（不会被二次处理）
    special_tool_names: list[str] = Field(default_factory=list)

    async def think(self) -> bool:
        """思考下一步做什么，返回 False 则停止"""
        # 可以在这里添加自定义逻辑
        return await super().think()

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """执行工具调用"""
        # 可以在这里添加预处理/后处理逻辑
        return await super().execute(tool_name, tool_input)
```

#### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | - | Agent 名称 |
| `description` | str | - | Agent 描述 |
| `system_prompt` | str | - | 系统提示词 |
| `max_observe` | int | 1000 | 最大观察消息数 |
| `max_steps` | int | 10 | 最大执行步数 |
| `max_plan_tries` | int | 3 | 最大计划尝试次数 |
| `allow_delegation` | bool | False | 是否允许委派 |
| `available_tools` | ToolCollection | - | 可用工具集合 |
| `special_tool_names` | list[str] | [] | 特殊工具名称 |

### 继承现有 Agent

#### 扩展 Manus Agent

```python
from app.agent.manus import Manus
from app.tool import ToolCollection
from app.tool.my_custom_tool import MyCustomTool
from pydantic import Field


class MyManus(Manus):
    """扩展 Manus Agent"""

    name: str = "my_manus"
    description: str = "增强版 Manus"

    # 扩展工具集合
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # 保留原有工具
            *Manus.available_tools.default_factory().tools,
            # 添加新工具
            MyCustomTool(),
        )
    )

    # 可以覆盖父类方法
    async def think(self) -> bool:
        # 添加自定义逻辑
        return await super().think()
```

#### 扩展 ToolCallAgent

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.str_replace_editor import StrReplaceEditor
from pydantic import Field


class WebEditorAgent(ToolCallAgent):
    """专门处理网页编辑任务的 Agent"""

    name: str = "web_editor"
    description: str = "使用浏览器和编辑器处理网页任务"

    system_prompt: str = """你是一个网页编辑专家。
    你可以使用浏览器工具浏览网页，使用编辑器修改文件。
    请根据用户需求选择合适的工具。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            BrowserUseTool(),
            StrReplaceEditor(),
        )
    )
```

### Agent 使用示例

#### 基本使用

```python
import asyncio
from app.agent.my_agent import MyAgent


async def main():
    agent = MyAgent()
    result = await agent.run("你的任务描述")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

#### 自定义系统提示词

```python
agent = MyAgent(
    system_prompt="你是一个 Python 专家，只回答 Python 相关问题。"
)
```

#### 动态添加工具

```python
agent = MyAgent()
agent.available_tools.add_tools(AnotherTool())

# 执行
result = await agent.run("任务")
```

### 生命周期方法

可重写的方法：

```python
class MyAgent(ToolCallAgent):

    async def think(self) -> bool:
        """思考下一步，返回 False 停止执行"""
        return True

    async def execute(self, tool_name: str, tool_input: dict) -> ToolResult:
        """执行工具"""
        return await super().execute(tool_name, tool_input)

    async def run(self, request: str) -> str:
        """运行 Agent"""
        return await super().run(request)

    async def cleanup(self):
        """清理资源"""
        await super().cleanup()
```

### 创建专用 Agent 示例

#### 数据分析 Agent

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from pydantic import Field


class DataAnalysisAgent(ToolCallAgent):
    """数据分析专用 Agent"""

    name: str = "data_analysis"
    description: str = "处理数据分析任务"

    system_prompt: str = """你是一个数据分析专家。
    你可以使用 Python 进行数据分析，使用编辑器保存结果。
    请进行数据分析并给出清晰的结论。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            StrReplaceEditor(),
        )
    )
```

#### 浏览器操作 Agent

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.web_search import WebSearch
from pydantic import Field


class BrowserAgent(ToolCallAgent):
    """浏览器操作 Agent"""

    name: str = "browser_agent"
    description: str = "使用浏览器搜索和操作网页"

    system_prompt: str = """你是一个浏览器助手。
    用户提出需求时，使用搜索工具查找信息，或使用浏览器工具操作网页。
    请提供准确的信息和操作反馈。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            WebSearch(),
            BrowserUseTool(),
        )
    )
```

---

## LLM 模块

**文件**: `app/llm.py`

### 主要功能

- 支持多种 API 类型（OpenAI、Azure、AWS Bedrock）
- 多模态模型支持（图像、文本）
- Token 计数和管理
- 流式响应支持
- 重试机制（使用 tenacity 库）

### API 类型

| 类型 | 说明 | 配置项 |
| ---- | ---- | ------ |
| **openai** | OpenAI 兼容 API | api_key, base_url |
| **azure** | Azure OpenAI | api_key, base_url, api_version |
| **aws** | AWS Bedrock | 无 api_key，使用 IAM 认证 |

### 主要方法

```python
class LLM:
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """发送请求获取回复"""

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
        **kwargs
    ) -> ChatCompletionMessage:
        """发送请求获取工具调用"""

    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
    ) -> str:
        """发送带图像的请求"""
```

### Token 管理

```python
class LLM:
    total_input_tokens: int = 0           # 累计输入 tokens
    total_completion_tokens: int = 0      # 累计输出 tokens
    max_input_tokens: Optional[int] = None # 最大输入限制

    def count_tokens(self, text: str) -> int
    def count_message_tokens(self, messages: List[dict]) -> int
    def check_token_limit(self, input_tokens: int) -> bool
    def update_token_count(self, input_tokens: int, completion_tokens: int = 0)
```

### 支持的模型

| 类型 | 模型 |
| ---- | ---- |
| **推理模型** | o1, o3-mini |
| **多模态模型** | GPT-4o, GPT-4o-mini, Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku |

---

## 生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent 生命周期                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   创建实例 ──▶ 初始化 ──▶ 运行循环 ──▶ 清理资源                  │
│                                                                 │
│   __init__()  initialize()    run()        cleanup()           │
│      │            │              │              │              │
│      ▼            ▼              ▼              ▼              │
│   1. 创建       1. 初始化      1. 添加       1. 清理工具        │
│   2. 依赖注入   2. MCP 连接    2. think()     2. 关闭连接       │
│                  3. 工具加载    3. act()                       │
│                                   4. 循环                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 最佳实践

1. **明确职责**：每个 Agent 应该有明确的职责范围
2. **工具匹配**：为 Agent 配备与其职责相关的工具
3. **提示词设计**：设计清晰、有指导性的系统提示词
4. **参数调优**：根据任务复杂度调整 `max_steps` 等参数
5. **资源清理**：实现 `cleanup` 方法释放资源
6. **错误处理**：在 `think` 和 `execute` 中处理异常情况