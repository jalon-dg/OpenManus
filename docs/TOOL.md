# OpenManus Tool 开发指南

本文档详细介绍 OpenManus 中 Tool 模块的架构设计和开发指南。

## 目录

- [概述](#概述)
- [Tool 继承层次](#tool-继承层次)
- [架构设计](#架构设计)
  - [BaseTool](#basetool)
  - [ToolCollection](#toolcollection)
  - [ToolResult](#toolresult)
- [内置工具](#内置工具)
- [工具分类](#工具分类)
- [MCP 集成](#mcp-集成)
- [创建自定义工具](#创建自定义工具)
- [工具使用流程](#工具使用流程)
- [最佳实践](#最佳实践)

---

## 概述

Tool（工具）是 OpenManus 中扩展 Agent 能力的核心模块。Agent 通过调用 Tool 来执行具体的操作，如执行代码、操作文件、浏览网页等。

**工具的核心功能：**

- 提供标准化的接口定义
- 支持参数验证和类型检查
- 返回结构化的执行结果
- 支持工具集合管理
- 支持 MCP 协议远程工具

## Tool 继承层次

```
BaseTool (抽象基类)
    │
    ├── name: str - 工具名称
    ├── description: str - 工具描述
    ├── parameters: dict - 参数 schema
    │
    ├── async execute(**kwargs) -> ToolResult - 执行工具（抽象方法）
    ├── async __call__(**kwargs) -> Any - 调用入口
    ├── to_param() -> dict - 转换为 OpenAI 函数格式
    ├── success_response(data) -> ToolResult - 成功响应
    └── fail_response(msg) -> ToolResult - 失败响应
            │
            ├── PythonExecute - Python 代码执行
            ├── BrowserUseTool - 浏览器自动化
            ├── StrReplaceEditor - 文件编辑
            ├── Bash - Shell 命令执行
            ├── WebSearch - 网络搜索
            ├── CreateChatCompletion - LLM 调用
            ├── PlanningTool - 任务规划
            ├── Terminate - 终止任务
            ├── AskHuman - 请求人类输入
            ├── MCPClientTool - MCP 客户端代理
            ├── ComputerUseTool - 计算机使用工具
            ├── Crawl4ai - 网页抓取工具
            ├── FileOperators - 文件操作工具
            ├── ChartVisualization - 图表可视化工具
            │
            └── MCPClients (继承 ToolCollection)
                │
                ├── sessions: Dict[str, ClientSession]
                ├── exit_stacks: Dict[str, AsyncExitStack]
                │
                ├── connect_sse() - SSE 连接
                ├── connect_stdio() - Stdio 连接
                ├── disconnect() - 断开连接
                └── list_tools() - 列出工具
```

---

## 架构设计

### BaseTool

`BaseTool` 是所有工具的抽象基类，提供标准化的工具接口。

#### 核心属性

```python
class BaseTool(ABC, BaseModel):
    """所有工具的基类"""

    name: str                                    # 工具唯一名称
    description: str                             # 工具描述
    parameters: Optional[dict] = None            # 参数 schema (OpenAI 格式)
```

#### 核心方法

```python
class BaseTool(ABC, BaseModel):
    async def __call__(self, **kwargs) -> Any:
        """执行工具"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具逻辑（子类实现）"""
        pass

    def to_param(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def success_response(self, data: Union[Dict[str, Any], str]) -> ToolResult:
        """创建成功响应"""
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data, indent=2)
        return ToolResult(output=text)

    def fail_response(self, msg: str) -> ToolResult:
        """创建失败响应"""
        return ToolResult(error=msg)
```

---

### ToolCollection

`ToolCollection` 用于管理和组织多个工具。

```python
class ToolCollection:
    """工具集合，管理多个工具"""

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        """获取所有工具参数"""
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        """执行指定工具"""
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """执行所有工具"""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def add_tool(self, tool: BaseTool):
        """添加工具"""
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists, skipping")
            return
        self.tools += (tool,)
        self.tool_map[tool.name] = tool

    def add_tools(self, *tools: BaseTool):
        """批量添加工具"""
        for tool in tools:
            self.add_tool(tool)

    def get_tool(self, name: str) -> BaseTool:
        """获取工具"""
        return self.tool_map.get(name)
```

---

### ToolResult

`ToolResult` 表示工具的执行结果。

```python
class ToolResult(BaseModel):
    """工具执行结果"""

    output: Any = Field(default=None)            # 成功输出
    error: Optional[str] = Field(default=None)   # 错误信息
    base64_image: Optional[str] = Field(default=None)  # Base64 图像
    system: Optional[str] = Field(default=None)  # 系统消息

    def __bool__(self):
        """判断是否有有效结果"""
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        """合并结果"""
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """替换字段"""
        return type(self)(**{**self.dict(), **kwargs})

    def __str__(self):
        return f"Error: {self.error}" if self.error else self.output
```

#### 其他结果类型

```python
class CLIResult(ToolResult):
    """可渲染为 CLI 输出的 ToolResult"""

class ToolFailure(ToolResult):
    """表示失败的 ToolResult"""
```

---

## 内置工具

### PythonExecute

执行 Python 代码的沙箱环境工具。

```python
class PythonExecute(BaseTool):
    """执行 Python 代码"""

    name: str = "python"
    description: str = "Execute Python code in a sandboxed environment."

    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            }
        },
        "required": ["code"]
    }

    async def execute(self, code: str) -> ToolResult:
        """执行 Python 代码"""
        result = await self._run_in_sandbox(code)
        return self.success_response(result)
```

### BrowserUseTool

浏览器自动化工具，支持网页导航和交互。

```python
class BrowserUseTool(BaseTool):
    """浏览器自动化工具"""

    name: str = "browser"
    description: str = "Use browser to navigate to URLs and interact with pages"

    parameters: dict = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What to achieve"},
            "url": {"type": "string", "description": "URL to navigate to"},
            "step": {"type": "integer", "description": "Current step"}
        }
    }

    async def execute(self, goal: str, url: str = "", **kwargs) -> ToolResult:
        """执行浏览器操作（基于 Playwright）"""
        pass
```

### StrReplaceEditor

文件编辑工具，支持创建、查看、编辑、搜索等操作。

```python
class StrReplaceEditor(BaseTool):
    """文件编辑工具"""

    name: str = "str_replace_editor"
    description: str = "Edit a file by replacing strings"

    async def execute(
        self,
        command: str,
        path: str = "",
        text: str = "",
        replace: str = "",
        **kwargs
    ) -> ToolResult:
        """编辑文件

        支持的命令:
        - create: 创建新文件
        - view: 查看文件内容
        - edit: 编辑文件
        - glob: 查找文件
        - lf递归查询: 递归搜索文件
        - insert: 插入内容
        - undo: 撤销操作
        """
        pass
```

### Bash

Shell 命令执行工具。

```python
class Bash(BaseTool):
    """Shell 命令执行工具"""

    name: str = "bash"
    description: str = "Execute shell commands"

    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"}
        },
        "required": ["command"]
    }

    async def execute(self, command: str, timeout: int = 60) -> ToolResult:
        """执行 shell 命令"""
        pass
```

### WebSearch

网络搜索工具。

```python
class WebSearch(BaseTool):
    """网络搜索工具"""

    name: str = "web_search"
    description: str = "Search the web for information"

    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }

    async def execute(self, query: str) -> ToolResult:
        """执行搜索"""
        pass
```

### CreateChatCompletion

LLM 调用工具。

```python
class CreateChatCompletion(BaseTool):
    """LLM 调用工具"""

    name: str = "create_chat_completion"
    description: str = "Create a chat completion"

    async def execute(
        self,
        messages: List[dict],
        model: str = "gpt-4",
        temperature: float = 1.0,
        **kwargs
    ) -> ToolResult:
        """调用 LLM"""
        pass
```

### PlanningTool

任务规划工具，用于创建和管理执行计划。

```python
class PlanningTool(BaseTool):
    """任务规划工具"""

    name: str = "planning"
    description: str = "Create and manage plans"

    async def execute(
        self,
        command: str,
        plan_id: str = "",
        title: str = "",
        steps: List[str] = None,
        **kwargs
    ) -> ToolResult:
        """执行规划操作

        支持的命令:
        - create: 创建计划
        - get: 获取计划
        - update: 更新计划
        - mark_step: 标记步骤状态
        - delete: 删除计划
        """
        pass
```

### Terminate

终止任务工具。

```python
class Terminate(BaseTool):
    """终止工具"""

    name: str = "terminate"
    description: str = "Finish the task"

    async def execute(self, thought: str = "") -> ToolResult:
        """终止任务"""
        return self.success_response("Task terminated by user")
```

### AskHuman

请求人类输入工具。

```python
class AskHuman(BaseTool):
    """请求人类输入工具"""

    name: str = "ask_human"
    description: str = "Ask human for input"

    async def execute(
        self,
        question: str,
        input_type: str = "text",
        **kwargs
    ) -> ToolResult:
        """请求人类输入

        暂停执行，等待用户输入。
        """
        pass
```

---

## 工具分类

### 搜索工具 (`app/tool/search/`)

| 工具 | 文件 | 功能 |
| ---- | ---- | ---- |
| BaiduSearch | `baidu_search.py` | 百度搜索 |
| BingSearch | `bing_search.py` | Bing 搜索 |
| DuckDuckGoSearch | `duckduckgo_search.py` | DuckDuckGo 搜索 |
| GoogleSearch | `google_search.py` | Google 搜索 |

### 沙箱工具 (`app/tool/sandbox/`)

| 工具 | 功能 |
| ---- | ---- |
| SandboxBrowser | 隔离浏览器环境 |
| SandboxFiles | 隔离文件系统 |
| SandboxShell | 隔离 Shell |
| SandboxVision | 隔离视觉处理 |

### 专用工具

| 工具 | 文件 | 功能 |
| ---- | ---- | ---- |
| ChartVisualization | `chart_visualization/` | 图表可视化 |
| FileOperators | `file_operators.py` | 文件操作 |
| ComputerUseTool | `computer_use_tool.py` | 计算机使用 |
| Crawl4ai | `crawl4ai.py` | 网页抓取 |

---

## MCP 集成

### MCPClientTool

MCP 客户端代理工具，用于调用远程 MCP 服务器上的工具。

```python
class MCPClientTool(BaseTool):
    """MCP 客户端代理工具"""

    session: Optional[ClientSession] = None
    server_id: str = ""
    original_name: str = ""

    name: str                          # mcp_{server_id}_{original_name}
    description: str
    parameters: dict                   # 从 MCP 服务器获取

    async def execute(self, **kwargs) -> ToolResult:
        """通过 MCP 协议调用远程工具"""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        # 调用远程工具
        result = await self.session.call_tool(self.original_name, kwargs)
        content_str = ", ".join(
            item.text for item in result.content if isinstance(item, TextContent)
        )
        return ToolResult(output=content_str or "No output returned.")
```

### MCPClients

MCP 客户端集合，管理多个服务器连接。

```python
class MCPClients(ToolCollection):
    """MCP 客户端集合，管理多个服务器连接"""

    sessions: Dict[str, ClientSession] = {}
    exit_stacks: Dict[str, AsyncExitStack] = {}

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """连接到 MCP 服务器 (SSE)"""
        server_id = server_id or server_url

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        # 建立 SSE 连接
        streams = await exit_stack.enter_async_context(sse_client(url=server_url))
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        # 列出并注册工具
        await self._initialize_and_list_tools(server_id)

    async def connect_stdio(self, command: str, args: List[str], server_id: str = "") -> None:
        """连接到 MCP 服务器 (stdio)"""
        server_id = server_id or command

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        # 建立 stdio 连接
        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(*stdio_transport))
        self.sessions[server_id] = session

        # 列出并注册工具
        await self._initialize_and_list_tools(server_id)

    async def _initialize_and_list_tools(self, server_id: str) -> None:
        """初始化会话并注册工具"""
        session = self.sessions[server_id]
        await session.initialize()
        response = await session.list_tools()

        for tool in response.tools:
            # 转换工具名
            tool_name = f"mcp_{server_id}_{tool.name}"
            tool_name = self._sanitize_tool_name(tool_name)

            # 创建代理工具
            mcp_tool = MCPClientTool(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=tool.name,
            )
            self.tool_map[tool_name] = mcp_tool

        self.tools = tuple(self.tool_map.values())

    async def disconnect(self, server_id: str = "") -> None:
        """断开连接"""
        if server_id:
            # 断开指定服务器
            if server_id in self.sessions:
                exit_stack = self.exit_stacks.get(server_id)
                if exit_stack:
                    await exit_stack.aclose()
                self.sessions.pop(server_id, None)
                self.exit_stacks.pop(server_id, None)
        else:
            # 断开所有服务器
            for sid in sorted(list(self.sessions.keys())):
                await self.disconnect(sid)
```

---

## 创建自定义工具

### 步骤 1：创建工具类

继承 `BaseTool` 类，定义必要的属性和方法：

```python
from app.tool.base import BaseTool, ToolResult


class MyCustomTool(BaseTool):
    """自定义工具示例"""

    # 工具名称，必须唯一
    name: str = "my_custom_tool"

    # 工具描述，会发送给 LLM 用于决定是否调用
    description: str = "这是一个自定义工具的描述，解释工具的用途"

    # 参数 schema（OpenAI function calling 格式）
    parameters: dict = {
        "type": "object",
        "properties": {
            "input_text": {
                "type": "string",
                "description": "输入文本的说明"
            }
        },
        "required": ["input_text"]  # 必填参数列表
    }

    async def execute(self, input_text: str) -> ToolResult:
        """执行工具逻辑

        Args:
            input_text: 从 parameters 中定义的参数

        Returns:
            ToolResult 对象
        """
        # 工具逻辑实现
        result = f"处理了: {input_text}"

        # 使用成功响应 helper
        return self.success_response(result)

        # 或者返回失败响应
        # return self.fail_response("错误信息")
```

### 步骤 2：在 Agent 中注册工具

#### 方式一：在 Agent 定义时静态注册

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.my_custom_tool import MyCustomTool


class MyAgent(ToolCallAgent):
    name: str = "my_agent"
    description: str = "我的自定义 Agent"

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            MyCustomTool(),      # 你的自定义工具
            # 可以添加多个工具...
        )
    )
```

#### 方式二：运行时动态添加工具

```python
# 创建 Agent 实例后，使用 add_tools 方法动态添加
agent = MyAgent()
agent.available_tools.add_tools(MyCustomTool())

# 也可以一次性添加多个工具
agent.available_tools.add_tools(
    ToolA(),
    ToolB(),
    ToolC()
)

# 检查工具是否存在
if agent.available_tools.get_tool("tool_name"):
    print("工具已存在")
```

#### 方式三：继承现有 Agent（如 Manus）

```python
from app.agent.manus import Manus
from app.tool.my_custom_tool import MyCustomTool


class MyManus(Manus):
    """扩展 Manus Agent，添加自定义工具"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # 使用解包保留原有工具
            *Manus.available_tools.default_factory().tools,
            # 添加新工具
            MyCustomTool(),
        )
    )
```

### 完整示例：计算器工具

#### 创建工具文件 `app/tool/calculator.py`

```python
from app.tool.base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """计算器工具 - 执行简单的数学计算"""

    name: str = "calculator"
    description: str = "执行简单的数学计算，支持加减乘除和括号"
    parameters: dict = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 '(10 + 5) / 2'"
            }
        },
        "required": ["expression"]
    }

    async def execute(self, expression: str) -> ToolResult:
        """计算数学表达式"""
        try:
            # 注意：eval() 仅用于演示，生产环境应使用安全的计算方式
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return self.fail_response("表达式包含非法字符")

            result = eval(expression)
            return self.success_response({
                "expression": expression,
                "result": result
            })
        except Exception as e:
            return self.fail_response(f"计算错误: {str(e)}")
```

#### 在 Agent 中使用

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection
from app.tool.calculator import CalculatorTool


class CalculatorAgent(ToolCallAgent):
    """专门处理数学计算任务的 Agent"""

    name: str = "calculator_agent"
    description: str = "专门处理数学计算问题"
    system_prompt: str = """你是一个专业的数学助手。
    当用户提出计算请求时，使用 calculator 工具进行计算。
    保持回答简洁准确。"""

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            CalculatorTool(),
        )
    )
```

#### 测试

```python
import asyncio
from app.agent.calculator_agent import CalculatorAgent


async def main():
    agent = CalculatorAgent()
    result = await agent.run("计算 100 * 50 + 25")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

### ToolResult 响应

#### 成功响应

```python
# 返回字符串
return self.success_response("处理完成")

# 返回字典（会自动 JSON 格式化）
return self.success_response({
    "status": "success",
    "data": {"key": "value"}
})
```

#### 失败响应

```python
return self.fail_response("错误信息")
```

#### ToolResult 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `output` | Any | 成功时的输出 |
| `error` | str | 错误信息 |
| `base64_image` | str | Base64 编码的图像 |
| `system` | str | 系统消息 |

#### 合并结果

```python
result1 = ToolResult(output="第一部分")
result2 = ToolResult(output="第二部分")
combined = result1 + result2  # output: "第一部分第二部分"
```

### 工具放置位置

| 类型 | 位置 |
|------|------|
| 通用工具 | `app/tool/` |
| 搜索工具 | `app/tool/search/` |
| 沙箱工具 | `app/tool/sandbox/` |
| 专用工具 | `app/tool/chart_visualization/` |

### 完整工具模板

```python
from app.tool.base import BaseTool, ToolResult
from app.logger import logger


class MyTool(BaseTool):
    """工具描述"""

    name: str = "my_tool"
    description: str = "工具的详细描述"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1说明"
            },
            "param2": {
                "type": "integer",
                "description": "参数2说明"
            }
        },
        "required": ["param1"]
    }

    async def execute(self, param1: str, param2: int = 0) -> ToolResult:
        """工具执行逻辑"""
        logger.info(f"执行工具: {param1}")

        try:
            # 实现逻辑
            result = do_something(param1, param2)
            return self.success_response(result)
        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return self.fail_response(str(e))
```

---

## 工具使用流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      工具调用流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Agent.think()                                              │
│      └── LLM.ask_tool() ──▶ 获取工具调用决策                      │
│                                                                 │
│   2. ToolCallAgent.act()                                        │
│      └── 解析 tool_calls                                        │
│                                                                 │
│   3. ToolCallAgent.execute_tool()                               │
│      └── 查找工具: tool_map[name]                                │
│                                                                 │
│   4. ToolCollection.execute()                                   │
│      └── 调用工具: await tool.execute(**args)                    │
│                                                                 │
│   5. BaseTool.execute()                                         │
│      └── 执行业务逻辑                                            │
│                                                                 │
│   6. 返回 ToolResult                                            │
│      └── output / error / base64_image                          │
│                                                                 │
│   7. Message.tool_message()                                     │
│      └── 添加工具结果到 Memory                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 最佳实践

1. **参数验证**：在 `execute` 方法中验证输入参数

2. **错误处理**：使用 `try/except` 捕获异常，返回 `fail_response`

3. **安全计算**：避免使用 `eval()` 处理用户输入，生产环境应使用 ast.literal_eval 或专门的数学解析库

4. **日志记录**：使用 `logger` 记录关键操作

5. **返回值结构**：返回结构化的字典，便于后续处理

6. **命名规范**：工具名称应唯一，使用小写字母和下划线

7. **描述清晰**：提供清晰的工具描述，帮助 LLM 决定何时使用该工具

8. **参数设计**：设计合理的参数 schema，必填参数应有说明

9. **资源清理**：如果工具使用外部资源，实现 `cleanup` 方法进行清理

10. **异步执行**：所有工具方法应为异步方法（async）