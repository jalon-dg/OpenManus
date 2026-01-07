# OpenManus MCP 开发指南

本文档详细介绍 OpenManus 中 MCP（Model Context Protocol）模块的架构设计和开发指南。

## 目录

- [概述](#概述)
- [MCP 协议简介](#mcp-协议简介)
- [架构设计](#架构设计)
  - [MCPClients](#mcpclients)
  - [MCPClientTool](#mcpclienttool)
  - [MCPAgent](#mcpagent)
- [配置说明](#配置说明)
- [使用方式](#使用方式)
  - [命令行运行](#命令行运行)
  - [在代码中使用](#在代码中使用)
- [创建 MCP 服务器](#创建-mcp-服务器)
- [最佳实践](#最佳实践)

---

## 概述

MCP（Model Context Protocol）是一种开放协议，使 Agent 能够动态访问外部工具和资源。OpenManus 提供了完整的 MCP 客户端实现，支持：

- 连接到远程 MCP 服务器
- 自动发现和注册服务器工具
- 动态工具刷新
- SSE 和 Stdio 两种传输方式

## MCP 协议简介

MCP（Model Context Protocol）是 Anthropic 提出的一个开放协议，用于：

1. **工具发现**：Agent 可以发现服务器提供的所有工具
2. **工具调用**：通过标准化接口调用远程工具
3. **资源访问**：访问服务器提供的资源和上下文

### 传输方式

| 方式 | 说明 | 使用场景 |
| ---- | ---- | -------- |
| **SSE** | Server-Sent Events，通过 HTTP 长连接 | 远程服务器、Web 服务 |
| **Stdio** | 标准输入输出，本地进程通信 | 本地 MCP 服务器 |

---

## 架构设计

### MCPClients

`MCPClients` 是 MCP 客户端集合，管理多个服务器连接。

```python
from app.tool.mcp import MCPClients

mcp_clients = MCPClients()
```

#### 核心属性

```python
class MCPClients(ToolCollection):
    """MCP 客户端集合"""

    sessions: Dict[str, ClientSession] = {}           # 会话管理
    exit_stacks: Dict[str, AsyncExitStack] = {}       # 资源管理
    description: str = "MCP client tools for server interaction"
```

#### 连接方法

##### connect_sse()

通过 SSE 方式连接到 MCP 服务器：

```python
async def connect_sse(self, server_url: str, server_id: str = "") -> None:
    """连接到 MCP 服务器 (SSE)"""
    server_id = server_id or server_url

    # 如果已连接，先断开
    if server_id in self.sessions:
        await self.disconnect(server_id)

    exit_stack = AsyncExitStack()
    self.exit_stacks[server_id] = exit_stack

    # 建立 SSE 连接
    streams_context = sse_client(url=server_url)
    streams = await exit_stack.enter_async_context(streams_context)
    session = await exit_stack.enter_async_context(ClientSession(*streams))
    self.sessions[server_id] = session

    # 初始化并获取工具列表
    await self._initialize_and_list_tools(server_id)
```

##### connect_stdio()

通过 Stdio 方式连接到 MCP 服务器：

```python
async def connect_stdio(
    self, command: str, args: List[str], server_id: str = ""
) -> None:
    """连接到 MCP 服务器 (stdio)"""
    server_id = server_id or command

    if server_id in self.sessions:
        await self.disconnect(server_id)

    exit_stack = AsyncExitStack()
    self.exit_stacks[server_id] = exit_stack

    # 建立 stdio 连接
    server_params = StdioServerParameters(command=command, args=args)
    stdio_transport = await exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    read, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    self.sessions[server_id] = session

    await self._initialize_and_list_tools(server_id)
```

#### 工具管理

##### list_tools()

获取所有可用工具：

```python
async def list_tools(self) -> ListToolsResult:
    """列出所有可用工具"""
    tools_result = ListToolsResult(tools=[])
    for session in self.sessions.values():
        response = await session.list_tools()
        tools_result.tools += response.tools
    return tools_result
```

##### 内部方法

```python
async def _initialize_and_list_tools(self, server_id: str) -> None:
    """初始化会话并注册工具"""
    session = self.sessions.get(server_id)
    if not session:
        raise RuntimeError(f"Session not initialized for server {server_id}")

    await session.initialize()
    response = await session.list_tools()

    # 为每个服务器工具创建代理工具
    for tool in response.tools:
        original_name = tool.name
        tool_name = f"mcp_{server_id}_{original_name}"
        tool_name = self._sanitize_tool_name(tool_name)

        server_tool = MCPClientTool(
            name=tool_name,
            description=tool.description,
            parameters=tool.inputSchema,
            session=session,
            server_id=server_id,
            original_name=original_name,
        )
        self.tool_map[tool_name] = server_tool

    self.tools = tuple(self.tool_map.values())

def _sanitize_tool_name(self, name: str) -> str:
    """清理工具名称以符合规范"""
    import re
    # 替换非法字符为下划线
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # 移除连续的下划线
    sanitized = re.sub(r"_+", "_", sanitized)
    # 移除首尾下划线
    sanitized = sanitized.strip("_")
    # 截断到 64 字符
    if len(sanitized) > 64:
        sanitized = sanitized[:64]
    return sanitized
```

#### 断开连接

```python
async def disconnect(self, server_id: str = "") -> None:
    """断开连接"""
    if server_id:
        if server_id in self.sessions:
            exit_stack = self.exit_stacks.get(server_id)
            if exit_stack:
                await exit_stack.aclose()

            self.sessions.pop(server_id, None)
            self.exit_stacks.pop(server_id, None)

            # 移除该服务器的所有工具
            self.tool_map = {
                k: v for k, v in self.tool_map.items()
                if v.server_id != server_id
            }
            self.tools = tuple(self.tool_map.values())
    else:
        # 断开所有服务器
        for sid in sorted(list(self.sessions.keys())):
            await self.disconnect(sid)
```

---

### MCPClientTool

`MCPClientTool` 是 MCP 服务器工具的本地代理。

```python
class MCPClientTool(BaseTool):
    """MCP 工具代理"""

    session: Optional[ClientSession] = None
    server_id: str = ""
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        """执行远程工具调用"""
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            logger.info(f"Executing tool: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content
                if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")
```

---

### MCPAgent

`MCPAgent` 是专门用于连接 MCP 服务器的 Agent。

```python
from app.agent.mcp import MCPAgent

agent = MCPAgent()
```

#### 核心属性

```python
class MCPAgent(ToolCallAgent):
    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # MCP 客户端
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None

    max_steps: int = 20
    connection_type: str = "stdio"  # "stdio" or "sse"

    # 工具刷新配置
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # 每 N 步刷新一次工具

    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])
```

#### 初始化

```python
async def initialize(
    self,
    connection_type: Optional[str] = None,
    server_url: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
) -> None:
    """初始化 MCP 连接"""
    if connection_type:
        self.connection_type = connection_type

    if self.connection_type == "sse":
        if not server_url:
            raise ValueError("Server URL is required for SSE connection")
        await self.mcp_clients.connect_sse(server_url=server_url)
    elif self.connection_type == "stdio":
        if not command:
            raise ValueError("Command is required for stdio connection")
        await self.mcp_clients.connect_stdio(command=command, args=args or [])

    # 设置可用工具
    self.available_tools = self.mcp_clients

    # 刷新并存储工具 schemas
    await self._refresh_tools()

    # 添加系统消息
    tool_names = list(self.mcp_clients.tool_map.keys())
    self.memory.add_message(
        Message.system_message(
            f"{self.system_prompt}\n\nAvailable MCP tools: {', '.join(tool_names)}"
        )
    )
```

#### 工具刷新

```python
async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
    """刷新可用工具列表

    Returns:
        (added_tools, removed_tools) - 新增和移除的工具列表
    """
    if not self.mcp_clients.sessions:
        return [], []

    # 获取当前工具 schemas
    response = await self.mcp_clients.list_tools()
    current_tools = {tool.name: tool.inputSchema for tool in response.tools}

    # 比较变化
    current_names = set(current_tools.keys())
    previous_names = set(self.tool_schemas.keys())

    added_tools = list(current_names - previous_names)
    removed_tools = list(previous_names - current_names)

    # 检查现有工具的 schema 变化
    changed_tools = []
    for name in current_names.intersection(previous_names):
        if current_tools[name] != self.tool_schemas.get(name):
            changed_tools.append(name)

    # 更新存储的 schemas
    self.tool_schemas = current_tools

    # 记录变化
    if added_tools:
        logger.info(f"Added MCP tools: {added_tools}")
        self.memory.add_message(
            Message.system_message(f"New tools available: {', '.join(added_tools)}")
        )
    if removed_tools:
        logger.info(f"Removed MCP tools: {removed_tools}")
        self.memory.add_message(
            Message.system_message(
                f"Tools no longer available: {', '.join(removed_tools)}"
            )
        )

    return added_tools, removed_tools
```

#### think 方法

```python
async def think(self) -> bool:
    """处理当前状态并决定下一步操作"""
    # 检查 MCP 连接状态
    if not self.mcp_clients.sessions or not self.mcp_clients.tool_map:
        logger.info("MCP service is no longer available, ending interaction")
        self.state = AgentState.FINISHED
        return False

    # 定期刷新工具
    if self.current_step % self._refresh_tools_interval == 0:
        await self._refresh_tools()
        if not self.mcp_clients.tool_map:
            logger.info("MCP service has shut down, ending interaction")
            self.state = AgentState.FINISHED
            return False

    return await super().think()
```

#### 清理

```python
async def cleanup(self) -> None:
    """清理 MCP 连接"""
    if self.mcp_clients.sessions:
        await self.mcp_clients.disconnect()
        logger.info("MCP connection closed")
```

---

## 配置说明

### MCP 配置文件

MCP 服务器配置存储在 `config/mcp.json` 文件中：

```json
{
    "mcpServers": {
        "server_name": {
            "type": "sse",
            "url": "http://localhost:8000/sse"
        }
    }
}
```

或者使用 stdio 方式：

```json
{
    "mcpServers": {
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextplugin/server-filesystem", "/path/to/dir"]
        }
    }
}
```

### 配置类定义

```python
class MCPServerConfig(BaseModel):
    """单个 MCP 服务器配置"""
    type: str = Field(..., description="Server connection type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(default_factory=list, description="Arguments for stdio command")


class MCPSettings(BaseModel):
    """MCP 配置"""
    server_reference: str = Field(default="app.mcp.server", description="MCP 服务器模块引用")
    servers: Dict[str, MCPServerConfig] = Field(default_factory=dict, description="服务器配置字典")
```

---

## 使用方式

### 命令行运行

#### 使用 SSE 方式

```bash
python run_mcp.py --connection sse --server-url http://localhost:8000/sse
```

#### 使用 Stdio 方式

```bash
python run_mcp.py --connection stdio
```

#### 交互模式

```bash
python run_mcp.py --interactive
```

#### 单次提示

```bash
python run_mcp.py --prompt "请帮我搜索最新的 AI 新闻"
```

### 在代码中使用

#### 基本用法

```python
import asyncio
from app.agent.mcp import MCPAgent


async def main():
    # 创建 Agent
    agent = MCPAgent()

    # 初始化连接（SSE 方式）
    await agent.initialize(
        connection_type="sse",
        server_url="http://localhost:8000/sse"
    )

    # 执行任务
    result = await agent.run("请帮我完成某个任务")
    print(result)

    # 清理
    await agent.cleanup()


asyncio.run(main())
```

#### Stdio 方式

```python
import asyncio
from app.agent.mcp import MCPAgent
import sys


async def main():
    agent = MCPAgent()

    # 通过 stdio 连接（使用 Python 模块作为服务器）
    await agent.initialize(
        connection_type="stdio",
        command=sys.executable,
        args=["-m", "app.mcp.server"]  # MCP 服务器模块
    )

    result = await agent.run("使用文件系统工具列出当前目录")
    print(result)

    await agent.cleanup()


asyncio.run(main())
```

### 在 Manus Agent 中使用 MCP

`Manus` Agent 内置了 MCP 支持，可以自动连接配置的 MCP 服务器：

```python
import asyncio
from app.agent.manus import Manus


async def main():
    # Manus.create() 会自动初始化 MCP 服务器
    agent = await Manus.create()

    # 执行任务，MCP 工具会自动可用
    result = await agent.run("使用文件系统工具读取 config.toml")
    print(result)

    # 清理时会自动断开 MCP 连接
    await agent.cleanup()


asyncio.run(main())
```

#### 动态添加 MCP 服务器

```python
from app.agent.manus import Manus


async def main():
    agent = await Manus.create()

    # 动态连接额外的 MCP 服务器
    await agent.connect_mcp_server(
        server_url="http://localhost:8000/sse",
        server_id="custom_server"
    )

    # 现在可以使用新服务器的工具
    result = await agent.run("使用 custom_server 的工具")
    print(result)

    await agent.cleanup()


asyncio.run(main())
```

---

## 创建 MCP 服务器

### 简单 MCP 服务器示例

创建一个 Python 模块作为 MCP 服务器：

```python
# app/mcp/server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("my-server")


@app.list_tools()
async def list_tools():
    """列出可用工具"""
    return [
        Tool(
            name="greet",
            description="Greet a person by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """调用工具"""
    if name == "greet":
        result = f"Hello, {arguments['name']}!"
        return [TextContent(type="text", text=result)]
```

### 使用 FastMCP 创建服务器

```python
# app/mcp/server.py
from mcp.server.fastmcp import FastMCP

app = FastMCP("my-server")


@app.tool()
def greet(name: str) -> str:
    """Greet a person by name"""
    return f"Hello, {name}!"


@app.tool()
def calculate(a: int, b: int, operation: str = "add") -> dict:
    """Perform a calculation"""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    else:
        return {"error": f"Unknown operation: {operation}"}
    return {"result": result}
```

### 运行 MCP 服务器

```python
# run_mcp_server.py
import asyncio
from mcp.server.stdio import stdio_server
from app.mcp.server import app


async def main():
    async with stdio_server(app) as (read, write):
        await asyncio.Future()  # 保持运行


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 提示词模板

### SYSTEM_PROMPT

```python
SYSTEM_PROMPT = """You are an AI assistant with access to a Model Context Protocol (MCP) server.
You can use the tools provided by the MCP server to complete tasks.
The MCP server will dynamically expose tools that you can use - always check the available tools first.

When using an MCP tool:
1. Choose the appropriate tool based on your task requirements
2. Provide properly formatted arguments as required by the tool
3. Observe the results and use them to determine next steps
4. Tools may change during operation - new tools might appear or existing ones might disappear

Follow these guidelines:
- Call tools with valid parameters as documented in their schemas
- Handle errors gracefully by understanding what went wrong and trying again with corrected parameters
- For multimedia responses (like images), you'll receive a description of the content
- Complete user requests step by step, using the most appropriate tools
- If multiple tools need to be called in sequence, make one call at a time and wait for results

Remember to clearly explain your reasoning and actions to the user.
"""
```

### NEXT_STEP_PROMPT

```python
NEXT_STEP_PROMPT = """Based on the current state and available tools, what should be done next?
Think step by step about the problem and identify which MCP tool would be most helpful for the current stage.
If you've already made progress, consider what additional information you need or what actions would move you closer to completing the task.
"""
```

### TOOL_ERROR_PROMPT

```python
TOOL_ERROR_PROMPT = """You encountered an error with the tool '{tool_name}'.
Try to understand what went wrong and correct your approach.
Common issues include:
- Missing or incorrect parameters
- Invalid parameter formats
- Using a tool that's no longer available
- Attempting an operation that's not supported

Please check the tool specifications and try again with corrected parameters.
"""
```

---

## 工具调用流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP 工具调用流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 连接到 MCP 服务器                                           │
│      ├── connect_sse()  ──▶ SSE 方式                             │
│      └── connect_stdio() ──▶ Stdio 方式                          │
│                                                                 │
│   2. 初始化并获取工具列表                                         │
│      └── _initialize_and_list_tools()                            │
│           └── 创建 MCPClientTool 代理                             │
│                                                                 │
│   3. Agent 调用工具                                              │
│      └── ToolCallAgent.execute_tool()                            │
│                                                                 │
│   4. MCPClientTool.execute()                                     │
│      └── session.call_tool(original_name, kwargs)                │
│                                                                 │
│   5. 返回结果                                                    │
│      └── ToolResult(output/error)                                │
│                                                                 │
│   6. 定期刷新工具列表（每 N 步）                                   │
│      └── _refresh_tools()                                        │
│                                                                 │
│   7. 断开连接                                                    │
│      └── disconnect()                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 最佳实践

1. **连接管理**
   - 使用 `try/finally` 确保连接被正确关闭
   - 在 Agent 生命周期结束时调用 `cleanup()`

2. **错误处理**
   - 检查 `session` 是否存在再调用工具
   - 捕获并记录工具调用异常

3. **工具刷新**
   - 定期刷新工具以获取动态变化
   - 监听工具添加/移除通知

4. **命名规范**
   - MCP 工具名格式：`mcp_{server_id}_{original_name}`
   - 服务器 ID 应具有描述性

5. **资源清理**
   - 使用 `AsyncExitStack` 管理异步上下文
   - 断开连接时清理所有会话

6. **配置管理**
   - 将服务器配置放在 `config/mcp.json`
   - 使用环境变量管理敏感信息

7. **工具发现**
   - 初始化时记录可用工具
   - 工具变化时更新系统提示