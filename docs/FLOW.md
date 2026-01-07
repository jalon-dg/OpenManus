# OpenManus Flow 开发指南

本文档详细介绍 OpenManus 中 Flow 模块的架构设计和开发指南。

## 目录

- [概述](#概述)
- [Flow 继承层次](#flow-继承层次)
- [架构设计](#架构设计)
  - [BaseFlow](#baseflow)
  - [PlanningFlow](#planningflow)
  - [FlowFactory](#flowfactory)
- [数据模式](#数据模式)
- [创建自定义 Flow](#创建自定义-flow)
- [使用示例](#使用示例)
- [最佳实践](#最佳实践)

---

## 概述

Flow（流程）是 OpenManus 中用于编排多个 Agent 或步骤的高级模块，实现复杂的任务流程控制。Flow 可以：

- 管理多个 Agent 的协作
- 实现分步骤的任务规划与执行
- 支持计划创建、执行状态追踪和完成汇总
- 提供灵活的流程定制能力

## Flow 继承层次

```
BaseFlow (抽象基类)
    │
    ├── agents: Dict[str, BaseAgent]    - 管理的 Agent
    ├── tools: Optional[List]            - 工具集合
    ├── primary_agent_key: str           - 主 Agent 键
    │
    ├── primary_agent -> BaseAgent       - 获取主 Agent
    ├── get_agent(key) -> BaseAgent      - 获取指定 Agent
    ├── add_agent(key, agent)            - 添加 Agent
    │
    └── execute(input_text) -> str       - 执行流程（抽象方法）
            │
            └── PlanningFlow
                │
                ├── llm: LLM                     - 规划用 LLM
                ├── planning_tool: PlanningTool  - 规划工具
                ├── executor_keys: List[str]     - 执行器键列表
                ├── active_plan_id: str          - 当前计划 ID
                ├── current_step_index: int      - 当前步骤索引
                │
                ├── execute(input_text)          - 执行规划流程
                ├── _create_initial_plan()       - 创建初始计划
                ├── _get_current_step_info()     - 获取当前步骤
                ├── _execute_step()              - 执行步骤
                ├── _mark_step_completed()       - 标记完成
                ├── _get_plan_text()             - 获取计划文本
                └── _finalize_plan()             - 完成计划
```

---

## 架构设计

### BaseFlow

`BaseFlow` 是所有 Flow 的抽象基类，提供多 Agent 管理的基础功能。

#### 核心属性

```python
class BaseFlow(BaseModel, ABC):
    """Base class for execution flows supporting multiple agents"""

    agents: Dict[str, BaseAgent]           # 管理的 Agent 字典
    tools: Optional[List] = None           # 可选工具列表
    primary_agent_key: Optional[str] = None # 主 Agent 键
```

#### 初始化

```python
class BaseFlow(BaseModel, ABC):
    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 处理不同形式的 agents 参数
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        # 如果主 Agent 未指定，使用第一个 Agent
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        data["agents"] = agents_dict
        super().__init__(**data)
```

#### 核心方法

```python
class BaseFlow(BaseModel, ABC):
    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取主 Agent"""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """获取指定 Agent"""
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """添加 Agent"""
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """执行流程（子类实现）"""
        pass
```

---

### PlanningFlow

`PlanningFlow` 是基于规划的任务执行流程，支持自动计划生成和多 Agent 协作执行。

#### 核心属性

```python
class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    # LLM 和规划工具
    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)

    # 执行器配置
    executor_keys: List[str] = Field(default_factory=list)

    # 计划状态
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
```

#### PlanStepStatus 状态枚举

```python
class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"      # 未开始
    IN_PROGRESS = "in_progress"      # 进行中
    COMPLETED = "completed"          # 完成
    BLOCKED = "blocked"              # 阻塞

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }
```

#### 执行流程

```python
class PlanningFlow(BaseFlow):
    async def execute(self, input_text: str) -> str:
        """执行规划流程"""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # 1. 如果有输入，创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

            result = ""

            # 2. 循环执行计划步骤
            while True:
                # 获取当前步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 没有更多步骤，结束
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 3. 选择执行器
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)

                # 4. 执行步骤
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 5. 检查是否需要终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result

        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"
```

#### 创建计划

```python
class PlanningFlow(BaseFlow):
    async def _create_initial_plan(self, request: str) -> None:
        """使用 LLM 创建初始计划"""
        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )

        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append({
                    "name": key.upper(),
                    "description": self.agents[key].description,
                })

        if len(agents_description) > 1:
            system_message_content += (
                f"\nNow we have {agents_description} agents. "
                "When creating steps in the planning tool, "
                "please specify the agent names using the format '[agent_name]'."
            )

        # 调用 LLM 生成计划
        response = await self.llm.ask_tool(
            messages=[Message.user_message(f"Create a plan for: {request}")],
            system_msgs=[Message.system_message(system_message_content)],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 执行计划创建工具
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    args = json.loads(tool_call.function.arguments)
                    args["plan_id"] = self.active_plan_id
                    await self.planning_tool.execute(**args)
```

#### 获取当前步骤

```python
class PlanningFlow(BaseFlow):
    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """获取第一个未完成的步骤"""
        plan_data = self.planning_tool.plans[self.active_plan_id]
        steps = plan_data.get("steps", [])
        step_statuses = plan_data.get("step_statuses", [])

        # 查找第一个活动状态的步骤
        for i, step in enumerate(steps):
            status = step_statuses[i] if i < len(step_statuses) else PlanStepStatus.NOT_STARTED.value

            if status in PlanStepStatus.get_active_statuses():
                step_info = {"text": step}

                # 提取步骤类型（如 [SEARCH], [CODE]）
                import re
                type_match = re.search(r"\[([A-Z_]+)\]", step)
                if type_match:
                    step_info["type"] = type_match.group(1).lower()

                # 标记为进行中
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=i,
                    step_status=PlanStepStatus.IN_PROGRESS.value,
                )

                return i, step_info

        return None, None
```

#### 执行步骤

```python
class PlanningFlow(BaseFlow):
    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行单个步骤"""
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please only execute this current step using the appropriate tools.
        When you're done, provide a summary of what you accomplished.
        """

        try:
            step_result = await executor.run(step_prompt)
            await self._mark_step_completed()
            return step_result

        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error: {str(e)}"
```

#### 获取执行器

```python
class PlanningFlow(BaseFlow):
    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """根据步骤类型选择执行器"""
        # 1. 如果指定了类型且匹配，优先使用
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 2. 按执行器键顺序查找
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 3. 使用主 Agent
        return self.primary_agent
```

#### 完成计划

```python
class PlanningFlow(BaseFlow):
    async def _finalize_plan(self) -> str:
        """完成计划并生成总结"""
        plan_text = await self._get_plan_text()

        try:
            system_message = Message.system_message(
                "You are a planning assistant. Summarize the completed plan."
            )
            user_message = Message.user_message(
                f"The plan has been completed:\n\n{plan_text}\n\n"
                "Please provide a summary of what was accomplished."
            )

            response = await self.llm.ask(
                messages=[user_message],
                system_msgs=[system_message]
            )
            return f"Plan completed:\n\n{response}"

        except Exception as e:
            logger.error(f"Error finalizing plan: {e}")
            return "Plan completed. Error generating summary."
```

---

### FlowFactory

```python
class FlowType(str, Enum):
    PLANNING = "planning"


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        """创建指定类型的 Flow 实例"""
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)
```

---

## 数据模式

### PlanStepStatus 状态机

```
NOT_STARTED ──────────▶ IN_PROGRESS ──────────▶ COMPLETED
      │                       │                     │
      │                       │ 阻塞                │
      │                       ▼                     │
      │                   BLOCKED ◀─────────────────┘
      │                       │
      │                       │ 解决阻塞
      └───────────────────────┘

状态显示标记:
  [✓] COMPLETED   - 已完成
  [→] IN_PROGRESS - 进行中
  [!] BLOCKED     - 阻塞
  [ ] NOT_STARTED - 未开始
```

---

## 创建自定义 Flow

### 创建基础 Flow

#### 最小实现

```python
from app.flow.base import BaseFlow


class MyFlow(BaseFlow):
    """我的自定义流程"""

    name: str = "my_flow"
    description: str = "我的自定义流程描述"

    async def execute(self, task: str) -> str:
        """执行流程"""
        # 实现流程逻辑
        result = await self.step1(task)
        result = await self.step2(result)
        return result

    async def step1(self, task: str) -> str:
        """步骤1"""
        return f"步骤1处理: {task}"

    async def step2(self, input_data: str) -> str:
        """步骤2"""
        return f"步骤2处理: {input_data}"
```

### 使用 FlowFactory 注册 Flow

```python
from app.flow.flow_factory import FlowFactory, FlowType

# 注册自定义 Flow
FlowFactory.create_flow(FlowType.PLANNING, agents)
```

### 完整示例：多步骤处理流程

```python
from app.flow.base import BaseFlow
from app.agent.manus import Manus
from app.agent.data_analysis import DataAnalysis


class MultiStepFlow(BaseFlow):
    """多步骤任务处理流程"""

    name: str = "multi_step"
    description: str = "分步骤处理复杂任务"

    # 配置
    use_data_analysis_agent: bool = False
    timeout: int = 3600  # 1小时超时

    async def execute(self, task: str) -> str:
        """执行多步骤流程"""
        # 步骤1: 分析任务并规划
        plan = await self._create_plan(task)
        results = []

        # 步骤2: 逐步执行
        for step in plan:
            step_result = await self._execute_step(step)
            results.append(step_result)

        # 步骤3: 汇总结果
        summary = self._summarize_results(results)
        return summary

    async def _create_plan(self, task: str) -> list[dict]:
        """创建执行计划"""
        return [
            {"step": 1, "description": "分析任务需求"},
            {"step": 2, "description": "执行具体操作"},
            {"step": 3, "description": "验证结果"}
        ]

    async def _execute_step(self, step: dict) -> dict:
        """执行单个步骤"""
        description = step.get("description", "")

        # 根据步骤类型选择 Agent
        if "分析" in description:
            agent = self.get_agent("data_analysis")
        else:
            agent = self.primary_agent

        result = await agent.run(description)
        return {
            "step": step["step"],
            "description": description,
            "result": result
        }

    def _summarize_results(self, results: list[dict]) -> str:
        """汇总结果"""
        summary = "执行结果汇总:\n\n"
        for r in results:
            summary += f"步骤 {r['step']}: {r['description']}\n"
            summary += f"结果: {r['result'][:200]}...\n\n"
        return summary
```

### 继承 PlanningFlow

```python
from app.flow.planning import PlanningFlow
from app.agent.manus import Manus


class CustomPlanningFlow(PlanningFlow):
    """自定义规划流程"""

    name: str = "custom_planning"
    description: str = "带有数据分析的规划流程"

    # 配置选项
    use_data_analysis_agent: bool = True

    async def _create_agents(self) -> dict:
        """创建 Agent 集合"""
        agents = {
            "manus": await Manus.create()
        }
        return agents

    async def _execute_step(
        self, step: dict, agents: dict, current_state: dict
    ) -> dict:
        """执行单个步骤"""
        step_type = step.get("type", "default")
        agent = agents.get(step_type, self.primary_agent)

        if not agent:
            return {"success": False, "error": "Agent not found"}

        result = await agent.run(step.get("description", ""))
        return {"success": True, "result": result}
```

### Flow 中的 Agent 管理

```python
class AgentFlow(BaseFlow):
    """管理 Agent 生命周期的 Flow"""

    name: str = "agent_flow"
    description: str = "管理多个 Agent"

    async def execute(self, task: str) -> str:
        """执行流程"""
        try:
            # 创建 Agent
            agents = await self._create_agents()
            result = await self._use_agents(task, agents)
            return result
        finally:
            # 清理 Agent
            await self._cleanup_agents()

    async def _create_agents(self) -> dict:
        """创建 Agent"""
        return {
            "agent1": await Manus.create(),
        }

    async def _use_agents(self, task: str, agents: dict) -> str:
        """使用 Agent"""
        result = await agents["agent1"].run(task)
        return result

    async def _cleanup_agents(self):
        """清理 Agent 资源"""
        for agent in self.agents.values():
            if hasattr(agent, "cleanup"):
                await agent.cleanup()
```

### 配置 Flow

```python
# 在 config.toml 中配置
[flow.default]
timeout = 3600

[flow.planning]
max_steps = 10
```

---

## 使用示例

### 基本使用

```python
import asyncio
from app.flow.planning import PlanningFlow
from app.agent.manus import Manus


async def main():
    # 创建 Agent
    manus = await Manus.create()

    # 创建 Flow
    flow = PlanningFlow(agents=manus)

    # 执行
    result = await flow.execute("分析销售数据并生成图表")
    print(result)


asyncio.run(main())
```

### 多 Agent 协作

```python
import asyncio
from app.flow.planning import PlanningFlow
from app.agent.manus import Manus
from app.agent.browser import BrowserAgent


async def main():
    # 创建多个专业 Agent
    manus = await Manus.create()
    browser_agent = BrowserAgent()

    # 创建 Flow
    flow = PlanningFlow(
        agents={
            "manus": manus,
            "browser": browser_agent,
        },
        executor_keys=["browser", "manus"],  # 指定执行顺序
    )

    # 执行任务，Flow 会自动选择合适的 Agent
    result = await flow.execute("搜索最新 AI 新闻并总结")
    print(result)


asyncio.run(main())
```

### 自定义 Flow

```python
import asyncio
from app.flow.base import BaseFlow


class MyFlow(BaseFlow):
    """自定义流程"""

    async def execute(self, task: str) -> str:
        """执行流程"""
        results = []

        # 使用主 Agent
        primary = self.primary_agent
        if primary:
            result = await primary.run(f"第一步: {task}")
            results.append(result)

        # 使用其他 Agent
        for key, agent in self.agents.items():
            if key != self.primary_agent_key:
                result = await agent.run(f"处理: {task}")
                results.append(result)

        return "\n\n".join(results)


async def main():
    agent = await Manus.create()
    flow = MyFlow(agents=agent)

    result = await flow.execute("我的任务")
    print(result)


asyncio.run(main())
```

### 与 Tool 集成

```python
from app.flow.base import BaseFlow
from app.tool import ToolCollection
from app.tool.web_search import WebSearch


class SearchFlow(BaseFlow):
    """搜索流程"""

    name: str = "search_flow"
    description: str = "多引擎搜索流程"

    async def execute(self, query: str) -> str:
        """执行搜索"""
        # 直接使用工具
        web_search = WebSearch()
        result = await web_search.execute(query=query)
        return str(result)
```

### 与外部服务集成

```python
from app.flow.base import BaseFlow
import aiohttp


class APIFlow(BaseFlow):
    """调用外部 API 的流程"""

    name: str = "api_flow"
    description: str = "调用外部 API"

    async def execute(self, request: str) -> str:
        """调用 API"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.example.com/endpoint",
                params={"q": request}
            ) as response:
                data = await response.json()
                return str(data)
```

---

## 流程状态图

```
┌─────────────────────────────────────────────────────────────────┐
│                      PlanningFlow 状态机                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   execute()                                                     │
│       │                                                         │
│       ▼                                                         │
│   _create_initial_plan()                                        │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────────────┐                                   │
│   │    循环执行计划步骤       │                                   │
│   │  ┌───────────────────┐  │                                   │
│   │  │ _get_current_step │  │                                   │
│   │  └─────────┬─────────┘  │                                   │
│   │            │            │                                   │
│   │            ▼            │                                   │
│   │  ┌───────────────────┐  │                                   │
│   │  │  get_executor()   │  │                                   │
│   │  └─────────┬─────────┘  │                                   │
│   │            │            │                                   │
│   │            ▼            │                                   │
│   │  ┌───────────────────┐  │                                   │
│   │  │  _execute_step()  │  │                                   │
│   │  └─────────┬─────────┘  │                                   │
│   │            │            │                                   │
│   │            ▼            │                                   │
│   │  ┌───────────────────┐  │                                   │
│   │  │_mark_step_complet │  │                                   │
│   │  └─────────┬─────────┘  │                                   │
│   │            │            │                                   │
│   └────────────┼────────────┘                                   │
│                │                                                 │
│                ▼                                                 │
│   _finalize_plan()                                              │
│       │                                                         │
│       ▼                                                         │
│   完成                                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 最佳实践

1. **职责分离**：每个 Flow 应该有明确的职责
2. **资源管理**：始终在 `finally` 块中清理资源
3. **错误处理**：捕获并处理各步骤中的错误
4. **超时控制**：设置合理的超时时间
5. **状态追踪**：记录每个步骤的执行状态
6. **可复用性**：设计可复用的步骤方法
7. **配置化**：将可变参数暴露为配置选项
8. **Agent 选择**：合理配置 `executor_keys` 实现智能 Agent 选择
9. **计划生成**：使用 LLM 生成清晰、可执行的步骤
10. **进度跟踪**：利用 `PlanningTool` 跟踪计划执行进度