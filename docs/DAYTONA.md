# Daytona Sandbox

本项目使用 [Daytona](https://www.daytona.io/) 作为沙箱环境，为 Agent 提供隔离的可执行环境。Daytona 是一个云端开发环境平台，支持在安全的容器中运行代码、浏览器操作和文件管理。

## 概述

Daytona Sandbox 为 OpenManus 提供以下能力：

- **隔离的执行环境**：每个任务在独立的容器中运行，互不干扰
- **完整的 Linux 环境**：支持运行命令行工具、安装软件包等
- **内置浏览器**：预配置的 Chrome 浏览器，支持自动化操作
- **VNC 访问**：可通过 VNC 实时查看浏览器操作
- **HTTP 服务**：沙箱内置 HTTP 服务器，可通过 8080 端口访问

## 安装依赖

```bash
# 激活 Python 环境
conda activate 'Your OpenManus python env'

# 安装 Daytona SDK
pip install daytona==0.21.8 structlog==25.4.0
```

## 配置

### 1. 复制配置模板

```bash
cd OpenManus
cp config/config.example-daytona.toml config/config.toml
```

### 2. 获取 API Key

访问 [Daytona Dashboard](https://app.daytona.io/dashboard/keys) 创建你的 API Key。

### 3. 配置说明

编辑 `config/config.toml` 中的 `[daytona]` 部分：

```toml
[daytona]
# 必填：你的 Daytona API Key
daytona_api_key = "your-api-key"

# 可选：服务器地址，默认使用官方服务器
# daytona_server_url = "https://app.daytona.io/api"

# 可选：目标区域，当前支持 us (美国) 和 eu (欧洲)
# daytona_target = "us"

# 可选：沙箱镜像名称
# sandbox_image_name = "whitezxj/sandbox:0.1.0"

# 可选：沙箱入口点
# sandbox_entrypoint = "/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf"

# 可选：VNC 密码，默认 123456
# VNC_password = "your-password"
```

### 配置项说明

| 配置项 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `daytona_api_key` | 是 | - | API Key，从 Daytona 仪表板获取 |
| `daytona_server_url` | 否 | `https://app.daytona.io/api` | Daytona API 服务器地址 |
| `daytona_target` | 否 | `us` | 区域：`us` 或 `eu` |
| `sandbox_image_name` | 否 | `whitezxj/sandbox:0.1.0` | 沙箱使用的 Docker 镜像 |
| `sandbox_entrypoint` | 否 | supervisord 启动命令 | 沙箱启动入口 |
| `VNC_password` | 否 | `123456` | VNC 连接密码 |

## 使用方法

### 运行 Agent

```bash
cd OpenManus
python sandbox_main.py --prompt "你的任务描述"
```

或者交互式运行：

```bash
python sandbox_main.py
Enter your prompt: 你的任务描述
```

### 查看执行状态

Agent 启动后会输出两个重要的访问链接：

1. **VNC URL** (端口 6080)：实时查看浏览器操作
2. **Website URL** (端口 8080)：访问沙箱内启动的 HTTP 服务

例如：
```
VNC URL: https://6080-sandbox-xxxxxx.h7890.daytona.work
Website URL: https://8080-sandbox-xxxxxx.h7890.daytona.work
```

## 可用工具

Sandbox Agent 提供了以下四类工具：

### 1. 浏览器工具 (sandbox_browser)

在沙箱内的浏览器环境中执行自动化操作。

| 动作 | 描述 | 参数 |
|------|------|------|
| `navigate_to` | 导航到指定 URL | `url` |
| `click_element` | 点击元素 | `index` |
| `input_text` | 输入文本 | `index`, `text` |
| `send_keys` | 发送键盘命令 | `keys` |
| `scroll_down` | 向下滚动 | `amount` |
| `scroll_up` | 向上滚动 | `amount` |
| `scroll_to_text` | 滚动到指定文本 | `text` |
| `switch_tab` | 切换标签页 | `page_id` |
| `close_tab` | 关闭标签页 | `page_id` |
| `go_back` | 返回上一页 | - |

### 2. 文件工具 (sandbox_files)

在沙箱工作区 `/workspace` 中管理文件。

| 动作 | 描述 | 参数 |
|------|------|------|
| `create_file` | 创建新文件 | `file_path`, `file_contents` |
| `str_replace` | 替换文件中的字符串 | `file_path`, `old_str`, `new_str` |
| `full_file_rewrite` | 完全重写文件 | `file_path`, `file_contents` |
| `delete_file` | 删除文件 | `file_path` |

### 3. Shell 工具 (sandbox_shell)

在沙箱中执行 Shell 命令。

| 动作 | 描述 | 参数 |
|------|------|------|
| `execute_command` | 执行命令 | `command`, `folder`, `blocking`, `timeout` |
| `check_command_output` | 检查命令输出 | `session_name`, `kill_session` |
| `terminate_command` | 终止命令 | `session_name` |
| `list_commands` | 列出所有会话 | - |

### 4. 视觉工具 (sandbox_vision)

读取沙箱内的图片文件。

| 动作 | 描述 | 参数 |
|------|------|------|
| `see_image` | 读取并压缩图片 | `file_path` |

## 示例

### 示例 1：网页信息采集

```
输入任务：帮我在 https://hk.trip.com/travel-guide/guidebook/nanjing-9696/?ishideheader=true 查询南京旅游信息，并保存为 index.html
```

执行过程：
1. Agent 使用 `sandbox_browser` 打开指定网页
2. 通过 VNC 链接可实时查看浏览器操作
3. 采集信息后使用 `sandbox_files` 创建 `index.html`
4. 通过 Website URL (8080端口) 可查看生成的网页

### 示例 2：运行开发任务

```
输入任务：创建一个 Python 脚本，实现快速排序算法并运行测试
```

执行过程：
1. Agent 使用 `sandbox_shell` 创建 Python 文件
2. 执行脚本并查看输出
3. 可通过 `check_command_output` 查看执行结果

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenManus Agent                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SandboxManus Agent                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │   │
│  │  │ Browser  │ │  Files   │ │  Shell   │ │Vision│ │   │
│  │  │  Tool    │ │  Tool    │ │  Tool    │ │ Tool │ │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┘ │   │
│  └───────┼────────────┼────────────┼──────────────┘   │
│          │            │            │                    │
└──────────┼────────────┼────────────┼────────────────────┘
           │            │            │
           ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Daytona SDK                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Daytona Sandbox                        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │         沙箱容器 (Docker Container)          │    │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐  │    │   │
│  │  │  │supervisord    │ │Chrome  │ │HTTP Server│  │    │   │
│  │  │  │(进程管理)     │ │浏览器  │ │  (8080)   │  │    │   │
│  │  │  └─────────┘ └─────────┘ └─────────────┘  │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 资源限制

默认配置的资源限制：

- **CPU**: 2 核
- **内存**: 4 GB
- **磁盘**: 5 GB
- **自动停止**: 15 分钟无活动后停止
- **自动归档**: 24 小时后归档

## 故障排除

### 1. API Key 无效

确保在 [Daytona Dashboard](https://app.daytona.io/dashboard/keys) 创建了有效的 API Key，并正确填入配置文件。

### 2. 沙箱创建失败

- 检查网络连接
- 确认账户有足够的沙箱配额
- 查看日志获取详细错误信息

### 3. 浏览器无法启动

沙箱镜像预配置了 Chrome 浏览器，如果遇到问题：
- 检查 `sandbox_image_name` 配置是否正确
- 确认镜像支持 `supervisord` 启动

### 4. VNC 连接失败

- 检查 VNC 密码是否正确
- 确认沙箱正在运行中
- 尝试刷新页面

## 相关文件

- `config/config.example-daytona.toml` - 配置文件模板
- `app/daytona/sandbox.py` - 沙箱管理模块
- `app/daytona/tool_base.py` - 工具基类
- `app/agent/sandbox_agent.py` - Sandbox Agent 实现
- `app/tool/sandbox/` - 各工具实现
- `sandbox_main.py` - 入口文件

## 了解更多

- [Daytona 官方文档](https://www.daytona.io/docs/)
- [Daytona Dashboard](https://app.daytona.io/dashboard)
- [OpenManus 项目](https://github.com/OpenManus)