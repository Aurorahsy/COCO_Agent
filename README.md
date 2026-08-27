# coco_agent

`coco_agent` 是一个基于 LangGraph 的部署调优 Agent。当前版本实现了最小核心链路：

```text
用户自然语言
→ LLM 解释并发起 Function Call
→ Agent 校验 Schema 并执行工具
→ 收集环境与运行 Benchmark
→ 分析指标并推荐一个参数
→ 再次测试
→ LLM 根据工具结果输出对比结论
```

当前环境采集和 Benchmark 使用安全、确定性的模拟实现，不会修改真实机器配置。真实部署、Benchmark 和经验库会沿现有端口逐步接入。

## 系统要求

- Windows 10/11
- Python 3.11 或更高版本
- 推荐使用 Miniconda/Conda 管理环境
- 一个支持 OpenAI-compatible Chat Completions 和 Function Calling 的模型服务

已验证 OpenAI-compatible 配置方式，可用于 OpenAI、DeepSeek 或兼容服务。模型本身是否支持 Function Calling，需要以服务商说明为准。

## 安装

### 1. 克隆 GitHub 仓库

```powershell
git clone https://github.com/Aurorahsy/COCO_Agent.git coco_agent
cd coco_agent
```

### 2. 创建并激活 Conda 环境

```powershell
conda create -n coco_agent python=3.11 -y
conda activate coco_agent
```

确认当前 Python 来自新环境：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

### 3. 安装项目和运行依赖

```powershell
python -m pip install .
```

pip 会读取项目根目录中的 `pyproject.toml`，构建 `coco-agent-runtime`，并自动下载、安装全部运行依赖。无需手动逐个安装 LangGraph 或 SQLite checkpoint 包。

如果需要修改源码并运行测试，改用开发模式：

```powershell
python -m pip install -e ".[test]"
```

当前声明的直接依赖包括：

- `langgraph>=1.2,<2`
- `langgraph-checkpoint-sqlite>=3.1,<4`
- 测试模式额外安装 `pytest>=8,<9`

如果受限网络环境已经预装构建依赖，但 pip 无法下载隔离构建环境，可以使用：

```powershell
python -m pip install -e ".[test]" --no-build-isolation
```

### 4. 验证安装

```powershell
coco-agent --help
```

看到 `chat` 和 `config` 子命令即表示安装成功。

## 首次启动与模型配置

启动 Agent：

```powershell
coco-agent chat
```

首次启动会自动进入一次性配置向导：

```text
OpenAI-compatible Base URL:
模型名称:
模型服务 API Key（隐藏输入）:
```

OpenAI 示例：

```text
Base URL: https://api.openai.com/v1
Model:    gpt-4.1-mini
```

DeepSeek 示例：

```text
Base URL: https://api.deepseek.com
Model:    deepseek-chat
```

API Key 输入时不会显示。配置成功后，后续启动会自动复用，不需要重复输入。

如果找不到 `coco-agent` 命令，可以使用等价入口：

```powershell
python -m deployopt_agent.cli chat
```

## 使用

```powershell
coco-agent chat
```

示例输入：

```text
You> 帮我把推理吞吐量优化到 100 requests/s
```

模型会根据本地 Skill 和工具 Schema 决定是追问缺失信息，还是调用：

- `submit_tuning_task`
- `run_tuning_task`

Agent 不直接信任模型输出。每个 Function Call 都会经过工具白名单和参数 Schema 校验，工具结果会回填给模型，再由模型生成最终回答。

输入 `exit`、`quit` 或 `退出` 结束会话。

## 配置管理

主动更换模型、服务地址或 API Key：

```powershell
coco-agent config
```

查看非敏感配置状态：

```powershell
coco-agent config show
```

查看唯一配置文件位置：

```powershell
coco-agent config path
```

配置文件固定保存在代码仓库之外：

```text
%APPDATA%\coco_agent\config.json
```

在当前 Windows 用户下一般对应：

```text
C:\Users\<用户名>\AppData\Roaming\coco_agent\config.json
```

