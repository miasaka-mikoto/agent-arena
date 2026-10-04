# Agent Arena（AI Agent 竞技场）

Agent Arena 是一个本地、可重复、完全安全的 Agent 基准测试环境。它把一次运行明确建模为：

`Environment → Task → Observation → Agent → Action → Tool → Result → Evaluation → Replay`

它不是聊天机器人排行榜，也不会连接真实邮箱、文件、日历、账号或收费模型 API。开发阶段的默认 Agent 是 `RandomAgent`、`RuleBasedAgent`、`ScriptedAgent` 和 `MockLLMAgent`。

## 快速运行

需要 Python 3.10+，无第三方运行时依赖：

```bash
python -m agent_arena demo --tasks 50 --seeds 0 1 2 --output artifacts/demo
python -m agent_arena replay --run-id <RUN_ID> --input artifacts/demo/results.json
python -m agent_arena inspect --input artifacts/demo/results.json
```

`demo` 会创建 5 类安全模拟任务，运行 3 个本地 Agent × 多个 seed，保存任务数据、逐步 Trace、SQLite/JSON 结果和可直接打开的 HTML 报告。报告中的 Mock/RuleBased 结果均标记为合成基线。

## 安全边界

- 所有文件、邮件、日历、网页、代码执行器和 Grid World 都是内存中的模拟环境。
- 工具调用只在当前 Task 的白名单内执行。
- 不实现真实网络、shell、邮件发送、账号登录或任意代码执行。
- Trace 只记录公开的行为摘要，不记录模型私有 chain-of-thought。
- `OpenAI`、`Anthropic`、`Google`、本地模型和自定义 HTTP Provider 只保留接口扩展点；默认不会发起请求。

## 主要模块

| 模块 | 内容 |
| --- | --- |
| `core.py` | Agent、Action、Observation、ToolResult、Trace 等稳定数据模型 |
| `environments.py` / `tools.py` | 六种安全模拟环境与统一工具注册器 |
| `tasking.py` | TaskDefinition、模板生成器、对抗任务与 Demo 数据集 |
| `runner.py` / `evaluator.py` | 单任务执行、Trace、失败分类和指标计算 |
| `tournament.py` | Agent × Task × Seed 的可重复竞赛 |
| `replay.py` / `reporting.py` | 逐步回放、排行榜、HTML Dashboard |
| `cli.py` | Demo、单任务、回放、报告、数据集生成命令 |

## 开发

```bash
python -m pytest
python -m agent_arena demo --tasks 50 --seeds 0 1 2 --output artifacts/demo
```

Windows 打包入口见 `scripts/build_windows.ps1`。如果本机安装了 PyInstaller，会生成 `dist/AgentArena.exe`；没有安装时脚本仍会生成可运行的源码发布目录。Linux/macOS 可直接用 `pyinstaller --onefile --name AgentArena agent_arena/__main__.py` 生成对应平台的可执行文件。

本工作区已验证并附带 Linux x86_64 可执行文件 `dist/AgentArena`。PyInstaller 不跨操作系统生成 PE 文件，因此 Windows `.exe` 请在 Windows 上运行上述脚本生成；源码、spec 和构建脚本均已包含。

## 结果解释

排行榜同时显示 Success、Efficiency、Reliability、Recovery 和 Failure Breakdown。Mock/RuleBased 结果会明确标记 `synthetic=true`，绝不冒充真实模型表现。
