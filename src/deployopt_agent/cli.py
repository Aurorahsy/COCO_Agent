"""Interactive LLM + function-call CLI."""

from __future__ import annotations

import argparse
import getpass
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .capabilities import MockTuningCapability, SqliteOperationStore
from .configuration import (
    LLMSettings,
    config_path,
    is_configured,
    load_settings,
    public_settings,
    save_settings,
)
from .conversation import LLMInterpreter, OpenAICompatibleLLM
from .conversation.skill import load_tuning_skill
from .conversation.tuning_tools import TuningToolset
from .demo import (
    AllowLowRiskActions,
    DoubleConcurrencyPolicy,
    InMemoryExperience,
    NumericGoalVerifier,
    SimpleReportBuilder,
)
from .graph_app import TuningApplication


def _application(checkpointer, root: Path) -> TuningApplication:
    return TuningApplication(
        checkpointer=checkpointer,
        capability=MockTuningCapability(SqliteOperationStore(root / "operations.sqlite")),
        authorization=AllowLowRiskActions(),
        verifier=NumericGoalVerifier(),
        experience=InMemoryExperience(),
        tuning_policy=DoubleConcurrencyPolicy(),
        report_builder=SimpleReportBuilder(),
    )


def run_chat(llm=None, input_fn=input, output_fn=print, secret_fn=getpass.getpass) -> int:
    if llm is None:
        if not is_configured():
            output_fn("首次启动：尚未发现模型配置，现在进行一次性初始化。")
            configure(input_fn=input_fn, secret_fn=secret_fn, output_fn=output_fn)
        settings = load_settings()
        llm = OpenAICompatibleLLM.from_settings(settings)
        output_fn(f"已加载持久化模型配置：{config_path()}")
    output_fn("coco_agent（LLM Function-Call Slice 0）")
    output_fn("输入 exit 结束会话。")
    with tempfile.TemporaryDirectory(prefix="deployopt_chat_") as temp_dir:
        root = Path(temp_dir)
        with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as saver:
            toolset = TuningToolset(_application(saver, root))
            interpreter = LLMInterpreter(
                llm=llm,
                tools=toolset.registry(),
                skill=load_tuning_skill(),
            )
            while True:
                try:
                    user_message = input_fn("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    output_fn("")
                    return 0
                if user_message.lower() in {"exit", "quit", "退出"}:
                    return 0
                if not user_message:
                    continue
                output_fn(f"Agent> {interpreter.handle(user_message)}")


def configure(input_fn=input, secret_fn=getpass.getpass, output_fn=print) -> int:
    path = config_path()
    output_fn(f"配置将保存到：{path}")
    output_fn("API Key 使用 Windows DPAPI 加密；输入内容不会显示或写入日志。")
    base_url = input_fn("OpenAI-compatible Base URL [https://api.openai.com/v1]: ").strip()
    model = input_fn("模型名称 [gpt-4.1-mini]: ").strip()
    api_key = secret_fn("模型服务 API Key（隐藏输入）: ").strip()
    save_settings(LLMSettings(
        base_url=base_url or "https://api.openai.com/v1",
        model=model or "gpt-4.1-mini",
        api_key=api_key,
    ))
    output_fn(f"配置已保存：{path}")
    output_fn("未记录 API Key 真值；show 命令只显示加密状态。")
    return 0


def show_configuration(output_fn=print) -> int:
    output_fn(json.dumps(public_settings(), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coco-agent")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("chat", help="start LLM tool-calling conversation")
    config_parser = subcommands.add_parser("config", help="configure persistent LLM access")
    config_parser.add_argument(
        "action", nargs="?", choices=("set", "show", "path"), default="set"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return run_chat()
        if args.action == "show":
            return show_configuration()
        if args.action == "path":
            print(config_path())
            return 0
        return configure()
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
