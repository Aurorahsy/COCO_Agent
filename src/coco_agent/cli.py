"""Interactive LLM + function-call CLI."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from .benchmark import (
    BenchmarkSettings,
    BenchmarkAdapterRegistry,
    LocalCocoBenchmarkAdapter,
    LocalAisBenchAdapter,
    benchmark_config_path,
    load_benchmark_settings,
    save_benchmark_settings,
    credential_path,
    load_credential,
    save_credential,
)
from .configuration import (
    LLMSettings,
    config_path,
    is_configured,
    load_settings,
    public_settings,
    save_settings,
)
from .conversation import LLMInterpreter, OpenAICompatibleLLM
from .conversation.openai_compat import LLMServiceError
from .conversation.skill import load_tuning_skill
from .conversation.tuning_tools import TuningToolset
from .terminal import TerminalRenderer


def run_chat(llm=None, input_fn=input, output_fn=print, secret_fn=getpass.getpass) -> int:
    renderer = TerminalRenderer(output_fn, animate=input_fn is input and output_fn is print)
    if llm is None:
        if not is_configured():
            output_fn("首次启动：尚未发现模型配置，现在进行一次性初始化。")
            configure(input_fn=input_fn, secret_fn=secret_fn, output_fn=output_fn)
        settings = load_settings()
        llm = OpenAICompatibleLLM.from_settings(settings)
        output_fn(f"已加载持久化模型配置：{config_path()}")
    output_fn("COCO_Agent  ·  输入 /exit 结束会话")
    benchmark_settings = load_benchmark_settings("coco_benchmark")
    ais_bench_settings = load_benchmark_settings("ais_bench")
    benchmark_registry = BenchmarkAdapterRegistry()
    if benchmark_settings:
        benchmark_registry.register(LocalCocoBenchmarkAdapter(
            benchmark_settings, credential=load_credential("coco_benchmark")
        ))
    if ais_bench_settings:
        benchmark_registry.register(LocalAisBenchAdapter(
            ais_bench_settings, credential=load_credential("ais_bench")
        ))
    toolset = TuningToolset(benchmark_registry)
    runtime_capabilities = benchmark_registry.inspect()
    interpreter = LLMInterpreter(
        llm=llm,
        tools=toolset.registry(),
        skill=(
            load_tuning_skill()
            + "\n\n# Runtime Benchmark capability snapshot\n"
            + json.dumps(runtime_capabilities, ensure_ascii=False)
        ),
    )
    while True:
        try:
            user_message = input_fn(renderer.prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return 0
        if user_message.lower() in {"/exit", "exit", "quit", "退出"}:
            return 0
        if not user_message:
            continue
        retrying = False
        while True:
            try:
                with renderer.thinking():
                    response = interpreter.resume() if retrying else interpreter.handle(user_message)
                renderer.events(interpreter.last_events)
                renderer.response(response)
                break
            except LLMServiceError as exc:
                renderer.error(str(exc))
                try:
                    action = input_fn(
                        "选择操作：[r] 重试  [c] 重新配置  [q] 退出："
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    output_fn("")
                    return 0
                if action in {"q", "quit", "exit", "退出"}:
                    return 0
                if action in {"c", "config", "配置"}:
                    try:
                        configure(input_fn=input_fn, secret_fn=secret_fn, output_fn=output_fn)
                        interpreter.replace_llm(OpenAICompatibleLLM.from_settings(load_settings()))
                    except LLMServiceError as config_error:
                        renderer.error(f"新配置验证失败：{config_error}")
                        continue
                elif action not in {"r", "retry", "重试"}:
                    renderer.error("无法识别该操作，请选择 r、c 或 q。")
                    continue
                retrying = True


def configure(
    input_fn=input,
    secret_fn=getpass.getpass,
    output_fn=print,
    validator=None,
) -> int:
    path = config_path()
    output_fn(f"配置将保存到：{path}")
    output_fn("API Key 输入内容不会显示，也不会写入日志或配置状态输出。")
    base_url = input_fn("OpenAI-compatible Base URL [https://api.openai.com/v1]: ").strip()
    model = input_fn("模型名称 [gpt-4.1-mini]: ").strip()
    api_key = secret_fn("模型服务 API Key（隐藏输入）: ").strip()
    candidate = LLMSettings(
        base_url=base_url or "https://api.openai.com/v1",
        model=model or "gpt-4.1-mini",
        api_key=api_key,
    )
    output_fn("正在验证模型服务配置……")
    if validator is None:
        validator = lambda settings: OpenAICompatibleLLM.from_settings(settings).probe()
    validator(candidate)
    save_settings(candidate)
    output_fn(f"配置已保存：{path}")
    output_fn("show 命令不会显示 API Key 真值。")
    return 0


def show_configuration(output_fn=print) -> int:
    output_fn(json.dumps(public_settings(), ensure_ascii=False, indent=2))
    return 0


def configure_benchmark(
    adapter="coco_benchmark", input_fn=input, output_fn=print, validator=None
) -> int:
    command_name = "coco-benchmark" if adapter == "coco_benchmark" else "ais_bench"
    display_name = "COCO_Benchmark" if adapter == "coco_benchmark" else "AISBench"
    discovered = shutil.which(command_name) or ""
    executable = input_fn(f"{display_name} 可执行文件 [{discovered}]: ").strip() or discovered
    if not executable:
        raise RuntimeError(f"未找到 {command_name}，请输入其完整可执行文件路径")
    workdir = input_fn(f"{display_name} 项目目录: ").strip()
    if not workdir:
        raise RuntimeError("COCO_Benchmark 项目目录不能为空")
    default_output = str(Path(workdir) / "results")
    output_dir = input_fn(f"结果目录 [{default_output}]: ").strip() or default_output
    api_key_env = input_fn("目标服务 API Key 环境变量名 [OPENAI_API_KEY]: ").strip()
    settings = BenchmarkSettings(
        executable=executable,
        workdir=workdir,
        output_dir=output_dir,
        api_key_env=api_key_env or "OPENAI_API_KEY",
        adapter=adapter,
    )
    if validator is None:
        validator = _validate_benchmark_settings
    validator(settings)
    path = save_benchmark_settings(settings)
    output_fn(f"{display_name} 适配器配置已保存：{path}")
    output_fn("配置仅保存环境变量名，不保存目标服务 API Key 真值。")
    return 0


def _validate_benchmark_settings(settings: BenchmarkSettings) -> None:
    executable = Path(settings.executable)
    if not executable.is_file():
        raise RuntimeError(f"COCO_Benchmark 可执行文件不存在：{executable}")
    if not Path(settings.workdir).is_dir():
        raise RuntimeError(f"COCO_Benchmark 项目目录不存在：{settings.workdir}")
    check_argument = "--version" if settings.adapter == "coco_benchmark" else "-h"
    result = subprocess.run(
        [str(executable), check_argument],
        cwd=settings.workdir,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"COCO_Benchmark 验证失败：{detail}")


def show_benchmark_configuration(adapter="coco_benchmark", output_fn=print) -> int:
    settings = load_benchmark_settings(adapter)
    output_fn(json.dumps({
        "configured": settings is not None,
        "adapter": adapter,
        "path": str(benchmark_config_path(adapter)),
        "credential_configured": load_credential(adapter) is not None,
        "credential_path": str(credential_path()),
        **({
            "executable": settings.executable,
            "workdir": settings.workdir,
            "output_dir": settings.output_dir,
            "api_key_env": settings.api_key_env,
        } if settings else {}),
    }, ensure_ascii=False, indent=2))
    return 0


def configure_benchmark_credential(
    adapter="coco_benchmark", secret_fn=getpass.getpass, output_fn=print
) -> int:
    output_fn(f"凭据将保存到代码仓外：{credential_path()}")
    api_key = secret_fn("目标服务 API Key（隐藏输入）: ").strip()
    if not api_key:
        raise RuntimeError("API Key 不能为空")
    path = save_credential(adapter, api_key)
    output_fn(f"凭据已保存：{path}")
    output_fn("show 命令只显示是否已配置，不显示真值。")
    return 0


def build_workload_from_cli(args, output_fn=print) -> int:
    settings = load_benchmark_settings("coco_benchmark")
    if settings is None:
        raise RuntimeError(
            "COCO_Benchmark 尚未配置，请先运行 coco benchmark config --adapter coco_benchmark"
        )
    task_id = args.task_id or f"workload-{uuid4().hex[:12]}"
    result = LocalCocoBenchmarkAdapter(settings).generate_workload(task_id, {
        "target": {
            "model_ref": args.model,
            **({"tokenizer_ref": args.tokenizer} if args.tokenizer else {}),
            **({"tokenizer_revision": args.tokenizer_revision} if args.tokenizer_revision else {}),
        },
        "workload": {
            "input_tokens_min": args.input_min,
            "input_tokens_max": args.input_max,
            "output_tokens_max": args.output_max,
            "chains": args.chains,
            "turns_per_chain": args.turns,
            "growth_factor": args.growth_factor,
            "length_jitter": args.jitter,
            "arrival_pattern": "poisson",
            "chain_starts_per_second": args.chain_starts_per_second,
            "seed": args.seed,
        },
    })
    output_fn(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "workload_ready" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coco")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("chat", help="start LLM tool-calling conversation")
    config_parser = subcommands.add_parser("config", help="configure persistent LLM access")
    config_parser.add_argument(
        "action", nargs="?", choices=("set", "show", "path"), default="set"
    )
    benchmark_parser = subcommands.add_parser(
        "benchmark", help="configure the dedicated COCO_Benchmark adapter"
    )
    benchmark_parser.add_argument(
        "action", nargs="?", choices=("config", "credential", "show", "path"), default="config"
    )
    benchmark_parser.add_argument(
        "--adapter", choices=("coco_benchmark", "ais_bench"), default="coco_benchmark"
    )
    workload_parser = subcommands.add_parser(
        "workload", help="build an offline COCO_Benchmark workload without an endpoint"
    )
    workload_parser.add_argument("action", choices=("build",))
    workload_parser.add_argument("--model", required=True)
    workload_parser.add_argument("--tokenizer")
    workload_parser.add_argument("--tokenizer-revision")
    workload_parser.add_argument("--input-min", type=int, required=True)
    workload_parser.add_argument("--input-max", type=int, required=True)
    workload_parser.add_argument("--output-max", type=int, required=True)
    workload_parser.add_argument("--chains", type=int, default=4)
    workload_parser.add_argument("--turns", type=int, default=4)
    workload_parser.add_argument("--growth-factor", type=float, default=2.0)
    workload_parser.add_argument("--jitter", type=float, default=0.05)
    workload_parser.add_argument("--chain-starts-per-second", type=float, default=1.0)
    workload_parser.add_argument("--seed", type=int, default=42)
    workload_parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return run_chat()
        if args.command == "benchmark":
            if args.action == "show":
                return show_benchmark_configuration(args.adapter)
            if args.action == "path":
                print(benchmark_config_path(args.adapter))
                return 0
            if args.action == "credential":
                return configure_benchmark_credential(args.adapter)
            return configure_benchmark(args.adapter)
        if args.command == "workload":
            return build_workload_from_cli(args)
        if args.action == "show":
            return show_configuration()
        if args.action == "path":
            print(config_path())
            return 0
        return configure()
    except RuntimeError as exc:
        parser.exit(1, f"coco: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
