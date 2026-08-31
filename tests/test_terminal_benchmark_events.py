import json

from coco_agent.terminal import TerminalRenderer


def test_benchmark_plan_has_framework_owned_visible_state():
    output = []
    renderer = TerminalRenderer(output.append, animate=False)
    renderer.events([
        (
            "ui_event",
            json.dumps({
                "kind": "benchmark_plan",
                "task_id": "tuning-1",
                "state": "adapter_unconfigured",
                "message": "Benchmark 计划已生成，执行适配器尚未配置",
            }, ensure_ascii=False),
        )
    ])
    rendered = "\n".join(output)
    assert "◇ Benchmark" in rendered
    assert "tuning-1" in rendered
    assert "尚未启动" in rendered
    assert "请稍候" not in rendered
