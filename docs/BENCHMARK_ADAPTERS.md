# Benchmark Adapter 使用与结果口径

COCO_Agent 通过 Benchmark Adapter Registry 选择验证工具。`auto` 根据目标指标、聚合方式、workload 特征和已配置状态选择，能力相同时优先 COCO_Benchmark；用户可以显式指定 Adapter。

## COCO_Benchmark

```powershell
coco benchmark config --adapter coco_benchmark
coco benchmark credential --adapter coco_benchmark
coco benchmark show --adapter coco_benchmark
```

凭据通过隐藏输入保存到个人配置目录；`show` 只显示是否已配置，不显示真值。
数据集可以在没有 endpoint、部署配置和凭据时独立生成：

```powershell
coco workload build --model MODEL_ID --input-min 1000 --input-max 10000 --output-max 512
```

适用能力：渐进式请求链、长上下文增长、结构化内容、Tool Schema、自定义到达轨迹、workload hash、request event、artifact hash、客户端自检、`BenchmarkRunReceipt` 和 SLA goodput。

当前指标包括 TTFT、TPOT、E2E、inter-output/ITL 和吞吐相关结果。TPOT 根据每请求的
`(E2E - TTFT) / (output_tokens - 1)` 派生，再按任务要求进行 p50/p95/p99 等聚合；
计算要求请求返回可信的 `output_tokens` 且至少生成两个 token。

## AISBench

官方源码安装方式：

```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
git clone https://github.com/AISBench/benchmark.git
cd benchmark
pip install -e . --use-pep517
pip install -r requirements/api.txt
pip install -r requirements/extra.txt
ais_bench -h
```

配置 Agent Adapter：

```powershell
coco benchmark config --adapter ais_bench
coco benchmark show --adapter ais_bench
```

性能模式计划使用：

```bash
ais_bench --models MODEL_PROFILE --datasets DATASET_PROFILE --mode perf
```

适用能力：服务化模型标准性能评测、极限压测、稳态与真实流量分布、TTFT、TPOT、ITL、E2E、请求/Token 吞吐，以及 OpenCompass 数据集和准确率评测。

官方性能汇总提供 avg、min、max、median、p75、p90 和 p99 等口径。Agent 根据 Adapter capability profile 检查用户要求的聚合方式。

## 结果关联

每次结果记录 Adapter/版本、workload 或 dataset 配置与 hash、指标定义/单位/aggregation、运行参数与时间窗口、原始 artifact、工具原生报告和 capability gaps。跨工具对比先执行口径兼容性检查，再进入性能结论和经验库。

参考：

- AISBench 官方仓库：https://github.com/AISBench/benchmark
- AISBench 运行模式：https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md
- AISBench 性能指标：https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/performance_metric.md
