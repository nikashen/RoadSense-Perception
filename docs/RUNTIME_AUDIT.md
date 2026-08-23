# Runtime audit record

`roadsense benchmark`（别名 `roadsense runtime-audit`）运行一次或多次
确定性城市环路 fixture，并写出 `roadsense.runtime-audit/v1` JSON 记录：

```powershell
.\.venv\Scripts\python.exe -m roadsense benchmark `
  --iterations 3 `
  --output reports/runtime_fixture_v1.json
```

若需要同时在终端查看完整 JSON，可加 `--json`。记录是原子写入的，默认不会
覆盖任何输入数据。

## 记录内容

每个记录包含：

- `device`：操作系统、架构、Python 实现和 CPU 数量；不写入主机名、序列号
  等主机标识；
- `dependencies`：RoadSense、NumPy、Pydantic 以及已安装的服务依赖版本；
- `input` / `output`：fixture 标识、帧数、迭代次数和 canonical SHA-256；输入
  hash 绑定帧 JSON 以及真值/预测 mask 的 dtype、shape、二进制 SHA-256，输出
  hash 绑定完整 replay payload 与指标；顶层 `input_sha256`、`output_sha256`
  必须与嵌套字段一致；
- `stages`：fixture 生成、评估和 replay payload 序列化的独立耗时与处理量；
  没有对应运行阶段的 `inference`、`rendering` 会显式标记
  `measured=false` 并给出原因，不会伪造 0ms；
- `wall_time_ms`、`throughput_fps`：本次 Python fixture pipeline 的诊断值；
- `record_id`：对除自身外完整 canonical JSON 的 16 位 SHA-256 前缀。

`started_at_utc` 采用带时区的 UTC 时间。修改任意字段都会使 `record_id`
校验失败，因此记录可作为审计附件保存。

## 证据边界

当前 schema 将 `benchmark_claim_available` 固定为 `false`，fixture 记录的
`evaluation_authorized` 与 `frozen` 也必须为 `false`。该命令不加载模型、不读取
BDD100K/COCO/MOT 数据、不执行浏览器渲染；因此输出的吞吐和阶段耗时只能用于：

- 检测运行时回归（例如依赖升级后 fixture pipeline 是否变慢）；
- 验证输入/输出 hash 和报告管线；
- 展示未来真实评测所需的记录格式。

它们不是模型 latency、端到端 FPS、实时性 SLA、鲁棒性或公开数据集性能。真实
benchmark 必须另行绑定数据集 manifest、模型 artifact hash、预处理配置、设备、
锁定依赖和独立 evaluator，并在新的授权证据 schema 中发布。

## 程序化使用

```python
from roadsense.runtime import build_fixture_runtime_record

record = build_fixture_runtime_record(iterations=2)
print(record.record_id)
print(record.model_dump(mode="json")["claim_boundary"])
```

验证外部 JSON 时使用：

```python
from roadsense.json_io import load_strict_json
from roadsense.runtime import RuntimeAuditRecord

record = RuntimeAuditRecord.model_validate(load_strict_json(path))
```
