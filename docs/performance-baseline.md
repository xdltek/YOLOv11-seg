# Performance records and baseline / 性能记录与基准

These are the original 1.6.11 RGB measurements, retained unchanged as the
optimization baseline. They are not a numerical qualification of newly supplied
ONNX files or the accuracy corrections made on 2026-11-20. Historical PASS means
the original runtime/smoke validation completed; current accuracy support is
listed in the root README. Only release-supported profiles are shown.

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 3.882 ms | 4.935 ms | 9.306 ms | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 3.916 ms | 9.216 ms | 10.071 ms | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 4.079 ms | 20.554 ms | 11.554 ms | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 3.997 ms | 25.465 ms | 11.415 ms | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 3.985 ms | 56.183 ms | 7.952 ms | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 5.469 ms | 12.372 ms | 24.546 ms | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 5.411 ms | 20.985 ms | 19.068 ms | 45.463 ms | 21.996 | PASS |

One warmup followed by ten measured frames; preprocessing includes H2D and
postprocessing includes D2H. Engine construction and visualization are excluded.
The measurements above remain the original reference values.

以上为原 1.6.11 的 RGB 实测值，原样保留为性能优化基准。历史 PASS 表示原运行检查完成，
不等于新提供 ONNX 文件或本轮精度修正已经获得数值精度认证。当前精度支持范围见根 README。
口径为一次预热后测量十帧，PRE 包含 H2D、POST 包含 D2H，不包含 engine 构建和可视化。
上表继续保留原始参考数值。

## Shared pipeline measurements / 公共路径实测

### Measured RGB Performance (2026-11-21)

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.697 ms | 11.110 ms | 28.775 ms | 47.583 ms | 21.016 | PASS |
| YOLO11s-seg | 640×640 | 7.713 ms | 26.578 ms | 29.622 ms | 63.913 ms | 15.646 | PASS |
| YOLO11m-seg | 640×640 | 7.581 ms | 47.658 ms | 29.243 ms | 84.482 ms | 11.837 | PASS |
| YOLO11l-seg | 640×640 | 7.746 ms | 52.741 ms | 28.502 ms | 88.988 ms | 11.237 | PASS |
| YOLO11x-seg | 640×640 | 7.662 ms | 124.878 ms | 28.331 ms | 160.872 ms | 6.216 | PASS |
| YOLO11n-seg | 960×960 | 10.672 ms | 28.248 ms | 42.491 ms | 81.411 ms | 12.283 | PASS |
| YOLO11s-seg | 960×960 | 10.791 ms | 44.532 ms | 43.120 ms | 98.442 ms | 10.158 | PASS |

### Measured I420 Performance (2026-11-21)

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 6.982 ms | 11.298 ms | 16.084 ms | 34.363 ms | 29.101 | PASS |
| YOLO11s-seg | 640×640 | 7.122 ms | 26.013 ms | 16.672 ms | 49.807 ms | 20.077 | PASS |
| YOLO11m-seg | 640×640 | 6.986 ms | 47.831 ms | 16.548 ms | 71.366 ms | 14.012 | PASS |
| YOLO11l-seg | 640×640 | 7.041 ms | 52.464 ms | 16.342 ms | 75.847 ms | 13.184 | PASS |
| YOLO11x-seg | 640×640 | 7.048 ms | 125.098 ms | 16.338 ms | 148.485 ms | 6.735 | PASS |
| YOLO11n-seg | 960×960 | 10.238 ms | 28.069 ms | 29.133 ms | 67.439 ms | 14.828 | PASS |
| YOLO11s-seg | 960×960 | 10.283 ms | 44.100 ms | 29.264 ms | 83.648 ms | 11.955 | PASS |

All seven profiles use the shared paths and original ONNX models, measured in paired runs
against preserved `f47941a` binaries. Each input uses one warmup and ten measured frames,
capture disabled, at a 920 MHz RPP clock; all device tasks run serially.
PRE includes input H2D; POST includes output D2H. Engine construction, image decoding and
visualization are excluded. PASS denotes completed runtime/performance measurement.
All fourteen measured end-to-end latencies were below the paired `f47941a` baseline.

These results remain slower than the original 1.6.11 target; the full-application one-H2D/one-D2H
objective is still incomplete. Original target, paired comparisons and commands:
[performance record](../README.md).

### RGB 实测性能（2026-11-21）

| 模型 | 输入 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.697 ms | 11.110 ms | 28.775 ms | 47.583 ms | 21.016 | PASS |
| YOLO11s-seg | 640×640 | 7.713 ms | 26.578 ms | 29.622 ms | 63.913 ms | 15.646 | PASS |
| YOLO11m-seg | 640×640 | 7.581 ms | 47.658 ms | 29.243 ms | 84.482 ms | 11.837 | PASS |
| YOLO11l-seg | 640×640 | 7.746 ms | 52.741 ms | 28.502 ms | 88.988 ms | 11.237 | PASS |
| YOLO11x-seg | 640×640 | 7.662 ms | 124.878 ms | 28.331 ms | 160.872 ms | 6.216 | PASS |
| YOLO11n-seg | 960×960 | 10.672 ms | 28.248 ms | 42.491 ms | 81.411 ms | 12.283 | PASS |
| YOLO11s-seg | 960×960 | 10.791 ms | 44.532 ms | 43.120 ms | 98.442 ms | 10.158 | PASS |

### I420 实测性能（2026-11-21）

| 模型 | 输入 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 6.982 ms | 11.298 ms | 16.084 ms | 34.363 ms | 29.101 | PASS |
| YOLO11s-seg | 640×640 | 7.122 ms | 26.013 ms | 16.672 ms | 49.807 ms | 20.077 | PASS |
| YOLO11m-seg | 640×640 | 6.986 ms | 47.831 ms | 16.548 ms | 71.366 ms | 14.012 | PASS |
| YOLO11l-seg | 640×640 | 7.041 ms | 52.464 ms | 16.342 ms | 75.847 ms | 13.184 | PASS |
| YOLO11x-seg | 640×640 | 7.048 ms | 125.098 ms | 16.338 ms | 148.485 ms | 6.735 | PASS |
| YOLO11n-seg | 960×960 | 10.238 ms | 28.069 ms | 29.133 ms | 67.439 ms | 14.828 | PASS |
| YOLO11s-seg | 960×960 | 10.283 ms | 44.100 ms | 29.264 ms | 83.648 ms | 11.955 | PASS |

七个规格均使用本次公共路径及原始 ONNX，与 `f47941a` 对照二进制逐项配对实测。
每种输入预热 1 帧、测量 10 帧，关闭精度采集，RPP 主频 920 MHz；所有设备任务串行。
前处理包含输入 H2D，后处理包含输出 D2H，不包含 engine 构建、图片读取和可视化。
表中 PASS 表示运行与测量通过。相对 `f47941a`，十四项端到端实测均未出现回退。

当前结果仍慢于原 1.6.11 性能目标；完整应用的一次 H2D/一次 D2H 目标尚未完成。
原始目标、配对对照及复现命令见[性能记录](../README.md)。

## Paired comparison with f47941a / 与 f47941a 配对对照

| Model | Input | RGB baseline | RGB shared | Reduction | I420 baseline | I420 shared | Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n-seg | 640×640 | 56.433 ms | 47.583 ms | 15.68% | 36.848 ms | 34.363 ms | 6.74% |
| YOLO11s-seg | 640×640 | 73.025 ms | 63.913 ms | 12.48% | 52.577 ms | 49.807 ms | 5.27% |
| YOLO11m-seg | 640×640 | 94.807 ms | 84.482 ms | 10.89% | 74.179 ms | 71.366 ms | 3.79% |
| YOLO11l-seg | 640×640 | 99.292 ms | 88.988 ms | 10.38% | 79.249 ms | 75.847 ms | 4.29% |
| YOLO11x-seg | 640×640 | 167.282 ms | 160.872 ms | 3.83% | 149.565 ms | 148.485 ms | 0.72% |
| YOLO11n-seg | 960×960 | 95.113 ms | 81.411 ms | 14.41% | 75.471 ms | 67.439 ms | 10.64% |
| YOLO11s-seg | 960×960 | 112.141 ms | 98.442 ms | 12.22% | 90.893 ms | 83.648 ms | 7.97% |

The baseline is an immutable copy of `f47941a` bin/lib files. Each baseline/candidate
pair uses the same source model, input, clock and loop count, with isolated libraries.
Engine creation remains fresh in both revisions. Inference includes the FP32 DFL/box
kernel, so its timing can increase while shared mask work reduces end-to-end latency.
The historical 1.6.11 target has no original model hashes and remains a separate target.

对照是 `f47941a` 的 bin/lib 完整副本，每一对使用相同原始模型、输入、时钟和测量次数，
通过独立库路径隔离；两版均重新构建 engine。推理计时包含 FP32 DFL/框 kernel，可能增加，
公共 mask 计算减少后，端到端延迟仍降低。原 1.6.11 未保留模型 hash，继续作为单独性能目标。

## Reproduce / 复现

```shell
unset YOLO_VALIDATION_DIR
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 10
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 10

python3 tests/validate_supported_pipeline.py \
  --models /path/to/model_a.onnx /path/to/model_b.onnx \
  --baseline-build /path/to/baseline-build --candidate-build build \
  --output build/paired_audit --loop 10
```

The paired tool serializes accuracy and performance jobs and stops if device health
fails. Supply all seven models for the release matrix. It records SHA256 hashes,
per-case logs, raw captures and paired metrics. Use a new output directory each time.

RGB input: `assets/test.jpg`; I420: `assets/test_0.yuv`, 1280×720. RPP clock: 920 MHz;
server event partition: 512 common / 3456 graph events. Tracing and diagnostic capture
are disabled during performance measurement. Results exclude engine construction,
input decoding and visualization; they are repeated-frame application latency.

Local evidence: `build/unified_pipeline_20261121/qualified_audit/summary.json` and
adjacent case logs/captures; immutable baseline provenance: `baseline.json`.

配对工具串行执行精度与性能任务，设备状态异常时停止提交；发布矩阵需提供七个模型。
每次使用新输出目录，保留 hash、逐项日志、原始张量和配对指标。性能采集关闭 trace 与
诊断搬运，不包含构建 engine、输入解码及可视化，口径为重复帧应用延迟。证据路径见上。
