# Release Notes

Language: [English](#english) | [中文](#chinese)

<a id="english"></a>

## 1.6.11 — Supported Model Scope

This release consolidates the OpenRT/RPP segmentation application around the
following model profiles, for RGB/BGR images and I420 frames.

| Model profile | Input H×W | Runtime model | Inputs | Status |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11m-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11l-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11x-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11n-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |

### Changes

- Retained the preprocessing, inference and postprocessing modules as independent
  OpenRT/RPP stages, with the existing console reports and output visualization.
- Builds a fresh in-memory engine from the original ONNX in every process;
  engine serialization, deserialization and cache loading are disabled.
- Retained convolution bias/rounding corrections, FP32 device box decoding,
  supported attention adaptation and stable RT NMS candidate compaction.
- Removed experimental nearest-upsample replacement, confidence-sigmoid utilities
  and their dedicated diagnostic tests. No experimental environment switch remains.
- Added graph-based release-profile checks before engine construction.
- The startup script prepares a local server configuration with enough graph events
  for the supported models; explicit `RPP_CFG_FILE` settings are respected. Installed SDK files are unchanged.
- Kept only `build.sh`, `run.sh` and `benchmark.sh` in `scripts/`.
- ONNX files remain user-provided and excluded from the repository and packages.

### Measured RGB Performance (2026-11-21)

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.875 ms | 7.109 ms | 41.581 ms | 56.565 ms | 17.679 | PASS |
| YOLO11s-seg | 640×640 | 7.737 ms | 22.997 ms | 42.443 ms | 73.177 ms | 13.665 | PASS |
| YOLO11m-seg | 640×640 | 7.871 ms | 44.528 ms | 42.782 ms | 95.181 ms | 10.506 | PASS |
| YOLO11l-seg | 640×640 | 7.717 ms | 49.193 ms | 41.177 ms | 98.087 ms | 10.195 | PASS |
| YOLO11x-seg | 640×640 | 7.795 ms | 120.255 ms | 40.465 ms | 168.516 ms | 5.934 | PASS |
| YOLO11n-seg | 960×960 | 10.734 ms | 28.990 ms | 54.607 ms | 94.330 ms | 10.601 | PASS |
| YOLO11s-seg | 960×960 | 10.668 ms | 44.781 ms | 56.132 ms | 111.581 ms | 8.962 | PASS |

### Measured I420 Performance (2026-11-21)

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.189 ms | 7.329 ms | 22.579 ms | 37.096 ms | 26.957 | PASS |
| YOLO11s-seg | 640×640 | 7.262 ms | 22.701 ms | 22.839 ms | 52.803 ms | 18.938 | PASS |
| YOLO11m-seg | 640×640 | 7.229 ms | 43.981 ms | 22.732 ms | 73.941 ms | 13.524 | PASS |
| YOLO11l-seg | 640×640 | 7.179 ms | 48.249 ms | 22.400 ms | 77.828 ms | 12.849 | PASS |
| YOLO11x-seg | 640×640 | 7.434 ms | 120.362 ms | 22.916 ms | 150.712 ms | 6.635 | PASS |
| YOLO11n-seg | 960×960 | 10.467 ms | 29.064 ms | 36.084 ms | 75.615 ms | 13.225 | PASS |
| YOLO11s-seg | 960×960 | 10.493 ms | 44.733 ms | 35.984 ms | 91.210 ms | 10.964 | PASS |

All seven profiles use the consolidated source and original ONNX models, measured serially after a power cycle.
Each input uses one warmup and ten measured frames, capture disabled, at a 920 MHz RPP clock.
PRE includes input H2D; POST includes output D2H. Engine construction, image decoding and visualization
are excluded. PASS in these tables denotes completed runtime/performance measurement.

The accuracy-corrected pipeline remains slower than the original 1.6.11 baseline; no speedup is claimed.
The full-application one-H2D/one-D2H objective is still incomplete. Baseline, measurement details and
commands: [performance record](docs/performance-baseline.md).

### Validation Record

On 2026-11-21, all seven profiles passed fresh RGB/I420 sample accuracy checks
(28 captured full-pipeline frames) and performance runs (154 frames including
warmup). Counts/classes matched ONNX Runtime, box IoU ≥ 0.90, mask IoU ≥ 0.95 and
confidence difference ≤ 0.05; captured tensors were finite and repeatable.
The device box/DFL, stable candidate-compaction and precision-module regressions
also passed. No column/sequencer busy condition was reproduced during these runs.
These are sample-level checks; see the [accuracy guide](docs/accuracy-validation.md).

---

<a id="chinese"></a>

## 1.6.11 — 模型支持范围收敛

本版本围绕下列规格收敛 OpenRT/RPP 实例分割应用，输入支持 RGB/BGR 图片和 I420 帧。

| 模型 | 输入 H×W | 运行模型 | 输入格式 | 状态 |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11m-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11l-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11x-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11n-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |

### 变更

- 保留前处理、推理、后处理独立的 OpenRT/RPP 模块、原有控制台报告及结果可视化。
- 每个进程从原始 ONNX 新建内存 engine，关闭 engine 序列化、反序列化及 cache 加载。
- 保留卷积 bias/舍入修正、设备 FP32 框解码、已支持的 attention 适配和 RT NMS 稳定候选压缩。
- 移除实验性最近邻上采样替换、置信度 sigmoid 工具及其专用诊断测试，不再保留实验环境开关。
- 构建 engine 前按模型图检查发布规格。
- 启动脚本生成本地服务配置，为支持模型预留图事件；尊重显式 `RPP_CFG_FILE`，不修改 SDK 文件。
- `scripts/` 仅保留 `build.sh`、`run.sh`、`benchmark.sh`。
- ONNX 继续由用户提供，不进入仓库或发布包。

### RGB 实测性能（2026-11-21）

| 模型 | 输入 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.875 ms | 7.109 ms | 41.581 ms | 56.565 ms | 17.679 | PASS |
| YOLO11s-seg | 640×640 | 7.737 ms | 22.997 ms | 42.443 ms | 73.177 ms | 13.665 | PASS |
| YOLO11m-seg | 640×640 | 7.871 ms | 44.528 ms | 42.782 ms | 95.181 ms | 10.506 | PASS |
| YOLO11l-seg | 640×640 | 7.717 ms | 49.193 ms | 41.177 ms | 98.087 ms | 10.195 | PASS |
| YOLO11x-seg | 640×640 | 7.795 ms | 120.255 ms | 40.465 ms | 168.516 ms | 5.934 | PASS |
| YOLO11n-seg | 960×960 | 10.734 ms | 28.990 ms | 54.607 ms | 94.330 ms | 10.601 | PASS |
| YOLO11s-seg | 960×960 | 10.668 ms | 44.781 ms | 56.132 ms | 111.581 ms | 8.962 | PASS |

### I420 实测性能（2026-11-21）

| 模型 | 输入 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 7.189 ms | 7.329 ms | 22.579 ms | 37.096 ms | 26.957 | PASS |
| YOLO11s-seg | 640×640 | 7.262 ms | 22.701 ms | 22.839 ms | 52.803 ms | 18.938 | PASS |
| YOLO11m-seg | 640×640 | 7.229 ms | 43.981 ms | 22.732 ms | 73.941 ms | 13.524 | PASS |
| YOLO11l-seg | 640×640 | 7.179 ms | 48.249 ms | 22.400 ms | 77.828 ms | 12.849 | PASS |
| YOLO11x-seg | 640×640 | 7.434 ms | 120.362 ms | 22.916 ms | 150.712 ms | 6.635 | PASS |
| YOLO11n-seg | 960×960 | 10.467 ms | 29.064 ms | 36.084 ms | 75.615 ms | 13.225 | PASS |
| YOLO11s-seg | 960×960 | 10.493 ms | 44.733 ms | 35.984 ms | 91.210 ms | 10.964 | PASS |

七个规格均使用当前收敛代码及原始 ONNX，在断电重启后串行实测。每个输入预热 1 帧、
测量 10 帧，关闭精度采集，RPP 主频 920 MHz。前处理包含输入 H2D，后处理包含输出 D2H；
不包含 engine 构建、图片读取和可视化。表中 PASS 表示该项运行与性能测量通过。

当前精度修复版仍慢于原 1.6.11 基准，不宣称性能提升；完整应用的一次 H2D/一次 D2H 目标尚未完成。
原基准、逐阶段口径及复现命令见[性能记录](docs/performance-baseline.md)。

### 验证记录

2026-11-21，七个规格均通过新的 RGB/I420 样例精度检查（28 帧完整流水线采集）与
性能测试（含预热共 154 帧）。目标数量、类别与 ONNX Runtime 一致，框 IoU ≥ 0.90、
mask IoU ≥ 0.95、置信度差 ≤ 0.05，采集张量为有限值且重复一致。
设备框/DFL 解码、稳定候选压缩及卷积精度模块回归亦通过；本轮运行未复现
column/sequencer busy。样例检查不等于数据集 mAP 认证，详见[精度指南](docs/accuracy-validation.md)。
