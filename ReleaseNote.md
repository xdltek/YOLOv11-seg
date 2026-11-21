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

- All seven profiles now share shape-driven attention adaptation and a single FP32
  DFL/box decoder. Graph matching uses connectivity/shapes rather than exported names.
- One RT NMS graph, anchor-map parser and sparse-mask implementation serve all profiles.
  A single SDK input policy retains direct 8400-anchor input and compact 18900-anchor input;
  forcing compaction for every profile reproduced l640 missing a weak object.
- Merged duplicated interpolation taps into three taps per axis. Removed unused CPU
  NMS/mask helpers, obsolete CUDA kernels and the dense-mask environment switch.
- Accuracy tooling now validates the unchanged source ONNX through the application;
  offline graph-rewrite alternatives have been removed. Added a serial paired-audit tool.

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
[performance record](docs/performance-baseline.md).

### Validation Record

The shared implementation passed fresh RGB/I420 checks for all seven profiles
(28 captured full-pipeline frames), and paired performance runs against `f47941a`
(308 frames including warmup across both revisions). Counts/classes match CPU ONNX
Runtime, box IoU ≥ 0.90, mask IoU ≥ 0.95 and score difference ≤ 0.05; raw outputs
are finite and repeatable. All 14 paired latency checks passed.
Device DFL, stable compaction, precision-module, host model-policy and merged
interpolation regressions passed. No column/sequencer busy was reproduced.
These are sample-level checks; see the [accuracy guide](docs/accuracy-validation.md).
Additional n/m640 switching/replay checks passed 12 frames; a renamed n640 model
passed two captured frames with identical weight payloads and new node/tensor names.

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

- 七个规格共用按形状驱动的 attention 适配及一个 FP32 DFL/框解码实现，图匹配依赖
  连接关系和形状，不依赖导出名称。
- 共用一个 RT NMS 图、anchor 映射解析及稀疏 mask 实现。仅保留一处 SDK 输入策略：
  8400-anchor 直接输入、18900-anchor 稳定压缩；强制全部压缩会复现 l640 弱目标漏检。
- 将重复插值采样合并为每轴三个 tap，删除未调用的 CPU NMS/mask、旧 CUDA kernel 和稠密 mask 开关。
- 精度工具仅通过应用验证原始 ONNX，移除离线导图的多种替代模式，新增串行配对回归工具。

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
原始目标、配对对照及复现命令见[性能记录](docs/performance-baseline.md)。

### 验证记录

公共实现通过七规格 RGB/I420 新精度检查（28 帧完整流水线采集），并与 `f47941a`
逐项配对跑分（两版含预热共 308 帧）。目标数量、类别与 CPU ONNX Runtime 一致，
框 IoU ≥ 0.90、mask IoU ≥ 0.95、置信度差 ≤ 0.05，原始输出有限且重复一致；十四项配对延迟检查通过。
设备 DFL、稳定候选压缩、精度模块、主机规格策略及合并插值回归通过，未复现 column/sequencer busy。
样例检查不代表数据集 mAP 认证，详见[精度指南](docs/accuracy-validation.md)。
附加 n/m640 切图重放共十二帧通过，保持权重不变、替换节点及张量名称的 n640 模型两帧亦通过。
