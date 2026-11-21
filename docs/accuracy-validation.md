# Supported-model accuracy validation

Language: [English](#english) | [中文](#chinese)

<a id="english"></a>

## Release Profiles

| Model profile | Input H×W | Runtime model | Inputs | Status |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11m-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11l-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11x-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11n-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |

The application loads static FP32 NCHW COCO80 segmentation exports with batch 1,
prediction `[1,116,N]` and prototype `[1,32,H/4,W/4]`. Every process builds a fresh
engine from the original ONNX; no engine cache or prepared ONNX is required.

## Acceptance and Current Coverage

- Use ONNX Runtime on the exact normalized tensor captured from RPP preprocessing.
- Require equal object count/classes, box IoU ≥ 0.90, mask IoU ≥ 0.95 and maximum
  confidence difference ≤ 0.05. Reject NaN/Inf and non-repeatable raw tensors.
- On 2026-11-21, all seven profiles passed fresh RGB and I420 checks after a power
  cycle using the shared implementation: one process per input, one warmup plus one measured frame, 28 captured
  full-pipeline frames in total. Tables aggregate the worst metric across both inputs.
- All captured prediction/prototype/DFL tensors were finite and repeated exactly
  between the two frames. Both semantic and same-prediction postprocess gates passed.
- Fresh n/m640 three-image switching and replay checks passed six frames each.
  A renamed n640 fixture also passed after replacing every node/tensor name while
  preserving weights. These additional checks are separate from the table below.
- These are sample-level checks, not dataset mAP certification.

| Profile | RGB objects | I420 objects | Min. box IoU | Min. mask IoU | Max. score difference | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n-seg 640x640 | 14 | 2 | 0.983607 | 0.976041 | 0.011898 | PASS |
| YOLO11s-seg 640x640 | 14 | 2 | 0.981729 | 0.991925 | 0.013122 | PASS |
| YOLO11m-seg 640x640 | 14 | 2 | 0.982914 | 0.991439 | 0.007390 | PASS |
| YOLO11l-seg 640x640 | 13 | 2 | 0.985075 | 0.990559 | 0.009253 | PASS |
| YOLO11x-seg 640x640 | 12 | 2 | 0.979399 | 0.986940 | 0.008786 | PASS |
| YOLO11n-seg 960x960 | 14 | 2 | 0.976190 | 0.969609 | 0.007968 | PASS |
| YOLO11s-seg 960x960 | 14 | 2 | 0.992424 | 0.975035 | 0.007608 | PASS |

Fresh evidence: `build/unified_pipeline_20261121/accuracy_summary.json`, with original
per-model reports and hashes under `qualified_audit/accuracy_rgb/` and `qualified_audit/accuracy_yuv/`. Previous
switching records remain archived under `build/accuracy_960_20261121/release_check/`.
Fresh switching reports are `build/unified_pipeline_20261121/sequence_n640/summary.json`
and `sequence_m640/summary.json`; naming evidence is `customer_rename_summary.json`.
Performance was measured separately with capture disabled; see the
[measurement record](performance-baseline.md).

## Reproduce

Supply models locally; the repository contains no ONNX files. Run one model at a time:

```shell
python tests/validate_models.py \
  --models /path/to/yolo11s_seg_640x640.onnx \
  --output build/accuracy_rgb --frames 1 --processes 2 \
  --process-timeout 600 --max-score-diff 0.05

python tests/validate_models.py \
  --models /path/to/yolo11s_seg_960x960.onnx \
  --input-format yuv --binary build/bin/yolov11_seg_yuv_frame_demo \
  --image assets/test_0.yuv --source-width 1280 --source-height 720 \
  --output build/accuracy_i420 --frames 1 --processes 2 \
  --process-timeout 600 --max-score-diff 0.05
```

Use a new output directory per audit. Validation captures warmup and measured
frames, input, prediction, prototype, DFL, geometry and masks. Reports retain model,
binary and library hashes. Capture adds diagnostic host transfers; use
`scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 10` without
`YOLO_VALIDATION_DIR` for performance. See [measurement scope](performance-baseline.md).

---

<a id="chinese"></a>

## 发布支持规格

| 模型 | 输入 H×W | 运行模型 | 输入格式 | 状态 |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11m-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11l-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11x-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11n-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |

使用 batch 1、静态 FP32 NCHW 的 COCO80 分割导出，prediction 为 `[1,116,N]`，
prototype 为 `[1,32,H/4,W/4]`。每个进程从原始 ONNX 新建 engine，不读取 engine cache，
也不要求预先生成适配 ONNX。

## 门槛与本次验证覆盖

- 使用 RPP 前处理实际输出的归一化张量，交由 ONNX Runtime 生成参考。
- 目标数量与类别一致，框 IoU ≥ 0.90、mask IoU ≥ 0.95、最大置信度差 ≤ 0.05；
  拒绝 NaN/Inf 及原始张量重复不一致。
- 2026-11-21，公共实现的七个规格 RGB/I420 均完成新验证：每种输入一个进程，
  一帧预热及一帧测量，共 28 帧完整流水线采集。下表取两种输入中的最差指标。
- prediction/prototype/DFL 全部为有限值，两帧原始张量完全一致；语义精度及相同
  prediction 的后处理对照均通过。样例验证不代表数据集 mAP 认证。
- n/m640 新三图切换与原样重放各六帧通过；保持权重不变、替换所有节点与张量名称的
  n640 模型也通过新验证。这些附加检查不混计到下表的两种输入指标中。

| 模型 | RGB 目标数 | I420 目标数 | 最小框 IoU | 最小 mask IoU | 最大置信度差 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n-seg 640x640 | 14 | 2 | 0.983607 | 0.976041 | 0.011898 | PASS |
| YOLO11s-seg 640x640 | 14 | 2 | 0.981729 | 0.991925 | 0.013122 | PASS |
| YOLO11m-seg 640x640 | 14 | 2 | 0.982914 | 0.991439 | 0.007390 | PASS |
| YOLO11l-seg 640x640 | 13 | 2 | 0.985075 | 0.990559 | 0.009253 | PASS |
| YOLO11x-seg 640x640 | 12 | 2 | 0.979399 | 0.986940 | 0.008786 | PASS |
| YOLO11n-seg 960x960 | 14 | 2 | 0.976190 | 0.969609 | 0.007968 | PASS |
| YOLO11s-seg 960x960 | 14 | 2 | 0.992424 | 0.975035 | 0.007608 | PASS |

新证据位于 `build/unified_pipeline_20261121/accuracy_summary.json`，逐模型报告及 hash 位于
`qualified_audit/accuracy_rgb/` 和 `qualified_audit/accuracy_yuv/`。此前切图记录保留在
`build/accuracy_960_20261121/release_check/`。本次切图报告为
`build/unified_pipeline_20261121/sequence_n640/summary.json` 和 `sequence_m640/summary.json`，
命名验证证据为 `customer_rename_summary.json`。复现命令见英文部分，每次使用新输出目录。

验证保存预热和测量帧的输入、prediction、prototype、DFL、几何与 mask，并记录模型、
二进制和库 hash。采集会增加诊断搬运，不用于性能测量。性能模式清除
`YOLO_VALIDATION_DIR` 后运行 `scripts/benchmark.sh`，口径见[性能记录](performance-baseline.md)。
