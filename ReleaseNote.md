# Release Notes

## CPU_NMS Complete Version

This version completes the optimized CPU_NMS pipeline:

- RPP decodes and thresholds fixed candidate slots; CPU performs compact, sort, and greedy NMS.
- Selected mask coefficients remain on the RPP path and are gathered by anchor index.
- Sparse Prototype ROI replaces full `160×160` per-instance mask generation.
- Packed uint8 ROI masks use direct SRAM-to-pinned-Host return.
- RGB and I420 inputs share persistent pipeline resources and a common runtime stream.
- Failed GEMM/BF16 experiments and their `rppblas` dependency are not part of the release path.

Validation completed with a full project build, synthetic postprocess accuracy regression, and RGB/I420 end-to-end runs on RPP. RT_NMS and full graph integration are deferred to the next version.

---

## CPU_NMS完成版

本版本完成优化后的CPU_NMS执行链：

- RPP生成并过滤候选槽，CPU执行compact、排序和greedy NMS。
- selected mask coefficient按anchor index在RPP侧完成gather。
- 使用Sparse Prototype ROI替代每实例完整`160×160` mask计算。
- packed uint8 ROI mask直接从SRAM回传pinned Host。
- RGB和I420输入复用持久化pipeline资源及统一Runtime stream。
- 已失败的GEMM/BF16实验和`rppblas`依赖不进入正式版本。

已完成完整工程构建、synthetic后处理精度回归，以及RGB/I420端到端RPP运行验证。RT_NMS和全流程图内集成留待下一版本。

---

## YOLOv11-SEG RPP Det Demo Initialization

This release initializes the YOLOv11-SEG RPP det demo repository.

### What Was Added

- Added a complete YOLOv11-SEG object detection demo for RPP hardware.
- Added RGB image and I420 YUV frame demo entry points.
- Added default build, run, test, and benchmark scripts under `scripts/`.
- Added sample assets under `assets/` for quick local validation.
- Added default output, log, and trace directory handling through the build and run scripts.
- Added public README documentation for building, running, benchmarking, and adapting the demo.
- Added model directory preparation through `weights/`, while keeping local model files out of Git.

### User Workflow

- Build the demo with `scripts/build.sh`.
- Run the default demos with `scripts/run.sh`.
- Run a smoke check with `scripts/test.sh`.
- Run benchmark loops with `scripts/benchmark.sh`.

### Notes

- The default model path is `weights/yolo11s-seg.onnx`.
- If the default model is missing, place `yolo11s-seg.onnx` in the repository root or under `weights/`.
- Generated build outputs, logs, traces, and model files are intentionally ignored by Git.

---

## YOLOv11-SEG RPP Det Demo 初始化

本次发布用于初始化 YOLOv11-SEG RPP det demo 仓库。

### 做了什么

- 新增完整的 YOLOv11-SEG 目标检测 demo，用于 RPP 硬件运行。
- 新增 RGB 图片和 I420 YUV 帧两类 demo 入口。
- 新增 `scripts/` 下的默认构建、运行、测试和 benchmark 脚本。
- 新增 `assets/` 示例资源，便于快速本地验证。
- 新增默认输出、日志和 trace 目录处理，随构建和运行脚本自动使用。
- 新增公开 README 文档，覆盖构建、运行、benchmark 和 demo 适配方式。
- 新增 `weights/` 模型目录准备，同时保持本地模型文件不进入 Git。

### 用户流程

- 使用 `scripts/build.sh` 构建 demo。
- 使用 `scripts/run.sh` 运行默认 demo。
- 使用 `scripts/test.sh` 进行基础检查。
- 使用 `scripts/benchmark.sh` 运行 benchmark loop。

### 说明

- 默认模型路径为 `weights/yolo11s-seg.onnx`。
- 如果默认模型不存在，可将 `yolo11s-seg.onnx` 放在仓库根目录或 `weights/` 下。
- 构建产物、日志、trace 和模型文件会被 Git 忽略。
