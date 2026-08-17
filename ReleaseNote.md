# Release Notes

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
