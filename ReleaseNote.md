# Release Notes

## Unreleased

- Standardized the build, run, benchmark, CLI, console banner, and model-package behavior.
- Added persistent OpenRT engine caching. The first execution builds and stores an engine; later executions deserialize it directly.
- Added model-content/OpenRT-version cache identity, engine checksum validation, and automatic ONNX rebuild for corrupted or incompatible caches.
- Added structured application output: stage banners, warmup/benchmark messages, progress bar, dual FPS highlights, pipeline breakdown, result section, and success status.

## YOLOv11-SEG RPP Inference Demo V2.0

This release provides a standardized, standalone YOLOv11-SEG project structure and an end-to-end RPP segmentation path.

### Validated Models

| Model | Input | End-to-end | FPS | Status |
| --- | --- | ---: | ---: | --- |
| YOLO11n-seg | 640×640 | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 45.463 ms | 21.996 | PASS |
| YOLO11n-seg | 1024×1280 | 46.386 ms | 21.558 | PASS |

Each model passed ONNX checking, OpenRT/RPP engine construction, one warmup plus 10 measured RGB loops, an I420 smoke test, and rendered-output validation. Performance describes the current local RPP environment.

### Added

- Multi-output YOLOv11 segmentation inference support.
- RPP RT NMS with selected-anchor mask coefficient handling.
- Sparse prototype ROI combination, bilinear resize, threshold, and packed uint8 mask output on RPP.
- RGB/BGR image and I420 YUV frame demos.
- Standalone postprocess accuracy and performance test under `tests/`.
- Supported-model and measured-performance tables in the root README.

### Changed

- Standardized root README sections, bilingual presentation, repository navigation, and performance format.
- Moved `AI_Log.md` to the repository root.
- Restricted `demos/` to the two customer-facing application entries.
- Added explicit PRE/POST transient SRAM lifetime boundaries for supported large static models.
- Standardized the default model as `weights/yolo11s-seg_640x640_sim.onnx` across scripts and binaries.

### Removed

- Removed the unused legacy `yolov11_nms_pre` plugin directory.
- Removed unsupported-model details from customer-facing README and release tables.

---

## YOLOv11-SEG RPP 推理 Demo V2.0

本版本提供标准化、独立的 YOLOv11-SEG 工程结构和端到端 RPP 实例分割执行链。

### 已验证模型

| 模型 | 输入 | 端到端 | FPS | 状态 |
| --- | --- | ---: | ---: | --- |
| YOLO11n-seg | 640×640 | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 45.463 ms | 21.996 | PASS |
| YOLO11n-seg | 1024×1280 | 46.386 ms | 21.558 | PASS |

所有模型均通过 ONNX 检查、OpenRT/RPP engine 构建、一次 warmup 加 10 次 RGB 测量、I420 smoke test 和渲染结果检查。性能仅代表当前本地 RPP 环境。

### 新增

- YOLOv11 双输出 segmentation inference。
- RPP RT NMS 和 selected-anchor mask coefficient 处理。
- RPP Sparse Prototype ROI combine、双线性 resize、threshold 和 packed uint8 mask 输出。
- RGB/BGR 图片和 I420 YUV 帧 Demo。
- `tests/` 下的独立后处理精度与性能测试。
- 根 README 中的支持模型和实测性能表。

### 变更

- 统一根 README 章节、中英双语、仓库导航和性能表格式。
- `AI_Log.md` 移到仓库根目录。
- `demos/` 仅保留两个用户应用入口。
- 为已支持的大尺寸静态模型增加明确的 PRE/POST transient SRAM 生命周期。
- 脚本和二进制默认模型统一为 `weights/yolo11s-seg_640x640_sim.onnx`。

### 移除

- 删除未使用的旧 `yolov11_nms_pre` plugin 目录。
- 从对外 README 和 Release 表格移除不支持模型的说明。
