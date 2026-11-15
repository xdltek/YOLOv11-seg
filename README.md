<a id="english"></a>

# RPP-Accelerated YOLOv11-SEG Application

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

![XDL Logo](docs/logo/logo_color_horizontal.png)

This C++17 application runs YOLOv11 instance segmentation through OpenRT on RPP hardware. It provides RGB/BGR and I420 inputs, RPP preprocessing, multi-output inference, RT NMS, sparse mask reconstruction, visualization, progress reporting, and pipeline performance measurements.

> **Model policy:** Git and release packages do not contain or download ONNX files. Provide a compatible model with `--model`, or locally place the optional fallback at `weights/yolo11s-seg_640x640_sim.onnx`.

**Table Of Contents**

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Demo Guides](#demo-guides)
- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
- [How to Build](#how-to-build)
- [How to Run](#how-to-run)
- [Direct Binary Usage](#direct-binary-usage)
- [Code Execution Workflow](#code-execution-workflow)
- [Key Source Files](#key-source-files)
- [Reusable Components](#reusable-components)
- [How to Adapt This Demo](#how-to-adapt-this-demo)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Validated Models](#validated-models)
- [Appendix: Export ONNX From a PT Model](#appendix-export-onnx-from-a-pt-model)

## Quick Start

Run from the repository root. Building does not require a model.

```shell
scripts/build.sh
scripts/run.sh --model /path/to/yolo11-seg.onnx
```

Successful execution prints stage progress, end-to-end and OpenRT FPS, pipeline timing, and saves rendered images under `build/output/`.

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Cleans generated outputs, configures CMake, and builds without preparing a model. |
| `scripts/run.sh` | Runs both demos, or one selected workload with configurable paths and loops. |
| `scripts/test.sh` | Checks the selected model, executables, and sample inputs. |
| `scripts/benchmark.sh` | Runs repeated measurements and archives the console log. |

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
scripts/run.sh --model /path/to/model.onnx --demo yuv
scripts/benchmark.sh --model /path/to/model.onnx --loop 20
```

Paths are resolved from the repository root. RGB and YUV run serially unless `--demo` selects one workload. If `/usr/local/rpp/bin/rpp_server` exists, the script ensures it is running and logs under `build/logs/`.

## Repository Layout

The public repository layout separates application entry points, reusable pipeline code, RPP modules, tests, scripts, documentation, and local model storage:

```text
yolov11-seg/
├── CMakeLists.txt
├── README.md
├── ReleaseNote.md
├── AI_Log.md
├── 3rd_party/argparse/       # command-line parser
├── assets/                   # RGB and I420 sample inputs
├── demos/
│   ├── rgb_image/            # RGB/BGR executable and guide
│   └── yuv_frame/            # I420 executable and guide
├── docs/                     # public docs, API index and assets
├── rpp/
│   ├── inference/            # multi-output OpenRT and engine cache
│   ├── preprocess/           # upload, letterbox, color, normalize
│   └── postprocess/          # RT NMS and sparse mask reconstruction
├── scripts/                  # build, run, test, benchmark
├── src/
│   ├── perf/                 # optional rpp_perf integration
│   ├── utils/                # logging, parser and path helpers
│   └── yolov11_seg/          # pipeline, reporter, labels, visualization
├── tests/                    # focused postprocess validation
└── weights/                  # local models; ONNX is Git-ignored
```

| Path | Responsibility |
|---|---|
| `CMakeLists.txt` | Root build entry for libraries, applications, tests, and assets. |
| `src/yolov11_seg/` | Application pipeline and customer-facing reporting. |
| [`rpp/`](rpp/README.md) | Hardware-facing inference, preprocessing, and postprocessing. |
| `demos/` | RGB and I420 customer application entries. |
| [`docs/`](docs/README.md) | Public documentation and API navigation. |
| `scripts/` | Reproducible build, run, test, and benchmark commands. |
| `weights/` | Empty release placeholder for user-provided models. |

## Demo Guides

- [RGB image demo](demos/rgb_image/README.md): input, CLI, workflow, and output.
- [I420 YUV frame demo](demos/yuv_frame/README.md): frame geometry, CLI, workflow, and output.

## Documentation

- [Project documentation](docs/README.md)
- [API documentation](docs/api/README.md)
- [RPP modules](rpp/README.md)
- [Performance tracing](src/perf/README.md)
- [Release notes](ReleaseNote.md)
- [AI development log](AI_Log.md)

## Prerequisites

- Linux development environment
- CMake `>= 3.16.3` and a C++17 compiler
- Required OpenCV components
- OpenRT/RPP installed under `/usr/local/rpp`
- Target RPP driver, firmware, and runtime libraries

Update paths in `CMakeLists.txt` when OpenRT/RPP is installed elsewhere.

## How to Build

```shell
scripts/build.sh
```

Build does not require, download, or verify ONNX. Expected outputs:

```text
build/bin/      applications and test executables
build/lib/      pipeline and RPP shared libraries
build/assets/   copied sample inputs
build/output/   rendered segmentation images
build/trace/    optional rpp_perf traces
build/logs/     server and benchmark logs
```

## How to Run

```shell
scripts/run.sh --model /path/to/yolo11-seg.onnx
scripts/run.sh --model /path/to/yolo11-seg.onnx --demo rgb
scripts/run.sh --model /path/to/yolo11-seg.onnx --demo yuv

scripts/benchmark.sh --model /path/to/yolo11-seg.onnx --loop 20
```

All scripts support `-h` and `--help`. Benchmark logs use `build/logs/benchmark_<workload>_<timestamp>.log`.

## Direct Binary Usage

Scripts are recommended because they validate paths and manage `rpp_server`. Direct usage is also supported:

```shell
build/bin/yolov11_seg_rgb_image_demo \
  -m /path/to/yolo11-seg.onnx \
  -i build/assets/test.jpg \
  -o build/output/output_rgb_seg.jpg -l 1

build/bin/yolov11_seg_yuv_frame_demo \
  -m /path/to/yolo11-seg.onnx \
  -i build/assets/test_0.yuv \
  --yuv-width 1280 --yuv-height 720 \
  -o build/output/output_yuv_seg.jpg -l 1
```

### OpenRT Engine Cache

The first run builds a BF16 engine and stores it beside the model under `.engine_cache/`. Later processes deserialize it when the ONNX content fingerprint, OpenRT version, precision, and integrity checksum match.

```text
/path/to/.engine_cache/
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine.fnv1a
```

Changed models, missing checksums, truncated or corrupt files, incompatible formats, and deserialization failures trigger an automatic ONNX rebuild. Engine files are local runtime artifacts and are not portable models.

## Code Execution Workflow

```text
+------------------------+
| Load RGB or I420 input |
+------------------------+
            |
            v
+------------------------+
| Reuse valid engine or  |
| build it from ONNX     |
+------------------------+
            |
            v
+------------------------+
| RPP H2D, color,        |
| letterbox, normalize   |
+------------------------+
            |
            v
+------------------------+
| OpenRT prediction and  |
| prototype inference    |
+------------------------+
            |
            v
+------------------------+
| RPP RT NMS and sparse  |
| instance-mask decode   |
+------------------------+
            |
            v
+------------------------+
| Render and report      |
+------------------------+
```

One full pipeline warmup precedes measured loops. CPU handles decoding, compact result assembly, and visualization; preprocessing, inference, NMS, coefficient gathering, prototype combination, ROI resize, and thresholding use RPP/OpenRT.

## Key Source Files

- `src/yolov11_seg/yolo_pipeline.h/.cpp`: end-to-end application pipeline.
- `src/yolov11_seg/segmentation_types.h`: results, geometry, configuration, and timing.
- `src/yolov11_seg/demo_reporter.h`: standardized progress and performance output.
- `src/yolov11_seg/labels.cpp`: COCO 80-class labels.
- `src/yolov11_seg/visualization.cpp`: masks, boxes, labels, and image output.
- `rpp/inference/rpp_infer_engine.h/.cpp`: multi-output inference and engine cache.
- `rpp/preprocess/rpp_yolo_preprocessor.h/.cpp`: RGB/BGR and I420 preprocessing.
- `rpp/postprocess/rpp_yolov11_seg_postprocessor.h/.cpp`: RT NMS and masks.

## Reusable Components

- `YoloV11SegPipeline`: application API for RGB/BGR and packed I420.
- `RppInferEngine`: engine, bindings, multi-output execution, and cache.
- `RppYoloPreprocessor`: upload, color conversion, letterbox, normalization, layout.
- `YoloV11SegPostprocessor`: instance selection and RPP mask reconstruction.
- `PerfTraceSession`: optional performance tracing.
- `demo_reporter`: standardized progress and performance output.

## How to Adapt This Demo

1. Keep the standard scripts as external entry points.
2. Export a static YOLO11 segmentation ONNX using the contract below.
3. Pass the model through `--model`; do not add it to Git.
4. Add a demo loader when RGB/BGR and packed I420 are insufficient.
5. Keep generic inference separate from masks, labels, rendering, and reporting.
6. Revalidate shapes, accuracy, images, and performance after model changes.

## Known Limitations

- Runtime support is limited to the static FP32 contracts under [Validated Models](#validated-models).
- Batch size is `1`; prediction and prototype must use the documented two-output layout.
- The application embeds COCO 80-class labels.
- The repository does not distribute, download, or validate ONNX model files.
- Performance is local validation data, not a cross-platform guarantee.

## Troubleshooting

- **No model:** use `scripts/run.sh --model /path/to/model.onnx`.
- **Executable missing:** run `scripts/build.sh`.
- **Model rejected:** verify static input, prediction, prototype, opset, and export settings.
- **Cache rejected:** automatic rebuild is expected after model/runtime changes or damage.
- **Cache not persisted:** ensure the model directory is writable.
- **Server failure:** inspect `<build-dir>/logs/rpp_server.log`.
- **Unexpected masks:** confirm the segmentation class and mask-channel contract.

## Validated Models

Only runtime-validated models are listed. ONNX files were local test inputs and are not Git or release artifacts.

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 3.882 ms | 4.935 ms | 9.306 ms | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 3.916 ms | 9.216 ms | 10.071 ms | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 4.079 ms | 20.554 ms | 11.554 ms | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 3.997 ms | 25.465 ms | 11.415 ms | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 3.985 ms | 56.183 ms | 7.952 ms | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 5.469 ms | 12.372 ms | 24.546 ms | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 5.411 ms | 20.985 ms | 19.068 ms | 45.463 ms | 21.996 | PASS |
| YOLO11n-seg | 1024×1280 | 6.537 ms | 18.127 ms | 21.722 ms | 46.386 ms | 21.558 | PASS |

Results average 10 RGB loops after one warmup. Initialization and visualization are excluded. Preprocess includes input H2D and postprocess includes output D2H in this dataset.

> **Review required:** Confirm accelerator, SDK, driver, firmware, clocks, utilization, and power before external commitments.

## Appendix: Export ONNX From a PT Model

### Export Requirements

| Setting | Required value |
|---|---|
| Task | YOLO11 instance segmentation |
| Input | FP32 NCHW `[1,3,H,W]`, batch `1`, static dimensions |
| Prediction | FP32 `[1,4+classes+mask_channels,N]` |
| Prototype | FP32 `[1,mask_channels,mask_h,mask_w]` |
| COCO contract | Prediction `116` channels; prototype `32` channels |
| Embedded NMS | Disabled; application performs RPP RT NMS |
| Simplification | Enabled |

### Install Ultralytics and Export

```shell
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python3 -m pip install -e .
python3 -m pip install onnx onnxruntime onnxslim

yolo export \
  model=yolo11s-seg.pt task=segment format=onnx imgsz=640 \
  batch=1 opset=17 dynamic=False simplify=True nms=False
```

For rectangular models, Ultralytics `imgsz` uses height then width.

### Verify the Exported Contract

```shell
python3 - <<'PY'
import onnx
p = "/path/to/yolo11-seg.onnx"
m = onnx.load(p, load_external_data=False)
onnx.checker.check_model(m)
shape = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
print("inputs:", [(v.name, shape(v)) for v in m.graph.input])
print("outputs:", [(v.name, shape(v)) for v in m.graph.output])
print("ONNX check: PASS")
PY
```

Expected 640×640 COCO tensors:

```text
input:   [1, 3, 640, 640]
output0: [1, 116, 8400]
output1: [1, 32, 160, 160]
```

---

<a id="chinese"></a>

# RPP 加速 YOLOv11-SEG 应用

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应章节。

![XDL Logo](docs/logo/logo_color_horizontal.png)

本 C++17 应用通过 OpenRT 在 RPP 硬件上运行 YOLOv11 实例分割，支持 RGB/BGR 和 I420 输入、RPP 前处理、双输出推理、RT NMS、稀疏 mask 重建、可视化、进度显示和 pipeline 性能测量。

> **模型策略：** Git 和发布包不包含、也不下载 ONNX。运行时通过 `--model` 指定模型，或由用户自行放置 `weights/yolo11s-seg_640x640_sim.onnx`。

**目录**

- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [Demo 指南](#demo-指南)
- [文档导航](#文档导航)
- [环境依赖](#环境依赖)
- [如何构建](#如何构建)
- [如何运行](#如何运行)
- [直接执行二进制](#直接执行二进制)
- [代码执行流程](#代码执行流程)
- [关键源码](#关键源码)
- [可复用组件](#可复用组件)
- [如何适配此 Demo](#如何适配此-demo)
- [已知限制](#已知限制)
- [故障排查](#故障排查)
- [已验证模型](#已验证模型)
- [附录：从 PT 模型导出 ONNX](#附录从-pt-模型导出-onnx)

## 快速开始

```shell
scripts/build.sh
scripts/run.sh --model /path/to/yolo11-seg.onnx
```

构建不需要模型。运行成功后打印阶段进度、端到端/OpenRT FPS 和 pipeline 耗时，并在 `build/output/` 保存结果。

| 脚本 | 作用 |
|---|---|
| `scripts/build.sh` | 清理生成物、配置 CMake 并构建，不处理模型。 |
| `scripts/run.sh` | 运行两个 Demo，或选择 workload、输入输出和循环次数。 |
| `scripts/test.sh` | 检查模型、程序和样例输入。 |
| `scripts/benchmark.sh` | 多次测量并归档控制台日志。 |

## 仓库结构

公开仓库结构将应用入口、可复用 pipeline、RPP 模块、测试、脚本、文档和本地模型目录分层组织：

```text
yolov11-seg/
├── CMakeLists.txt
├── README.md
├── ReleaseNote.md
├── AI_Log.md
├── 3rd_party/argparse/       # 命令行解析
├── assets/                   # RGB 与 I420 样例
├── demos/
│   ├── rgb_image/            # RGB/BGR 程序与文档
│   └── yuv_frame/            # I420 程序与文档
├── docs/                     # 对外文档与 API
├── rpp/
│   ├── inference/            # 双输出推理与 engine 缓存
│   ├── preprocess/           # 上传、letterbox、颜色、归一化
│   └── postprocess/          # RT NMS 与稀疏 mask
├── scripts/                  # build、run、test、benchmark
├── src/
│   ├── perf/                 # 可选 trace
│   ├── utils/                # 通用工具
│   └── yolov11_seg/          # pipeline、打印、标签、可视化
├── tests/                    # 后处理验证
└── weights/                  # 本地模型；ONNX 被 Git 忽略
```

## Demo 指南

- [RGB 图片 Demo](demos/rgb_image/README.md)
- [I420 YUV Demo](demos/yuv_frame/README.md)

## 文档导航

- [项目文档](docs/README.md)
- [API 文档](docs/api/README.md)
- [RPP 模块](rpp/README.md)
- [性能 Trace](src/perf/README.md)
- [Release Notes](ReleaseNote.md)
- [AI 开发日志](AI_Log.md)

## 环境依赖

- Linux、CMake `>= 3.16.3`、C++17 编译器
- 工程所需 OpenCV 组件
- `/usr/local/rpp` 下的 OpenRT/RPP
- 目标 RPP Driver、Firmware 和 Runtime

## 如何构建

```shell
scripts/build.sh
```

构建不下载、检查或要求 ONNX。生成物位于 `build/bin`、`build/lib`、`build/assets`、`build/output`、`build/trace` 和 `build/logs`。

## 如何运行

```shell
scripts/run.sh --model /path/to/yolo11-seg.onnx
scripts/run.sh --model /path/to/yolo11-seg.onnx --demo rgb
scripts/run.sh --model /path/to/yolo11-seg.onnx --demo yuv
scripts/benchmark.sh --model /path/to/yolo11-seg.onnx --loop 20
```

所有脚本支持 `-h` 和 `--help`。

## 直接执行二进制

```shell
build/bin/yolov11_seg_rgb_image_demo \
  -m /path/to/yolo11-seg.onnx -i build/assets/test.jpg \
  -o build/output/output_rgb_seg.jpg -l 1

build/bin/yolov11_seg_yuv_frame_demo \
  -m /path/to/yolo11-seg.onnx -i build/assets/test_0.yuv \
  --yuv-width 1280 --yuv-height 720 \
  -o build/output/output_yuv_seg.jpg -l 1
```

### OpenRT Engine 缓存

首次运行从 ONNX 构建 BF16 engine 并保存到模型同目录的 `.engine_cache/`。后续在 ONNX 指纹、OpenRT 版本、精度和完整性校验匹配时直接加载。模型变化、校验缺失、文件损坏、格式不兼容或反序列化失败时自动重建。Engine 是本机生成物，不能作为可移植模型发布。

## 代码执行流程

```text
加载 RGB/I420
    -> 校验并加载 engine，或从 ONNX 构建
    -> RPP H2D、颜色转换、letterbox、normalize
    -> OpenRT prediction + prototype
    -> RPP RT NMS、系数 gather、稀疏 mask
    -> 绘制实例、保存图片、打印报告
```

CPU 负责解码、紧凑结果组装和可视化；前处理、推理、NMS、prototype combine、ROI resize 和 threshold 走 RPP/OpenRT。

## 关键源码

- `src/yolov11_seg/yolo_pipeline.h/.cpp`：端到端 pipeline。
- `src/yolov11_seg/segmentation_types.h`：结果、几何、配置和计时。
- `src/yolov11_seg/demo_reporter.h`：统一的进度和性能打印。
- `src/yolov11_seg/visualization.cpp`：mask、box、label 和图片。
- `rpp/inference/rpp_infer_engine.h/.cpp`：双输出推理和缓存。
- `rpp/preprocess/rpp_yolo_preprocessor.h/.cpp`：RGB/BGR 与 I420 前处理。
- `rpp/postprocess/rpp_yolov11_seg_postprocessor.h/.cpp`：RT NMS 与 mask。

## 可复用组件

- `YoloV11SegPipeline`
- `RppInferEngine`
- `RppYoloPreprocessor`
- `YoloV11SegPostprocessor`
- `PerfTraceSession`
- `demo_reporter`

## 如何适配此 Demo

1. 保留标准脚本入口。
2. 按下方契约导出静态 segmentation ONNX。
3. 使用 `--model` 传入，不提交模型。
4. 需要其他输入时新增 Demo loader。
5. 保持通用推理与 mask、标签、绘图、报告分层。
6. 模型变化后重新验证 shape、精度、结果和性能。

## 已知限制

- 仅支持 [已验证模型](#已验证模型) 中的静态 FP32 契约。
- Batch 为 `1`，必须是 prediction + prototype 双输出。
- 内置 COCO 80 类标签。
- 仓库不分发、下载或校验 ONNX。
- 性能为本地验证数据，不代表跨平台保证。

## 故障排查

- **没有模型：** 使用 `--model /path/to/model.onnx`。
- **找不到程序：** 先运行 `scripts/build.sh`。
- **模型被拒绝：** 检查输入、双输出、opset 和导出配置。
- **缓存被拒绝：** 模型/Runtime 变化或损坏后自动重建属于正常行为。
- **缓存无法保存：** 检查模型目录可写权限。
- **Server 失败：** 查看 `<build-dir>/logs/rpp_server.log`。

## 已验证模型

只列出 Runtime PASS 模型；ONNX 是本地验证输入，不属于 Git 或发布包。

| 模型规格 | 输入 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 3.882 ms | 4.935 ms | 9.306 ms | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 3.916 ms | 9.216 ms | 10.071 ms | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 4.079 ms | 20.554 ms | 11.554 ms | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 3.997 ms | 25.465 ms | 11.415 ms | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 3.985 ms | 56.183 ms | 7.952 ms | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 5.469 ms | 12.372 ms | 24.546 ms | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 5.411 ms | 20.985 ms | 19.068 ms | 45.463 ms | 21.996 | PASS |
| YOLO11n-seg | 1024×1280 | 6.537 ms | 18.127 ms | 21.722 ms | 46.386 ms | 21.558 | PASS |

数据为一次 warmup 后 10 次 RGB 平均值，不包含初始化和可视化。对外承诺前需复核硬件和完整软件环境。

## 附录：从 PT 模型导出 ONNX

### 导出要求

| 配置 | 要求 |
|---|---|
| 任务 | YOLO11 实例分割 |
| 输入 | FP32 NCHW `[1,3,H,W]`，batch `1`，静态尺寸 |
| Prediction | FP32 `[1,4+类别数+mask通道数,N]` |
| Prototype | FP32 `[1,mask通道数,mask_h,mask_w]` |
| COCO | Prediction 116 通道，prototype 32 通道 |
| 内嵌 NMS | 关闭，由应用执行 RPP RT NMS |
| Simplify | 开启 |

### 安装 Ultralytics 并导出

```shell
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python3 -m pip install -e .
python3 -m pip install onnx onnxruntime onnxslim
yolo export model=yolo11s-seg.pt task=segment format=onnx imgsz=640 \
  batch=1 opset=17 dynamic=False simplify=True nms=False
```

矩形模型的 `imgsz` 顺序为高、宽。

### 验证导出契约

使用英文 [Verify the Exported Contract](#verify-the-exported-contract) 中的 ONNX checker。640×640 COCO 模型应为：

```text
input:   [1, 3, 640, 640]
output0: [1, 116, 8400]
output1: [1, 32, 160, 160]
```
