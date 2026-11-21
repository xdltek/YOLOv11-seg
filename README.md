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

Use a model prepared and qualified by the [accuracy workflow](docs/accuracy-validation.md).

Successful execution prints stage progress, end-to-end and OpenRT FPS, pipeline timing, and saves rendered images under `build/output/`.

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Cleans generated outputs, configures CMake, and builds without preparing a model. |
| `scripts/run.sh` | Runs both demos, or one selected workload with configurable paths and loops. |
| `scripts/benchmark.sh` | Runs repeated measurements and archives the console log. |

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
scripts/run.sh --model /path/to/model.onnx --demo yuv
scripts/benchmark.sh --model /path/to/model.onnx --loop 20
```

When starting the service, the script creates `build/config/rpp_syscfg.ini` with a graph-event budget
for the supported models, without editing the installed SDK. An explicit `RPP_CFG_FILE` overrides this
configuration; an already running service is reused. For direct binary use, start the service with
this configuration as well.

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
│   ├── inference/            # multi-output OpenRT engine construction
│   ├── preprocess/           # upload, letterbox, color, normalize
│   └── postprocess/          # RT NMS and sparse mask reconstruction
├── scripts/                  # build, run, benchmark
├── src/
│   ├── perf/                 # optional rpp_perf integration
│   ├── utils/                # logging, parser and path helpers
│   └── yolov11_seg/          # pipeline, reporter, labels, visualization
├── tests/                    # model accuracy and postprocess validation
└── weights/                  # local models; ONNX is Git-ignored
```

| Path | Responsibility |
|---|---|
| `CMakeLists.txt` | Root build entry for libraries, applications, tests, and assets. |
| `src/yolov11_seg/` | Application pipeline and customer-facing reporting. |
| [`rpp/`](rpp/README.md) | Hardware-facing inference, preprocessing, and postprocessing. |
| `demos/` | RGB and I420 customer application entries. |
| [`docs/`](docs/README.md) | Public documentation and API navigation. |
| `scripts/` | Reproducible build, run, and benchmark commands. |
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
- [Model accuracy validation](docs/accuracy-validation.md)

## Prerequisites

- Linux development environment
- CMake `>= 3.16.3` and a C++17 compiler
- Required OpenCV components
- OpenRT/RPP installed under `/usr/local/rpp`
- Python 3 and binutils for build-time kernel preparation; SDK assembly sources and `librpp_asm` must be installed.
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
build/tools/    private build-time assembler helper
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

### OpenRT Engine Initialization

Each application process parses the supplied ONNX and builds a fresh BF16 OpenRT
engine in memory. That engine serves all warmup and measured frames in the process.
The application does not serialize engines or read/write local engine caches.
Existing engine files are ignored. Construction is initialization time and is
excluded from measured frame latency.

Standard 640×640 exports are adapted in memory for DFL channel normalization and
prototype upsampling. Convolution bias is accumulated in FP32 before nearest
BF16 output rounding. No intermediate ONNX or modified SDK is required.
The inference stage also includes FP32 box-coordinate correction on RPP.

## Code Execution Workflow

```text
+------------------------+
| Load RGB or I420 input |
+------------------------+
            |
            v
+------------------------+
| Build fresh OpenRT     |
| engine from ONNX       |
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
- `rpp/inference/rpp_infer_engine.h/.cpp`: ONNX engine construction and multi-output inference.
- `rpp/preprocess/rpp_yolo_preprocessor.h/.cpp`: RGB/BGR and I420 preprocessing.
- `rpp/postprocess/rpp_yolov11_seg_postprocessor.h/.cpp`: RT NMS and masks.

## Reusable Components

- `YoloV11SegPipeline`: application API for RGB/BGR and packed I420.
- `RppInferEngine`: fresh engine construction, bindings, and multi-output execution.
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
- ONNX files are user-provided and excluded from Git; numerical validation is an explicit local step.
- Performance is local validation data, not a cross-platform guarantee.

## Troubleshooting

For `Event: Out of events`, restart the service while the device is idle to load the
script-generated configuration. A device power cycle is not required for this configuration change.
The configuration is read when the service starts; setting `RPP_CFG_FILE` on an
application cannot reconfigure an existing server. For manual startup, use:

```shell
RPP_CFG_FILE="$PWD/build/config/rpp_syscfg.ini" /usr/local/rpp/bin/rpp_server
```

- **No model:** use `scripts/run.sh --model /path/to/model.onnx`.
- **Executable missing:** run `scripts/build.sh`.
- **Model rejected:** verify static input, prediction, prototype, opset, and export settings.
- **Engine initialization:** each process builds from ONNX; the model directory does not need to be writable.
- **Server failure:** inspect `<build-dir>/logs/rpp_server.log`.
- **Unexpected masks:** confirm the segmentation class and mask-channel contract.

## Validated Models

The release supports these seven static COCO80 segmentation profiles. ONNX files
are supplied by the user and are excluded from Git and release packages.

| Model profile | Input H×W | Runtime model | Inputs | Status |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11m-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11l-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11x-seg | 640×640 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11n-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |
| YOLO11s-seg | 960×960 | Original ONNX | RGB/BGR, I420 | Supported |

Each process builds a fresh OpenRT engine from the original ONNX. DFL/prototype
adaptations and precision corrections run in memory; no engine cache or prepared
ONNX file is used. The loader checks the graph's input dimensions and stem
convolution width before engine construction, independently of the filename.

Numerical acceptance compares the actual device input against ONNX Runtime:
matching counts/classes, box IoU ≥ 0.90, mask IoU ≥ 0.95 and confidence difference
≤ 0.05, with finite, repeatable prediction/prototype/DFL tensors. Support scope,
retained validation coverage and reproduction commands are documented in the
[accuracy guide](docs/accuracy-validation.md). Sample checks are not dataset mAP certification.

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

## Appendix: Export ONNX From a PT Model

Export is followed by model adaptation and numerical validation; an ONNX checker
PASS alone does not establish inference or segmentation accuracy.

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

Use `imgsz=960` when exporting a supported 960×960 profile.

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
│   ├── inference/            # OpenRT engine 构建与双输出推理
│   ├── preprocess/           # 上传、letterbox、颜色、归一化
│   └── postprocess/          # RT NMS 与稀疏 mask
├── scripts/                  # build、run、benchmark
├── src/
│   ├── perf/                 # 可选 trace
│   ├── utils/                # 通用工具
│   └── yolov11_seg/          # pipeline、打印、标签、可视化
├── tests/                    # 模型精度与后处理验证
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
- 构建时需要 Python 3、binutils，以及 SDK 的汇编源和 `librpp_asm`。
- 目标 RPP Driver、Firmware 和 Runtime

## 如何构建

```shell
scripts/build.sh
```

构建不下载、检查或要求 ONNX。生成物位于 `build/bin`、`build/lib`、`build/tools`、`build/assets`、`build/output`、`build/trace` 和 `build/logs`。

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

### OpenRT Engine 初始化

每个应用进程都解析指定 ONNX，并在内存中重新构建 BF16 OpenRT engine。
同一进程的 warmup 和测量帧共用该 engine。应用不序列化 engine，也不读写本地
engine 缓存，已有 engine 文件会被忽略。构建属于初始化时间，不计入帧耗时。

标准 640×640 导出模型会在内存中完成 DFL 通道归一化及 prototype 上采样适配。
卷积先用 FP32 累加 bias，再采用最近舍入转为 BF16，不要求中间 ONNX，也不修改 SDK。
推理阶段还包含在 RPP 上完成的 FP32 框坐标修正。

## 代码执行流程

```text
加载 RGB/I420
    -> 从 ONNX 重新构建 OpenRT engine
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
- `rpp/inference/rpp_infer_engine.h/.cpp`：从 ONNX 构建 engine 和双输出推理。
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
- ONNX 由用户提供且不进入 Git；数值校验是显式的本地操作。
- 性能为本地验证数据，不代表跨平台保证。

## 故障排查

脚本启动服务时会生成 `build/config/rpp_syscfg.ini`，为支持模型预留图事件资源，不修改 SDK。
显式设置的 `RPP_CFG_FILE` 优先，已运行的服务会直接复用。直接运行二进制时，服务也应使用
同样的配置。若出现 `Event: Out of events`，需在设备空闲时重启服务以加载配置，无需断电。
配置仅在服务启动时读取；在应用进程设置 `RPP_CFG_FILE` 不能改变已有服务。手工启动命令：

```shell
RPP_CFG_FILE="$PWD/build/config/rpp_syscfg.ini" /usr/local/rpp/bin/rpp_server
```

- **没有模型：** 使用 `--model /path/to/model.onnx`。
- **找不到程序：** 先运行 `scripts/build.sh`。
- **模型被拒绝：** 检查输入、双输出、opset 和导出配置。
- **Engine 初始化：** 每个进程都从 ONNX 构建，模型目录不需要可写。
- **Server 失败：** 查看 `<build-dir>/logs/rpp_server.log`。

## 已验证模型

本版本支持以下七种静态 COCO80 实例分割规格。ONNX 由用户提供，不进入 Git 或发布包。

| 模型 | 输入 H×W | 运行模型 | 输入格式 | 状态 |
|---|---:|---|---|---|
| YOLO11n-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11m-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11l-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11x-seg | 640×640 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11n-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |
| YOLO11s-seg | 960×960 | 原始 ONNX | RGB/BGR, I420 | 支持 |

每个进程直接从原始 ONNX 新建 OpenRT engine，在内存中完成 DFL/prototype 适配及
精度修正，不使用 engine cache 或中间 ONNX。加载器在构建前检查图的输入尺寸及
首层卷积宽度，不依赖模型文件名。

数值验收使用设备实际输入与 ONNX Runtime 参考比较：目标数量、类别一致，框 IoU
≥ 0.90、mask IoU ≥ 0.95、置信度差 ≤ 0.05，prediction/prototype/DFL 为有限值且
重复一致。支持范围、已有验证覆盖及复现命令见[精度指南](docs/accuracy-validation.md)。
样例检查不代表数据集 mAP 认证。

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

## 附录：从 PT 模型导出 ONNX

导出后还需进行模型适配及数值校验，ONNX checker PASS 不能替代推理与分割精度验收。

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

导出支持的 960×960 规格时，使用 `imgsz=960`。

### 验证导出契约

使用英文 [Verify the Exported Contract](#verify-the-exported-contract) 中的 ONNX checker。640×640 COCO 模型应为：

```text
input:   [1, 3, 640, 640]
output0: [1, 116, 8400]
output1: [1, 32, 160, 160]
```
