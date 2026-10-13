<a id="english"></a>

# RGB Image YOLOv11-SEG Demo

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This demo runs one RGB/BGR image through the reusable YOLOv11-SEG RPP pipeline per measured loop. It is a simple entry point for validating model loading, host-to-device input transfer, RPP preprocessing, OpenRT inference, YOLOv11-SEG postprocessing, device-to-host result transfer, and rendered detection output.

## What This Demo Covers

- Load a YOLOv11-SEG-compatible ONNX model.
- Read an image with OpenCV.
- Run one explicit warmup pass.
- Run measured inference loops and print stage-level latency.
- Draw detection boxes and save the output image.
- Automatically save an `rpp_perf` trace when the project is built with `YOLO_ENABLE_RPP_PERF=ON`.

## Build

Build from the repository root with the standard script:

```shell
scripts/build.sh
```

The executable is generated at:

```text
build/bin/yolov11_seg_rgb_image_demo
```

## Run

Run all demos with the default configuration:

```shell
scripts/run.sh
```

Run only the RGB demo from the repository root:

```shell
scripts/run.sh --demo rgb
```

Run a measured benchmark for the RGB demo:

```shell
scripts/benchmark.sh --demo rgb --loop 20
```

The script uses the default model at `weights/yolo11s-seg.onnx`, the copied input image at `build/assets/test.jpg`, and the output path `build/output/output_rgb_seg.jpg`.

To use another image through the script:

```shell
scripts/run.sh --demo rgb --input /path/to/image.jpg
```

After building, the binary can also be executed directly:

```shell
build/bin/yolov11_seg_rgb_image_demo \
  -o weights/yolo11s-seg.onnx \
  -i build/assets/test.jpg \
  --output build/output/output_rgb_seg.jpg \
  -l 1
```

## Arguments

```text
-o, --onnx      ONNX model path. Default: ../../weights/yolo11s-seg.onnx
-i, --image     RGB/BGR image path. Default: ../assets/test.jpg
--output        Rendered output image path. Default: ../output/output_rgb_seg.jpg
-l, --loop      Measured loop count after one warmup pass. Default: 1
-v, --verbose   Enable verbose runtime logging.
```

## Execution Flow

```text
Read CLI and paths
        |
        v
Load RGB/BGR image with OpenCV
        |
        v
Create perf trace session in perf-enabled builds
        |
        v
Initialize YoloV11SegPipeline and RppInferEngine
        |
        v
Warmup: run full pipeline once
        |
        v
Printed stages: H2D, preprocess, inference, postprocess, D2H
        |
        v
Draw boxes and save rendered image
```

## Output

The demo prints model and input paths first, then a timing summary:

```text
Model path: ../../weights/yolo11s-seg.onnx
Inference image path: ../assets/test.jpg

Summary:
Loop count: <loop>
Loop time: average per measured loop after warmup
Input H2D: <ms> ms, <bytes> bytes, <KiB> KiB
Preprocess compute: <ms> ms (excludes Input H2D)
Inference: <ms> ms
Postprocess compute: <ms> ms (excludes Output D2H)
Output D2H: <ms> ms, <bytes> bytes, <KiB> KiB
All time end to end: <ms> ms (input H2D+preprocess compute+inference+postprocess compute+output D2H)
FPS: <fps>
Output path: ../output/output_rgb_seg.jpg
```

`Input H2D` and `Output D2H` are printed separately. `Preprocess compute` excludes input H2D and `Postprocess compute` excludes output D2H. `All time end to end` sums all five mutually exclusive stages and is used for FPS.

The rendered image is saved to `build/output/output_rgb_seg.jpg` by default.

## Related Documents

- [Project API documentation](../../docs/api/README.md): reusable API index.
- [RPP acceleration modules](../../rpp/README.md): RPP execution layer guide.

## Reuse Notes

Use this demo when your application already receives compressed or decoded image files on the host. For new applications, copy the CLI pattern and call `YoloV11SegPipeline::runRgb()` with your own `cv::Mat`.

---

<a id="chinese"></a>

# RGB 图片 YOLOv11-SEG Demo

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本 demo 在每次测量 loop 中，将一张 RGB/BGR 图片送入可复用的 YOLOv11-SEG RPP pipeline。它是一个简单的执行入口，可用于验证模型加载、输入 H2D、RPP 前处理、OpenRT 推理、YOLOv11-SEG 后处理、结果 D2H 以及检测框绘制输出。

## 这个 Demo 包含什么

- 加载 YOLOv11-SEG 兼容 ONNX 模型。
- 使用 OpenCV 读取图片。
- 显式执行一次完整 warmup。
- 执行指定次数的测量 loop，并打印阶段级耗时。
- 绘制检测框并保存输出图片。
- 在开启 `YOLO_ENABLE_RPP_PERF=ON` 构建时，自动保存 `rpp_perf` trace。

## 构建

从仓库根目录使用标准脚本构建：

```shell
scripts/build.sh
```

可执行文件生成路径：

```text
build/bin/yolov11_seg_rgb_image_demo
```

## 运行

使用默认配置运行全部 demo：

```shell
scripts/run.sh
```

从仓库根目录只运行 RGB demo：

```shell
scripts/run.sh --demo rgb
```

执行 RGB demo benchmark：

```shell
scripts/benchmark.sh --demo rgb --loop 20
```

脚本默认使用 `weights/yolo11s-seg.onnx` 模型、`build/assets/test.jpg` 输入图片，并将结果保存到 `build/output/output_rgb_seg.jpg`。

通过脚本使用其他图片：

```shell
scripts/run.sh --demo rgb --input /path/to/image.jpg
```

构建完成后，也可以直接执行二进制：

```shell
build/bin/yolov11_seg_rgb_image_demo \
  -o weights/yolo11s-seg.onnx \
  -i build/assets/test.jpg \
  --output build/output/output_rgb_seg.jpg \
  -l 1
```

## 参数

```text
-o, --onnx      ONNX 模型路径。默认：../../weights/yolo11s-seg.onnx
-i, --image     RGB/BGR 图片路径。默认：../assets/test.jpg
--output        绘制检测框后的输出图片路径。默认：../output/output_rgb_seg.jpg
-l, --loop      warmup 之后的测量 loop 次数。默认：1
-v, --verbose   开启更详细的运行时日志。
```

## 执行流程

```text
读取命令行参数和路径
        |
        v
使用 OpenCV 加载 RGB/BGR 图片
        |
        v
perf 构建中创建 trace 会话
        |
        v
初始化 YoloV11SegPipeline 和 RppInferEngine
        |
        v
Warmup：完整执行一次 pipeline
        |
        v
打印阶段：H2D、preprocess、inference、postprocess、D2H
        |
        v
绘制检测框并保存结果图片
```

## 输出

demo 会先打印模型和输入路径，然后打印计时摘要：

```text
Model path: ../../weights/yolo11s-seg.onnx
Inference image path: ../assets/test.jpg

Summary:
Loop count: <loop>
Loop time: average per measured loop after warmup
Input H2D: <ms> ms, <bytes> bytes, <KiB> KiB
Preprocess compute: <ms> ms（不含 Input H2D）
Inference: <ms> ms
Postprocess compute: <ms> ms（不含 Output D2H）
Output D2H: <ms> ms, <bytes> bytes, <KiB> KiB
All time end to end: <ms> ms (input H2D+preprocess compute+inference+postprocess compute+output D2H)
FPS: <fps>
Output path: ../output/output_rgb_seg.jpg
```

`Preprocess compute` 不含输入 H2D，`Postprocess compute` 不含输出 D2H。`All time end to end` 为五个互斥阶段之和，并用于计算 FPS。

绘制后的图片默认保存到 `build/output/output_rgb_seg.jpg`。

## 相关文档

- [项目 API 文档](../../docs/api/README.md)：可复用 API 索引。
- [RPP 加速模块](../../rpp/README.md)：RPP 执行层说明。

## 复用说明

当应用已经在 host 侧收到压缩图片或解码后的图片时，可使用本 demo。新应用可以复用 CLI 组织方式，并用自己的 `cv::Mat` 调用 `YoloV11SegPipeline::runRgb()`。
