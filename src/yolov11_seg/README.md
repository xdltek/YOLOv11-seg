<a id="english"></a>

# YOLOv11-SEG Application Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module contains the reusable YOLOv11-SEG application layer.

## Responsibilities

- Define detection, letterbox, configuration, and timing data structures.
- Provide the COCO 80-class label table.
- Compose preprocessing, inference, and postprocessing through `YoloV11SegPipeline`.
- Draw segmentations and save rendered images.
- Keep YOLOv11-SEG output assumptions outside the generic OpenRT inference wrapper.

## Main API

- `YoloV11SegPipeline::init()`
  Initializes OpenRT inference and YOLOv11-SEG postprocessing with validated runtime configuration.
- `YoloV11SegPipeline::runRgb()`
  Runs one RGB/BGR image through the full pipeline.
- `YoloV11SegPipeline::runI420()`
  Runs one packed I420 frame through the full pipeline.
- `YoloV11SegPipeline::runtimeConfig()`
  Returns the validated runtime configuration derived from defaults and model metadata.
- `draw_segmentations()`
  Draws detection boxes on an OpenCV image.

## Default Configuration

`YoloV11SegConfig` defaults are used by the demos unless the source code is changed:

| Field | Default |
|---|---|
| `input_width` | `640` |
| `input_height` | `640` |
| `score_threshold` | `0.25` |
| `nms_threshold` | `0.45` |
| `agnostic` | `true` |
| `class_count` | `80` |

The actual input size and output dimensions are updated from `RppInferEngine` after model initialization.

## Input And Output Contract

- RGB path accepts an OpenCV BGR/RGB host image. Non-contiguous images are cloned before preprocessing.
- YUV path accepts one packed I420 frame in `Y`, `U`, `V` order. Width and height must be positive even values.
- Segmentations are returned as `std::vector<Segmentation>` with source-image coordinates restored from `LetterboxInfo`.
- Class names are provided by the caller; demos use `coco80_class_labels()`.

## Timing Contract

`YoloV11SegStageTimes` reports H2D and D2H separately. `end_to_end_ms` is the sum of the printed `Preprocess`, `Inference`, and `Postprocess` stages and is used for FPS.

## Call Flow

```text
Demo CLI
  -> YoloV11SegPipeline::init()
  -> YoloV11SegPipeline::runRgb() or runI420()
  -> RppYoloPreprocessor
  -> RppInferEngine
  -> YoloV11SegPostprocessor
  -> draw_segmentations() / save_segmentation_image()
```

## Module Boundary

This layer may know about YOLOv11-SEG output assumptions, COCO labels, visualization text, and pipeline timing. It should not own low-level OpenRT binding allocation, RPP preprocessing kernels, or RT_NMS engine construction details; those stay under `rpp/`.

---

<a id="chinese"></a>

# YOLOv11-SEG 应用模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块包含可复用的 YOLOv11-SEG 应用层。

## 模块职责

- 定义检测结果、letterbox、配置和计时数据结构。
- 提供 COCO 80 类 label 表。
- 通过 `YoloV11SegPipeline` 组合前处理、推理和后处理。
- 绘制检测框并保存结果图片。
- 将 YOLOv11-SEG 输出假设保留在通用 OpenRT 推理封装之外。

## 主要 API

- `YoloV11SegPipeline::init()`
  使用校验后的运行配置初始化 OpenRT 推理和 YOLOv11-SEG 后处理。
- `YoloV11SegPipeline::runRgb()`
  执行一张 RGB/BGR 图片的完整 pipeline。
- `YoloV11SegPipeline::runI420()`
  执行一帧 packed I420 数据的完整 pipeline。
- `YoloV11SegPipeline::runtimeConfig()`
  返回由默认值和模型元数据共同确定的运行配置。
- `draw_segmentations()`
  在 OpenCV 图片上绘制检测框。

## 默认配置

demo 默认使用 `YoloV11SegConfig` 中的配置，除非源码中显式修改：

| 字段 | 默认值 |
|---|---|
| `input_width` | `640` |
| `input_height` | `640` |
| `score_threshold` | `0.25` |
| `nms_threshold` | `0.45` |
| `agnostic` | `true` |
| `class_count` | `80` |

模型初始化后，实际输入尺寸和输出维度会由 `RppInferEngine` 更新。

## 输入输出约束

- RGB 路径接收 OpenCV BGR/RGB host image；非连续图像会在前处理前 clone。
- YUV 路径接收标准 I420 packed buffer，顺序为 `Y`、`U`、`V`，宽高必须为正偶数。
- 输出为 `std::vector<Segmentation>`，实例分割结果坐标已根据 `LetterboxInfo` 恢复到原图坐标。
- 类别名称由调用方传入，demo 默认使用 `coco80_class_labels()`。

## 计时约束

`YoloV11SegStageTimes` 单独上报 H2D 和 D2H。`end_to_end_ms` 为打印的 `Preprocess`、`Inference` 和 `Postprocess` 三个阶段之和，并用于计算 FPS。

## 调用流程

```text
Demo CLI
  -> YoloV11SegPipeline::init()
  -> YoloV11SegPipeline::runRgb() 或 runI420()
  -> RppYoloPreprocessor
  -> RppInferEngine
  -> YoloV11SegPostprocessor
  -> draw_segmentations() / save_segmentation_image()
```

## 模块边界

本层可以包含 YOLOv11-SEG 输出假设、COCO 标签、可视化文本和 pipeline 计时逻辑。不应持有底层 OpenRT binding 分配、RPP 前处理 kernel 或 RT_NMS engine 构建细节；这些职责保留在 `rpp/` 模块中。
