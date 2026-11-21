<a id="english"></a>

# RPP Inference Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module wraps OpenRT inference for YOLOv11-SEG-compatible ONNX models.

The release profiles are n/s/m/l/x at 640×640 and n/s at 960×960. Input shape
and stem-convolution width are checked from the parsed graph before engine
construction; changing a filename does not change profile eligibility.

## Responsibilities

- Parse the ONNX model.
- Build a fresh OpenRT execution engine from ONNX for every application process.
- Adapt DFL channel normalization and prototype upsampling in memory.
- Correct convolution bias accumulation and BF16 output rounding on RPP.
- Restore FP32 anchor/stride box coordinates from an internal device DFL binding.
- Own model input and output binding buffers.
- Execute synchronous inference.
- Expose model input shape, output shape, and tensor data types to the pipeline.

## Main API

- `RppInferEngine::init()`
  Builds the inference engine and allocates bindings.
- `RppInferEngine::execute()`
  Runs one inference pass using the current input binding.
- `RppInferEngine::getInputDeviceBuffer()`
  Returns the device buffer consumed by preprocessing.
- `RppInferEngine::getOutputDeviceBuffer()`
  Returns the device buffer consumed by postprocessing.

## Input And Output

- Input: YOLOv11-SEG-compatible ONNX model path.
- Input binding: FP32 model input tensor.
- Output bindings: FP32 prediction and prototype tensors.

Engine serialization and deserialization are disabled. Existing local engine
files are ignored; no engine or intermediate ONNX model is written. One in-memory
engine serves the process's warmup and measured frames.

The internal DFL binding is not exposed through `getOutputInfos()`.
After OpenRT execution, a row-based RPP kernel updates only the first four
prediction planes in FP32. It uses device/SRAM movement, with no host tensor copy.
Scratch SRAM is allocated after OpenRT execution and freed before POST or the next
frame, matching the existing PRE/POST workspace lifetime. Inference timing includes
this model-head coordinate correction.

640×640 retains its device DFL-distance decoder. At 960×960, OpenRT exposes
contiguous `[4,16,N]` logits before the large DFL transpose. An RPP kernel performs
max-subtracted FP32 normalization, expectation and coordinate decoding. No NaN/Inf
replacement or CPU numerical fallback is used. Validation mode also captures the
raw internal DFL tensor and checks finiteness before accepting the public outputs.

960-square attention with four heads uses independent head/query tiles,
padded keys with an explicit exclusion mask, and a final crop to the original
900 queries. The n960 two-head profile retains the native parser path. The application builds
these equivalent operations in memory through OpenRT. Source ONNX files remain
unchanged; validated model scope is in the root README.

Convolution corrections are assembled at build time from the installed SDK's
assembly sources. At first inference, the application checks the temporary OpenRT
module's assembly source, code sizes and parameter ABI before loading corrected
convolution sections through `rppModuleLoadData`. Other module sections and OpenRT
scheduling remain intact. The loader hook is scoped to synchronous inference;
the POST engine retains its own module. No installed SDK file is changed.
The enhanced convolution path serves the release-supported 640×640 and 960×960 profiles.

## Main Files

- `rpp_infer_engine.h`
- `rpp_infer_engine.cpp`
- `rpp_segmentation_network.h/.cpp`: in-memory graph adaptation.
- `rpp_box_decode.cu/.cuh`: FP32 RPP head coordinates.
- `rpp_precision_module.h/.cpp`: scoped module preparation and compatibility checks.
- `generate_precision_kernels.py`, `precision_assembler.cpp`: build-time correction generator.

## Reuse Notes

Reuse this module when the model can be loaded through the same OpenRT ONNX flow. Keep model-specific preprocessing, postprocessing, labels, and visualization outside this module.

---

<a id="chinese"></a>

# RPP 推理模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块封装 YOLOv11-SEG 兼容 ONNX 模型在 OpenRT 上的推理流程。

发布规格为 640×640 的 n/s/m/l/x 与 960×960 的 n/s。构建 engine 前按模型图检查
输入尺寸及首层卷积宽度，不依赖模型文件名判断规格。

## 模块职责

- 解析 ONNX 模型。
- 每个应用进程都从 ONNX 重新构建 OpenRT 执行 engine。
- 在内存中适配 DFL 通道归一化与 prototype 上采样。
- 在 RPP 上修正卷积 bias 累加及 BF16 输出舍入。
- 从内部设备 DFL binding 恢复 FP32 anchor/stride 框坐标。
- 管理模型输入和输出 binding buffer。
- 执行同步推理。
- 向 pipeline 暴露模型输入尺寸、输出尺寸和 tensor 数据类型。

## 主要 API

- `RppInferEngine::init()`
  构建推理 engine 并分配 binding。
- `RppInferEngine::execute()`
  使用当前输入 binding 执行一次推理。
- `RppInferEngine::getInputDeviceBuffer()`
  返回前处理写入的 device buffer。
- `RppInferEngine::getOutputDeviceBuffer()`
  返回后处理读取的 device buffer。

## 输入输出

- 输入：YOLOv11-SEG 兼容 ONNX 模型路径。
- 输入 binding：FP32 模型输入 tensor。
- 输出 binding：FP32 prediction 与 prototype tensor。

不执行 engine 序列化和反序列化，已有本地 engine 文件会被忽略。不写出 engine 或中间
ONNX 模型；同一内存 engine 服务于进程中的预热帧和测量帧。

内部 DFL binding 不通过 `getOutputInfos()` 暴露。OpenRT 执行后，按行启动的 RPP
kernel 以 FP32 更新 prediction 的前四个平面，只做设备/SRAM 搬运，不复制 tensor 到主机。
临时 SRAM 在 OpenRT 执行后分配，并在 POST 或下一帧之前释放，与现有 PRE/POST 的
workspace 生命周期一致。推理阶段计时包含这部分模型 head 坐标修正。

640×640 保留设备端 DFL 距离解码。960×960 直接输出大型 DFL 转置之前连续存储的
`[4,16,N]` logits，再由 RPP kernel 执行减最大值的 FP32 归一化、期望值和框坐标解码。
没有用替换 NaN/Inf 或 CPU 数值回退掩盖异常；校验模式还会采集内部 DFL 原始张量并检查有限性。

960 规格的四 head attention 使用独立 head/query 分块，补齐的 key 带显式排除 mask，
最后裁剪回原有的 900 个 query；n960 的双 head 使用原生解析路径。应用通过 OpenRT 在内存中构建等价操作，原始 ONNX
保持不变；实际通过验证的型号范围见根 README。

卷积修正使用已安装 SDK 的汇编源在构建时生成。首次推理时，应用核对 OpenRT 临时
模块的汇编源、代码长度及参数 ABI，再通过 `rppModuleLoadData` 加载已修正的卷积
section。其他 section 与 OpenRT 调度保持原有行为；加载 hook 仅作用于同步推理，
POST engine 继续使用自身模块。不会修改已安装 SDK 文件。
增强卷积路径用于发布支持的 640×640 和 960×960 规格。

## 主要文件

- `rpp_infer_engine.h`
- `rpp_infer_engine.cpp`
- `rpp_segmentation_network.h/.cpp`：内存内图适配。
- `rpp_box_decode.cu/.cuh`：FP32 RPP head 坐标。
- `rpp_precision_module.h/.cpp`：限定作用范围的模块准备及兼容检查。
- `generate_precision_kernels.py`、`precision_assembler.cpp`：构建时生成卷积修正。

## 复用说明

当模型可以通过相同的 OpenRT ONNX 流程加载时，可复用本模块。模型相关的前处理、后处理、标签和可视化逻辑应放在本模块之外。
