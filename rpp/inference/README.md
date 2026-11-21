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
- Expose a single DFL-logit contract and adapt attention/prototype operations in memory.
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

All release profiles expose contiguous `[4,16,N]` DFL logits. The same RPP kernel
performs max-subtracted FP32 normalization, expectation and coordinate decoding.
`N` is computed from the model input and strides 8/16/32. No profile selects a
second DFL algorithm. Validation captures the logits and public outputs and checks
finiteness and repeatability against ONNX Runtime.

All attention blocks use the same head/query decomposition: the adapter identifies
MatMul → Softmax → transpose → MatMul by graph connectivity, reads the head count,
token length and channel widths, aligns key length to 32 and query length to 96, applies an
explicit key exclusion mask, and crops to the original length. Prototype upsampling
uses the same 1×1 projection and phase shuffle for every model. Source ONNX files
remain unchanged; tensor/layer names do not select an implementation.

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

## Shared Execution Contract

```text
Original ONNX → graph contract / common adaptation → fresh BF16 OpenRT engine
             → FP32 RPP DFL / box decode → prediction + prototype
```

The application then uses one RT NMS graph → selected coefficient gather → sparse
prototype probabilities → composed ROI interpolation. Input preparation retains
one SDK compatibility policy: 8400-anchor heads bind the prediction directly;
18900-anchor heads use stable compaction. Both use the same original-anchor map
and result parser. Forcing compaction for l640 reproduced a missing weak object
on an idle device, so this policy remains until SDK candidate selection is corrected.
The shared interpolation merges repeated taps into three taps per axis. Input
format handling and error checks remain separate from model computation.
The release gate checks support at initialization; it does not select an inference
or mask implementation. A new export needs numerical qualification of the NMS
input policy as well as its model contract.

## Reuse Notes

Customer integration follows the same API sequence: `init()` → write the input binding
through preprocessing → `execute()` → consume prediction/prototype in postprocessing.
Read dimensions from the engine; do not introduce n/s/m/l/x or 640/960 dispatch.
The current export contract fixes batch 1, FP32 NCHW, COCO80 classes, 32 mask channels
and 16 DFL bins. Changes to this contract require a coordinated adaptation and accuracy audit.
Keep labels, input decoding and visualization outside this module. Retain the supplied
precision module and build helpers: their guarded SDK compatibility is part of this demo.

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
- 在内存中建立统一 DFL logits 契约，适配 attention/prototype 操作。
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

所有发布规格均输出连续的 `[4,16,N]` DFL logits，由同一 RPP kernel 完成减最大值的
FP32 归一化、期望值和坐标解码。`N` 由模型输入尺寸及 stride 8/16/32 推导，不按规格
选择第二种 DFL 算法。验证采集 logits 与公开输出，检查有限性、重复一致性及 ONNX Runtime 精度。

所有 attention 块均使用同一个 head/query 分块实现：按 MatMul → Softmax → 转置 → MatMul
的图连接识别，从图读取 head 数、token 长度和通道宽度；key 长度按 32 对齐，query 长度按 96 对齐，
显式屏蔽填充 key，再裁剪回原长度。所有模型的 prototype 上采样均使用相同的 1×1 投影
和 phase shuffle。原始 ONNX 不变，不根据 tensor/layer 名称选择实现。

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

## 统一执行契约

```text
原始 ONNX → 模型契约 / 公共适配 → 新建 BF16 OpenRT engine
          → RPP FP32 DFL / 框解码 → prediction + prototype
```

应用后续共用一个 RT NMS 图 → 选中系数收集 → 稀疏 prototype 概率 → ROI 插值。
输入准备保留一处 SDK 兼容策略：8400-anchor head 直接绑定 prediction，18900-anchor head
先稳定压缩；两者使用相同的原 anchor 映射及结果解析。强制为 l640 压缩时，在设备空闲
状态复现了弱目标漏检，因此在 SDK 候选选择修正之前保留此策略。
公共插值将重复采样合并为每轴三个 tap。输入格式处理和错误检查独立于模型计算。
发布边界仅在初始化时检查支持策略，不用于选择推理或 mask 实现。新导出模型除检查
张量契约外，还需对 NMS 输入策略进行数值验证。

## 复用说明

客户接入遵循相同 API 顺序：`init()` → 前处理写输入 binding → `execute()` → 后处理读取
prediction/prototype。尺寸从 engine 读取，不增加 n/s/m/l/x 或 640/960 分发。
当前导出契约固定 batch 1、FP32 NCHW、COCO80、32 个 mask 通道和 16 个 DFL bin；
契约变化需要统一调整适配并重新进行精度验证。标签、图片解码和可视化位于模块之外。
移植时需保留精度模块及构建辅助代码，其 SDK 兼容检查是本 demo 的组成部分。
