# Performance records and baseline / 性能记录与基准

These are the original 1.6.11 RGB measurements, retained unchanged as the
optimization baseline. They are not a numerical qualification of newly supplied
ONNX files or the accuracy corrections made on 2026-11-20. Historical PASS means
the original runtime/smoke validation completed; current accuracy support is
listed in the root README. Only release-supported profiles are shown.

| Model profile | Input | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11n-seg | 640×640 | 3.882 ms | 4.935 ms | 9.306 ms | 18.122 ms | 55.182 | PASS |
| YOLO11s-seg | 640×640 | 3.916 ms | 9.216 ms | 10.071 ms | 23.203 ms | 43.098 | PASS |
| YOLO11m-seg | 640×640 | 4.079 ms | 20.554 ms | 11.554 ms | 36.187 ms | 27.634 | PASS |
| YOLO11l-seg | 640×640 | 3.997 ms | 25.465 ms | 11.415 ms | 40.876 ms | 24.464 | PASS |
| YOLO11x-seg | 640×640 | 3.985 ms | 56.183 ms | 7.952 ms | 68.120 ms | 14.680 | PASS |
| YOLO11n-seg | 960×960 | 5.469 ms | 12.372 ms | 24.546 ms | 42.387 ms | 23.592 | PASS |
| YOLO11s-seg | 960×960 | 5.411 ms | 20.985 ms | 19.068 ms | 45.463 ms | 21.996 | PASS |

One warmup followed by ten measured frames; preprocessing includes H2D and
postprocessing includes D2H. Engine construction and visualization are excluded.
The measurements above remain the original reference values.

以上为原 1.6.11 的 RGB 实测值，原样保留为性能优化基准。历史 PASS 表示原运行检查完成，
不等于新提供 ONNX 文件或本轮精度修正已经获得数值精度认证。当前精度支持范围见根 README。
口径为一次预热后测量十帧，PRE 包含 H2D、POST 包含 D2H，不包含 engine 构建和可视化。
上表继续保留原始参考数值。

## 2026-11-21 reboot measurements / 重启后实测

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
commands: [performance record](../README.md).

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
原基准、逐阶段口径及复现命令见[性能记录](../README.md)。

## Reproduction and evidence / 复现与证据

```shell
unset YOLO_VALIDATION_DIR
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 10
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 10
```

RGB input is `assets/test.jpg`; I420 input is `assets/test_0.yuv`, 1280×720.
Only one application uses the accelerator at a time. The server uses
`glb_common_event=512` (3456 graph events), configured locally through `RPP_CFG_FILE`.
Measured execution uses BF16 OpenRT inference with the retained precision corrections,
FP32 device DFL/box decoding, RT NMS, mask reconstruction and normal result transfers.
The inference column includes coordinate correction. This is repeated-frame application
latency, excluding model construction and file/render I/O; it is not camera-to-display latency.

Local evidence: `build/reboot_release_20261121/performance_final.json`,
`benchmark_final_metadata.json`, and per-case logs/status/image files under
`performance_final/`. Model, executable and shared-library SHA256 hashes are retained.
The original baseline has no model hashes, so comparisons are against the user-provided
performance target rather than a controlled same-model revision benchmark.

输入为仓库样例 RGB 图片及 1280×720 I420 帧；所有设备任务串行执行。服务通过本地配置将
公共事件设为 512，为图事件保留 3456 个。推理使用 BF16 OpenRT 及保留的精度修正，
包含设备 FP32 DFL/框解码、RT NMS、mask 重建和正常结果搬运。推理列包含坐标修正。
这是重复帧应用延迟，排除建模和文件/绘制耗时，不代表摄像头到显示延迟。
本地证据保留模型、二进制、共享库 hash 及每项日志。原基准没有模型 hash，故比较对象为
用户提供的性能目标，而非严格控制同权重的版本对比。
