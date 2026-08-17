# YOLOv11-SEG RPP Demo

This repository contains a C++17/OpenRT on RPP demo for YOLOv11-SEG instance segmentation.

The project is organized from the existing `/home/azurengine/github/YOLOv11` detection demo and updated for a two-output segmentation model:

- `output0`: detection head, expected shape `[1, 116, 8400]`
- `output1`: prototype masks, expected shape `[1, 32, 160, 160]`
- default model: `weights/yolo11s-seg.onnx`

## Directory

| Path | Description |
| --- | --- |
| `CMakeLists.txt` | Builds shared libraries and RGB/YUV demo executables |
| `demos/rgb_image` | RGB/BGR image demo entry |
| `demos/yuv_frame` | I420 YUV single-frame demo entry |
| `rpp/inference` | OpenRT engine wrapper with multi-output binding support |
| `rpp/preprocess` | RPP preprocessing: upload, letterbox, color conversion, resize, normalize |
| `rpp/postprocess` | YOLOv11-SEG RPP output split, CPU det decode/NMS, RPP prototype combine, CPU mask finalize |
| `src/yolov11_seg` | Pipeline, labels, result types, visualization |
| `scripts` | Build, run, smoke test, benchmark helpers |
| `internal_docs` | Design documents |
| `weights` | Local ONNX model directory, not intended for version tracking |

## Build

```bash
scripts/build.sh
```

If the model is already prepared or you only want to compile:

```bash
scripts/build.sh --skip-model
```

Build outputs are written to:

```text
build/bin/
build/lib/
build/assets/
build/output/
build/logs/
```

## Run

RGB/BGR image:

```bash
scripts/run.sh --demo rgb --loop 1
```

I420 YUV frame:

```bash
scripts/run.sh --demo yuv --loop 1 --yuv-width 1280 --yuv-height 720
```

Both demos:

```bash
scripts/run.sh --demo all --loop 1
```

Direct binaries:

```bash
build/bin/yolov11_seg_rgb_image_demo \
  --onnx weights/yolo11s-seg.onnx \
  --image assets/test.jpg \
  --output build/output/output_rgb_seg.jpg

build/bin/yolov11_seg_yuv_frame_demo \
  --onnx weights/yolo11s-seg.onnx \
  --yuv assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  --output build/output/output_yuv_seg.jpg
```

## Output

The pipeline returns `std::vector<Segmentation>`.

Each result contains:

- `class_id`
- `confidence`
- `box` in original image coordinates
- dense binary `mask` with original image width and height

Visualization blends instance masks over the source image and draws the retained boxes and class labels.

## Postprocess

The current complete path is:

1. copy model `output0` from DDR to SRAM
2. run an RPP CUDA kernel to split `output0[116, anchors]` into `det[84, anchors]` and `seg_coeff[32, anchors]`
3. copy split `det` and `seg_coeff` heads back to host staging buffers for the current CPU det/NMS path
4. decode det `(cx, cy, w, h)` and class scores, then run greedy NMS
5. upload selected anchor indices and copy prototype output to SRAM
6. run an RPP CUDA kernel to gather selected `seg_coeff` rows from the full coefficient head
7. run an RPP CUDA kernel to combine `coefficients x prototype` into selected low-resolution raw masks
8. run an RPP CUDA kernel to threshold selected raw masks into low-resolution binary masks
9. copy low-resolution binary masks back to host
10. resize, box crop, and letterbox reverse mapping on CPU
11. render mask overlay and boxes

The `det[84, anchors]` output is now isolated in a YOLOv11 det-compatible layout. The next TODO is to replace the CPU greedy NMS call with the YOLOv11 det RT_NMS path and return selected anchor indices directly from the device path. Source-size dense mask resize/crop remains the next SEG optimization item.

## Test

```bash
scripts/test.sh
```

This smoke test checks that required model, binaries, and sample assets exist.

Standalone postprocess module test:

```bash
scripts/build_postprocess_test.sh
scripts/test_postprocess.sh
scripts/run_postprocess_test.sh --loop 20 --warmup 1
```

The standalone test does not run preprocessing or inference. By default it generates synthetic YOLOv11-SEG `output0` and `output1` tensors, uploads them to RPP DDR, calls `YoloV11SegPostprocessor::run()`, and prints:

- `output_d2h_ms`
- `output_d2h_bytes`
- `split_ms`
- `decode_nms_ms`
- `mask_decode_ms`
- `profile_total_ms`
- `outer_call_ms`
- `output_segments`

To test real model output dumps:

```bash
scripts/run_postprocess_test.sh \
  --detection-bin path/to/output0_f32.bin \
  --prototype-bin path/to/output1_f32.bin \
  --loop 20
```

## Benchmark

```bash
scripts/benchmark.sh --demo rgb --loop 20
scripts/benchmark.sh --demo yuv --loop 20
```

Benchmark logs are saved under `build/logs/`.

## Current Verification

- `scripts/build.sh --skip-model`: passed
- `scripts/test.sh`: passed
- `scripts/build_postprocess_test.sh`: passed
- `scripts/test_postprocess.sh`: passed
- `cmake --build build -j$(nproc)`: passed with `rpp_postprocess_kernels.cu`
- `scripts/run.sh --demo rgb --loop 1`: blocked by environment, `/usr/local/rpp/bin/rpp_server` reports `no RPP device found`
- `scripts/run_postprocess_test.sh --loop 1 --warmup 0`: blocked by environment, `/usr/local/rpp/bin/rpp_server` reports `no RPP device found`

## Known Limitations

- Runtime execution requires an available RPP device and working `rpp_server`.
- Output split, selected coefficient gather, selected mask prototype combination, and low-resolution mask threshold run on RPP.
- NMS remains CPU-based in this version; the split det head is prepared for the YOLOv11 det RT_NMS path.
- Source-size mask resize/crop remains CPU-based.
- The output layout is validated for the local `yolo11s-seg.onnx` model shape. Other exports may need layout handling updates.
