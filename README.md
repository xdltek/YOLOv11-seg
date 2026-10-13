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
| `rpp/postprocess` | RPP candidate decode and sparse mask kernels, CPU sort/NMS, packed uint8 result return |
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
2. decode box/class data and apply score threshold on RPP into fixed seven-field candidate slots
3. copy candidate slots directly from SRAM to pinned Host memory
4. compact valid candidates and run class-aware or agnostic greedy NMS on CPU
5. upload selected anchor indices and packed ROI metadata directly to SRAM
6. gather the selected 32-channel mask coefficients on RPP
7. combine only the prototype regions required by the selected source-image ROIs
8. bilinearly resize, threshold, and pack all ROI masks as uint8 on RPP
9. copy the packed masks directly from SRAM to pinned Host memory and build final segmentations
10. render mask overlays, boxes, and labels

This release intentionally completes the CPU_NMS path. RT_NMS and full PRE/Infer/POST graph integration are deferred to the next version; they are not part of the current default execution path.

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

Enable code-side `rpp_perf` scopes for the standalone postprocess test:

```bash
scripts/build_postprocess_test.sh --perf
scripts/run_postprocess_test.sh --loop 20 --warmup 1 --accuracy-check
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

Postprocess precision validation:

```bash
scripts/run_postprocess_test.sh \
  --loop 20 \
  --warmup 1 \
  --accuracy-check
```

In `--accuracy-check` mode, the test builds a CPU OpenCV reference from the same `output0/output1` tensors, then compares the RPP postprocess result against it. It prints per-segment `box_iou`, `max_box_diff_px`, `mask_iou`, and `mask_diff_ratio`, followed by a PASS/FAIL summary. Default thresholds are:

- `--min-box-iou 0.98`
- `--max-box-diff 1.0`
- `--min-mask-iou 0.90`
- `--max-mask-diff-ratio 0.15`

To test real model output dumps:

```bash
scripts/run_postprocess_test.sh \
  --detection-bin path/to/output0_f32.bin \
  --prototype-bin path/to/output1_f32.bin \
  --loop 20 \
  --accuracy-check
```

## Benchmark

```bash
scripts/benchmark.sh --demo rgb --loop 20
scripts/benchmark.sh --demo yuv --loop 20
```

Benchmark logs are saved under `build/logs/`.

## Current Verification

- `cmake -S . -B build && cmake --build build -j2`: passed
- `scripts/test.sh` and `scripts/test_postprocess.sh`: passed
- `scripts/run_postprocess_test.sh --warmup 1 --loop 3 --accuracy-check --min-mask-iou 0.995`: passed; `min_box_iou=1.000000`, `min_mask_iou=0.999110`, `max_mask_diff_ratio=0.000105`
- `scripts/run.sh --demo all --loop 1`: RGB and I420 demos passed on RPP
- Latest single-loop RGB result: End-to-end `20.158 ms`, `49.607 FPS`
- Latest single-loop I420 result: End-to-end `17.281 ms`, `57.866 FPS`

## Known Limitations

- Runtime execution requires an available RPP device and working `rpp_server`.
- Candidate decode, selected coefficient gather, Sparse Prototype ROI combine, and packed uint8 ROI resize/threshold run on RPP.
- NMS remains CPU-based in this release; RT_NMS and selected-index device residency are deferred to the next version.
- Final masks are returned as bbox-local packed uint8 buffers and exposed through the existing `Segmentation` result structure.
- The output layout is validated for the local `yolo11s-seg.onnx` model shape. Other exports may need layout handling updates.
