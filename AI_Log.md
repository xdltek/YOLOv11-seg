# AI Development Log

| Date | Stage | Prompt Summary | AI Output Summary | Manual Changes | Final Status | Reviewer | Artifact/Commit |
|---|---|---|---|---|---|---|---|
| 2026-08-14 | Design | 基于 `internal_docs/YOLOv11_Demo_Design_v2.0.md` 生成 YOLOv11-SEG 设计文档，重点修改后处理设计 | 新增 `internal_docs/YOLOv11_SEG_Demo_Design_v2.0.md`，补充 detection/prototype 输出、mask decode、NMS、CPU fallback、坐标恢复设计 | TBD | Draft | TBD | local workspace |
| 2026-08-14 | Implementation | 参考 `/home/azurengine/github/YOLOv11` 和 SEG 设计文档整理完整 demo，模型已拷贝 | 迁移参考工程，新增多输出 inference metadata，改为 YOLOv11-SEG pipeline，新增 CPU SEG postprocess、mask 可视化、脚本默认模型和 README | TODO: human review required for OpenRT/RPP runtime on target device | Build passed, runtime blocked by missing RPP device | TBD | local workspace |
| 2026-08-14 | RPP CUDA Postprocess | 后处理耗时过长，参考前处理实现添加 RPP CUDA 算子 | 新增 `rpp_postprocess_kernels.cu/.cuh`，将 selected mask 的 `coefficients x prototype` 线性组合迁移到 RPP SRAM kernel，CPU 保留 sigmoid/resize/crop fallback/收尾 | TODO: validate speed and mask parity on RPP device | Build passed, runtime blocked by missing RPP device | TBD | local workspace |
| 2026-08-14 | RPP Postprocess Split | 希望后处理都放在 RPP 上，拆成基础算子并在 SRAM 上持续计算 | 尝试新增第二段 RPP finalize kernel 做 resize/crop/threshold，但当前 RPP assembler 对该 kernel 触发断言；已回退到稳定可编译的 RPP prototype combine 基线 | TODO: continue with smaller finalize sub-kernels or RT_NMS + gather path on RPP device | Build passed, full finalize kernel blocked by assembler | TBD | local workspace |
| 2026-08-14 | Standalone Postprocess Test | 基于 YOLOv11-SEG 后处理构建独立测试 demo、测试脚本、构建脚本，并打印完整后处理调用时间 | 新增 `demos/postprocess_test/main.cpp`、`yolov11_seg_postprocess_test` CMake target、`scripts/build_postprocess_test.sh`、`scripts/run_postprocess_test.sh`、`scripts/test_postprocess.sh`；支持 synthetic tensors 和真实 output dump | TODO: run timing on RPP device | Build and artifact test passed; runtime blocked by missing RPP device | TBD | local workspace |
| 2026-08-14 | RPP Output Split | 将 YOLOv11-SEG 模型 `output0` 在 RPP 上拆分为 det + seg，并按 det NMS、seg 板上运算顺序实现 TODO | 新增 `yolov11_seg_split_det_coeff_kernel`，路径变为 output0 DDR->SRAM->RPP split->det/coeff staging；det decode/NMS 读取拆分后的 `[84, anchors]`，seg coefficient gather 读取 `[32, anchors]`；保留 det RT_NMS 和完整 seg finalize TODO | TODO: replace CPU NMS with YOLOv11 det RT_NMS after selected anchor index plumbing; analyze SEG finalize split kernels separately | Build and artifact test passed; runtime blocked by missing RPP device | TBD | local workspace |
| 2026-08-14 | SEG RPP Algorithm Design | 确认 det split 可在 RPP 上运行，分析 SEG 后处理如何按并行要求拆分算法；后续不再由 sandbox 运行 RPP 脚本 | 在设计文档新增 SEG RPP 拆分算法，给出 Level 1 低风险板上化和 Level 2 dense mask 完整板上化方案，明确 kernel 切分、线程映射、SRAM/DDR 数据流、实施顺序和正确性验证 | TODO: implement selected anchor RT_NMS plumbing and SEG split kernels on target board | Design updated; no RPP runtime script executed | TBD | local workspace |
| 2026-08-14 | SEG RPP Level 1 Implementation | 按设计实现 SEG 后处理拆分算法；不运行 RPP runtime 脚本 | 新增 `yolov11_seg_gather_selected_coefficients_kernel` 和 `yolov11_seg_sigmoid_threshold_lowres_kernel`，路径变为 selected anchor H2D/SRAM -> RPP coeff gather -> RPP prototype combine -> RPP low-res threshold -> CPU resize/crop；最初 `uint16_t + exp` 版本触发 RPP assembler 断言，已改为 float binary 和 `raw > logit(threshold)` | TODO: target board runtime parity/performance validation; next step source-space resize/crop RPP fused kernel | Build passed; no RPP runtime script executed | TBD | local workspace |

## Notes

- Local model shape verified with ONNX metadata: input `[1, 3, 640, 640]`, outputs `[1, 116, 8400]` and `[1, 32, 160, 160]`.
- `scripts/build.sh --skip-model` passed.
- `cmake --build build -j$(nproc)` passed after adding RPP postprocess kernel.
- `scripts/test.sh` passed.
- `scripts/build_postprocess_test.sh` passed.
- `scripts/test_postprocess.sh` passed.
- `cmake --build build -j$(nproc)` passed after adding RPP output split kernel.
- `scripts/build_postprocess_test.sh` passed after adding `split_ms`.
- `scripts/test_postprocess.sh` passed after adding `split_ms`.
- `cmake --build build -j$(nproc)` passed after adding SEG RPP Level 1 gather and low-res threshold kernels.
- `scripts/run.sh --demo rgb --loop 1` could not run because `rpp_server` reported `no RPP device found`.
- `scripts/run_postprocess_test.sh --loop 1 --warmup 0` could not run because `rpp_server` reported `no RPP device found`.
