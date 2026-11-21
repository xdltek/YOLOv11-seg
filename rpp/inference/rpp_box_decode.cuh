#ifndef YOLO11_RPP_BOX_DECODE_CUH
#define YOLO11_RPP_BOX_DECODE_CUH
#include <rpp_runtime.h>

// Inputs and outputs are SRAM FP32 [4,N].
void launch_yolo11_box_decode(rtStream_t stream, const float* distances, float* boxes,
                             int width, int height, int anchors);
// Logits are FP32 [4,16,N], boxes FP32 [4,N], both in SRAM.
void launch_yolo11_dfl_decode(rtStream_t stream, const float* logits, float* boxes,
                             int width, int height, int anchors);
#endif
