/**
 * @file rpp_postprocess_kernels.cuh
 * @brief RPP postprocessing kernel launcher declarations for YOLOv11-SEG masks.
 */
#ifndef XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH
#define XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH

#include <rpp_runtime.h>

// All pointers refer to SRAM. Prediction coefficients use channel-major [32,N].

void launch_yolov11_seg_compact_candidates(rtStream_t stream, const float* input,
    float* output, int* ping, int* pong, int* map, int anchors, int classes, float threshold);

void launch_yolov11_seg_resample_probabilities(
    rtStream_t stream, const float* probabilities, const int* tile_base, const int* tile_stride,
    const int* tile_y, const int* tile_x, const int* tile_output, const int* tile_pixels,
    const int* x_indices, const float* x_weights, const int* y_indices, const float* y_weights,
    unsigned char* output, int scratch, int tiles, float threshold);

void launch_yolov11_seg_gather_selected_coefficients(rtStream_t stream,
                                                     const float* combined_head,
                                                     const int* selected_anchor_indices,
                                                     float* selected_coefficients,
                                                     int segment_count,
                                                     int coeff_count,
                                                     int anchors);

void launch_yolov11_seg_combine_prototypes_sparse(
    rtStream_t stream, const float* prototype, const float* coefficients,
    const int* point_segments, const int* point_proto_indices, float* packed_raw_masks,
    int point_count, int coeff_count, int prototype_height, int prototype_width);

#endif
