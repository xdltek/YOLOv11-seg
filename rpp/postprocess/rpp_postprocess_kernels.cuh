/**
 * @file rpp_postprocess_kernels.cuh
 * @brief RPP postprocessing kernel launcher declarations for YOLOv11-SEG masks.
 */
#ifndef XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH
#define XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH

#include <rpp_runtime.h>

// Stable, threshold-only compaction. All pointers refer to SRAM; no host count
// readback and no candidate truncation. Ping/pong/map each contain N int32s.
void launch_yolov11_seg_compact_candidates(rtStream_t stream, const float* input,
    float* output, int* ping, int* pong, int* map, int anchors, int classes, float threshold);

// Four taps per axis compose the two reference bilinear resize operations.
void launch_yolov11_seg_resample_probabilities(
    rtStream_t stream, const float* probabilities, const int* tile_base, const int* tile_stride,
    const int* tile_y, const int* tile_x, const int* tile_output, const int* tile_pixels,
    const int* x_indices, const float* x_weights, const int* y_indices, const float* y_weights,
    unsigned char* output, int scratch, int tiles, float threshold);

/**
 * @brief Launch YOLOv11-SEG output split into detection head and mask coefficients.
 */
void launch_yolov11_seg_split_det_coeff(rtStream_t stream,
                                        const float* input,
                                        float* det_output,
                                        float* coeff_output,
                                        int anchors,
                                        int det_channels,
                                        int coeff_count,
                                        int input_channels,
                                        int input_channels_first);

/** Decode class scores and threshold anchors into seven-float fixed slots. */
void launch_yolov11_seg_decode_candidate_slots(rtStream_t stream,
                                               const float* combined_head,
                                               float* candidate_slots,
                                               int anchors,
                                               int class_count,
                                               int input_channels,
                                               int input_channels_first,
                                               float score_threshold);

/**
 * @brief Launch selected mask coefficient gather by selected anchor indices.
 */
void launch_yolov11_seg_gather_selected_coefficients(rtStream_t stream,
                                                     const float* combined_head,
                                                     const int* selected_anchor_indices,
                                                     float* selected_coefficients,
                                                     int segment_count,
                                                     int coeff_count,
                                                     int anchors,
                                                     int det_channels,
                                                     int input_channels,
                                                     int input_channels_first);

/**
 * @brief Launch prototype linear combination and sigmoid; raw_masks contains probabilities.
 */
void launch_yolov11_seg_combine_prototypes(rtStream_t stream,
                                           const float* prototype,
                                           const float* coefficients,
                                           float* raw_masks,
                                           int segment_count,
                                           int coeff_count,
                                           int prototype_height,
                                           int prototype_width);

/** Combine and sigmoid only prototype pixels referenced by selected source-image ROIs. */
void launch_yolov11_seg_combine_prototypes_sparse(
    rtStream_t stream, const float* prototype, const float* coefficients,
    const int* point_segments, const int* point_proto_indices, float* packed_raw_masks,
    int point_count, int coeff_count, int prototype_height, int prototype_width);

/**
 * @brief Launch low-resolution sigmoid and threshold for selected raw masks.
 */
void launch_yolov11_seg_sigmoid_threshold_lowres(rtStream_t stream,
                                                 const float* raw_masks,
                                                 float* binary_masks,
                                                 int segment_count,
                                                 int mask_height,
                                                 int mask_width,
                                                 float logit_threshold);

/**
 * @brief Launch source-size resize and box crop from one low-resolution binary mask.
 */
void launch_yolov11_seg_resize_raw_to_source_crop(rtStream_t stream,
                                                  const float* raw_mask,
                                                  const int* box_xywh,
                                                     float* source_mask,
                                                     int source_height,
                                                     int source_width,
                                                     int model_height,
                                                     int model_width,
                                                     int mask_height,
                                                     int mask_width,
                                                     int resized_height,
                                                     int resized_width,
                                                     int pad_top,
                                                     int pad_left,
                                                     float logit_threshold);

/**
 * @brief Launch bbox-local mask resize/crop from one low-resolution raw mask.
 */
void launch_yolov11_seg_resize_raw_to_box_mask(rtStream_t stream,
                                               const float* raw_mask,
                                               const int* box_xywh,
                                               float* box_mask,
                                               int box_height,
                                               int box_width,
                                               int source_height,
                                               int source_width,
                                               int model_height,
                                               int model_width,
                                               int mask_height,
                                               int mask_width,
                                               int resized_height,
                                               int resized_width,
                                               int pad_top,
                                               int pad_left,
                                               float logit_threshold);

/**
 * @brief Resize all selected raw masks into one packed uint8 bbox-mask buffer.
 */
void launch_yolov11_seg_resize_raw_to_packed_box_masks(rtStream_t stream,
                                                       const float* raw_masks,
                                                       const int* boxes_xywh,
                                                       const int* mask_offsets,
                                                       float* packed_box_masks,
                                                       int segment_count,
                                                       int max_box_height,
                                                       int max_box_width,
                                                       int source_height,
                                                       int source_width,
                                                       int model_height,
                                                       int model_width,
                                                       int mask_height,
                                                       int mask_width,
                                                       int resized_height,
                                                       int resized_width,
                                                       int pad_top,
                                                       int pad_left,
                                                       float logit_threshold);

/** Resize one raw mask directly into a packed uint8 ROI without padded threads. */
void launch_yolov11_seg_resize_raw_to_packed_u8_box_mask(rtStream_t stream,
                                                         const float* raw_mask,
                                                         const int* box_xywh,
                                                         unsigned char* packed_box_masks,
                                                         int output_offset,
                                                         int box_height,
                                                         int box_width,
                                                         int source_height,
                                                         int source_width,
                                                         int model_height,
                                                         int model_width,
                                                         int mask_height,
                                                         int mask_width,
                                                         int resized_height,
                                                         int resized_width,
                                                         int pad_top,
                                                         int pad_left,
                                                         float logit_threshold);

/** Resize all selected raw masks into one packed uint8 buffer with one kernel launch. */
void launch_yolov11_seg_resize_raw_to_packed_u8_box_masks(rtStream_t stream,
                                                          const float* raw_masks,
                                                          const int* tile_segments,
                                                          const int* tile_source_y,
                                                          const int* tile_source_x,
                                                          const int* tile_output,
                                                          const int* tile_pixels,
                                                          const int* source_x0,
                                                          const float* source_wx,
                                                          const int* source_y0,
                                                          const float* source_wy,
                                                          unsigned char* packed_box_masks,
                                                          int scratch_offset,
                                                          int tile_count,
                                                          int mask_height,
                                                          int mask_width,
                                                          float logit_threshold);

/** Resize packed local prototype logits into packed uint8 source-image ROI masks. */
void launch_yolov11_seg_resize_sparse_raw_to_packed_u8_box_masks(
    rtStream_t stream, const float* packed_raw_masks,
    const int* tile_raw_row0, const int* tile_raw_row1, const float* tile_wy,
    const int* tile_source_x, const int* tile_output, const int* tile_pixels,
    const int* source_x0, const float* source_wx,
    unsigned char* packed_box_masks, int scratch_offset,
    int tile_count, int mask_width, float logit_threshold);

#endif // XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH
