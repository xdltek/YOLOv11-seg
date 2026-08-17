/**
 * @file rpp_postprocess_kernels.cuh
 * @brief RPP postprocessing kernel launcher declarations for YOLOv11-SEG masks.
 */
#ifndef XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH
#define XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH

#include <rpp_runtime.h>

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

/**
 * @brief Launch selected mask coefficient gather by selected anchor indices.
 */
void launch_yolov11_seg_gather_selected_coefficients(rtStream_t stream,
                                                     const float* coeff_head,
                                                     const int* selected_anchor_indices,
                                                     float* selected_coefficients,
                                                     int segment_count,
                                                     int coeff_count,
                                                     int anchors);

/**
 * @brief Launch selected-mask prototype linear combination.
 */
void launch_yolov11_seg_combine_prototypes(rtStream_t stream,
                                           const float* prototype,
                                           const float* coefficients,
                                           float* raw_masks,
                                           int segment_count,
                                           int coeff_count,
                                           int prototype_height,
                                           int prototype_width);

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

#endif // XDLTEK_SAMPLES_RPP_POSTPROCESS_KERNELS_CUH
