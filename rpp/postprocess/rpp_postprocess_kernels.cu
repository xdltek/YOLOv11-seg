/**
 * @file rpp_postprocess_kernels.cu
 * @brief RPP-like CUDA kernels for YOLOv11-SEG mask prototype combination.
 */
#include "rpp_postprocess_kernels.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <drvapi_error_string.h>
#include <rpp_com.h>
#include <rpp_drv_api.h>
#include <rpp_math.h>
#include <rpp_runtime.h>

/**
 * @brief Split output0 [det channels + mask coeff channels, anchors] on RPP SRAM.
 */
__global__ void yolov11_seg_split_det_coeff_kernel(const float* input,
                                                   float* det_output,
                                                   float* coeff_output,
                                                   int anchors,
                                                   int det_channels,
                                                   int coeff_count,
                                                   int input_channels,
                                                   int input_channels_first)
{
    int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.y;
    short valid = static_cast<short>(anchor < anchors);
    int safe_anchor = rpp_min(anchor, anchors - 1);

    int input_offset_cf = channel * anchors + safe_anchor;
    int input_offset_cl = safe_anchor * input_channels + channel;
    int input_offset = rpp_select(input_offset_cl,
                                  input_offset_cf,
                                  static_cast<short>(input_channels_first != 0));
    float value = input[input_offset];

    if (channel < det_channels) {
        int det_offset = channel * anchors + safe_anchor;
        rpp_storemask(&det_output[det_offset], value, valid);
    }
    else {
        int coeff = channel - det_channels;
        int coeff_offset = coeff * anchors + safe_anchor;
        short coeff_valid = static_cast<short>(valid & (coeff < coeff_count));
        rpp_storemask(&coeff_output[coeff_offset], value, coeff_valid);
    }
}

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
                                        int input_channels_first)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 256;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = (anchors + static_cast<int>(threadsPerBlock.x) - 1) /
                      static_cast<int>(threadsPerBlock.x);
    blocksPerGrid.y = det_channels + coeff_count;
    blocksPerGrid.z = 1;

    yolov11_seg_split_det_coeff_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        input,
        det_output,
        coeff_output,
        anchors,
        det_channels,
        coeff_count,
        input_channels,
        input_channels_first);
}

constexpr int kCandidateFields = 7;

__global__ void yolov11_seg_decode_candidate_slots_kernel(const float* combined_head,
                                                           float* candidate_slots,
                                                           int anchors,
                                                           int class_count,
                                                           int input_channels,
                                                           int input_channels_first,
                                                           float score_threshold)
{
    int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    short valid_anchor = static_cast<short>(anchor < anchors);
    int safe_anchor = rpp_min(anchor, anchors - 1);
    float best_score = 0.0f;
    int best_class = -1;
    for (int class_id = 0; class_id < class_count; ++class_id) {
        int channel = 4 + class_id;
        int offset_cf = channel * anchors + safe_anchor;
        int offset_cl = safe_anchor * input_channels + channel;
        int input_offset = rpp_select(offset_cl, offset_cf,
                                      static_cast<short>(input_channels_first != 0));
        float score = combined_head[input_offset];
        if (score > best_score) {
            best_score = score;
            best_class = class_id;
        }
    }
    short keep = static_cast<short>(valid_anchor & (best_class >= 0) & (best_score >= score_threshold));
    int offset = safe_anchor * kCandidateFields;
    for (int channel = 0; channel < 4; ++channel) {
        int offset_cf = channel * anchors + safe_anchor;
        int offset_cl = safe_anchor * input_channels + channel;
        int input_offset = rpp_select(offset_cl, offset_cf,
                                      static_cast<short>(input_channels_first != 0));
        rpp_storemask(&candidate_slots[offset + channel], combined_head[input_offset], valid_anchor);
    }
    rpp_storemask(&candidate_slots[offset + 4], rpp_select(0.0f, best_score, keep), valid_anchor);
    rpp_storemask(&candidate_slots[offset + 5], static_cast<float>(best_class), valid_anchor);
    rpp_storemask(&candidate_slots[offset + 6], static_cast<float>(safe_anchor), valid_anchor);
}

void launch_yolov11_seg_decode_candidate_slots(rtStream_t stream,
                                               const float* combined_head,
                                               float* candidate_slots,
                                               int anchors,
                                               int class_count,
                                               int input_channels,
                                               int input_channels_first,
                                               float score_threshold)
{
    dim3 decode_threads = {0};
    dim3 decode_blocks = {0};
    decode_threads.x = 256;
    decode_blocks.x = (anchors + 255) / 256;
    yolov11_seg_decode_candidate_slots_kernel<<<decode_blocks, decode_threads, 0, stream>>>(
        combined_head, candidate_slots, anchors, class_count,
        input_channels, input_channels_first, score_threshold);
}

/**
 * @brief Gather selected mask coefficients from the full coefficient head.
 */
__global__ void yolov11_seg_gather_selected_coefficients_kernel(const float* combined_head,
                                                                const int* selected_anchor_indices,
                                                                float* selected_coefficients,
                                                                int coeff_count,
                                                                int anchors,
                                                                int det_channels,
                                                                int input_channels,
                                                                int input_channels_first)
{
    int coeff = threadIdx.x;
    int segment = blockIdx.y;
    short valid_coeff = static_cast<short>(coeff < coeff_count);
    int safe_coeff = rpp_min(coeff, coeff_count - 1);
    int anchor = selected_anchor_indices[segment];
    short valid_anchor = static_cast<short>((anchor >= 0) & (anchor < anchors));
    int safe_anchor = rpp_min(rpp_max(anchor, 0), anchors - 1);
    short valid = static_cast<short>(valid_coeff & valid_anchor);

    int channel = det_channels + safe_coeff;
    int offset_cf = channel * anchors + safe_anchor;
    int offset_cl = safe_anchor * input_channels + channel;
    int input_offset = rpp_select(offset_cl, offset_cf,
                                  static_cast<short>(input_channels_first != 0));
    float value = combined_head[input_offset];
    int out_offset = segment * coeff_count + safe_coeff;
    rpp_storemask(&selected_coefficients[out_offset], value, valid);
}

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
                                                     int input_channels_first)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 64;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = 1;
    blocksPerGrid.y = segment_count;
    blocksPerGrid.z = 1;

    yolov11_seg_gather_selected_coefficients_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        combined_head,
        selected_anchor_indices,
        selected_coefficients,
        coeff_count,
        anchors,
        det_channels,
        input_channels,
        input_channels_first);
}

/**
 * @brief Combine prototype masks with selected coefficients into low-resolution raw masks.
 */
__global__ void yolov11_seg_combine_prototypes_kernel(const float* prototype,
                                                      const float* coefficients,
                                                      float* raw_masks,
                                                      int coeff_count,
                                                      int prototype_height,
                                                      int prototype_width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int segment = blockIdx.z;
    int proto_pixel = y * prototype_width + x;
    float sum = 0.0f;
    for (int coeff = 0; coeff < coeff_count; ++coeff) {
        float c = coefficients[segment * coeff_count + coeff];
        float proto = prototype[coeff * prototype_height * prototype_width + proto_pixel];
        sum += c * proto;
    }
    raw_masks[segment * prototype_height * prototype_width + proto_pixel] = sum;
}

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
                                           int prototype_width)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = prototype_width;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = 1;
    blocksPerGrid.y = prototype_height;
    blocksPerGrid.z = segment_count;

    yolov11_seg_combine_prototypes_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        prototype,
        coefficients,
        raw_masks,
        coeff_count,
        prototype_height,
        prototype_width);
}

__global__ void yolov11_seg_combine_prototypes_sparse_kernel(
    const float* prototype, const float* coefficients,
    const int* point_segments, const int* point_proto_indices,
    float* packed_raw_masks, int point_count, int coeff_count, int prototype_pixels)
{
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    short valid = static_cast<short>(point < point_count);
    int safe_point = rpp_min(point, point_count - 1);
    int segment = point_segments[safe_point];
    int proto_pixel = point_proto_indices[safe_point];
    float sum = 0.0f;
    for (int coeff = 0; coeff < coeff_count; ++coeff) {
        sum += coefficients[segment * coeff_count + coeff] *
               prototype[coeff * prototype_pixels + proto_pixel];
    }
    rpp_storemask(&packed_raw_masks[safe_point], sum, valid);
}

void launch_yolov11_seg_combine_prototypes_sparse(
    rtStream_t stream, const float* prototype, const float* coefficients,
    const int* point_segments, const int* point_proto_indices, float* packed_raw_masks,
    int point_count, int coeff_count, int prototype_height, int prototype_width)
{
    dim3 threads = {0};
    dim3 blocks = {0};
    threads.x = 256;
    blocks.x = (point_count + 255) / 256;
    yolov11_seg_combine_prototypes_sparse_kernel<<<blocks, threads, 0, stream>>>(
        prototype, coefficients, point_segments, point_proto_indices, packed_raw_masks,
        point_count, coeff_count, prototype_height * prototype_width);
}

/**
 * @brief Convert selected raw masks to low-resolution binary masks.
 */
__global__ void yolov11_seg_sigmoid_threshold_lowres_kernel(const float* raw_masks,
                                                           float* binary_masks,
                                                           int mask_height,
                                                           int mask_width,
                                                           float logit_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int segment = blockIdx.z;
    short valid = static_cast<short>(x < mask_width);
    int safe_x = rpp_min(x, mask_width - 1);
    int offset = segment * mask_height * mask_width + y * mask_width + safe_x;

    float raw = raw_masks[offset];
    short above = static_cast<short>(raw > logit_threshold);
    float binary = rpp_select(0.0f, 255.0f, above);
    rpp_storemask(&binary_masks[offset], binary, valid);
}

/**
 * @brief Launch low-resolution sigmoid and threshold for selected raw masks.
 */
void launch_yolov11_seg_sigmoid_threshold_lowres(rtStream_t stream,
                                                 const float* raw_masks,
                                                 float* binary_masks,
                                                 int segment_count,
                                                 int mask_height,
                                                 int mask_width,
                                                 float logit_threshold)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 256;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = (mask_width + static_cast<int>(threadsPerBlock.x) - 1) /
                      static_cast<int>(threadsPerBlock.x);
    blocksPerGrid.y = mask_height;
    blocksPerGrid.z = segment_count;

    yolov11_seg_sigmoid_threshold_lowres_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        raw_masks,
        binary_masks,
        mask_height,
        mask_width,
        logit_threshold);
}

/**
 * @brief Resize one raw low-resolution mask to source space and apply box crop.
 */
__global__ void yolov11_seg_resize_raw_to_source_crop_kernel(const float* raw_mask,
                                                            const int* box_xywh,
                                                            float* source_mask,
                                                            int source_width,
                                                            int model_height,
                                                            int model_width,
                                                            int mask_height,
                                                            int mask_width,
                                                            int resized_height,
                                                            int resized_width,
                                                            int pad_top,
                                                            int pad_left,
                                                            float logit_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int source_height = gridDim.y;
    short valid = static_cast<short>(x < source_width);
    int safe_x = rpp_min(x, source_width - 1);

    int box_x = box_xywh[0];
    int box_y = box_xywh[1];
    int box_w = box_xywh[2];
    int box_h = box_xywh[3];
    short inside_box = static_cast<short>((safe_x >= box_x) & (safe_x < box_x + box_w) &
                                          (y >= box_y) & (y < box_y + box_h));

    const int q = 8;
    const int one = 1 << q;
    int model_x_q = (safe_x * resized_width * one) / source_width + (pad_left << q);
    int model_y_q = (y * resized_height * one) / source_height + (pad_top << q);
    int mask_x_q = model_x_q * mask_width / model_width;
    int mask_y_q = model_y_q * mask_height / model_height;

    int x0 = rpp_min(rpp_max(mask_x_q >> q, 0), mask_width - 1);
    int y0 = rpp_min(rpp_max(mask_y_q >> q, 0), mask_height - 1);
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    int y1 = rpp_min(y0 + 1, mask_height - 1);
    int wx_i = mask_x_q & (one - 1);
    int wy_i = mask_y_q & (one - 1);
    float wx = static_cast<float>(wx_i) * (1.0f / 256.0f);
    float wy = static_cast<float>(wy_i) * (1.0f / 256.0f);

    float v00 = raw_mask[y0 * mask_width + x0];
    float v01 = raw_mask[y0 * mask_width + x1];
    float v10 = raw_mask[y1 * mask_width + x0];
    float v11 = raw_mask[y1 * mask_width + x1];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    float mask_value = top + (bottom - top) * wy;
    short foreground = static_cast<short>(mask_value > logit_threshold);
    short keep = static_cast<short>(inside_box & foreground);
    float output = rpp_select(0.0f, 255.0f, keep);
    rpp_storemask(&source_mask[y * source_width + safe_x], output, valid);
}

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
                                                  float logit_threshold)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 256;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = (source_width + static_cast<int>(threadsPerBlock.x) - 1) /
                      static_cast<int>(threadsPerBlock.x);
    blocksPerGrid.y = source_height;
    blocksPerGrid.z = 1;

    yolov11_seg_resize_raw_to_source_crop_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        raw_mask,
        box_xywh,
        source_mask,
        source_width,
        model_height,
        model_width,
        mask_height,
        mask_width,
        resized_height,
        resized_width,
        pad_top,
        pad_left,
        logit_threshold);
}

/**
 * @brief Resize one raw low-resolution mask directly into bbox-local source pixels.
 */
__global__ void yolov11_seg_resize_raw_to_box_mask_kernel(const float* raw_mask,
                                                          const int* box_xywh,
                                                          float* box_mask,
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
                                                          float logit_threshold)
{
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y;
    int box_x = box_xywh[0];
    int box_y = box_xywh[1];
    int box_w = box_xywh[2];
    int box_h = box_xywh[3];
    short valid = static_cast<short>((local_x < box_w) & (local_y < box_h));
    int safe_local_x = rpp_min(local_x, box_w - 1);
    int source_x = rpp_min(rpp_max(box_x + safe_local_x, 0), source_width - 1);
    int source_y = rpp_min(rpp_max(box_y + local_y, 0), source_height - 1);

    const int q = 8;
    const int one = 1 << q;
    int model_x_q = (source_x * resized_width * one) / source_width + (pad_left << q);
    int model_y_q = (source_y * resized_height * one) / source_height + (pad_top << q);
    int mask_x_q = model_x_q * mask_width / model_width;
    int mask_y_q = model_y_q * mask_height / model_height;

    int x0 = rpp_min(rpp_max(mask_x_q >> q, 0), mask_width - 1);
    int y0 = rpp_min(rpp_max(mask_y_q >> q, 0), mask_height - 1);
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    int y1 = rpp_min(y0 + 1, mask_height - 1);
    int wx_i = mask_x_q & (one - 1);
    int wy_i = mask_y_q & (one - 1);
    float wx = static_cast<float>(wx_i) * (1.0f / 256.0f);
    float wy = static_cast<float>(wy_i) * (1.0f / 256.0f);

    float v00 = raw_mask[y0 * mask_width + x0];
    float v01 = raw_mask[y0 * mask_width + x1];
    float v10 = raw_mask[y1 * mask_width + x0];
    float v11 = raw_mask[y1 * mask_width + x1];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    float mask_value = top + (bottom - top) * wy;
    short foreground = static_cast<short>(mask_value > logit_threshold);
    float output = rpp_select(0.0f, 255.0f, foreground);
    rpp_storemask(&box_mask[local_y * box_w + safe_local_x], output, valid);
}

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
                                               float logit_threshold)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 256;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = (box_width + static_cast<int>(threadsPerBlock.x) - 1) /
                      static_cast<int>(threadsPerBlock.x);
    blocksPerGrid.y = box_height;
    blocksPerGrid.z = 1;

    yolov11_seg_resize_raw_to_box_mask_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        raw_mask,
        box_xywh,
        box_mask,
        source_height,
        source_width,
        model_height,
        model_width,
        mask_height,
        mask_width,
        resized_height,
        resized_width,
        pad_top,
        pad_left,
        logit_threshold);
}

/**
 * @brief Resize selected raw masks to packed bbox-local uint8 masks.
 */
__global__ void yolov11_seg_resize_raw_to_packed_box_masks_kernel(
    const float* raw_masks,
    const int* boxes_xywh,
    const int* mask_offsets,
    float* packed_box_masks,
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
    float logit_threshold)
{
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_y = blockIdx.y;
    int segment = blockIdx.z;
    const int* box = boxes_xywh + segment * 4;
    int box_x = box[0];
    int box_y = box[1];
    int box_w = box[2];
    int box_h = box[3];
    short valid = static_cast<short>((local_x < box_w) & (local_y < box_h));
    int safe_local_x = rpp_min(local_x, box_w - 1);
    int safe_local_y = rpp_min(local_y, box_h - 1);
    int source_x = rpp_min(rpp_max(box_x + safe_local_x, 0), source_width - 1);
    int source_y = rpp_min(rpp_max(box_y + safe_local_y, 0), source_height - 1);

    const int q = 8;
    const int one = 1 << q;
    int model_x_q = (source_x * resized_width * one) / source_width + (pad_left << q);
    int model_y_q = (source_y * resized_height * one) / source_height + (pad_top << q);
    int mask_x_q = model_x_q * mask_width / model_width;
    int mask_y_q = model_y_q * mask_height / model_height;
    int x0 = rpp_min(rpp_max(mask_x_q >> q, 0), mask_width - 1);
    int y0 = rpp_min(rpp_max(mask_y_q >> q, 0), mask_height - 1);
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    int y1 = rpp_min(y0 + 1, mask_height - 1);
    float wx = static_cast<float>(mask_x_q & (one - 1)) * (1.0f / 256.0f);
    float wy = static_cast<float>(mask_y_q & (one - 1)) * (1.0f / 256.0f);

    const float* raw_mask = raw_masks + segment * mask_height * mask_width;
    float v00 = raw_mask[y0 * mask_width + x0];
    float v01 = raw_mask[y0 * mask_width + x1];
    float v10 = raw_mask[y1 * mask_width + x0];
    float v11 = raw_mask[y1 * mask_width + x1];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    short foreground = static_cast<short>((top + (bottom - top) * wy) > logit_threshold);
    float output = rpp_select(0.0f, 255.0f, foreground);
    int output_offset = mask_offsets[segment] + safe_local_y * box_w + safe_local_x;
    rpp_storemask(&packed_box_masks[output_offset], output, valid);
}

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
                                                       float logit_threshold)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = 256;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = (max_box_width + static_cast<int>(threadsPerBlock.x) - 1) /
                      static_cast<int>(threadsPerBlock.x);
    blocksPerGrid.y = max_box_height;
    blocksPerGrid.z = segment_count;

    yolov11_seg_resize_raw_to_packed_box_masks_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        raw_masks, boxes_xywh, mask_offsets, packed_box_masks,
        source_height, source_width, model_height, model_width,
        mask_height, mask_width, resized_height, resized_width,
        pad_top, pad_left, logit_threshold);
}

__global__ void yolov11_seg_resize_raw_to_packed_u8_box_mask_kernel(
    const float* raw_mask,
    const int* box_xywh,
    unsigned char* packed_box_masks,
    int output_offset,
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
    float logit_threshold)
{
    int local_x = threadIdx.x;
    int local_y = blockIdx.y;
    int box_x = box_xywh[0];
    int box_y = box_xywh[1];
    int box_w = box_xywh[2];
    int source_x = rpp_min(rpp_max(box_x + local_x, 0), source_width - 1);
    int source_y = rpp_min(rpp_max(box_y + local_y, 0), source_height - 1);
    const int q = 8;
    const int one = 1 << q;
    int model_x_q = (source_x * resized_width * one) / source_width + (pad_left << q);
    int model_y_q = (source_y * resized_height * one) / source_height + (pad_top << q);
    int mask_x_q = model_x_q * mask_width / model_width;
    int mask_y_q = model_y_q * mask_height / model_height;
    int x0 = rpp_min(rpp_max(mask_x_q >> q, 0), mask_width - 1);
    int y0 = rpp_min(rpp_max(mask_y_q >> q, 0), mask_height - 1);
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    int y1 = rpp_min(y0 + 1, mask_height - 1);
    float wx = static_cast<float>(mask_x_q & (one - 1)) * (1.0f / 256.0f);
    float wy = static_cast<float>(mask_y_q & (one - 1)) * (1.0f / 256.0f);
    float v00 = raw_mask[y0 * mask_width + x0];
    float v01 = raw_mask[y0 * mask_width + x1];
    float v10 = raw_mask[y1 * mask_width + x0];
    float v11 = raw_mask[y1 * mask_width + x1];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    packed_box_masks[output_offset + local_y * box_w + local_x] =
        static_cast<unsigned char>((top + (bottom - top) * wy) > logit_threshold ? 255 : 0);
}

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
                                                         float logit_threshold)
{
    dim3 threads = {0};
    dim3 blocks = {0};
    threads.x = box_width;
    blocks.x = 1;
    blocks.y = box_height;
    yolov11_seg_resize_raw_to_packed_u8_box_mask_kernel<<<blocks, threads, 0, stream>>>(
        raw_mask, box_xywh, packed_box_masks, output_offset,
        source_height, source_width, model_height, model_width,
        mask_height, mask_width, resized_height, resized_width,
        pad_top, pad_left, logit_threshold);
}

__global__ void yolov11_seg_resize_raw_to_packed_u8_box_masks_kernel(
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
    int mask_height,
    int mask_width,
    float logit_threshold)
{
    int tile_index = blockIdx.x;
    int segment = tile_segments[tile_index];
    int source_y = tile_source_y[tile_index];
    int source_x_base = tile_source_x[tile_index];
    int output_base = tile_output[tile_index];
    int pixel_count = tile_pixels[tile_index];
    int tile_x = threadIdx.x;
    short valid = static_cast<short>(tile_x < pixel_count);
    int safe_tile_x = rpp_min(tile_x, pixel_count - 1);
    int source_x = source_x_base + safe_tile_x;
    int y0 = source_y0[source_y];
    int y1 = rpp_min(y0 + 1, mask_height - 1);
    float wy = source_wy[source_y];
    const float* raw_mask = raw_masks + segment * mask_height * mask_width;
    int x0 = source_x0[source_x];
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    float wx = source_wx[source_x];
    float top = raw_mask[y0 * mask_width + x0] +
                (raw_mask[y0 * mask_width + x1] - raw_mask[y0 * mask_width + x0]) * wx;
    float bottom = raw_mask[y1 * mask_width + x0] +
                   (raw_mask[y1 * mask_width + x1] - raw_mask[y1 * mask_width + x0]) * wx;
    int output = rpp_select(scratch_offset, output_base + safe_tile_x, valid);
    packed_box_masks[output] = static_cast<unsigned char>(
        (top + (bottom - top) * wy) > logit_threshold ? 255 : 0);
}

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
                                                          float logit_threshold)
{
    dim3 threads = {0};
    dim3 blocks = {0};
    threads.x = 256;
    blocks.x = tile_count;
    yolov11_seg_resize_raw_to_packed_u8_box_masks_kernel<<<blocks, threads, 0, stream>>>(
        raw_masks, tile_segments, tile_source_y, tile_source_x, tile_output, tile_pixels,
        source_x0, source_wx, source_y0, source_wy,
        packed_box_masks, scratch_offset,
        mask_height, mask_width, logit_threshold);
}

__global__ void yolov11_seg_resize_sparse_raw_to_packed_u8_box_masks_kernel(
    const float* packed_raw_masks,
    const int* tile_raw_row0,
    const int* tile_raw_row1,
    const float* tile_wy,
    const int* tile_source_x,
    const int* tile_output,
    const int* tile_pixels,
    const int* source_x0,
    const float* source_wx,
    unsigned char* packed_box_masks,
    int scratch_offset,
    int mask_width,
    float logit_threshold)
{
    int tile_index = blockIdx.x;
    int source_x_base = tile_source_x[tile_index];
    int output_base = tile_output[tile_index];
    int pixel_count = tile_pixels[tile_index];
    int tile_x = threadIdx.x;
    short valid = static_cast<short>(tile_x < pixel_count);
    int safe_tile_x = rpp_min(tile_x, pixel_count - 1);
    int source_x = source_x_base + safe_tile_x;
    int x0 = source_x0[source_x];
    int x1 = rpp_min(x0 + 1, mask_width - 1);
    float v00 = packed_raw_masks[tile_raw_row0[tile_index] + x0];
    float v01 = packed_raw_masks[tile_raw_row0[tile_index] + x1];
    float v10 = packed_raw_masks[tile_raw_row1[tile_index] + x0];
    float v11 = packed_raw_masks[tile_raw_row1[tile_index] + x1];
    float wx = source_wx[source_x];
    float wy = tile_wy[tile_index];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    int output = rpp_select(scratch_offset, output_base + safe_tile_x, valid);
    packed_box_masks[output] = static_cast<unsigned char>(
        (top + (bottom - top) * wy) > logit_threshold ? 255 : 0);
}

void launch_yolov11_seg_resize_sparse_raw_to_packed_u8_box_masks(
    rtStream_t stream, const float* packed_raw_masks,
    const int* tile_raw_row0, const int* tile_raw_row1, const float* tile_wy,
    const int* tile_source_x, const int* tile_output, const int* tile_pixels,
    const int* source_x0, const float* source_wx,
    unsigned char* packed_box_masks, int scratch_offset,
    int tile_count, int mask_width, float logit_threshold)
{
    dim3 threads = {0};
    dim3 blocks = {0};
    threads.x = 256;
    blocks.x = tile_count;
    yolov11_seg_resize_sparse_raw_to_packed_u8_box_masks_kernel<<<blocks, threads, 0, stream>>>(
        packed_raw_masks, tile_raw_row0, tile_raw_row1, tile_wy,
        tile_source_x, tile_output, tile_pixels, source_x0, source_wx,
        packed_box_masks, scratch_offset, mask_width, logit_threshold);
}
