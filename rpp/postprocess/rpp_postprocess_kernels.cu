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

/**
 * @brief Gather selected mask coefficients from the full coefficient head.
 */
__global__ void yolov11_seg_gather_selected_coefficients_kernel(const float* coeff_head,
                                                                const int* selected_anchor_indices,
                                                                float* selected_coefficients,
                                                                int coeff_count,
                                                                int anchors)
{
    int coeff = threadIdx.x;
    int segment = blockIdx.y;
    short valid_coeff = static_cast<short>(coeff < coeff_count);
    int safe_coeff = rpp_min(coeff, coeff_count - 1);
    int anchor = selected_anchor_indices[segment];
    short valid_anchor = static_cast<short>((anchor >= 0) & (anchor < anchors));
    int safe_anchor = rpp_min(rpp_max(anchor, 0), anchors - 1);
    short valid = static_cast<short>(valid_coeff & valid_anchor);

    float value = coeff_head[safe_coeff * anchors + safe_anchor];
    int out_offset = segment * coeff_count + safe_coeff;
    rpp_storemask(&selected_coefficients[out_offset], value, valid);
}

/**
 * @brief Launch selected mask coefficient gather by selected anchor indices.
 */
void launch_yolov11_seg_gather_selected_coefficients(rtStream_t stream,
                                                     const float* coeff_head,
                                                     const int* selected_anchor_indices,
                                                     float* selected_coefficients,
                                                     int segment_count,
                                                     int coeff_count,
                                                     int anchors)
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
        coeff_head,
        selected_anchor_indices,
        selected_coefficients,
        coeff_count,
        anchors);
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
