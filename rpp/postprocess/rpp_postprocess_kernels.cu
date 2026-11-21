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

__global__ void seg_compact_flags(const float* input, int* flags, int anchors, int classes, float threshold)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int safe=rpp_min(i,anchors-1);
    float score=input[4*anchors+safe];
    for(int c=1;c<classes;++c) score=rpp_max(score,input[(4+c)*anchors+safe]);
    rpp_storemask(flags+safe,int(score>=threshold),short(i<anchors));
}
__global__ void seg_compact_scan(const int* input, int* output, int anchors, int offset)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int safe=rpp_min(i,anchors-1);
    const int previous=input[rpp_max(safe-offset,0)];
    const int value=input[safe]+rpp_select(0,previous,short(safe>=offset));
    rpp_storemask(output+safe,value,short(i<anchors));
}
__global__ void seg_compact_zero(float* output, int* map, int anchors)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int safe=rpp_min(i,anchors-1);
    rpp_storemask(output+blockIdx.y*anchors+safe,0.0f,short(i<anchors));
    if(blockIdx.y==0) rpp_storemask(map+safe,-1,short(i<anchors));
}
__global__ void seg_compact_scatter(const float* input, float* output, const int* prefix, int* map, int anchors)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int safe=rpp_min(i,anchors-1);
    const int count=prefix[safe];
    const int previous=rpp_select(0,prefix[rpp_max(safe-1,0)],short(safe>0));
    const int rank=rpp_max(count-1,0);
    const short valid=short(i<anchors && count>previous);
    rpp_storemask(output+blockIdx.y*anchors+rank,input[blockIdx.y*anchors+safe],valid);
    if(blockIdx.y==0) rpp_storemask(map+rank,safe,valid);
}
void launch_yolov11_seg_compact_candidates(rtStream_t stream, const float* input,
    float* output, int* ping, int* pong, int* map, int anchors, int classes, float threshold)
{
    dim3 block={0},grid={0};block.x=256;block.y=block.z=1;
    grid.x=(anchors+255)/256;grid.y=grid.z=1;
    seg_compact_flags<<<grid,block,0,stream>>>(input,ping,anchors,classes,threshold);
    for(int offset=1;offset<anchors;offset*=2) {
        seg_compact_scan<<<grid,block,0,stream>>>(ping,pong,anchors,offset);
        int* temporary=ping;ping=pong;pong=temporary;
    }
    grid.y=classes+4;
    seg_compact_zero<<<grid,block,0,stream>>>(output,map,anchors);
    seg_compact_scatter<<<grid,block,0,stream>>>(input,output,ping,map,anchors);
}

__global__ void yolov11_seg_gather_selected_coefficients_kernel(const float* combined_head,
                                                                const int* selected_anchor_indices,
                                                                float* selected_coefficients,
                                                                int coeff_count,
                                                                int anchors)
{
    int coeff = threadIdx.x;
    int segment = blockIdx.y;
    int safe_coeff = rpp_min(coeff, coeff_count - 1);
    int anchor = selected_anchor_indices[segment];
    int safe_anchor = rpp_min(rpp_max(anchor, 0), anchors - 1);
    float value = combined_head[safe_coeff * anchors + safe_anchor];
    int out_offset = segment * coeff_count + safe_coeff;
    // The launcher covers exactly coeff_count lanes. No masked duplicate
    // tail lanes may alias and overwrite the final coefficient.
    selected_coefficients[out_offset] = value;
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
                                                     int anchors)
{
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = coeff_count;
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
        anchors);
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
    float exponent = rpp_exp2_precise(-rpp_min(40.0f, rpp_max(-40.0f, sum)) * 1.442695041f);
    rpp_storemask(&packed_raw_masks[safe_point], rpp_div(1.0f, 1.0f + exponent), valid);
}

// Compose prototype->model and cropped-model->source bilinear resizes. Three
// merged taps per axis preserve the two bilinear resize operations.
__global__ void yolov11_seg_resample_probabilities_kernel(
    const float* probabilities, const int* tile_base, const int* tile_stride,
    const int* tile_y, const int* tile_x, const int* tile_output, const int* tile_pixels,
    const int* x_indices, const float* x_weights, const int* y_indices, const float* y_weights,
    unsigned char* output, int scratch, float threshold)
{
    int tile = blockIdx.x;
    int lane = threadIdx.x;
    short valid = static_cast<short>(lane < tile_pixels[tile]);
    int x = tile_x[tile] + rpp_min(lane, tile_pixels[tile] - 1);
    int y = tile_y[tile];
    int base = tile_base[tile];
    int stride = tile_stride[tile];
    float value = 0.0f;
    for (int j = 0; j < 3; ++j) {
        float row = 0.0f;
        int offset = base + y_indices[3*y+j] * stride;
        for (int i = 0; i < 3; ++i)
            row += probabilities[offset + x_indices[3*x+i]] * x_weights[3*x+i];
        value += row * y_weights[3*y+j];
    }
    int destination = rpp_select(scratch, tile_output[tile]+lane, valid);
    output[destination] = static_cast<unsigned char>(value > threshold ? 255 : 0);
}

void launch_yolov11_seg_resample_probabilities(
    rtStream_t stream, const float* probabilities, const int* tile_base, const int* tile_stride,
    const int* tile_y, const int* tile_x, const int* tile_output, const int* tile_pixels,
    const int* x_indices, const float* x_weights, const int* y_indices, const float* y_weights,
    unsigned char* output, int scratch, int tiles, float threshold)
{
    dim3 threads = {0}, blocks = {0};
    threads.x = 256; blocks.x = tiles;
    yolov11_seg_resample_probabilities_kernel<<<blocks, threads, 0, stream>>>(
        probabilities, tile_base, tile_stride, tile_y, tile_x, tile_output, tile_pixels,
        x_indices, x_weights, y_indices, y_weights, output, scratch, threshold);
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
