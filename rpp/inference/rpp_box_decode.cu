#include "rpp_box_decode.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <drvapi_error_string.h>
#include <rpp_com.h>
#include <rpp_drv_api.h>
#include <rpp_math.h>

__device__ __attribute__((always_inline)) inline float dfl_expectation(const float* logits, int direction, int anchors, int index)
{
    float maximum = logits[direction * 16 * anchors + index];
    for (int bin = 1; bin < 16; ++bin)
        maximum = rpp_max(maximum, logits[(direction * 16 + bin) * anchors + index]);
    float total = 0.0f, weighted = 0.0f;
    float weight = 0.0f;
    for (int bin = 0; bin < 16; ++bin) {
        const float value = rpp_expf(rpp_max(logits[(direction * 16 + bin) * anchors + index] - maximum, -80.0f));
        total += value;
        weighted += value * weight;
        weight += 1.0f;
    }
    return weighted / total;
}

__global__ void yolo11_dfl_decode_kernel(const float* logits, float* boxes,
                                        int start, int columns, float scale, int anchors)
{
    const int x = rpp_min(int(threadIdx.x), columns - 1);
    const int y = blockIdx.y + threadIdx.y;
    const int index = start + y * columns + x;
    const float left = dfl_expectation(logits, 0, anchors, index);
    const float top = dfl_expectation(logits, 1, anchors, index);
    const float right = dfl_expectation(logits, 2, anchors, index);
    const float bottom = dfl_expectation(logits, 3, anchors, index);
    boxes[index] = (static_cast<float>(x) + 0.5f + (right - left) * 0.5f) * scale;
    boxes[anchors + index] = (static_cast<float>(y) + 0.5f + (bottom - top) * 0.5f) * scale;
    boxes[2 * anchors + index] = (left + right) * scale;
    boxes[3 * anchors + index] = (top + bottom) * scale;
}

// Preserve FP32 anchor/stride arithmetic for small boxes at large coordinates.
// BF16 center/width calculations can move a 20-pixel-wide tie by several pixels.
__global__ void yolo11_box_decode_kernel(const float* distances, float* boxes,
                                        int start, int columns, float scale, int anchors)
{
    // Tail lanes evaluate the same valid anchor and store identical FP32 bits.
    // Row launches avoid division in the coordinate path entirely.
    const int x = rpp_min(int(threadIdx.x), columns - 1);
    const int y = blockIdx.y + threadIdx.y;
    const int index = start + y * columns + x;
    const float left = distances[index];
    const float top = distances[anchors + index];
    const float right = distances[2 * anchors + index];
    const float bottom = distances[3 * anchors + index];
    boxes[index] = (static_cast<float>(x) + 0.5f + (right - left) * 0.5f) * scale;
    boxes[anchors + index] = (static_cast<float>(y) + 0.5f + (bottom - top) * 0.5f) * scale;
    boxes[2 * anchors + index] = (left + right) * scale;
    boxes[3 * anchors + index] = (top + bottom) * scale;
}

void launch_yolo11_dfl_decode(rtStream_t stream, const float* logits, float* boxes,
                             int width, int height, int anchors)
{
    dim3 block = {0}, grid = {0};
    block.y = block.z = grid.x = grid.z = 1;
    int start = 0;
    for (int stride = 8; stride <= 32; stride *= 2) {
        const int columns = width / stride;
        block.x = ((columns + 31) / 32) * 32;
        grid.y = height / stride;
        yolo11_dfl_decode_kernel<<<grid, block, 0, stream>>>(logits, boxes, start, columns,
                                                           static_cast<float>(stride), anchors);
        start += columns * (height / stride);
    }
}

void launch_yolo11_box_decode(rtStream_t stream, const float* distances, float* boxes,
                             int width, int height, int anchors)
{
    dim3 block = {0};
    dim3 grid = {0};
    block.y = block.z = 1;
    grid.x = grid.z = 1;
    int start = 0;
    for (int stride = 8; stride <= 32; stride *= 2) {
        const int columns = width / stride;
        const int count = columns * (height / stride);
        block.x = ((columns + 31) / 32) * 32;
        grid.y = height / stride;
        yolo11_box_decode_kernel<<<grid, block, 0, stream>>>(distances, boxes, start, columns,
                                                           static_cast<float>(stride), anchors);
        start += count;
    }
}
