/**
 * @file segmentation_types.h
 * @brief Shared YOLOv11-SEG segmentation data structures and pipeline configuration.
 */
#ifndef XDLTEK_SAMPLES_YOLOV11_SEGMENTATION_TYPES_H
#define XDLTEK_SAMPLES_YOLOV11_SEGMENTATION_TYPES_H

#include <Infer.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

/**
 * @brief One YOLOv11-SEG instance restored to original image coordinates.
 */
struct Segmentation
{
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect box;
    int mask_width = 0;
    int mask_height = 0;
    std::vector<uint8_t> mask;
};

/**
 * @brief Geometry metadata shared by preprocessing and postprocessing.
 */
struct LetterboxInfo
{
    int source_width = 0;
    int source_height = 0;
    int model_width = 0;
    int model_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

/**
 * @brief Runtime parameters for a standard Ultralytics YOLOv11-SEG model.
 */
struct YoloV11SegConfig
{
    int input_width = 640;
    int input_height = 640;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.50f;
    bool agnostic = true;
    int class_count = 80;
    int mask_coeff_count = 32;
    int max_segments = 100;
    infer1::Dims detection_dimensions{};
    infer1::Dims prototype_dimensions{};
    size_t detection_elements = 0;
    size_t prototype_elements = 0;
};

#endif // XDLTEK_SAMPLES_YOLOV11_SEGMENTATION_TYPES_H
