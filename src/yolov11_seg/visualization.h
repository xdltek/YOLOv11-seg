/**
 * @file visualization.h
 * @brief YOLOv11-SEG visualization helpers.
 */
#ifndef XDLTEK_SAMPLES_YOLOV11_VISUALIZATION_H
#define XDLTEK_SAMPLES_YOLOV11_VISUALIZATION_H

#include "segmentation_types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

/**
 * @brief Draw retained instance masks, boxes, and labels on a host BGR image.
 * @param image Image modified in place with boxes and labels.
 * @param segments Segmentation records in source-image coordinates.
 * @param class_names Label table used for class id display.
 */
void draw_segmentations(cv::Mat& image, const std::vector<Segmentation>& segments, const std::vector<std::string>& class_names);
/**
 * @brief Save the rendered segmentation image to disk.
 * @param output_path Destination image path accepted by OpenCV image codecs.
 * @param image Rendered host image.
 */
bool save_segmentation_image(const std::string& output_path, const cv::Mat& image);

#endif // XDLTEK_SAMPLES_YOLOV11_VISUALIZATION_H
