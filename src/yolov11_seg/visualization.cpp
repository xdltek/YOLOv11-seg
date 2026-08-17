/**
 * @file visualization.cpp
 * @brief YOLOv11-SEG visualization helpers.
 */
#include "visualization.h"

#include "labels.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

namespace
{
/**
 * @brief Return the fixed palette used when drawing instance masks and boxes.
 */
const std::vector<cv::Scalar>& segmentation_colors()
{
    static const std::vector<cv::Scalar> kColors = {
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 0),
    };
    return kColors;
}
}

/**
 * @brief Draw segmentation masks, boxes, and labels onto an image.
 */
void draw_segmentations(cv::Mat& image,
                        const std::vector<Segmentation>& segments,
                        const std::vector<std::string>& class_names)
{
    const auto& colors = segmentation_colors();

    for (const auto& segment : segments) {
        cv::Rect box = segment.box & cv::Rect(0, 0, image.cols, image.rows);
        if (box.empty()) {
            continue;
        }

        size_t color_index = segment.class_id >= 0 ? static_cast<size_t>(segment.class_id) % colors.size() : 0;
        const auto color = colors[color_index];
        if (segment.mask_width == image.cols &&
            segment.mask_height == image.rows &&
            segment.mask.size() == static_cast<size_t>(image.cols * image.rows)) {
            for (int y = 0; y < image.rows; ++y) {
                const uint8_t* mask_row = segment.mask.data() + static_cast<size_t>(y) * static_cast<size_t>(image.cols);
                cv::Vec3b* image_row = image.ptr<cv::Vec3b>(y);
                for (int x = 0; x < image.cols; ++x) {
                    if (mask_row[x] == 0) {
                        continue;
                    }
                    image_row[x][0] = static_cast<uint8_t>(0.55f * image_row[x][0] + 0.45f * color[0]);
                    image_row[x][1] = static_cast<uint8_t>(0.55f * image_row[x][1] + 0.45f * color[1]);
                    image_row[x][2] = static_cast<uint8_t>(0.55f * image_row[x][2] + 0.45f * color[2]);
                }
            }
        }
        else if (segment.mask_width == box.width &&
                 segment.mask_height == box.height &&
                 segment.mask.size() == static_cast<size_t>(box.width * box.height)) {
            for (int y = 0; y < box.height; ++y) {
                const uint8_t* mask_row = segment.mask.data() + static_cast<size_t>(y) * static_cast<size_t>(box.width);
                cv::Vec3b* image_row = image.ptr<cv::Vec3b>(box.y + y);
                for (int x = 0; x < box.width; ++x) {
                    if (mask_row[x] == 0) {
                        continue;
                    }
                    int image_x = box.x + x;
                    image_row[image_x][0] = static_cast<uint8_t>(0.55f * image_row[image_x][0] + 0.45f * color[0]);
                    image_row[image_x][1] = static_cast<uint8_t>(0.55f * image_row[image_x][1] + 0.45f * color[1]);
                    image_row[image_x][2] = static_cast<uint8_t>(0.55f * image_row[image_x][2] + 0.45f * color[2]);
                }
            }
        }
        cv::rectangle(image, box, color, 3);

        int label_top = std::max(0, box.y - 20);
        cv::rectangle(image,
                      cv::Point(box.x, label_top),
                      cv::Point(std::min(image.cols - 1, box.x + box.width), box.y),
                      color,
                      cv::FILLED);
        std::string label = detection_class_label(segment.class_id, class_names);
        cv::putText(image,
                    label,
                    cv::Point(box.x, std::max(12, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0));
    }
}

/**
 * @brief Save a segmentation visualization image to disk using OpenCV image codecs.
 */
bool save_segmentation_image(const std::string& output_path, const cv::Mat& image)
{
    return cv::imwrite(output_path, image);
}
