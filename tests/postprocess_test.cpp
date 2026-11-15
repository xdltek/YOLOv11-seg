/**
 * @file main.cpp
 * @brief Standalone YOLOv11-SEG postprocess validation and timing demo.
 */
#include "argparse/argparse.hpp"
#include "perf_trace_session.h"
#include "rpp_yolov11_seg_postprocessor.h"

#include <opencv2/imgproc.hpp>
#include <rpp_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::high_resolution_clock;
constexpr int kClassCount = 80;
constexpr int kMaskCoeffCount = 32;
constexpr int kAnchors = 8400;
constexpr int kDetectionChannels = 4 + kClassCount + kMaskCoeffCount;
constexpr int kPrototypeHeight = 160;
constexpr int kPrototypeWidth = 160;
#ifdef YOLO_ENABLE_RPP_PERF
constexpr bool kPerfTraceEnabled = true;
#else
constexpr bool kPerfTraceEnabled = false;
#endif
constexpr const char* kPerfTraceDir = "../trace";

struct ReferenceCandidate
{
    int anchor = -1;
    int class_id = -1;
    float score = 0.0f;
    cv::Rect box;
};

struct AccuracyThresholds
{
    float max_score_diff = 0.002f;
    float min_box_iou = 0.98f;
    float max_box_diff_px = 1.0f;
    float min_mask_iou = 0.90f;
    float max_mask_diff_ratio = 0.15f;
};

struct AccuracySummary
{
    bool passed = true;
    int output_segments = 0;
    int reference_segments = 0;
    int matched_segments = 0;
    int extra_output_segments = 0;
    int missing_reference_segments = 0;
    float min_box_iou = 1.0f;
    float max_box_diff_px = 0.0f;
    float min_mask_iou = 1.0f;
    float max_mask_diff_ratio = 0.0f;
};

bool check_rt(rtError_t status, const char* what)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << what << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

double elapsed_ms(Clock::time_point start, Clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double bytes_to_kib(size_t bytes)
{
    return static_cast<double>(bytes) / 1024.0;
}

bool read_binary_f32(const std::string& path, std::vector<float>& values)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open binary file: " << path << std::endl;
        return false;
    }
    input.seekg(0, std::ios::end);
    std::streamoff bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(float))) != 0) {
        std::cerr << "Invalid f32 binary size: " << path << std::endl;
        return false;
    }
    values.resize(static_cast<size_t>(bytes) / sizeof(float));
    return static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), bytes));
}

void set_detection_value(std::vector<float>& detection, int channel, int anchor, float value)
{
    detection[static_cast<size_t>(channel) * static_cast<size_t>(kAnchors) +
              static_cast<size_t>(anchor)] = value;
}

float detection_value_at(const std::vector<float>& detection, int channel, int anchor)
{
    return detection[static_cast<size_t>(channel) * static_cast<size_t>(kAnchors) +
                     static_cast<size_t>(anchor)];
}

float sigmoid_ref(float value)
{
    if (value >= 0.0f) {
        float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(value);
    return z / (1.0f + z);
}

cv::Rect clip_rect_to_image_ref(float left, float top, float right, float bottom, int width, int height)
{
    left = std::max(0.0f, std::min(left, static_cast<float>(width - 1)));
    top = std::max(0.0f, std::min(top, static_cast<float>(height - 1)));
    right = std::max(0.0f, std::min(right, static_cast<float>(width - 1)));
    bottom = std::max(0.0f, std::min(bottom, static_cast<float>(height - 1)));

    int x0 = static_cast<int>(std::floor(left));
    int y0 = static_cast<int>(std::floor(top));
    int x1 = static_cast<int>(std::ceil(right));
    int y1 = static_cast<int>(std::ceil(bottom));
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

float intersection_over_union_ref(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    float intersection = 0.0f;
    if (x2 > x1 && y2 > y1) {
        intersection = static_cast<float>((x2 - x1) * (y2 - y1));
    }
    float union_area = static_cast<float>(a.area() + b.area()) - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

float max_box_coordinate_diff(const cv::Rect& a, const cv::Rect& b)
{
    float dx0 = static_cast<float>(std::abs(a.x - b.x));
    float dy0 = static_cast<float>(std::abs(a.y - b.y));
    float dx1 = static_cast<float>(std::abs((a.x + a.width) - (b.x + b.width)));
    float dy1 = static_cast<float>(std::abs((a.y + a.height) - (b.y + b.height)));
    return std::max(std::max(dx0, dy0), std::max(dx1, dy1));
}

void decode_reference_candidates(const std::vector<float>& detection,
                                 const YoloV11SegConfig& config,
                                 const LetterboxInfo& letterbox,
                                 std::vector<ReferenceCandidate>& candidates)
{
    candidates.clear();
    for (int anchor = 0; anchor < kAnchors; ++anchor) {
        int best_class = -1;
        float best_score = 0.0f;
        for (int class_id = 0; class_id < config.class_count; ++class_id) {
            float score = detection_value_at(detection, 4 + class_id, anchor);
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (best_class < 0 || best_score < config.score_threshold) {
            continue;
        }

        float cx = detection_value_at(detection, 0, anchor);
        float cy = detection_value_at(detection, 1, anchor);
        float width = detection_value_at(detection, 2, anchor);
        float height = detection_value_at(detection, 3, anchor);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        float left = ((cx - 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float top = ((cy - 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        float right = ((cx + 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float bottom = ((cy + 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        cv::Rect box = clip_rect_to_image_ref(left,
                                              top,
                                              right,
                                              bottom,
                                              letterbox.source_width,
                                              letterbox.source_height);
        if (box.empty()) {
            continue;
        }

        ReferenceCandidate candidate;
        candidate.anchor = anchor;
        candidate.class_id = best_class;
        candidate.score = best_score;
        candidate.box = box;
        candidates.push_back(candidate);
    }
}

void run_reference_nms(const std::vector<ReferenceCandidate>& candidates,
                       const YoloV11SegConfig& config,
                       std::vector<int>& keep)
{
    keep.clear();
    std::vector<int> order(candidates.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&candidates](int a, int b) {
        return candidates[static_cast<size_t>(a)].score > candidates[static_cast<size_t>(b)].score;
    });

    while (!order.empty() && static_cast<int>(keep.size()) < config.max_segments) {
        int best = order.front();
        keep.push_back(best);

        std::vector<int> next;
        next.reserve(order.size());
        for (size_t i = 1; i < order.size(); ++i) {
            int idx = order[i];
            if (!config.agnostic &&
                candidates[static_cast<size_t>(idx)].class_id != candidates[static_cast<size_t>(best)].class_id) {
                next.push_back(idx);
                continue;
            }
            if (intersection_over_union_ref(candidates[static_cast<size_t>(best)].box,
                                            candidates[static_cast<size_t>(idx)].box) <= config.nms_threshold) {
                next.push_back(idx);
            }
        }
        order.swap(next);
    }
}

void build_reference_mask(const std::vector<float>& detection,
                          const std::vector<float>& prototype,
                          const YoloV11SegConfig& config,
                          const LetterboxInfo& letterbox,
                          const ReferenceCandidate& candidate,
                          Segmentation& segment)
{
    cv::Mat proto_prob(kPrototypeHeight, kPrototypeWidth, CV_32FC1);
    for (int y = 0; y < kPrototypeHeight; ++y) {
        float* row = proto_prob.ptr<float>(y);
        for (int x = 0; x < kPrototypeWidth; ++x) {
            float value = 0.0f;
            for (int coeff = 0; coeff < config.mask_coeff_count; ++coeff) {
                float coefficient = detection_value_at(detection, 4 + config.class_count + coeff, candidate.anchor);
                size_t prototype_offset = (static_cast<size_t>(coeff) * kPrototypeHeight +
                                           static_cast<size_t>(y)) *
                                              kPrototypeWidth +
                                          static_cast<size_t>(x);
                value += coefficient * prototype[prototype_offset];
            }
            row[x] = sigmoid_ref(value);
        }
    }

    cv::Mat model_mask;
    cv::resize(proto_prob,
               model_mask,
               cv::Size(letterbox.model_width, letterbox.model_height),
               0.0,
               0.0,
               cv::INTER_LINEAR);

    int valid_x = std::max(0, static_cast<int>(std::round(letterbox.pad_x)));
    int valid_y = std::max(0, static_cast<int>(std::round(letterbox.pad_y)));
    int valid_w = std::min(letterbox.resized_width, letterbox.model_width - valid_x);
    int valid_h = std::min(letterbox.resized_height, letterbox.model_height - valid_y);
    if (valid_w <= 0 || valid_h <= 0) {
        return;
    }

    cv::Rect valid_region(valid_x, valid_y, valid_w, valid_h);
    cv::Mat unpadded = model_mask(valid_region);
    cv::Mat source_prob;
    cv::resize(unpadded,
               source_prob,
               cv::Size(letterbox.source_width, letterbox.source_height),
               0.0,
               0.0,
               cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(source_prob, binary, config.mask_threshold, 255.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1);

    cv::Mat cropped = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Rect clipped_box = candidate.box & cv::Rect(0, 0, binary.cols, binary.rows);
    if (clipped_box.empty()) {
        return;
    }
    binary(clipped_box).copyTo(cropped(clipped_box));
    if (cv::countNonZero(cropped) == 0) {
        return;
    }

    segment.class_id = candidate.class_id;
    segment.confidence = candidate.score;
    segment.box = candidate.box;
    segment.mask_width = cropped.cols;
    segment.mask_height = cropped.rows;
    segment.mask.assign(cropped.datastart, cropped.dataend);
}

void build_reference_segments(const std::vector<float>& detection,
                              const std::vector<float>& prototype,
                              const YoloV11SegConfig& config,
                              const LetterboxInfo& letterbox,
                              std::vector<Segmentation>& reference)
{
    std::vector<ReferenceCandidate> candidates;
    std::vector<int> keep;
    decode_reference_candidates(detection, config, letterbox, candidates);
    run_reference_nms(candidates, config, keep);

    reference.clear();
    reference.reserve(keep.size());
    for (int kept_index : keep) {
        Segmentation segment;
        build_reference_mask(detection,
                             prototype,
                             config,
                             letterbox,
                             candidates[static_cast<size_t>(kept_index)],
                             segment);
        if (!segment.mask.empty()) {
            reference.push_back(std::move(segment));
        }
    }
}

bool mask_is_box_local(const Segmentation& segment)
{
    return segment.mask_width == segment.box.width &&
           segment.mask_height == segment.box.height &&
           segment.mask.size() == static_cast<size_t>(segment.box.width * segment.box.height);
}

uint8_t mask_value_at_source(const Segmentation& segment, int x, int y)
{
    if (segment.mask.empty() || segment.mask_width <= 0 || segment.mask_height <= 0) {
        return 0;
    }
    if (mask_is_box_local(segment)) {
        int local_x = x - segment.box.x;
        int local_y = y - segment.box.y;
        if (local_x < 0 || local_y < 0 ||
            local_x >= segment.mask_width || local_y >= segment.mask_height) {
            return 0;
        }
        return segment.mask[static_cast<size_t>(local_y) * static_cast<size_t>(segment.mask_width) +
                            static_cast<size_t>(local_x)];
    }
    if (x < 0 || y < 0 || x >= segment.mask_width || y >= segment.mask_height ||
        segment.mask.size() != static_cast<size_t>(segment.mask_width * segment.mask_height)) {
        return 0;
    }
    return segment.mask[static_cast<size_t>(y) * static_cast<size_t>(segment.mask_width) +
                        static_cast<size_t>(x)];
}

void comparison_canvas(const Segmentation& a,
                       const Segmentation& b,
                       int& width,
                       int& height,
                       cv::Rect& region)
{
    bool a_local = mask_is_box_local(a);
    bool b_local = mask_is_box_local(b);
    width = 0;
    height = 0;
    if (!a_local) {
        width = std::max(width, a.mask_width);
        height = std::max(height, a.mask_height);
    }
    if (!b_local) {
        width = std::max(width, b.mask_width);
        height = std::max(height, b.mask_height);
    }
    if (width > 0 && height > 0) {
        region = cv::Rect(0, 0, width, height);
        return;
    }

    int x0 = std::min(a.box.x, b.box.x);
    int y0 = std::min(a.box.y, b.box.y);
    int x1 = std::max(a.box.x + a.box.width, b.box.x + b.box.width);
    int y1 = std::max(a.box.y + a.box.height, b.box.y + b.box.height);
    width = std::max(0, x1);
    height = std::max(0, y1);
    region = cv::Rect(std::max(0, x0),
                      std::max(0, y0),
                      std::max(0, x1 - std::max(0, x0)),
                      std::max(0, y1 - std::max(0, y0)));
}

float mask_iou(const Segmentation& a, const Segmentation& b)
{
    if (a.mask.empty() || b.mask.empty()) {
        return 0.0f;
    }

    int canvas_width = 0;
    int canvas_height = 0;
    cv::Rect region;
    comparison_canvas(a, b, canvas_width, canvas_height, region);
    int intersection = 0;
    int union_count = 0;
    for (int y = region.y; y < region.y + region.height; ++y) {
        for (int x = region.x; x < region.x + region.width; ++x) {
            bool av = mask_value_at_source(a, x, y) != 0;
            bool bv = mask_value_at_source(b, x, y) != 0;
            if (av && bv) {
                ++intersection;
            }
            if (av || bv) {
                ++union_count;
            }
        }
    }
    return union_count > 0 ? static_cast<float>(intersection) / static_cast<float>(union_count) : 1.0f;
}

float mask_diff_ratio(const Segmentation& a, const Segmentation& b)
{
    if (a.mask.empty() || b.mask.empty()) {
        return 1.0f;
    }

    int canvas_width = 0;
    int canvas_height = 0;
    cv::Rect region;
    comparison_canvas(a, b, canvas_width, canvas_height, region);
    int diff = 0;
    for (int y = region.y; y < region.y + region.height; ++y) {
        for (int x = region.x; x < region.x + region.width; ++x) {
            if ((mask_value_at_source(a, x, y) != 0) != (mask_value_at_source(b, x, y) != 0)) {
                ++diff;
            }
        }
    }
    int pixels = region.width * region.height;
    return pixels > 0 ? static_cast<float>(diff) / static_cast<float>(pixels) : 0.0f;
}

AccuracySummary compare_accuracy(const std::vector<Segmentation>& output,
                                 const std::vector<Segmentation>& reference,
                                 const AccuracyThresholds& thresholds)
{
    AccuracySummary summary;
    summary.output_segments = static_cast<int>(output.size());
    summary.reference_segments = static_cast<int>(reference.size());
    summary.matched_segments = std::min(summary.output_segments, summary.reference_segments);
    summary.extra_output_segments = std::max(0, summary.output_segments - summary.reference_segments);
    summary.missing_reference_segments = std::max(0, summary.reference_segments - summary.output_segments);

    if (summary.matched_segments == 0) {
        summary.min_box_iou = reference.empty() && output.empty() ? 1.0f : 0.0f;
        summary.min_mask_iou = reference.empty() && output.empty() ? 1.0f : 0.0f;
    }

    for (int i = 0; i < summary.matched_segments; ++i) {
        const Segmentation& out = output[static_cast<size_t>(i)];
        const Segmentation& ref = reference[static_cast<size_t>(i)];
        float box_iou = intersection_over_union_ref(out.box, ref.box);
        float box_diff = max_box_coordinate_diff(out.box, ref.box);
        float current_mask_iou = mask_iou(out, ref);
        float current_mask_diff_ratio = mask_diff_ratio(out, ref);

        summary.min_box_iou = std::min(summary.min_box_iou, box_iou);
        summary.max_box_diff_px = std::max(summary.max_box_diff_px, box_diff);
        summary.min_mask_iou = std::min(summary.min_mask_iou, current_mask_iou);
        summary.max_mask_diff_ratio = std::max(summary.max_mask_diff_ratio, current_mask_diff_ratio);

        bool metadata_match = out.class_id == ref.class_id &&
                              std::fabs(out.confidence - ref.confidence) <= thresholds.max_score_diff;
        bool match_passed = metadata_match &&
                            box_iou >= thresholds.min_box_iou &&
                            box_diff <= thresholds.max_box_diff_px &&
                            current_mask_iou >= thresholds.min_mask_iou &&
                            current_mask_diff_ratio <= thresholds.max_mask_diff_ratio;
        summary.passed = summary.passed && match_passed;

        std::cout << std::fixed << std::setprecision(6)
                  << "  segment[" << i << "]: "
                  << "class(out/ref)=" << out.class_id << "/" << ref.class_id
                  << ", score(out/ref)=" << out.confidence << "/" << ref.confidence
                  << ", box_iou=" << box_iou
                  << ", max_box_diff_px=" << box_diff
                  << ", mask_iou=" << current_mask_iou
                  << ", mask_diff_ratio=" << current_mask_diff_ratio << "\n";
    }

    if (summary.extra_output_segments != 0 || summary.missing_reference_segments != 0) {
        summary.passed = false;
    }
    if (summary.min_box_iou < thresholds.min_box_iou ||
        summary.max_box_diff_px > thresholds.max_box_diff_px ||
        summary.min_mask_iou < thresholds.min_mask_iou ||
        summary.max_mask_diff_ratio > thresholds.max_mask_diff_ratio) {
        summary.passed = false;
    }
    return summary;
}

void generate_synthetic_outputs(std::vector<float>& detection, std::vector<float>& prototype)
{
    std::fill(detection.begin(), detection.end(), 0.0f);
    std::fill(prototype.begin(), prototype.end(), -2.0f);

    const int anchor0 = 0;
    set_detection_value(detection, 0, anchor0, 320.0f);
    set_detection_value(detection, 1, anchor0, 320.0f);
    set_detection_value(detection, 2, anchor0, 220.0f);
    set_detection_value(detection, 3, anchor0, 220.0f);
    set_detection_value(detection, 4, anchor0, 0.95f);
    set_detection_value(detection, 4 + kClassCount, anchor0, 4.0f);

    const int anchor1 = 100;
    set_detection_value(detection, 0, anchor1, 455.0f);
    set_detection_value(detection, 1, anchor1, 320.0f);
    set_detection_value(detection, 2, anchor1, 160.0f);
    set_detection_value(detection, 3, anchor1, 180.0f);
    set_detection_value(detection, 5, anchor1, 0.82f);
    set_detection_value(detection, 4 + kClassCount + 1, anchor1, 4.0f);

    for (int y = 0; y < kPrototypeHeight; ++y) {
        for (int x = 0; x < kPrototypeWidth; ++x) {
            float dx = (static_cast<float>(x) - 80.0f) / 38.0f;
            float dy = (static_cast<float>(y) - 80.0f) / 38.0f;
            float circle = (dx * dx + dy * dy) < 1.0f ? 2.0f : -2.0f;
            prototype[static_cast<size_t>(y) * kPrototypeWidth + x] = circle;

            float rx = std::fabs(static_cast<float>(x) - 114.0f);
            float ry = std::fabs(static_cast<float>(y) - 80.0f);
            float rect = (rx < 24.0f && ry < 32.0f) ? 2.0f : -2.0f;
            prototype[(static_cast<size_t>(1) * kPrototypeHeight + y) * kPrototypeWidth + x] = rect;
        }
    }
}

YoloV11SegConfig make_config(float score_threshold,
                             float nms_threshold,
                             float mask_threshold,
                             int max_segments)
{
    YoloV11SegConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.score_threshold = score_threshold;
    config.nms_threshold = nms_threshold;
    config.mask_threshold = mask_threshold;
    config.class_count = kClassCount;
    config.mask_coeff_count = kMaskCoeffCount;
    config.max_segments = max_segments;
    config.detection_dimensions.nbDims = 3;
    config.detection_dimensions.d[0] = 1;
    config.detection_dimensions.d[1] = kDetectionChannels;
    config.detection_dimensions.d[2] = kAnchors;
    config.detection_elements = static_cast<size_t>(kDetectionChannels) * kAnchors;
    config.prototype_dimensions.nbDims = 4;
    config.prototype_dimensions.d[0] = 1;
    config.prototype_dimensions.d[1] = kMaskCoeffCount;
    config.prototype_dimensions.d[2] = kPrototypeHeight;
    config.prototype_dimensions.d[3] = kPrototypeWidth;
    config.prototype_elements = static_cast<size_t>(kMaskCoeffCount) * kPrototypeHeight * kPrototypeWidth;
    return config;
}

LetterboxInfo make_letterbox(int source_width, int source_height)
{
    LetterboxInfo letterbox;
    letterbox.source_width = source_width;
    letterbox.source_height = source_height;
    letterbox.model_width = 640;
    letterbox.model_height = 640;
    float scale = std::min(640.0f / static_cast<float>(source_width),
                           640.0f / static_cast<float>(source_height));
    letterbox.scale = scale;
    letterbox.resized_width = static_cast<int>(std::round(static_cast<float>(source_width) * scale));
    letterbox.resized_height = static_cast<int>(std::round(static_cast<float>(source_height) * scale));
    letterbox.pad_x = static_cast<float>(640 - letterbox.resized_width) * 0.5f;
    letterbox.pad_y = static_cast<float>(640 - letterbox.resized_height) * 0.5f;
    return letterbox;
}

void print_profile(const YoloV11SegPostprocessProfile& profile,
                   double outer_ms,
                   int segment_count)
{
    std::cout << std::fixed << std::setprecision(3)
              << "Postprocess timing:\n"
              << "  split_ms: " << profile.split_ms << " ms\n"
              << "    split_d2s_ms: " << profile.split_d2s_ms << " ms\n"
              << "    split_kernel_ms: " << profile.split_kernel_ms << " ms\n"
              << "    split_s2d_ms: " << profile.split_s2d_ms << " ms\n"
              << "    split_d2h_ms: " << profile.split_d2h_ms << " ms\n"
              << "  output_d2h_ms: " << profile.device_to_host_ms << " ms\n"
              << "  output_d2h_bytes: " << profile.device_to_host_bytes
              << " bytes (" << bytes_to_kib(profile.device_to_host_bytes) << " KiB)\n"
              << "  decode_nms_ms: " << profile.decode_nms_ms << " ms\n"
              << "    candidate_decode_device_ms: " << profile.candidate_decode_ms << " ms\n"
              << "    candidate_decode_wall_ms: " << profile.candidate_decode_wall_ms << " ms\n"
              << "    candidate_return_wall_ms: " << profile.split_d2h_ms << " ms\n"
              << "    candidate_return_device_ms: " << profile.candidate_return_device_ms << " ms\n"
              << "    candidate_runtime_overhead_ms: " << profile.candidate_runtime_overhead_ms << " ms\n"
              << "    candidate_sort_ms: " << profile.candidate_sort_ms << " ms\n"
              << "    nms_ms: " << profile.nms_ms << " ms\n"
              << "    candidate_count: " << profile.candidate_count << "\n"
              << "  mask_decode_ms: " << profile.mask_decode_ms << " ms\n"
              << "    mask_setup_copy_ms: " << profile.mask_setup_copy_ms << " ms\n"
              << "      prototype_d2s_ms: " << profile.mask_prototype_d2s_ms << " ms\n"
              << "      selected_metadata_h2s_ms: " << profile.mask_metadata_h2s_ms << " ms\n"
              << "    mask_coeff_gather_ms: " << profile.mask_coeff_gather_ms << " ms\n"
              << "    mask_proto_combine_ms: " << profile.mask_proto_combine_ms << " ms\n"
              << "    mask_lowres_threshold_ms: " << profile.mask_lowres_threshold_ms << " ms\n"
              << "    mask_box_copy_ms: " << profile.mask_box_copy_ms << " ms\n"
              << "    mask_source_resize_crop_ms: " << profile.mask_source_resize_crop_ms << " ms\n"
              << "    mask_fused_kernel_ms: " << profile.mask_fused_kernel_ms << " ms\n"
              << "    mask_device_pipeline_ms: " << profile.mask_device_pipeline_ms << " ms\n"
              << "    mask_runtime_overhead_ms: " << profile.mask_runtime_overhead_ms << " ms\n"
              << "    mask_source_copy_ms: " << profile.mask_source_copy_ms << " ms\n"
              << "    mask_build_cpu_ms: " << profile.mask_build_cpu_ms << " ms\n"
              << "    full_proto_pixels: " << profile.mask_full_proto_pixels << "\n"
              << "    sparse_proto_pixels: " << profile.mask_sparse_proto_pixels << "\n"
              << "    sparse_proto_coverage: " << profile.mask_sparse_proto_coverage << "\n"
              << "    output_roi_pixels: " << profile.mask_output_roi_pixels << "\n"
              << "  profile_total_ms: " << profile.total_ms << " ms\n"
              << "  outer_call_ms: " << outer_ms << " ms\n"
              << "  output_segments: " << segment_count << std::endl;
}
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("yolov11-seg postprocess test demo", "0.1", argparse::default_arguments::help);
    program.add_argument("--detection-bin")
            .default_value(std::string(""))
            .help("optional raw f32 detection output file, shape [1,116,8400].");
    program.add_argument("--prototype-bin")
            .default_value(std::string(""))
            .help("optional raw f32 prototype output file, shape [1,32,160,160].");
    program.add_argument("--loop")
            .default_value(10)
            .help("measured postprocess loop count after warmup.")
            .scan<'i', int>();
    program.add_argument("--warmup")
            .default_value(1)
            .help("warmup loop count.")
            .scan<'i', int>();
    program.add_argument("--source-width")
            .default_value(640)
            .scan<'i', int>();
    program.add_argument("--source-height")
            .default_value(640)
            .scan<'i', int>();
    program.add_argument("--score-threshold")
            .default_value(0.25f)
            .scan<'g', float>();
    program.add_argument("--nms-threshold")
            .default_value(0.45f)
            .scan<'g', float>();
    program.add_argument("--mask-threshold")
            .default_value(0.50f)
            .scan<'g', float>();
    program.add_argument("--max-segments")
            .default_value(100)
            .scan<'i', int>();
    program.add_argument("--accuracy-check")
            .help("compare RPP postprocess output against CPU reference and return non-zero on mismatch.")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("--min-box-iou")
            .default_value(0.98f)
            .help("minimum accepted box IoU in --accuracy-check mode.")
            .scan<'g', float>();
    program.add_argument("--max-box-diff")
            .default_value(1.0f)
            .help("maximum accepted restored box coordinate difference in pixels.")
            .scan<'g', float>();
    program.add_argument("--min-mask-iou")
            .default_value(0.90f)
            .help("minimum accepted source-size binary mask IoU in --accuracy-check mode.")
            .scan<'g', float>();
    program.add_argument("--max-mask-diff-ratio")
            .default_value(0.15f)
            .help("maximum accepted binary mask pixel mismatch ratio.")
            .scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program << std::endl;
        return EXIT_FAILURE;
    }

    int loop = program.get<int>("--loop");
    int warmup = program.get<int>("--warmup");
    int source_width = program.get<int>("--source-width");
    int source_height = program.get<int>("--source-height");
    bool accuracy_check = program.get<bool>("--accuracy-check");
    if (loop <= 0 || warmup < 0 || source_width <= 0 || source_height <= 0) {
        std::cerr << "Invalid loop/warmup/source dimensions." << std::endl;
        return EXIT_FAILURE;
    }
    AccuracyThresholds accuracy_thresholds;
    accuracy_thresholds.min_box_iou = program.get<float>("--min-box-iou");
    accuracy_thresholds.max_box_diff_px = program.get<float>("--max-box-diff");
    accuracy_thresholds.min_mask_iou = program.get<float>("--min-mask-iou");
    accuracy_thresholds.max_mask_diff_ratio = program.get<float>("--max-mask-diff-ratio");
    if (accuracy_thresholds.min_box_iou < 0.0f || accuracy_thresholds.min_box_iou > 1.0f ||
        accuracy_thresholds.max_box_diff_px < 0.0f ||
        accuracy_thresholds.min_mask_iou < 0.0f || accuracy_thresholds.min_mask_iou > 1.0f ||
        accuracy_thresholds.max_mask_diff_ratio < 0.0f || accuracy_thresholds.max_mask_diff_ratio > 1.0f) {
        std::cerr << "Invalid accuracy thresholds." << std::endl;
        return EXIT_FAILURE;
    }

    YoloV11SegConfig config = make_config(program.get<float>("--score-threshold"),
                                          program.get<float>("--nms-threshold"),
                                          program.get<float>("--mask-threshold"),
                                          program.get<int>("--max-segments"));
    std::vector<float> detection(config.detection_elements, 0.0f);
    std::vector<float> prototype(config.prototype_elements, 0.0f);

    std::string detection_bin = program.get<std::string>("--detection-bin");
    std::string prototype_bin = program.get<std::string>("--prototype-bin");
    if (!detection_bin.empty() || !prototype_bin.empty()) {
        if (detection_bin.empty() || prototype_bin.empty()) {
            std::cerr << "--detection-bin and --prototype-bin must be provided together." << std::endl;
            return EXIT_FAILURE;
        }
        if (!read_binary_f32(detection_bin, detection) || !read_binary_f32(prototype_bin, prototype)) {
            return EXIT_FAILURE;
        }
        if (detection.size() != config.detection_elements || prototype.size() != config.prototype_elements) {
            std::cerr << "Input binary element count does not match expected YOLOv11-SEG shapes." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        generate_synthetic_outputs(detection, prototype);
    }

    PerfTraceSession trace_session("yolov11_seg_postprocess_test", kPerfTraceEnabled, kPerfTraceDir);
    trace_session.enableDriverTrace(0);

    void* detection_device = nullptr;
    void* prototype_device = nullptr;
    const size_t detection_bytes = detection.size() * sizeof(float);
    const size_t prototype_bytes = prototype.size() * sizeof(float);
    if (!check_rt(rtMalloc(&detection_device, detection_bytes), "rtMalloc detection") ||
        !check_rt(rtMalloc(&prototype_device, prototype_bytes), "rtMalloc prototype") ||
        !check_rt(rtMemcpy(detection_device, detection.data(), detection_bytes, rtMemcpyHostToDevice),
                  "rtMemcpy detection H2D") ||
        !check_rt(rtMemcpy(prototype_device, prototype.data(), prototype_bytes, rtMemcpyHostToDevice),
                  "rtMemcpy prototype H2D")) {
        if (detection_device != nullptr) {
            rtFree(detection_device);
        }
        if (prototype_device != nullptr) {
            rtFree(prototype_device);
        }
        return EXIT_FAILURE;
    }

    YoloV11SegPostprocessor postprocessor;
    if (!postprocessor.init(config)) {
        rtFree(detection_device);
        rtFree(prototype_device);
        return EXIT_FAILURE;
    }

    LetterboxInfo letterbox = make_letterbox(source_width, source_height);
    std::vector<Segmentation> segments;
    YoloV11SegPostprocessProfile profile;
    for (int i = 0; i < warmup; ++i) {
        if (!postprocessor.run(detection_device, prototype_device, letterbox, segments, nullptr, &profile)) {
            rtFree(detection_device);
            rtFree(prototype_device);
            return EXIT_FAILURE;
        }
    }

    YoloV11SegPostprocessProfile total_profile;
    double total_outer_ms = 0.0;
    for (int i = 0; i < loop; ++i) {
        auto start = Clock::now();
        if (!postprocessor.run(detection_device, prototype_device, letterbox, segments, nullptr, &profile)) {
            rtFree(detection_device);
            rtFree(prototype_device);
            return EXIT_FAILURE;
        }
        auto stop = Clock::now();
        total_outer_ms += elapsed_ms(start, stop);
        total_profile.device_to_host_ms += profile.device_to_host_ms;
        total_profile.device_to_host_bytes += profile.device_to_host_bytes;
        total_profile.split_ms += profile.split_ms;
        total_profile.split_d2s_ms += profile.split_d2s_ms;
        total_profile.split_kernel_ms += profile.split_kernel_ms;
        total_profile.split_s2d_ms += profile.split_s2d_ms;
        total_profile.split_d2h_ms += profile.split_d2h_ms;
        total_profile.decode_nms_ms += profile.decode_nms_ms;
        total_profile.candidate_decode_ms += profile.candidate_decode_ms;
        total_profile.candidate_decode_wall_ms += profile.candidate_decode_wall_ms;
        total_profile.candidate_return_device_ms += profile.candidate_return_device_ms;
        total_profile.candidate_runtime_overhead_ms += profile.candidate_runtime_overhead_ms;
        total_profile.candidate_sort_ms += profile.candidate_sort_ms;
        total_profile.nms_ms += profile.nms_ms;
        total_profile.candidate_count += profile.candidate_count;
        total_profile.mask_decode_ms += profile.mask_decode_ms;
        total_profile.mask_setup_copy_ms += profile.mask_setup_copy_ms;
        total_profile.mask_prototype_d2s_ms += profile.mask_prototype_d2s_ms;
        total_profile.mask_metadata_h2s_ms += profile.mask_metadata_h2s_ms;
        total_profile.mask_coeff_gather_ms += profile.mask_coeff_gather_ms;
        total_profile.mask_proto_combine_ms += profile.mask_proto_combine_ms;
        total_profile.mask_lowres_threshold_ms += profile.mask_lowres_threshold_ms;
        total_profile.mask_box_copy_ms += profile.mask_box_copy_ms;
        total_profile.mask_source_resize_crop_ms += profile.mask_source_resize_crop_ms;
        total_profile.mask_fused_kernel_ms += profile.mask_fused_kernel_ms;
        total_profile.mask_device_pipeline_ms += profile.mask_device_pipeline_ms;
        total_profile.mask_runtime_overhead_ms += profile.mask_runtime_overhead_ms;
        total_profile.mask_source_copy_ms += profile.mask_source_copy_ms;
        total_profile.mask_build_cpu_ms += profile.mask_build_cpu_ms;
        total_profile.mask_full_proto_pixels += profile.mask_full_proto_pixels;
        total_profile.mask_sparse_proto_pixels += profile.mask_sparse_proto_pixels;
        total_profile.mask_output_roi_pixels += profile.mask_output_roi_pixels;
        total_profile.mask_sparse_proto_coverage += profile.mask_sparse_proto_coverage;
        total_profile.total_ms += profile.total_ms;
    }

    const double inv_loop = 1.0 / static_cast<double>(loop);
    YoloV11SegPostprocessProfile average_profile;
    average_profile.device_to_host_ms = total_profile.device_to_host_ms * inv_loop;
    average_profile.device_to_host_bytes = total_profile.device_to_host_bytes / static_cast<size_t>(loop);
    average_profile.split_ms = total_profile.split_ms * inv_loop;
    average_profile.split_d2s_ms = total_profile.split_d2s_ms * inv_loop;
    average_profile.split_kernel_ms = total_profile.split_kernel_ms * inv_loop;
    average_profile.split_s2d_ms = total_profile.split_s2d_ms * inv_loop;
    average_profile.split_d2h_ms = total_profile.split_d2h_ms * inv_loop;
    average_profile.decode_nms_ms = total_profile.decode_nms_ms * inv_loop;
    average_profile.candidate_decode_ms = total_profile.candidate_decode_ms * inv_loop;
    average_profile.candidate_decode_wall_ms = total_profile.candidate_decode_wall_ms * inv_loop;
    average_profile.candidate_return_device_ms = total_profile.candidate_return_device_ms * inv_loop;
    average_profile.candidate_runtime_overhead_ms = total_profile.candidate_runtime_overhead_ms * inv_loop;
    average_profile.candidate_sort_ms = total_profile.candidate_sort_ms * inv_loop;
    average_profile.nms_ms = total_profile.nms_ms * inv_loop;
    average_profile.candidate_count = total_profile.candidate_count / static_cast<size_t>(loop);
    average_profile.mask_decode_ms = total_profile.mask_decode_ms * inv_loop;
    average_profile.mask_setup_copy_ms = total_profile.mask_setup_copy_ms * inv_loop;
    average_profile.mask_prototype_d2s_ms = total_profile.mask_prototype_d2s_ms * inv_loop;
    average_profile.mask_metadata_h2s_ms = total_profile.mask_metadata_h2s_ms * inv_loop;
    average_profile.mask_coeff_gather_ms = total_profile.mask_coeff_gather_ms * inv_loop;
    average_profile.mask_proto_combine_ms = total_profile.mask_proto_combine_ms * inv_loop;
    average_profile.mask_lowres_threshold_ms = total_profile.mask_lowres_threshold_ms * inv_loop;
    average_profile.mask_box_copy_ms = total_profile.mask_box_copy_ms * inv_loop;
    average_profile.mask_source_resize_crop_ms = total_profile.mask_source_resize_crop_ms * inv_loop;
    average_profile.mask_fused_kernel_ms = total_profile.mask_fused_kernel_ms * inv_loop;
    average_profile.mask_device_pipeline_ms = total_profile.mask_device_pipeline_ms * inv_loop;
    average_profile.mask_runtime_overhead_ms = total_profile.mask_runtime_overhead_ms * inv_loop;
    average_profile.mask_source_copy_ms = total_profile.mask_source_copy_ms * inv_loop;
    average_profile.mask_build_cpu_ms = total_profile.mask_build_cpu_ms * inv_loop;
    average_profile.mask_full_proto_pixels =
        total_profile.mask_full_proto_pixels / static_cast<size_t>(loop);
    average_profile.mask_sparse_proto_pixels =
        total_profile.mask_sparse_proto_pixels / static_cast<size_t>(loop);
    average_profile.mask_output_roi_pixels =
        total_profile.mask_output_roi_pixels / static_cast<size_t>(loop);
    average_profile.mask_sparse_proto_coverage =
        total_profile.mask_sparse_proto_coverage * inv_loop;
    average_profile.total_ms = total_profile.total_ms * inv_loop;

    std::cout << "YOLOv11-SEG postprocess standalone test\n"
              << "  detection_elements: " << config.detection_elements << "\n"
              << "  prototype_elements: " << config.prototype_elements << "\n"
              << "  source_size: " << source_width << "x" << source_height << "\n"
              << "  warmup: " << warmup << "\n"
              << "  loop: " << loop << "\n"
              << "  accuracy_check: " << (accuracy_check ? "enabled" : "disabled") << "\n";
    print_profile(average_profile, total_outer_ms * inv_loop, static_cast<int>(segments.size()));

    bool accuracy_passed = true;
    if (accuracy_check) {
        auto reference_start = Clock::now();
        std::vector<Segmentation> reference_segments;
        build_reference_segments(detection, prototype, config, letterbox, reference_segments);
        double reference_ms = elapsed_ms(reference_start, Clock::now());

        std::cout << std::fixed << std::setprecision(3)
                  << "Accuracy thresholds:\n"
                  << "  max_score_diff: " << accuracy_thresholds.max_score_diff << "\n"
                  << "  min_box_iou: " << accuracy_thresholds.min_box_iou << "\n"
                  << "  max_box_diff_px: " << accuracy_thresholds.max_box_diff_px << "\n"
                  << "  min_mask_iou: " << accuracy_thresholds.min_mask_iou << "\n"
                  << "  max_mask_diff_ratio: " << accuracy_thresholds.max_mask_diff_ratio << "\n"
                  << "CPU reference:\n"
                  << "  reference_ms: " << reference_ms << " ms\n"
                  << "  reference_segments: " << reference_segments.size() << "\n"
                  << "Accuracy detail:\n";

        AccuracySummary summary = compare_accuracy(segments, reference_segments, accuracy_thresholds);
        accuracy_passed = summary.passed;
        std::cout << std::fixed << std::setprecision(6)
                  << "Accuracy summary:\n"
                  << "  output_segments: " << summary.output_segments << "\n"
                  << "  reference_segments: " << summary.reference_segments << "\n"
                  << "  matched_segments: " << summary.matched_segments << "\n"
                  << "  extra_output_segments: " << summary.extra_output_segments << "\n"
                  << "  missing_reference_segments: " << summary.missing_reference_segments << "\n"
                  << "  min_box_iou: " << summary.min_box_iou << "\n"
                  << "  max_box_diff_px: " << summary.max_box_diff_px << "\n"
                  << "  min_mask_iou: " << summary.min_mask_iou << "\n"
                  << "  max_mask_diff_ratio: " << summary.max_mask_diff_ratio << "\n"
                  << "  result: " << (summary.passed ? "PASS" : "FAIL") << std::endl;
    }

    rtFree(detection_device);
    rtFree(prototype_device);
    return accuracy_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
