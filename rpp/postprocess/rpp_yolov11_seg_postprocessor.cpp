/**
 * @file rpp_yolov11_seg_postprocessor.cpp
 * @brief YOLOv11-SEG output copy, decode, NMS, and mask reconstruction module.
 */
#include "rpp_yolov11_seg_postprocessor.h"

#include "perf_trace_session.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{
using ProfileClock = std::chrono::high_resolution_clock;

double elapsed_ms(ProfileClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

void reset_profile(YoloV11SegPostprocessProfile* profile)
{
    if (profile != nullptr) {
        *profile = YoloV11SegPostprocessProfile{};
    }
}

bool check_rt(rtError_t status, const char* what)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << what << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

std::vector<int> compact_dims(const infer1::Dims& dims)
{
    std::vector<int> out;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) {
            out.push_back(dims.d[i]);
        }
    }
    return out;
}

float sigmoid(float value)
{
    if (value >= 0.0f) {
        float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(value);
    return z / (1.0f + z);
}

cv::Rect clip_rect_to_image(float left, float top, float right, float bottom, int width, int height)
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

float intersection_over_union(const cv::Rect& a, const cv::Rect& b)
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
}

YoloV11SegPostprocessor::~YoloV11SegPostprocessor()
{
    releaseMaskDecodeBuffers();
}

bool YoloV11SegPostprocessor::init(const YoloV11SegConfig& config)
{
    if (config.detection_elements == 0 || config.prototype_elements == 0 ||
        config.score_threshold < 0.0f || config.nms_threshold < 0.0f ||
        config.nms_threshold > 1.0f || config.mask_threshold < 0.0f ||
        config.mask_threshold > 1.0f) {
        std::cerr << "Invalid YOLOv11-SEG postprocess configuration." << std::endl;
        return false;
    }

    config_ = config;
    det_channels_ = 4 + config_.class_count;
    total_head_channels_ = det_channels_ + config_.mask_coeff_count;
    if (!resolveDetectionLayout() || !resolvePrototypeLayout()) {
        std::cerr << "Unsupported YOLOv11-SEG output dimensions." << std::endl;
        return false;
    }

    host_det_head_.assign(static_cast<size_t>(det_channels_) *
                          static_cast<size_t>(detection_layout_.anchors), 0.0f);
    host_coeff_head_.assign(static_cast<size_t>(config_.mask_coeff_count) *
                            static_cast<size_t>(detection_layout_.anchors), 0.0f);
    host_prototype_.assign(config_.prototype_elements, 0.0f);
    initialized_ = true;
    return true;
}

bool YoloV11SegPostprocessor::run(const void* detection_output_device,
                                  const void* prototype_output_device,
                                  const LetterboxInfo& letterbox,
                                  std::vector<Segmentation>& segments,
                                  rtStream_t stream,
                                  YoloV11SegPostprocessProfile* profile)
{
    (void)stream;
    reset_profile(profile);
    segments.clear();

    if (!initialized_ || detection_output_device == nullptr || prototype_output_device == nullptr) {
        std::cerr << "YOLOv11-SEG postprocessor has not been initialized." << std::endl;
        return false;
    }
    if (letterbox.scale <= 0.0f || letterbox.source_width <= 0 || letterbox.source_height <= 0) {
        std::cerr << "Invalid letterbox metadata for YOLOv11-SEG postprocess." << std::endl;
        return false;
    }

    auto total_start = ProfileClock::now();
    {
        PERF_SCOPE_CATE("postprocess_split_output", "postprocess");
        if (!runRppSplitOutput(detection_output_device, profile)) {
            return false;
        }
    }

    auto stage_start = ProfileClock::now();
    std::vector<Candidate> candidates;
    {
        PERF_SCOPE_CATE("postprocess_decode_candidates_cpu", "postprocess");
        decodeCandidates(letterbox, candidates);
    }
    std::vector<int> keep;
    {
        PERF_SCOPE_CATE("postprocess_nms_cpu", "postprocess");
        runNms(candidates, keep);
    }
    if (profile != nullptr) {
        profile->decode_nms_ms = elapsed_ms(stage_start);
    }

    stage_start = ProfileClock::now();
    {
        PERF_SCOPE_CATE("postprocess_mask_decode_rpp", "postprocess");
        if (runRppMaskDecode(prototype_output_device, letterbox, candidates, keep, segments, profile)) {
            if (profile != nullptr) {
                profile->total_ms = elapsed_ms(total_start);
            }
            return true;
        }
    }

    {
        PERF_SCOPE_CATE("postprocess_mask_decode_cpu_fallback", "postprocess");
    }
    size_t prototype_bytes = config_.prototype_elements * sizeof(float);
    stage_start = ProfileClock::now();
    if (!check_rt(rtMemcpy(host_prototype_.data(), prototype_output_device, prototype_bytes, rtMemcpyDeviceToHost),
                  "rtMemcpy YOLOv11-SEG prototype D2H")) {
        return false;
    }
    if (profile != nullptr) {
        profile->device_to_host_ms += elapsed_ms(stage_start);
        profile->device_to_host_bytes += prototype_bytes;
    }

    segments.reserve(keep.size());
    {
        PERF_SCOPE_CATE("postprocess_build_masks_cpu_fallback", "postprocess");
        for (int kept_index : keep) {
            if (static_cast<int>(segments.size()) >= config_.max_segments) {
                break;
            }
            Segmentation segment;
            const Candidate& candidate = candidates[static_cast<size_t>(kept_index)];
            segment.class_id = candidate.class_id;
            segment.confidence = candidate.score;
            segment.box = candidate.box;
            buildMask(candidate, letterbox, segment);
            if (!segment.mask.empty()) {
                segments.push_back(std::move(segment));
            }
        }
    }
    if (profile != nullptr) {
        profile->mask_decode_ms = elapsed_ms(stage_start);
        profile->total_ms = elapsed_ms(total_start);
    }

    if (std::getenv("YOLO_DEBUG_POST") != nullptr) {
        std::cout << "YOLOv11-SEG postprocess debug: mode=cpu_seg"
                  << ", channels=" << detection_layout_.channels
                  << ", anchors=" << detection_layout_.anchors
                  << ", proto=" << prototype_channels_ << "x"
                  << prototype_height_ << "x" << prototype_width_
                  << ", candidates=" << candidates.size()
                  << ", kept=" << segments.size()
                  << std::endl;
    }
    return true;
}

bool YoloV11SegPostprocessor::resolveDetectionLayout()
{
    detection_layout_ = OutputLayout{};
    detection_layout_.dims = compact_dims(config_.detection_dimensions);
    if (detection_layout_.dims.size() < 2) {
        return false;
    }

    const int expected_channels = total_head_channels_;
    int feature_axis = -1;
    for (int i = 0; i < static_cast<int>(detection_layout_.dims.size()); ++i) {
        if (detection_layout_.dims[static_cast<size_t>(i)] == expected_channels) {
            feature_axis = i;
            break;
        }
    }
    if (feature_axis < 0) {
        return false;
    }

    detection_layout_.feature_axis = feature_axis;
    detection_layout_.channels = expected_channels;
    detection_layout_.strides.assign(detection_layout_.dims.size(), 1U);
    size_t volume = 1U;
    for (int i = static_cast<int>(detection_layout_.dims.size()) - 1; i >= 0; --i) {
        detection_layout_.strides[static_cast<size_t>(i)] = volume;
        volume *= static_cast<size_t>(detection_layout_.dims[static_cast<size_t>(i)]);
    }
    if (volume != config_.detection_elements) {
        return false;
    }
    size_t anchors = volume / static_cast<size_t>(detection_layout_.channels);
    if (anchors == 0 || anchors > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    detection_layout_.anchors = static_cast<int>(anchors);

    std::vector<int> tensor_dims = detection_layout_.dims;
    if (tensor_dims.size() == 3 && tensor_dims[0] == 1) {
        tensor_dims.erase(tensor_dims.begin());
    }
    split_input_channels_first_ = tensor_dims.size() == 2 && tensor_dims[0] == expected_channels;
    return true;
}

bool YoloV11SegPostprocessor::resolvePrototypeLayout()
{
    std::vector<int> dims = compact_dims(config_.prototype_dimensions);
    if (dims.size() == 4 && dims[0] == 1) {
        dims.erase(dims.begin());
    }
    if (dims.size() != 3) {
        return false;
    }
    prototype_channels_ = dims[0];
    prototype_height_ = dims[1];
    prototype_width_ = dims[2];
    return prototype_channels_ == config_.mask_coeff_count &&
           prototype_height_ > 0 && prototype_width_ > 0;
}

bool YoloV11SegPostprocessor::runRppSplitOutput(const void* detection_output_device,
                                                YoloV11SegPostprocessProfile* profile)
{
    if (detection_output_device == nullptr) {
        return false;
    }

    size_t input_bytes = config_.detection_elements * sizeof(float);
    size_t det_bytes = static_cast<size_t>(det_channels_) *
                       static_cast<size_t>(detection_layout_.anchors) *
                       sizeof(float);
    size_t coeff_bytes = static_cast<size_t>(config_.mask_coeff_count) *
                         static_cast<size_t>(detection_layout_.anchors) *
                         sizeof(float);

    if (!ensureSramBuffer(&split_input_sram_, &split_input_sram_capacity_, input_bytes) ||
        !ensureDeviceBuffer(&det_head_device_, &det_head_device_capacity_, det_bytes) ||
        !ensureSramBuffer(&det_head_sram_, &det_head_sram_capacity_, det_bytes) ||
        !ensureDeviceBuffer(&coeff_head_device_, &coeff_head_device_capacity_, coeff_bytes) ||
        !ensureSramBuffer(&coeff_head_sram_, &coeff_head_sram_capacity_, coeff_bytes)) {
        return false;
    }

    rtStream_t stream = nullptr;
    if (!check_rt(rtStreamCreate(&stream), "rtStreamCreate YOLOv11-SEG split")) {
        return false;
    }

    auto split_start = ProfileClock::now();
    auto detail_start = ProfileClock::now();
    bool ok = check_rt(rtMemcpy(split_input_sram_,
                                detection_output_device,
                                input_bytes,
                                rtMemcpyDeviceToSram),
                       "rtMemcpy output0 DDR to SRAM");
    if (profile != nullptr) {
        profile->split_d2s_ms = elapsed_ms(detail_start);
    }
    if (ok) {
        detail_start = ProfileClock::now();
        launch_yolov11_seg_split_det_coeff(stream,
                                           static_cast<const float*>(split_input_sram_),
                                           static_cast<float*>(det_head_sram_),
                                           static_cast<float*>(coeff_head_sram_),
                                           detection_layout_.anchors,
                                           det_channels_,
                                           config_.mask_coeff_count,
                                           total_head_channels_,
                                           split_input_channels_first_ ? 1 : 0);
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG split kernel launch") &&
             check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize output split");
        if (profile != nullptr) {
            profile->split_kernel_ms = elapsed_ms(detail_start);
        }
    }
    if (ok) {
        detail_start = ProfileClock::now();
        ok = check_rt(rtMemcpy(det_head_device_, det_head_sram_, det_bytes, rtMemcpySramToDevice),
                      "rtMemcpy det head SRAM to DDR") &&
             check_rt(rtMemcpy(coeff_head_device_, coeff_head_sram_, coeff_bytes, rtMemcpySramToDevice),
                      "rtMemcpy coeff head SRAM to DDR");
        if (profile != nullptr) {
            profile->split_s2d_ms = elapsed_ms(detail_start);
        }
    }

    if (profile != nullptr) {
        profile->split_ms = elapsed_ms(split_start);
    }

    auto d2h_start = ProfileClock::now();
    if (ok) {
        ok = check_rt(rtMemcpy(host_det_head_.data(), det_head_device_, det_bytes, rtMemcpyDeviceToHost),
                      "rtMemcpy split det D2H") &&
             check_rt(rtMemcpy(host_coeff_head_.data(), coeff_head_device_, coeff_bytes, rtMemcpyDeviceToHost),
                      "rtMemcpy split coeff D2H");
    }
    if (profile != nullptr) {
        profile->split_d2h_ms = elapsed_ms(d2h_start);
        profile->device_to_host_ms = profile->split_d2h_ms;
        profile->device_to_host_bytes = det_bytes + coeff_bytes;
    }

    check_rt(rtStreamDestroy(stream), "rtStreamDestroy YOLOv11-SEG split");
    return ok;
}

float YoloV11SegPostprocessor::detectionValueAt(int channel, int anchor) const
{
    if (channel < 0 || channel >= det_channels_ || anchor < 0 || anchor >= detection_layout_.anchors) {
        return 0.0f;
    }
    return host_det_head_[static_cast<size_t>(channel) *
                          static_cast<size_t>(detection_layout_.anchors) +
                          static_cast<size_t>(anchor)];
}

float YoloV11SegPostprocessor::coefficientValueAt(int coeff, int anchor) const
{
    if (coeff < 0 || coeff >= config_.mask_coeff_count ||
        anchor < 0 || anchor >= detection_layout_.anchors) {
        return 0.0f;
    }
    return host_coeff_head_[static_cast<size_t>(coeff) *
                            static_cast<size_t>(detection_layout_.anchors) +
                            static_cast<size_t>(anchor)];
}

float YoloV11SegPostprocessor::prototypeValueAt(int coeff, int y, int x) const
{
    size_t offset = (static_cast<size_t>(coeff) * static_cast<size_t>(prototype_height_) +
                     static_cast<size_t>(y)) * static_cast<size_t>(prototype_width_) +
                    static_cast<size_t>(x);
    return host_prototype_[offset];
}

void YoloV11SegPostprocessor::decodeCandidates(const LetterboxInfo& letterbox,
                                               std::vector<Candidate>& candidates) const
{
    candidates.clear();
    for (int anchor = 0; anchor < detection_layout_.anchors; ++anchor) {
        int best_class = -1;
        float best_score = 0.0f;
        for (int class_id = 0; class_id < config_.class_count; ++class_id) {
            float score = detectionValueAt(4 + class_id, anchor);
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (best_class < 0 || best_score < config_.score_threshold) {
            continue;
        }

        float cx = detectionValueAt(0, anchor);
        float cy = detectionValueAt(1, anchor);
        float width = detectionValueAt(2, anchor);
        float height = detectionValueAt(3, anchor);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        float left = ((cx - 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float top = ((cy - 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        float right = ((cx + 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float bottom = ((cy + 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        cv::Rect box = clip_rect_to_image(left, top, right, bottom,
                                          letterbox.source_width,
                                          letterbox.source_height);
        if (box.empty()) {
            continue;
        }

        Candidate candidate;
        candidate.anchor = anchor;
        candidate.class_id = best_class;
        candidate.score = best_score;
        candidate.box = box;
        candidates.push_back(candidate);
    }
}

void YoloV11SegPostprocessor::runNms(const std::vector<Candidate>& candidates, std::vector<int>& keep) const
{
    keep.clear();
    std::vector<int> order(candidates.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&candidates](int a, int b) {
        return candidates[static_cast<size_t>(a)].score > candidates[static_cast<size_t>(b)].score;
    });

    while (!order.empty() && static_cast<int>(keep.size()) < config_.max_segments) {
        int best = order.front();
        keep.push_back(best);

        std::vector<int> next;
        next.reserve(order.size());
        for (size_t i = 1; i < order.size(); ++i) {
            int idx = order[i];
            if (!config_.agnostic &&
                candidates[static_cast<size_t>(idx)].class_id != candidates[static_cast<size_t>(best)].class_id) {
                next.push_back(idx);
                continue;
            }
            if (intersection_over_union(candidates[static_cast<size_t>(best)].box,
                                        candidates[static_cast<size_t>(idx)].box) <= config_.nms_threshold) {
                next.push_back(idx);
            }
        }
        order.swap(next);
    }
}

bool YoloV11SegPostprocessor::runRppMaskDecode(const void* prototype_output_device,
                                               const LetterboxInfo& letterbox,
                                               const std::vector<Candidate>& candidates,
                                               const std::vector<int>& keep,
                                               std::vector<Segmentation>& segments,
                                               YoloV11SegPostprocessProfile* profile)
{
    segments.clear();
    if (keep.empty()) {
        return true;
    }
    if (letterbox.source_width <= 0 || letterbox.source_width > 8191 ||
        letterbox.source_height <= 0 || prototype_output_device == nullptr) {
        return false;
    }

    int segment_count = std::min(static_cast<int>(keep.size()), config_.max_segments);
    size_t coeff_bytes = static_cast<size_t>(segment_count) *
                         static_cast<size_t>(config_.mask_coeff_count) *
                         sizeof(float);
    size_t prototype_bytes = config_.prototype_elements * sizeof(float);
    size_t raw_mask_elements_per_segment = static_cast<size_t>(prototype_height_) *
                                           static_cast<size_t>(prototype_width_);
    size_t raw_masks_bytes = static_cast<size_t>(segment_count) *
                             raw_mask_elements_per_segment *
                             sizeof(float);
    size_t binary_masks_bytes = static_cast<size_t>(segment_count) *
                                raw_mask_elements_per_segment *
                                sizeof(float);
    size_t source_mask_elements = static_cast<size_t>(letterbox.source_width) *
                                  static_cast<size_t>(letterbox.source_height);
    size_t source_mask_bytes = source_mask_elements * sizeof(float);
    size_t selected_indices_bytes = static_cast<size_t>(segment_count) * sizeof(int);
    size_t selected_box_bytes = 4U * sizeof(int);

    host_selected_anchor_indices_.assign(static_cast<size_t>(segment_count), 0);
    host_selected_box_.assign(4U, 0);
    host_source_mask_.assign(source_mask_elements, 0);

    // TODO(postprocess-rpp-det-nms): replace CPU decode/NMS with the YOLOv11 det RT_NMS
    // path once it can return selected anchor indices for the SEG coefficient gather below.
    for (int i = 0; i < segment_count; ++i) {
        const Candidate& candidate = candidates[static_cast<size_t>(keep[static_cast<size_t>(i)])];
        host_selected_anchor_indices_[static_cast<size_t>(i)] = candidate.anchor;
    }

    if (!ensureSramBuffer(&prototype_sram_, &prototype_sram_capacity_, prototype_bytes) ||
        !ensureSramBuffer(&coefficients_sram_, &coefficients_sram_capacity_, coeff_bytes) ||
        !ensureSramBuffer(&raw_masks_sram_, &raw_masks_sram_capacity_, raw_masks_bytes) ||
        !ensureDeviceBuffer(&selected_indices_device_, &selected_indices_device_capacity_, selected_indices_bytes) ||
        !ensureSramBuffer(&selected_indices_sram_, &selected_indices_sram_capacity_, selected_indices_bytes) ||
        !ensureSramBuffer(&binary_masks_sram_, &binary_masks_sram_capacity_, binary_masks_bytes) ||
        !ensureDeviceBuffer(&selected_box_device_, &selected_box_device_capacity_, selected_box_bytes) ||
        !ensureSramBuffer(&selected_box_sram_, &selected_box_sram_capacity_, selected_box_bytes) ||
        !ensureDeviceBuffer(&source_mask_device_, &source_mask_device_capacity_, source_mask_bytes) ||
        !ensureSramBuffer(&source_mask_sram_, &source_mask_sram_capacity_, source_mask_bytes)) {
        return false;
    }

    rtStream_t stream = nullptr;
    if (!check_rt(rtStreamCreate(&stream), "rtStreamCreate YOLOv11-SEG mask decode")) {
        return false;
    }

    auto stage_start = ProfileClock::now();
    auto detail_start = ProfileClock::now();
    bool ok = check_rt(rtMemcpy(prototype_sram_, prototype_output_device, prototype_bytes, rtMemcpyDeviceToSram),
                       "rtMemcpy prototype DDR to SRAM") &&
              check_rt(rtMemcpy(selected_indices_device_,
                                host_selected_anchor_indices_.data(),
                                selected_indices_bytes,
                                rtMemcpyHostToDevice),
                       "rtMemcpy selected anchor indices H2D") &&
              check_rt(rtMemcpy(selected_indices_sram_,
                                selected_indices_device_,
                                selected_indices_bytes,
                                rtMemcpyDeviceToSram),
                       "rtMemcpy selected anchor indices DDR to SRAM");
    if (profile != nullptr) {
        profile->mask_setup_copy_ms = elapsed_ms(detail_start);
    }
    if (ok) {
        detail_start = ProfileClock::now();
        launch_yolov11_seg_gather_selected_coefficients(stream,
                                                        static_cast<const float*>(coeff_head_sram_),
                                                        static_cast<const int*>(selected_indices_sram_),
                                                        static_cast<float*>(coefficients_sram_),
                                                        segment_count,
                                                        config_.mask_coeff_count,
                                                        detection_layout_.anchors);
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG coefficient gather kernel launch") &&
             check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize coefficient gather");
        if (profile != nullptr) {
            profile->mask_coeff_gather_ms = elapsed_ms(detail_start);
        }
    }
    if (ok) {
        detail_start = ProfileClock::now();
        launch_yolov11_seg_combine_prototypes(stream,
                                              static_cast<const float*>(prototype_sram_),
                                              static_cast<const float*>(coefficients_sram_),
                                              static_cast<float*>(raw_masks_sram_),
                                              segment_count,
                                              config_.mask_coeff_count,
                                              prototype_height_,
                                              prototype_width_);
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG prototype combine kernel launch") &&
             check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize prototype combine");
        if (profile != nullptr) {
            profile->mask_proto_combine_ms = elapsed_ms(detail_start);
        }
    }
    if (ok) {
        detail_start = ProfileClock::now();
        launch_yolov11_seg_sigmoid_threshold_lowres(stream,
                                                    static_cast<const float*>(raw_masks_sram_),
                                                    static_cast<float*>(binary_masks_sram_),
                                                    segment_count,
                                                    prototype_height_,
                                                    prototype_width_,
                                                    std::log(std::max(1.0e-6f, config_.mask_threshold) /
                                                             std::max(1.0e-6f, 1.0f - config_.mask_threshold)));
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG low-res threshold kernel launch") &&
             check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize low-res threshold");
        if (profile != nullptr) {
            profile->mask_lowres_threshold_ms = elapsed_ms(detail_start);
        }
    }
    if (!ok) {
        check_rt(rtStreamDestroy(stream), "rtStreamDestroy YOLOv11-SEG mask decode");
        return false;
    }

    segments.reserve(static_cast<size_t>(segment_count));
    for (int i = 0; i < segment_count; ++i) {
        const Candidate& candidate = candidates[static_cast<size_t>(keep[static_cast<size_t>(i)])];
        host_selected_box_[0] = candidate.box.x;
        host_selected_box_[1] = candidate.box.y;
        host_selected_box_[2] = candidate.box.width;
        host_selected_box_[3] = candidate.box.height;

        const float* raw_mask = static_cast<const float*>(raw_masks_sram_) +
                                static_cast<size_t>(i) * raw_mask_elements_per_segment;
        detail_start = ProfileClock::now();
        ok = check_rt(rtMemcpy(selected_box_device_,
                               host_selected_box_.data(),
                               selected_box_bytes,
                               rtMemcpyHostToDevice),
                      "rtMemcpy selected box H2D") &&
             check_rt(rtMemcpy(selected_box_sram_,
                               selected_box_device_,
                               selected_box_bytes,
                               rtMemcpyDeviceToSram),
                      "rtMemcpy selected box DDR to SRAM");
        if (profile != nullptr) {
            profile->mask_box_copy_ms += elapsed_ms(detail_start);
        }
        if (ok) {
            detail_start = ProfileClock::now();
            launch_yolov11_seg_resize_raw_to_source_crop(stream,
                                                         raw_mask,
                                                         static_cast<const int*>(selected_box_sram_),
                                                         static_cast<float*>(source_mask_sram_),
                                                         letterbox.source_height,
                                                         letterbox.source_width,
                                                         letterbox.model_height,
                                                         letterbox.model_width,
                                                         prototype_height_,
                                                         prototype_width_,
                                                         letterbox.resized_height,
                                                         letterbox.resized_width,
                                                         static_cast<int>(std::round(letterbox.pad_y)),
                                                         static_cast<int>(std::round(letterbox.pad_x)),
                                                         std::log(std::max(1.0e-6f, config_.mask_threshold) /
                                                                 std::max(1.0e-6f, 1.0f - config_.mask_threshold)));
            ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG source resize/crop kernel launch") &&
                 check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize source resize/crop");
            if (profile != nullptr) {
                profile->mask_source_resize_crop_ms += elapsed_ms(detail_start);
            }
        }
        if (ok) {
            detail_start = ProfileClock::now();
            ok = check_rt(rtMemcpy(source_mask_device_, source_mask_sram_, source_mask_bytes, rtMemcpySramToDevice),
                          "rtMemcpy source mask SRAM to DDR") &&
                 check_rt(rtMemcpy(host_source_mask_.data(), source_mask_device_, source_mask_bytes, rtMemcpyDeviceToHost),
                          "rtMemcpy source mask D2H");
            if (profile != nullptr) {
                profile->mask_source_copy_ms += elapsed_ms(detail_start);
            }
        }
        if (!ok) {
            check_rt(rtStreamDestroy(stream), "rtStreamDestroy YOLOv11-SEG mask decode");
            return false;
        }

        Segmentation segment;
        segment.class_id = candidate.class_id;
        segment.confidence = candidate.score;
        segment.box = candidate.box;
        detail_start = ProfileClock::now();
        buildMaskFromSourceBinary(host_source_mask_.data(),
                                  letterbox.source_width,
                                  letterbox.source_height,
                                  segment);
        if (profile != nullptr) {
            profile->mask_build_cpu_ms += elapsed_ms(detail_start);
        }
        if (!segment.mask.empty()) {
            segments.push_back(std::move(segment));
        }
    }
    check_rt(rtStreamDestroy(stream), "rtStreamDestroy YOLOv11-SEG mask decode");
    if (profile != nullptr) {
        profile->mask_decode_ms = elapsed_ms(stage_start);
        profile->device_to_host_ms += profile->mask_source_copy_ms;
        profile->device_to_host_bytes += static_cast<size_t>(segment_count) * source_mask_bytes;
    }
    return true;
}

void YoloV11SegPostprocessor::buildMask(const Candidate& candidate,
                                        const LetterboxInfo& letterbox,
                                        Segmentation& segment) const
{
    cv::Mat proto_mask(prototype_height_, prototype_width_, CV_32FC1);
    for (int y = 0; y < prototype_height_; ++y) {
        float* row = proto_mask.ptr<float>(y);
        for (int x = 0; x < prototype_width_; ++x) {
            float value = 0.0f;
            for (int coeff = 0; coeff < config_.mask_coeff_count; ++coeff) {
                value += coefficientValueAt(coeff, candidate.anchor) *
                         prototypeValueAt(coeff, y, x);
            }
            row[x] = sigmoid(value);
        }
    }

    cv::Mat model_mask;
    cv::resize(proto_mask, model_mask, cv::Size(letterbox.model_width, letterbox.model_height), 0.0, 0.0, cv::INTER_LINEAR);

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
    cv::resize(unpadded, source_prob, cv::Size(letterbox.source_width, letterbox.source_height), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(source_prob, binary, config_.mask_threshold, 255.0, cv::THRESH_BINARY);
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

    segment.mask_width = cropped.cols;
    segment.mask_height = cropped.rows;
    segment.mask.assign(cropped.datastart, cropped.dataend);
}

void YoloV11SegPostprocessor::buildMaskFromRaw(const Candidate& candidate,
                                               const LetterboxInfo& letterbox,
                                               const float* raw_mask,
                                               Segmentation& segment) const
{
    if (raw_mask == nullptr) {
        return;
    }

    cv::Mat proto_prob(prototype_height_, prototype_width_, CV_32FC1);
    for (int y = 0; y < prototype_height_; ++y) {
        float* row = proto_prob.ptr<float>(y);
        for (int x = 0; x < prototype_width_; ++x) {
            row[x] = sigmoid(raw_mask[y * prototype_width_ + x]);
        }
    }

    cv::Mat model_mask;
    cv::resize(proto_prob, model_mask, cv::Size(letterbox.model_width, letterbox.model_height), 0.0, 0.0, cv::INTER_LINEAR);

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
    cv::resize(unpadded, source_prob, cv::Size(letterbox.source_width, letterbox.source_height), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat binary;
    cv::threshold(source_prob, binary, config_.mask_threshold, 255.0, cv::THRESH_BINARY);
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

    segment.mask_width = cropped.cols;
    segment.mask_height = cropped.rows;
    segment.mask.assign(cropped.datastart, cropped.dataend);
}

void YoloV11SegPostprocessor::buildMaskFromLowResBinary(const Candidate& candidate,
                                                        const LetterboxInfo& letterbox,
                                                        const float* lowres_mask,
                                                        Segmentation& segment) const
{
    if (lowres_mask == nullptr) {
        return;
    }

    cv::Mat lowres_f32(prototype_height_,
                       prototype_width_,
                       CV_32FC1,
                       const_cast<float*>(lowres_mask));

    cv::Mat model_mask;
    cv::resize(lowres_f32,
               model_mask,
               cv::Size(letterbox.model_width, letterbox.model_height),
               0.0,
               0.0,
               cv::INTER_LINEAR);

    cv::Mat model_binary;
    cv::threshold(model_mask, model_binary, 127.0, 255.0, cv::THRESH_BINARY);
    model_binary.convertTo(model_binary, CV_8UC1);

    int valid_x = std::max(0, static_cast<int>(std::round(letterbox.pad_x)));
    int valid_y = std::max(0, static_cast<int>(std::round(letterbox.pad_y)));
    int valid_w = std::min(letterbox.resized_width, letterbox.model_width - valid_x);
    int valid_h = std::min(letterbox.resized_height, letterbox.model_height - valid_y);
    if (valid_w <= 0 || valid_h <= 0) {
        return;
    }

    cv::Rect valid_region(valid_x, valid_y, valid_w, valid_h);
    cv::Mat unpadded = model_binary(valid_region);
    cv::Mat source_binary;
    cv::resize(unpadded,
               source_binary,
               cv::Size(letterbox.source_width, letterbox.source_height),
               0.0,
               0.0,
               cv::INTER_NEAREST);

    cv::Mat cropped = cv::Mat::zeros(source_binary.size(), CV_8UC1);
    cv::Rect clipped_box = candidate.box & cv::Rect(0, 0, source_binary.cols, source_binary.rows);
    if (clipped_box.empty()) {
        return;
    }
    source_binary(clipped_box).copyTo(cropped(clipped_box));
    if (cv::countNonZero(cropped) == 0) {
        return;
    }

    segment.mask_width = cropped.cols;
    segment.mask_height = cropped.rows;
    segment.mask.assign(cropped.datastart, cropped.dataend);
}

void YoloV11SegPostprocessor::buildMaskFromSourceBinary(const float* source_mask,
                                                        int source_width,
                                                        int source_height,
                                                        Segmentation& segment) const
{
    if (source_mask == nullptr || source_width <= 0 || source_height <= 0) {
        return;
    }

    cv::Mat source_f32(source_height,
                      source_width,
                      CV_32FC1,
                      const_cast<float*>(source_mask));
    cv::Mat source_u8;
    source_f32.convertTo(source_u8, CV_8UC1);
    if (cv::countNonZero(source_u8) == 0) {
        return;
    }

    segment.mask_width = source_u8.cols;
    segment.mask_height = source_u8.rows;
    segment.mask.assign(source_u8.datastart, source_u8.dataend);
}

bool YoloV11SegPostprocessor::ensureDeviceBuffer(void** buffer, size_t* capacity, size_t required_bytes)
{
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFree(*buffer), "rtFree postprocess DDR")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtMalloc(buffer, required_bytes), "rtMalloc postprocess DDR")) {
        return false;
    }
    *capacity = required_bytes;
    return true;
}

bool YoloV11SegPostprocessor::ensureSramBuffer(void** buffer, size_t* capacity, size_t required_bytes)
{
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFreeSram(*buffer), "rtFreeSram postprocess")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtMallocSram(buffer, required_bytes), "rtMallocSram postprocess")) {
        return false;
    }
    *capacity = required_bytes;
    return true;
}

void YoloV11SegPostprocessor::releaseMaskDecodeBuffers()
{
    if (prototype_sram_ != nullptr) {
        rtFreeSram(prototype_sram_);
        prototype_sram_ = nullptr;
    }
    if (coefficients_device_ != nullptr) {
        rtFree(coefficients_device_);
        coefficients_device_ = nullptr;
    }
    if (coefficients_sram_ != nullptr) {
        rtFreeSram(coefficients_sram_);
        coefficients_sram_ = nullptr;
    }
    if (raw_masks_device_ != nullptr) {
        rtFree(raw_masks_device_);
        raw_masks_device_ = nullptr;
    }
    if (raw_masks_sram_ != nullptr) {
        rtFreeSram(raw_masks_sram_);
        raw_masks_sram_ = nullptr;
    }
    if (selected_indices_device_ != nullptr) {
        rtFree(selected_indices_device_);
        selected_indices_device_ = nullptr;
    }
    if (selected_indices_sram_ != nullptr) {
        rtFreeSram(selected_indices_sram_);
        selected_indices_sram_ = nullptr;
    }
    if (binary_masks_device_ != nullptr) {
        rtFree(binary_masks_device_);
        binary_masks_device_ = nullptr;
    }
    if (binary_masks_sram_ != nullptr) {
        rtFreeSram(binary_masks_sram_);
        binary_masks_sram_ = nullptr;
    }
    if (selected_box_device_ != nullptr) {
        rtFree(selected_box_device_);
        selected_box_device_ = nullptr;
    }
    if (selected_box_sram_ != nullptr) {
        rtFreeSram(selected_box_sram_);
        selected_box_sram_ = nullptr;
    }
    if (source_mask_device_ != nullptr) {
        rtFree(source_mask_device_);
        source_mask_device_ = nullptr;
    }
    if (source_mask_sram_ != nullptr) {
        rtFreeSram(source_mask_sram_);
        source_mask_sram_ = nullptr;
    }
    if (split_input_sram_ != nullptr) {
        rtFreeSram(split_input_sram_);
        split_input_sram_ = nullptr;
    }
    if (det_head_device_ != nullptr) {
        rtFree(det_head_device_);
        det_head_device_ = nullptr;
    }
    if (det_head_sram_ != nullptr) {
        rtFreeSram(det_head_sram_);
        det_head_sram_ = nullptr;
    }
    if (coeff_head_device_ != nullptr) {
        rtFree(coeff_head_device_);
        coeff_head_device_ = nullptr;
    }
    if (coeff_head_sram_ != nullptr) {
        rtFreeSram(coeff_head_sram_);
        coeff_head_sram_ = nullptr;
    }
    prototype_sram_capacity_ = 0;
    coefficients_device_capacity_ = 0;
    coefficients_sram_capacity_ = 0;
    raw_masks_device_capacity_ = 0;
    raw_masks_sram_capacity_ = 0;
    selected_indices_device_capacity_ = 0;
    selected_indices_sram_capacity_ = 0;
    binary_masks_device_capacity_ = 0;
    binary_masks_sram_capacity_ = 0;
    selected_box_device_capacity_ = 0;
    selected_box_sram_capacity_ = 0;
    source_mask_device_capacity_ = 0;
    source_mask_sram_capacity_ = 0;
    split_input_sram_capacity_ = 0;
    det_head_device_capacity_ = 0;
    det_head_sram_capacity_ = 0;
    coeff_head_device_capacity_ = 0;
    coeff_head_sram_capacity_ = 0;
}
