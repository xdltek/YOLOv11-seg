/**
 * @file rpp_yolov11_seg_postprocessor.cpp
 * @brief YOLOv11-SEG output copy, decode, NMS, and mask reconstruction module.
 */
#include "rpp_yolov11_seg_postprocessor.h"
#include "rpp_mask_geometry.h"

#include "perf_trace_session.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <rpp_drv_api.h>

namespace
{
using ProfileClock = std::chrono::high_resolution_clock;
constexpr int kRppNmsMaxOutputBoxesPerClass = 512;

class PostprocessLogger : public infer1::ILogger
{
public:
    void log(Severity severity, const char* message) override
    {
        if (severity <= Severity::kERROR) {
            std::cerr << "[YOLOv11-SEG RPP postprocess] " << message << std::endl;
        }
    }
};

PostprocessLogger& postprocess_logger()
{
    static PostprocessLogger logger;
    return logger;
}

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

bool check_rpp(RPPresult status, const char* what)
{
    if (status == RPP_SUCCESS) {
        return true;
    }
    std::cerr << what << " failed, RPPresult=" << static_cast<int>(status) << std::endl;
    return false;
}

int dims_volume(const infer1::Dims& dims)
{
    int result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0 || result > std::numeric_limits<int>::max() / dims.d[i]) {
            return 0;
        }
        result *= dims.d[i];
    }
    return result;
}

uint16_t float_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
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

}

YoloV11SegPostprocessor::~YoloV11SegPostprocessor()
{
    releaseMaskDecodeBuffers();
    resetRppNmsEngine();
    if (mask_event_resize_end_ != nullptr) {
        rtEventDestroy(mask_event_resize_end_);
    }
    if (mask_event_combine_end_ != nullptr) {
        rtEventDestroy(mask_event_combine_end_);
    }
    if (mask_event_gather_end_ != nullptr) {
        rtEventDestroy(mask_event_gather_end_);
    }
    if (mask_event_start_ != nullptr) {
        rtEventDestroy(mask_event_start_);
    }
    if (mask_event_setup_start_ != nullptr) {
        rtEventDestroy(mask_event_setup_start_);
    }
    if (stream_ != nullptr) {
        rtStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void YoloV11SegPostprocessor::releaseSramBuffers()
{
    auto release = [](void** buffer, size_t* capacity) {
        if (*buffer != nullptr) {
            rtFreeSram(*buffer);
            *buffer = nullptr;
        }
        *capacity = 0;
    };
    release(&prototype_sram_, &prototype_sram_capacity_);
    release(&coefficients_sram_, &coefficients_sram_capacity_);
    release(&raw_masks_sram_, &raw_masks_sram_capacity_);
    release(&metadata_sram_, &metadata_sram_capacity_);
    release(&source_mask_sram_, &source_mask_sram_capacity_);
    release(&split_input_sram_, &split_input_sram_capacity_);
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

    resetRppNmsEngine();
    if (!buildRppNmsEngine() || !allocateRppNmsBuffers() || !copyRppNmsConstants()) {
        std::cerr << "Failed to initialize YOLOv11-SEG RPP RT-NMS path." << std::endl;
        resetRppNmsEngine();
        return false;
    }

    host_det_head_.assign(4U * static_cast<size_t>(detection_layout_.anchors), 0.0f);
    if (stream_ == nullptr && !check_rt(rtStreamCreate(&stream_), "rtStreamCreate YOLOv11-SEG postprocess")) {
        return false;
    }
    if (std::getenv("YOLO_POST_PROFILE") != nullptr && mask_event_start_ == nullptr) {
        if (!check_rt(rtEventCreateWithFlags(&mask_event_setup_start_, rtEventEnableTiming), "rtEventCreate mask setup start") ||
            !check_rt(rtEventCreateWithFlags(&mask_event_start_, rtEventEnableTiming), "rtEventCreate mask start") ||
            !check_rt(rtEventCreateWithFlags(&mask_event_gather_end_, rtEventEnableTiming), "rtEventCreate gather end") ||
            !check_rt(rtEventCreateWithFlags(&mask_event_combine_end_, rtEventEnableTiming), "rtEventCreate combine end") ||
            !check_rt(rtEventCreateWithFlags(&mask_event_resize_end_, rtEventEnableTiming), "rtEventCreate resize end")) {
            return false;
        }
    }
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
    rtStream_t previous_stream = stream_;
    struct StreamRestore
    {
        rtStream_t& target;
        rtStream_t previous;
        ~StreamRestore() { target = previous; }
    } stream_restore{stream_, previous_stream};
    if (stream != nullptr) {
        stream_ = stream;
    }
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

    if (const char* dump_dir = std::getenv("YOLO_DUMP_RAW_OUTPUTS")) {
        std::vector<float> detection_dump(config_.detection_elements);
        std::vector<float> prototype_dump(config_.prototype_elements);
        const size_t detection_bytes = detection_dump.size() * sizeof(float);
        const size_t prototype_bytes = prototype_dump.size() * sizeof(float);
        if (!check_rt(rtMemcpy(detection_dump.data(), detection_output_device,
                               detection_bytes, rtMemcpyDeviceToHost), "dump detection D2H") ||
            !check_rt(rtMemcpy(prototype_dump.data(), prototype_output_device,
                               prototype_bytes, rtMemcpyDeviceToHost), "dump prototype D2H")) {
            return false;
        }
        std::ofstream detection_file(std::string(dump_dir) + "/detection.bin", std::ios::binary);
        std::ofstream prototype_file(std::string(dump_dir) + "/prototype.bin", std::ios::binary);
        detection_file.write(reinterpret_cast<const char*>(detection_dump.data()), detection_bytes);
        prototype_file.write(reinterpret_cast<const char*>(prototype_dump.data()), prototype_bytes);
    }

    auto total_start = ProfileClock::now();
    std::vector<Candidate> candidates;
    std::vector<int> keep;
    auto stage_start = ProfileClock::now();
    {
        PERF_SCOPE_CATE("postprocess_rt_nms_rpp", "postprocess");
        if (!runRppNms(detection_output_device, letterbox, candidates, keep, profile)) {
            return false;
        }
    }
    if (profile != nullptr) {
        profile->candidate_count = candidates.size();
        profile->decode_nms_ms = elapsed_ms(stage_start);
    }

    stage_start = ProfileClock::now();
    {
        PERF_SCOPE_CATE("postprocess_mask_decode_rpp", "postprocess");
        // RT NMS consumes the DDR prediction directly. Mask gather independently
        // needs the current FP32 prediction in SRAM; it must not read the old
        // split path's uninitialized/null workspace.
        // Channel-major heads only need the 32 coefficient planes. Keeping the
        // entire 116-channel head resident exhausts SRAM at large input sizes.
        const size_t prediction_bytes = size_t(config_.mask_coeff_count) *
                                        detection_layout_.anchors * sizeof(float);
        const void* mask_input = static_cast<const float*>(detection_output_device) +
                                 size_t(det_channels_) * detection_layout_.anchors;
        if (!ensureSramBuffer(&split_input_sram_, &split_input_sram_capacity_, prediction_bytes) ||
            !check_rt(rtMemcpy(split_input_sram_, mask_input, prediction_bytes,
                               rtMemcpyDeviceToSram), "mask prediction DDR to SRAM")) {
            return false;
        }
        if (runRppMaskDecode(prototype_output_device, letterbox, candidates, keep, segments, profile)) {
            if (profile != nullptr) {
                profile->total_ms = elapsed_ms(total_start);
            }
            if (std::getenv("YOLO_DEBUG_POST") != nullptr) {
                std::cout << "YOLOv11-SEG postprocess debug: mode=rpp_seg"
                          << ", candidates=" << candidates.size()
                          << ", kept=" << segments.size()
                          << ", decode_ms=" << (profile != nullptr ? profile->candidate_decode_ms : 0.0)
                          << ", sort_ms=" << (profile != nullptr ? profile->candidate_sort_ms : 0.0)
                          << ", nms_ms=" << (profile != nullptr ? profile->nms_ms : 0.0)
                          << std::endl;
            }
            return true;
        }
    }

    // The mask workspace now contains coefficient planes only. A failed RPP
    // stage cannot be reinterpreted as the legacy full-head CPU split layout.
    std::cerr << "RPP mask reconstruction failed; no fallback result is accepted." << std::endl;
    return false;
}

bool YoloV11SegPostprocessor::buildRppNmsEngine()
{
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(postprocess_logger())};
    std::unique_ptr<infer1::INetworkDefinition> network {
        builder == nullptr ? nullptr : builder->createNetwork()};
    std::unique_ptr<infer1::IBuilderConfig> builder_config {
        builder == nullptr ? nullptr : builder->createBuilderConfig()};
    if (builder == nullptr || network == nullptr || builder_config == nullptr) {
        return false;
    }

    infer1::Dims raw_dims{};
    raw_dims.nbDims = 2;
    raw_dims.d[0] = det_channels_;
    raw_dims.d[1] = detection_layout_.anchors;
    infer1::ITensor* raw = network->addInput("yolov11_seg_prediction", infer1::DataType::kFLOAT, raw_dims);
    if (raw == nullptr) {
        return false;
    }

    infer1::IShuffleLayer* transpose = network->addShuffle(*raw);
    if (transpose == nullptr) return false;
    infer1::Permutation permutation{};
    permutation.order[0] = 1;
    permutation.order[1] = 0;
    transpose->setFirstTranspose(permutation);
    infer1::ITensor* rows = transpose->getOutput(0);
    infer1::IIdentityLayer* cast = network->addIdentity(*rows);
    if (cast == nullptr) return false;
    cast->setPrecision(infer1::DataType::kBF);
    cast->setOutputType(0, infer1::DataType::kBF);
    cast->getOutput(0)->setType(infer1::DataType::kBF);

    infer1::Dims start{};
    start.nbDims = 2;
    infer1::Dims stride{};
    stride.nbDims = 2;
    stride.d[0] = stride.d[1] = 1;
    infer1::Dims box_size{};
    box_size.nbDims = 2;
    box_size.d[0] = detection_layout_.anchors;
    box_size.d[1] = 4;
    infer1::ISliceLayer* boxes = network->addSlice(*cast->getOutput(0), start, box_size, stride);
    if (boxes == nullptr) return false;
    boxes->setPrecision(infer1::DataType::kBF);
    boxes->setOutputType(0, infer1::DataType::kBF);
    boxes->getOutput(0)->setType(infer1::DataType::kBF);
    boxes->getOutput(0)->setName("rpp_nms_boxes");

    start.d[1] = 4;
    infer1::Dims score_size{};
    score_size.nbDims = 2;
    score_size.d[0] = detection_layout_.anchors;
    score_size.d[1] = config_.class_count;
    infer1::ISliceLayer* scores = network->addSlice(*cast->getOutput(0), start, score_size, stride);
    if (scores == nullptr) return false;
    scores->setPrecision(infer1::DataType::kBF);
    scores->setOutputType(0, infer1::DataType::kBF);
    scores->getOutput(0)->setType(infer1::DataType::kBF);
    scores->getOutput(0)->setName("rpp_nms_scores");

    infer1::Dims scalar{};
    scalar.nbDims = 1;
    scalar.d[0] = 1;
    infer1::ITensor* max_output = network->addInput("rpp_nms_max_output", infer1::DataType::kINT32, scalar);
    infer1::ITensor* iou = network->addInput("rpp_nms_iou_threshold", infer1::DataType::kBF, scalar);
    infer1::ITensor* score = network->addInput("rpp_nms_score_threshold", infer1::DataType::kBF, scalar);
    if (max_output == nullptr || iou == nullptr || score == nullptr) {
        return false;
    }

    infer1::ITensor* nms_scores = scores->getOutput(0);
    if (config_.agnostic) {
        infer1::IReduceLayer* best = network->addReduce(
            *nms_scores, infer1::ReduceOperation::kMAX, 1U << 1U, true);
        if (best == nullptr) return false;
        best->setPrecision(infer1::DataType::kBF);
        best->setOutputType(0, infer1::DataType::kBF);
        best->getOutput(0)->setType(infer1::DataType::kBF);
        nms_scores = best->getOutput(0);
    }
    infer1::INMSLayer* nms = network->addNMS(*boxes->getOutput(0), *nms_scores, *max_output);
    if (nms == nullptr) {
        return false;
    }
    nms->setBoundingBoxFormat(infer1::BoundingBoxFormat::kCENTER_SIZES);
    nms->setInput(3, *iou);
    nms->setInput(4, *score);
    nms->setPrecision(infer1::DataType::kBF);
    nms->setOutputType(0, infer1::DataType::kINT32);
    nms->setOutputType(1, infer1::DataType::kINT32);
    nms->setHasBoxConfidence(false);
    nms->setIsAgnostic(false);
    nms->getOutput(0)->setName("rpp_nms_indices");
    nms->getOutput(1)->setName("rpp_nms_count");

    network->markOutput(*boxes->getOutput(0));
    network->markOutput(*scores->getOutput(0));
    network->markOutput(*nms->getOutput(0));
    network->markOutput(*nms->getOutput(1));
    builder->setMaxBatchSize(1);
    builder_config->setMaxWorkspaceSize(static_cast<size_t>(16) << 20);
    builder_config->setFlag(infer1::BuilderFlag::kBF16);
    nms_engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
    nms_context_.reset(nms_engine_ == nullptr ? nullptr : nms_engine_->createExecutionContext());
    if (nms_engine_ == nullptr || nms_context_ == nullptr) {
        return false;
    }

    nms_input_index_ = nms_engine_->getBindingIndex("yolov11_seg_prediction");
    nms_boxes_index_ = nms_engine_->getBindingIndex("rpp_nms_boxes");
    nms_scores_index_ = nms_engine_->getBindingIndex("rpp_nms_scores");
    nms_max_output_index_ = nms_engine_->getBindingIndex("rpp_nms_max_output");
    nms_iou_threshold_index_ = nms_engine_->getBindingIndex("rpp_nms_iou_threshold");
    nms_score_threshold_index_ = nms_engine_->getBindingIndex("rpp_nms_score_threshold");
    nms_indices_index_ = nms_engine_->getBindingIndex("rpp_nms_indices");
    nms_count_index_ = nms_engine_->getBindingIndex("rpp_nms_count");
    nms_anchor_map_elements_ = detection_layout_.anchors;
    // SDK compatibility policy only: forcing dense compaction on the smaller
    // head can discard a weak object at the SDK preselection boundary. Larger
    // sparse heads require compaction to avoid anchor-bucket collisions.
    nms_requires_compaction_ = detection_layout_.anchors > 8400;
    if (nms_input_index_ < 0 || nms_boxes_index_ < 0 || nms_scores_index_ < 0 ||
        nms_max_output_index_ < 0 || nms_iou_threshold_index_ < 0 ||
        nms_score_threshold_index_ < 0 || nms_indices_index_ < 0 || nms_count_index_ < 0) {
        return false;
    }
    const infer1::Dims indices_dims = nms_engine_->getBindingDimensions(nms_indices_index_);
    nms_indices_elements_ = dims_volume(indices_dims);
    nms_count_elements_ = dims_volume(nms_engine_->getBindingDimensions(nms_count_index_));
    if (indices_dims.nbDims > 0) {
        const int tuple = indices_dims.d[indices_dims.nbDims - 1];
        if (tuple == 2 || tuple == 3) {
            nms_index_tuple_size_ = tuple;
        }
    }
    return nms_indices_elements_ > 0 && nms_count_elements_ > 0;
}

bool YoloV11SegPostprocessor::allocateRppNmsBuffers()
{
    nms_boxes_bytes_ = static_cast<size_t>(detection_layout_.anchors) * 4U * sizeof(uint16_t);
    nms_scores_bytes_ = static_cast<size_t>(detection_layout_.anchors) *
                        static_cast<size_t>(config_.class_count) * sizeof(uint16_t);
    const size_t indices_bytes = static_cast<size_t>(nms_indices_elements_) * sizeof(int32_t);
    nms_outputs_bytes_ = indices_bytes + static_cast<size_t>(nms_count_elements_ + nms_anchor_map_elements_) * sizeof(int32_t);
    if (!check_rt(rtMalloc(&nms_compact_device_,
            size_t(det_channels_) * detection_layout_.anchors * sizeof(float)), "rtMalloc compact NMS input")) return false;
    return check_rt(rtMalloc(&nms_boxes_device_, nms_boxes_bytes_), "rtMalloc RT-NMS boxes") &&
           check_rt(rtMalloc(&nms_scores_device_, nms_scores_bytes_), "rtMalloc RT-NMS scores") &&
           check_rt(rtMalloc(&nms_constants_device_, sizeof(int32_t) + 2U * sizeof(uint16_t)), "rtMalloc RT-NMS constants") &&
           check_rt(rtMalloc(&nms_outputs_device_, nms_outputs_bytes_), "rtMalloc RT-NMS outputs") &&
           check_rt(rtHostAlloc(&nms_boxes_host_, nms_boxes_bytes_, rtHostAllocDefault), "rtHostAlloc RT-NMS boxes") &&
           check_rt(rtHostAlloc(&nms_scores_host_, nms_scores_bytes_, rtHostAllocDefault), "rtHostAlloc RT-NMS scores") &&
           check_rt(rtHostAlloc(&nms_outputs_host_, nms_outputs_bytes_, rtHostAllocDefault), "rtHostAlloc RT-NMS outputs");
}

bool YoloV11SegPostprocessor::copyRppNmsConstants()
{
    const size_t map_offset = size_t(nms_indices_elements_ + nms_count_elements_) * sizeof(int32_t);
    auto* host_map = reinterpret_cast<int32_t*>(static_cast<unsigned char*>(nms_outputs_host_) + map_offset);
    for (int i = 0; i < nms_anchor_map_elements_; ++i) host_map[i] = i;
    if (!check_rt(rtMemcpy(static_cast<unsigned char*>(nms_outputs_device_) + map_offset, host_map,
                          size_t(nms_anchor_map_elements_) * sizeof(int32_t), rtMemcpyHostToDevice),
                  "initialize original-anchor map")) return false;
    struct Constants { int32_t max_output; uint16_t iou; uint16_t score; } constants {
        kRppNmsMaxOutputBoxesPerClass,
        float_to_bf16(config_.nms_threshold),
        float_to_bf16(config_.score_threshold)};
    return check_rt(rtMemcpy(nms_constants_device_, &constants, sizeof(constants), rtMemcpyHostToDevice),
                    "rtMemcpy RT-NMS constants");
}

bool YoloV11SegPostprocessor::compactRppNmsInput(const void* input, int32_t* device_map)
{
    // One aligned allocation owns all temporary compaction arrays. Its lifetime
    // ends before OpenRT NMS reserves SRAM, just as for inference head decoding.
    struct Scratch {
        void* data = nullptr;
        ~Scratch() { if (data) rtFreeSram(data); }
    } scratch;
    const int anchors = detection_layout_.anchors;
    const size_t bytes = size_t(det_channels_) * anchors * sizeof(float);
    const size_t index_bytes = size_t(anchors) * sizeof(int32_t);
    auto aligned = [](size_t n) { return (n + 127) & ~size_t(127); };
    const size_t data_stride = aligned(bytes), index_stride = aligned(index_bytes);
    if (!check_rt(rtMallocSram(&scratch.data, 2 * data_stride + 3 * index_stride),
                   "compact SRAM workspace")) return false;
    auto* base = static_cast<unsigned char*>(scratch.data);
    auto* output = reinterpret_cast<float*>(base + data_stride);
    auto* ping = reinterpret_cast<int*>(base + 2 * data_stride);
    auto* pong = reinterpret_cast<int*>(base + 2 * data_stride + index_stride);
    auto* map = reinterpret_cast<int*>(base + 2 * data_stride + 2 * index_stride);
    if (!check_rt(rtMemcpy(base, input, bytes, rtMemcpyDeviceToSram), "compact input D2S")) return false;
    launch_yolov11_seg_compact_candidates(stream_, reinterpret_cast<const float*>(base),
        output, ping, pong, map, anchors, config_.class_count,
        bf16_to_float(float_to_bf16(config_.score_threshold)));
    return check_rt(rtStreamSynchronize(stream_), "compact candidates synchronize") &&
        check_rt(rtMemcpy(nms_compact_device_, output, bytes, rtMemcpySramToDevice), "compact input S2D") &&
        check_rt(rtMemcpy(device_map, map, index_bytes, rtMemcpySramToDevice), "compact map S2D");
}

bool YoloV11SegPostprocessor::runRppNms(const void* detection_output_device,
                                        const LetterboxInfo& letterbox,
                                        std::vector<Candidate>& candidates,
                                        std::vector<int>& keep,
                                        YoloV11SegPostprocessProfile* profile)
{
    std::vector<void*> bindings(static_cast<size_t>(nms_engine_->getNbBindings()), nullptr);
    auto* constants = static_cast<unsigned char*>(nms_constants_device_);
    const size_t indices_bytes = static_cast<size_t>(nms_indices_elements_) * sizeof(int32_t);
    bindings[static_cast<size_t>(nms_input_index_)] = const_cast<void*>(detection_output_device);
    bindings[static_cast<size_t>(nms_boxes_index_)] = nms_boxes_device_;
    bindings[static_cast<size_t>(nms_scores_index_)] = nms_scores_device_;
    bindings[static_cast<size_t>(nms_max_output_index_)] = constants;
    bindings[static_cast<size_t>(nms_iou_threshold_index_)] = constants + sizeof(int32_t);
    bindings[static_cast<size_t>(nms_score_threshold_index_)] = constants + sizeof(int32_t) + sizeof(uint16_t);
    bindings[static_cast<size_t>(nms_indices_index_)] = nms_outputs_device_;
    bindings[static_cast<size_t>(nms_count_index_)] = static_cast<unsigned char*>(nms_outputs_device_) + indices_bytes;
    auto stage_start = ProfileClock::now();
    // All models share this NMS graph and original-anchor map contract. The
    // input preparation policy is selected once for SDK compatibility.
    auto* map = reinterpret_cast<int32_t*>(static_cast<unsigned char*>(nms_outputs_device_) +
        indices_bytes + size_t(nms_count_elements_) * sizeof(int32_t));
    if (nms_requires_compaction_) {
        if (!compactRppNmsInput(detection_output_device, map)) return false;
        bindings[static_cast<size_t>(nms_input_index_)] = nms_compact_device_;
    }
    if (!nms_context_->enqueue(1, bindings.data(), stream_, nullptr) ||
        !check_rt(rtStreamSynchronize(stream_), "rtStreamSynchronize YOLOv11-SEG RT-NMS")) {
        return false;
    }
    if (profile != nullptr) {
        profile->nms_ms = elapsed_ms(stage_start);
        profile->candidate_decode_ms = profile->nms_ms;
    }

    stage_start = ProfileClock::now();
    if (!check_rt(rtMemcpyAsync(nms_outputs_host_, nms_outputs_device_, nms_outputs_bytes_,
                                rtMemcpyDeviceToHost, stream_), "rtMemcpyAsync RT-NMS outputs") ||
        !check_rt(rtMemcpyAsync(nms_boxes_host_, nms_boxes_device_, nms_boxes_bytes_,
                                rtMemcpyDeviceToHost, stream_), "rtMemcpyAsync RT-NMS boxes") ||
        !check_rt(rtMemcpyAsync(nms_scores_host_, nms_scores_device_, nms_scores_bytes_,
                                rtMemcpyDeviceToHost, stream_), "rtMemcpyAsync RT-NMS scores") ||
        !check_rt(rtStreamSynchronize(stream_), "rtStreamSynchronize RT-NMS D2H")) {
        return false;
    }
    // NMS uses BF16 internally; restore selected boxes from the original FP32
    // coordinates to avoid a second rounding of small source-image ROIs.
    const size_t fp32_box_bytes = size_t(detection_layout_.anchors) * 4 * sizeof(float);
    if (!check_rt(rtMemcpy(host_det_head_.data(), detection_output_device, fp32_box_bytes,
                           rtMemcpyDeviceToHost), "original FP32 boxes D2H")) return false;
    if (profile != nullptr) {
        profile->split_d2h_ms = elapsed_ms(stage_start);
        profile->device_to_host_ms += profile->split_d2h_ms;
        profile->device_to_host_bytes += nms_outputs_bytes_ + nms_boxes_bytes_ + nms_scores_bytes_;
        profile->device_to_host_bytes += fp32_box_bytes;
    }
    return parseRppNmsResults(letterbox, candidates, keep);
}

bool YoloV11SegPostprocessor::parseRppNmsResults(const LetterboxInfo& letterbox,
                                                  std::vector<Candidate>& candidates,
                                                  std::vector<int>& keep) const
{
    const auto* indices = static_cast<const int32_t*>(nms_outputs_host_);
    const auto* count = reinterpret_cast<const int32_t*>(
        static_cast<const unsigned char*>(nms_outputs_host_) +
        static_cast<size_t>(nms_indices_elements_) * sizeof(int32_t));
    const auto* scores = static_cast<const uint16_t*>(nms_scores_host_);
    const int capacity = nms_indices_elements_ / nms_index_tuple_size_;
    if (count[0] < 0 || count[0] > capacity) {
        std::cerr << "Invalid device NMS count: " << count[0] << " (capacity " << capacity << ")" << std::endl;
        return false;
    }
    const int selected = std::max(0, std::min({count[0], capacity, config_.max_segments}));
    candidates.reserve(static_cast<size_t>(selected));
    keep.reserve(static_cast<size_t>(selected));
    for (int i = 0; i < selected; ++i) {
        const size_t tuple = static_cast<size_t>(i) * static_cast<size_t>(nms_index_tuple_size_);
        int class_id = indices[tuple + static_cast<size_t>(nms_index_tuple_size_ - 2)];
        const int score_anchor = indices[tuple + static_cast<size_t>(nms_index_tuple_size_ - 1)];
        if (score_anchor < 0 || score_anchor >= nms_anchor_map_elements_) return false;
        const int anchor = count[nms_count_elements_ + score_anchor];
        if (anchor < 0 || anchor >= detection_layout_.anchors) {
            continue;
        }
        const uint16_t* anchor_scores = scores + static_cast<size_t>(score_anchor) * config_.class_count;
        if (config_.agnostic) {
            class_id = 0;
            for (int candidate = 1; candidate < config_.class_count; ++candidate) {
                if (bf16_to_float(anchor_scores[candidate]) > bf16_to_float(anchor_scores[class_id])) {
                    class_id = candidate;
                }
            }
        }
        if (class_id < 0 || class_id >= config_.class_count) {
            continue;
        }
        const int n = detection_layout_.anchors;
        const float cx = host_det_head_[anchor];
        const float cy = host_det_head_[n + anchor];
        const float width = host_det_head_[2*n + anchor];
        const float height = host_det_head_[3*n + anchor];
        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(width) || !std::isfinite(height)) {
            std::cerr << "Non-finite selected box from inference at anchor " << anchor << std::endl;
            return false;
        }
        Candidate result;
        result.anchor = anchor;
        result.class_id = class_id;
        result.score = bf16_to_float(anchor_scores[class_id]);
        result.box = clip_rect_to_image(
            (cx - width * 0.5f - letterbox.pad_x) / letterbox.scale,
            (cy - height * 0.5f - letterbox.pad_y) / letterbox.scale,
            (cx + width * 0.5f - letterbox.pad_x) / letterbox.scale,
            (cy + height * 0.5f - letterbox.pad_y) / letterbox.scale,
            letterbox.source_width, letterbox.source_height);
        if (!result.box.empty() && result.score >= config_.score_threshold) {
            keep.push_back(static_cast<int>(candidates.size()));
            candidates.push_back(result);
        }
    }
    return true;
}

void YoloV11SegPostprocessor::resetRppNmsEngine()
{
    if (nms_boxes_device_ != nullptr) rtFree(nms_boxes_device_);
    if (nms_scores_device_ != nullptr) rtFree(nms_scores_device_);
    if (nms_constants_device_ != nullptr) rtFree(nms_constants_device_);
    if (nms_outputs_device_ != nullptr) rtFree(nms_outputs_device_);
    if (nms_compact_device_ != nullptr) rtFree(nms_compact_device_);
    nms_compact_device_ = nullptr;
    if (nms_boxes_host_ != nullptr) rtFreeHost(nms_boxes_host_);
    if (nms_scores_host_ != nullptr) rtFreeHost(nms_scores_host_);
    if (nms_outputs_host_ != nullptr) rtFreeHost(nms_outputs_host_);
    nms_boxes_device_ = nms_scores_device_ = nms_constants_device_ = nms_outputs_device_ = nullptr;
    nms_boxes_host_ = nms_scores_host_ = nms_outputs_host_ = nullptr;
    nms_context_.reset();
    nms_engine_.reset();
}

bool YoloV11SegPostprocessor::resolveDetectionLayout()
{
    // One documented export layout: optional batch 1 followed by [116,N].
    auto shape = compact_dims(config_.detection_dimensions);
    if (shape.size() == 3 && shape[0] == 1) shape.erase(shape.begin());
    if (shape.size() != 2 || shape[0] != total_head_channels_ || shape[1] <= 0 ||
        size_t(shape[0]) * size_t(shape[1]) != config_.detection_elements) return false;
    detection_layout_.anchors = shape[1];
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
    size_t selected_indices_bytes = static_cast<size_t>(segment_count) * sizeof(int);

    host_selected_anchor_indices_.assign(static_cast<size_t>(segment_count), 0);
    host_selected_box_.assign(static_cast<size_t>(segment_count) * 4U, 0);

    // RT NMS supplies original-anchor indices for the shared coefficient gather.
    size_t packed_mask_elements = 0;
    host_mask_offsets_.assign(static_cast<size_t>(segment_count), 0);
    std::vector<int> tile_segments;
    std::vector<int> tile_source_y;
    std::vector<int> tile_source_x;
    std::vector<int> tile_output;
    std::vector<int> tile_pixels;
    for (int i = 0; i < segment_count; ++i) {
        const Candidate& candidate = candidates[static_cast<size_t>(keep[static_cast<size_t>(i)])];
        host_selected_anchor_indices_[static_cast<size_t>(i)] = candidate.anchor;
        host_mask_offsets_[static_cast<size_t>(i)] = static_cast<int>(packed_mask_elements);
        size_t box_offset = static_cast<size_t>(i) * 4U;
        host_selected_box_[box_offset + 0U] = candidate.box.x;
        host_selected_box_[box_offset + 1U] = candidate.box.y;
        host_selected_box_[box_offset + 2U] = candidate.box.width;
        host_selected_box_[box_offset + 3U] = candidate.box.height;
        packed_mask_elements += static_cast<size_t>(candidate.box.width) *
                                static_cast<size_t>(candidate.box.height);
        constexpr int tile_width = 256;
        for (int local_y = 0; local_y < candidate.box.height; ++local_y) {
            for (int local_x = 0; local_x < candidate.box.width; local_x += tile_width) {
                tile_segments.push_back(i);
                tile_source_y.push_back(candidate.box.y + local_y);
                tile_source_x.push_back(candidate.box.x + local_x);
                tile_output.push_back(host_mask_offsets_[static_cast<size_t>(i)] +
                                      local_y * candidate.box.width + local_x);
                tile_pixels.push_back(std::min(tile_width, candidate.box.width - local_x));
            }
        }
    }
    size_t source_mask_bytes = packed_mask_elements * sizeof(uint8_t);
    const size_t selected_indices_offset = 0;
    const size_t source_x0_offset = selected_indices_offset + selected_indices_bytes;
    const size_t source_x0_bytes = static_cast<size_t>(letterbox.source_width) * yolo11::kMaskResizeTaps * sizeof(int);
    const size_t source_wx_offset = source_x0_offset + source_x0_bytes;
    const size_t source_wx_bytes = static_cast<size_t>(letterbox.source_width) * yolo11::kMaskResizeTaps * sizeof(float);
    const size_t source_y0_offset = source_wx_offset + source_wx_bytes;
    const size_t source_y0_bytes = static_cast<size_t>(letterbox.source_height) * yolo11::kMaskResizeTaps * sizeof(int);
    const size_t source_wy_offset = source_y0_offset + source_y0_bytes;
    const size_t source_wy_bytes = static_cast<size_t>(letterbox.source_height) * yolo11::kMaskResizeTaps * sizeof(float);
    const size_t tile_field_bytes = tile_segments.size() * sizeof(int);
    const size_t tile_segments_offset = source_wy_offset + source_wy_bytes;
    const size_t tile_source_y_offset = tile_segments_offset + tile_field_bytes;
    const size_t tile_source_x_offset = tile_source_y_offset + tile_field_bytes;
    const size_t tile_output_offset = tile_source_x_offset + tile_field_bytes;
    const size_t tile_pixels_offset = tile_output_offset + tile_field_bytes;
    const size_t base_metadata_bytes = tile_pixels_offset + tile_field_bytes;

    const int pad_x = static_cast<int>(std::round(letterbox.pad_x));
    const int pad_y = static_cast<int>(std::round(letterbox.pad_y));
    std::vector<int> coordinate_lut_key = {
        letterbox.source_width, letterbox.source_height,
        letterbox.resized_width, letterbox.resized_height,
        letterbox.model_width, letterbox.model_height,
        prototype_width_, prototype_height_, pad_x, pad_y};
    if (host_coordinate_lut_key_ != coordinate_lut_key) {
        host_coordinate_lut_key_ = coordinate_lut_key;
        yolo11::buildMaskResizeAxis(letterbox.source_width,letterbox.resized_width,letterbox.model_width,
                  prototype_width_,pad_x,host_source_x0_lut_,host_source_wx_lut_);
        yolo11::buildMaskResizeAxis(letterbox.source_height,letterbox.resized_height,letterbox.model_height,
                  prototype_height_,pad_y,host_source_y0_lut_,host_source_wy_lut_);
    }

    size_t sparse_proto_pixels = 0;
    std::vector<int> proto_roi_x(static_cast<size_t>(segment_count));
    std::vector<int> proto_roi_y(static_cast<size_t>(segment_count));
    std::vector<int> proto_roi_width(static_cast<size_t>(segment_count));
    std::vector<int> proto_roi_offset(static_cast<size_t>(segment_count));
    std::vector<int> sparse_point_segments;
    std::vector<int> sparse_point_proto_indices;
    for (int i = 0; i < segment_count; ++i) {
        const size_t box_offset = static_cast<size_t>(i) * 4U;
        const int box_x0 = host_selected_box_[box_offset + 0U];
        const int box_y0 = host_selected_box_[box_offset + 1U];
        const int box_x1 = box_x0 + host_selected_box_[box_offset + 2U] - 1;
        const int box_y1 = box_y0 + host_selected_box_[box_offset + 3U] - 1;
        const int proto_x0 = host_source_x0_lut_[static_cast<size_t>(box_x0)*yolo11::kMaskResizeTaps];
        const int proto_y0 = host_source_y0_lut_[static_cast<size_t>(box_y0)*yolo11::kMaskResizeTaps];
        // Include every tap, including zero-weight halo taps read by SIMD lanes.
        const int proto_x1 = host_source_x0_lut_[static_cast<size_t>(box_x1)*yolo11::kMaskResizeTaps+2];
        const int proto_y1 = host_source_y0_lut_[static_cast<size_t>(box_y1)*yolo11::kMaskResizeTaps+2];
        const int proto_width = proto_x1 - proto_x0 + 1;
        const int proto_height = proto_y1 - proto_y0 + 1;
        proto_roi_x[static_cast<size_t>(i)] = proto_x0;
        proto_roi_y[static_cast<size_t>(i)] = proto_y0;
        proto_roi_width[static_cast<size_t>(i)] = proto_width;
        proto_roi_offset[static_cast<size_t>(i)] = static_cast<int>(sparse_proto_pixels);
        for (int y = proto_y0; y <= proto_y1; ++y) {
            for (int x = proto_x0; x <= proto_x1; ++x) {
                sparse_point_segments.push_back(i);
                sparse_point_proto_indices.push_back(y * prototype_width_ + x);
            }
        }
        sparse_proto_pixels += static_cast<size_t>(proto_width) *
                               static_cast<size_t>(proto_height);
    }
    if (profile != nullptr) {
        profile->mask_full_proto_pixels = static_cast<size_t>(segment_count) *
                                          raw_mask_elements_per_segment;
        profile->mask_sparse_proto_pixels = sparse_proto_pixels;
        profile->mask_output_roi_pixels = packed_mask_elements;
        profile->mask_sparse_proto_coverage = profile->mask_full_proto_pixels > 0 ?
            static_cast<double>(sparse_proto_pixels) /
            static_cast<double>(profile->mask_full_proto_pixels) : 0.0;
    }

    raw_masks_bytes = sparse_proto_pixels * sizeof(float);
    const size_t sparse_point_field_bytes = sparse_proto_pixels * sizeof(int);
    const size_t sparse_point_segments_offset = base_metadata_bytes;
    const size_t sparse_point_indices_offset = sparse_point_segments_offset + sparse_point_field_bytes;
    const size_t sparse_tile_raw_row0_offset = sparse_point_indices_offset + sparse_point_field_bytes;
    const size_t sparse_tile_raw_row1_offset = sparse_tile_raw_row0_offset + tile_field_bytes;
    const size_t metadata_bytes = sparse_tile_raw_row1_offset + tile_field_bytes;

    std::vector<int> sparse_tile_raw_row0(tile_segments.size());
    std::vector<int> sparse_tile_raw_row1(tile_segments.size());
    for (size_t tile = 0; tile < tile_segments.size(); ++tile) {
        const int segment = tile_segments[tile];
        const int roi_y = proto_roi_y[static_cast<size_t>(segment)];
        const int roi_x = proto_roi_x[static_cast<size_t>(segment)];
        const int roi_width = proto_roi_width[static_cast<size_t>(segment)];
        const int roi_offset = proto_roi_offset[static_cast<size_t>(segment)];
        // Every tile addresses the same packed sparse-prototype contract.
        sparse_tile_raw_row0[tile] = roi_offset - roi_y * roi_width - roi_x;
        sparse_tile_raw_row1[tile] = roi_width;
    }

    if (!ensureSramBuffer(&prototype_sram_, &prototype_sram_capacity_, prototype_bytes) ||
        !ensureSramBuffer(&coefficients_sram_, &coefficients_sram_capacity_, coeff_bytes) ||
        !ensureSramBuffer(&raw_masks_sram_, &raw_masks_sram_capacity_, raw_masks_bytes) ||
        !ensureSramBuffer(&metadata_sram_, &metadata_sram_capacity_, metadata_bytes) ||
        !ensureSramBuffer(&source_mask_sram_, &source_mask_sram_capacity_, source_mask_bytes + 1U) ||
        !ensureHostPinnedBuffer(&host_metadata_pinned_,
                                &host_metadata_pinned_capacity_,
                                metadata_bytes) ||
        !ensureHostPinnedBuffer(&host_source_mask_pinned_,
                                &host_source_mask_pinned_capacity_,
                                source_mask_bytes)) {
        return false;
    }
    auto* host_metadata = static_cast<uint8_t*>(host_metadata_pinned_);
    std::memcpy(host_metadata + selected_indices_offset,
                host_selected_anchor_indices_.data(), selected_indices_bytes);
    std::memcpy(host_metadata + tile_segments_offset, tile_segments.data(), tile_field_bytes);
    std::memcpy(host_metadata + tile_source_y_offset, tile_source_y.data(), tile_field_bytes);
    std::memcpy(host_metadata + tile_source_x_offset, tile_source_x.data(), tile_field_bytes);
    std::memcpy(host_metadata + tile_output_offset, tile_output.data(), tile_field_bytes);
    std::memcpy(host_metadata + tile_pixels_offset, tile_pixels.data(), tile_field_bytes);
    std::memcpy(host_metadata + source_x0_offset, host_source_x0_lut_.data(), source_x0_bytes);
    std::memcpy(host_metadata + source_wx_offset, host_source_wx_lut_.data(), source_wx_bytes);
    std::memcpy(host_metadata + source_y0_offset, host_source_y0_lut_.data(), source_y0_bytes);
    std::memcpy(host_metadata + source_wy_offset, host_source_wy_lut_.data(), source_wy_bytes);
    std::memcpy(host_metadata + sparse_point_segments_offset,
                sparse_point_segments.data(), sparse_point_field_bytes);
    std::memcpy(host_metadata + sparse_point_indices_offset,
                sparse_point_proto_indices.data(), sparse_point_field_bytes);
    std::memcpy(host_metadata + sparse_tile_raw_row0_offset,
                sparse_tile_raw_row0.data(), tile_field_bytes);
    std::memcpy(host_metadata + sparse_tile_raw_row1_offset,
                sparse_tile_raw_row1.data(), tile_field_bytes);

    auto stage_start = ProfileClock::now();
    auto detail_start = ProfileClock::now();
    bool ok = check_rt(rtMemcpy(prototype_sram_, prototype_output_device, prototype_bytes,
                               rtMemcpyDeviceToSram), "rtMemcpy prototype DDR to SRAM");
    if (profile != nullptr) {
        profile->mask_prototype_d2s_ms = elapsed_ms(detail_start);
    }
    const bool use_timing_events = mask_event_start_ != nullptr;
    if (ok && use_timing_events) {
        ok = check_rt(rtEventRecord(mask_event_setup_start_, stream_),
                      "rtEventRecord mask setup start");
    }
    ok = ok && check_rpp(rppMemcpyHtoSAsync(
                            reinterpret_cast<RPPdeviceptr>(metadata_sram_),
                            host_metadata_pinned_,
                            metadata_bytes,
                            stream_),
                        "rppMemcpyHtoSAsync packed postprocess metadata Host to SRAM");
    if (profile != nullptr) {
        profile->mask_setup_copy_ms = elapsed_ms(detail_start);
    }
    auto fused_start = ProfileClock::now();
    if (ok && use_timing_events) {
        ok = check_rt(rtEventRecord(mask_event_start_, stream_), "rtEventRecord mask start");
    }
    if (ok) {
        detail_start = ProfileClock::now();
        const auto* metadata_sram = static_cast<const uint8_t*>(metadata_sram_);
        launch_yolov11_seg_gather_selected_coefficients(stream_,
                                                        static_cast<const float*>(split_input_sram_),
                                                        reinterpret_cast<const int*>(metadata_sram + selected_indices_offset),
                                                        static_cast<float*>(coefficients_sram_),
                                                        segment_count,
                                                        config_.mask_coeff_count,
                                                        detection_layout_.anchors);
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG coefficient gather kernel launch");
        if (ok && use_timing_events) {
            ok = check_rt(rtEventRecord(mask_event_gather_end_, stream_), "rtEventRecord gather end");
        }
        if (profile != nullptr && !use_timing_events) {
            profile->mask_coeff_gather_ms = elapsed_ms(detail_start);
        }
    }
    if (ok) {
        detail_start = ProfileClock::now();
        const auto* metadata_sram = static_cast<const uint8_t*>(metadata_sram_);
        launch_yolov11_seg_combine_prototypes_sparse(
            stream_, static_cast<const float*>(prototype_sram_),
            static_cast<const float*>(coefficients_sram_),
            reinterpret_cast<const int*>(metadata_sram + sparse_point_segments_offset),
            reinterpret_cast<const int*>(metadata_sram + sparse_point_indices_offset),
            static_cast<float*>(raw_masks_sram_), static_cast<int>(sparse_proto_pixels),
            config_.mask_coeff_count, prototype_height_, prototype_width_);
        ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG prototype combine kernel launch");
        if (ok && use_timing_events) {
            ok = check_rt(rtEventRecord(mask_event_combine_end_, stream_), "rtEventRecord combine end");
        }
        if (profile != nullptr && !use_timing_events) {
            profile->mask_proto_combine_ms = elapsed_ms(detail_start);
        }
    }
    if (!ok) {
        return false;
    }
    detail_start = ProfileClock::now();
    const auto* metadata_sram = static_cast<const uint8_t*>(metadata_sram_);
    launch_yolov11_seg_resample_probabilities(
        stream_, static_cast<const float*>(raw_masks_sram_),
        reinterpret_cast<const int*>(metadata_sram + sparse_tile_raw_row0_offset),
        reinterpret_cast<const int*>(metadata_sram + sparse_tile_raw_row1_offset),
        reinterpret_cast<const int*>(metadata_sram + tile_source_y_offset),
        reinterpret_cast<const int*>(metadata_sram + tile_source_x_offset),
        reinterpret_cast<const int*>(metadata_sram + tile_output_offset),
        reinterpret_cast<const int*>(metadata_sram + tile_pixels_offset),
        reinterpret_cast<const int*>(metadata_sram + source_x0_offset),
        reinterpret_cast<const float*>(metadata_sram + source_wx_offset),
        reinterpret_cast<const int*>(metadata_sram + source_y0_offset),
        reinterpret_cast<const float*>(metadata_sram + source_wy_offset),
        static_cast<unsigned char*>(source_mask_sram_), static_cast<int>(source_mask_bytes),
        static_cast<int>(tile_segments.size()), config_.mask_threshold);
    ok = check_rt(rtGetLastError(), "RPP YOLOv11-SEG batched uint8 ROI resize/crop kernel launch") &&
         (!use_timing_events || check_rt(rtEventRecord(mask_event_resize_end_, stream_),
                                        "rtEventRecord resize end")) &&
         check_rt(rtStreamSynchronize(stream_), "rtStreamSynchronize fused mask kernels");
    if (profile != nullptr) {
        profile->mask_fused_kernel_ms = elapsed_ms(fused_start);
        if (use_timing_events) {
            float metadata_ms = 0.0f;
            float gather_ms = 0.0f;
            float combine_ms = 0.0f;
            float resize_ms = 0.0f;
            if (check_rt(rtEventElapsedTime(&metadata_ms, mask_event_setup_start_, mask_event_start_),
                         "rtEventElapsedTime selected metadata H2S") &&
                check_rt(rtEventElapsedTime(&gather_ms, mask_event_start_, mask_event_gather_end_),
                         "rtEventElapsedTime coefficient gather") &&
                check_rt(rtEventElapsedTime(&combine_ms, mask_event_gather_end_, mask_event_combine_end_),
                         "rtEventElapsedTime prototype combine") &&
                check_rt(rtEventElapsedTime(&resize_ms, mask_event_combine_end_, mask_event_resize_end_),
                         "rtEventElapsedTime ROI resize")) {
                profile->mask_metadata_h2s_ms = metadata_ms;
                profile->mask_coeff_gather_ms = gather_ms;
                profile->mask_proto_combine_ms = combine_ms;
                profile->mask_source_resize_crop_ms = resize_ms;
                profile->mask_device_pipeline_ms = metadata_ms + gather_ms + combine_ms + resize_ms;
                profile->mask_runtime_overhead_ms = std::max(
                    0.0, profile->mask_fused_kernel_ms - profile->mask_device_pipeline_ms);
                profile->mask_setup_copy_ms =
                    profile->mask_prototype_d2s_ms + profile->mask_metadata_h2s_ms;
            }
        }
        else {
            profile->mask_source_resize_crop_ms = profile->mask_fused_kernel_ms;
        }
    }
    if (ok) {
        detail_start = ProfileClock::now();
        ok = check_rpp(rppMemcpyStoHAsync(
                           host_source_mask_pinned_,
                           reinterpret_cast<RPPdeviceptr>(source_mask_sram_),
                           source_mask_bytes,
                           stream_),
                       "rppMemcpyStoHAsync packed box masks SRAM to Host") &&
             check_rt(rtStreamSynchronize(stream_), "rtStreamSynchronize packed box mask S2H");
        if (profile != nullptr) {
            profile->mask_source_copy_ms = elapsed_ms(detail_start);
        }
    }
    if (!ok) {
        return false;
    }

    segments.reserve(static_cast<size_t>(segment_count));
    for (int i = 0; i < segment_count; ++i) {
        const Candidate& candidate = candidates[static_cast<size_t>(keep[static_cast<size_t>(i)])];
        Segmentation segment;
        segment.class_id = candidate.class_id;
        segment.confidence = candidate.score;
        segment.box = candidate.box;
        detail_start = ProfileClock::now();
        buildMaskFromBoxBinary(static_cast<const uint8_t*>(host_source_mask_pinned_) +
                                   host_mask_offsets_[static_cast<size_t>(i)],
                               candidate.box.width,
                               candidate.box.height,
                               segment);
        if (profile != nullptr) {
            profile->mask_build_cpu_ms += elapsed_ms(detail_start);
        }
        if (!segment.mask.empty()) {
            segments.push_back(std::move(segment));
        }
    }
    if (profile != nullptr) {
        profile->mask_decode_ms = elapsed_ms(stage_start);
        profile->device_to_host_ms += profile->mask_source_copy_ms;
        profile->device_to_host_bytes += source_mask_bytes;
    }
    return true;
}

void YoloV11SegPostprocessor::buildMaskFromBoxBinary(const uint8_t* box_mask,
                                                     int box_width,
                                                     int box_height,
                                                     Segmentation& segment) const
{
    if (box_mask == nullptr || box_width <= 0 || box_height <= 0) {
        return;
    }

    cv::Mat box_u8(box_height,
                   box_width,
                   CV_8UC1,
                   const_cast<uint8_t*>(box_mask));
    if (cv::countNonZero(box_u8) == 0) {
        return;
    }

    segment.mask_width = box_u8.cols;
    segment.mask_height = box_u8.rows;
    segment.mask.assign(box_u8.datastart, box_u8.dataend);
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

bool YoloV11SegPostprocessor::ensureHostPinnedBuffer(void** buffer,
                                                     size_t* capacity,
                                                     size_t required_bytes)
{
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFreeHost(*buffer), "rtFreeHost postprocess pinned buffer")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtHostAlloc(buffer, required_bytes, rtHostAllocDefault),
                  "rtHostAlloc postprocess pinned buffer")) {
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
    if (coefficients_sram_ != nullptr) {
        rtFreeSram(coefficients_sram_);
        coefficients_sram_ = nullptr;
    }
    if (raw_masks_sram_ != nullptr) {
        rtFreeSram(raw_masks_sram_);
        raw_masks_sram_ = nullptr;
    }
    if (metadata_sram_ != nullptr) {
        rtFreeSram(metadata_sram_);
        metadata_sram_ = nullptr;
    }
    if (source_mask_sram_ != nullptr) {
        rtFreeSram(source_mask_sram_);
        source_mask_sram_ = nullptr;
    }
    if (split_input_sram_ != nullptr) {
        rtFreeSram(split_input_sram_);
        split_input_sram_ = nullptr;
    }
    if (host_source_mask_pinned_ != nullptr) {
        rtFreeHost(host_source_mask_pinned_);
        host_source_mask_pinned_ = nullptr;
    }
    if (host_metadata_pinned_ != nullptr) {
        rtFreeHost(host_metadata_pinned_);
        host_metadata_pinned_ = nullptr;
    }
    prototype_sram_capacity_ = 0;
    coefficients_sram_capacity_ = 0;
    raw_masks_sram_capacity_ = 0;
    metadata_sram_capacity_ = 0;
    source_mask_sram_capacity_ = 0;
    split_input_sram_capacity_ = 0;
    host_source_mask_pinned_capacity_ = 0;
    host_metadata_pinned_capacity_ = 0;
}
