/**
 * @file yolo_pipeline.cpp
 * @brief Reusable YOLOv11-SEG pipeline implementation.
 */
#include "yolo_pipeline.h"

#include "perf_trace_session.h"
#include "rpp_infer_engine.h"
#include "rpp_yolov11_seg_postprocessor.h"
#include "rpp_yolo_preprocessor.h"

#include <Infer.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <iostream>

namespace
{
/**
 * @brief Fill runtime YOLOv11-SEG tensor shape fields from the initialized inference engine.
 */
bool dims_contains(const infer1::Dims& dims, int value)
{
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] == value) {
            return true;
        }
    }
    return false;
}

bool looks_like_prototype(const infer1::Dims& dims, int mask_coeff_count)
{
    return dims.nbDims >= 3 && dims_contains(dims, mask_coeff_count);
}

bool update_config_from_model(const RppInferEngine& infer_engine,
                              YoloV11SegConfig& config,
                              int& detection_output_index,
                              int& prototype_output_index)
{
    if (infer_engine.getInputDataType() != infer1::DataType::kFLOAT) {
        std::cerr << "The current YOLOv11-SEG path only supports float input tensors." << std::endl;
        return false;
    }

    config.input_width = infer_engine.getInputWidth();
    config.input_height = infer_engine.getInputHeight();
    detection_output_index = -1;
    prototype_output_index = -1;

    const auto& outputs = infer_engine.getOutputInfos();
    const int expected_detection_channels = 4 + config.class_count + config.mask_coeff_count;
    for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
        if (outputs[static_cast<size_t>(i)].data_type != infer1::DataType::kFLOAT) {
            std::cerr << "The current YOLOv11-SEG path only supports float output tensors." << std::endl;
            return false;
        }
        const infer1::Dims& dims = outputs[static_cast<size_t>(i)].dimensions;
        if (detection_output_index < 0 && dims_contains(dims, expected_detection_channels)) {
            detection_output_index = i;
        }
        else if (prototype_output_index < 0 && looks_like_prototype(dims, config.mask_coeff_count)) {
            prototype_output_index = i;
        }
    }

    if (detection_output_index < 0 || prototype_output_index < 0 ||
        detection_output_index == prototype_output_index) {
        std::cerr << "Unable to identify YOLOv11-SEG detection/prototype outputs." << std::endl;
        return false;
    }

    const auto& detection_output = outputs[static_cast<size_t>(detection_output_index)];
    const auto& prototype_output = outputs[static_cast<size_t>(prototype_output_index)];
    config.detection_dimensions = detection_output.dimensions;
    config.detection_elements = detection_output.elements;
    config.prototype_dimensions = prototype_output.dimensions;
    config.prototype_elements = prototype_output.elements;

    if (config.input_width <= 0 || config.input_height <= 0 ||
        config.detection_elements == 0 || config.prototype_elements == 0) {
        std::cerr << "Invalid YOLOv11-SEG model tensor dimensions." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Measure elapsed milliseconds between two host timestamps.
 */
double elapsed_ms(const std::chrono::high_resolution_clock::time_point& start,
                  const std::chrono::high_resolution_clock::time_point& stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}
}

/**
 * @brief Construct an empty pipeline; call init() before running frames.
 */
YoloV11SegPipeline::YoloV11SegPipeline() = default;
/**
 * @brief Destroy module owners and release runtime resources.
 */
YoloV11SegPipeline::~YoloV11SegPipeline() = default;

/**
 * @brief Initialize model runtime, derive tensor shapes, and build postprocessing.
 */
bool YoloV11SegPipeline::init(const YoloV11SegPipelineOptions& options, const std::vector<std::string>& class_names)
{
    PERF_SCOPE_CATE("yolov11_pipeline_init", "yolov11");
    // Preserve caller options, but enforce at least one inference pass per pipeline run.
    options_ = options;
    if (options_.inference_count <= 0) {
        options_.inference_count = 1;
    }
    config_ = options.config;
    if (!class_names.empty()) {
        config_.class_count = static_cast<int>(class_names.size());
    }

    // Build the model engine first so input/output dimensions can drive downstream modules.
    infer_engine_ = std::make_unique<RppInferEngine>(options.model_path);
    if (!infer_engine_->init()) {
        std::cerr << "Failed to initialize RPP inference engine." << std::endl;
        return false;
    }

    if (!update_config_from_model(*infer_engine_, config_, detection_output_index_, prototype_output_index_)) {
        return false;
    }
    // YOLOv11-SEG postprocessing depends on the final output tensor shape and class count.
    if (!initPostprocessor(class_names)) {
        return false;
    }

    initialized_ = true;
    return true;
}

/**
 * @brief Build the YOLOv11-SEG postprocessor after model output dimensions are known.
 */
bool YoloV11SegPipeline::initPostprocessor(const std::vector<std::string>& class_names)
{
    PERF_SCOPE_CATE("yolov11_pipeline_init_postprocess", "postprocess");
    if (!class_names.empty()) {
        config_.class_count = static_cast<int>(class_names.size());
    }

    postprocessor_ = std::make_unique<YoloV11SegPostprocessor>();
    return postprocessor_->init(config_);
}

/**
 * @brief Create or reuse a preprocessor sized for the requested source image.
 */
bool YoloV11SegPipeline::ensurePreprocessor(int width, int height)
{
    if (preprocessor_ != nullptr &&
        preprocessor_width_ >= width &&
        preprocessor_height_ >= height) {
        return true;
    }

    PERF_SCOPE_CATE("yolov11_pipeline_init_preprocess", "preprocess");
    preprocessor_ = std::make_unique<RppYoloPreprocessor>();
    if (!preprocessor_->init(width, height, config_.input_width, config_.input_height)) {
        preprocessor_.reset();
        return false;
    }
    preprocessor_width_ = width;
    preprocessor_height_ = height;
    return true;
}

/**
 * @brief Run a host BGR OpenCV image through the complete YOLOv11-SEG pipeline.
 */
bool YoloV11SegPipeline::runRgb(const cv::Mat& image,
                            std::vector<Segmentation>& segments,
                            YoloV11SegStageTimes* times)
{
    if (image.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return false;
    }

    // RPP preprocessing expects a contiguous host buffer.
    cv::Mat continuous_image = image.isContinuous() ? image : image.clone();
    return runInput(continuous_image.data,
                    continuous_image.total() * continuous_image.elemSize(),
                    continuous_image.cols,
                    continuous_image.rows,
                    static_cast<int>(continuous_image.step[0]),
                    continuous_image.channels(),
                    continuous_image.isContinuous(),
                    static_cast<int>(PreprocessInputFormat::BGR_HWC),
                    segments,
                    times);
}

/**
 * @brief Run a host I420 frame through the complete YOLOv11-SEG pipeline.
 */
bool YoloV11SegPipeline::runI420(const void* yuv_data,
                             size_t yuv_bytes,
                             int width,
                             int height,
                             std::vector<Segmentation>& segments,
                             YoloV11SegStageTimes* times)
{
    return runInput(yuv_data,
                    yuv_bytes,
                    width,
                    height,
                    width,
                    1,
                    true,
                    static_cast<int>(PreprocessInputFormat::YUV_I420),
                    segments,
                    times);
}

/**
 * @brief Shared input path used by RGB and YUV demos to execute one full pipeline pass.
 */
bool YoloV11SegPipeline::runInput(const void* data,
                              size_t bytes,
                              int width,
                              int height,
                              int row_stride_bytes,
                              int channels,
                              bool continuous,
                              int format,
                              std::vector<Segmentation>& segments,
                              YoloV11SegStageTimes* times)
{
    PERF_SCOPE_CATE("yolov11_pipeline_run", "yolov11");
    if (!initialized_ || infer_engine_ == nullptr || postprocessor_ == nullptr) {
        std::cerr << "YOLOv11-SEG pipeline has not been initialized." << std::endl;
        return false;
    }
    if (data == nullptr || width <= 0 || height <= 0 || bytes == 0) {
        std::cerr << "Invalid YOLOv11-SEG input." << std::endl;
        return false;
    }
    if (!continuous) {
        std::cerr << "Only continuous host input is supported by the current pipeline." << std::endl;
        return false;
    }
    // row_stride_bytes and channels are accepted to keep the call signature ready for future formats.
    (void)row_stride_bytes;
    (void)channels;

    // Each call owns its output vector and reports compute latency separately from transfer fields.
    segments.clear();
    YoloV11SegStageTimes local_times;

    LetterboxInfo letterbox;
    RppPreprocessProfile preprocess_profile;
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    {
        PERF_SCOPE_CATE("yolov11_pipeline_preprocess_stage", "preprocess");
        // Preprocessing workspaces are sized lazily to match the largest source seen so far.
        if (!ensurePreprocessor(width, height)) {
            return false;
        }

        PreprocessInput input;
        input.data = data;
        input.width = width;
        input.height = height;
        input.format = static_cast<PreprocessInputFormat>(format);
        input.memory_type = PreprocessMemoryType::Host;
        input.bytes = bytes;

        // RPP writes the normalized model tensor directly into the inference input binding.
        if (!preprocessor_->run(input,
                                infer_engine_->getInputDeviceBuffer(),
                                letterbox,
                                nullptr,
                                &preprocess_profile)) {
            return false;
        }
        // SRAM is temporary workspace; release it before postprocess needs its own runtime buffers.
        preprocessor_->releaseSramBuffers();
    }
    auto preprocess_stop = std::chrono::high_resolution_clock::now();
    local_times.preprocess_ms = preprocess_profile.total_ms > 0.0 ?
                                preprocess_profile.total_ms :
                                elapsed_ms(preprocess_start, preprocess_stop);
    local_times.input_h2d_ms = preprocess_profile.host_to_device_ms;
    local_times.input_h2d_bytes = preprocess_profile.host_to_device_bytes;

    auto inference_start = std::chrono::high_resolution_clock::now();
    {
        PERF_SCOPE_CATE("yolov11_pipeline_inference_stage", "inference");
        // Demos control warmup separately; this loop is the explicit per-run inference count.
        for (int i = 0; i < options_.inference_count; ++i) {
            if (!infer_engine_->execute()) {
                return false;
            }
        }
    }
    auto inference_stop = std::chrono::high_resolution_clock::now();
    local_times.inference_ms = elapsed_ms(inference_start, inference_stop);

    YoloV11SegPostprocessProfile postprocess_profile;
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    {
        PERF_SCOPE_CATE("yolov11_pipeline_postprocess_stage", "postprocess");
        // Postprocessing consumes detection/prototype output buffers and restores segments.
        if (!postprocessor_->run(infer_engine_->getOutputDeviceBuffer(static_cast<size_t>(detection_output_index_)),
                                 infer_engine_->getOutputDeviceBuffer(static_cast<size_t>(prototype_output_index_)),
                                 letterbox,
                                 segments,
                                 nullptr,
                                 &postprocess_profile)) {
            return false;
        }
    }
    auto postprocess_stop = std::chrono::high_resolution_clock::now();
    local_times.postprocess_ms = postprocess_profile.total_ms > 0.0 ?
                                 postprocess_profile.total_ms :
                                 elapsed_ms(postprocess_start, postprocess_stop);
    local_times.output_d2h_ms = postprocess_profile.device_to_host_ms;
    local_times.output_d2h_bytes = postprocess_profile.device_to_host_bytes;

    // End-to-end compute time follows the printed stage summary.
    double compute_ms = local_times.preprocess_ms +
                        local_times.inference_ms +
                        local_times.postprocess_ms;
    local_times.end_to_end_ms = compute_ms > 0.0 ? compute_ms : 0.0;

    if (times != nullptr) {
        *times = local_times;
    }
    return true;
}
