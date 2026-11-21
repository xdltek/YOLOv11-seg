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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>

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

// Explicit diagnostic mode only. Captures are not performance measurements.
bool capture_device_tensor(const std::filesystem::path& path, const void* device, size_t elements)
{
    std::vector<float> values(elements);
    if (rtMemcpy(values.data(), device, elements * sizeof(float), rtMemcpyDeviceToHost) != rtSuccess)
        return false;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), elements * sizeof(float));
    return static_cast<bool>(out);
}
}

/**
 * @brief Construct an empty pipeline; call init() before running frames.
 */
YoloV11SegPipeline::YoloV11SegPipeline() = default;
/**
 * @brief Destroy module owners and release runtime resources.
 */
YoloV11SegPipeline::~YoloV11SegPipeline()
{
    // Destroy stream consumers before releasing the pipeline-wide orchestration stream.
    postprocessor_.reset();
    preprocessor_.reset();
    infer_engine_.reset();
    if (stream_ != nullptr) {
        rtStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

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
    if (stream_ == nullptr && rtStreamCreate(&stream_) != rtError_t::rtSuccess) {
        std::cerr << "Failed to create YOLOv11-SEG pipeline stream." << std::endl;
        return false;
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
                                stream_,
                                &preprocess_profile)) {
            return false;
        }
        // The normalized tensor has already been copied to the inference DDR binding.
        // Release PRE SRAM before inference/POST so supported large tensors
        // have enough SRAM for their larger prototype and sparse-mask workspaces.
        preprocessor_->releaseSramBuffers();
    }
    auto preprocess_stop = std::chrono::high_resolution_clock::now();
    const double preprocess_total_ms = preprocess_profile.total_ms > 0.0 ?
                                       preprocess_profile.total_ms :
                                       elapsed_ms(preprocess_start, preprocess_stop);
    local_times.input_h2d_ms = preprocess_profile.host_to_device_ms;
    local_times.input_h2d_bytes = preprocess_profile.host_to_device_bytes;
    // RppYoloPreprocessor::total_ms starts after copyHostInputToDevice(), so it
    // already represents preprocess compute/internal movement with H2D excluded.
    local_times.preprocess_ms = std::max(0.0, preprocess_total_ms);

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

    std::filesystem::path capture_dir;
    if (const char* root = std::getenv("YOLO_VALIDATION_DIR")) {
        capture_dir = std::filesystem::path(root) / ("frame" + std::to_string(validation_frame_index_++));
        std::filesystem::create_directories(capture_dir);
        if (!capture_device_tensor(capture_dir / "input.f32", infer_engine_->getInputDeviceBuffer(),
                                    size_t(config_.input_width) * config_.input_height * 3) ||
            !capture_device_tensor(capture_dir / "prediction.f32",
                infer_engine_->getOutputDeviceBuffer(detection_output_index_), config_.detection_elements) ||
            !capture_device_tensor(capture_dir / "prototype.f32",
                infer_engine_->getOutputDeviceBuffer(prototype_output_index_), config_.prototype_elements))
            return false;
        if (infer_engine_->getBoxHeadDeviceBuffer() &&
            !capture_device_tensor(capture_dir / "dfl_input.f32", infer_engine_->getBoxHeadDeviceBuffer(),
                                   infer_engine_->getBoxHeadElements())) return false;
        std::ofstream meta(capture_dir / "geometry.txt");
        meta << std::setprecision(9);
        meta << letterbox.source_width << ' ' << letterbox.source_height << ' '
             << letterbox.model_width << ' ' << letterbox.model_height << ' '
             << letterbox.resized_width << ' ' << letterbox.resized_height << ' '
             << letterbox.pad_x << ' ' << letterbox.pad_y << ' ' << letterbox.scale << '\n';
        if (!meta) return false;
    }

    YoloV11SegPostprocessProfile postprocess_profile;
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    {
        PERF_SCOPE_CATE("yolov11_pipeline_postprocess_stage", "postprocess");
        // Postprocessing consumes detection/prototype output buffers and restores segments.
        if (!postprocessor_->run(infer_engine_->getOutputDeviceBuffer(static_cast<size_t>(detection_output_index_)),
                                 infer_engine_->getOutputDeviceBuffer(static_cast<size_t>(prototype_output_index_)),
                                 letterbox,
                                 segments,
                                 stream_,
                                 &postprocess_profile)) {
            return false;
        }
        // Release transient POST SRAM before the next frame's PRE allocation.
        // This is required when large model-input and prototype tensors cannot
        // coexist in the fixed SRAM capacity.
        postprocessor_->releaseSramBuffers();
    }
    auto postprocess_stop = std::chrono::high_resolution_clock::now();
    const double postprocess_total_ms = postprocess_profile.total_ms > 0.0 ?
                                        postprocess_profile.total_ms :
                                        elapsed_ms(postprocess_start, postprocess_stop);
    local_times.output_d2h_ms = postprocess_profile.device_to_host_ms;
    local_times.output_d2h_bytes = postprocess_profile.device_to_host_bytes;
    // Report postprocess compute/control separately from D2H. The runtime calls are
    // synchronous, so subtracting the measured D2H regions avoids double counting.
    local_times.postprocess_ms = std::max(0.0, postprocess_total_ms - local_times.output_d2h_ms);

    if (!capture_dir.empty()) {
        std::ofstream records(capture_dir / "segments.csv");
        records << std::setprecision(9);
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& s = segments[i];
            records << i << ',' << s.class_id << ',' << s.confidence << ','
                    << s.box.x << ',' << s.box.y << ',' << s.box.width << ',' << s.box.height
                    << ',' << s.mask_width << ',' << s.mask_height << '\n';
            std::ofstream mask(capture_dir / ("mask" + std::to_string(i) + ".u8"), std::ios::binary);
            mask.write(reinterpret_cast<const char*>(s.mask.data()), s.mask.size());
            if (!mask) return false;
        }
        if (!records) return false;
    }

    if (std::getenv("YOLO_POST_PROFILE") != nullptr) {
        std::cout << "POST profile:"
                  << " total=" << postprocess_total_ms
                  << " output_to_host=" << postprocess_profile.device_to_host_ms
                  << " split=" << postprocess_profile.split_ms
                  << " split_d2s=" << postprocess_profile.split_d2s_ms
                  << " split_kernel=" << postprocess_profile.split_kernel_ms
                  << " candidate_decode=" << postprocess_profile.candidate_decode_ms
                  << " candidate_decode_wall=" << postprocess_profile.candidate_decode_wall_ms
                  << " candidate_return_wall=" << postprocess_profile.split_d2h_ms
                  << " candidate_return_device=" << postprocess_profile.candidate_return_device_ms
                  << " candidate_runtime_overhead=" << postprocess_profile.candidate_runtime_overhead_ms
                  << " candidate_sort=" << postprocess_profile.candidate_sort_ms
                  << " nms=" << postprocess_profile.nms_ms
                  << " candidates=" << postprocess_profile.candidate_count
                  << " segments=" << segments.size()
                  << " mask_total=" << postprocess_profile.mask_decode_ms
                  << " mask_setup=" << postprocess_profile.mask_setup_copy_ms
                  << " prototype_d2s=" << postprocess_profile.mask_prototype_d2s_ms
                  << " selected_metadata_h2s=" << postprocess_profile.mask_metadata_h2s_ms
                  << " coeff_gather=" << postprocess_profile.mask_coeff_gather_ms
                  << " proto_combine=" << postprocess_profile.mask_proto_combine_ms
                  << " roi_resize=" << postprocess_profile.mask_source_resize_crop_ms
                  << " fused_kernel_wall=" << postprocess_profile.mask_fused_kernel_ms
                  << " mask_device_pipeline=" << postprocess_profile.mask_device_pipeline_ms
                  << " mask_runtime_overhead=" << postprocess_profile.mask_runtime_overhead_ms
                  << " mask_s2h=" << postprocess_profile.mask_source_copy_ms
                  << " mask_cpu=" << postprocess_profile.mask_build_cpu_ms
                  << " full_proto_pixels=" << postprocess_profile.mask_full_proto_pixels
                  << " sparse_proto_pixels=" << postprocess_profile.mask_sparse_proto_pixels
                  << " sparse_proto_coverage=" << postprocess_profile.mask_sparse_proto_coverage
                  << " output_roi_pixels=" << postprocess_profile.mask_output_roi_pixels
                  << std::endl;
    }

    // End-to-end latency contains every mutually exclusive printed stage exactly once.
    double total_ms = local_times.input_h2d_ms +
                      local_times.preprocess_ms +
                      local_times.inference_ms +
                      local_times.postprocess_ms +
                      local_times.output_d2h_ms;
    local_times.end_to_end_ms = total_ms > 0.0 ? total_ms : 0.0;

    if (times != nullptr) {
        *times = local_times;
    }
    return true;
}
