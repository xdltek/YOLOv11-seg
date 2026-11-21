/**
 * @file rpp_infer_engine.cpp
 * @brief Reusable OpenRT inference engine wrapper implementation.
 */
#include "rpp_infer_engine.h"
#include "rpp_precision_module.h"
#include "rpp_segmentation_network.h"
#include "rpp_box_decode.cuh"

#include "logger.h"
#include "parser_api.h"
#include "sampleCommon.h"

#include <exception>
#include <iostream>

namespace
{
/**
 * @brief Read model input tensor dimensions and infer the spatial width and height.
 * @param dims Runtime binding dimensions.
 * @param width Output spatial width.
 * @param height Output spatial height.
 */
bool parseInputSpatialDims(const infer1::Dims& dims, int& width, int& height)
{
    if (dims.nbDims < 2) {
        return false;
    }

    if (dims.nbDims >= 3 && dims.d[dims.nbDims - 1] == 3) {
        height = dims.d[dims.nbDims - 3];
        width = dims.d[dims.nbDims - 2];
    }
    else {
        height = dims.d[dims.nbDims - 2];
        width = dims.d[dims.nbDims - 1];
    }

    return width > 0 && height > 0;
}

}

/**
 * @brief Build a fresh OpenRT engine from ONNX for this application process.
 */
bool RppInferEngine::init()
{
    context_ptr_.reset();
    releaseBoxBuffers();
    buffer_ptr_.reset();
    engine_ptr_.reset();
    output_infos_.clear();
    std::cout << "[ENGINE  ] Building OpenRT engine from ONNX: "
              << onnx_model_path_ << std::endl;
    if (!buildFromOnnx() || !initializeBindings()) return false;
    std::cout << "[ENGINE  ] OpenRT engine ready" << std::endl;
    return true;
}

bool RppInferEngine::buildFromOnnx()
{
    // Create the builder, network, and config objects that turn ONNX into an executable engine.
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(sample::gLogger.getLogger())};
    if (builder == nullptr) {
        std::cerr << "Unable to create builder object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::INetworkDefinition> network {builder->createNetwork()};
    if (network == nullptr) {
        std::cerr << "Unable to create network object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::IBuilderConfig> config {builder->createBuilderConfig()};
    if (config == nullptr) {
        std::cerr << "Unable to create config object." << std::endl;
        return false;
    }

    // Parse the configured ONNX file into the runtime network definition.
    std::unique_ptr<onnxparser::IParser> parser {onnxparser::createParser(*network, sample::gLogger.getLogger())};
    if (parser == nullptr) {
        std::cerr << "Unable to parse ONNX model file: " << onnx_model_path_ << std::endl;
        return false;
    }
    try {
        if (onnx_parser(onnx_model_path_, builder.get(), network.get(), parser.get()) != 0) {
            return false;
        }
    }
    catch (const std::exception& error) {
        std::cerr << "OpenRT could not import the ONNX model: " << error.what() << std::endl;
        return false;
    }

    std::vector<std::vector<float>> adapted_weights;
    try {
        validateSupportedSegmentationModel(*network);
        distances_name_ = adaptSegmentationNetwork(*network, adapted_weights);
    } catch (const std::exception& error) {
        std::cerr << "OpenRT segmentation adaptation failed: " << error.what() << std::endl;
        return false;
    }

    // Build a single-batch BF16-capable engine for the demo inference path.
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((size_t)1_GiB);
    config->setFlag(BuilderFlag::kBF16);

    try {
        engine_ptr_ = std::shared_ptr<infer1::IEngine>(builder->buildEngineWithConfig(*network, *config));
    } catch (const std::exception& error) {
        std::cerr << "OpenRT could not build the ONNX engine: " << error.what() << std::endl;
        return false;
    }
    if (!engine_ptr_) {
        return false;
    }

    if (config->getFlag(infer1::BuilderFlag::kINT8)) {
        sample::user_visible_log("Since the model includes quantization layers, inference is performed with int8 precision.");
    }

    return true;
}

bool RppInferEngine::initializeBindings()
{
    // Cache binding names, shapes, data types, and element counts for pipeline stages.
    output_infos_.clear();
    for (int i = 0; i < engine_ptr_->getNbBindings(); i++) {
        if (engine_ptr_->bindingIsInput(i)) {
            input_name_ = engine_ptr_->getBindingName(i);
            infer1::Dims input_dimensions = engine_ptr_->getBindingDimensions(i);
            input_data_type_ = engine_ptr_->getBindingDataType(i);

            if (!parseInputSpatialDims(input_dimensions, input_width_, input_height_)) {
                std::cerr << "Invalid model input dimensions." << std::endl;
                return false;
            }
        }
        else {
            if (distances_name_ == engine_ptr_->getBindingName(i)) continue;
            RppInferOutputInfo output;
            output.name = engine_ptr_->getBindingName(i);
            output.dimensions = engine_ptr_->getBindingDimensions(i);
            output.data_type = engine_ptr_->getBindingDataType(i);
            output.elements = samplesCommon::volume(output.dimensions);
            output_infos_.push_back(output);
        }
    }
    if (output_infos_.empty()) {
        std::cerr << "The model does not expose any output binding." << std::endl;
        return false;
    }
    output_name_ = output_infos_[0].name;
    output_dimensions_ = output_infos_[0].dimensions;
    output_data_type_ = output_infos_[0].data_type;
    output_tensor_size_ = output_infos_[0].elements;

    // Allocate host and device buffers for every engine binding.
    buffer_ptr_ = std::make_shared<samplesCommon::RppBufferManager>(engine_ptr_);
    if (buffer_ptr_ == nullptr) {
        return false;
    }

    // Create the execution context used by execute().
    context_ptr_.reset(engine_ptr_->createExecutionContext());
    if (!context_ptr_) return false;
    if (!distances_name_.empty()) {
        box_anchors_ = (input_width_ / 8) * (input_height_ / 8) +
                       (input_width_ / 16) * (input_height_ / 16) +
                       (input_width_ / 32) * (input_height_ / 32);
        const size_t head_bytes = buffer_ptr_->size(distances_name_);
        device_dfl_ = head_bytes == size_t(64) * box_anchors_ * sizeof(float);
        if (!device_dfl_ && head_bytes != size_t(4) * box_anchors_ * sizeof(float))
            return false;
        prediction_name_.clear();
        for (const auto& output : output_infos_)
            if (output.elements == size_t(116) * box_anchors_) prediction_name_ = output.name;
        if (prediction_name_.empty()) return false;
        std::cout << "[ENGINE  ] FP32 RPP box coordinates enabled" << std::endl;
    }
    return true;
}

RppInferEngine::~RppInferEngine()
{
    context_ptr_.reset();
    releaseBoxBuffers();
}

void RppInferEngine::releaseBoxBuffers()
{
    if (distances_sram_) rtFreeSram(distances_sram_);
    if (boxes_sram_) rtFreeSram(boxes_sram_);
    distances_sram_ = boxes_sram_ = nullptr;
}

bool RppInferEngine::decodeBoxes()
{
    if (distances_name_.empty()) return true;
    const size_t bytes = size_t(4) * box_anchors_ * sizeof(float);
    const size_t input_bytes = device_dfl_ ? bytes * 16 : bytes;
    // OpenRT and the custom PRE/POST stages share SRAM. Allocate only after
    // synchronous inference and release before the next stage/frame; a custom
    // persistent allocation may otherwise overlap the SDK's execution workspace.
    if (rtMallocSram(&distances_sram_, input_bytes) != rtSuccess ||
        rtMallocSram(&boxes_sram_, bytes) != rtSuccess) {
        releaseBoxBuffers();
        return false;
    }
    if (rtMemcpy(distances_sram_, buffer_ptr_->getDeviceBuffer(distances_name_), input_bytes,
                 rtMemcpyDeviceToSram) != rtSuccess) {
        releaseBoxBuffers();
        return false;
    }
    if (device_dfl_) launch_yolo11_dfl_decode(nullptr, static_cast<const float*>(distances_sram_),
                            static_cast<float*>(boxes_sram_), input_width_, input_height_, box_anchors_);
    else launch_yolo11_box_decode(nullptr, static_cast<const float*>(distances_sram_),
                            static_cast<float*>(boxes_sram_), input_width_, input_height_,
                            box_anchors_);
    const bool passed = rtStreamSynchronize(nullptr) == rtSuccess &&
                        rtMemcpy(buffer_ptr_->getDeviceBuffer(prediction_name_), boxes_sram_, bytes,
                                 rtMemcpySramToDevice) == rtSuccess;
    releaseBoxBuffers();
    return passed;
}

/**
 * @brief Return the device buffer associated with the model input binding.
 */
void* RppInferEngine::getInputDeviceBuffer() const
{
    return buffer_ptr_ == nullptr ? nullptr : buffer_ptr_->getDeviceBuffer(input_name_);
}

/**
 * @brief Return the device buffer associated with the model output binding.
 */
void* RppInferEngine::getOutputDeviceBuffer() const
{
    return buffer_ptr_ == nullptr ? nullptr : buffer_ptr_->getDeviceBuffer(output_name_);
}

/**
 * @brief Return the device buffer associated with a model output binding.
 */
void* RppInferEngine::getOutputDeviceBuffer(size_t output_index) const
{
    if (buffer_ptr_ == nullptr || output_index >= output_infos_.size()) {
        return nullptr;
    }
    return buffer_ptr_->getDeviceBuffer(output_infos_[output_index].name);
}

void* RppInferEngine::getBoxHeadDeviceBuffer() const
{
    return buffer_ptr_ && !distances_name_.empty() ? buffer_ptr_->getDeviceBuffer(distances_name_) : nullptr;
}

size_t RppInferEngine::getBoxHeadElements() const
{
    return buffer_ptr_ && !distances_name_.empty() ? buffer_ptr_->size(distances_name_) / sizeof(float) : 0;
}

/**
 * @brief Execute one synchronous model inference using current device bindings.
 */
bool RppInferEngine::executeContext()
{
    // Synchronous execution is the inference timing region reported by the demos.
    if (context_ptr_ == nullptr || buffer_ptr_ == nullptr) {
        sample::LOG_ERROR() << "RPP inference engine has not been initialized." << std::endl;
        return false;
    }

    bool ok;
    {
        yolo11::ScopedConvolutionPrecision precision(
            (input_width_ == 640 && input_height_ == 640) ||
            (input_width_ == 960 && input_height_ == 960));
        ok = context_ptr_->execute(1, buffer_ptr_->getDeviceBindings().data());
    }
    if (!ok) {
        sample::LOG_ERROR() << "do inference failed." << std::endl;
        return false;
    }
    return decodeBoxes();
}

/**
 * @brief Public inference entry point used by the YOLO pipeline.
 */
bool RppInferEngine::execute()
{
    return executeContext();
}
