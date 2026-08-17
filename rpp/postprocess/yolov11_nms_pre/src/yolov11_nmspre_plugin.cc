/**
 * @file yolov11_nmspre_plugin.cc
 * @brief OpenRT plugin implementation for YOLOv11 PreNMS tensor rearrangement.
 */
#include "yolov11_nmspre_plugin.h"
#include "yolov11_nmspre_kernel.cuh"

#include <cassert>

const char *YoloV11NMSPREPlugin::getPluginType() const {
    return "YoloV11NMSPREPlugin";
}

const char *YoloV11NMSPREPlugin::getPluginVersion() const {
    return "1";
}

int YoloV11NMSPREPlugin::getNbOutputs() const {
    return 2;
}

infer1::Dims YoloV11NMSPREPlugin::getOutputDimensions(int index, const infer1::Dims *inputs, int nbInputDims) {
    infer1::Dims dims{};
    if (nbInputDims != 1 || index < 0)
    {
        dims.nbDims = 0;
        dims.d[0] = 0; 
    }
    else
    {
        int box_num = inputs[0].d[0];
        int class_num = inputs[0].d[1] - 4;
        if(is_tanspose_)
        {
            box_num = inputs[0].d[1];
            class_num = inputs[0].d[0] - 4;
        }
        switch (index)
        {
        case 0:
            dims.nbDims = 2;
            dims.d[0] = box_num;
            dims.d[1] = 4;
            break;
        case 1:
            dims.nbDims = 2;
            dims.d[0] = box_num;
            dims.d[1] = class_num;
            break;
        default:
            break;
        }
    }

    return dims;
}

bool YoloV11NMSPREPlugin::supportsFormat(infer1::DataType type, infer1::PluginFormat format) const {
    return ((type == infer1::DataType::kFLOAT || type == infer1::DataType::kINT8 || type == infer1::DataType::kBF) && format == infer1::PluginFormat::kLINEAR);
}

void YoloV11NMSPREPlugin::configureWithFormat(const infer1::Dims *inputDims, int nbInputs, const infer1::Dims *outputDims,
                                        int nbOutputs, infer1::DataType type, infer1::PluginFormat format, int maxBatchSize) {
    const infer1::Dims& input_dims = inputDims[0];
    height_ = input_dims.d[0];
    width_ = input_dims.d[1];
    if (input_dims.d[0] * input_dims.d[1] > kSramSize)
    {
        assert(false);
    }
}

int YoloV11NMSPREPlugin::initialize() {
    return 0;
}

void YoloV11NMSPREPlugin::terminate() {

}

size_t YoloV11NMSPREPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int YoloV11NMSPREPlugin::enqueue(int32_t batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                           rtStream_t stream) {
    YoloV11RearrangeMatrix(stream, (const float*)inputs[0], width_, height_, is_tanspose_, type_size_, obj_threshold_, nms_threshold_, max_class_ = 200, outputs[0], outputs[1]);
    return 0;
}

void YoloV11NMSPREPlugin::setParams(float obj_threshold, float nms_threshold, int max_class, bool transpose, int type_size) { 
    obj_threshold_ = obj_threshold; 
    nms_threshold_ = nms_threshold; 
    max_class_ = max_class; 
    is_tanspose_ = transpose;
    type_size_ = type_size;
}

size_t YoloV11NMSPREPlugin::getSerializationSize() const {
    return 0;
}

void YoloV11NMSPREPlugin::serialize(void *buffer) const {

}

void YoloV11NMSPREPlugin::destroy() {
}

infer1::IPluginV2 *YoloV11NMSPREPlugin::clone() const {
    auto* plugin = new YoloV11NMSPREPlugin();
    plugin->setHeight(height_);
    plugin->setWidth(width_);
    plugin->setParams(obj_threshold_, nms_threshold_, max_class_, is_tanspose_, type_size_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

void YoloV11NMSPREPlugin::setPluginNamespace(const char *pluginNamespace) {
    namespace_ = pluginNamespace;
}

const char *YoloV11NMSPREPlugin::getPluginNamespace() const {
    return namespace_.c_str();
}

const char *YoloV11NMSPREPluginCreator::getPluginName() const {
    return plugin_name_.c_str();
}

const char *YoloV11NMSPREPluginCreator::getPluginVersion() const {
    return "1";
}

const infer1::PluginFieldCollection *YoloV11NMSPREPluginCreator::getFieldNames() {
    return &field_collection_;
}

infer1::IPluginV2 *YoloV11NMSPREPluginCreator::createPlugin(const char *name, const infer1::PluginFieldCollection *fc) {
    auto* plugin = new YoloV11NMSPREPlugin();
    field_collection_ = *fc;
    plugin_name_ = name;
    return plugin;
}

infer1::IPluginV2 *
YoloV11NMSPREPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    auto plugin = new YoloV11NMSPREPlugin();
    plugin_name_ = name;
    return plugin;
}

void YoloV11NMSPREPluginCreator::setPluginNamespace(const char *pluginNamespace) {
    namespace_ = pluginNamespace;
}

const char *YoloV11NMSPREPluginCreator::getPluginNamespace() const {
    return namespace_.c_str();
}
