#include "rpp_segmentation_network.h"
#include "rpp_model_support.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void replaceConsumers(infer1::INetworkDefinition& network, infer1::ITensor& old,
                      infer1::ITensor& replacement)
{
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto* layer = network.getLayer(i);
        for (int input = 0; input < layer->getNbInputs(); ++input)
            if (layer->getInput(input) == &old) layer->setInput(input, replacement);
    }
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        if (network.getOutput(i) != &old) continue;
        const std::string name = old.getName();
        network.unmarkOutput(old);
        old.setName((name + "_unused").c_str());
        replacement.setName(name.c_str());
        network.markOutput(replacement);
        break;
    }
}

infer1::Dims dims(std::initializer_list<int> values)
{
    infer1::Dims result{};
    result.nbDims = values.size();
    int i = 0;
    for (int value : values) result.d[i++] = value;
    return result;
}

// Expose the unnormalized DFL logits and retain class/mask planes in the
// public prediction. A device FP32 decoder fills its four coordinate planes.
// Pruning the original DFL subgraph also avoids a large channel reduction
// that this SDK cannot safely split for 960-square heads.
std::string deviceDfl(infer1::INetworkDefinition& network, int anchors)
{
    using namespace infer1;
    ITensor *logits = nullptr, *prediction = nullptr, *classes = nullptr, *coefficients = nullptr;
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto* layer = network.getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            auto* t = layer->getOutput(j);
            const auto s = t->getDimensions();
            const std::string name = t->getName() ? t->getName() : "";
            if (name.find("/dfl/Reshape_output_0") != std::string::npos &&
                s.nbDims == 3 && s.d[0] == 4 && s.d[1] == 16 && s.d[2] == anchors) logits = t;
        }
    }
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        auto* t = network.getOutput(i);
        const auto s = t->getDimensions();
        if (s.nbDims == 2 && s.d[0] == 116 && s.d[1] == anchors) prediction = t;
    }
    if (!logits || !prediction) throw std::runtime_error("Missing static COCO80 DFL head contract");
    // Read the public head's concatenation rather than depend on a
    // particular exported layer number or accidentally select an earlier head.
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto* final = network.getLayer(i);
        if (final->getNbOutputs() != 1 || final->getOutput(0) != prediction ||
            final->getType() != LayerType::kCONCATENATION) continue;
        for (int j = 0; j < final->getNbInputs(); ++j) {
            auto* part = final->getInput(j);
            const auto s = part->getDimensions();
            if (s.nbDims == 2 && s.d[0] == 32 && s.d[1] == anchors) coefficients = part;
            if (s.nbDims == 2 && s.d[0] == 80 && s.d[1] == anchors) classes = part;
            if (s.nbDims != 2 || s.d[0] != 84 || s.d[1] != anchors) continue;
            for (int k = 0; k < network.getNbLayers(); ++k) {
                auto* det = network.getLayer(k);
                if (det->getNbOutputs() != 1 || det->getOutput(0) != part ||
                    det->getType() != LayerType::kCONCATENATION) continue;
                for (int n = 0; n < det->getNbInputs(); ++n) {
                    auto* candidate = det->getInput(n);
                    const auto cs = candidate->getDimensions();
                    if (cs.nbDims == 2 && cs.d[0] == 80 && cs.d[1] == anchors) classes = candidate;
                }
            }
        }
    }
    if (!classes || !coefficients) throw std::runtime_error("Missing segmentation class/mask head contract");
    auto* first_bin = network.addSlice(*logits, dims({0, 0, 0}), dims({4, 1, anchors}), dims({1, 1, 1}));
    if (!first_bin) throw std::runtime_error("Cannot create device DFL coordinate storage");
    auto* planes = network.addShuffle(*first_bin->getOutput(0));
    if (!planes) throw std::runtime_error("Cannot reshape device DFL coordinate storage");
    planes->setReshapeDimensions(dims({4, anchors}));
    ITensor* parts[] = {planes->getOutput(0), classes, coefficients};
    auto* concat = network.addConcatenation(parts, 3);
    if (!concat) throw std::runtime_error("Cannot retain device segmentation prediction");
    concat->setAxis(0);
    const std::string name = prediction->getName();
    network.unmarkOutput(*prediction);
    prediction->setName((name + "_unused").c_str());
    concat->getOutput(0)->setName(name.c_str());
    network.markOutput(*concat->getOutput(0));
    network.markOutput(*logits);
    std::cout << "[ENGINE  ] Stable FP32 DFL normalization and box decode on RPP" << std::endl;
    return logits->getName();
}

void streamAttention(infer1::INetworkDefinition& network, std::vector<std::vector<float>>& weights)
{
    using namespace infer1;
    const int original_count = network.getNbLayers();
    auto producer = [&](ITensor* tensor) -> ILayer* {
        for (int i=0;i<original_count;++i) {
            auto* layer=network.getLayer(i);
            for(int j=0;j<layer->getNbOutputs();++j) if(layer->getOutput(j)==tensor) return layer;
        }
        return nullptr;
    };
    auto reshape = [&](ITensor* input, Dims shape) -> ITensor* {
        auto* layer=network.addShuffle(*input);
        if(!layer) throw std::runtime_error("Cannot reshape attention tensor");
        layer->setReshapeDimensions(shape);return layer->getOutput(0);
    };
    auto slice = [&](ITensor* input, Dims start, Dims size) -> ITensor* {
        Dims stride{};stride.nbDims=start.nbDims;
        for(int i=0;i<stride.nbDims;++i) stride.d[i]=1;
        auto* layer=network.addSlice(*input,start,size,stride);
        if(!layer) throw std::runtime_error("Cannot slice attention tensor");
        return layer->getOutput(0);
    };
    auto concat = [&](std::vector<ITensor*> inputs, int axis) -> ITensor* {
        auto* layer=network.addConcatenation(inputs.data(),inputs.size());
        if(!layer) throw std::runtime_error("Cannot join attention tiles");
        layer->setAxis(axis);return layer->getOutput(0);
    };
    auto pad = [&](ITensor* input, int rows, int columns, int extra_rows, int extra_columns) -> ITensor* {
        if(extra_rows) {
            weights.emplace_back(extra_rows*columns,0.0f);
            auto* zero=network.addConstant(dims({extra_rows,columns}),Weights{DataType::kFLOAT,weights.back().data(),int64_t(weights.back().size())});
            if(!zero) throw std::runtime_error("Cannot pad attention queries");
            input=concat({input,zero->getOutput(0)},0);rows+=extra_rows;
        }
        if(extra_columns) {
            weights.emplace_back(rows*extra_columns,0.0f);
            auto* zero=network.addConstant(dims({rows,extra_columns}),Weights{DataType::kFLOAT,weights.back().data(),int64_t(weights.back().size())});
            if(!zero) throw std::runtime_error("Cannot pad attention keys");
            input=concat({input,zero->getOutput(0)},1);
        }
        return input;
    };
    for(int i=0;i<original_count;++i) {
        auto* soft=network.getLayer(i);
        if(soft->getType()!=LayerType::kSOFTMAX || soft->getNbOutputs()!=1) continue;
        const std::string name=soft->getOutput(0)->getName()?soft->getOutput(0)->getName():"";
        if(name.find("/attn/Softmax_output_0")==std::string::npos) continue;
        auto* qk=producer(soft->getInput(0));
        ILayer *transpose=nullptr,*context=nullptr;
        for(int j=0;j<original_count;++j) {
            auto* candidate=network.getLayer(j);
            if(candidate->getNbInputs()==1 && candidate->getInput(0)==soft->getOutput(0)) transpose=candidate;
        }
        if(transpose) for(int j=0;j<original_count;++j) {
            auto* candidate=network.getLayer(j);
            if(candidate->getNbInputs()==2 && candidate->getInput(1)==transpose->getOutput(0)) context=candidate;
        }
        if(!qk || qk->getNbInputs()!=2 || !context) throw std::runtime_error("Unexpected attention graph contract");
        auto *q=qk->getInput(0),*k=qk->getInput(1),*v=context->getInput(0);
        const auto qs=q->getDimensions(),ks=k->getDimensions(),vs=v->getDimensions();
        // The tiled form is qualified for four-head 960 exports. Preserve the
        // native parser path for other head counts, including native n960.
        if(qs.nbDims==3 && qs.d[0]!=4) continue;
        if(qs.nbDims!=3 || ks.nbDims!=3 || vs.nbDims!=3 || qs.d[1]!=900 || qs.d[2]!=32 ||
            ks.d[0]!=qs.d[0] || ks.d[1]!=32 || ks.d[2]!=900 || vs.d[0]!=qs.d[0] || vs.d[1]!=64 || vs.d[2]!=900)
            throw std::runtime_error("Unsupported 960-square attention dimensions");
        constexpr int length=900,padded=928,queries=960,tile=96;
        weights.emplace_back(padded,0.0f);
        for(int j=length;j<padded;++j) weights.back()[j]=-10000.0f;
        const float* key_mask=weights.back().data();
        std::vector<ITensor*> heads;
        for(int h=0;h<qs.d[0];++h) {
            auto* qh=reshape(slice(q,dims({h,0,0}),dims({1,length,32})),dims({length,32}));
            auto* kh=reshape(slice(k,dims({h,0,0}),dims({1,32,length})),dims({32,length}));
            auto* vh=reshape(slice(v,dims({h,0,0}),dims({1,64,length})),dims({64,length}));
            qh=pad(qh,length,32,queries-length,0);
            kh=pad(kh,32,length,0,padded-length);
            vh=pad(vh,64,length,0,padded-length);
            std::vector<ITensor*> chunks;
            for(int start=0;start<queries;start+=tile) {
                // The current scheduler expects a constant node to have a
                // single consumer. Reuse its immutable payload, not its node.
                auto* mask=network.addConstant(dims({1,padded}),Weights{DataType::kFLOAT,key_mask,padded});
                if(!mask) throw std::runtime_error("Cannot create attention key mask");
                auto* query=slice(qh,dims({start,0}),dims({tile,32}));
                auto* scores=network.addMatrixMultiply(*query,MatrixOperation::kNONE,*kh,MatrixOperation::kNONE);
                if(!scores) throw std::runtime_error("Cannot project attention query tile");
                auto* masked=network.addElementWise(*scores->getOutput(0),*mask->getOutput(0),ElementWiseOperation::kSUM);
                if(!masked) throw std::runtime_error("Cannot mask padded attention keys");
                auto* probabilities=network.addSoftMax(*masked->getOutput(0));
                if(!probabilities) throw std::runtime_error("Cannot normalize attention tile");
                probabilities->setAxes(1U<<1);
                auto* pt=network.addShuffle(*probabilities->getOutput(0));
                if(!pt) throw std::runtime_error("Cannot transpose attention tile");
                Permutation order{};order.order[0]=1;order.order[1]=0;pt->setFirstTranspose(order);
                auto* result=network.addMatrixMultiply(*vh,MatrixOperation::kNONE,*pt->getOutput(0),MatrixOperation::kNONE);
                if(!result) throw std::runtime_error("Cannot reconstruct attention tile");
                chunks.push_back(result->getOutput(0));
            }
            auto* joined=concat(chunks,1);
            heads.push_back(reshape(slice(joined,dims({0,0}),dims({64,length})),dims({1,64,length})));
        }
        auto* result=concat(heads,0);
        replaceConsumers(network,*context->getOutput(0),*result);
        std::cout << "[ENGINE  ] Streamed attention on RPP: " << qs.d[0] << " heads, " << tile << "-query tiles" << std::endl;
    }
}

} // namespace

void validateSupportedSegmentationModel(infer1::INetworkDefinition& network)
{
    using namespace infer1;
    const char* contract = "Supported profiles: YOLO11n/s/m/l/x-seg 640x640 and YOLO11n/s-seg 960x960; "
                           "static FP32 RGB NCHW input, batch 1.";
    if (network.getNbInputs() != 1) throw std::runtime_error(contract);
    auto* input = network.getInput(0);
    const auto shape = input->getDimensions();
    const bool chw = shape.nbDims == 3 && shape.d[0] == 3;
    const bool nchw = shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] == 3;
    if ((!chw && !nchw) || input->getType() != DataType::kFLOAT)
        throw std::runtime_error(contract);
    int stem_channels = 0;
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto* layer = network.getLayer(i);
        if (layer->getType() == LayerType::kCONVOLUTION && layer->getNbInputs() &&
            layer->getInput(0) == input) {
            stem_channels = static_cast<IConvolutionLayer*>(layer)->getNbOutputMaps();
            break;
        }
    }
    if (!yolo11::supportsModelProfile(shape.d[shape.nbDims-1], shape.d[shape.nbDims-2], stem_channels))
        throw std::runtime_error(contract);
}

std::string adaptSegmentationNetwork(infer1::INetworkDefinition& network,
                                     std::vector<std::vector<float>>& weights)
{
    using namespace infer1;
    const auto input_shape = network.getInput(0)->getDimensions();
    const bool device_dfl = input_shape.nbDims >= 2 &&
        input_shape.d[input_shape.nbDims - 2] == 960 && input_shape.d[input_shape.nbDims - 1] == 960;
    if (device_dfl) streamAttention(network, weights);
    const int count = network.getNbLayers();
    for (int i = 0; i < count; ++i) {
        auto* layer = network.getLayer(i);
        // Some parser importers do not set the layer name. ONNX tensor names
        // still identify DFL normalization and the prototype upsample.
        std::string name = layer->getName() ? layer->getName() : "";
        if (layer->getNbOutputs() && layer->getOutput(0)->getName())
            name += layer->getOutput(0)->getName();
        if (!device_dfl && layer->getType() == LayerType::kSOFTMAX && name.find("dfl") != std::string::npos) {
            const auto shape = layer->getInput(0)->getDimensions();
            int axis = -1;
            for (int d = 0; d < shape.nbDims; ++d) if (shape.d[d] == 16) { axis = d; break; }
            if (axis < 0) throw std::runtime_error("Unexpected DFL channel dimension");
            auto* exp = network.addUnary(*layer->getInput(0), UnaryOperation::kEXP);
            if (!exp) throw std::runtime_error("Cannot create DFL exponential");
            auto* sum = network.addReduce(*exp->getOutput(0), ReduceOperation::kSUM, 1u << axis, true);
            if (!sum) throw std::runtime_error("Cannot create DFL channel sum");
            auto* div = network.addElementWise(*exp->getOutput(0), *sum->getOutput(0), ElementWiseOperation::kDIV);
            if (!div) throw std::runtime_error("Cannot create DFL normalization");
            exp->setName((name + "_rpp_exp").c_str());
            sum->setName((name + "_rpp_sum").c_str());
            div->setName((name + "_rpp_div").c_str());
            replaceConsumers(network, *layer->getOutput(0), *div->getOutput(0));
            std::cout << "[ENGINE  ] Adapted DFL channel normalization in memory" << std::endl;
        }
        if (layer->getType() != LayerType::kDECONVOLUTION || name.find("proto") == std::string::npos)
            continue;
        auto* transpose = static_cast<IDeconvolutionLayer*>(layer);
        const auto shape = layer->getInput(0)->getDimensions();
        const auto kernel = transpose->getKernelSize();
        const auto stride = transpose->getStride();
        const auto padding = transpose->getPadding();
        const auto old_weights = transpose->getKernelWeights();
        const auto old_bias = transpose->getBiasWeights();
        const int channels = transpose->getNbOutputMaps();
        if (shape.nbDims != 3 || kernel.h() != 2 || kernel.w() != 2 || stride.h() != 2 ||
            stride.w() != 2 || padding.h() || padding.w() || transpose->getNbGroups() != 1 ||
            old_weights.type != DataType::kFLOAT || old_bias.type != DataType::kFLOAT ||
            old_weights.count != int64_t(shape.d[0]) * channels * 4 || old_bias.count != channels)
            throw std::runtime_error("Unsupported prototype upsample contract");
        weights.emplace_back(old_weights.count);
        auto& packed = weights.back();
        const auto* source = static_cast<const float*>(old_weights.values);
        for (int out = 0; out < channels; ++out)
            for (int phase = 0; phase < 4; ++phase)
                for (int in = 0; in < shape.d[0]; ++in)
                    packed[(out * 4 + phase) * shape.d[0] + in] = source[(in * channels + out) * 4 + phase];
        const auto* bias = static_cast<const float*>(old_bias.values);
        const float* packed_data = packed.data();
        weights.emplace_back(channels * 4);
        for (int out = 0; out < channels; ++out)
            for (int phase = 0; phase < 4; ++phase) weights.back()[out * 4 + phase] = bias[out];
        auto* conv = network.addConvolutionNd(*layer->getInput(0), channels * 4, DimsHW{1, 1},
            Weights{DataType::kFLOAT, packed_data, old_weights.count},
            Weights{DataType::kFLOAT, weights.back().data(), channels * 4});
        if (!conv) throw std::runtime_error("Cannot create prototype projection");
        conv->setName((name + "_rpp_projection").c_str());
        auto* phase = network.addShuffle(*conv->getOutput(0));
        if (!phase) throw std::runtime_error("Cannot create prototype phase shuffle");
        phase->setReshapeDimensions(dims({channels, 2, 2, shape.d[1], shape.d[2]}));
        Permutation order{};
        order.order[0] = 0; order.order[1] = 3; order.order[2] = 1;
        order.order[3] = 4; order.order[4] = 2;
        phase->setSecondTranspose(order);
        phase->setName((name + "_rpp_phase").c_str());
        auto* output = network.addShuffle(*phase->getOutput(0));
        if (!output) throw std::runtime_error("Cannot create prototype reshape");
        output->setReshapeDimensions(dims({channels, shape.d[1] * 2, shape.d[2] * 2}));
        output->setName((name + "_rpp_output").c_str());
        replaceConsumers(network, *layer->getOutput(0), *output->getOutput(0));
        std::cout << "[ENGINE  ] Adapted prototype upsample in memory" << std::endl;
    }
    // Keep the four DFL distances on device for FP32 anchor/stride arithmetic.
    // Public prediction/prototype outputs retain their original contract.
    // The 960-square path decodes logits directly, avoiding the SDK's large
    // DFL transpose/reduction. The existing 640-square distance path is kept.
    if (device_dfl) return deviceDfl(network, 18900);
    if (input_shape.nbDims < 2) return {};
    const int height = input_shape.d[input_shape.nbDims - 2];
    const int width = input_shape.d[input_shape.nbDims - 1];
    if (!((width == 640 && height == 640) || (width == 960 && height == 960))) return {};
    const int anchors = width * height / 64 + width * height / 256 + width * height / 1024;
    for (int i = 0; i < network.getNbLayers(); ++i) {
        auto* layer = network.getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            auto* tensor = layer->getOutput(j);
            const std::string name = tensor->getName() ? tensor->getName() : "";
            const auto shape = tensor->getDimensions();
            if (name.find("/dfl/") != std::string::npos && shape.nbDims == 2 && shape.d[0] == 4 && shape.d[1] == anchors) {
                network.markOutput(*tensor);
                return name;
            }
        }
    }
    return {};
}
