// Diagnostic only: replay an exact normalized input through OpenRT without PRE
// or POST. Output tensors may include explicitly marked ONNX intermediates.
#include "logger.h"
#include "parser_api.h"
#include "rpp_buffer_manager.h"
#include "rpp_segmentation_network.h"
#include "rpp_precision_module.h"

#include <Infer.h>
#include <RppConfig.h>
#include <rpp_runtime.h>

#include <filesystem>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 6) {
        std::cerr << "Usage: inference_probe model.onnx input.f32 output_dir [--application|--application-unpatched|--precision-only [tensor_name]]\n";
        return 2;
    }
    try {
        if (std::filesystem::exists(argv[3]))
            throw std::runtime_error("Output directory already exists; use a new capture path");
        config::RppConfig runtime_config;
        std::unique_ptr<infer1::IBuilder> builder{infer1::createInferBuilder(sample::gLogger.getLogger())};
        if (!builder) throw std::runtime_error("Cannot create builder");
        builder->setConfigPtr(&runtime_config);
        std::unique_ptr<infer1::INetworkDefinition> network{builder->createNetwork()};
        std::unique_ptr<infer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!network || !config) throw std::runtime_error("Cannot create network/config");
        std::unique_ptr<onnxparser::IParser> parser{onnxparser::createParser(*network,sample::gLogger.getLogger())};
        if (!parser || onnx_parser(argv[1],builder.get(),network.get(),parser.get()) != 0)
            throw std::runtime_error("Cannot parse model");
        const bool application = argc >= 5 && (std::string(argv[4]) == "--application" ||
            std::string(argv[4]) == "--application-unpatched");
        // Isolate convolution arithmetic tests from the segmentation contract.
        const bool precision_only = argc >= 5 && std::string(argv[4]) == "--precision-only";
        const bool precision_enabled = precision_only || (application && std::string(argv[4]) == "--application");
        if (argc >= 5 && !application && !precision_only) throw std::runtime_error("Unknown probe mode");
        std::vector<std::vector<float>> adapted_weights;
        if (application) adaptSegmentationNetwork(*network, adapted_weights);
        if (argc == 6) {
            const bool append = argv[5][0] == '+';
            const std::string selection = argv[5] + (append ? 1 : 0);
            std::vector<std::string> names;
            if (!selection.empty() && selection[0] == '@') {
                std::ifstream list(selection.substr(1));
                if (!list) throw std::runtime_error("Cannot read probe tensor list");
                for (std::string line; std::getline(list,line);) if (!line.empty()) names.push_back(line);
            } else names.push_back(selection);
            std::vector<infer1::ITensor*> selected;
            for (const auto& name:names) {
                infer1::ITensor* found=nullptr;
                for (int i = 0; i < network->getNbLayers(); ++i)
                    for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j) {
                        auto* tensor = network->getLayer(i)->getOutput(j);
                        if (tensor->getName() && std::string(tensor->getName()) == name) found=tensor;
                    }
                if (!found) throw std::runtime_error("Probe tensor not found: "+name);
                selected.push_back(found);
            }
            if(selected.empty()) throw std::runtime_error("Empty probe selection");
            if (!append) while (network->getNbOutputs()) network->unmarkOutput(*network->getOutput(0));
            for(auto* tensor:selected) network->markOutput(*tensor);
        }
        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(size_t(1)<<30);
        config->setFlag(infer1::BuilderFlag::kBF16);
        std::cout << "[PROBE] Build OpenRT engine" << std::endl;
        std::shared_ptr<infer1::IEngine> engine{builder->buildEngineWithConfig(*network,*config)};
        if (!engine) throw std::runtime_error("Engine build failed");
        samplesCommon::RppBufferManager buffers(engine);
        std::unique_ptr<infer1::IExecutionContext> context{engine->createExecutionContext()};
        if (!context) throw std::runtime_error("Context creation failed");
        int inputs=0;
        for (int i=0;i<engine->getNbBindings();++i) {
            if (engine->getBindingDataType(i) != infer1::DataType::kFLOAT)
                throw std::runtime_error("Probe requires FP32 bindings");
            if (!engine->bindingIsInput(i)) continue;
            ++inputs;
            size_t count=samplesCommon::volume(engine->getBindingDimensions(i));
            std::vector<float> data(count);
            std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
            if (!input || input.tellg()!=std::streampos(count*sizeof(float)))
                throw std::runtime_error("Input tensor byte count mismatch");
            input.seekg(0); input.read(reinterpret_cast<char*>(data.data()),count*sizeof(float));
            for (float value:data) if (!std::isfinite(value))
                throw std::runtime_error("Non-finite probe input");
            if (!input || rtMemcpy(buffers.getDeviceBuffer(engine->getBindingName(i)),data.data(),
                                  count*sizeof(float),rtMemcpyHostToDevice)!=rtSuccess)
                throw std::runtime_error("Input H2D failed");
        }
        if (inputs!=1) throw std::runtime_error("Probe requires exactly one input");
        std::filesystem::create_directories(argv[3]);
        const auto metadata_path=std::filesystem::path(argv[3])/"bindings.json";
        std::ofstream metadata(metadata_path.string()+".tmp");
        metadata << "[\n";bool first=true, finite=true;
        for (int frame=0;frame<2;++frame) {
            std::cout << "[PROBE] Execute frame " << frame << std::endl;
            yolo11::ScopedConvolutionPrecision precision(precision_enabled);
            if (!context->execute(1,buffers.getDeviceBindings().data()))
                throw std::runtime_error("Execution failed");
            for (int i=0;i<engine->getNbBindings();++i) {
                if (engine->bindingIsInput(i)) continue;
                auto dims=engine->getBindingDimensions(i);
                size_t count=samplesCommon::volume(dims);
                std::vector<float> data(count);
                if (rtMemcpy(data.data(),buffers.getDeviceBuffer(engine->getBindingName(i)),
                             count*sizeof(float),rtMemcpyDeviceToHost)!=rtSuccess)
                    throw std::runtime_error("Output D2H failed");
                std::string filename="frame"+std::to_string(frame)+"_binding"+std::to_string(i)+".f32";
                std::ofstream output(std::filesystem::path(argv[3])/filename,std::ios::binary);
                output.write(reinterpret_cast<const char*>(data.data()),count*sizeof(float));
                if (!output) throw std::runtime_error("Cannot write tensor");
                for (float value:data) if (!std::isfinite(value)) finite=false;
                if (!first) metadata << ",\n";first=false;
                metadata << "{\"frame\":" << frame << ",\"name\":" << std::quoted(engine->getBindingName(i))
                         << ",\"file\":" << std::quoted(filename) << ",\"shape\":[";
                for(int d=0;d<dims.nbDims;++d) {if(d)metadata<<',';metadata<<dims.d[d];}
                metadata << "]}";
            }
        }
        metadata << "\n]\n";
        metadata.close();
        if (!metadata) throw std::runtime_error("Cannot write binding metadata");
        std::filesystem::rename(metadata_path.string()+".tmp",metadata_path);
        if (!finite) throw std::runtime_error("Captured non-finite inference outputs");
        std::cout << "[PROBE] Completed two inference-only frames" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[PROBE] " << error.what() << std::endl;
        return 1;
    }
}
