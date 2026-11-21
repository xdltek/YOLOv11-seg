#ifndef YOLO11_RPP_SEGMENTATION_NETWORK_H
#define YOLO11_RPP_SEGMENTATION_NETWORK_H
#include <Infer.h>
#include <vector>
#include <string>

// Validate the release profile before constructing an executable engine.
void validateSupportedSegmentationModel(infer1::INetworkDefinition& network);

// Keep generated FP32 weight payloads alive until buildEngineWithConfig returns.
std::string adaptSegmentationNetwork(infer1::INetworkDefinition& network,
                                     std::vector<std::vector<float>>& weights);
#endif
