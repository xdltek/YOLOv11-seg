// Reuse one engine across different images to catch stale input/workspace data.
#include "yolo_pipeline.h"
#include "labels.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    YoloV11SegPipeline pipeline;
    YoloV11SegPipelineOptions options;
    options.model_path = argv[1];
    if (!pipeline.init(options, coco80_class_labels())) return 1;
    for (int i = 2; i < argc; ++i) {
        auto image = cv::imread(argv[i]);
        std::vector<Segmentation> segments;
        if (image.empty() || !pipeline.runRgb(image, segments)) return 1;
        std::cout << "[SEQUENCE] frame=" << i-2 << " instances=" << segments.size() << '\n';
    }
    return 0;
}
