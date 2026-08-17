/**
 * @file main.cpp
 * @brief Standalone YOLOv11-SEG postprocess validation and timing demo.
 */
#include "argparse/argparse.hpp"
#include "rpp_yolov11_seg_postprocessor.h"

#include <rpp_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::high_resolution_clock;
constexpr int kClassCount = 80;
constexpr int kMaskCoeffCount = 32;
constexpr int kAnchors = 8400;
constexpr int kDetectionChannels = 4 + kClassCount + kMaskCoeffCount;
constexpr int kPrototypeHeight = 160;
constexpr int kPrototypeWidth = 160;

bool check_rt(rtError_t status, const char* what)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << what << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

double elapsed_ms(Clock::time_point start, Clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double bytes_to_kib(size_t bytes)
{
    return static_cast<double>(bytes) / 1024.0;
}

bool read_binary_f32(const std::string& path, std::vector<float>& values)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open binary file: " << path << std::endl;
        return false;
    }
    input.seekg(0, std::ios::end);
    std::streamoff bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(float))) != 0) {
        std::cerr << "Invalid f32 binary size: " << path << std::endl;
        return false;
    }
    values.resize(static_cast<size_t>(bytes) / sizeof(float));
    return static_cast<bool>(input.read(reinterpret_cast<char*>(values.data()), bytes));
}

void set_detection_value(std::vector<float>& detection, int channel, int anchor, float value)
{
    detection[static_cast<size_t>(channel) * static_cast<size_t>(kAnchors) +
              static_cast<size_t>(anchor)] = value;
}

void generate_synthetic_outputs(std::vector<float>& detection, std::vector<float>& prototype)
{
    std::fill(detection.begin(), detection.end(), 0.0f);
    std::fill(prototype.begin(), prototype.end(), -2.0f);

    const int anchor0 = 0;
    set_detection_value(detection, 0, anchor0, 320.0f);
    set_detection_value(detection, 1, anchor0, 320.0f);
    set_detection_value(detection, 2, anchor0, 220.0f);
    set_detection_value(detection, 3, anchor0, 220.0f);
    set_detection_value(detection, 4, anchor0, 0.95f);
    set_detection_value(detection, 4 + kClassCount, anchor0, 4.0f);

    const int anchor1 = 100;
    set_detection_value(detection, 0, anchor1, 455.0f);
    set_detection_value(detection, 1, anchor1, 320.0f);
    set_detection_value(detection, 2, anchor1, 160.0f);
    set_detection_value(detection, 3, anchor1, 180.0f);
    set_detection_value(detection, 5, anchor1, 0.82f);
    set_detection_value(detection, 4 + kClassCount + 1, anchor1, 4.0f);

    for (int y = 0; y < kPrototypeHeight; ++y) {
        for (int x = 0; x < kPrototypeWidth; ++x) {
            float dx = (static_cast<float>(x) - 80.0f) / 38.0f;
            float dy = (static_cast<float>(y) - 80.0f) / 38.0f;
            float circle = (dx * dx + dy * dy) < 1.0f ? 2.0f : -2.0f;
            prototype[static_cast<size_t>(y) * kPrototypeWidth + x] = circle;

            float rx = std::fabs(static_cast<float>(x) - 114.0f);
            float ry = std::fabs(static_cast<float>(y) - 80.0f);
            float rect = (rx < 24.0f && ry < 32.0f) ? 2.0f : -2.0f;
            prototype[(static_cast<size_t>(1) * kPrototypeHeight + y) * kPrototypeWidth + x] = rect;
        }
    }
}

YoloV11SegConfig make_config(float score_threshold,
                             float nms_threshold,
                             float mask_threshold,
                             int max_segments)
{
    YoloV11SegConfig config;
    config.input_width = 640;
    config.input_height = 640;
    config.score_threshold = score_threshold;
    config.nms_threshold = nms_threshold;
    config.mask_threshold = mask_threshold;
    config.class_count = kClassCount;
    config.mask_coeff_count = kMaskCoeffCount;
    config.max_segments = max_segments;
    config.detection_dimensions.nbDims = 3;
    config.detection_dimensions.d[0] = 1;
    config.detection_dimensions.d[1] = kDetectionChannels;
    config.detection_dimensions.d[2] = kAnchors;
    config.detection_elements = static_cast<size_t>(kDetectionChannels) * kAnchors;
    config.prototype_dimensions.nbDims = 4;
    config.prototype_dimensions.d[0] = 1;
    config.prototype_dimensions.d[1] = kMaskCoeffCount;
    config.prototype_dimensions.d[2] = kPrototypeHeight;
    config.prototype_dimensions.d[3] = kPrototypeWidth;
    config.prototype_elements = static_cast<size_t>(kMaskCoeffCount) * kPrototypeHeight * kPrototypeWidth;
    return config;
}

LetterboxInfo make_letterbox(int source_width, int source_height)
{
    LetterboxInfo letterbox;
    letterbox.source_width = source_width;
    letterbox.source_height = source_height;
    letterbox.model_width = 640;
    letterbox.model_height = 640;
    float scale = std::min(640.0f / static_cast<float>(source_width),
                           640.0f / static_cast<float>(source_height));
    letterbox.scale = scale;
    letterbox.resized_width = static_cast<int>(std::round(static_cast<float>(source_width) * scale));
    letterbox.resized_height = static_cast<int>(std::round(static_cast<float>(source_height) * scale));
    letterbox.pad_x = static_cast<float>(640 - letterbox.resized_width) * 0.5f;
    letterbox.pad_y = static_cast<float>(640 - letterbox.resized_height) * 0.5f;
    return letterbox;
}

void print_profile(const YoloV11SegPostprocessProfile& profile,
                   double outer_ms,
                   int segment_count)
{
    std::cout << std::fixed << std::setprecision(3)
              << "Postprocess timing:\n"
              << "  split_ms: " << profile.split_ms << " ms\n"
              << "  output_d2h_ms: " << profile.device_to_host_ms << " ms\n"
              << "  output_d2h_bytes: " << profile.device_to_host_bytes
              << " bytes (" << bytes_to_kib(profile.device_to_host_bytes) << " KiB)\n"
              << "  decode_nms_ms: " << profile.decode_nms_ms << " ms\n"
              << "  mask_decode_ms: " << profile.mask_decode_ms << " ms\n"
              << "  profile_total_ms: " << profile.total_ms << " ms\n"
              << "  outer_call_ms: " << outer_ms << " ms\n"
              << "  output_segments: " << segment_count << std::endl;
}
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("yolov11-seg postprocess test demo", "0.1", argparse::default_arguments::help);
    program.add_argument("--detection-bin")
            .default_value(std::string(""))
            .help("optional raw f32 detection output file, shape [1,116,8400].");
    program.add_argument("--prototype-bin")
            .default_value(std::string(""))
            .help("optional raw f32 prototype output file, shape [1,32,160,160].");
    program.add_argument("--loop")
            .default_value(10)
            .help("measured postprocess loop count after warmup.")
            .scan<'i', int>();
    program.add_argument("--warmup")
            .default_value(1)
            .help("warmup loop count.")
            .scan<'i', int>();
    program.add_argument("--source-width")
            .default_value(640)
            .scan<'i', int>();
    program.add_argument("--source-height")
            .default_value(640)
            .scan<'i', int>();
    program.add_argument("--score-threshold")
            .default_value(0.25f)
            .scan<'g', float>();
    program.add_argument("--nms-threshold")
            .default_value(0.45f)
            .scan<'g', float>();
    program.add_argument("--mask-threshold")
            .default_value(0.50f)
            .scan<'g', float>();
    program.add_argument("--max-segments")
            .default_value(100)
            .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program << std::endl;
        return EXIT_FAILURE;
    }

    int loop = program.get<int>("--loop");
    int warmup = program.get<int>("--warmup");
    int source_width = program.get<int>("--source-width");
    int source_height = program.get<int>("--source-height");
    if (loop <= 0 || warmup < 0 || source_width <= 0 || source_height <= 0) {
        std::cerr << "Invalid loop/warmup/source dimensions." << std::endl;
        return EXIT_FAILURE;
    }

    YoloV11SegConfig config = make_config(program.get<float>("--score-threshold"),
                                          program.get<float>("--nms-threshold"),
                                          program.get<float>("--mask-threshold"),
                                          program.get<int>("--max-segments"));
    std::vector<float> detection(config.detection_elements, 0.0f);
    std::vector<float> prototype(config.prototype_elements, 0.0f);

    std::string detection_bin = program.get<std::string>("--detection-bin");
    std::string prototype_bin = program.get<std::string>("--prototype-bin");
    if (!detection_bin.empty() || !prototype_bin.empty()) {
        if (detection_bin.empty() || prototype_bin.empty()) {
            std::cerr << "--detection-bin and --prototype-bin must be provided together." << std::endl;
            return EXIT_FAILURE;
        }
        if (!read_binary_f32(detection_bin, detection) || !read_binary_f32(prototype_bin, prototype)) {
            return EXIT_FAILURE;
        }
        if (detection.size() != config.detection_elements || prototype.size() != config.prototype_elements) {
            std::cerr << "Input binary element count does not match expected YOLOv11-SEG shapes." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else {
        generate_synthetic_outputs(detection, prototype);
    }

    void* detection_device = nullptr;
    void* prototype_device = nullptr;
    const size_t detection_bytes = detection.size() * sizeof(float);
    const size_t prototype_bytes = prototype.size() * sizeof(float);
    if (!check_rt(rtMalloc(&detection_device, detection_bytes), "rtMalloc detection") ||
        !check_rt(rtMalloc(&prototype_device, prototype_bytes), "rtMalloc prototype") ||
        !check_rt(rtMemcpy(detection_device, detection.data(), detection_bytes, rtMemcpyHostToDevice),
                  "rtMemcpy detection H2D") ||
        !check_rt(rtMemcpy(prototype_device, prototype.data(), prototype_bytes, rtMemcpyHostToDevice),
                  "rtMemcpy prototype H2D")) {
        if (detection_device != nullptr) {
            rtFree(detection_device);
        }
        if (prototype_device != nullptr) {
            rtFree(prototype_device);
        }
        return EXIT_FAILURE;
    }

    YoloV11SegPostprocessor postprocessor;
    if (!postprocessor.init(config)) {
        rtFree(detection_device);
        rtFree(prototype_device);
        return EXIT_FAILURE;
    }

    LetterboxInfo letterbox = make_letterbox(source_width, source_height);
    std::vector<Segmentation> segments;
    YoloV11SegPostprocessProfile profile;
    for (int i = 0; i < warmup; ++i) {
        if (!postprocessor.run(detection_device, prototype_device, letterbox, segments, nullptr, &profile)) {
            rtFree(detection_device);
            rtFree(prototype_device);
            return EXIT_FAILURE;
        }
    }

    YoloV11SegPostprocessProfile total_profile;
    double total_outer_ms = 0.0;
    for (int i = 0; i < loop; ++i) {
        auto start = Clock::now();
        if (!postprocessor.run(detection_device, prototype_device, letterbox, segments, nullptr, &profile)) {
            rtFree(detection_device);
            rtFree(prototype_device);
            return EXIT_FAILURE;
        }
        auto stop = Clock::now();
        total_outer_ms += elapsed_ms(start, stop);
        total_profile.device_to_host_ms += profile.device_to_host_ms;
        total_profile.device_to_host_bytes += profile.device_to_host_bytes;
        total_profile.split_ms += profile.split_ms;
        total_profile.decode_nms_ms += profile.decode_nms_ms;
        total_profile.mask_decode_ms += profile.mask_decode_ms;
        total_profile.total_ms += profile.total_ms;
    }

    const double inv_loop = 1.0 / static_cast<double>(loop);
    YoloV11SegPostprocessProfile average_profile;
    average_profile.device_to_host_ms = total_profile.device_to_host_ms * inv_loop;
    average_profile.device_to_host_bytes = total_profile.device_to_host_bytes / static_cast<size_t>(loop);
    average_profile.split_ms = total_profile.split_ms * inv_loop;
    average_profile.decode_nms_ms = total_profile.decode_nms_ms * inv_loop;
    average_profile.mask_decode_ms = total_profile.mask_decode_ms * inv_loop;
    average_profile.total_ms = total_profile.total_ms * inv_loop;

    std::cout << "YOLOv11-SEG postprocess standalone test\n"
              << "  detection_elements: " << config.detection_elements << "\n"
              << "  prototype_elements: " << config.prototype_elements << "\n"
              << "  source_size: " << source_width << "x" << source_height << "\n"
              << "  warmup: " << warmup << "\n"
              << "  loop: " << loop << "\n";
    print_profile(average_profile, total_outer_ms * inv_loop, static_cast<int>(segments.size()));

    rtFree(detection_device);
    rtFree(prototype_device);
    return EXIT_SUCCESS;
}
