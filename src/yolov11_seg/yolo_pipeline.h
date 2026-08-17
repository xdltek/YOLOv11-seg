/**
 * @file yolo_pipeline.h
 * @brief Reusable YOLOv11-SEG pipeline built from preprocessing, inference, and postprocessing modules.
 */
#ifndef XDLTEK_SAMPLES_YOLOV11_PIPELINE_H
#define XDLTEK_SAMPLES_YOLOV11_PIPELINE_H

#include "segmentation_types.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class RppInferEngine;
class RppYoloPreprocessor;
class YoloV11SegPostprocessor;

/**
 * @brief User-visible options used to initialize a YOLOv11-SEG pipeline instance.
 */
struct YoloV11SegPipelineOptions
{
    std::string model_path;
    int inference_count = 1;
    YoloV11SegConfig config;
};

/**
 * @brief Stage-level timing summary reported by demos after one measured pipeline run.
 */
struct YoloV11SegStageTimes
{
    double input_h2d_ms = 0.0;
    size_t input_h2d_bytes = 0;
    double output_d2h_ms = 0.0;
    size_t output_d2h_bytes = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double end_to_end_ms = 0.0;
};

/**
 * @brief Reusable YOLOv11-SEG pipeline that wires preprocessing, inference, and postprocessing.
 */
class YoloV11SegPipeline
{
public:
    /**
     * @brief Construct an empty pipeline; call init() before runRgb() or runI420().
     */
    YoloV11SegPipeline();
    /**
     * @brief Destroy owned preprocessing, inference, and postprocessing modules.
     */
    ~YoloV11SegPipeline();

    /**
     * @brief Disable copying because the pipeline owns runtime resources.
     */
    YoloV11SegPipeline(const YoloV11SegPipeline&) = delete;
    /**
     * @brief Disable copy assignment because the pipeline owns runtime resources.
     */
    YoloV11SegPipeline& operator=(const YoloV11SegPipeline&) = delete;

    /**
     * @brief Initialize the model runtime and YOLOv11-SEG postprocess path.
     * @param options Model path, inference loop count, and configurable thresholds.
     * @param class_names Class name table used to derive class count and visualization text.
     */
    bool init(const YoloV11SegPipelineOptions& options, const std::vector<std::string>& class_names);

    /**
     * @brief Run one BGR/RGB host image through preprocessing, inference, and postprocessing.
     * @param image Host OpenCV image. Non-contiguous images are cloned before preprocessing.
     * @param segments Output segments restored to source-image coordinates.
     * @param times Optional per-stage timing output.
     */
    bool runRgb(const cv::Mat& image,
                std::vector<Segmentation>& segments,
                YoloV11SegStageTimes* times = nullptr);

    /**
     * @brief Run one I420 host frame through preprocessing, inference, and postprocessing.
     * @param yuv_data Host buffer packed as Y plane, U plane, then V plane.
     * @param yuv_bytes Number of bytes available in yuv_data.
     * @param width Source frame width; must be even for I420.
     * @param height Source frame height; must be even for I420.
     * @param segments Output segments restored to source-frame coordinates.
     * @param times Optional per-stage timing output.
     */
    bool runI420(const void* yuv_data,
                 size_t yuv_bytes,
                 int width,
                 int height,
                 std::vector<Segmentation>& segments,
                 YoloV11SegStageTimes* times = nullptr);

    /**
     * @brief Return the validated runtime configuration used by this pipeline.
     */
    const YoloV11SegConfig& runtimeConfig() const { return config_; }

private:
    /**
     * @brief Run one generic input descriptor through preprocessing, inference, and postprocessing.
     */
    bool runInput(const void* data,
                  size_t bytes,
                  int width,
                  int height,
                  int row_stride_bytes,
                  int channels,
                  bool continuous,
                  int format,
                  std::vector<Segmentation>& segments,
                  YoloV11SegStageTimes* times);

    /**
     * @brief Create or reuse preprocessing workspace for a source shape.
     */
    bool ensurePreprocessor(int width, int height);
    /**
     * @brief Initialize YOLOv11-SEG postprocessing after model output dimensions are known.
     */
    bool initPostprocessor(const std::vector<std::string>& class_names);

    std::unique_ptr<RppInferEngine> infer_engine_;
    std::unique_ptr<RppYoloPreprocessor> preprocessor_;
    std::unique_ptr<YoloV11SegPostprocessor> postprocessor_;

    YoloV11SegPipelineOptions options_;
    YoloV11SegConfig config_;
    int preprocessor_width_ = 0;
    int preprocessor_height_ = 0;
    int detection_output_index_ = -1;
    int prototype_output_index_ = -1;
    bool initialized_ = false;
};

#endif // XDLTEK_SAMPLES_YOLOV11_PIPELINE_H
