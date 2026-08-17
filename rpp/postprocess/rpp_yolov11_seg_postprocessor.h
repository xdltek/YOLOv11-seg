/**
 * @file rpp_yolov11_seg_postprocessor.h
 * @brief YOLOv11-SEG output copy, decode, NMS, and mask reconstruction module.
 */
#ifndef XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H
#define XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H

#include "segmentation_types.h"
#include "rpp_postprocess_kernels.cuh"

#include <rpp_runtime.h>

#include <cstddef>
#include <vector>

/**
 * @brief Timing and transfer counters reported by YOLOv11-SEG postprocessing.
 */
struct YoloV11SegPostprocessProfile
{
    double device_to_host_ms = 0.0;
    size_t device_to_host_bytes = 0;
    double split_ms = 0.0;
    double split_d2s_ms = 0.0;
    double split_kernel_ms = 0.0;
    double split_s2d_ms = 0.0;
    double split_d2h_ms = 0.0;
    double decode_nms_ms = 0.0;
    double mask_decode_ms = 0.0;
    double mask_setup_copy_ms = 0.0;
    double mask_coeff_gather_ms = 0.0;
    double mask_proto_combine_ms = 0.0;
    double mask_lowres_threshold_ms = 0.0;
    double mask_box_copy_ms = 0.0;
    double mask_source_resize_crop_ms = 0.0;
    double mask_source_copy_ms = 0.0;
    double mask_build_cpu_ms = 0.0;
    double total_ms = 0.0;
};

/**
 * @brief YOLOv11-SEG CPU postprocessor.
 *
 * The first complete demo path keeps preprocessing and inference on RPP, then copies
 * detection/prototype outputs to host and performs SEG decode. This preserves correctness
 * for two-output Ultralytics segmentation ONNX models while leaving RT_NMS/device mask
 * decode as a focused optimization step.
 */
class YoloV11SegPostprocessor
{
public:
    /**
     * @brief Release host, DDR, and SRAM mask decode workspaces.
     */
    ~YoloV11SegPostprocessor();

    /**
     * @brief Validate output layouts and prepare host staging buffers.
     */
    bool init(const YoloV11SegConfig& config);

    /**
     * @brief Postprocess one model output pair into final source-image segmentation results.
     */
    bool run(const void* detection_output_device,
             const void* prototype_output_device,
             const LetterboxInfo& letterbox,
             std::vector<Segmentation>& segments,
             rtStream_t stream = nullptr,
             YoloV11SegPostprocessProfile* profile = nullptr);

private:
    struct OutputLayout
    {
        std::vector<int> dims;
        std::vector<size_t> strides;
        int feature_axis = -1;
        int channels = 0;
        int anchors = 0;
    };

    struct Candidate
    {
        int anchor = -1;
        int class_id = -1;
        float score = 0.0f;
        cv::Rect box;
    };

    bool resolveDetectionLayout();
    bool resolvePrototypeLayout();
    bool runRppSplitOutput(const void* detection_output_device, YoloV11SegPostprocessProfile* profile);
    float detectionValueAt(int channel, int anchor) const;
    float coefficientValueAt(int coeff, int anchor) const;
    float prototypeValueAt(int coeff, int y, int x) const;
    void decodeCandidates(const LetterboxInfo& letterbox, std::vector<Candidate>& candidates) const;
    void runNms(const std::vector<Candidate>& candidates, std::vector<int>& keep) const;
    bool runRppMaskDecode(const void* prototype_output_device,
                          const LetterboxInfo& letterbox,
                          const std::vector<Candidate>& candidates,
                          const std::vector<int>& keep,
                          std::vector<Segmentation>& segments,
                          YoloV11SegPostprocessProfile* profile);
    void buildMaskFromRaw(const Candidate& candidate,
                          const LetterboxInfo& letterbox,
                          const float* raw_mask,
                          Segmentation& segment) const;
    void buildMaskFromLowResBinary(const Candidate& candidate,
                                   const LetterboxInfo& letterbox,
                                   const float* lowres_mask,
                                   Segmentation& segment) const;
    void buildMaskFromSourceBinary(const float* source_mask,
                                   int source_width,
                                   int source_height,
                                   Segmentation& segment) const;
    void buildMaskFromBoxBinary(const float* box_mask,
                                int box_width,
                                int box_height,
                                Segmentation& segment) const;
    void buildMask(const Candidate& candidate, const LetterboxInfo& letterbox, Segmentation& segment) const;
    bool ensureDeviceBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    bool ensureSramBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    void releaseMaskDecodeBuffers();

    YoloV11SegConfig config_;
    OutputLayout detection_layout_;
    int prototype_channels_ = 0;
    int prototype_height_ = 0;
    int prototype_width_ = 0;
    int det_channels_ = 0;
    int total_head_channels_ = 0;
    bool split_input_channels_first_ = true;
    std::vector<float> host_det_head_;
    std::vector<float> host_coeff_head_;
    std::vector<float> host_prototype_;
    std::vector<float> host_coefficients_;
    std::vector<float> host_raw_masks_;
    std::vector<int> host_selected_anchor_indices_;
    std::vector<int> host_selected_box_;
    std::vector<float> host_lowres_masks_;
    std::vector<float> host_source_mask_;
    void* prototype_sram_ = nullptr;
    void* coefficients_device_ = nullptr;
    void* coefficients_sram_ = nullptr;
    void* raw_masks_device_ = nullptr;
    void* raw_masks_sram_ = nullptr;
    void* selected_indices_device_ = nullptr;
    void* selected_indices_sram_ = nullptr;
    void* binary_masks_device_ = nullptr;
    void* binary_masks_sram_ = nullptr;
    void* selected_box_device_ = nullptr;
    void* selected_box_sram_ = nullptr;
    void* source_mask_device_ = nullptr;
    void* source_mask_sram_ = nullptr;
    void* split_input_sram_ = nullptr;
    void* det_head_device_ = nullptr;
    void* det_head_sram_ = nullptr;
    void* coeff_head_device_ = nullptr;
    void* coeff_head_sram_ = nullptr;
    size_t prototype_sram_capacity_ = 0;
    size_t coefficients_device_capacity_ = 0;
    size_t coefficients_sram_capacity_ = 0;
    size_t raw_masks_device_capacity_ = 0;
    size_t raw_masks_sram_capacity_ = 0;
    size_t selected_indices_device_capacity_ = 0;
    size_t selected_indices_sram_capacity_ = 0;
    size_t binary_masks_device_capacity_ = 0;
    size_t binary_masks_sram_capacity_ = 0;
    size_t selected_box_device_capacity_ = 0;
    size_t selected_box_sram_capacity_ = 0;
    size_t source_mask_device_capacity_ = 0;
    size_t source_mask_sram_capacity_ = 0;
    size_t split_input_sram_capacity_ = 0;
    size_t det_head_device_capacity_ = 0;
    size_t det_head_sram_capacity_ = 0;
    size_t coeff_head_device_capacity_ = 0;
    size_t coeff_head_sram_capacity_ = 0;
    bool initialized_ = false;
};

#endif // XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H
