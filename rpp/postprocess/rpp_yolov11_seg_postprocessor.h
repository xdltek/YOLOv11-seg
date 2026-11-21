/**
 * @file rpp_yolov11_seg_postprocessor.h
 * @brief YOLOv11-SEG output copy, decode, NMS, and mask reconstruction module.
 */
#ifndef XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H
#define XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H

#include "segmentation_types.h"
#include "rpp_postprocess_kernels.cuh"

#include <InferRuntime.h>
#include <rpp_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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
    double candidate_decode_ms = 0.0;
    double candidate_decode_wall_ms = 0.0;
    double candidate_return_device_ms = 0.0;
    double candidate_runtime_overhead_ms = 0.0;
    double candidate_sort_ms = 0.0;
    double nms_ms = 0.0;
    double decode_nms_ms = 0.0;
    size_t candidate_count = 0;
    double mask_decode_ms = 0.0;
    double mask_setup_copy_ms = 0.0;
    double mask_prototype_d2s_ms = 0.0;
    double mask_metadata_h2s_ms = 0.0;
    double mask_coeff_gather_ms = 0.0;
    double mask_proto_combine_ms = 0.0;
    double mask_lowres_threshold_ms = 0.0;
    double mask_box_copy_ms = 0.0;
    double mask_source_resize_crop_ms = 0.0;
    double mask_fused_kernel_ms = 0.0;
    double mask_device_pipeline_ms = 0.0;
    double mask_runtime_overhead_ms = 0.0;
    double mask_source_copy_ms = 0.0;
    double mask_build_cpu_ms = 0.0;
    size_t mask_full_proto_pixels = 0;
    size_t mask_sparse_proto_pixels = 0;
    size_t mask_output_roi_pixels = 0;
    double mask_sparse_proto_coverage = 0.0;
    double total_ms = 0.0;
};

/**
 * @brief YOLOv11-SEG RPP postprocessor.
 *
 * Candidate preparation and RT NMS execute in an OpenRT engine on RPP. Selected
 * mask coefficients, prototype combination, resize, crop, and threshold also run
 * on RPP; the host only assembles the compact result objects.
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

    /**
     * @brief Release transient SRAM while retaining engines and host/DDR buffers.
     */
    void releaseSramBuffers();

private:
    struct OutputLayout
    {
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
    bool buildRppNmsEngine();
    bool allocateRppNmsBuffers();
    bool copyRppNmsConstants();
    bool runRppNms(const void* detection_output_device,
                   const LetterboxInfo& letterbox,
                   std::vector<Candidate>& candidates,
                   std::vector<int>& keep,
                   YoloV11SegPostprocessProfile* profile);
    bool parseRppNmsResults(const LetterboxInfo& letterbox,
                            std::vector<Candidate>& candidates,
                            std::vector<int>& keep) const;
    void resetRppNmsEngine();
    bool runRppMaskDecode(const void* prototype_output_device,
                          const LetterboxInfo& letterbox,
                          const std::vector<Candidate>& candidates,
                          const std::vector<int>& keep,
                          std::vector<Segmentation>& segments,
                          YoloV11SegPostprocessProfile* profile);
    void buildMaskFromBoxBinary(const uint8_t* box_mask,
                                int box_width,
                                int box_height,
                                Segmentation& segment) const;
    bool ensureSramBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    bool ensureHostPinnedBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    void releaseMaskDecodeBuffers();

    YoloV11SegConfig config_;
    OutputLayout detection_layout_;
    int prototype_channels_ = 0;
    int prototype_height_ = 0;
    int prototype_width_ = 0;
    int det_channels_ = 0;
    int total_head_channels_ = 0;
    std::unique_ptr<infer1::ICudaEngine> nms_engine_;
    std::unique_ptr<infer1::IExecutionContext> nms_context_;
    int nms_input_index_ = -1;
    int nms_boxes_index_ = -1;
    int nms_scores_index_ = -1;
    int nms_max_output_index_ = -1;
    int nms_iou_threshold_index_ = -1;
    int nms_score_threshold_index_ = -1;
    int nms_indices_index_ = -1;
    int nms_count_index_ = -1;
    int nms_anchor_map_elements_ = 0;
    bool nms_requires_compaction_ = false;
    int nms_indices_elements_ = 0;
    int nms_count_elements_ = 0;
    int nms_index_tuple_size_ = 3;
    void* nms_boxes_device_ = nullptr;
    void* nms_scores_device_ = nullptr;
    void* nms_constants_device_ = nullptr;
    void* nms_outputs_device_ = nullptr;
    void* nms_compact_device_ = nullptr;
    bool compactRppNmsInput(const void* input, int32_t* device_map);
    void* nms_boxes_host_ = nullptr;
    void* nms_scores_host_ = nullptr;
    void* nms_outputs_host_ = nullptr;
    size_t nms_boxes_bytes_ = 0;
    size_t nms_scores_bytes_ = 0;
    size_t nms_outputs_bytes_ = 0;
    std::vector<float> host_det_head_;
    std::vector<int> host_selected_anchor_indices_;
    std::vector<int> host_selected_box_;
    std::vector<int> host_mask_offsets_;
    std::vector<int> host_coordinate_lut_key_;
    std::vector<int> host_source_x0_lut_;
    std::vector<float> host_source_wx_lut_;
    std::vector<int> host_source_y0_lut_;
    std::vector<float> host_source_wy_lut_;
    void* prototype_sram_ = nullptr;
    void* coefficients_sram_ = nullptr;
    void* raw_masks_sram_ = nullptr;
    void* metadata_sram_ = nullptr;
    void* source_mask_sram_ = nullptr;
    void* split_input_sram_ = nullptr;
    void* host_source_mask_pinned_ = nullptr;
    void* host_metadata_pinned_ = nullptr;
    size_t prototype_sram_capacity_ = 0;
    size_t coefficients_sram_capacity_ = 0;
    size_t raw_masks_sram_capacity_ = 0;
    size_t metadata_sram_capacity_ = 0;
    size_t source_mask_sram_capacity_ = 0;
    size_t split_input_sram_capacity_ = 0;
    size_t host_source_mask_pinned_capacity_ = 0;
    size_t host_metadata_pinned_capacity_ = 0;
    rtStream_t stream_ = nullptr;
    rtEvent_t mask_event_setup_start_ = nullptr;
    rtEvent_t mask_event_start_ = nullptr;
    rtEvent_t mask_event_gather_end_ = nullptr;
    rtEvent_t mask_event_combine_end_ = nullptr;
    rtEvent_t mask_event_resize_end_ = nullptr;
    bool initialized_ = false;
};

#endif // XDLTEK_SAMPLES_RPP_YOLOV11_SEG_POSTPROCESSOR_H
