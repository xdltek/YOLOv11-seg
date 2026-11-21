cmake_minimum_required(VERSION 3.16.3)

project(yolov11_seg_demo_workspace VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-write-strings -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_COMPILER /usr/local/bin/clang++)
set(CMAKE_CUDA_ARCHITECTURES RPP)
set(_CMAKE_CUDA_WHOLE_FLAG "-c")
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "-fPIC -gdwarf-4 -fno-unroll-loops -O3 -ffp-contract=off -nocudalib -nocudainc -save-temps -std=c++14")

option(YOLO_ENABLE_RPP_PERF "Enable rpp_perf trace integration" OFF)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(RppRT_DIR /usr/local/rpp)
set(RppRT_LIBS ${RppRT_DIR}/lib)
set(RppRT_INCLUDE_DIRS "${RppRT_DIR}/include")
set(ThirdPartyLib_Include ${ROOT_DIRECTORY}/3rd_party)
set(YOLO_BUILD_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
set(YOLO_BUILD_LIB_DIR ${CMAKE_BINARY_DIR}/lib)
set(YOLO_BUILD_ASSETS_DIR ${CMAKE_BINARY_DIR}/assets)
set(YOLO_BUILD_OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(YOLO_BUILD_TRACE_DIR ${CMAKE_BINARY_DIR}/trace)
set(YOLO_BUILD_LOGS_DIR ${CMAKE_BINARY_DIR}/logs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${YOLO_BUILD_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${YOLO_BUILD_LIB_DIR})

message(STATUS "ROOT_DIRECTORY: ${ROOT_DIRECTORY}")

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
)

link_directories(${RppRT_LIBS})
set(rpprt_libs RppRT nvonnxparser caffeparser urpp)

set(YOLOV11_INCLUDE_DIR ${ROOT_DIRECTORY}/src/yolov11_seg)
set(YOLO_UTILS_INCLUDE_DIR ${ROOT_DIRECTORY}/src/utils)
set(YOLO_PERF_INCLUDE_DIR ${ROOT_DIRECTORY}/src/perf)
set(RPP_INFERENCE_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/inference)
set(RPP_PREPROCESS_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/preprocess)
set(RPP_POSTPROCESS_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/postprocess)

# Build the application's convolution correction with the installed SDK ISA.
# Neither the SDK nor supplied ONNX models are modified.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(yolov11_precision_assembler EXCLUDE_FROM_ALL
    ${RPP_INFERENCE_INCLUDE_DIR}/precision_assembler.cpp
)
target_link_libraries(yolov11_precision_assembler PRIVATE rpp_asm)
set_target_properties(yolov11_precision_assembler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools
    BUILD_RPATH ${RppRT_LIBS}
)
file(GLOB YOLO_CONV_ASM_DEPENDENCIES CONFIGURE_DEPENDS
    ${RppRT_INCLUDE_DIRS}/kernel/asm/mac8/*.s
)
set(YOLO_PRECISION_HEADER ${CMAKE_BINARY_DIR}/generated/rpp_precision_kernels.inc)
add_custom_command(OUTPUT ${YOLO_PRECISION_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${RPP_INFERENCE_INCLUDE_DIR}/generate_precision_kernels.py
        --sdk ${RppRT_INCLUDE_DIRS}/kernel/asm
        --assembler $<TARGET_FILE:yolov11_precision_assembler>
        --output ${YOLO_PRECISION_HEADER}
    DEPENDS yolov11_precision_assembler
        ${RPP_INFERENCE_INCLUDE_DIR}/generate_precision_kernels.py
        ${RppRT_INCLUDE_DIRS}/kernel/asm/kernel_cfg.json
        ${YOLO_CONV_ASM_DEPENDENCIES}
    VERBATIM
)
add_library(yolov11_seg_utils SHARED
    ${ROOT_DIRECTORY}/src/utils/logger.cpp
    ${ROOT_DIRECTORY}/src/utils/logging.cpp
)
target_include_directories(yolov11_seg_utils PUBLIC
    ${YOLO_UTILS_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
)
set_target_properties(yolov11_seg_utils PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(perf_trace SHARED
    ${ROOT_DIRECTORY}/src/perf/perf_trace_session.cpp
)
target_include_directories(perf_trace PUBLIC
    ${YOLO_PERF_INCLUDE_DIR}
)
target_include_directories(perf_trace PRIVATE
    ${RppRT_INCLUDE_DIRS}
)
target_link_libraries(perf_trace PRIVATE pthread)
if(YOLO_ENABLE_RPP_PERF)
    target_compile_definitions(perf_trace PUBLIC YOLO_ENABLE_RPP_PERF=1)
    target_compile_options(perf_trace PRIVATE -Wno-c99-designator)
    target_link_libraries(perf_trace PRIVATE rpp_perf urpp)
endif()
set_target_properties(perf_trace PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(rpp_yolo_preprocess SHARED
    ${ROOT_DIRECTORY}/rpp/preprocess/rpp_yolo_preprocessor.cpp
    ${ROOT_DIRECTORY}/rpp/preprocess/rpp_preprocess_kernels.cu
)
target_include_directories(rpp_yolo_preprocess PUBLIC
    ${RPP_PREPROCESS_INCLUDE_DIR}
    ${YOLOV11_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(rpp_yolo_preprocess PRIVATE
    ${YOLO_PERF_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
)
target_link_libraries(rpp_yolo_preprocess PUBLIC perf_trace yolov11_seg_utils ${rpprt_libs})
set_target_properties(rpp_yolo_preprocess PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(rpp_inference SHARED
    ${ROOT_DIRECTORY}/rpp/inference/rpp_infer_engine.cpp
    ${RPP_INFERENCE_INCLUDE_DIR}/rpp_precision_module.cpp
    ${RPP_INFERENCE_INCLUDE_DIR}/rpp_segmentation_network.cpp
    ${RPP_INFERENCE_INCLUDE_DIR}/rpp_box_decode.cu
    ${YOLO_PRECISION_HEADER}
)
target_include_directories(rpp_inference PUBLIC
    ${RPP_INFERENCE_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
)
target_include_directories(rpp_inference PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(rpp_inference PUBLIC yolov11_seg_utils ${rpprt_libs} PRIVATE dl)
set_target_properties(rpp_inference PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_executable(yolov11_precision_module_test EXCLUDE_FROM_ALL
    ${ROOT_DIRECTORY}/tests/test_precision_module.cpp
)
target_include_directories(yolov11_precision_module_test PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(yolov11_precision_module_test PRIVATE rpp_inference)
set_target_properties(yolov11_precision_module_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

add_executable(yolov11_box_decode_test EXCLUDE_FROM_ALL ${ROOT_DIRECTORY}/tests/test_box_decode.cpp)
target_link_libraries(yolov11_box_decode_test PRIVATE rpp_inference)
set_target_properties(yolov11_box_decode_test PROPERTIES BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}")

add_executable(yolov11_model_support_test EXCLUDE_FROM_ALL ${ROOT_DIRECTORY}/tests/test_model_support.cpp)
target_include_directories(yolov11_model_support_test PRIVATE ${RPP_INFERENCE_INCLUDE_DIR})

add_library(rpp_yolov11_seg_postprocess SHARED
    ${ROOT_DIRECTORY}/rpp/postprocess/rpp_yolov11_seg_postprocessor.cpp
    ${ROOT_DIRECTORY}/rpp/postprocess/rpp_postprocess_kernels.cu
)
target_include_directories(rpp_yolov11_seg_postprocess PUBLIC
    ${RPP_POSTPROCESS_INCLUDE_DIR}
    ${YOLOV11_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(rpp_yolov11_seg_postprocess PRIVATE
    ${YOLO_PERF_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
)
target_link_libraries(rpp_yolov11_seg_postprocess PUBLIC perf_trace yolov11_seg_utils ${rpprt_libs} ${OpenCV_LIBS})
target_compile_options(rpp_yolov11_seg_postprocess PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(rpp_yolov11_seg_postprocess PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(yolov11_seg_core SHARED
    ${ROOT_DIRECTORY}/src/yolov11_seg/labels.cpp
    ${ROOT_DIRECTORY}/src/yolov11_seg/yolo_pipeline.cpp
)
target_include_directories(yolov11_seg_core PUBLIC
    ${YOLOV11_INCLUDE_DIR}
    ${RPP_INFERENCE_INCLUDE_DIR}
    ${RPP_POSTPROCESS_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolov11_seg_core PUBLIC
    perf_trace
    rpp_inference
    rpp_yolo_preprocess
    rpp_yolov11_seg_postprocess
    yolov11_seg_utils
    ${OpenCV_LIBS}
    ${rpprt_libs}
    pthread
)
set_target_properties(yolov11_seg_core PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(yolov11_seg_visualization SHARED
    ${ROOT_DIRECTORY}/src/yolov11_seg/visualization.cpp
)
target_include_directories(yolov11_seg_visualization PUBLIC
    ${YOLOV11_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolov11_seg_visualization PUBLIC yolov11_seg_core ${OpenCV_LIBS})
set_target_properties(yolov11_seg_visualization PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_executable(yolov11_seg_rgb_image_demo
    ${ROOT_DIRECTORY}/demos/rgb_image/main.cpp
)
target_include_directories(yolov11_seg_rgb_image_demo PRIVATE
    ${ThirdPartyLib_Include}
)
target_link_libraries(yolov11_seg_rgb_image_demo PRIVATE yolov11_seg_core yolov11_seg_visualization)
set_target_properties(yolov11_seg_rgb_image_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

add_executable(yolov11_seg_yuv_frame_demo
    ${ROOT_DIRECTORY}/demos/yuv_frame/main.cpp
)
target_include_directories(yolov11_seg_yuv_frame_demo PRIVATE
    ${ThirdPartyLib_Include}
)
target_link_libraries(yolov11_seg_yuv_frame_demo PRIVATE yolov11_seg_core yolov11_seg_visualization)
set_target_properties(yolov11_seg_yuv_frame_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

add_executable(yolov11_sequence_probe EXCLUDE_FROM_ALL ${ROOT_DIRECTORY}/tests/sequence_probe.cpp)
target_link_libraries(yolov11_sequence_probe PRIVATE yolov11_seg_core ${OpenCV_LIBS})
set_target_properties(yolov11_sequence_probe PROPERTIES BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}")

add_executable(yolov11_seg_postprocess_test
    ${ROOT_DIRECTORY}/tests/postprocess_test.cpp
)
target_include_directories(yolov11_seg_postprocess_test PRIVATE
    ${ThirdPartyLib_Include}
    ${RPP_POSTPROCESS_INCLUDE_DIR}
    ${YOLOV11_INCLUDE_DIR}
    ${YOLO_PERF_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
)
target_link_libraries(yolov11_seg_postprocess_test PRIVATE rpp_yolov11_seg_postprocess)
set_target_properties(yolov11_seg_postprocess_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

add_executable(yolov11_seg_inference_probe EXCLUDE_FROM_ALL tests/inference_probe.cpp)
target_include_directories(yolov11_seg_inference_probe PRIVATE ${YOLO_UTILS_INCLUDE_DIR} ${RppRT_INCLUDE_DIRS} ${RPP_INFERENCE_INCLUDE_DIR})
target_link_libraries(yolov11_seg_inference_probe PRIVATE rpp_inference yolov11_seg_utils ${rpprt_libs} pthread)
set_target_properties(yolov11_seg_inference_probe PROPERTIES BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}")

add_executable(yolov11_nms_compaction_test EXCLUDE_FROM_ALL tests/test_nms_compaction.cpp)
target_include_directories(yolov11_nms_compaction_test PRIVATE ${RPP_POSTPROCESS_INCLUDE_DIR} ${RppRT_INCLUDE_DIRS})
target_link_libraries(yolov11_nms_compaction_test PRIVATE rpp_yolov11_seg_postprocess ${rpprt_libs})
set_target_properties(yolov11_nms_compaction_test PROPERTIES BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}")

file(MAKE_DIRECTORY ${YOLO_BUILD_ASSETS_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_OUTPUT_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_TRACE_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_LOGS_DIR})
configure_file("${ROOT_DIRECTORY}/assets/test.jpg" "${YOLO_BUILD_ASSETS_DIR}/test.jpg" COPYONLY)
configure_file("${ROOT_DIRECTORY}/assets/test_0.yuv" "${YOLO_BUILD_ASSETS_DIR}/test_0.yuv" COPYONLY)
