#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

build_dir="${ROOT_DIR}/build"
model_path="${ROOT_DIR}/weights/yolo11s-seg.onnx"

show_help() {
    cat <<'EOF'
Usage: scripts/test.sh [options]

Options:
  -b, --build-dir DIR   Build directory. Default: build
  -m, --model PATH      ONNX model path. Default: weights/yolo11s-seg.onnx
  -h, --help            Show this help message.
EOF
}

resolve_repo_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${ROOT_DIR}/${path}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build-dir)
            build_dir="$(resolve_repo_path "$2")"
            shift 2
            ;;
        -m|--model)
            model_path="$(resolve_repo_path "$2")"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

required_files=(
    "${model_path}"
    "${build_dir}/bin/yolov11_seg_rgb_image_demo"
    "${build_dir}/bin/yolov11_seg_yuv_frame_demo"
    "${build_dir}/assets/test.jpg"
    "${build_dir}/assets/test_0.yuv"
)

for path in "${required_files[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required test artifact: ${path}" >&2
        exit 1
    fi
done

echo "YOLOv11-SEG project smoke prerequisites are present."
