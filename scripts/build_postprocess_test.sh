#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
ENABLE_PERF="OFF"

show_help() {
    cat <<'EOF'
Usage: scripts/build_postprocess_test.sh [options]

Options:
  -b, --build-dir DIR   Build directory. Default: build
      --perf            Enable YOLO_ENABLE_RPP_PERF.
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
            BUILD_DIR="$(resolve_repo_path "$2")"
            shift 2
            ;;
        --perf)
            ENABLE_PERF="ON"
            shift
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

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G "Unix Makefiles" -DYOLO_ENABLE_RPP_PERF="${ENABLE_PERF}"
cmake --build "${BUILD_DIR}" --target yolov11_seg_postprocess_test -j"$(nproc)"
