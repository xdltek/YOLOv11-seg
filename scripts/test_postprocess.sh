#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

show_help() {
    cat <<'EOF'
Usage: scripts/test_postprocess.sh [options]

Options:
  -b, --build-dir DIR   Build directory. Default: build
  -h, --help            Show this help message.

This script verifies that the standalone postprocess test target is built.
It does not require an RPP device.
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

required=(
    "${BUILD_DIR}/bin/yolov11_seg_postprocess_test"
    "${BUILD_DIR}/lib/librpp_yolov11_seg_postprocess.so"
)

for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing postprocess test artifact: ${path}" >&2
        exit 1
    fi
done

echo "YOLOv11-SEG postprocess standalone test artifacts are present."
