#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RPP_SERVER_BIN="/usr/local/rpp/bin/rpp_server"

build_dir="${ROOT_DIR}/build"
loop="20"
warmup="1"
source_width="640"
source_height="640"
detection_bin=""
prototype_bin=""
accuracy_check="0"
min_box_iou=""
max_box_diff=""
min_mask_iou=""
max_mask_diff_ratio=""

show_help() {
    cat <<'EOF'
Usage: scripts/run_postprocess_test.sh [options]

Options:
  -b, --build-dir DIR       Build directory. Default: build
  -l, --loop COUNT          Measured loop count. Default: 20
      --warmup COUNT        Warmup loop count. Default: 1
      --source-width N      Source image width for letterbox restore. Default: 640
      --source-height N     Source image height for letterbox restore. Default: 640
      --detection-bin PATH  Optional f32 detection output, shape [1,116,8400]
      --prototype-bin PATH  Optional f32 prototype output, shape [1,32,160,160]
      --accuracy-check      Compare RPP output with CPU OpenCV reference
      --min-box-iou F       Accuracy mode minimum box IoU. Binary default: 0.98
      --max-box-diff F      Accuracy mode maximum box coordinate diff in pixels. Binary default: 1.0
      --min-mask-iou F      Accuracy mode minimum mask IoU. Binary default: 0.90
      --max-mask-diff-ratio F
                            Accuracy mode maximum mask pixel diff ratio. Binary default: 0.15
  -h, --help                Show this help message.

Without --detection-bin/--prototype-bin, the demo generates synthetic YOLOv11-SEG outputs.
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

is_rpp_server_running() {
    pgrep -f "${RPP_SERVER_BIN}" >/dev/null 2>&1 || pgrep -x "rpp_server" >/dev/null 2>&1
}

ensure_rpp_server() {
    if [[ ! -x "${RPP_SERVER_BIN}" ]]; then
        return 0
    fi
    if is_rpp_server_running; then
        echo "RPP server: already running"
        return 0
    fi

    mkdir -p "${build_dir}/logs"
    local log_path="${build_dir}/logs/rpp_server.log"
    echo "RPP server: starting ${RPP_SERVER_BIN}"
    nohup "${RPP_SERVER_BIN}" > "${log_path}" 2>&1 &
    sleep 1

    if ! is_rpp_server_running; then
        echo "Failed to start rpp_server. See ${log_path}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build-dir)
            build_dir="$(resolve_repo_path "$2")"
            shift 2
            ;;
        -l|--loop)
            loop="$2"
            shift 2
            ;;
        --warmup)
            warmup="$2"
            shift 2
            ;;
        --source-width)
            source_width="$2"
            shift 2
            ;;
        --source-height)
            source_height="$2"
            shift 2
            ;;
        --detection-bin)
            detection_bin="$(resolve_repo_path "$2")"
            shift 2
            ;;
        --prototype-bin)
            prototype_bin="$(resolve_repo_path "$2")"
            shift 2
            ;;
        --accuracy-check)
            accuracy_check="1"
            shift
            ;;
        --min-box-iou)
            min_box_iou="$2"
            shift 2
            ;;
        --max-box-diff)
            max_box_diff="$2"
            shift 2
            ;;
        --min-mask-iou)
            min_mask_iou="$2"
            shift 2
            ;;
        --max-mask-diff-ratio)
            max_mask_diff_ratio="$2"
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

bin="${build_dir}/bin/yolov11_seg_postprocess_test"
if [[ ! -x "${bin}" ]]; then
    echo "Missing postprocess test binary: ${bin}" >&2
    echo "Run scripts/build_postprocess_test.sh first." >&2
    exit 1
fi

args=(--loop "${loop}" --warmup "${warmup}" --source-width "${source_width}" --source-height "${source_height}")
if [[ -n "${detection_bin}" || -n "${prototype_bin}" ]]; then
    if [[ -z "${detection_bin}" || -z "${prototype_bin}" ]]; then
        echo "--detection-bin and --prototype-bin must be provided together." >&2
        exit 1
    fi
    args+=(--detection-bin "${detection_bin}" --prototype-bin "${prototype_bin}")
fi
if [[ "${accuracy_check}" == "1" ]]; then
    args+=(--accuracy-check)
fi
if [[ -n "${min_box_iou}" ]]; then
    args+=(--min-box-iou "${min_box_iou}")
fi
if [[ -n "${max_box_diff}" ]]; then
    args+=(--max-box-diff "${max_box_diff}")
fi
if [[ -n "${min_mask_iou}" ]]; then
    args+=(--min-mask-iou "${min_mask_iou}")
fi
if [[ -n "${max_mask_diff_ratio}" ]]; then
    args+=(--max-mask-diff-ratio "${max_mask_diff_ratio}")
fi

ensure_rpp_server
echo "Binary: ${bin}"
"${bin}" "${args[@]}"
