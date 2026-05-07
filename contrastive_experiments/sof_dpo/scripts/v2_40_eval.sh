#!/bin/bash
# v2_40_eval.sh — Run final_1k_benchmark eval on a v2 fine-tuned LoRA adapter.
#
# Required env:
#   ADAPTER_DIR     — absolute path to the LoRA checkpoint to evaluate
# Optional:
#   ADAPTER_TAG     — short name (default = basename of ADAPTER_DIR)
#   EVAL_OUTPUT_FOLDER — output subdir (default = final_1k_benchmark_v2)
#   VIDEO_FPS / VIDEO_MAX_FRAMES — must match training (default 2 / 32)
#
# Output: results/qwen3_vl_ft/<EVAL_OUTPUT_FOLDER>_ft_<ADAPTER_TAG>/
set -euo pipefail

# scripts/  ->  sof_dpo/  ->  contrastive_experiments/  ->  Pupil/
REPO="$(cd "$(dirname "$0")/../../.." && pwd)/mllm_evaluation"

[[ -z "${ADAPTER_DIR:-}" ]] && { echo "❌ ADAPTER_DIR is required"; exit 2; }
[[ -d "${ADAPTER_DIR}" ]]  || { echo "❌ ADAPTER_DIR not a dir: $ADAPTER_DIR"; exit 2; }

# Prefer a checkpoint-* subdir if ADAPTER_DIR is the run's parent.
if ls -d "$ADAPTER_DIR"/checkpoint-* >/dev/null 2>&1; then
    LATEST_CKPT=$(ls -d "$ADAPTER_DIR"/checkpoint-* | sort -t- -k2 -n | tail -1)
    echo "  ADAPTER_DIR was a run dir → resolved to latest ckpt: $LATEST_CKPT"
    export ADAPTER_DIR="$LATEST_CKPT"
fi

export ADAPTER_TAG="${ADAPTER_TAG:-$(basename "$ADAPTER_DIR")}"
export EVAL_OUTPUT_FOLDER="${EVAL_OUTPUT_FOLDER:-final_1k_benchmark_v2}"

# Match training preprocessing
export VIDEO_FPS="${VIDEO_FPS:-2}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-32}"
export VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-262144}"        # 256 * 32 * 32
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-32768}"
export VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-$((VIDEO_MAX_FRAMES * VIDEO_MAX_PIXELS))}"

# Greedy decoding (DPO models prefer this)
export GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-0}"

cat <<EOF
════════════════════════════════════════════════════════════════════
  v2 EVAL  ($EVAL_OUTPUT_FOLDER)
  ADAPTER_DIR: $ADAPTER_DIR
  ADAPTER_TAG: $ADAPTER_TAG
  Video      : fps=$VIDEO_FPS  max_frames=$VIDEO_MAX_FRAMES  max_px=$VIDEO_MAX_PIXELS
  Output dir : results/qwen3_vl_ft/${EVAL_OUTPUT_FOLDER}_ft_${ADAPTER_TAG}
════════════════════════════════════════════════════════════════════
EOF

cd "$REPO"
exec bash run_final_benchmark.sh qwen3_vl_ft
