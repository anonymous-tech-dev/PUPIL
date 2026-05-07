#!/bin/bash
# v2_06_curriculum.sh — Build the easy→hard curriculum-ordered variants
# (Run 2). Uses the qwen3_vl baseline benchmark to estimate per-(axis,
# cognitive_category) difficulty.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO/old_dpo_revised_data_8b"

IN_TRAIN="${IN_TRAIN:-$DATA_DIR/sof_dpo_train.judged.json}"
IN_SFT="${IN_SFT:-$DATA_DIR/sof_sft_warmstart.no_transcript.judged.json}"
OUT_TRAIN="${OUT_TRAIN:-$DATA_DIR/sof_dpo_train.judged.curriculum.json}"
OUT_SFT="${OUT_SFT:-$DATA_DIR/sof_sft_warmstart.no_transcript.judged.curriculum.json}"
BASELINE_DIR="${BASELINE_DIR:-/workspace/Pupil/mllm_evaluation/results/qwen3_vl/final_1k_benchmark}"

python3 "$REPO/build_pairs/sof_dpo_curriculum_sort.py" \
    --in-train "$IN_TRAIN" \
    --in-sft   "$IN_SFT" \
    --out-train "$OUT_TRAIN" \
    --out-sft   "$OUT_SFT" \
    --baseline-results-dir "$BASELINE_DIR"
