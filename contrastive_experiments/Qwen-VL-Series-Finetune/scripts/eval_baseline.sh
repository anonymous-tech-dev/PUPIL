#!/bin/bash
# ============================================================================
# Base Model Evaluation — Qwen3-VL-8B-Instruct (No Fine-tuning)
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/baseline_qwen3vl_8b}"
mkdir -p "$OUTPUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

echo "════════════════════════════════════════════════════════════════════"
echo "  BASE MODEL EVALUATION"
echo "  Model: $MODEL_ID"
echo "  Test Data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--video_dir $VIDEO_DIR"

python tools/evaluate_model.py \
    --model_id "$MODEL_ID" \
    --test_data_path "$TEST_DATA" \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  BASELINE EVALUATION COMPLETE"
echo "  Results: $OUTPUT_DIR/metrics.json"
echo "════════════════════════════════════════════════════════════════════"
