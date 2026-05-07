#!/bin/bash
# ============================================================================
# Test Script — Load LoRA Adapters + Evaluate on Test Set
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

ADAPTER_DIR="${ADAPTER_DIR:?ERROR: Set ADAPTER_DIR to your checkpoint, e.g. ../outputs/V-02_generative_...}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

OUTPUT_DIR="${OUTPUT_DIR:-$ADAPTER_DIR/test_results}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST: LoRA Adapter Evaluation"
echo "  Base Model:  $MODEL_ID"
echo "  Adapter:     $ADAPTER_DIR"
echo "  Test Data:   $TEST_DATA"
echo "  Output:      $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--video_dir $VIDEO_DIR"

python tools/evaluate_model.py \
    --model_id "$MODEL_ID" \
    --adapter_path "$ADAPTER_DIR" \
    --test_data_path "$TEST_DATA" \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$OUTPUT_DIR/test.log"

# Append _16_frames suffix to outputs (clue_vids @ native fps ≈ 16 frames)
for f in predictions metrics judge_details; do
    [[ -f "$OUTPUT_DIR/${f}.json" ]] && mv "$OUTPUT_DIR/${f}.json" "$OUTPUT_DIR/${f}_16_frames.json"
done

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST COMPLETE"
echo "  Metrics:     $OUTPUT_DIR/metrics_16_frames.json"
echo "  Predictions: $OUTPUT_DIR/predictions_16_frames.json"
echo "════════════════════════════════════════════════════════════════════"
