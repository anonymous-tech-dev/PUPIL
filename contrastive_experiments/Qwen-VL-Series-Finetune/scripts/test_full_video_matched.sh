#!/bin/bash
# ============================================================================
# Test Script — Matched to training params (max_seq_length, FPS, video_max_pixels)
# ============================================================================
# Evaluates on FULL-LENGTH videos using the same pixel budget and FPS that
# the model was trained with. This ensures train-test consistency.
#
# Key params inherited from training:
#   MAX_SEQ_LENGTH → auto-computes video_max_pixels = (MSL/1000)*32*32
#   FPS            → same FPS used during training
# ============================================================================
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
# GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

FULL_VIDEO_DIR="${FULL_VIDEO_DIR:-/data/Pupil/CGBench/train_vids}"

ADAPTER_DIR="${ADAPTER_DIR:?ERROR: Set ADAPTER_DIR to your checkpoint}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ════════════════════════════════════════════════════════════════════
# Matched training params — these MUST match what the model was trained with
# ════════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
FPS="${FPS:-1}"

# Auto-compute video_max_pixels from max_seq_length (Qwen3-VL token math)
VIDEO_MAX_PIXELS=$((MAX_SEQ_LENGTH / 1000 * 32 * 32))
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((VIDEO_MAX_PIXELS / 4))}"

OUTPUT_DIR="${OUTPUT_DIR:-$ADAPTER_DIR/test_results_full_video}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST: Matched Training Params (Full-Length Videos)"
echo "  Base Model:  $MODEL_ID"
echo "  Adapter:     $ADAPTER_DIR"
echo "  Test Data:   $TEST_DATA"
echo "  Full Videos: $FULL_VIDEO_DIR"
echo "  Matched Settings:"
echo "    max_seq_length:   $MAX_SEQ_LENGTH"
echo "    FPS:              $FPS"
echo "    video_max_pixels: $VIDEO_MAX_PIXELS"
echo "    video_min_pixels: $VIDEO_MIN_PIXELS"
echo "  GPUs: $GPU_IDS  (num_shards=$NUM_GPUS)"
echo "  Output: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--video_dir $VIDEO_DIR"

# ── Launch one eval process per GPU ──
PIDS=()
for IDX in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$IDX]}"
    SHARD_LOG="$OUTPUT_DIR/test.shard${IDX}of${NUM_GPUS}.log"
    echo "  → Launching shard $IDX/$NUM_GPUS on GPU $GPU (log: $SHARD_LOG)"
    (
        CUDA_VISIBLE_DEVICES="$GPU" \
        python tools/evaluate_model.py \
            --model_id "$MODEL_ID" \
            --adapter_path "$ADAPTER_DIR" \
            --test_data_path "$TEST_DATA" \
            $VIDEO_ARG \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --sampling_mode fps \
            --fps "$FPS" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --video_max_pixels "$VIDEO_MAX_PIXELS" \
            --video_min_pixels "$VIDEO_MIN_PIXELS" \
            --use_full_video \
            --full_video_dir "$FULL_VIDEO_DIR" \
            --shard_index "$IDX" \
            --num_shards "$NUM_GPUS" \
            > "$SHARD_LOG" 2>&1
    ) &
    PIDS+=($!)
done

# ── Wait for all shards ──
echo ""
echo "  Waiting for all $NUM_GPUS shards to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  ✗ Shard $i FAILED (see $OUTPUT_DIR/test.shard${i}of${NUM_GPUS}.log)"
        FAILED=1
    else
        echo "  ✓ Shard $i finished"
    fi
done

if [[ $FAILED -ne 0 ]]; then
    echo "ERROR: At least one shard failed. Not merging."
    exit 1
fi

# ── Merge ──
echo ""
echo "  All shards complete. Cooling down for 60s before merge..."
sleep 60

echo "  Merging shards..."
python tools/merge_shards.py \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_GPUS" \
    2>&1 | tee "$OUTPUT_DIR/merge.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST COMPLETE (matched training params, full-length videos)"
echo "  Metrics:     $OUTPUT_DIR/metrics.json"
echo "  Predictions: $OUTPUT_DIR/predictions.json"
echo "════════════════════════════════════════════════════════════════════"
