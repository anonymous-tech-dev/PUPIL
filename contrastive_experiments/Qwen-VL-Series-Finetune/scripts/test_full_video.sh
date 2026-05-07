#!/bin/bash
# ============================================================================
# Test Script v2 — Evaluate on FULL-LENGTH videos (realistic CGBench eval)
# ============================================================================
# Data-parallel across NUM_GPUS: each GPU loads its own Qwen3-VL and
# processes ~N/NUM_GPUS samples. After all shards complete, the outputs
# are merged into single predictions.json / metrics.json and the shard
# files are deleted.
# ============================================================================
set -euo pipefail

# Comma-separated list of GPU ids to use; one Qwen3-VL per GPU.
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

# Full-length CGBench videos directory
FULL_VIDEO_DIR="${FULL_VIDEO_DIR:-/data/Pupil/CGBench/train_vids}"

ADAPTER_DIR="${ADAPTER_DIR:?ERROR: Set ADAPTER_DIR to your checkpoint}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NFRAMES="${NFRAMES:-}"
FPS="${FPS:-2.0}"
SAMPLING_MODE="${SAMPLING_MODE:-native}"

OUTPUT_DIR="${OUTPUT_DIR:-$ADAPTER_DIR/test_results_full_video}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST v2: Full-Length Video Evaluation (data-parallel)"
echo "  Base Model:  $MODEL_ID"
echo "  Adapter:     $ADAPTER_DIR"
echo "  Test Data:   $TEST_DATA"
echo "  Full Videos: $FULL_VIDEO_DIR"
echo "  Sampling:    $SAMPLING_MODE (fps=$FPS)"
echo "  GPUs:        $GPU_IDS  (num_shards=$NUM_GPUS)"
echo "  Output:      $OUTPUT_DIR"
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
            --sampling_mode "$SAMPLING_MODE" \
            --fps "$FPS" \
            --use_full_video \
            --full_video_dir "$FULL_VIDEO_DIR" \
            --shard_index "$IDX" \
            --num_shards "$NUM_GPUS" \
            > "$SHARD_LOG" 2>&1
    ) &
    PIDS+=($!)
done

# ── Wait for all shards to finish ──
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

# ── Cool-down, then merge ──
echo ""
echo "  All shards complete. Cooling down for 60s before merge..."
sleep 60

echo "  Merging shards..."
python tools/merge_shards.py \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_GPUS" \
    2>&1 | tee "$OUTPUT_DIR/merge.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  TEST COMPLETE (full-length videos, distributed)"
echo "  Metrics:     $OUTPUT_DIR/metrics.json"
echo "  Predictions: $OUTPUT_DIR/predictions.json"
echo "════════════════════════════════════════════════════════════════════"
