#!/bin/bash
# ============================================================================
# Evaluate ALL baselines on FULL-LENGTH videos (4-GPU sharded)
# ============================================================================
# Runs 3 baselines:
#   1. Qwen3-VL-8B-Instruct (base, no adapter)
#   2. Qwen3.5-9B (thinking OFF)
#   3. Qwen3.5-9B (thinking ON)
#
# Each baseline is sharded across 4 GPUs, then merged.
# ============================================================================
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

TEST_DATA="${TEST_DATA:-$CE_DIR/final_sft_data/test.json}"
FULL_VIDEO_DIR="${FULL_VIDEO_DIR:-/data/Pupil/CGBench/train_vids}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$CE_DIR/outputs}"

MAX_NEW_TOKENS_VL="${MAX_NEW_TOKENS_VL:-512}"
MAX_NEW_TOKENS_35="${MAX_NEW_TOKENS_35:-32768}"

# ════════════════════════════════════════════════════════════════════
# Helper: run sharded eval and merge
# ════════════════════════════════════════════════════════════════════
run_sharded_eval() {
    local EVAL_CMD="$1"    # python command template (use {GPU} {SHARD_IDX} {NUM_SHARDS} placeholders)
    local OUTPUT_DIR="$2"
    local LABEL="$3"

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  $LABEL"
    echo "  Output: $OUTPUT_DIR"
    echo "  GPUs: $GPU_IDS (${NUM_GPUS} shards)"
    echo "════════════════════════════════════════════════════════════════════"

    local PIDS=()
    for IDX in "${!GPU_ARR[@]}"; do
        local GPU="${GPU_ARR[$IDX]}"
        local SHARD_LOG="$OUTPUT_DIR/eval.shard${IDX}of${NUM_GPUS}.log"
        echo "  → Launching shard $IDX on GPU $GPU (log: $SHARD_LOG)"

        # Replace placeholders in command
        local CMD="${EVAL_CMD//\{GPU\}/$GPU}"
        CMD="${CMD//\{SHARD_IDX\}/$IDX}"
        CMD="${CMD//\{NUM_SHARDS\}/$NUM_GPUS}"

        (
            eval "CUDA_VISIBLE_DEVICES=$GPU $CMD" > "$SHARD_LOG" 2>&1
        ) &
        PIDS+=($!)
    done

    echo "  Waiting for $NUM_GPUS shards..."
    local FAILED=0
    for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
            echo "  ✗ Shard $i FAILED (see $OUTPUT_DIR/eval.shard${i}of${NUM_GPUS}.log)"
            FAILED=1
        else
            echo "  ✓ Shard $i finished"
        fi
    done

    if [[ $FAILED -ne 0 ]]; then
        echo "  ERROR: At least one shard failed for $LABEL. Skipping merge."
        return 1
    fi

    echo "  Merging shards..."
    python tools/merge_shards.py \
        --output_dir "$OUTPUT_DIR" \
        --num_shards "$NUM_GPUS" \
        2>&1 | tee "$OUTPUT_DIR/merge.log"

    echo "  ✓ $LABEL COMPLETE → $OUTPUT_DIR/predictions.json"
}


# ════════════════════════════════════════════════════════════════════
# 1. Qwen3-VL-8B-Instruct (base model, no adapter)
# ════════════════════════════════════════════════════════════════════
QWEN3VL_OUT="$OUTPUTS_ROOT/baseline_qwen3vl_8b_full_video"
QWEN3VL_CMD="python tools/evaluate_model.py \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --test_data_path $TEST_DATA \
    --output_dir $QWEN3VL_OUT \
    --batch_size 1 \
    --max_new_tokens $MAX_NEW_TOKENS_VL \
    --use_full_video \
    --full_video_dir $FULL_VIDEO_DIR \
    --shard_index {SHARD_IDX} \
    --num_shards {NUM_SHARDS}"

run_sharded_eval "$QWEN3VL_CMD" "$QWEN3VL_OUT" "Baseline: Qwen3-VL-8B-Instruct (full video)"


# ════════════════════════════════════════════════════════════════════
# 2. Qwen3.5-9B (thinking OFF)
# ════════════════════════════════════════════════════════════════════
QWEN35_OUT="$OUTPUTS_ROOT/baseline_qwen35_9b_full_video"
QWEN35_CMD="python $CE_DIR/eval_qwen35_baseline.py \
    --model_id Qwen/Qwen3.5-9B \
    --test_data_path $TEST_DATA \
    --output_dir $QWEN35_OUT \
    --max_new_tokens $MAX_NEW_TOKENS_35 \
    --disable_thinking \
    --use_full_video \
    --full_video_dir $FULL_VIDEO_DIR \
    --shard_index {SHARD_IDX} \
    --num_shards {NUM_SHARDS}"

run_sharded_eval "$QWEN35_CMD" "$QWEN35_OUT" "Baseline: Qwen3.5-9B (thinking OFF, full video)"


# ════════════════════════════════════════════════════════════════════
# 3. Qwen3.5-9B (thinking ON)
# ════════════════════════════════════════════════════════════════════
QWEN35T_OUT="$OUTPUTS_ROOT/baseline_qwen35_9b_thinking_full_video"
QWEN35T_CMD="python $CE_DIR/eval_qwen35_baseline.py \
    --model_id Qwen/Qwen3.5-9B \
    --test_data_path $TEST_DATA \
    --output_dir $QWEN35T_OUT \
    --max_new_tokens $MAX_NEW_TOKENS_35 \
    --enable_thinking \
    --use_full_video \
    --full_video_dir $FULL_VIDEO_DIR \
    --shard_index {SHARD_IDX} \
    --num_shards {NUM_SHARDS}"

run_sharded_eval "$QWEN35T_CMD" "$QWEN35T_OUT" "Baseline: Qwen3.5-9B (thinking ON, full video)"


# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL BASELINES COMPLETE (full-length videos)"
echo "  Results in: $OUTPUTS_ROOT/baseline_*_full_video/"
echo "════════════════════════════════════════════════════════════════════"
