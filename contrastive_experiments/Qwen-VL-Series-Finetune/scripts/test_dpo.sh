#!/bin/bash
# ============================================================================
# DPO test — evaluate a DPO-trained LoRA adapter on Pupil.
# Thin wrapper around test_full_video_matched.sh that points at the EduBench
# test JSON and the DPO output dir by default.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"

ADAPTER_DIR="${ADAPTER_DIR:?ERROR: set ADAPTER_DIR to a DPO checkpoint or final dir}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

# Default to the Pupil test set used by the SFT pipeline.
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TEST_DATA="${TEST_DATA:-$DATA_DIR/test.json}"

# Match training defaults
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
FPS="${FPS:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-$ADAPTER_DIR/test_results_full_video}"
# CGBench full-length lecture videos (matches what SFT baselines use).
# This MUST exist on disk; otherwise the evaluator silently falls back to
# clue_vids and the result is not comparable to SFT baselines.
FULL_VIDEO_DIR="${FULL_VIDEO_DIR:-/data/Pupil/CGBench/train_vids}"
if [[ ! -d "$FULL_VIDEO_DIR" ]]; then
    echo "ERROR: FULL_VIDEO_DIR=$FULL_VIDEO_DIR does not exist." >&2
    echo "Eval would silently fall back to clue_vids and produce uncomparable numbers." >&2
    exit 1
fi

export ADAPTER_DIR MODEL_ID TEST_DATA MAX_SEQ_LENGTH FPS GPU_IDS BATCH_SIZE \
       MAX_NEW_TOKENS OUTPUT_DIR FULL_VIDEO_DIR

bash "$REPO_ROOT/scripts/test_full_video_matched.sh"

# ─── GPT-5 judge ────────────────────────────────────────────────────
PREDICTIONS="$OUTPUT_DIR/predictions.json"
if [[ -f "$PREDICTIONS" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  GPT-5 JUDGE — $PREDICTIONS"
    echo "════════════════════════════════════════════════════════════════════"
    python "$REPO_ROOT/tools/gpt5_mcq_judge.py" \
        --predictions_path "$PREDICTIONS" \
        --num_samples -1 --max_workers 16
else
    echo "WARNING: predictions.json not found at $PREDICTIONS — skipping judge." >&2
fi
