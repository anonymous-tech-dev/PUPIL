#!/bin/bash
# 02_score_margins.sh — Forward-pass reference Qwen3-VL-8B over each filtered
# pair under the FULL-context prompt that DPO will see at training time.
# Logs the implicit reward margin (raw and length-normalised) to the JSONL.
#
# Sharded across 8 GPUs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NUM_SHARDS="${NUM_SHARDS:-8}"
IN="${IN:-$REPO/data/pairs_after_filter.jsonl}"
OUT="${OUT:-$REPO/data/pairs_with_margin.jsonl}"   # per-shard suffix added by script
N_FRAMES="${N_FRAMES:-24}"
MAX_ROWS="${MAX_ROWS:--1}"
LOG_DIR="$REPO/logs/02_score_margins"
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG="$LOG_DIR/shard${SHARD}.log"
    echo "  launch margin-shard $SHARD on GPU $SHARD  -> $LOG"
    CUDA_VISIBLE_DEVICES=$SHARD \
    NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD \
    python3 "$REPO/build_pairs/sof_dpo_score_margins.py" \
        --in-jsonl "$IN" \
        --out-jsonl "$OUT" \
        --n-frames "$N_FRAMES" \
        --max-rows "$MAX_ROWS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done
echo "[02] waiting for ${#PIDS[@]} shards ..."
FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then FAIL=$((FAIL+1)); fi
done
echo "[02] done. failed shards: $FAIL"

# Quick histogram report
HIST_GLOB="$(dirname "$OUT")/$(basename "${OUT%.jsonl}").shard*.jsonl"
python3 "$REPO/build_pairs/sof_dpo_margin_histogram.py" --in-glob "$HIST_GLOB"
