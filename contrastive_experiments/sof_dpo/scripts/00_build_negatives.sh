#!/bin/bash
# 00_build_negatives.sh — Generate axis-ablated rejected responses on 8 GPUs.
# Each GPU runs an independent shard of the train set across all 4 axes.
#
# Usage:   bash 00_build_negatives.sh                 # full run
#          MAX_ROWS=20 bash 00_build_negatives.sh     # debug per-axis cap
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-$REPO/data/negatives_qwen3vl8b}"
MAX_ROWS="${MAX_ROWS:--1}"
N_FRAMES_FULL="${N_FRAMES_FULL:-24}"
N_FRAMES_CLIP="${N_FRAMES_CLIP:-8}"
# Audio axis uses a higher frame budget than other axes — the frames diagnostic
# (frames_diag/audio_24v64.judged.jsonl) showed N=24 sits at 57.9% "high-win",
# borderline-contaminated by frame-starvation rather than missing-audio failures.
N_FRAMES_AUDIO="${N_FRAMES_AUDIO:-48}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
LOG_DIR="$REPO/logs/00_build_negatives"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[00] $(date)  out=$OUT_DIR  shards=$NUM_SHARDS  max_rows=$MAX_ROWS"

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG="$LOG_DIR/shard${SHARD}.log"
    echo "  launch shard $SHARD on GPU $SHARD  -> $LOG"
    CUDA_VISIBLE_DEVICES=$SHARD \
    NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD \
    python3 "$REPO/build_pairs/sof_dpo_generate_negatives.py" \
        --out-dir "$OUT_DIR" \
        --max-rows "$MAX_ROWS" \
        --n-frames-full "$N_FRAMES_FULL" \
        --n-frames-clip "$N_FRAMES_CLIP" \
        --n-frames-audio "$N_FRAMES_AUDIO" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[00] waiting for ${#PIDS[@]} shards ..."
FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then FAIL=$((FAIL+1)); fi
done
echo "[00] done. failed shards: $FAIL"

# Report per-axis counts
echo "[00] per-axis row counts:"
for ax in visual audio time priority; do
    n=$(cat "$OUT_DIR"/negatives_${ax}.shard*.jsonl 2>/dev/null | wc -l || echo 0)
    echo "    $ax  $n"
done
