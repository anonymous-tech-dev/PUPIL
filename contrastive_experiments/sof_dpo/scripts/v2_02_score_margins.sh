#!/bin/bash
# v2_02_score_margins.sh — Reference-policy log-prob margins (8-GPU sharded).
# Reuses the trusted v1 sof_dpo_score_margins.py (no transcript-related logic
# changes needed; the FT runs all use the no-transcript prompt at training
# time, but the margin under the FULL context is still the right saturation
# proxy — and we then drop saturated pairs in 03_assemble.sh).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NUM_SHARDS="${NUM_SHARDS:-8}"
IN="${IN:-$REPO/old_dpo_revised_data_8b/pairs_after_filter.jsonl}"
OUT="${OUT:-$REPO/old_dpo_revised_data_8b/pairs_with_margin.jsonl}"
N_FRAMES="${N_FRAMES:-24}"
MAX_ROWS="${MAX_ROWS:--1}"
LOG_DIR="$REPO/old_dpo_revised_data_8b/logs/v2_02_score_margins"
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

# decord guard preflight
python3 -c "import sys; sys.path.insert(0, '$REPO/scripts'); import decord_only_guard" \
    || { echo "decord_only_guard import failed"; exit 1; }

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG="$LOG_DIR/shard${SHARD}.log"
    echo "  launch margin-shard $SHARD on GPU $SHARD  -> $LOG"
    CUDA_VISIBLE_DEVICES=$SHARD \
    NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD \
    PYTHONPATH="$REPO/scripts:$REPO:${PYTHONPATH:-}" \
    python3 -u -c "
import decord_only_guard
import runpy, sys
sys.argv = [
    'sof_dpo_score_margins.py',
    '--in-jsonl', '$IN',
    '--out-jsonl', '$OUT',
    '--n-frames', '$N_FRAMES',
    '--max-rows', '$MAX_ROWS',
]
runpy.run_path('$REPO/build_pairs/sof_dpo_score_margins.py', run_name='__main__')
" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[v2-02] waiting for ${#PIDS[@]} shards ..."
FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then FAIL=$((FAIL+1)); fi
done
echo "[v2-02] done. failed shards: $FAIL"

# Histogram report (best-effort).
HIST_GLOB="$(dirname "$OUT")/$(basename "${OUT%.jsonl}").shard*.jsonl"
python3 "$REPO/build_pairs/sof_dpo_margin_histogram.py" --in-glob "$HIST_GLOB" || true
