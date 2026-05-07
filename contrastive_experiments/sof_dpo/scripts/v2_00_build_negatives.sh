#!/bin/bash
# v2_00_build_negatives.sh — Stage-0 v2: anti-abstention prompts + retries
# Generates rejected responses for each train sample under axis-ablated context.
# Sharded across 8 GPUs.  Hot-resumable.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-$REPO/old_dpo_revised_data_8b/negatives_v2}"
MAX_ROWS="${MAX_ROWS:--1}"
N_FRAMES_FULL="${N_FRAMES_FULL:-24}"
N_FRAMES_CLIP="${N_FRAMES_CLIP:-8}"
N_FRAMES_AUDIO="${N_FRAMES_AUDIO:-48}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LOG_DIR="$REPO/old_dpo_revised_data_8b/logs/v2_00_build_negatives"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Belt-and-suspenders: silence torchvision OOM fallback in qwen-vl-utils.
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"
python3 -c "import sys; sys.path.insert(0, '$REPO/scripts'); import decord_only_guard" \
    || { echo "decord_only_guard import failed"; exit 1; }

echo "[v2-00] $(date)  out=$OUT_DIR  shards=$NUM_SHARDS  max_rows=$MAX_ROWS  attempts=$MAX_ATTEMPTS"
echo "[v2-00] model=$MODEL_ID  fr_full=$N_FRAMES_FULL  fr_clip=$N_FRAMES_CLIP  fr_audio=$N_FRAMES_AUDIO"

PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
    LOG="$LOG_DIR/shard${SHARD}.log"
    echo "  launch shard $SHARD on GPU $SHARD  -> $LOG"
    CUDA_VISIBLE_DEVICES=$SHARD \
    NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD \
    PYTHONPATH="$REPO/scripts:$REPO:${PYTHONPATH:-}" \
    python3 -c "import decord_only_guard" 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=$SHARD \
    NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD \
    PYTHONPATH="$REPO/scripts:$REPO:${PYTHONPATH:-}" \
    python3 -u -c "
import decord_only_guard  # patches qwen_vl_utils + decord
import runpy, sys
sys.argv = [
    'sof_dpo_generate_negatives_v2.py',
    '--out-dir', '$OUT_DIR',
    '--model-id', '$MODEL_ID',
    '--max-rows', '$MAX_ROWS',
    '--n-frames-full', '$N_FRAMES_FULL',
    '--n-frames-clip', '$N_FRAMES_CLIP',
    '--n-frames-audio', '$N_FRAMES_AUDIO',
    '--max-new-tokens', '$MAX_NEW_TOKENS',
    '--max-attempts', '$MAX_ATTEMPTS',
]
runpy.run_path('$REPO/build_pairs/sof_dpo_generate_negatives_v2.py', run_name='__main__')
" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[v2-00] waiting for ${#PIDS[@]} shards ..."
FAIL=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then FAIL=$((FAIL+1)); fi
done
echo "[v2-00] done. failed shards: $FAIL"

echo "[v2-00] per-axis row counts (final):"
for ax in visual audio time priority; do
    n=$(cat "$OUT_DIR"/final_${ax}.shard*.jsonl 2>/dev/null | wc -l || echo 0)
    a=$(cat "$OUT_DIR"/attempts_${ax}.shard*.jsonl 2>/dev/null | wc -l || echo 0)
    echo "    $ax  final=$n  attempts=$a"
done
