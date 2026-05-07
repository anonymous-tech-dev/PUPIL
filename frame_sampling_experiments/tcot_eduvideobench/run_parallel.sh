#!/usr/bin/env bash
# ==============================================================================
#  TCoT Pupil — Data-Parallel Launcher
#  Spawns N shards (one per GPU), waits for all, merges by concatenation.
#
#  Usage:
#    bash run_parallel.sh [num_gpus] [extra_args...]
#
#  Examples:
#    bash run_parallel.sh                         # 8 GPUs, base Qwen3-VL
#    bash run_parallel.sh 8 --max-samples 5       # 5 samples per shard (smoke)
#    ADAPTER_DIR=/path/to/checkpoint-200 \
#      ADAPTER_TAG=T04_gradfix_ckpt200 \
#      bash run_parallel.sh 8                     # fine-tuned LoRA, 8 shards
# ==============================================================================
set -euo pipefail

_cleanup_done=0
_cleanup() {
    [[ $_cleanup_done -eq 1 ]] && return
    _cleanup_done=1
    trap '' INT TERM EXIT
    echo
    echo "🛑 Caught signal — killing all shard children…"
    pkill -P $$ -TERM 2>/dev/null || true
    sleep 2
    pkill -P $$ -KILL 2>/dev/null || true
    exit 130
}
trap _cleanup INT TERM

NUM_GPUS="${1:-8}"
shift 1 2>/dev/null || true
EXTRA_ARGS="$*"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

ADAPTER_INFO="${ADAPTER_DIR:-<none>} (tag=${ADAPTER_TAG:-<none>})"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TCoT × Pupil — Parallel Evaluation              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Shards   : $NUM_GPUS"
echo "║  Adapter  : $ADAPTER_INFO"
echo "║  Extra    : ${EXTRA_ARGS:-<none>}"
echo "║  Logs     : $LOG_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${LOG_DIR}/${TIMESTAMP}_shard${SHARD_ID}of${NUM_GPUS}.log"
    CMD="cd ${SCRIPT_DIR} && CUDA_VISIBLE_DEVICES=${SHARD_ID} \
        ADAPTER_DIR=\"${ADAPTER_DIR:-}\" ADAPTER_TAG=\"${ADAPTER_TAG:-}\" \
        python main.py --shard-id ${SHARD_ID} --num-shards ${NUM_GPUS} ${EXTRA_ARGS}"
    echo "🚀 Launching shard ${SHARD_ID} → ${LOG_FILE}"
    bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    sleep 1
done

echo ""
echo "⏳ Waiting for ${#PIDS[@]} shards to complete..."
echo "   Tail any shard:  tail -f ${LOG_DIR}/${TIMESTAMP}_shard0of${NUM_GPUS}.log"
echo ""

FAILURES=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "  ✅ Shard $i (PID $PID) finished"
    else
        echo "  ❌ Shard $i (PID $PID) FAILED (exit $?)"
        FAILURES=$((FAILURES + 1))
    fi
done

if [[ $FAILURES -gt 0 ]]; then
    echo ""
    echo "⚠️  $FAILURES shard(s) failed. Check logs in $LOG_DIR"
    echo "   Re-run the same command — it will hot-resume from where each shard left off."
fi

# ── Cooldown so all shard fsync/exit hooks settle before merge ──
COOLDOWN="${COOLDOWN:-60}"
echo ""
echo "⏱️  Cooldown ${COOLDOWN}s before merge..."
sleep "$COOLDOWN"

# Merge: concatenate all per-shard JSONLs into a single _merged.jsonl
echo ""
echo "🔀 Merging shard JSONLs..."
RESULTS_DIR="${SCRIPT_DIR}/results/Pupil"
if [[ -d "$RESULTS_DIR" ]]; then
    # Find all shard files matching this run's adapter+variant signature
    # For simplicity merge anything ending in shard*of${NUM_GPUS}_results.jsonl
    cd "$RESULTS_DIR"
    for SHARD0 in *_shard0of${NUM_GPUS}_results.jsonl; do
        [[ -e "$SHARD0" ]] || continue
        BASE="${SHARD0%_shard0of${NUM_GPUS}_results.jsonl}"
        MERGED="${BASE}_merged.jsonl"
        echo "  → $MERGED"
        : > "$MERGED"
        for s in $(seq 0 $((NUM_GPUS - 1))); do
            FN="${BASE}_shard${s}of${NUM_GPUS}_results.jsonl"
            [[ -f "$FN" ]] && cat "$FN" >> "$MERGED"
        done
        echo "    $(wc -l < "$MERGED") lines"
    done
fi

echo ""
echo "🎉 Done! Results at: ${RESULTS_DIR}/"
