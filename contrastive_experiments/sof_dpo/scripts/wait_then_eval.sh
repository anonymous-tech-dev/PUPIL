#!/usr/bin/env bash
# ==============================================================================
#  wait_then_eval.sh
#
#  1. Polls all 8 GPUs every 30s.
#  2. Waits until ALL GPUs are simultaneously idle (util < IDLE_UTIL_PCT  AND
#     mem-used < IDLE_MEM_MB) for IDLE_WINDOW_SEC continuously.
#     If utilization spikes on ANY GPU, the idle timer resets to 0.
#  3. Once the idle window is satisfied, runs the Pupil evaluation on
#     the latest checkpoint of the just-finished SFT run, using the EXACT same
#     video config (frames, pixels, fps) as training.
#
#  Run me in a screen / tmux:
#     screen -S eval
#     bash /workspace/Pupil/contrastive_experiments/sof_dpo/scripts/wait_then_eval.sh
#     # Ctrl-A D to detach
# ==============================================================================
set -euo pipefail

# ───────────────────── Config ──────────────────────────────────────────────
RUN_TAG="sof_sft_warmstart_NOTX_lr2e-5_ep3_bs64_24576seq_256fr"
RUN_DIR="/workspace/Pupil/contrastive_experiments/sof_dpo/outputs/${RUN_TAG}"

EVAL_REPO="/workspace/Pupil/mllm_evaluation"
EVAL_LAUNCHER="${EVAL_REPO}/run_final_benchmark.sh"   # uses run_model() inside
EVAL_MODEL="qwen3_vl_ft"

# Idle detection thresholds
IDLE_UTIL_PCT=5          # GPU util %  below which counts as "idle"
IDLE_MEM_MB=2000         # GPU mem MB below which counts as "idle"
                         # (training holds ~50 GB/GPU; 2 GB is conservative)
IDLE_WINDOW_SEC=$((20 * 60))   # 20 minutes
POLL_SEC=30              # poll every 30s

# Video config — MUST match training exactly
export VIDEO_MAX_FRAMES=256
export VIDEO_FPS=1
export VIDEO_MAX_PIXELS=98304
export VIDEO_MIN_PIXELS=32768
export VIDEO_TOTAL_PIXELS=25165824

# Optional: limit max new tokens / generation knobs (uncomment if needed)
# export GEN_MAX_NEW_TOKENS=2048
# export GEN_DO_SAMPLE=False
# export GEN_TEMPERATURE=0.0

# Memory hygiene for inference (same tricks as training)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=16777216
export DECORD_NUM_THREADS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

LOG_DIR="${RUN_DIR}/eval_logs"
mkdir -p "$LOG_DIR"
WAIT_LOG="${LOG_DIR}/wait_then_eval.$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAIT_LOG"; }

# ───────────────────── Step 1: Wait for GPUs idle ──────────────────────────
log "═══════════════════════════════════════════════════════════════"
log "Watching GPUs. Need ALL 8 idle (util<${IDLE_UTIL_PCT}% AND mem<${IDLE_MEM_MB}MB) for ${IDLE_WINDOW_SEC}s."
log "Polling every ${POLL_SEC}s."
log "═══════════════════════════════════════════════════════════════"

idle_start=0
while true; do
    # nvidia-smi outputs e.g. "0, 5, 96384"  (idx, util%, memMB)
    busy=0
    while IFS=, read -r idx util mem; do
        util=$(echo "$util" | tr -d ' %')
        mem=$(echo "$mem"   | tr -d ' MiB')
        if [[ "$util" -ge "$IDLE_UTIL_PCT" || "$mem" -ge "$IDLE_MEM_MB" ]]; then
            busy=1
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                       --format=csv,noheader,nounits)

    now=$(date +%s)
    if [[ $busy -eq 1 ]]; then
        if [[ $idle_start -ne 0 ]]; then
            log "Activity detected — resetting idle timer."
        fi
        idle_start=0
    else
        if [[ $idle_start -eq 0 ]]; then
            idle_start=$now
            log "All GPUs idle — starting 20-min countdown."
        else
            elapsed=$(( now - idle_start ))
            remaining=$(( IDLE_WINDOW_SEC - elapsed ))
            if [[ $elapsed -ge $IDLE_WINDOW_SEC ]]; then
                log "✅ GPUs idle for ${elapsed}s (≥ ${IDLE_WINDOW_SEC}s). Proceeding to eval."
                break
            fi
            # status print every 5 min during countdown
            if (( elapsed % 300 < POLL_SEC )); then
                log "Still idle. Elapsed=${elapsed}s, remaining=${remaining}s."
            fi
        fi
    fi
    sleep "$POLL_SEC"
done

# ───────────────────── Step 2: Pick latest checkpoint ──────────────────────
LATEST_CKPT=$(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null \
              | awk -F'checkpoint-' '{print $2"\t"$0}' \
              | sort -n \
              | tail -1 \
              | cut -f2-)

if [[ -z "$LATEST_CKPT" || ! -d "$LATEST_CKPT" ]]; then
    log "❌ No checkpoint found under ${RUN_DIR}. Aborting."
    exit 1
fi

# Sanity: make sure adapter files exist
if [[ ! -f "${LATEST_CKPT}/adapter_config.json" && ! -f "${LATEST_CKPT}/adapter_model.safetensors" ]]; then
    log "⚠️  ${LATEST_CKPT} doesn't look like a LoRA adapter dir (no adapter_config.json/adapter_model.safetensors)."
    log "    Listing contents for debugging:"
    ls -la "$LATEST_CKPT" | tee -a "$WAIT_LOG"
    log "    Continuing anyway — qwen3_vl_ft loader will error if invalid."
fi

CKPT_STEP=$(basename "$LATEST_CKPT" | sed 's/checkpoint-//')
ADAPTER_TAG="${RUN_TAG}__step${CKPT_STEP}"

export ADAPTER_DIR="$LATEST_CKPT"
export ADAPTER_TAG="$ADAPTER_TAG"
# tells run_final_benchmark.sh to put outputs under
# results/qwen3_vl_ft/final_1k_benchmark_ft_${ADAPTER_TAG}/
export EVAL_OUTPUT_FOLDER="final_1k_benchmark"

log "═══════════════════════════════════════════════════════════════"
log "Launching evaluation:"
log "  MODEL         = ${EVAL_MODEL}"
log "  ADAPTER_DIR   = ${ADAPTER_DIR}"
log "  ADAPTER_TAG   = ${ADAPTER_TAG}"
log "  VIDEO_MAX_FRAMES   = ${VIDEO_MAX_FRAMES}"
log "  VIDEO_MAX_PIXELS   = ${VIDEO_MAX_PIXELS}"
log "  VIDEO_MIN_PIXELS   = ${VIDEO_MIN_PIXELS}"
log "  VIDEO_TOTAL_PIXELS = ${VIDEO_TOTAL_PIXELS}"
log "  VIDEO_FPS          = ${VIDEO_FPS}"
log "═══════════════════════════════════════════════════════════════"

# ───────────────────── Step 3: Run benchmark ───────────────────────────────
cd "$EVAL_REPO"
EVAL_LOG="${LOG_DIR}/eval.${ADAPTER_TAG}.$(date +%Y%m%d_%H%M%S).log"
log "Running: bash ${EVAL_LAUNCHER} ${EVAL_MODEL}"
log "Eval log: ${EVAL_LOG}"

bash "$EVAL_LAUNCHER" "$EVAL_MODEL" 2>&1 | tee -a "$EVAL_LOG"
RC=${PIPESTATUS[0]}

if [[ $RC -eq 0 ]]; then
    log "✅ Evaluation finished successfully."
else
    log "❌ Evaluation exited with code $RC."
fi
exit $RC
