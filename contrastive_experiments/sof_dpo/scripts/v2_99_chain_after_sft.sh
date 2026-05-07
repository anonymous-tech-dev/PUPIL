#!/bin/bash
# v2_99_chain_after_sft.sh — Background watcher.  Polls for the v2 SFT
# warmstart final-adapter file; when it appears AND no training process
# is still running, launches DPO Run-1, then DPO Run-2, then both evals.
#
# Usage:
#   nohup bash scripts/v2_99_chain_after_sft.sh > /tmp/v2_chain.log 2>&1 &
set -uo pipefail   # no -e: we want to keep going if a step fails

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

# Find the most recent v2 SFT run dir
SFT_DIR=$(ls -1d "$CE_DIR/outputs/sof_sft_warmstart_v2_8b_"* 2>/dev/null | sort | tail -1)
[[ -z "$SFT_DIR" ]] && { echo "❌ no v2 SFT run dir found"; exit 2; }
echo "[chain] watching SFT_DIR=$SFT_DIR"

# Sentinel: trainer writes adapter_model.safetensors at end of training,
# then no train_sft.py / deepspeed processes are running on those args.
SENTINEL="$SFT_DIR/adapter_model.safetensors"

while true; do
    if [[ -f "$SENTINEL" ]]; then
        # Belt-and-suspenders: also confirm no running train_sft.py for this dir
        if ! pgrep -af "train_sft.py" | grep -q "$(basename "$SFT_DIR")"; then
            echo "[chain] $(date)  SFT done — sentinel exists & no train_sft running"
            break
        fi
    fi
    # Print progress dot every 60s so the log shows life
    echo -n "."; sleep 60
done
echo ""

# ── Step 1: DPO Run-1 (random) ─────────────────────────────────────────────
echo "[chain] $(date)  ▶ launching v2 DPO Run-1 (random)"
SFT_CKPT="$SFT_DIR" bash "$REPO/scripts/v2_30_dpo_run1_random.sh"
RUN1_RC=$?
RUN1_DIR=$(ls -1d "$CE_DIR/outputs/sof_dpo_v2_run1_random_8b_"* 2>/dev/null | sort | tail -1)
echo "[chain] $(date)  Run-1 done rc=$RUN1_RC  out=$RUN1_DIR"

# ── Step 2: DPO Run-2 (curriculum) ─────────────────────────────────────────
echo "[chain] $(date)  ▶ launching v2 DPO Run-2 (curriculum)"
SFT_CKPT="$SFT_DIR" bash "$REPO/scripts/v2_31_dpo_run2_curriculum.sh"
RUN2_RC=$?
RUN2_DIR=$(ls -1d "$CE_DIR/outputs/sof_dpo_v2_run2_curriculum_8b_"* 2>/dev/null | sort | tail -1)
echo "[chain] $(date)  Run-2 done rc=$RUN2_RC  out=$RUN2_DIR"

# ── Step 3: Eval Run-1 ─────────────────────────────────────────────────────
if [[ -n "$RUN1_DIR" ]]; then
    LATEST_CKPT=$(ls -1d "$RUN1_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "[chain] $(date)  ▶ eval Run-1 ckpt=$LATEST_CKPT"
    ADAPTER_DIR="$LATEST_CKPT" \
    ADAPTER_TAG="v2_run1_random_$(basename "$LATEST_CKPT")" \
    EVAL_OUTPUT_FOLDER="final_1k_benchmark_v2" \
    bash "$REPO/scripts/v2_40_eval.sh"
    echo "[chain] $(date)  Run-1 eval done"
fi

# ── Step 4: Eval Run-2 ─────────────────────────────────────────────────────
if [[ -n "$RUN2_DIR" ]]; then
    LATEST_CKPT=$(ls -1d "$RUN2_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "[chain] $(date)  ▶ eval Run-2 ckpt=$LATEST_CKPT"
    ADAPTER_DIR="$LATEST_CKPT" \
    ADAPTER_TAG="v2_run2_curriculum_$(basename "$LATEST_CKPT")" \
    EVAL_OUTPUT_FOLDER="final_1k_benchmark_v2" \
    bash "$REPO/scripts/v2_40_eval.sh"
    echo "[chain] $(date)  Run-2 eval done"
fi

# ── Step 5: Eval SFT-only ──────────────────────────────────────────────────
LATEST_SFT_CKPT=$(ls -1d "$SFT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [[ -z "$LATEST_SFT_CKPT" ]]; then LATEST_SFT_CKPT="$SFT_DIR"; fi
echo "[chain] $(date)  ▶ eval SFT-only ckpt=$LATEST_SFT_CKPT"
ADAPTER_DIR="$LATEST_SFT_CKPT" \
ADAPTER_TAG="v2_sft_only_$(basename "$LATEST_SFT_CKPT")" \
EVAL_OUTPUT_FOLDER="final_1k_benchmark_v2" \
bash "$REPO/scripts/v2_40_eval.sh"
echo "[chain] $(date)  SFT-only eval done"

echo "[chain] $(date)  🏁 ALL DONE"
echo "  SFT:  $SFT_DIR"
echo "  Run1: $RUN1_DIR"
echo "  Run2: $RUN2_DIR"
