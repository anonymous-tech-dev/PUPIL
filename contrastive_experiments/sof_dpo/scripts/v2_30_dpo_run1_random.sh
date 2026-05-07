#!/bin/bash
# v2_30_dpo_run1_random.sh — Run 1 = DPO on the v2 (anti-abstention) data,
# random shuffle (default HF behaviour).  Continues from the v2 SFT warm-start.
#
# Required env:
#   SFT_CKPT        — path to v2 SFT warmstart adapter dir
#                     (defaults to most-recent sof_sft_warmstart_v2_8b_*)
# Override-able:
#   GLOBAL_BATCH (32), LR (5e-7), BETA (0.1), EPOCHS (1)
#   VIDEO_MAX_FRAMES (32), FPS (2), MAX_SEQ_LENGTH (24576)
#
# Output: $CE_DIR/outputs/sof_dpo_v2_run1_random_8b_<tag>_<ts>/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

# Auto-pick latest v2 SFT warm-start if not given.
if [[ -z "${SFT_CKPT:-}" ]]; then
    SFT_CKPT=$(ls -1d "$CE_DIR/outputs/sof_sft_warmstart_v2_8b_"* 2>/dev/null | sort | tail -1)
    [[ -z "$SFT_CKPT" ]] && { echo "❌ no v2 SFT warmstart found, set SFT_CKPT=..."; exit 2; }
fi
[[ -d "$SFT_CKPT" ]] || { echo "❌ SFT_CKPT not a dir: $SFT_CKPT"; exit 2; }
echo "  SFT_CKPT auto = $SFT_CKPT"

TS=$(date +%Y%m%d_%H%M%S)
DATA_DIR="$REPO/old_dpo_revised_data_8b"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/sof_dpo_train.judged.mix80.json}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/sof_dpo_train.val.judged.mix80.json}"

# Recipe — mirror v1 DPO winner (32fr clip, lr 5e-7, beta 0.1, ep 1, bs 32).
: "${GLOBAL_BATCH:=32}"
: "${PER_DEV_BS:=1}"
: "${EPOCHS:=1}"
: "${LR:=5e-7}"
: "${BETA:=0.1}"
: "${DPO_LOSS:=sigmoid}"
: "${LORA_R:=128}"
: "${LORA_ALPHA:=128}"
: "${FPS:=2}"
: "${MAX_SEQ_LENGTH:=24576}"
: "${VIDEO_MAX_FRAMES:=32}"
: "${VIDEO_MAX_PIXELS:=$((96 * 32 * 32))}"        # 98304
: "${VIDEO_MIN_PIXELS:=$((32 * 32 * 32))}"        # 32768
: "${VIDEO_TOTAL_PIXELS:=$((96 * 32 * 32 * 32))}" # 3.1M
: "${MASTER_PORT:=29621}"

# Random shuffle (default HF) — explicitly keep NO_SHUFFLE_TRAIN unset.
unset NO_SHUFFLE_TRAIN || true

TAG="sof_dpo_v2_run1_random_8b_${DPO_LOSS}_b${BETA}_lr${LR}_ep${EPOCHS}_bs${GLOBAL_BATCH}_${VIDEO_MAX_FRAMES}fr_${TS}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$TAG}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  v2 DPO Run-1 (RANDOM SHUFFLE)
  TRAIN  : $TRAIN_DATA
  EVAL   : $EVAL_DATA
  SFT_CKPT: $SFT_CKPT
  OUT    : $OUTPUT_DIR
EOF

export TRAIN_DATA EVAL_DATA SFT_CKPT GLOBAL_BATCH PER_DEV_BS EPOCHS LR BETA \
       DPO_LOSS LORA_R LORA_ALPHA FPS MAX_SEQ_LENGTH VIDEO_MAX_FRAMES \
       VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS VIDEO_TOTAL_PIXELS MASTER_PORT

exec bash "$REPO/scripts/20_sof_dpo.sh"
