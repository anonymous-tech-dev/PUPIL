#!/bin/bash
# v2_31_dpo_run2_curriculum.sh — Run 2 = DPO on the v2 *curriculum* data,
# easy→hard, sequential sampler (NO_SHUFFLE_TRAIN=1).  Continues from the
# same v2 SFT warmstart as Run 1 so the only difference is data ordering.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

if [[ -z "${SFT_CKPT:-}" ]]; then
    SFT_CKPT=$(ls -1d "$CE_DIR/outputs/sof_sft_warmstart_v2_8b_"* 2>/dev/null | sort | tail -1)
    [[ -z "$SFT_CKPT" ]] && { echo "❌ no v2 SFT warmstart found, set SFT_CKPT=..."; exit 2; }
fi
[[ -d "$SFT_CKPT" ]] || { echo "❌ SFT_CKPT not a dir: $SFT_CKPT"; exit 2; }
echo "  SFT_CKPT auto = $SFT_CKPT"

TS=$(date +%Y%m%d_%H%M%S)
DATA_DIR="$REPO/old_dpo_revised_data_8b"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/sof_dpo_train.judged.curriculum.mix80.json}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/sof_dpo_train.val.judged.mix80.json}"

# Same recipe as Run-1 — only difference is data order + NO_SHUFFLE_TRAIN.
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
: "${VIDEO_MAX_PIXELS:=$((96 * 32 * 32))}"
: "${VIDEO_MIN_PIXELS:=$((32 * 32 * 32))}"
: "${VIDEO_TOTAL_PIXELS:=$((96 * 32 * 32 * 32))}"
: "${MASTER_PORT:=29622}"

# CURRICULUM = SequentialSampler. The trainer reads this env var.
export NO_SHUFFLE_TRAIN=1

TAG="sof_dpo_v2_run2_curriculum_8b_${DPO_LOSS}_b${BETA}_lr${LR}_ep${EPOCHS}_bs${GLOBAL_BATCH}_${VIDEO_MAX_FRAMES}fr_${TS}"
export OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$TAG}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  v2 DPO Run-2 (CURRICULUM, easy → hard, NO_SHUFFLE_TRAIN=1)
  TRAIN  : $TRAIN_DATA
  EVAL   : $EVAL_DATA
  SFT_CKPT: $SFT_CKPT
  OUT    : $OUTPUT_DIR
EOF

export TRAIN_DATA EVAL_DATA SFT_CKPT GLOBAL_BATCH PER_DEV_BS EPOCHS LR BETA \
       DPO_LOSS LORA_R LORA_ALPHA FPS MAX_SEQ_LENGTH VIDEO_MAX_FRAMES \
       VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS VIDEO_TOTAL_PIXELS MASTER_PORT

exec bash "$REPO/scripts/20_sof_dpo.sh"
