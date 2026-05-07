#!/bin/bash
# 20_sof_dpo.sh — Stage-2: SoF-targeted DPO from the SoF-SFT warm-start ckpt.
#
# Uses our new sof_dpo_train.py + sof_dpo_trainer.py + sof_dpo_dataset.py,
# which are renamed copies of the trusted /old/ DPO stack (the current
# /train/train_dpo.py and /trainer/dpo_trainer.py are known-broken).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# CPU-RAM hygiene (see 10_sof_sft_warmstart.sh).
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4}"
export TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-67108864}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-67108864}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29521}"
export PYTHONPATH="$REPO/scripts:$QWEN_REPO:${PYTHONPATH:-}"

# ---- Hard-verify the decord guard is importable BEFORE deepspeed launches ----
python -c "import sys; sys.path.insert(0, '$REPO/scripts'); import decord_only_guard" \
    || { echo "❌ decord_only_guard NOT importable — aborting before pod-killer fires"; exit 1; }
echo "✅ decord_only_guard preflight OK"

# Default: continue from the SoF-SFT warm-start adapter dir.  The default
# below is the latest mixed-clip warm-start. Override with
#   SFT_CKPT=/path/to/other/sof_sft_warmstart_xxx bash 20_sof_dpo.sh
SFT_CKPT="${SFT_CKPT:-$CE_DIR/outputs/sof_sft_warmstart_NOTX_lr2e-5_ep3_bs64_24576seq_256fr_mix80}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

TRAIN_DATA="${TRAIN_DATA:-$REPO/data/sof_dpo_train.json}"
EVAL_DATA="${EVAL_DATA:-$REPO/data/sof_dpo_train.val.json}"

GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
PER_DEV_BS="${PER_DEV_BS:-1}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
[[ "$GRAD_ACCUM" -lt 1 ]] && GRAD_ACCUM=1
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-7}"
BETA="${BETA:-0.1}"
DPO_LOSS="${DPO_LOSS:-sigmoid}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
# Match the SFT warm-start visual distribution: 2 fps, up to 256 frames per
# video, max ~262k px/frame. qwen-vl-utils auto-shrinks per-frame pixels
# down to honor (max_frames, total_pixels) — short clips get sharp frames,
# long full videos get coarser frames at the same total budget.
FPS="${FPS:-2}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-24576}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((256 * 32 * 32))}"   # 262144
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((32 * 32 * 32))}"    # 32768
# Hard cap on frames per video — matches the SFT warm-start safety belt.
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
# 25.2M total pixels per video — same proven envelope as the 256fr SFT run.
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-$((96 * 32 * 32 * 256))}"
MAX_STEPS="${MAX_STEPS:--1}"

RUN_TAG="sof_dpo_${DPO_LOSS}_beta${BETA}_lr${LR}_ep${EPOCHS}_bs${GLOBAL_BATCH}_${MAX_SEQ_LENGTH}seq_${VIDEO_MAX_FRAMES}fr"
[[ "$MAX_STEPS" != "-1" ]] && RUN_TAG="${RUN_TAG}_${MAX_STEPS}steps"

# Checkpoints AND log land under the workspace.
# (Set LOG_DIR=/some/other/path to mirror logs elsewhere for crash safety.)
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  SoF-DPO  (sof_dpo_train.py / sof_dpo_trainer.py)
  GPUs:        $NUM_GPUS  global=$GLOBAL_BATCH per-dev=$PER_DEV_BS accum=$GRAD_ACCUM
  LR / beta:   $LR / $BETA       loss=$DPO_LOSS
  base model:  $MODEL_ID
  warm-start:  $SFT_CKPT
  Train data:  $TRAIN_DATA
  Eval  data:  $EVAL_DATA
  Output:      $OUTPUT_DIR
  Log:         $LOG_DIR/training.log
EOF
echo "════════════════════════════════════════════════════════════════════"

EVAL_ARG=""
if [[ -f "$EVAL_DATA" ]]; then
    EVAL_ARG="--eval_path $EVAL_DATA --eval_strategy steps --eval_steps 25 --per_device_eval_batch_size 1"
else
    EVAL_ARG="--eval_strategy no"
fi

cd "$QWEN_REPO"
deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/sof_dpo_train.py \
    --model_id "$MODEL_ID" \
    --sft_adapter_path "$SFT_CKPT" \
    --data_path "$TRAIN_DATA" \
    $EVAL_ARG \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --video_max_pixels "$VIDEO_MAX_PIXELS" \
    --video_min_pixels "$VIDEO_MIN_PIXELS" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --video_total_pixels "$VIDEO_TOTAL_PIXELS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --remove_unused_columns False \
    --bf16 True --fp16 False \
    --beta "$BETA" \
    --dpo_loss "$DPO_LOSS" \
    --precompute_ref_log_probs False \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_strategy steps --save_steps 25 --save_total_limit 5 \
    --save_only_model True \
    --learning_rate "$LR" \
    --vision_lr 1e-7 --merger_lr 5e-7 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps "${LOGGING_STEPS:-1}" --max_grad_norm 1.0 \
    --gradient_checkpointing False \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --deepspeed "${DEEPSPEED_CONFIG:-scripts/zero2_dpo.json}" \
    --lora_enable True \
    --lora_rank "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_llm True --freeze_vision_tower True --freeze_merger False \
    2>&1 | tee "$LOG_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  SoF-DPO COMPLETE — $OUTPUT_DIR"
