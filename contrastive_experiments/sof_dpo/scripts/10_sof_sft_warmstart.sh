#!/bin/bash
# 10_sof_sft_warmstart.sh — Stage-1: short SFT warm-start on EduBench-Train
# chosen responses, used as the initialisation for SoF-DPO.
#
# Mirrors train_vanilla_sft_fps.sh but points at the SoF-curated SFT data.
# 8xB200 defaults; override via env vars.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_DISABLED=true
# CPU-RAM hygiene: tcmalloc returns memory to OS more aggressively than glibc;
# the glibc knobs make malloc_trim trigger when arenas exceed 64 MB.
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4}"
export TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-67108864}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-67108864}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"
export PYTHONPATH="$REPO/scripts:$QWEN_REPO:${PYTHONPATH:-}"

# ---- Hard-verify the decord guard is importable BEFORE deepspeed launches ----
# Without this, torchvision silently falls back, full-decodes multi-minute
# videos into CPU RAM, and OOM-kills the pod (lost 24fr SFT run this way).
python -c "import sys; sys.path.insert(0, '$REPO/scripts'); import decord_only_guard" \
    || { echo "❌ decord_only_guard NOT importable — aborting before pod-killer fires"; exit 1; }
echo "✅ decord_only_guard preflight OK"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$REPO/data/sof_sft_warmstart.json}"
EVAL_DATA="${EVAL_DATA:-$REPO/data/sof_sft_warmstart.val.json}"

GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
PER_DEV_BS="${PER_DEV_BS:-1}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
[[ "$GRAD_ACCUM" -lt 1 ]] && GRAD_ACCUM=1
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
FPS="${FPS:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((MAX_SEQ_LENGTH / 1000 * 32 * 32))}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((VIDEO_MAX_PIXELS / 4))}"
# Hard cap on frames per video — mirrored from the DPO stage so SFT and DPO
# train at the SAME visual distribution and eval can match it.
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-$((VIDEO_MAX_FRAMES * VIDEO_MAX_PIXELS))}"
MAX_STEPS="${MAX_STEPS:--1}"

RUN_TAG="sof_sft_warmstart_lr${LR}_ep${EPOCHS}_bs${GLOBAL_BATCH}_${MAX_SEQ_LENGTH}seq_${VIDEO_MAX_FRAMES}fr"
[[ "$MAX_STEPS" != "-1" ]] && RUN_TAG="${RUN_TAG}_${MAX_STEPS}steps"

# Checkpoints AND log land under the workspace.
# (Set LOG_DIR=/some/other/path to mirror logs elsewhere for crash safety.)
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  SoF-SFT WARM-START
  GPUs:     $NUM_GPUS  global=$GLOBAL_BATCH per-dev=$PER_DEV_BS accum=$GRAD_ACCUM
  Model:    $MODEL_ID
  Train:    $TRAIN_DATA
  Eval :    $EVAL_DATA
  Out  :    $OUTPUT_DIR
  Log  :    $LOG_DIR/training.log
EOF
echo "════════════════════════════════════════════════════════════════════"

cd "$QWEN_REPO"
deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/train_sft.py \
    --use_liger_kernel "${USE_LIGER:-True}" \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_path "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --fps "$FPS" \
    --video_max_pixels "$VIDEO_MAX_PIXELS" \
    --video_min_pixels "$VIDEO_MIN_PIXELS" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --video_total_pixels "$VIDEO_TOTAL_PIXELS" \
    --remove_unused_columns False \
    --bf16 True --fp16 False \
    --disable_flash_attn2 False \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_strategy steps --eval_steps "${EVAL_STEPS:-200}" \
    --save_strategy steps --save_steps "${SAVE_STEPS:-200}" --save_total_limit "${SAVE_LIMIT:-3}" \
    --learning_rate "$LR" \
    --vision_lr "${VISION_LR:-2e-6}" --merger_lr "${MERGER_LR:-2e-5}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 5 --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
    --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR:-1}" \
    --report_to tensorboard \
    --deepspeed "${DEEPSPEED_CONFIG:-scripts/zero1.json}" \
    --lora_enable True \
    --lora_rank "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_llm True --freeze_vision_tower True --freeze_merger False \
    2>&1 | tee "$LOG_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  SoF-SFT WARM-START COMPLETE — $OUTPUT_DIR"
