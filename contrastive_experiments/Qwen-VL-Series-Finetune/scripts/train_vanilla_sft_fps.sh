#!/bin/bash
# ============================================================================
# Vanilla SFT — FPS-based frame sampling (recommended over nframes)
# ============================================================================
# Uses FPS instead of fixed frame count. For long videos, video_max_pixels
# is computed from max_seq_length following the Qwen3-VL token math:
#   video_max_pixels = (max_seq_length / 1000) * 32 * 32
# This lets Qwen gracefully reduce spatial resolution for longer videos
# instead of crashing on OOM.
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_DISABLED=true
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.json}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/val.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
PER_DEV_BS="${PER_DEV_BS:-1}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
RUN_TEST="${RUN_TEST:-true}"
MAX_STEPS="${MAX_STEPS:--1}"

# ════════════════════════════════════════════════════════════════════
# FPS-based sampling (the new default)
# ════════════════════════════════════════════════════════════════════
FPS="${FPS:-1}"

# ════════════════════════════════════════════════════════════════════
# Sequence length & video pixel budget
# ════════════════════════════════════════════════════════════════════
# Qwen3-VL token math: each frame produces (pixels / 32 / 32) tokens.
# These are PER-FRAME caps (not totals).  README-recommended Qwen3-VL range:
#   video_min_pixels = 128 * 32 * 32  (~128 vision tokens/frame)
#   video_max_pixels = 768 * 32 * 32  (~768 vision tokens/frame)
# qwen_vl_utils may further downscale per-frame if the total exceeds budget.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((768 * 32 * 32))}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((128 * 32 * 32))}"
# Hard cap on frames decoded per video — the missing safety belt.
# Without this, FPS=1 on a 5-min lecture decodes hundreds of frames into
# CPU RAM per dataloader worker → unbounded host-RAM growth → OOM.
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
# Explicit total-pixel budget so qwen_vl_utils' auto-compute (which scales
# with max_seq_length) doesn't fight us when we shrink seq length.
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-$((VIDEO_MAX_FRAMES * VIDEO_MAX_PIXELS))}"

RUN_TAG="vanilla_sft_fps${FPS}_lr${LR}_ep${EPOCHS}_${MAX_SEQ_LENGTH}seq"
[[ "$MAX_STEPS" != "-1" ]] && RUN_TAG="${RUN_TAG}_${MAX_STEPS}steps"
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  VANILLA SFT — FPS-based (QwenSFTTrainer)"
echo "  GPUs: $NUM_GPUS  BS: ${GLOBAL_BATCH}  LR: ${LR}  Epochs: ${EPOCHS}"
echo "  FPS: $FPS   max_seq_length: $MAX_SEQ_LENGTH"
echo "  video_max_pixels: $VIDEO_MAX_PIXELS  video_min_pixels: $VIDEO_MIN_PIXELS"
echo "  Model: $MODEL_ID"
echo "  Data:  $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
if [[ "$MAX_STEPS" != "-1" ]]; then
    echo "  Max Steps: $MAX_STEPS (debug run)"
fi
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--image_folder $VIDEO_DIR"

deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/train_sft.py \
    --use_liger_kernel True \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_path "$EVAL_DATA" \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --fps "$FPS" \
    --video_max_pixels "$VIDEO_MAX_PIXELS" \
    --video_min_pixels "$VIDEO_MIN_PIXELS" \
    --video_max_frames "$VIDEO_MAX_FRAMES" \
    --video_total_pixels "$VIDEO_TOTAL_PIXELS" \
    --remove_unused_columns False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --vision_lr 2e-6 \
    --merger_lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --deepspeed scripts/zero1.json \
    --lora_enable True \
    --lora_rank "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_llm True \
    --freeze_vision_tower True \
    --freeze_merger False \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  VANILLA SFT (FPS) COMPLETE — $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

# ════════════════════════════════════════════════════════════════════
# Auto-evaluate on test set after training
# ════════════════════════════════════════════════════════════════════
if [[ "$RUN_TEST" == "true" ]]; then
    echo ""
    echo "  Cooling down for 3 minutes before evaluation..."
    sleep 180
    echo "  Starting test evaluation..."
    ADAPTER_DIR="$OUTPUT_DIR" \
    MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
    FPS="$FPS" \
    OUTPUT_DIR="$OUTPUT_DIR/test_results_full_video" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        bash "$REPO_ROOT/scripts/test_full_video_matched.sh" 2>&1 | tee "$OUTPUT_DIR/test_after_train.log"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  AUTO-TEST COMPLETE — $OUTPUT_DIR/test_results_full_video"
    echo "════════════════════════════════════════════════════════════════════"
fi
