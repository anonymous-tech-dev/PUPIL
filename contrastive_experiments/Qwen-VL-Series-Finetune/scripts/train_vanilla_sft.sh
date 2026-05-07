#!/bin/bash
# ============================================================================
# Vanilla SFT — Uses the proven QwenSFTTrainer pipeline (no contrastive loss)
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
PER_DEV_BS="${PER_DEV_BS:-2}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
NFRAMES="${NFRAMES:-32}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
RUN_TEST="${RUN_TEST:-true}"

RUN_TAG="vanilla_sft_fair_frozen_lr${LR}_ep${EPOCHS}_${NFRAMES}frames_65kseq"
# Matches contrastive runs: frozen VLM, lora_alpha=128, max_seq=65536, 32 frames
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  VANILLA SFT (QwenSFTTrainer — proven pipeline)"
echo "  GPUs: $NUM_GPUS  BS: ${GLOBAL_BATCH}  LR: ${LR}  Epochs: ${EPOCHS}"
echo "  Model: $MODEL_ID"
echo "  Data:  $TRAIN_DATA"
echo "  Output: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--image_folder $VIDEO_DIR"

# ----- Video pixel budget (optional overrides) -----
# total_pixels / max_frames are auto-computed from max_seq_length by default.
# Override only if you want finer control.  Qwen recommends:
#   total_pixels < max_seq_length * 32 * 32  (for Qwen3-VL)
#   video_max_pixels = 1664*28*28   video_min_pixels = 256*28*28  (for Qwen2.5-VL)
# With 4xB200 192 GB each, max_seq_length=32768 is comfortable.
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-}"
EXTRA_VIDEO_ARGS=""
[[ -n "$VIDEO_TOTAL_PIXELS" ]] && EXTRA_VIDEO_ARGS+=" --video_total_pixels $VIDEO_TOTAL_PIXELS"
[[ -n "$VIDEO_MAX_FRAMES" ]]   && EXTRA_VIDEO_ARGS+=" --video_max_frames $VIDEO_MAX_FRAMES"

deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/train_sft.py \
    --use_liger_kernel True \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_path "$EVAL_DATA" \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 65536 \
    --nframes "$NFRAMES" \
    --remove_unused_columns False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 400 \
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
    $EXTRA_VIDEO_ARGS \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  VANILLA SFT COMPLETE — $OUTPUT_DIR"
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
    OUTPUT_DIR="$OUTPUT_DIR/test_results_full_video" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        bash "$REPO_ROOT/scripts/test_full_video_matched.sh" 2>&1 | tee "$OUTPUT_DIR/test_after_train.log"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  AUTO-TEST COMPLETE — $OUTPUT_DIR/test_results_full_video"
    echo "════════════════════════════════════════════════════════════════════"
fi
