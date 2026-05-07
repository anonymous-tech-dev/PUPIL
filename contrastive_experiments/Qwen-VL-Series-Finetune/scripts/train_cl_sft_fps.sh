#!/bin/bash
# ============================================================================
# Contrastive SFT — FPS-based frame sampling (recommended over nframes)
# ============================================================================
# Uses FPS instead of fixed frame count. video_max_pixels is locked to
# max_seq_length via Qwen3-VL token math:
#   video_max_pixels = (max_seq_length / 1000) * 32 * 32
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_DISABLED=true
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Model
# ════════════════════════════════════════════════════════════════════
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Data paths
# ════════════════════════════════════════════════════════════════════
CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/final_sft_data}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/train.json}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/val.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

CGBENCH_TRAIN_VIDS_DIR="${CGBENCH_TRAIN_VIDS_DIR:-/data/Pupil/CGBench/train_vids}"
CGBENCH_ANCHORS_PATH="${CGBENCH_ANCHORS_PATH:-$CE_DIR/cgbench_setup/cgbench.json}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Experiment ID
# ════════════════════════════════════════════════════════════════════
EXPERIMENT_ID="${EXPERIMENT_ID:-V-04}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Contrastive mode & hyperparameters
# ════════════════════════════════════════════════════════════════════
CONTRASTIVE_MODE="${CONTRASTIVE_MODE:-generative}"
CL_LAMBDA="${CL_LAMBDA:-0.4}"
CL_ALPHA="${CL_ALPHA:-1.0}"
CL_TEMPERATURE="${CL_TEMPERATURE:-0.07}"
NUM_TEMPORAL_CLIPS="${NUM_TEMPORAL_CLIPS:-3}"
GRAD_THROUGH_NEGATIVES="${GRAD_THROUGH_NEGATIVES:-True}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Per-source sample counts
# ════════════════════════════════════════════════════════════════════
MAX_SAMPLES_CGBENCH="${MAX_SAMPLES_CGBENCH:--1}"
MAX_SAMPLES_FINEVIDEO="${MAX_SAMPLES_FINEVIDEO:--1}"
MAX_SAMPLES_EDUBENCH="${MAX_SAMPLES_EDUBENCH:--1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:--1}"
USE_REASONING_TRACES="${USE_REASONING_TRACES:-False}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Training hyperparameters
# ════════════════════════════════════════════════════════════════════
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
PER_DEV_BS="${PER_DEV_BS:-1}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
MAX_STEPS="${MAX_STEPS:--1}"

# ════════════════════════════════════════════════════════════════════
# FPS-based sampling (the new default)
# ════════════════════════════════════════════════════════════════════
FPS="${FPS:-1}"

# ════════════════════════════════════════════════════════════════════
# Sequence length & video pixel budget (Qwen3-VL token math)
# ════════════════════════════════════════════════════════════════════
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
VIDEO_MAX_PIXELS=$((MAX_SEQ_LENGTH / 1000 * 32 * 32))
# video_min_pixels must be <= video_max_pixels; default to 1/4 of max
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((VIDEO_MAX_PIXELS / 4))}"

# ════════════════════════════════════════════════════════════════════
# KNOB: LoRA parameters
# ════════════════════════════════════════════════════════════════════
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Freeze settings
# ════════════════════════════════════════════════════════════════════
FREEZE_LLM="${FREEZE_LLM:-True}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-False}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-2e-5}"

# ════════════════════════════════════════════════════════════════════
# KNOB: DeepSpeed & logging
# ════════════════════════════════════════════════════════════════════
DS_CONFIG="${DS_CONFIG:-scripts/zero1_cl.json}"
LOG_CL_METRICS="${LOG_CL_METRICS:-True}"
DEBUG_NEGATIVES="${DEBUG_NEGATIVES:-False}"
RUN_TEST="${RUN_TEST:-true}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# ════════════════════════════════════════════════════════════════════
# Output directory
# ════════════════════════════════════════════════════════════════════
RUN_TAG="${RUN_TAG:-${EXPERIMENT_ID}_${CONTRASTIVE_MODE}_fps${FPS}_lambda${CL_LAMBDA}_alpha${CL_ALPHA}_lr${LR}_ep${EPOCHS}_${MAX_SEQ_LENGTH}seq}"
[[ "$MAX_STEPS" != "-1" ]] && RUN_TAG="${RUN_TAG}_${MAX_STEPS}steps"
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  CONTRASTIVE SFT — FPS-based"
echo "  Experiment:     $EXPERIMENT_ID"
echo "  CL Mode:        $CONTRASTIVE_MODE"
echo "  Lambda (λ):     $CL_LAMBDA  Alpha (α): $CL_ALPHA  Temp (τ): $CL_TEMPERATURE"
echo "  Temporal clips:  $NUM_TEMPORAL_CLIPS"
echo "  Grad thru negs:  $GRAD_THROUGH_NEGATIVES"
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
    src/train/train_cl_sft.py \
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
    --eval_steps "$EVAL_STEPS" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LR" \
    --vision_lr "$VISION_LR" \
    --merger_lr "$MERGER_LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --deepspeed "$DS_CONFIG" \
    --lora_enable True \
    --lora_rank "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --freeze_llm "$FREEZE_LLM" \
    --freeze_vision_tower "$FREEZE_VISION_TOWER" \
    --freeze_merger "$FREEZE_MERGER" \
    --contrastive_weight "$CL_LAMBDA" \
    --contrastive_temperature "$CL_TEMPERATURE" \
    --negative_strategy "$EXPERIMENT_ID" \
    --alpha_grounding_penalty "$CL_ALPHA" \
    --contrastive_mode "$CONTRASTIVE_MODE" \
    --grad_through_negatives "$GRAD_THROUGH_NEGATIVES" \
    --num_temporal_clips "$NUM_TEMPORAL_CLIPS" \
    --cgbench_train_vids_dir "$CGBENCH_TRAIN_VIDS_DIR" \
    --cgbench_anchors_path "$CGBENCH_ANCHORS_PATH" \
    --max_samples_cgbench "$MAX_SAMPLES_CGBENCH" \
    --max_samples_finevideo "$MAX_SAMPLES_FINEVIDEO" \
    --max_samples_edubench "$MAX_SAMPLES_EDUBENCH" \
    --max_val_samples "$MAX_VAL_SAMPLES" \
    --use_reasoning_traces "$USE_REASONING_TRACES" \
    --log_contrastive_metrics "$LOG_CL_METRICS" \
    --debug_negatives "$DEBUG_NEGATIVES" \
    --seed "$SEED" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  CONTRASTIVE SFT (FPS) COMPLETE — $OUTPUT_DIR"
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
