#!/bin/bash
# ============================================================================
# Contrastive SFT Training Script
# ============================================================================
# Trains Qwen3-VL-8B with contrastive regularization:
#   L_total = L_next_token + λ · L_contrastive
#
# All knobs are configurable via environment variables below.
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_DISABLED=true
NUM_GPUS="${NUM_GPUS:-4}"
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

# CGBench full-length videos for temporal negative extraction (V-04, V-05)
CGBENCH_TRAIN_VIDS_DIR="${CGBENCH_TRAIN_VIDS_DIR:-/data/Pupil/CGBench/train_vids}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Experiment ID — controls which negative strategy to use
# ════════════════════════════════════════════════════════════════════
# Options:
#   V-01  Batch negatives only (baseline InfoNCE)
#   V-02  Blackened frames grounding penalty (α=5)
#   V-03  Gaussian noise frames (α=5)
#   V-04  Temporal shift short (±30s from same video)
#   V-05  Temporal shift long (>2min from same video)
#   T-01  Batch answer negatives
#   T-02  Temporal answer mismatch short
#   T-03  Temporal answer mismatch long
#   FULL  Combined: batch + blackened + temporal
#   CUSTOM  All flags via CLI
EXPERIMENT_ID="${EXPERIMENT_ID:-V-04}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Contrastive mode — how S(V, T) is computed
# ════════════════════════════════════════════════════════════════════
# "generative" — log-likelihood method (generation probability)
# "vector"     — EOS hidden state projection + cosine sim (Amazon paper)
CONTRASTIVE_MODE="${CONTRASTIVE_MODE:-generative}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Contrastive hyperparameters
# ════════════════════════════════════════════════════════════════════
# Lambda (λ) — weight of contrastive loss in total loss
CL_LAMBDA="${CL_LAMBDA:-0.4}"

# Alpha (α) — grounding penalty weight for corrupted negatives
CL_ALPHA="${CL_ALPHA:-1.0}"

# Temperature (τ) — InfoNCE temperature
CL_TEMPERATURE="${CL_TEMPERATURE:-0.07}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Per-source sample counts (-1 = use all)
# ════════════════════════════════════════════════════════════════════
MAX_SAMPLES_CGBENCH="${MAX_SAMPLES_CGBENCH:--1}"
MAX_SAMPLES_FINEVIDEO="${MAX_SAMPLES_FINEVIDEO:--1}"
MAX_SAMPLES_EDUBENCH="${MAX_SAMPLES_EDUBENCH:--1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:--1}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Reasoning traces (append CoT to CGBench answers)
# ════════════════════════════════════════════════════════════════════
USE_REASONING_TRACES="${USE_REASONING_TRACES:-False}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Training hyperparameters
# ════════════════════════════════════════════════════════════════════
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
PER_DEV_BS="${PER_DEV_BS:-2}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
NFRAMES="${NFRAMES:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
RUN_TEST="${RUN_TEST:-true}"

# ════════════════════════════════════════════════════════════════════
# KNOB: LoRA parameters
# ════════════════════════════════════════════════════════════════════
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Freeze settings — what stays frozen during training
# ════════════════════════════════════════════════════════════════════
FREEZE_LLM="${FREEZE_LLM:-True}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-True}"
FREEZE_MERGER="${FREEZE_MERGER:-False}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Separate learning rates for vision components
# ════════════════════════════════════════════════════════════════════
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-2e-5}"

# ════════════════════════════════════════════════════════════════════
# KNOB: Video pixel budget overrides
# ════════════════════════════════════════════════════════════════════
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-}"

# ════════════════════════════════════════════════════════════════════
# KNOB: DeepSpeed config
# ════════════════════════════════════════════════════════════════════
DS_CONFIG="${DS_CONFIG:-scripts/zero1_cl.json}"

# ════════════════════════════════════════════════════════════════════
# KNOB: CL logging
# ════════════════════════════════════════════════════════════════════
LOG_CL_METRICS="${LOG_CL_METRICS:-True}"
DEBUG_NEGATIVES="${DEBUG_NEGATIVES:-False}"

# ════════════════════════════════════════════════════════════════════
# Output directory — auto-named from experiment config
# ════════════════════════════════════════════════════════════════════
RUN_TAG="${RUN_TAG:-${EXPERIMENT_ID}_${CONTRASTIVE_MODE}_lambda${CL_LAMBDA}_alpha${CL_ALPHA}_lr${LR}_ep${EPOCHS}_${NFRAMES}frames_bl_false}"
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  CONTRASTIVE SFT TRAINING"
echo "  Experiment:     $EXPERIMENT_ID"
echo "  CL Mode:        $CONTRASTIVE_MODE"
echo "  Lambda (λ):     $CL_LAMBDA"
echo "  Alpha (α):      $CL_ALPHA"
echo "  Temperature (τ): $CL_TEMPERATURE"
echo "  GPUs: $NUM_GPUS  BS: ${GLOBAL_BATCH}  LR: ${LR}  Epochs: ${EPOCHS}"
echo "  Model: $MODEL_ID"
echo "  Data:  $TRAIN_DATA"
echo "  CGBench Full Vids: $CGBENCH_TRAIN_VIDS_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Freeze: LLM=$FREEZE_LLM  VisionTower=$FREEZE_VISION_TOWER  Merger=$FREEZE_MERGER"
echo "════════════════════════════════════════════════════════════════════"

# Build optional args
VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--image_folder $VIDEO_DIR"

EXTRA_VIDEO_ARGS=""
[[ -n "$VIDEO_TOTAL_PIXELS" ]] && EXTRA_VIDEO_ARGS+=" --video_total_pixels $VIDEO_TOTAL_PIXELS"
[[ -n "$VIDEO_MAX_FRAMES" ]]   && EXTRA_VIDEO_ARGS+=" --video_max_frames $VIDEO_MAX_FRAMES"

deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/train_cl_sft.py \
    --use_liger_kernel True \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_path "$EVAL_DATA" \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
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
    --cgbench_train_vids_dir "$CGBENCH_TRAIN_VIDS_DIR" \
    --max_samples_cgbench "$MAX_SAMPLES_CGBENCH" \
    --max_samples_finevideo "$MAX_SAMPLES_FINEVIDEO" \
    --max_samples_edubench "$MAX_SAMPLES_EDUBENCH" \
    --max_val_samples "$MAX_VAL_SAMPLES" \
    --use_reasoning_traces "$USE_REASONING_TRACES" \
    --log_contrastive_metrics "$LOG_CL_METRICS" \
    --debug_negatives "$DEBUG_NEGATIVES" \
    $EXTRA_VIDEO_ARGS \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo "════════════════════════════════════════════════════════════════════"
echo "  CONTRASTIVE SFT COMPLETE — $OUTPUT_DIR"
echo "  Adapters saved (NOT merged model) for disk-space efficiency."
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
