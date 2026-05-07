#!/bin/bash
# ============================================================================
# DPO training — Qwen3-VL on Pupil preference pairs
# ============================================================================
# Mirrors train_vanilla_sft_fps.sh: same FPS-based video sampling, same
# dynamic max_seq_length → video_max_pixels math, same LoRA defaults.
#
# Defaults assume 4×B200 (192GB).  Set NUM_GPUS=8 and re-run on the 8-GPU node;
# the script will recompute GRAD_ACCUM so the global batch size is preserved.
#
# Quick smoke test (≈ 5 min on 4 GPUs):
#   MAX_STEPS=10 PER_DEV_BS=1 GLOBAL_BATCH=4 SMOKE=1 \
#   bash scripts/train_dpo.sh
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
# Reduce CUDA allocator fragmentation — critical for DPO where peak memory
# spikes on long-video outliers (policy + ref forward both hold activations).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"

CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
DATA_DIR="${DATA_DIR:-$CE_DIR/dpo_data}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/dpo_train.json}"
EVAL_DATA="${EVAL_DATA:-$DATA_DIR/dpo_train.val.json}"
VIDEO_DIR="${VIDEO_DIR:-}"

# ════════════════════════════════════════════════════════════════════
# Optimisation knobs
# ════════════════════════════════════════════════════════════════════
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"     # smaller than SFT — DPO is 2× memory
PER_DEV_BS="${PER_DEV_BS:-1}"
GRAD_ACCUM=$((GLOBAL_BATCH / (PER_DEV_BS * NUM_GPUS)))
[[ "$GRAD_ACCUM" -lt 1 ]] && GRAD_ACCUM=1
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"                        # DPO standard ≈ 5e-7 .. 1e-5
BETA="${BETA:-0.1}"
DPO_LOSS="${DPO_LOSS:-sigmoid}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_STEPS="${MAX_STEPS:--1}"
RUN_TEST="${RUN_TEST:-true}"
SMOKE="${SMOKE:-0}"

# ════════════════════════════════════════════════════════════════════
# Frame sampling + sequence-length budget (identical math to SFT)
# ════════════════════════════════════════════════════════════════════
FPS="${FPS:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-65536}"
# Upstream-recommended pixel formula (same as SFT):
#   video_max_pixels = (max_seq_length / 1000) * 32 * 32
# DPO does 2× forward (policy + ref), so we use FPS=1 (not 2) to keep
# memory in check.  Qwen team uses FPS=2 for their reported benchmarks
# (see QwenLM/Qwen3-VL#1540), but our SFT baselines already train at
# FPS=1, so matching that is the right apples-to-apples comparison.
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((MAX_SEQ_LENGTH / 1000 * 32 * 32))}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((VIDEO_MAX_PIXELS / 4))}"

RUN_TAG="dpo_${DPO_LOSS}_beta${BETA}_lr${LR}_ep${EPOCHS}_${MAX_SEQ_LENGTH}seq_fps${FPS}"
[[ "$MAX_STEPS" != "-1" ]] && RUN_TAG="${RUN_TAG}_${MAX_STEPS}steps"
[[ "$SMOKE" == "1" ]] && RUN_TAG="${RUN_TAG}_smoke"
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  DPO TRAIN — QwenDPOTrainer
  GPUs:            $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)
  Global batch:    $GLOBAL_BATCH  (per-dev=$PER_DEV_BS  grad-accum=$GRAD_ACCUM)
  LR / beta:       $LR / $BETA      Loss: $DPO_LOSS
  Epochs:          $EPOCHS
  FPS:             $FPS
  max_seq_length:  $MAX_SEQ_LENGTH
  video pixels:    [$VIDEO_MIN_PIXELS, $VIDEO_MAX_PIXELS]
  Model:           $MODEL_ID
  Train data:      $TRAIN_DATA
  Eval  data:      $EVAL_DATA
  Output:          $OUTPUT_DIR
EOF
[[ "$MAX_STEPS" != "-1" ]] && echo "  Max steps:       $MAX_STEPS  (debug)"
echo "════════════════════════════════════════════════════════════════════"

VIDEO_ARG=""
[[ -n "$VIDEO_DIR" ]] && VIDEO_ARG="--image_folder $VIDEO_DIR"

EVAL_ARG=""
if [[ -f "$EVAL_DATA" ]]; then
    EVAL_ARG="--eval_path $EVAL_DATA --eval_strategy steps --eval_steps 100 --per_device_eval_batch_size 1"
else
    EVAL_ARG="--eval_strategy no"
fi

# DPO needs fp32 gradient accumulation under bf16 — the default zero2.json
# (used by SFT) accumulates grads in bf16, which silently overflows during
# backward at long video sequences and produces NaN LoRA updates after step 0
# (loss freezes at log(2)=0.6931, grad_norm at sqrt(3), accuracy at 0).
# zero2_dpo.json is identical to zero2.json plus:
#   "data_types": {"grad_accum_dtype": "fp32"}
#   "communication_data_type": "fp32"
# This costs ~0.5 GB extra per rank for the fp32 grad buffer (negligible on B200)
# but is REQUIRED for DPO at MAX_SEQ_LENGTH > ~16k or video_max_pixels > ~16k.
DEEPSPEED_CFG="${DEEPSPEED_CFG:-scripts/zero2_dpo.json}"

deepspeed --num_gpus "$NUM_GPUS" --master_port "$MASTER_PORT" \
    src/train/train_dpo.py \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    $EVAL_ARG \
    $VIDEO_ARG \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --fps "$FPS" \
    --video_max_pixels "$VIDEO_MAX_PIXELS" \
    --video_min_pixels "$VIDEO_MIN_PIXELS" \
    --remove_unused_columns False \
    --bf16 True --fp16 False \
    --beta "$BETA" \
    --dpo_loss "$DPO_LOSS" \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --vision_lr 1e-6 \
    --merger_lr 5e-6 \
    --weight_decay 0.0 \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --deepspeed "$DEEPSPEED_CFG" \
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
echo "  DPO TRAIN COMPLETE — $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"

if [[ "$RUN_TEST" == "true" ]]; then
    echo ""
    echo "  Cooling down 60s before test eval ..."
    sleep 60
    EVAL_ADAPTER_DIR="${EVAL_ADAPTER_DIR:-$OUTPUT_DIR}"
    ADAPTER_DIR="$EVAL_ADAPTER_DIR" \
    MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
    FPS="$FPS" \
    OUTPUT_DIR="$EVAL_ADAPTER_DIR/test_results_full_video" \
        bash "$REPO_ROOT/scripts/test_full_video_matched.sh" 2>&1 | tee "$EVAL_ADAPTER_DIR/test_after_train.log"
fi
