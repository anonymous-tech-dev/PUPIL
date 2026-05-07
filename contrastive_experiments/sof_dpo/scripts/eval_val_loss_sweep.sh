#!/bin/bash
# eval_val_loss_sweep.sh — compute trainer.evaluate() loss on the held-out
# val set for each saved checkpoint of the NOTX SFT run.  Mirrors the launch
# config of 10_sof_sft_warmstart.sh so the loss is comparable to the train
# curve in trainer_state.json.
#
# Output: <ckpt>/eval_metrics.json  (eval_loss, runtime, samples_per_sec)
set -uo pipefail

# ── Defensive: unset any video/gen env vars leaking from prior shell runs.
# The benchmark eval scripts export these and they'd otherwise override the
# defaults below (e.g. VIDEO_MAX_FRAMES=768 leaked from a base-settings eval).
unset VIDEO_FPS VIDEO_MAX_FRAMES VIDEO_MIN_FRAMES \
      VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS VIDEO_TOTAL_PIXELS \
      GEN_MAX_NEW_TOKENS GEN_DO_SAMPLE GEN_TEMPERATURE GEN_TOP_P GEN_TOP_K \
      ADAPTER_DIR ADAPTER_TAG

REPO=/workspace/Pupil/contrastive_experiments/sof_dpo
QWEN_REPO=/workspace/Pupil/contrastive_experiments/Qwen-VL-Series-Finetune
CE_DIR=/workspace/Pupil/contrastive_experiments

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_DISABLED=true
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4}"
export TCMALLOC_RELEASE_RATE="${TCMALLOC_RELEASE_RATE:-10}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-67108864}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-67108864}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29512}"
export PYTHONPATH="$REPO/scripts:$QWEN_REPO:${PYTHONPATH:-}"

# Same decord guard preflight as training.
python -c "import sys; sys.path.insert(0, '$REPO/scripts'); import decord_only_guard" \
    || { echo "❌ decord_only_guard not importable, aborting"; exit 1; }

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
EVAL_DATA="${EVAL_DATA:-$REPO/data/sof_sft_warmstart.val.no_transcript.json}"
TRAIN_DATA="${TRAIN_DATA:-$EVAL_DATA}"

# Training-time visual distribution (32 fr, fps=1, max_seq=16384)
MAX_SEQ_LENGTH=16384
FPS=1
VIDEO_MAX_FRAMES=32
VIDEO_MAX_PIXELS=524288
VIDEO_MIN_PIXELS=131072
VIDEO_TOTAL_PIXELS=$((VIDEO_MAX_FRAMES * VIDEO_MAX_PIXELS))

LORA_R=128
LORA_ALPHA=128

SFT_OUT="${SFT_OUT:-$CE_DIR/sof_dpo/outputs/sof_sft_warmstart_NOTX_lr2e-5_ep3_bs64_16384seq_32fr}"
CKPTS="${CKPTS:-checkpoint-20 checkpoint-40 checkpoint-60 checkpoint-80 checkpoint-100 .}"

cd "$QWEN_REPO"

for ck in $CKPTS; do
  if [ "$ck" = "." ]; then
    ADAPTER_DIR_VAL="$SFT_OUT"
    TAG="final_step102"
  else
    ADAPTER_DIR_VAL="$SFT_OUT/$ck"
    TAG="$ck"
  fi

  if [ ! -f "$ADAPTER_DIR_VAL/adapter_model.safetensors" ]; then
    echo "⚠️  $ADAPTER_DIR_VAL missing adapter_model.safetensors, skipping"
    continue
  fi

  OUT_SCRATCH="${SFT_OUT}/_eval_only/${TAG}"
  mkdir -p "$OUT_SCRATCH"

  echo "════════════════════════════════════════════════════════════════════"
  echo "  EVAL_ONLY  ckpt=$TAG"
  echo "  ADAPTER:  $ADAPTER_DIR_VAL"
  echo "  EVAL:     $EVAL_DATA"
  echo "  WRITE:    $OUT_SCRATCH/eval_metrics.json"
  echo "════════════════════════════════════════════════════════════════════"

  export ADAPTER_DIR="$ADAPTER_DIR_VAL"

  # torchrun (native DDP) instead of deepspeed — ZeRO-1 refuses inference.
  torchrun --nproc_per_node "$NUM_GPUS" --master_port "$MASTER_PORT" \
      src/train/eval_only.py \
      --use_liger_kernel False \
      --model_id "$MODEL_ID" \
      --data_path "$TRAIN_DATA" \
      --eval_path "$EVAL_DATA" \
      --output_dir "$OUT_SCRATCH" \
      --max_seq_length "$MAX_SEQ_LENGTH" \
      --fps "$FPS" \
      --video_max_pixels "$VIDEO_MAX_PIXELS" \
      --video_min_pixels "$VIDEO_MIN_PIXELS" \
      --video_max_frames "$VIDEO_MAX_FRAMES" \
      --video_total_pixels "$VIDEO_TOTAL_PIXELS" \
      --remove_unused_columns False \
      --bf16 True --fp16 False \
      --disable_flash_attn2 False \
      --num_train_epochs 1 \
      --max_steps 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 1 \
      --eval_strategy "no" \
      --save_strategy "no" \
      --learning_rate 1e-5 \
      --vision_lr 2e-6 --merger_lr 2e-5 \
      --weight_decay 0.01 \
      --warmup_ratio 0.0 \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --gradient_checkpointing False \
      --dataloader_num_workers 2 \
      --dataloader_prefetch_factor 1 \
      --report_to none \
      --lora_enable True \
      --lora_rank "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout 0.05 \
      --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
      --freeze_llm True --freeze_vision_tower True --freeze_merger False \
      2>&1 | tee "$OUT_SCRATCH/eval.log"
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  ALL EVALS DONE — summary"
echo "════════════════════════════════════════════════════════════════════"
for ck in $CKPTS; do
  TAG=$( [ "$ck" = "." ] && echo "final_step102" || echo "$ck" )
  M="${SFT_OUT}/_eval_only/${TAG}/eval_metrics.json"
  if [ -f "$M" ]; then
    LOSS=$(python3 -c "import json; print(json.load(open('$M')).get('eval_loss','?'))")
    printf "  %-22s eval_loss=%s\n" "$TAG" "$LOSS"
  else
    printf "  %-22s (no metrics file)\n" "$TAG"
  fi
done
