#!/bin/bash
# 11_sof_sft_warmstart_clipmode.sh — convenience wrapper around
# 10_sof_sft_warmstart.sh that picks the right TRAIN/EVAL paths and
# visual-budget defaults for each clip mode.
#
# Usage:
#   MODE=clip   bash scripts/11_sof_sft_warmstart_clipmode.sh    # 2104 clips
#   MODE=mix80  bash scripts/11_sof_sft_warmstart_clipmode.sh    # 1680c+478f
#   MODE=full   bash scripts/11_sof_sft_warmstart_clipmode.sh    # 2158 full
#
# Honors all the same env overrides as 10_sof_sft_warmstart.sh
# (LR, EPOCHS, GLOBAL_BATCH, FPS, VIDEO_MAX_*, etc.) — those override
# the per-mode defaults set here.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_REPO="${QWEN_REPO:-$REPO/../Qwen-VL-Series-Finetune}"
CE_DIR="$(cd "$QWEN_REPO/.." && pwd)"

MODE="${MODE:?MODE env var is required: clip | mix80 | full}"

# ── Per-mode data paths ──────────────────────────────────────────────────────
# These files were produced by build_pairs/make_sft_with_clips.py — see
# the 'Build clip-augmented SFT data variants' step.
case "$MODE" in
    clip)
        : "${TRAIN_DATA:=$REPO/data/sof_sft_warmstart.no_transcript.clip.json}"
        : "${EVAL_DATA:=$REPO/data/sof_sft_warmstart.val.no_transcript.clip.json}"
        # Pure clip: median 40s × 2fps = 80 frames, well under the 256 cap.
        # Bump per-frame pixels — short clips have headroom under the
        # total-pixel envelope (effective px = min(MAX_PIXELS, TOTAL/frames)).
        # 25.2M / 80 = 314k → MAX_PIXELS=524288 unlocks that ceiling for
        # short clips while leaving long clips bounded by TOTAL_PIXELS.
        : "${VIDEO_MAX_PIXELS:=524288}"
        ;;
    mix80)
        : "${TRAIN_DATA:=$REPO/data/sof_sft_warmstart.no_transcript.mix80.json}"
        : "${EVAL_DATA:=$REPO/data/sof_sft_warmstart.val.no_transcript.mix80.json}"
        # Mix: 78% short clips + 22% full lectures. Lower per-frame ceiling
        # so the full-video portion doesn't blow the token budget when
        # qwen-vl-utils auto-balances. (Total cap protects either way, but
        # this keeps the visual distribution closer between the two halves.)
        : "${VIDEO_MAX_PIXELS:=262144}"
        ;;
    full)
        : "${TRAIN_DATA:=$REPO/data/sof_sft_warmstart.no_transcript.full.json}"
        : "${EVAL_DATA:=$REPO/data/sof_sft_warmstart.val.no_transcript.full.json}"
        # Same regime as the failing 256fr full-video run, here as a
        # baseline for the ablation table.
        : "${VIDEO_MAX_PIXELS:=98304}"
        ;;
    *)
        echo "❌ MODE must be clip|mix80|full, got '$MODE'"; exit 2
        ;;
esac

# ── Defaults shared across all three modes ──────────────────────────────────
: "${USE_LIGER:=False}"
: "${LR:=2e-5}"
: "${EPOCHS:=3}"
: "${GLOBAL_BATCH:=64}"
: "${PER_DEV_BS:=1}"
: "${FPS:=2}"
: "${MAX_SEQ_LENGTH:=24576}"
: "${VIDEO_MAX_FRAMES:=256}"
: "${VIDEO_MIN_PIXELS:=32768}"
: "${VIDEO_TOTAL_PIXELS:=25165824}"        # 24 * 1024 * 1024
: "${MAX_GRAD_NORM:=0.5}"
: "${SAVE_STEPS:=20}"
: "${SAVE_LIMIT:=15}"
: "${EVAL_STEPS:=999}"                     # de-facto disabled; eval_only.py used post-hoc
: "${DATALOADER_NUM_WORKERS:=1}"
: "${DATALOADER_PREFETCH_FACTOR:=1}"
: "${MASTER_PORT:=29511}"

# ── Run tag + output dir keyed by MODE ───────────────────────────────────────
TAG="sof_sft_warmstart_NOTX_lr${LR}_ep${EPOCHS}_bs${GLOBAL_BATCH}_${MAX_SEQ_LENGTH}seq_${VIDEO_MAX_FRAMES}fr_${MODE}"
: "${OUTPUT_DIR:=$CE_DIR/outputs/$TAG}"
: "${LOG_DIR:=$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

cat <<EOF
════════════════════════════════════════════════════════════════════
  SoF-SFT WARM-START  (clip-mode wrapper)
  MODE:                $MODE
  Train:               $TRAIN_DATA
  Eval :               $EVAL_DATA
  Output:              $OUTPUT_DIR
  fps / max_seq:       $FPS  /  $MAX_SEQ_LENGTH
  max_frames:          $VIDEO_MAX_FRAMES
  max_pixels/frame:    $VIDEO_MAX_PIXELS
  total_pixels/video:  $VIDEO_TOTAL_PIXELS
  batch / lr / epochs: $GLOBAL_BATCH / $LR / $EPOCHS
EOF
echo "════════════════════════════════════════════════════════════════════"

export TRAIN_DATA EVAL_DATA OUTPUT_DIR LOG_DIR USE_LIGER LR EPOCHS GLOBAL_BATCH \
       PER_DEV_BS FPS MAX_SEQ_LENGTH VIDEO_MAX_FRAMES VIDEO_MAX_PIXELS \
       VIDEO_MIN_PIXELS VIDEO_TOTAL_PIXELS MAX_GRAD_NORM \
       SAVE_STEPS SAVE_LIMIT EVAL_STEPS DATALOADER_NUM_WORKERS \
       DATALOADER_PREFETCH_FACTOR MASTER_PORT

exec bash "$REPO/scripts/10_sof_sft_warmstart.sh"
