#!/bin/bash
# v2_98_post_sft_pipeline.sh — Background watcher.
#
# Polls the in-flight v2 SFT-curriculum output dir for its final adapter
# file.  Once the file appears AND no train_sft.py / deepspeed worker is
# still alive, it:
#   1. filters the 14 known-broken training videos out of all SFT+DPO json files
#      (writes *.vidclean.json next to each)
#   2. evals the SFT (rank slot for the leaderboard)
#   3. launches v2 DPO Run-1 (random shuffle) on top of the SFT
#   4. evals Run-1
#   5. launches v2 DPO Run-2 (curriculum) on top of the SFT
#   6. evals Run-2
#
# All steps log under $SFT_OUT/postsft_pipeline.log.
#
# Usage:
#   SFT_OUT=/abs/path/to/sft_out  nohup bash scripts/v2_98_post_sft_pipeline.sh \
#       > /tmp/v2_98.log 2>&1 &
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/old_dpo_revised_data_8b"

[[ -z "${SFT_OUT:-}" ]] && { echo "❌ SFT_OUT is required"; exit 2; }
[[ -d "$SFT_OUT" ]] || { echo "❌ not a dir: $SFT_OUT"; exit 2; }

LOG="$SFT_OUT/postsft_pipeline.log"
exec >>"$LOG" 2>&1
echo "════════════════════════════════════════════════════════════════════"
echo "[$(date)] watcher start; SFT_OUT=$SFT_OUT"

# ── 1. Wait for SFT to finish ────────────────────────────────────────────
# 'Finished' = (adapter_model.safetensors exists in last checkpoint) AND
#              (no train_sft.py or deepspeed launcher process running).
while :; do
    LAST_CKPT=$(ls -1d "$SFT_OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    HAVE_FILE=0
    [[ -n "$LAST_CKPT" && -f "$LAST_CKPT/adapter_model.safetensors" ]] && HAVE_FILE=1
    PROCS=$(pgrep -af "train_sft.py|deepspeed.launcher.launch.*train_sft" | wc -l)
    if [[ "$HAVE_FILE" -eq 1 && "$PROCS" -eq 0 ]]; then
        echo "[$(date)] SFT done. last ckpt = $LAST_CKPT"
        break
    fi
    sleep 60
done
SFT_CKPT="$LAST_CKPT"
echo "$SFT_CKPT" > /tmp/v2_sft_curriculum_ckpt.txt

# ── 2. Filter bad videos out of all SFT+DPO json files ─────────────────
echo "[$(date)] filtering bad videos from data files…"
python3 "$REPO/build_pairs/filter_bad_vids.py" \
    "$DATA/sof_sft_warmstart.no_transcript.judged.curriculum.json" \
    "$DATA/sof_sft_warmstart.no_transcript.judged.json" \
    "$DATA/sof_sft_warmstart.no_transcript.val.judged.json" \
    "$DATA/sof_dpo_train.judged.curriculum.json" \
    "$DATA/sof_dpo_train.judged.json" \
    "$DATA/sof_dpo_train.val.judged.json" \
    "$DATA/sof_dpo_train.judged.mix80.json" \
    "$DATA/sof_dpo_train.val.judged.mix80.json" \
    "$DATA/sof_dpo_train.judged.curriculum.mix80.json"
echo "[$(date)] filter done."

# ── 3. Eval the SFT adapter for the leaderboard ─────────────────────────
echo "[$(date)] launching SFT eval…"
ADAPTER_TAG=v2_sft_curriculum_8b_clip ADAPTER_DIR="$SFT_CKPT" \
    bash "$REPO/scripts/v2_40_eval.sh" || echo "  ⚠ SFT eval errored, continuing"

# ── 4-5. Run-1 random + eval ────────────────────────────────────────────
TS1=$(date +%Y%m%d_%H%M%S)
RUN1_OUT="$REPO/../../outputs/sof_dpo_v2_run1_random_8b_onCurrSFT_${TS1}"
echo "[$(date)] launching DPO Run-1 (random) on $SFT_CKPT  ->  $RUN1_OUT"
SFT_CKPT="$SFT_CKPT" OUTPUT_DIR="$RUN1_OUT" \
    TRAIN_DATA="$DATA/sof_dpo_train.judged.vidclean.json" \
    EVAL_DATA="$DATA/sof_dpo_train.val.judged.vidclean.json" \
    bash "$REPO/scripts/v2_30_dpo_run1_random.sh" || echo "  ⚠ Run-1 train errored"

ADAPTER_TAG=v2_dpo_run1_random_onCurrSFT \
    ADAPTER_DIR="$(ls -1d "$RUN1_OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "$RUN1_OUT")" \
    bash "$REPO/scripts/v2_40_eval.sh" || echo "  ⚠ Run-1 eval errored"

# ── 6-7. Run-2 curriculum + eval ────────────────────────────────────────
TS2=$(date +%Y%m%d_%H%M%S)
RUN2_OUT="$REPO/../../outputs/sof_dpo_v2_run2_curriculum_8b_onCurrSFT_${TS2}"
echo "[$(date)] launching DPO Run-2 (curriculum) on $SFT_CKPT  ->  $RUN2_OUT"
SFT_CKPT="$SFT_CKPT" OUTPUT_DIR="$RUN2_OUT" \
    TRAIN_DATA="$DATA/sof_dpo_train.judged.curriculum.vidclean.json" \
    EVAL_DATA="$DATA/sof_dpo_train.val.judged.vidclean.json" \
    bash "$REPO/scripts/v2_31_dpo_run2_curriculum.sh" || echo "  ⚠ Run-2 train errored"

ADAPTER_TAG=v2_dpo_run2_curriculum_onCurrSFT \
    ADAPTER_DIR="$(ls -1d "$RUN2_OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "$RUN2_OUT")" \
    bash "$REPO/scripts/v2_40_eval.sh" || echo "  ⚠ Run-2 eval errored"

echo "[$(date)] ✅ pipeline complete."
