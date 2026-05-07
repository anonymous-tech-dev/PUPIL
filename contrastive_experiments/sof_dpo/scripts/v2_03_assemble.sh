#!/bin/bash
# v2_03_assemble.sh — Build the final v2 DPO + SFT files (no transcript).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO/old_dpo_revised_data_8b"
IN_GLOB="${IN_GLOB:-$DATA_DIR/pairs_with_margin.shard*.jsonl}"
OUT_TRAIN="${OUT_TRAIN:-$DATA_DIR/sof_dpo_train.json}"
OUT_VAL="${OUT_VAL:-$DATA_DIR/sof_dpo_train.val.json}"
VAL_FRAC="${VAL_FRAC:-0.05}"
MAX_MARGIN_PER_TOK="${MAX_MARGIN_PER_TOK:-0.5}"
CAP_PER_AXIS="${CAP_PER_AXIS:-}"

CMD=(python3 "$REPO/build_pairs/sof_dpo_assemble_dpo_jsonl_v2.py"
     --in-glob "$IN_GLOB"
     --out-train "$OUT_TRAIN"
     --out-val   "$OUT_VAL"
     --val-frac  "$VAL_FRAC"
     --max-margin-per-tok "$MAX_MARGIN_PER_TOK")
[[ -n "$CAP_PER_AXIS" ]] && CMD+=(--cap-per-axis "$CAP_PER_AXIS")
"${CMD[@]}"

# Companion SFT-warmstart (chosen-only, no-transcript).
SFT_TRAIN="${SFT_TRAIN:-$DATA_DIR/sof_sft_warmstart.no_transcript.json}"
SFT_VAL="${SFT_VAL:-$DATA_DIR/sof_sft_warmstart.no_transcript.val.json}"
python3 "$REPO/build_pairs/sof_dpo_make_sft_warmstart.py" \
    --in-dpo-json "$OUT_TRAIN" \
    --out-train "$SFT_TRAIN" \
    --out-val   "$SFT_VAL" \
    --val-frac  "$VAL_FRAC"

echo "[v2-03] assembled (pre-judge)."
