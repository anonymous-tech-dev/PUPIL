#!/bin/bash
# 03_assemble.sh — Build the final DPO train/val JSON.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN_GLOB="${IN_GLOB:-$REPO/data/pairs_with_margin.shard*.jsonl}"
OUT_TRAIN="${OUT_TRAIN:-$REPO/data/sof_dpo_train.json}"
OUT_VAL="${OUT_VAL:-$REPO/data/sof_dpo_train.val.json}"
VAL_FRAC="${VAL_FRAC:-0.05}"
MAX_MARGIN_PER_TOK="${MAX_MARGIN_PER_TOK:-0.5}"   # drop saturated pairs
CAP_PER_AXIS="${CAP_PER_AXIS:-}"

CMD=(python3 "$REPO/build_pairs/sof_dpo_assemble_dpo_jsonl.py"
     --in-glob "$IN_GLOB"
     --out-train "$OUT_TRAIN"
     --out-val   "$OUT_VAL"
     --val-frac  "$VAL_FRAC"
     --max-margin-per-tok "$MAX_MARGIN_PER_TOK")
[[ -n "$CAP_PER_AXIS" ]] && CMD+=(--cap-per-axis "$CAP_PER_AXIS")
"${CMD[@]}"

# Also build SFT warm-start data from the same chosen responses
SFT_TRAIN="${SFT_TRAIN:-$REPO/data/sof_sft_warmstart.json}"
SFT_VAL="${SFT_VAL:-$REPO/data/sof_sft_warmstart.val.json}"
python3 "$REPO/build_pairs/sof_dpo_make_sft_warmstart.py" \
    --in-dpo-json "$OUT_TRAIN" \
    --out-train "$SFT_TRAIN" \
    --out-val   "$SFT_VAL" \
    --val-frac  "$VAL_FRAC"

echo "[03] assembled."
