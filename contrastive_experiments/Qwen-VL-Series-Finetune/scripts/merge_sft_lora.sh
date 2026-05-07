#!/bin/bash
# ============================================================================
# Merge an SFT LoRA adapter into the base Qwen3-VL weights → produces a
# self-contained HF model directory that can be used as MODEL_ID for DPO.
#
# Default target = T-04 grad-fix α=5 ckpt-200 (the 65.6% SOTA SFT run).
#
# Naming convention for merged outputs:
#     outputs/merged/<short-alias>-merged/
# Where <short-alias> is human-meaningful and short, e.g. sft-T04-a5-ck200.
#
# Override any of: BASE_MODEL, ADAPTER_PATH, OUTPUT_DIR, DTYPE.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CE_DIR="$(cd "$REPO_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-$CE_DIR/outputs/T-04_generative_grad_fix_fps1_lambda1.0_alpha5.0_lr2e-5_ep1_65536seq/checkpoint-200}"
ALIAS="${ALIAS:-sft-T04-a5-ck200}"
OUTPUT_DIR="${OUTPUT_DIR:-$CE_DIR/outputs/merged/${ALIAS}-merged}"
DTYPE="${DTYPE:-bf16}"

cat <<EOF
════════════════════════════════════════════════════════════════════
  SFT-LoRA → MERGED BASE
  Base model  : $BASE_MODEL
  Adapter     : $ADAPTER_PATH
  Output      : $OUTPUT_DIR
  Dtype       : $DTYPE
════════════════════════════════════════════════════════════════════
EOF

if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "[merge_sft_lora.sh] FATAL: adapter dir not found: $ADAPTER_PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

EXTRA=""
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA="--overwrite"

python "$REPO_ROOT/scripts/merge_sft_lora.py" \
    --base_model    "$BASE_MODEL" \
    --adapter_path  "$ADAPTER_PATH" \
    --output_dir    "$OUTPUT_DIR" \
    --dtype         "$DTYPE" \
    $EXTRA

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  MERGE COMPLETE"
echo "  Use as MODEL_ID:  $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════════"
