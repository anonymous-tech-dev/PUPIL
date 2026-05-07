#!/usr/bin/env bash
# =============================================================================
# run_qwen25vl.sh — Evaluate Qwen2.5-VL-7B-Instruct on LVBench  (GPU 0)
#
# Usage:
#   bash run_qwen25vl.sh          # default: GPU 0
#   GPU=2 bash run_qwen25vl.sh    # override GPU
# =============================================================================
set -euo pipefail

# ── GPU assignment ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
VLMEVAL_ROOT="${WORK_DIR}/VLMEvalKit"
CONFIG_FILE="${WORK_DIR}/config_qwen25vl.json"
OUTPUT_DIR="${WORK_DIR}/outputs/qwen25vl"

export LMUData="${WORK_DIR}/LMUData"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [ ! -d "$VLMEVAL_ROOT" ]; then
    echo "❌  VLMEvalKit not found. Run:  bash run_all.sh  (setup) first."
    exit 1
fi
if [ ! -f "${LMUData}/LVBench/LVBench.tsv" ]; then
    echo "❌  LVBench TSV not found. Run:  bash run_all.sh  (setup) first."
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  Qwen2.5-VL-7B-Instruct  ×  LVBench"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Output: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"
cd "$VLMEVAL_ROOT"

python run.py \
    --config  "${CONFIG_FILE}" \
    --work-dir "${OUTPUT_DIR}" \
    --mode all \
    2>&1 | tee "${OUTPUT_DIR}/eval_qwen25vl.log"

echo ""
echo "✅  Qwen2.5-VL-7B done.  Results: ${OUTPUT_DIR}"
