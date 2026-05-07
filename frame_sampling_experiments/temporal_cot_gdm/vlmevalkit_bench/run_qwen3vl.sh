#!/usr/bin/env bash
# =============================================================================
# run_qwen3vl.sh — Evaluate Qwen3-VL-8B-Instruct on LVBench  (GPU 1)
#
# Usage:
#   bash run_qwen3vl.sh          # default: GPU 1
#   GPU=3 bash run_qwen3vl.sh    # override GPU
# =============================================================================
set -euo pipefail

# ── GPU assignment ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
VLMEVAL_ROOT="${WORK_DIR}/VLMEvalKit"
CONFIG_FILE="${WORK_DIR}/config_qwen3vl.json"
OUTPUT_DIR="${WORK_DIR}/outputs/qwen3vl"

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
echo "  Qwen3-VL-8B-Instruct  ×  LVBench"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Output: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$OUTPUT_DIR"
cd "$VLMEVAL_ROOT"

python run.py \
    --config  "${CONFIG_FILE}" \
    --work-dir "${OUTPUT_DIR}" \
    --mode all \
    2>&1 | tee "${OUTPUT_DIR}/eval_qwen3vl.log"

echo ""
echo "✅  Qwen3-VL-8B-Instruct done.  Results: ${OUTPUT_DIR}"
