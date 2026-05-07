#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-time setup for VLMEvalKit + LVBench v2 evaluation
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════════"
echo "  VLMEvalKit — LVBench v2 Setup"
echo "═══════════════════════════════════════════════════════════════════"

# ── 1. Install VLMEvalKit ────────────────────────────────────────────────────
echo ""
echo "[1/3] Installing VLMEvalKit..."
pip install "vlmeval>=0.2" --quiet 2>/dev/null || \
    pip install git+https://github.com/open-compass/VLMEvalKit.git --quiet

# ── 2. Verify dependencies ──────────────────────────────────────────────────
echo "[2/3] Verifying dependencies..."
python -c "
import vlmeval; print(f'  VLMEvalKit {vlmeval.__version__} installed')
from vlmeval.config import supported_VLM
qwen = [k for k in supported_VLM if 'Qwen' in k and ('2.5-VL-7B' in k or '3-VL-8B' in k)]
print(f'  Qwen models available: {qwen}')
"

# ── 3. Prepare the TSV data file ────────────────────────────────────────────
echo "[3/3] Preparing LVBench v2 TSV..."
python "$SCRIPT_DIR/prepare_tsv.py"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Setup complete! Run evaluation with:"
echo ""
echo "    # Qwen2.5-VL-7B"
echo "    python run_eval.py --model Qwen2.5-VL-7B-Instruct --fps 1.0"
echo ""
echo "    # Qwen3-VL-8B"
echo "    python run_eval.py --model Qwen3-VL-8B-Instruct --fps 1.0"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
