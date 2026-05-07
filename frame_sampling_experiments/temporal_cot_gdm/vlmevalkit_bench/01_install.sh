#!/usr/bin/env bash
# =============================================================================
# 01_install.sh  —  Install VLMEvalKit + dependencies for LVBench eval
#
# References:
#   VLMEvalKit installation:
#     https://github.com/open-compass/VLMEvalKit#installation
#   VLMEvalKit custom benchmark guide:
#     https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md
# =============================================================================
set -euo pipefail

WORK_DIR="$(pwd)"

# ── 1. Clone VLMEvalKit if not already present ───────────────────────────────
# Use editable install so we can patch in the LVBench dataset class without
# re-installing.  (Community practice: all custom datasets are added directly
# to the cloned repo and registered in vlmeval/dataset/__init__.py)
if [ ! -d "VLMEvalKit" ]; then
    git clone https://github.com/open-compass/VLMEvalKit.git
fi

cd VLMEvalKit
# Use base install only — the B200 image already has torch, transformers,
# flash-attn, accelerate, qwen_vl_utils, etc.  ".[all]" pulls in every model
# backend (InternVL, LLaVA, …) which we don't need and risks version conflicts.
pip install -e . --quiet

# ── 2. Install LVBench annotation loader ─────────────────────────────────────
pip install --quiet datasets huggingface_hub decord av rouge

# ── 2b. Create empty .env so VLMEvalKit doesn't log a noisy ERROR ─────────────
touch .env

# ── 3. Patch LVBench dataset class into VLMEvalKit ───────────────────────────
# Copy our custom dataset implementation into VLMEvalKit's dataset directory.
# This follows the documented approach at:
#   https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md
cp "${WORK_DIR}/lvbench_dataset.py" vlmeval/dataset/lvbench.py

# Register the class in vlmeval/dataset/__init__.py
# We append only if the import isn't already there (idempotent).
INIT_FILE="vlmeval/dataset/__init__.py"
if ! grep -q "from .lvbench import LVBench" "$INIT_FILE"; then
    echo "" >> "$INIT_FILE"
    echo "# LVBench — custom long-video MCQ benchmark" >> "$INIT_FILE"
    echo "from .lvbench import LVBench" >> "$INIT_FILE"
    echo "Registered LVBench in $INIT_FILE"
else
    echo "LVBench already registered in $INIT_FILE"
fi

echo ""
echo "✅  VLMEvalKit installed and patched with LVBench dataset class."
echo "    Next: run  02_prepare_tsv.sh  to build the annotation TSV."
