#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot setup for LVBench × VLMEvalKit (NO evaluation)
#
# This script only installs dependencies, builds the annotation TSV, and
# runs diagnostics.  Evaluation is handled by the per-model run scripts:
#
#   bash run_qwen25vl.sh   # Qwen2.5-VL-7B on GPU 0
#   bash run_qwen3vl.sh    # Qwen3-VL-7B  on GPU 1
#
# You can launch both in parallel in separate terminals.
# =============================================================================
set -euo pipefail

echo "════════════════════════════════════════════════════════════"
echo "  LVBench × VLMEvalKit  —  Setup"
echo "════════════════════════════════════════════════════════════"

# ── Step 1: Install ────────────────────────────────────────────────────────────
echo ""
echo "[Step 1/3] Installing VLMEvalKit and patching LVBench ..."
bash 01_install.sh

# ── Step 2: Build TSV ──────────────────────────────────────────────────────────
echo ""
echo "[Step 2/3] Building LVBench annotation TSV ..."
export LMUData="$(pwd)/LMUData"
python 02_prepare_lvbench_tsv.py \
    --meta_jsonl /workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl \
    --video_root /data/Pupil/lvbench_v2 \
    --lmu_root   "${LMUData}"

# ── Step 3: Diagnose ───────────────────────────────────────────────────────────
echo ""
echo "[Step 3/3] Running pre-flight diagnostics ..."
python 04_diagnose.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete.  To run evaluation:"
echo ""
echo "    bash run_qwen25vl.sh   # GPU 0"
echo "    bash run_qwen3vl.sh    # GPU 1  (in a second terminal)"
echo "════════════════════════════════════════════════════════════"
