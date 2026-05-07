#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_qwen3vl.sh — Evaluate Qwen3-VL-8B on LVBench v2 via VLMEvalKit
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Qwen3-VL-8B-Instruct evaluation on LVBench v2..."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python run_eval.py \
    --model Qwen3-VL-8B-Instruct \
    --fps 1.0 \
    --work-dir "$SCRIPT_DIR/outputs" \
    --verbose
