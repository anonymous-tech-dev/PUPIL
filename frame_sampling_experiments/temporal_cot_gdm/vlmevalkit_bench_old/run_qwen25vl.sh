#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_qwen25vl.sh — Evaluate Qwen2.5-VL-7B on LVBench v2 via VLMEvalKit
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Qwen2.5-VL-7B-Instruct evaluation on LVBench v2..."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python run_eval.py \
    --model Qwen2.5-VL-7B-Instruct \
    --fps 1.0 \
    --work-dir "$SCRIPT_DIR/outputs" \
    --verbose
