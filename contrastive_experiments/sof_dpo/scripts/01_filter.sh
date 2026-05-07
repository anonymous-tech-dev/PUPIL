#!/bin/bash
# 01_filter.sh — ROUGE + keyword filter on the negatives.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN_GLOB="${IN_GLOB:-$REPO/data/negatives_qwen3vl8b/negatives_*.shard*.jsonl}"
OUT="${OUT:-$REPO/data/pairs_after_filter.jsonl}"
JUDGE_OUT="${JUDGE_OUT:-$REPO/data/judge_prompts.jsonl}"
ROUGE="${ROUGE:-0.55}"

python3 "$REPO/build_pairs/sof_dpo_filter_pairs.py" \
    --in-glob "$IN_GLOB" \
    --out-jsonl "$OUT" \
    --judge-prompts-jsonl "$JUDGE_OUT" \
    --rouge-thresh "$ROUGE"
