#!/bin/bash
# v2_01_filter.sh — ROUGE + keyword + abstention filter on the v2 negatives.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN_GLOB="${IN_GLOB:-$REPO/old_dpo_revised_data_8b/negatives_v2/final_*.shard*.jsonl}"
OUT="${OUT:-$REPO/old_dpo_revised_data_8b/pairs_after_filter.jsonl}"
JUDGE_OUT="${JUDGE_OUT:-$REPO/old_dpo_revised_data_8b/judge_prompts.jsonl}"
ROUGE="${ROUGE:-0.55}"
mkdir -p "$(dirname "$OUT")"

python3 "$REPO/build_pairs/sof_dpo_filter_pairs_v2.py" \
    --in-glob "$IN_GLOB" \
    --out-jsonl "$OUT" \
    --judge-prompts-jsonl "$JUDGE_OUT" \
    --rouge-thresh "$ROUGE"
