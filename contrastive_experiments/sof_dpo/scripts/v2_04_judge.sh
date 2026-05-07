#!/bin/bash
# v2_04_judge.sh — Run the LLM-as-judge on the filtered pairs.
# Uses the SAME GPT-5 judge wording as mllm_evaluation/evaluate_parallel.py
# (the v2 filter emitted prompts in that exact JSON-output format).
#
# v1's run_judge_parallel.py expects a one-word YES/PARTIAL/NO reply, but the
# v2 prompts ask for {"verdict": true|false, "reason": "..."} JSON.  We use a
# tiny JSON-aware judge runner (build_pairs/run_judge_parallel_v2.py) that
# parses {verdict: bool} and remaps true -> "YES" / false -> "NO" so all
# downstream tooling (apply_judge_to_dpo.py) keeps working unchanged.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO/old_dpo_revised_data_8b"
IN="${IN:-$DATA_DIR/judge_prompts.jsonl}"
OUT="${OUT:-$DATA_DIR/judge_results.jsonl}"
MODEL="${MODEL:-gpt-5.1_2025-11-13}"
WORKERS="${WORKERS:-24}"

python3 "$REPO/build_pairs/run_judge_parallel_v2.py" \
    --in "$IN" \
    --out "$OUT" \
    --model "$MODEL" \
    --workers "$WORKERS"
