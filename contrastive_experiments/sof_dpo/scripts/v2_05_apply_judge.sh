#!/bin/bash
# v2_05_apply_judge.sh — Drop pairs the judge said are YES/PARTIAL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO/old_dpo_revised_data_8b"

python3 "$REPO/build_pairs/apply_judge_to_dpo.py" \
    --judge "$DATA_DIR/judge_results.jsonl" \
    --train "$DATA_DIR/sof_dpo_train.json" \
    --val   "$DATA_DIR/sof_dpo_train.val.json"

# Also produce a judged SFT companion (drop the same query_ids).
python3 - <<'PY'
import json, os
DATA_DIR = "/workspace/Pupil/contrastive_experiments/sof_dpo/old_dpo_revised_data_8b"

def load_judge(p):
    out = {}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["query_id"]] = r.get("verdict", "ERROR")
    return out

verdicts = load_judge(os.path.join(DATA_DIR, "judge_results.jsonl"))

for tag, src in (("train", "sof_sft_warmstart.no_transcript.json"),
                 ("val",   "sof_sft_warmstart.no_transcript.val.json")):
    sp = os.path.join(DATA_DIR, src)
    if not os.path.exists(sp):
        print(f"[sft-judge] skip {tag}: {sp} missing")
        continue
    data = json.load(open(sp))
    kept = [r for r in data if verdicts.get(r["id"], "MISSING") not in ("YES", "PARTIAL")]
    out_p = sp.replace(".json", ".judged.json")
    json.dump(kept, open(out_p, "w"), ensure_ascii=False, indent=0)
    print(f"[sft-judge] {tag}: {len(data)} -> {len(kept)}  -> {out_p}")
PY
