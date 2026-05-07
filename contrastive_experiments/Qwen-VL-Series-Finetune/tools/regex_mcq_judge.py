#!/usr/bin/env python3
"""
Regex/String MCQ Judge — deterministic, CPU-only, no LLM/GPU.
Matches the model's free-form prediction to the gold MCQ option text from
CGBench using simple text-overlap rules.

Same output format as gpt5_mcq_judge.py.

Usage:
  python tools/regex_mcq_judge.py \
      --predictions_path outputs/.../predictions.json
"""
import argparse, json, os, re, string, sys, time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_CGBENCH = _HERE.parent.parent / "cgbench_setup" / "cgbench.json"

# ── Number-word ↔ digit aliases ───────────────────────────────────────────────
W2N = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
       "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
       "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
       "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
       "eighteen": "18", "nineteen": "19", "twenty": "20",
       "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
       "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}

STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','can','may','might','of',
    'in','on','at','to','for','with','by','from','about','into','as','and','or',
    'but','if','this','that','these','those','it','its','i','you','he','she','we',
    'they','them','his','her','their','my','our','your','said','says','say','also',
    'just','then','than','so','too','very','out','up','down','off','over','under',
    'video','protagonist','picture','screen','image','shows','shown','appears',
    'appear','appeared','scene','wearing','wore','one','two', # numbers handled separately
}


def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)        # strip punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    # Replace word-numbers with digits so "Two" matches "2"
    out = []
    for tok in t.split():
        out.append(W2N.get(tok, tok))
    return ' '.join(out)


def tokens(text: str) -> set:
    return {t for t in normalize(text).split() if len(t) > 0}


def content_tokens(text: str) -> set:
    return {t for t in tokens(text) if t not in STOPWORDS}


def score_choice(pred_norm: str, pred_toks: set, pred_content: set,
                 choice: str) -> float:
    """
    Score how well *prediction* matches *choice* using:
      • exact substring (very strong signal)
      • content-token recall  (|choice_content ∩ pred_content| / |choice_content|)
      • SequenceMatcher ratio (soft fallback)
    """
    c_norm = normalize(choice)
    if not c_norm:
        return 0.0

    # 1. Substring of normalised choice in prediction → very strong
    substr = 1.0 if c_norm in pred_norm else 0.0

    # 2. Recall of choice's CONTENT tokens inside prediction
    c_content = content_tokens(choice)
    if c_content:
        recall = len(c_content & pred_content) / len(c_content)
    else:
        # Choice has no content tokens (rare — e.g. "the") fall back to all toks
        c_toks = tokens(choice)
        recall = len(c_toks & pred_toks) / max(1, len(c_toks))

    # 3. Soft sequence ratio
    seq = SequenceMatcher(None, pred_norm, c_norm).ratio()

    # Weighted combination — substring dominates
    return 3.0 * substr + 2.0 * recall + 0.5 * seq


def judge(prediction: str, choices: list) -> tuple:
    """Returns (best_idx, best_score, margin, all_scores)."""
    pred_norm = normalize(prediction)
    pred_toks = tokens(prediction)
    pred_content = content_tokens(prediction)

    scores = [score_choice(pred_norm, pred_toks, pred_content, c) for c in choices]

    best_idx = max(range(len(choices)), key=lambda i: scores[i])
    sorted_s = sorted(scores, reverse=True)
    margin = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else sorted_s[0]
    return best_idx, scores[best_idx], margin, scores


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Regex-based MCQ judge (deterministic, CPU-only).")
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--cgbench_path", default=str(DEFAULT_CGBENCH))
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", default=None)
    p.add_argument("--question_lookup", default=None)
    return p.parse_args()


def load_question_lookup(path):
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for item in data:
        oid = str(item.get("metadata", {}).get("original_id", ""))
        for turn in item.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                out[oid] = turn.get("value", "").replace("<video>", "").strip()
                break
    return out


def main():
    args = parse_args()

    if not os.path.exists(args.cgbench_path):
        sys.exit(f"ERROR: cgbench.json not found at {args.cgbench_path}")
    with open(args.cgbench_path) as f:
        cg = {str(x["qid"]): x for x in json.load(f)}
    print(f"Loaded {len(cg)} CGBench MCQ items.")

    with open(args.predictions_path) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions from {args.predictions_path}")

    ql = load_question_lookup(args.question_lookup)

    matchable = []
    for p in preds:
        oid = str(p.get("metadata", {}).get("original_id", ""))
        if oid in cg:
            matchable.append((oid, p))
    print(f"  {len(matchable)} matchable to CGBench MCQ.")

    if 0 < args.num_samples < len(matchable):
        import random
        random.seed(args.seed)
        matchable = random.sample(matchable, args.num_samples)

    t0 = time.time()
    results = []
    for oid, p in matchable:
        item = cg[oid]
        gold_letter = item["right_answer"].upper()
        choices = item["choices"]
        question = ql.get(oid, item.get("question", ""))

        best_idx, best_score, margin, all_scores = judge(p["prediction"], choices)
        chosen = chr(65 + best_idx)

        if margin > 1.5 and best_score > 1.5:
            conf = "high"
        elif margin > 0.4:
            conf = "medium"
        else:
            conf = "low"

        results.append({
            "id": oid,
            "question": question,
            "choices": choices,
            "prediction": p["prediction"],
            "gold_letter": gold_letter,
            "gold_answer_text": item.get("answer", ""),
            "sub_category": p.get("metadata", {}).get("sub_category",
                              item.get("sub_category", "?")),
            "domain": p.get("metadata", {}).get("domain",
                        item.get("domain", "?")),
            "chosen_letter": chosen,
            "confidence": conf,
            "reason": f"score={best_score:.2f}, margin={margin:.2f}",
            "is_correct": chosen == gold_letter,
            "error": None,
        })

    elapsed = time.time() - t0
    n_total = len(results)
    n_correct = sum(r["is_correct"] for r in results)

    print(f"\n{'='*70}")
    print(f"  Regex MCQ Judge — {os.path.basename(args.predictions_path)}")
    print(f"{'-'*70}")
    print(f"  Judged in {elapsed:.2f}s on CPU")
    print(f"  Correct: {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%")

    by_conf = defaultdict(lambda: [0, 0])
    for r in results:
        by_conf[r["confidence"]][1] += 1
        if r["is_correct"]:
            by_conf[r["confidence"]][0] += 1
    if by_conf:
        print(f"\n  Confidence × accuracy:")
        for c, (cor, tot) in sorted(by_conf.items()):
            print(f"    {c:<8}  {cor}/{tot} = {100*cor/tot:.1f}%")

    by_sub = defaultdict(lambda: [0, 0])
    for r in results:
        by_sub[r["sub_category"]][1] += 1
        if r["is_correct"]:
            by_sub[r["sub_category"]][0] += 1
    if len(by_sub) > 1:
        print(f"\n  Per-sub-category:")
        for s, (cor, tot) in sorted(by_sub.items(),
                                     key=lambda kv: -kv[1][0]/max(1, kv[1][1])):
            print(f"    {s:<28} {cor}/{tot} = {100*cor/tot:.1f}%")

    out_path = args.output_path or os.path.join(
        os.path.dirname(args.predictions_path), "regex_judge.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "regex-judge-v1",
            "predictions_file": args.predictions_path,
            "num_total": n_total,
            "num_valid": n_total,
            "num_correct": n_correct,
            "accuracy": n_correct / n_total if n_total else None,
            "errors": 0,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
