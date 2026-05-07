#!/usr/bin/env python3
"""
================================================================================
GPT-5 Correctness Judge — binary, leak-free open-ended QA evaluator
================================================================================
Like gpt5_mcq_judge.py, but:
  • Does NOT send the letter list (A/B/C/...) — so no "pick a letter" framing.
  • Does NOT leak the gold letter (no leakage bias).
  • Sends ONLY: question, model prediction, gold short answer text.
  • Asks for a binary verdict: correct / incorrect.

Why this exists:
  The MCQ judge artificially frames evaluation as "match the prediction to one
  of N choices" even though the model never outputs letters — it generates
  free-form text. That framing creates distractor noise AND requires us to
  show the gold letter to the judge "for tracking", which biases the verdict.

  The correct evaluation is much simpler:
      Q: Given the question, is the prediction equivalent to the gold answer?
      A: yes / no.

Usage:
  python tools/gpt5_correctness_judge.py \
      --predictions_path outputs/.../predictions.json \
      --num_samples -1 --max_workers 16
================================================================================
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict


# ─── Azure defaults ──────────────────────────────────────────────────────────
DEFAULT_MODEL       = "gpt-5.1_2025-11-13"
DEFAULT_ENDPOINT    = "https://<AZURE_OPENAI_ENDPOINT>"
DEFAULT_API_VERSION = "2024-12-01-preview"

_HERE = Path(__file__).resolve().parent
DEFAULT_CGBENCH = _HERE.parent.parent / "cgbench_setup" / "cgbench.json"


# ─── Prompts ─────────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for video-question-answering predictions.

You will be given:
  • A question about a video.
  • The reference (gold) short answer.
  • A model's free-form prediction.

Your task: decide whether the prediction answers the question with the SAME
meaning as the gold answer.

Rules:
  • Match on meaning, not surface form. Accept paraphrases, numeral/word swaps
    ("3" ↔ "three"), synonyms, and partial mentions of the key entity, AS LONG
    AS the prediction unambiguously conveys the gold answer.
  • Reject predictions that contradict the gold answer, give a different number,
    a different entity, a different direction, etc., even if loosely related.
  • Reject predictions that are off-topic or that fail to answer the question.
  • If the prediction is ambiguous or hedges between multiple answers, only
    accept it if the gold answer is clearly the intended one.

Respond with ONLY a JSON object:
{"verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low",
 "reason": "<one short sentence>"}"""


def build_user_prompt(question: str, gold_answer: str, prediction: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Gold answer: {gold_answer}\n\n"
        f"Model prediction: {prediction}"
    )


# ─── Azure client ────────────────────────────────────────────────────────────
def make_Azure_client(endpoint: str, api_version: str):
    try:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AzureOpenAI
    except ImportError:
        print("ERROR: install azure-identity and openai → pip install azure-identity openai")
        sys.exit(1)

    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def judge_one(client, model: str, question: str, gold_answer: str,
              prediction: str, retries: int = 2) -> dict:
    """One Azure call. Returns {"verdict", "confidence", "reason"} or {"error"}."""
    user_msg = build_user_prompt(question, gold_answer, prediction)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=300,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            return {"error": str(e)}


# ─── Main pipeline ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="GPT-5 binary correctness judge (no letter leak).",
        epilog=__doc__,
    )
    p.add_argument("--predictions_path", required=True)
    p.add_argument("--cgbench_path", default=str(DEFAULT_CGBENCH))
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--max_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--api_version", default=DEFAULT_API_VERSION)
    p.add_argument("--question_lookup",
                   help="Optional sft test json with original question text.")
    p.add_argument("--gold_source", choices=["choice", "answer"], default="choice",
                   help="What to use as gold answer text. "
                        "'choice' = cgbench['choices'][right_answer] (short MCQ option, default). "
                        "'answer' = cgbench['answer'] (longer reference sentence).")
    return p.parse_args()


def load_question_lookup(path: str | None) -> dict:
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

    question_lookup = load_question_lookup(args.question_lookup)

    matchable = []
    for p in preds:
        oid = str(p.get("metadata", {}).get("original_id", ""))
        if oid in cg:
            matchable.append((oid, p))
    print(f"  {len(matchable)} matchable to CGBench MCQ.")

    if args.num_samples > 0 and args.num_samples < len(matchable):
        random.seed(args.seed)
        matchable = random.sample(matchable, args.num_samples)
    print(f"  Judging {len(matchable)} samples with {args.max_workers} workers.")
    print(f"  Gold source: '{args.gold_source}'")

    tasks = []
    for oid, p in matchable:
        item = cg[oid]
        gold_letter = item["right_answer"].upper()
        gold_idx = ord(gold_letter) - 65
        if args.gold_source == "choice":
            gold_text = item["choices"][gold_idx]
        else:
            gold_text = item.get("answer", "") or item["choices"][gold_idx]
        question = question_lookup.get(oid, item.get("question", ""))
        tasks.append({
            "id": oid,
            "question": question,
            "choices": item["choices"],
            "prediction": p["prediction"],
            "gold_letter": gold_letter,
            "gold_answer_text": gold_text,
            "sub_category": p.get("metadata", {}).get("sub_category",
                              item.get("sub_category", "?")),
            "domain": p.get("metadata", {}).get("domain",
                        item.get("domain", "?")),
        })

    client = make_Azure_client(args.endpoint, args.api_version)
    print(f"  Using model: {args.model}")

    results = [None] * len(tasks)
    t0 = time.time()
    completed = 0

    def worker(idx: int):
        t = tasks[idx]
        verdict = judge_one(client, args.model, t["question"],
                            t["gold_answer_text"], t["prediction"])
        return idx, verdict

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(len(tasks))]
        for fut in as_completed(futures):
            idx, verdict = fut.result()
            t = tasks[idx]
            v = (verdict.get("verdict") or "").strip().lower()
            is_correct = v.startswith("correct")
            results[idx] = {
                **t,
                "verdict":    v or verdict.get("error", ""),
                "confidence": verdict.get("confidence", "?"),
                "reason":     verdict.get("reason", verdict.get("error", "")),
                "is_correct": is_correct,
                "error":      verdict.get("error"),
            }
            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(tasks)}] {rate:.1f} samples/s  "
                      f"({elapsed:.1f}s elapsed)")

    n_total = len(results)
    valid = [r for r in results if r and r["verdict"] in ("correct", "incorrect")]
    n_valid = len(valid)
    n_correct = sum(1 for r in valid if r["is_correct"])
    n_errors = sum(1 for r in results if r and r.get("error"))

    print()
    print(f"{'═'*70}")
    print(f"  GPT-5 Correctness Judge — {os.path.basename(args.predictions_path)}")
    print(f"{'─'*70}")
    print(f"  Judged:        {n_valid}/{n_total}")
    if n_valid:
        print(f"  Correct:       {n_correct}/{n_valid}  =  {100*n_correct/n_valid:.1f}%")
    print(f"  Azure errors:  {n_errors}")

    by_conf = defaultdict(lambda: [0, 0])
    for r in valid:
        by_conf[r["confidence"]][1] += 1
        if r["is_correct"]:
            by_conf[r["confidence"]][0] += 1
    if by_conf:
        print(f"\n  Confidence × accuracy:")
        for c, (cor, tot) in sorted(by_conf.items()):
            print(f"    {c:<8}  {cor}/{tot} = {100*cor/tot:.1f}%")

    by_sub = defaultdict(lambda: [0, 0])
    for r in valid:
        by_sub[r["sub_category"]][1] += 1
        if r["is_correct"]:
            by_sub[r["sub_category"]][0] += 1
    if by_sub and len(by_sub) > 1:
        print(f"\n  Per-sub-category:")
        for s, (cor, tot) in sorted(by_sub.items(), key=lambda kv: -kv[1][0]/max(1,kv[1][1])):
            print(f"    {s:<28} {cor}/{tot} = {100*cor/tot:.1f}%")

    out_path = args.output_path or os.path.join(
        os.path.dirname(args.predictions_path), "gpt5_correctness_judge.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "predictions_file": args.predictions_path,
            "gold_source": args.gold_source,
            "num_total": n_total,
            "num_valid": n_valid,
            "num_correct": n_correct,
            "accuracy": (n_correct / n_valid) if n_valid else None,
            "errors": n_errors,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
