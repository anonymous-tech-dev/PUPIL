#!/usr/bin/env python3
"""
===============================================================================
GPT-5 MCQ Judge — concurrent LLM-as-judge for MCQ-grounded predictions
===============================================================================
Re-scores `predictions.json` using GPT-5 (via Azure) as the matcher between the
free-form prediction text and the original CGBench MCQ choices.

Why this exists:
  The default regex/word-overlap matcher (tools/eval_mcq_accuracy.py) is brittle
  to numeral/word ("3" vs "three"), paraphrasing, and partial mentions.
  GPT-5 picks the MCQ option a human would, giving us an accuracy figure free
  of those artifacts.

Concurrency:
  Uses ThreadPoolExecutor with --max_workers (default 10) to dispatch sample
  judgments in parallel. Each call is independent.

Usage:
  # Score 10 random samples from V-07 predictions
  python tools/gpt5_mcq_judge.py \
      --predictions_path outputs/V-07_.../test_results_full_video/predictions.json \
      --num_samples 10

  # Score the entire 750-sample test set
  python tools/gpt5_mcq_judge.py \
      --predictions_path outputs/V-07_.../test_results_full_video/predictions.json \
      --num_samples -1 --max_workers 16


  python tools/gpt5_mcq_judge.py \
      --predictions_path  /workspace/Pupil/contrastive_experiments/outputs/T-04_generative_grad_fix_fps1_lambda1.0_alpha5.0_lr2e-5_ep1_65536seq/checkpoint-200/test_results_matched/predictions.json \
    --num_samples -1 --max_workers 16

  python tools/gpt5_mcq_judge.py \
      --predictions_path   /workspace/Pupil/contrastive_experiments/outputs/dpo-from-sft-T04-a5_lr2e-6_beta0.1_ep1/test_results/predictions.json \
      --num_samples -1 --max_workers 16
===============================================================================
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


# ─── Azure defaults ────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "gpt-5.1_2025-11-13"
DEFAULT_ENDPOINT    = "https://<AZURE_OPENAI_ENDPOINT>"
DEFAULT_API_VERSION = "2024-12-01-preview"

# ─── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DEFAULT_CGBENCH = _HERE.parent.parent / "cgbench_setup" / "cgbench.json"


JUDGE_SYSTEM_PROMPT = """You are an evaluator for video-question-answering predictions.

You will be given:
  • A question
  • A list of candidate answer choices labelled A, B, C, ...
  • A model's free-form prediction text
  • The gold answer letter (revealed to you only for tracking — do NOT let it bias your choice)

Your task: pick the candidate letter that best matches the model's prediction.
Match on meaning, not surface form — accept numeral/word substitutions
("3" vs "three"), paraphrases, partial mentions of the key entity, and
re-orderings. If the prediction does not unambiguously support any choice,
pick the closest one and mark `confidence` accordingly.

Respond with ONLY a JSON object of the form:
{"chosen": "<letter>", "confidence": "<high|medium|low>", "reason": "<one short sentence>"}"""


def build_user_prompt(question: str, choices: list[str], prediction: str, gold_letter: str) -> str:
    choice_block = "\n".join(f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"Question: {question}\n\n"
        f"Choices:\n{choice_block}\n\n"
        f"Gold answer (for tracking only — do not let this bias you): {gold_letter}\n\n"
        f"Model prediction: {prediction}"
    )


# ─── Azure client ──────────────────────────────────────────────────────────────

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


def judge_one(client, model: str, question: str, choices: list[str],
              prediction: str, gold_letter: str, retries: int = 2) -> dict:
    """One Azure call. Returns {"chosen", "confidence", "reason"} or {"error"}."""
    user_msg = build_user_prompt(question, choices, prediction, gold_letter)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
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


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="GPT-5 MCQ judge with parallel Azure calls.",
        epilog=__doc__,
    )
    p.add_argument("--predictions_path", required=True,
                   help="Path to predictions.json from a test run.")
    p.add_argument("--cgbench_path", default=str(DEFAULT_CGBENCH),
                   help=f"Path to cgbench.json (default: {DEFAULT_CGBENCH}).")
    p.add_argument("--num_samples", type=int, default=10,
                   help="How many predictions to judge. -1 = all. (default: 10)")
    p.add_argument("--max_workers", type=int, default=10,
                   help="Concurrent Azure calls (default: 10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed when num_samples < total.")
    p.add_argument("--output_path", default=None,
                   help="Where to save per-sample judgements. "
                        "Default: alongside predictions.json as gpt5_judge.json.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Azure deployment (default: {DEFAULT_MODEL}).")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                   help=f"Azure endpoint (default: {DEFAULT_ENDPOINT}).")
    p.add_argument("--api_version", default=DEFAULT_API_VERSION,
                   help=f"Azure API version (default: {DEFAULT_API_VERSION}).")
    p.add_argument("--question_lookup",
                   help="Optional sft test json containing the original "
                        "question text (e.g. final_sft_data/test.json). "
                        "Falls back to cgbench question if absent.")
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

    # ── Load CGBench MCQ lookup ────────────────────────────────────────────
    if not os.path.exists(args.cgbench_path):
        sys.exit(f"ERROR: cgbench.json not found at {args.cgbench_path}")
    with open(args.cgbench_path) as f:
        cg = {str(x["qid"]): x for x in json.load(f)}
    print(f"Loaded {len(cg)} CGBench MCQ items.")

    # ── Load predictions ───────────────────────────────────────────────────
    with open(args.predictions_path) as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions from {args.predictions_path}")

    # ── Optional question lookup (sft test split) ──────────────────────────
    question_lookup = load_question_lookup(args.question_lookup)

    # ── Filter to CGBench-matchable samples ────────────────────────────────
    matchable = []
    for p in preds:
        oid = str(p.get("metadata", {}).get("original_id", ""))
        if oid in cg:
            matchable.append((oid, p))
    print(f"  {len(matchable)} matchable to CGBench MCQ.")

    # ── Sample ─────────────────────────────────────────────────────────────
    if args.num_samples > 0 and args.num_samples < len(matchable):
        random.seed(args.seed)
        matchable = random.sample(matchable, args.num_samples)
    print(f"  Judging {len(matchable)} samples with {args.max_workers} workers.")

    # ── Build per-sample task list ─────────────────────────────────────────
    tasks = []
    for oid, p in matchable:
        item = cg[oid]
        gold_letter = item["right_answer"].upper()
        question = question_lookup.get(oid, item.get("question", ""))
        tasks.append({
            "id": oid,
            "question": question,
            "choices": item["choices"],
            "prediction": p["prediction"],
            "gold_letter": gold_letter,
            "gold_answer_text": item.get("answer", ""),
            "sub_category": p.get("metadata", {}).get("sub_category",
                              item.get("sub_category", "?")),
            "domain": p.get("metadata", {}).get("domain",
                        item.get("domain", "?")),
        })

    # ── Dispatch to Azure in parallel ──────────────────────────────────────
    client = make_Azure_client(args.endpoint, args.api_version)
    print(f"  Using model: {args.model}")

    results = [None] * len(tasks)
    t0 = time.time()
    completed = 0

    def worker(idx: int):
        t = tasks[idx]
        verdict = judge_one(client, args.model, t["question"], t["choices"],
                            t["prediction"], t["gold_letter"])
        return idx, verdict

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(len(tasks))]
        for fut in as_completed(futures):
            idx, verdict = fut.result()
            t = tasks[idx]
            chosen = (verdict.get("chosen") or "").strip().upper()[:1]
            is_correct = (chosen == t["gold_letter"]) if chosen.isalpha() else False
            results[idx] = {
                **t,
                "chosen_letter": chosen,
                "confidence":   verdict.get("confidence", "?"),
                "reason":       verdict.get("reason", verdict.get("error", "")),
                "is_correct":   is_correct,
                "error":        verdict.get("error"),
            }
            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(tasks)}] {rate:.1f} samples/s  "
                      f"({elapsed:.1f}s elapsed)")

    # ── Aggregate ──────────────────────────────────────────────────────────
    valid = [r for r in results if r and r["chosen_letter"] in [chr(65+i) for i in range(20)]]
    n_total = len(results)
    n_valid = len(valid)
    n_correct = sum(1 for r in valid if r["is_correct"])
    n_errors = sum(1 for r in results if r and r.get("error"))

    print()
    print(f"{'═'*70}")
    print(f"  GPT-5 Judge results — {os.path.basename(args.predictions_path)}")
    print(f"{'─'*70}")
    print(f"  Judged:        {n_valid}/{n_total}")
    print(f"  Correct:       {n_correct}/{n_valid}  =  "
          f"{100*n_correct/n_valid:.1f}%" if n_valid else "  Correct: n/a")
    print(f"  Azure errors:  {n_errors}")

    # Confidence breakdown
    by_conf = defaultdict(lambda: [0, 0])
    for r in valid:
        by_conf[r["confidence"]][1] += 1
        if r["is_correct"]:
            by_conf[r["confidence"]][0] += 1
    if by_conf:
        print(f"\n  Confidence × accuracy:")
        for c, (cor, tot) in sorted(by_conf.items()):
            print(f"    {c:<8}  {cor}/{tot} = {100*cor/tot:.1f}%")

    # Sub-category
    by_sub = defaultdict(lambda: [0, 0])
    for r in valid:
        by_sub[r["sub_category"]][1] += 1
        if r["is_correct"]:
            by_sub[r["sub_category"]][0] += 1
    if by_sub and len(by_sub) > 1:
        print(f"\n  Per-sub-category:")
        for s, (cor, tot) in sorted(by_sub.items(), key=lambda kv: -kv[1][0]/max(1,kv[1][1])):
            print(f"    {s:<28} {cor}/{tot} = {100*cor/tot:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = args.output_path
    if not out_path:
        out_path = os.path.join(os.path.dirname(args.predictions_path),
                                "gpt5_judge.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "predictions_file": args.predictions_path,
            "num_total": n_total,
            "num_valid": n_valid,
            "num_correct": n_correct,
            "accuracy": (n_correct / n_valid) if n_valid else None,
            "errors": n_errors,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved per-sample judgements → {out_path}")


if __name__ == "__main__":
    main()
