#!/usr/bin/env python3
"""
===============================================================================
LLM-as-Judge Evaluation — Parallel Azure dispatcher
===============================================================================
Judges model predictions against ground truth using GPT-5.1 via Azure Azure.
Reads *_results.json files produced by script_parallel.py, fills in
judge_verdict / judge_reason fields, and (by default) writes the judged
copies into a mirrored directory under `results_v2/` so the original
`results/` files are never mutated.

  /…/results/<model>/<run>/foo_results.json   ──►   /…/results_v2/<model>/<run>/foo_results.json

If the source path doesn't contain a '/results/' segment, or `--in-place`
is passed, files are written back in place (legacy behaviour).

Hot-resumable: on second invocation we read from the existing `results_v2/`
copy (if it already exists) and skip entries where judge_verdict is set.

Usage:
    # Judge all results for a model (output → results_v2/...)
    python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark

    # Judge with more concurrency
    python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --max-workers 20

    # Overwrite existing judgements
    python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --overwrite

    # Dry run — just count pending
    python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --dry-run

    # Legacy: write verdicts back into the source results/ files
    python evaluate_parallel.py --results-dir results/qwen3_vl/final_1k_benchmark --in-place
===============================================================================
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Lock
from tqdm import tqdm

print = functools.partial(print, flush=True)

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "gpt-5.1_2025-11-13"
DEFAULT_ENDPOINT    = "https://<AZURE_OPENAI_ENDPOINT>"
DEFAULT_API_VERSION = "2024-12-01-preview"

# ─── Judge prompt ──────────────────────────────────────────────────────────────

def build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """Identical wording to the judge prompt in `evaluate.py`, but with a
    *balancing* second example (verdict=false) added to counter the anchoring
    bias of the original single verdict=true example."""
    return f"""
    You are an impartial judge evaluating a Multimodal AI's response.

    Question: "{question}"
    Ground Truth: "{ground_truth}"
    Model Prediction: "{prediction}"

    Task:
    1. Compare the factual core of the Model Prediction against the Ground Truth.
    2. Determine if the prediction matches the ground truth (ignore minor phrasing differences).
    3. Output strictly valid JSON.

    Format examples (illustrative only — do NOT copy their content):
    {{
        "reason": "Short explanation of why the prediction matches the ground truth...",
        "verdict": true
    }}
    {{
        "reason": "Short explanation of why the prediction does NOT match the ground truth...",
        "verdict": false
    }}
    """
# ─── Azure client ──────────────────────────────────────────────────────────────

def make_Azure_client(endpoint: str, api_version: str):
    from azure.identity import AzureCliCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def judge_one(client, model: str, question: str, ground_truth: str,
              prediction: str, retries: int = 3) -> dict:
    """Single Azure call. Returns {"verdict": bool, "reason": str} or {"error": str}."""
    user_msg = build_judge_prompt(question, ground_truth, prediction)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=300,
            )
            data = json.loads(resp.choices[0].message.content)
            verdict = data.get("verdict", False)
            if isinstance(verdict, str):
                verdict = verdict.lower() == "true"
            return {"verdict": verdict, "reason": data.get("reason", "")}
        except Exception as e:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            return {"error": str(e)}


# ─── File I/O with locking (thread-safe saves) ────────────────────────────────

class ResultFileManager:
    """Thread-safe read/write for per-video result JSON files."""

    def __init__(self):
        self._locks = defaultdict(Lock)

    def load(self, filepath):
        with self._locks[filepath]:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

    def save(self, filepath, data):
        with self._locks[filepath]:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel LLM-as-Judge evaluation for Pupil results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results-dir", required=True,
                   help="Directory containing *_results.json files.")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Concurrent Azure calls (default: 16).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-judge entries that already have a verdict.")
    p.add_argument("--dry-run", action="store_true",
                   help="Just count pending entries; don't call Azure.")
    p.add_argument("--in-place", action="store_true",
                   help="Write verdicts back to the source dir instead of "
                        "mirroring to results_v2/ (legacy behaviour).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Azure deployment (default: {DEFAULT_MODEL}).")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--api-version", default=DEFAULT_API_VERSION)
    return p.parse_args()


def resolve_output_dir(src_dir: str, in_place: bool) -> str:
    """Map source `…/results/…` → `…/results_v2/…`. If no '/results/' segment
    or --in-place was passed, the output dir equals the source dir."""
    abs_src = os.path.abspath(src_dir)
    if in_place:
        return abs_src
    sep = os.sep + "results" + os.sep
    if sep in abs_src:
        return abs_src.replace(sep, os.sep + "results_v2" + os.sep, 1)
    return abs_src


def main():
    args = parse_args()
    src_dir = args.results_dir

    if not os.path.isdir(src_dir):
        sys.exit(f"❌ Source directory not found: {src_dir}")

    src_files = sorted(glob.glob(os.path.join(src_dir, "*_results.json")))
    if not src_files:
        sys.exit(f"❌ No *_results.json files found in {src_dir}")

    # ── Resolve output directory (mirror /results/ → /results_v2/) ─────────
    output_dir = resolve_output_dir(src_dir, args.in_place)
    if output_dir == os.path.abspath(src_dir):
        if args.in_place:
            print(f"📝 In-place mode: writing back to {output_dir}")
        else:
            print(f"⚠️  No '/results/' segment in source path — writing in-place to {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        seeded = 0
        for src in src_files:
            dst = os.path.join(output_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                seeded += 1
        print(f"📦 Source : {src_dir}")
        print(f"📤 Output : {output_dir}")
        print(f"   Seeded {seeded}/{len(src_files)} files into output dir "
              f"({len(src_files) - seeded} already existed and will resume).")

    # All subsequent reads/writes operate on the OUTPUT directory.
    results_dir = output_dir
    result_files = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))

    # ── Scan: build task list ──────────────────────────────────────────────
    tasks = []  # (filepath, entry_index, question, ground_truth, prediction, query_id)
    already_judged = 0
    no_ground_truth = 0

    for fp in result_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Skipping corrupt file: {fp} ({e})")
            continue

        for idx, entry in enumerate(data):
            has_verdict = entry.get("judge_verdict") is not None
            if has_verdict and not args.overwrite:
                already_judged += 1
                continue

            gt = entry.get("ground_truth", "")
            if not gt:
                no_ground_truth += 1
                continue

            tasks.append((
                fp,
                idx,
                entry.get("question", ""),
                gt,
                entry.get("model_prediction", ""),
                entry.get("query_id", "unknown"),
            ))

    total_entries = already_judged + no_ground_truth + len(tasks)
    print(f"\n{'='*65}")
    print(f"  LLM-as-Judge Evaluation")
    print(f"  Directory:      {results_dir}")
    print(f"  Result files:   {len(result_files)}")
    print(f"  Total entries:  {total_entries}")
    print(f"  Already judged: {already_judged}")
    print(f"  No ground truth:{no_ground_truth}")
    print(f"  Pending:        {len(tasks)}")
    print(f"  Workers:        {args.max_workers}")
    print(f"  Model:          {args.model}")
    print(f"{'='*65}\n")

    if len(tasks) == 0:
        print("🎉 Nothing to judge — all entries already have verdicts!")
        return

    if args.dry_run:
        print("🧪 DRY RUN — would judge the above entries. Exiting.")
        return

    # ── Initialize Azure client ────────────────────────────────────────────
    client = make_Azure_client(args.endpoint, args.api_version)
    print("✅ Azure Azure client initialized\n")

    fm = ResultFileManager()
    t0 = time.time()
    correct = 0
    errors = 0
    completed = 0

    # ── Pre-load all files into memory for thread-safe updates ─────────────
    file_cache = {}
    for fp in set(t[0] for t in tasks):
        with open(fp, "r", encoding="utf-8") as f:
            file_cache[fp] = json.load(f)

    file_locks = defaultdict(Lock)
    save_counter = defaultdict(int)  # track saves per file for batching
    SAVE_EVERY = 5  # save after every N updates per file

    def worker(task_tuple):
        fp, idx, question, ground_truth, prediction, query_id = task_tuple
        result = judge_one(client, args.model, question, ground_truth, prediction)
        return fp, idx, query_id, result

    pbar = tqdm(total=len(tasks), desc="Judging", unit="q", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}

        for future in as_completed(futures):
            fp, idx, query_id, result = future.result()

            if "error" in result:
                errors += 1
                with file_locks[fp]:
                    file_cache[fp][idx]["judge_verdict"] = None
                    file_cache[fp][idx]["judge_reason"] = f"Azure_ERROR: {result['error']}"
                pbar.set_postfix_str(f"ERR {query_id}")
            else:
                verdict = result["verdict"]
                reason = result["reason"]
                with file_locks[fp]:
                    file_cache[fp][idx]["judge_verdict"] = verdict
                    file_cache[fp][idx]["judge_reason"] = reason
                if verdict:
                    correct += 1

            completed += 1

            # Periodic save (thread-safe, per-file)
            with file_locks[fp]:
                save_counter[fp] += 1
                if save_counter[fp] >= SAVE_EVERY:
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(file_cache[fp], f, indent=4, ensure_ascii=False)
                    save_counter[fp] = 0

            pbar.update(1)
            pbar.set_postfix_str(
                f"✓{correct} ✗{completed-correct-errors} err{errors}"
            )

    pbar.close()

    # ── Final save all files ───────────────────────────────────────────────
    for fp, data in file_cache.items():
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    elapsed = time.time() - t0
    n_judged = completed - errors

    # ── Category breakdown ─────────────────────────────────────────────────
    cat_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    sof_stats = defaultdict(lambda: [0, 0])
    for fp, data in file_cache.items():
        for entry in data:
            v = entry.get("judge_verdict")
            if v is None:
                continue
            cat = entry.get("category", "unknown")
            sof = entry.get("source_of_fact", "unknown")
            cat_stats[cat][1] += 1
            sof_stats[sof][1] += 1
            if v:
                cat_stats[cat][0] += 1
                sof_stats[sof][0] += 1

    total_judged_all = sum(v[1] for v in cat_stats.values())
    total_correct_all = sum(v[0] for v in cat_stats.values())

    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'─'*65}")
    print(f"  Judged this run: {n_judged}")
    print(f"  Azure errors:    {errors}")
    print(f"  Time:            {elapsed:.1f}s ({n_judged/elapsed:.1f} q/s)" if elapsed > 0 else "")
    print(f"{'─'*65}")
    print(f"  OVERALL (all entries with verdicts):")
    print(f"  Accuracy:  {total_correct_all}/{total_judged_all}"
          f"  =  {100*total_correct_all/total_judged_all:.1f}%" if total_judged_all else "")
    print(f"{'─'*65}")

    if cat_stats:
        print(f"  Per Cognitive Category:")
        for cat, (cor, tot) in sorted(cat_stats.items(), key=lambda kv: -kv[1][1]):
            print(f"    {cat:<30} {cor:>3}/{tot:<3}  = {100*cor/tot:.1f}%")

    if sof_stats:
        print(f"\n  Per Source-of-Fact:")
        for sof, (cor, tot) in sorted(sof_stats.items(), key=lambda kv: -kv[1][1]):
            print(f"    {sof:<30} {cor:>3}/{tot:<3}  = {100*cor/tot:.1f}%")

    print(f"{'='*65}")
    print(f"  Results saved at: {results_dir}")
    if not args.in_place and results_dir != os.path.abspath(src_dir):
        print(f"  (source files at {src_dir} were NOT modified)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
