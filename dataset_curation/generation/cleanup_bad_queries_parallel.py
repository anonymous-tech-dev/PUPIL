"""
Query Cleanup Script — Detect & Remove Bad (Failed-Generation) Questions
=========================================================================
PARALLEL version of cleanup_bad_queries.py

Phase 1 (DETECT): Uses GPT-5 via Azure to classify every question in each
    *_queries.json file under final_1k/ as GOOD or BAD.  Uses ThreadPoolExecutor
    with --max_workers to dispatch file-level LLM calls in parallel.
    Outputs a JSON manifest of files that need cleaning plus files where ALL
    questions are bad.

Phase 2 (CLEAN): Reads the manifest produced by Phase 1 and removes bad
    questions from the JSON files (in-place). (Same as original — no LLM.)

Phase 3 (TIMESTAMPS): Structurally cleans the dataset (no LLM required).
    (Same as original.)

Usage:
  # Phase 1 with 16 parallel workers
  python cleanup_bad_queries_parallel.py --phase 1 --max_workers 16

  # Phase 2 (no parallelism needed)
  python cleanup_bad_queries_parallel.py --phase 2

  # Phase 3
  python cleanup_bad_queries_parallel.py --phase 3
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             DEFAULTS                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

QUERIES_ROOT = "/workspace/Pupil/dataset_curation/dataset/queries_db/final_train"
SUB_FOLDERS = ["sof_visual", "sof_audio", "sof_priority", "sof_time"]
MANIFEST_PATH = "/workspace/Pupil/dataset_curation/cleanup_manifest.json"

DEPLOYMENT_NAME = "gpt-5.1_2025-11-13"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         LLM CLIENT SETUP                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def get_llm_client():
    """Initialise Azure OpenAI client using AzureCliCredential (Azure)."""
    print(f"🔑 Authenticating with Azure Azure ({DEPLOYMENT_NAME})...")
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    client = AzureOpenAI(
        azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
    )
    print("✅ Client ready.\n")
    return client


def classify_questions(client, questions: list[dict]) -> list[bool]:
    """
    Send all questions from one file to the LLM in a single call.
    Returns a list of booleans — True = BAD (failed generation), False = GOOD.
    """
    numbered = "\n".join(
        f"[Q{i+1}]\nQuestion: {q['question']}\nGround Truth: {q['ground_truth']}"
        for i, q in enumerate(questions)
    )

    system_prompt = (
        "You are a data-quality classifier. You will be given a list of question-answer pairs "
        "that were auto-generated from educational videos. Some of them are GOOD (sensible, "
        "well-formed questions with real answers) and some are BAD (the generator failed and "
        "the 'question' or 'ground_truth' is actually an error/refusal/apology message such as "
        "'Unable to generate…', 'I can't generate…', 'Not enough information…', or similar). "
        "A BAD question is one where the generator clearly could NOT produce a real question and "
        "instead output a refusal, apology, or meta-commentary about why it couldn't generate. "
        "A GOOD question is a genuine, answerable question about video content.\n\n"
        "Reply with ONLY a JSON array of labels, one per question, in order. "
        "Use exactly the strings \"GOOD\" or \"BAD\". Example: [\"GOOD\", \"BAD\", \"GOOD\"]"
    )

    user_prompt = f"Classify each of the following {len(questions)} questions:\n\n{numbered}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_completion_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Handle ```json ... ``` wrapping
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            labels = json.loads(raw)

            if len(labels) != len(questions):
                print(f"  ⚠️  LLM returned {len(labels)} labels for {len(questions)} questions, retrying...")
                continue

            return [label.upper() == "BAD" for label in labels]

        except Exception as e:
            print(f"  ⚠️  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ❌  All retries exhausted. Marking all as GOOD (safe default).")
                return [False] * len(questions)

    return [False] * len(questions)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      PHASE 1: DETECT (PARALLEL)                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _process_one_file(client, filepath: str) -> dict:
    """
    Worker function: classify questions in a single file.
    Returns a dict with the result for this file.
    """
    rel_path = os.path.relpath(filepath, QUERIES_ROOT)
    result = {
        "filepath": filepath,
        "rel_path": rel_path,
        "status": None,       # "good" | "mixed" | "all_bad" | "empty" | "error"
        "bad_indices": [],
        "num_questions": 0,
        "error": None,
    }

    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    video_key = list(data.keys())[0]
    questions = data[video_key]
    result["num_questions"] = len(questions)

    if len(questions) == 0:
        result["status"] = "empty"
        return result

    bad_flags = classify_questions(client, questions)
    bad_indices = [i for i, is_bad in enumerate(bad_flags) if is_bad]
    result["bad_indices"] = bad_indices

    if len(bad_indices) == 0:
        result["status"] = "good"
    elif len(bad_indices) == len(questions):
        result["status"] = "all_bad"
    else:
        result["status"] = "mixed"

    return result


def phase_detect(max_workers: int = 10):
    client = get_llm_client()

    # Discover all *_queries.json files
    all_query_files = []
    folders = SUB_FOLDERS if SUB_FOLDERS else os.listdir(QUERIES_ROOT)
    for folder in folders:
        folder_path = os.path.join(QUERIES_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            if f.endswith("_queries.json"):
                all_query_files.append(os.path.join(folder_path, f))

    n = len(all_query_files)
    print(f"📂 Found {n} query files across {len(folders)} folders.")
    print(f"🚀 Dispatching with {max_workers} parallel workers.\n")

    # ── Dispatch in parallel ───────────────────────────────────────────
    files_to_clean = {}
    all_bad_files = []
    total_bad_detected = 0
    total_questions_seen = 0
    completed = 0
    t0 = time.time()

    results = [None] * n

    def worker(idx: int):
        return idx, _process_one_file(client, all_query_files[idx])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i) for i in range(n)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            completed += 1

            # Progress
            rel = res["rel_path"]
            if res["status"] == "error":
                print(f"  [{completed}/{n}] ❌  {rel}: {res['error']}")
            elif res["status"] == "empty":
                print(f"  [{completed}/{n}] ⏭️  {rel}: empty")
            elif res["status"] == "good":
                print(f"  [{completed}/{n}] ✅  {rel}: all {res['num_questions']} GOOD")
            elif res["status"] == "all_bad":
                print(f"  [{completed}/{n}] 🔴  {rel}: ALL {res['num_questions']} BAD")
            elif res["status"] == "mixed":
                bad_n = len(res["bad_indices"])
                good_n = res["num_questions"] - bad_n
                print(f"  [{completed}/{n}] 🟡  {rel}: {bad_n} BAD / {good_n} GOOD")

            # Progress rate every ~10%
            if completed % max(1, n // 10) == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"       ↳ {completed}/{n} done  ({rate:.1f} files/s, {elapsed:.1f}s elapsed)")

    # ── Aggregate ──────────────────────────────────────────────────────
    for res in results:
        if res is None:
            continue
        total_questions_seen += res["num_questions"]
        if res["status"] == "all_bad":
            all_bad_files.append(res["filepath"])
            total_bad_detected += res["num_questions"]
        elif res["status"] == "mixed":
            files_to_clean[res["filepath"]] = res["bad_indices"]
            total_bad_detected += len(res["bad_indices"])

    # ── Write manifest ─────────────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_files_scanned": n,
        "total_questions_seen": total_questions_seen,
        "total_bad_detected": total_bad_detected,
        "files_to_clean": files_to_clean,
        "all_bad_files": all_bad_files,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("                     PHASE 1 — DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total files scanned       : {n}")
    print(f"  Total questions seen       : {total_questions_seen}")
    print(f"  Total BAD questions found  : {total_bad_detected}")
    print(f"  Files needing cleanup      : {len(files_to_clean)}  (mixed good/bad)")
    print(f"  Files ALL bad (untouched)  : {len(all_bad_files)}")
    print(f"  Wall-clock time            : {elapsed:.1f}s")
    print(f"  Throughput                 : {n/elapsed:.1f} files/s" if elapsed > 0 else "")
    print(f"  Manifest saved to          : {MANIFEST_PATH}")

    if files_to_clean:
        print(f"\n  📋 Files to clean (bad indices):")
        for fp, indices in files_to_clean.items():
            print(f"     {os.path.relpath(fp, QUERIES_ROOT)}  →  bad indices: {indices}")

    if all_bad_files:
        print(f"\n  🔴 Files where ALL questions are bad (will NOT be modified):")
        for fp in all_bad_files:
            print(f"     {os.path.relpath(fp, QUERIES_ROOT)}")

    print("=" * 70)

    print("\n\n# ---- COPY-PASTEABLE ARRAYS ----")
    print(f"files_to_clean = {json.dumps(files_to_clean, indent=2)}")
    print(f"\nall_bad_files = {json.dumps(all_bad_files, indent=2)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            PHASE 2: CLEAN                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def phase_clean():
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌  Manifest not found at {MANIFEST_PATH}.")
        print("   Run Phase 1 first (--phase 1).")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    files_to_clean = manifest.get("files_to_clean", {})
    all_bad_files = manifest.get("all_bad_files", [])

    if not files_to_clean:
        print("✅  No files to clean. Nothing to do.")
        return

    print(f"🧹 Phase 2: Cleaning {len(files_to_clean)} files...\n")
    total_deleted = 0

    for filepath, bad_indices in files_to_clean.items():
        rel_path = os.path.relpath(filepath, QUERIES_ROOT)

        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found (skipping): {rel_path}")
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  ❌  Could not read {rel_path}: {e}")
            continue

        video_key = list(data.keys())[0]
        questions = data[video_key]
        original_count = len(questions)

        for i in sorted(bad_indices, reverse=True):
            if i < len(questions):
                removed_q = questions.pop(i)
                total_deleted += 1
                print(f"  🗑️  {rel_path}: removed index {i} — \"{removed_q['question'][:80]}...\"")

        data[video_key] = questions
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        print(f"  ✅  {rel_path}: {original_count} → {len(questions)} questions\n")

    print("\n" + "=" * 70)
    print("                      PHASE 2 — CLEANING SUMMARY")
    print("=" * 70)
    print(f"  Files cleaned             : {len(files_to_clean)}")
    print(f"  Total questions deleted   : {total_deleted}")

    if all_bad_files:
        print(f"\n  🔴 Reminder — these files had ALL bad questions (untouched):")
        for fp in all_bad_files:
            print(f"     {os.path.relpath(fp, QUERIES_ROOT)}")

    print("=" * 70)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                      PHASE 3: CLEAN TIMESTAMPS                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def phase_clean_timestamps():
    print(f"🧹 Phase 3: Analyzing structural timestamp issues...\n")

    all_query_files = []
    folders = SUB_FOLDERS if SUB_FOLDERS else os.listdir(QUERIES_ROOT)
    for folder in folders:
        folder_path = os.path.join(QUERIES_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            if f.endswith("_queries.json"):
                all_query_files.append(os.path.join(folder_path, f))

    total_files_to_modify = 0
    total_empty_removed = 0
    total_multi_removed = 0
    pending_writes = {}

    for filepath in all_query_files:
        rel_path = os.path.relpath(filepath, QUERIES_ROOT)

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  ❌  Could not read {rel_path}: {e}")
            continue

        video_key = list(data.keys())[0]
        questions = data[video_key]

        valid_questions = []
        files_modified = False

        for q in questions:
            ts_segments = q.get("timestamp_segments", [])
            pipeline_mode = q.get("annotations", {}).get("pipeline_mode", "")

            if len(ts_segments) == 0:
                print(f"  🗑️  [PREVIEW] {rel_path}: Will remove query_id '{q.get('query_id')}' (Empty timestamps)")
                total_empty_removed += 1
                files_modified = True
                continue

            if len(ts_segments) > 1 and pipeline_mode != "time":
                print(f"  🗑️  [PREVIEW] {rel_path}: Will remove query_id '{q.get('query_id')}' (Multi-ts, mode: '{pipeline_mode}')")
                total_multi_removed += 1
                files_modified = True
                continue

            valid_questions.append(q)

        if files_modified:
            data[video_key] = valid_questions
            pending_writes[filepath] = data
            total_files_to_modify += 1

    total_to_remove = total_empty_removed + total_multi_removed

    print("\n" + "=" * 70)
    print("                   PHASE 3 — PROPOSED CHANGES")
    print("=" * 70)
    print(f"  Files to be modified                     : {total_files_to_modify}")
    print(f"  Questions to remove (Empty timestamps)   : {total_empty_removed}")
    print(f"  Questions to remove (Invalid multi-ts)   : {total_multi_removed}")
    print(f"  Total questions to remove                : {total_to_remove}")
    print("=" * 70)

    if total_files_to_modify == 0:
        print("\n✅ No structural timestamp issues found. Nothing to do.")
        return

    confirm = input("\n⚠️  Do you want to proceed with these deletions? (y/n): ").strip().lower()

    if confirm == 'y':
        print("\n💾 Committing changes to disk...")
        for filepath, new_data in pending_writes.items():
            rel_path = os.path.relpath(filepath, QUERIES_ROOT)
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(new_data, fh, indent=2, ensure_ascii=False)
            print(f"  ✅  Updated: {rel_path}")
        print("\n🎉 Phase 3 Complete.")
    else:
        print("\n🚫 Mission aborted. No files were modified.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parallel query cleanup — detect & remove bad questions.",
        epilog=__doc__,
    )
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                   help="Phase to run: 1=detect, 2=clean, 3=timestamps.")
    p.add_argument("--max_workers", type=int, default=10,
                   help="Number of parallel Azure calls for Phase 1 (default: 10).")
    p.add_argument("--queries_root", default=QUERIES_ROOT,
                   help=f"Root of query folders (default: {QUERIES_ROOT}).")
    p.add_argument("--manifest_path", default=MANIFEST_PATH,
                   help=f"Manifest file path (default: {MANIFEST_PATH}).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow CLI overrides of globals
    QUERIES_ROOT = args.queries_root
    MANIFEST_PATH = args.manifest_path

    if args.phase == 1:
        print(f"🚀 Running PHASE 1: DETECT (parallel, {args.max_workers} workers)\n")
        phase_detect(max_workers=args.max_workers)
    elif args.phase == 2:
        print("🚀 Running PHASE 2: CLEAN\n")
        phase_clean()
    elif args.phase == 3:
        print("🚀 Running PHASE 3: CLEAN TIMESTAMPS\n")
        phase_clean_timestamps()
