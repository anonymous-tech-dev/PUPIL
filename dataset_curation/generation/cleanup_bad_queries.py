"""

Query Cleanup Script — Detect & Remove Bad (Failed-Generation) Questions
=========================================================================
Phase 1 (DETECT): Uses GPT-5 via Azure to classify every question in each
    *_queries.json file under final_1k/ as GOOD or BAD. Outputs a JSON
    manifest of files that need cleaning plus files where ALL questions
    are bad.
Phase 2 (CLEAN): Reads the manifest produced by Phase 1 and removes bad
    questions from the JSON files (in-place). Files where ALL questions
    are bad are left untouched.
Phase 3 (TIMESTAMPS): Structurally cleans the dataset (no LLM required). 
    Removes questions that have missing (empty) timestamp segments OR 
    multiple timestamp segments when the pipeline_mode is NOT "time".

Set the PHASE knob below to 1, 2, or 3.
"""

import os
import sys
import json
import glob
import time
import traceback
from datetime import datetime

from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                             KNOBS                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

PHASE = 2  # 1 = detect, 2 = clean (LLM based), 3 = clean timestamps (Structural)

# Path to the root of the query folders
QUERIES_ROOT = "/workspace/Pupil/dataset_curation/dataset/queries_db/final_train"

# Sub-folders to scan (set to None to scan ALL sub-folders)
SUB_FOLDERS = ["sof_visual", "sof_audio", "sof_priority", "sof_time"]

# Manifest file (Phase 1 writes it, Phase 2 reads it)
MANIFEST_PATH = "/workspace/Pupil/dataset_curation/cleanup_manifest.json"

# LLM settings
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

            # Parse the JSON array from the response
            # Handle cases where the LLM wraps it in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]  # drop first line
                raw = raw.rsplit("```", 1)[0]  # drop trailing ```
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
# ║                            PHASE 1: DETECT                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def phase_detect():
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

    print(f"📂 Found {len(all_query_files)} query files across {len(folders)} folders.\n")

    # Results
    files_to_clean = {}      # filepath -> list of bad question indices (0-based)
    all_bad_files = []        # files where ALL questions are bad
    total_bad_detected = 0
    total_questions_seen = 0

    for idx, filepath in enumerate(all_query_files, 1):
        rel_path = os.path.relpath(filepath, QUERIES_ROOT)
        print(f"[{idx}/{len(all_query_files)}] Checking: {rel_path}")

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  ❌  Could not read file: {e}")
            continue

        # Each file has one key (video path) mapping to a list of questions
        video_key = list(data.keys())[0]
        questions = data[video_key]
        total_questions_seen += len(questions)

        if len(questions) == 0:
            print(f"  ⏭️  Empty question list, skipping.")
            continue

        # Classify
        bad_flags = classify_questions(client, questions)
        bad_indices = [i for i, is_bad in enumerate(bad_flags) if is_bad]
        good_count = len(questions) - len(bad_indices)
        bad_count = len(bad_indices)

        if bad_count == 0:
            print(f"  ✅  All {len(questions)} questions are GOOD.")
            continue

        if bad_count == len(questions):
            # ALL are bad — don't remove, just record
            all_bad_files.append(filepath)
            total_bad_detected += bad_count
            print(f"  🔴  ALL {len(questions)} questions are BAD — will NOT remove (flagged).")
        else:
            # Mix of good and bad — record for cleaning
            files_to_clean[filepath] = bad_indices
            total_bad_detected += bad_count
            print(f"  🟡  {bad_count} BAD / {good_count} GOOD — marked for cleaning (indices: {bad_indices}).")

    # Write manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_files_scanned": len(all_query_files),
        "total_questions_seen": total_questions_seen,
        "total_bad_detected": total_bad_detected,
        "files_to_clean": files_to_clean,        # filepath -> [bad_indices]
        "all_bad_files": all_bad_files,           # files where all questions are bad
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("                     PHASE 1 — DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total files scanned      : {len(all_query_files)}")
    print(f"  Total questions seen      : {total_questions_seen}")
    print(f"  Total BAD questions found : {total_bad_detected}")
    print(f"  Files needing cleanup     : {len(files_to_clean)}  (mixed good/bad)")
    print(f"  Files ALL bad (untouched) : {len(all_bad_files)}")
    print(f"  Manifest saved to         : {MANIFEST_PATH}")

    if files_to_clean:
        print(f"\n  📋 Files to clean (bad indices):")
        for fp, indices in files_to_clean.items():
            print(f"     {os.path.relpath(fp, QUERIES_ROOT)}  →  bad indices: {indices}")

    if all_bad_files:
        print(f"\n  🔴 Files where ALL questions are bad (will NOT be modified):")
        for fp in all_bad_files:
            print(f"     {os.path.relpath(fp, QUERIES_ROOT)}")

    print("=" * 70)

    # Also print the arrays for easy copy-paste
    print("\n\n# ---- COPY-PASTEABLE ARRAYS ----")
    print(f"files_to_clean = {json.dumps(files_to_clean, indent=2)}")
    print(f"\nall_bad_files = {json.dumps(all_bad_files, indent=2)}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                            PHASE 2: CLEAN                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def phase_clean():
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌  Manifest not found at {MANIFEST_PATH}.")
        print("   Run Phase 1 first (set PHASE = 1).")
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

        # Remove bad indices (sort descending so indices stay valid)
        for i in sorted(bad_indices, reverse=True):
            if i < len(questions):
                removed_q = questions.pop(i)
                total_deleted += 1
                print(f"  🗑️  {rel_path}: removed index {i} — \"{removed_q['question'][:80]}...\"")

        # Write back
        data[video_key] = questions
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        print(f"  ✅  {rel_path}: {original_count} → {len(questions)} questions\n")

    # Summary
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
    """
    Scans files, detects timestamp conditions, previews changes, and asks for confirmation:
    1. Empty timestamp_segments
    2. Multiple timestamp_segments when pipeline_mode is not "time"
    """
    print(f"🧹 Phase 3: Analyzing structural timestamp issues...\n")

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

    total_files_to_modify = 0
    total_empty_removed = 0
    total_multi_removed = 0

    # Dictionary to buffer our proposed writes: filepath -> updated_data
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
            
            # Rule 1: Empty timestamps
            if len(ts_segments) == 0:
                print(f"  🗑️  [PREVIEW] {rel_path}: Will remove query_id '{q.get('query_id')}' (Empty timestamps)")
                total_empty_removed += 1
                files_modified = True
                continue
                
            # Rule 2: Multiple timestamps but NOT time mode
            if len(ts_segments) > 1 and pipeline_mode != "time":
                print(f"  🗑️  [PREVIEW] {rel_path}: Will remove query_id '{q.get('query_id')}' (Multi-ts, mode: '{pipeline_mode}')")
                total_multi_removed += 1
                files_modified = True
                continue

            # Keep the question if it passes rules
            valid_questions.append(q)

        # Buffer changes if modifications were detected
        if files_modified:
            data[video_key] = valid_questions
            pending_writes[filepath] = data
            total_files_to_modify += 1

    total_to_remove = total_empty_removed + total_multi_removed

    # Print Summary of Planned Changes
    print("\n" + "=" * 70)
    print("                   PHASE 3 — PROPOSED CHANGES")
    print("=" * 70)
    print(f"  Files to be modified                     : {total_files_to_modify}")
    print(f"  Questions to remove (Empty timestamps)   : {total_empty_removed}")
    print(f"  Questions to remove (Invalid multi-ts)   : {total_multi_removed}")
    print(f"  Total questions to remove                : {total_to_remove}")
    print("=" * 70)

    # Fast exit if everything is already clean
    if total_files_to_modify == 0:
        print("\n✅ No structural timestamp issues found. Nothing to do.")
        return

    # Trigger the y/n Prompt
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

if __name__ == "__main__":
    if PHASE == 1:
        print("🚀 Running PHASE 1: DETECT\n")
        phase_detect()
    elif PHASE == 2:
        print("🚀 Running PHASE 2: CLEAN\n")
        phase_clean()
    elif PHASE == 3:
        print("🚀 Running PHASE 3: CLEAN TIMESTAMPS\n")
        phase_clean_timestamps()
    else:
        print(f"❌  Invalid PHASE value: {PHASE}. Set PHASE to 1, 2, or 3.")
        sys.exit(1)