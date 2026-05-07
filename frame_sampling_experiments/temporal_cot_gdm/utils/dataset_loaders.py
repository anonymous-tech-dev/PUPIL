"""
utils/dataset_loaders.py — Dataset Loaders for Egoschema and LVBench.

Each loader yields dicts with a standardised schema:
  {
    "uid"           : str,          unique question ID
    "video_path"    : str,          absolute path to .mp4
    "question"      : str,
    "answer_choices": List[str],    option strings (no letter prefix)
    "ground_truth"  : str,          correct letter "A"/"B"/... (if available)
    "question_type" : List[str],    (LVBench only)
    "time_reference": str,          (LVBench only)
  }
"""

import json
import os
from typing import Iterator, Dict, Any, Optional

import config


# ─── Egoschema ────────────────────────────────────────────────────────────────

def load_egoschema(
    questions_path : str = None,
    answers_path   : str = None,
    video_dir      : str = None,
    num_samples    : int = -1,
) -> Iterator[Dict[str, Any]]:
    """
    Yield egoschema items.

    Questions format:
      [{"q_uid": "...", "google_drive_id": "...", "question": "...",
        "option 0": "...", ..., "option 4": "..."}, ...]

    Answers format:
      {"q_uid": answer_index (0-4), ...}

    Video path: <video_dir>/<q_uid>_clean.mp4
    """
    questions_path = questions_path or config.EGOSCHEMA_QUESTIONS
    answers_path   = answers_path   or config.EGOSCHEMA_ANSWERS
    video_dir      = video_dir      or config.EGOSCHEMA_VIDEO_DIR

    with open(questions_path, "r") as f:
        questions = json.load(f)

    with open(answers_path, "r") as f:
        answers = json.load(f)

    letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    count = 0
    for item in questions:
        if num_samples != -1 and count >= num_samples:
            break

        uid = item["q_uid"]
        video_path = os.path.join(video_dir, f"{uid}_clean.mp4")

        if not os.path.exists(video_path):
            # Try without _clean suffix
            alt = os.path.join(video_dir, f"{uid}.mp4")
            if os.path.exists(alt):
                video_path = alt
            else:
                print(f"[Egoschema] Warning: video not found for uid={uid}, skipping.")
                continue

        # Collect options in order
        choices = []
        for i in range(5):
            key = f"option {i}"
            if key in item:
                choices.append(item[key])

        gt_idx = answers.get(uid)
        gt_letter = letter_map.get(gt_idx, "") if gt_idx is not None else ""

        yield {
            "uid"           : uid,
            "video_path"    : video_path,
            "question"      : item["question"],
            "answer_choices": choices,
            "ground_truth"  : gt_letter,
            "question_type" : [],
            "time_reference": "",
        }
        count += 1


# ─── LVBench ──────────────────────────────────────────────────────────────────

def _lvbench_paths(version: str):
    """Return (meta_path, video_dir) for 'lvbench_v1' or 'lvbench_v2'."""
    if version == "lvbench_v1":
        return config.LVBENCH_V1_META, config.LVBENCH_V1_VIDEO_DIR
    elif version == "lvbench_v2":
        return config.LVBENCH_V2_META, config.LVBENCH_V2_VIDEO_DIR
    else:
        raise ValueError(f"Unknown lvbench version: {version!r}. "
                         "Choose 'lvbench_v1' or 'lvbench_v2'.")


def load_lvbench(
    meta_path   : str = None,
    video_dir   : str = None,
    num_samples : int = -1,
    version     : str = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield LVBench items.

    Version is resolved in this order:
      1. Explicit meta_path / video_dir arguments (override everything).
      2. `version` argument (e.g. 'lvbench_v1').
      3. config.DATASET (e.g. 'lvbench_v2').

    Meta format (JSONL, one JSON object per line):
      {"key": "video_key", "qa": [
        {"uid": "...", "question": "...(A)...(B)...(C)...(D)...",
         "answer": "A", "question_type": [...], "time_reference": "..."},
        ...
      ], "video_info": {...}}

    Video path: <video_dir>/<key>.mp4
    """
    if meta_path is None or video_dir is None:
        v = version or config.DATASET   # e.g. "lvbench_v1"
        auto_meta, auto_video_dir = _lvbench_paths(v)
        meta_path = meta_path or auto_meta
        video_dir = video_dir or auto_video_dir

    count = 0
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key        = entry["key"]
            video_path = os.path.join(video_dir, f"{key}_clean.mp4")

            if not os.path.exists(video_path):
                print(f"[LVBench] Warning: video not found for key={key}, skipping.")
                continue

            for qa in entry.get("qa", []):
                if num_samples != -1 and count >= num_samples:
                    return

                uid          = qa["uid"]
                raw_question = qa["question"]
                gt_letter    = qa.get("answer", "")

                # LVBench embeds choices in the question string:
                # "Question text\n(A) ...\n(B) ...\n(C) ...\n(D) ..."
                question_text, choices = _parse_lvbench_question(raw_question)

                yield {
                    "uid"           : uid,
                    "video_path"    : video_path,
                    "question"      : question_text,
                    "answer_choices": choices,
                    "ground_truth"  : gt_letter,
                    "question_type" : qa.get("question_type", []),
                    "time_reference": qa.get("time_reference", ""),
                }
                count += 1


def _parse_lvbench_question(raw: str):
    """
    Split LVBench question string into (question_text, [choice_A, choice_B, ...]).
    Format: 'Question...\n(A) text\n(B) text\n(C) text\n(D) text'
    """
    import re
    # Try to split on (A) / (B) / (C) / (D) markers
    parts = re.split(r"\n?\(([A-E])\)\s*", raw)
    if len(parts) < 2:
        # No choices found embedded — return as-is with empty choices
        return raw.strip(), []

    question_text = parts[0].strip()
    choices = []
    # parts is now: [question, letter, text, letter, text, ...]
    i = 1
    while i + 1 < len(parts):
        choices.append(parts[i + 1].strip())
        i += 2

    return question_text, choices