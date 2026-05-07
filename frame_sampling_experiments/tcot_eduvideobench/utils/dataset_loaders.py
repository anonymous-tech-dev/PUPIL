"""
utils/dataset_loaders.py — Dataset loaders for TCoT-Pupil.

Includes Egoschema, LVBench (legacy) and Pupil (open-ended).

All loaders yield dicts with the standard schema:
  {
    "uid"           : str,
    "video_path"    : str,
    "question"      : str,
    "answer_choices": List[str],   # empty list for open-ended
    "ground_truth"  : str,         # letter or free-form text
    "question_type" : List[str],
    "time_reference": str,
  }
"""

import json
import os
import re
from typing import Iterator, Dict, Any

import config


# ─── Egoschema (legacy) ───────────────────────────────────────────────────────

def load_egoschema(num_samples: int = -1) -> Iterator[Dict[str, Any]]:
    questions_path = config.EGOSCHEMA_QUESTIONS
    answers_path   = config.EGOSCHEMA_ANSWERS
    video_dir      = config.EGOSCHEMA_VIDEO_DIR
    if not (questions_path and os.path.exists(questions_path)):
        return
    with open(questions_path) as f:
        questions = json.load(f)
    with open(answers_path) as f:
        answers = json.load(f)
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    count = 0
    for item in questions:
        if num_samples != -1 and count >= num_samples:
            break
        uid = item["q_uid"]
        video_path = os.path.join(video_dir, f"{uid}_clean.mp4")
        if not os.path.exists(video_path):
            alt = os.path.join(video_dir, f"{uid}.mp4")
            if not os.path.exists(alt):
                print(f"[Egoschema] Warning: video not found for uid={uid}, skipping.")
                continue
            video_path = alt
        choices = []
        for i in range(5):
            key = f"option {i}"
            if key in item:
                choices.append(item[key])
        gt_idx = answers.get(uid)
        gt_letter = letter_map.get(gt_idx, "") if gt_idx is not None else ""
        yield {
            "uid": uid,
            "video_path": video_path,
            "question": item["question"],
            "answer_choices": choices,
            "ground_truth": gt_letter,
            "question_type": [],
            "time_reference": "",
        }
        count += 1


# ─── LVBench (legacy) ─────────────────────────────────────────────────────────

def _parse_lvbench_question(raw: str):
    parts = re.split(r"\n?\(([A-E])\)\s*", raw)
    if len(parts) < 2:
        return raw.strip(), []
    question_text = parts[0].strip()
    choices = []
    i = 1
    while i + 1 < len(parts):
        choices.append(parts[i + 1].strip())
        i += 2
    return question_text, choices


def _lvbench_paths(version: str):
    if version == "lvbench_v1":
        return config.LVBENCH_V1_META, config.LVBENCH_V1_VIDEO_DIR
    elif version == "lvbench_v2":
        return config.LVBENCH_V2_META, config.LVBENCH_V2_VIDEO_DIR
    raise ValueError(f"Unknown lvbench version: {version!r}. "
                     "Choose 'lvbench_v1' or 'lvbench_v2'.")


def load_lvbench(num_samples: int = -1, version: str = None) -> Iterator[Dict[str, Any]]:
    v = version or config.DATASET
    meta_path, video_dir = _lvbench_paths(v)
    count = 0
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = entry["key"]
            video_path = os.path.join(video_dir, f"{key}_clean.mp4")
            if not os.path.exists(video_path):
                print(f"[LVBench] Warning: video not found for key={key}, skipping.")
                continue
            for qa in entry.get("qa", []):
                if num_samples != -1 and count >= num_samples:
                    return
                question_text, choices = _parse_lvbench_question(qa["question"])
                yield {
                    "uid": qa["uid"],
                    "video_path": video_path,
                    "question": question_text,
                    "answer_choices": choices,
                    "ground_truth": qa.get("answer", ""),
                    "question_type": qa.get("question_type", []),
                    "time_reference": qa.get("time_reference", ""),
                }
                count += 1


# ─── Pupil (open-ended JSONL) ─────────────────────────────────────────

def load_Pupil(num_samples: int = -1) -> Iterator[Dict[str, Any]]:
    """
    Load Pupil final_1k JSONL dataset for open-ended evaluation.

    JSONL line schema (per `final_1k_for_cot.jsonl`):
      {
        "uid"            : str,
        "video"          : str (basename),
        "question"       : str,
        "answer_choices" : [],         # always [] for open-ended
        "ground_truth"   : str,        # free-form text
        "question_type"  : [str],
        "time_reference" : str,
        "source_of_fact" : str,
        "category"       : str
      }
    """
    meta_path = config.Pupil_META
    video_dir = config.Pupil_VIDEO_DIR
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Pupil meta not found: {meta_path!r}")

    count = 0
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            video_name = item.get("video", "")
            video_path = (video_name if os.path.isabs(video_name)
                          else os.path.join(video_dir, video_name))
            if not os.path.exists(video_path):
                print(f"[Pupil] Warning: video not found: {video_path}, skipping.")
                continue
            yield {
                "uid": str(item.get("uid", count)),
                "video_path": video_path,
                "question": item.get("question", ""),
                "answer_choices": item.get("answer_choices", []),
                "ground_truth": item.get("ground_truth", ""),
                "question_type": item.get("question_type", []),
                "time_reference": item.get("time_reference", ""),
                "source_of_fact": item.get("source_of_fact", "unknown"),
                "category": item.get("category", "unknown"),
            }
            count += 1
            if num_samples != -1 and count >= num_samples:
                return
