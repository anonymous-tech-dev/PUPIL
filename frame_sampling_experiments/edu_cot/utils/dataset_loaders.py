"""
utils/dataset_loaders.py — Dataset iterators for LVBench and Pupil.

Each iterator yields dicts with a standard schema:
  {
    "uid"            : str,
    "video_path"     : str,
    "question"       : str,
    "answer_choices" : List[str],     # empty for open-ended
    "ground_truth"   : str,           # letter A-E or free-form
    "question_type"  : List[str],
    "time_reference" : str,
  }
"""

import json
import logging
import os
import re
from typing import Dict, Any, Iterator

from omegaconf import DictConfig

logger = logging.getLogger("educot.data")


# ─── LVBench question parser ─────────────────────────────────────────────

def _parse_lvbench_question(raw: str):
    """
    Split LVBench question string into (question_text, [choice_A, choice_B, ...]).

    LVBench embeds answer choices directly in the question string:
      "Question text\\n(A) choice1\\n(B) choice2\\n(C) choice3\\n(D) choice4"

    Returns (question_text, choices_list).  If no embedded choices are
    found, returns (raw_text, []).
    """
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


# ─── LVBench ─────────────────────────────────────────────────────────────

def load_lvbench(cfg: DictConfig) -> Iterator[Dict[str, Any]]:
    meta_file = cfg.dataset.lvbench.meta_file
    video_dir = cfg.dataset.lvbench.video_dir
    n = cfg.num_samples

    with open(meta_file) as f:
        items = [json.loads(line) for line in f]

    count = 0
    for item in items:
        key = item.get("key", item.get("video_id", ""))
        video_path = os.path.join(video_dir, f"{key}_clean.mp4")

        if not os.path.exists(video_path):
            # Fall back to without _clean suffix
            video_path = os.path.join(video_dir, f"{key}.mp4")

        if not os.path.exists(video_path):
            logger.warning("[LVBench] Video not found: key=%s, skipping.", key)
            continue

        for qa in item.get("qa", []):
            uid = str(qa.get("uid", qa.get("id", f"{key}_{count}")))
            raw_question = qa.get("question", "")
            gt_letter = qa.get("answer", "")

            # LVBench embeds choices in the question string:
            # "Question text\n(A) ...\n(B) ...\n(C) ...\n(D) ..."
            question_text, choices = _parse_lvbench_question(raw_question)

            yield {
                "uid": uid,
                "video_path": video_path,
                "question": question_text,
                "answer_choices": choices,
                "ground_truth": gt_letter,
                "question_type": qa.get("question_type", qa.get("task_type", [])),
                "time_reference": qa.get("time_reference", ""),
            }

            count += 1
            if 0 < n <= count:
                return


# ─── Pupil ────────────────────────────────────────────────────────

def load_Pupil(cfg: DictConfig) -> Iterator[Dict[str, Any]]:
    """
    Generic JSONL loader for Pupil.

    Expected fields per line:
      uid, video (or video_path), question, answer_choices (or choices),
      ground_truth (or answer), question_type, time_reference
    """
    meta_file = cfg.dataset.Pupil.meta_file
    video_dir = cfg.dataset.Pupil.video_dir

    if not meta_file or not os.path.exists(meta_file):
        raise FileNotFoundError(
            f"Pupil metadata not found: {meta_file!r}. "
            "Set dataset.Pupil.meta_file in config."
        )

    n = cfg.num_samples
    count = 0

    with open(meta_file) as f:
        items = [json.loads(line) for line in f]

    for item in items:
        video_name = item.get("video", item.get("video_path", ""))
        if not os.path.isabs(video_name):
            video_path = os.path.join(video_dir, video_name)
        else:
            video_path = video_name

        if not os.path.exists(video_path):
            logger.warning("[Pupil] Video not found: %s", video_path)
            continue

        yield {
            "uid": str(item.get("uid", item.get("id", count))),
            "video_path": video_path,
            "question": item.get("question", ""),
            "answer_choices": item.get("answer_choices", item.get("choices", [])),
            "ground_truth": item.get("ground_truth", item.get("answer", "")),
            "question_type": item.get("question_type", []),
            "time_reference": item.get("time_reference", ""),
        }

        count += 1
        if 0 < n <= count:
            return


# ─── Dispatcher ───────────────────────────────────────────────────────────

def get_dataset_iterator(cfg: DictConfig) -> Iterator[Dict[str, Any]]:
    name = cfg.dataset.name.lower()
    if "lvbench" in name:
        return load_lvbench(cfg)
    elif "eduvideo" in name or "edu" in name:
        return load_Pupil(cfg)
    else:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset.name!r}. "
            "Choose 'lvbench_v2' or 'Pupil'."
        )
