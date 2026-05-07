"""
stages/answering.py — Final VLM answering call  H(c, q).

Prompts follow the DeepMind TCoT paper:
  - Fig. 15  — Multiple-choice (Qwen style)
  - Fig. 16  — Open-ended
"""

import logging
import re
from typing import Dict, Any, List, Tuple

from PIL import Image

logger = logging.getLogger("educot.answering")

FrameBundle = List[Tuple[int, Image.Image]]


# ─── Prompt templates ────────────────────────────────────────────────────

ANSWERING_MC = (
    "Frames: {frame_label_list}\n\n"
    "Carefully watch the video and pay attention to the cause and sequence "
    "of events, the detail and movement of objects and the action and pose "
    "of persons. Based on your observations, select the best option that "
    "accurately addresses the question.\n"
    "Question: {question}\n"
    "Options: {answer_choices}\n"
    "Answer with the option's letter from the given choices directly and "
    "only give the best option."
)

ANSWERING_OPEN = (
    "Frames: {frame_label_list}\n\n"
    "You will be given a question about a video. You will be provided "
    "frames from the video, retrieved by an intelligent agent. It is "
    "crucial that you imagine the visual scene as vividly as possible to "
    "enhance the accuracy of your response.\n\n"
    "Question: {question}"
)


# ─── Answer extraction ───────────────────────────────────────────────────

def extract_answer_letter(raw: str) -> str:
    """Extract a single answer letter (A-E) from raw VLM output."""
    raw = raw.strip()
    # Bare letter
    if raw and raw[0] in "ABCDE":
        return raw[0]
    # Parenthesised letter
    m = re.search(r"\(([A-E])\)", raw)
    if m:
        return m.group(1)
    # "answer is X" / "option X"
    m = re.search(
        r"(?:answer|option)\s*(?:is\s*)?[:\s]*\(?([A-E])\)?",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    # Last resort: first letter found
    for ch in raw:
        if ch in "ABCDE":
            return ch
    return ""


# ─── Public API ───────────────────────────────────────────────────────────

def answer_question(
    model,
    context_bundle: FrameBundle,
    question: str,
    answer_choices: List[str],
) -> Dict[str, Any]:
    """Run the answering call and return predicted letter + raw output."""
    frame_ids = [fid for fid, _ in context_bundle]
    images = [img for _, img in context_bundle]

    frame_label_list = ", ".join(str(fid) for fid in frame_ids)

    if answer_choices:
        letters = "ABCDE"
        choices_str = " ".join(
            f"({letters[i]}) {c}" for i, c in enumerate(answer_choices)
        )
        prompt = ANSWERING_MC.format(
            frame_label_list=frame_label_list,
            question=question,
            answer_choices=choices_str,
        )
    else:
        prompt = ANSWERING_OPEN.format(
            frame_label_list=frame_label_list,
            question=question,
        )

    raw = model.call_answering(images, prompt)
    predicted = extract_answer_letter(raw) if answer_choices else ""

    return {
        "raw_response": raw,
        "predicted_letter": predicted,
        "frame_ids_used": frame_ids,
    }
