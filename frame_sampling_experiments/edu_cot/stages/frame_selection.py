"""
stages/frame_selection.py — VLM-based frame selection (TCoT-style).

Shows the VLM a set of frames and asks it to pick the ones most relevant
to the question.  Returns the selected frame IDs + justification.

Uses the same prompt structure as the DeepMind TCoT paper (Fig. 3).
"""

import json
import logging
import re
from typing import Dict, Any, List, Tuple

from PIL import Image
from omegaconf import DictConfig

logger = logging.getLogger("educot.selection")

FrameBundle = List[Tuple[int, Image.Image]]


# ─── Selection prompt (TCoT paper Fig. 3, Qwen variant) ──────────────────

SELECTION_PROMPT = (
    "You will be given a question about a video and {num_choices} possible "
    "answer options.\n"
    "{frame_labels}\n"
    "Question: {question}\n"
    "Possible answer choices: {answer_choices}\n"
    "Return the frame ids which are most relevant to answering the given "
    "question. You MUST select at least one frame. If no frames are clearly "
    "relevant, select the frames most likely to contain useful information.\n"
    "Respond with ONLY a JSON object and nothing else. Example format:\n"
    '{{\"frame_ids\": [1, 5, 12], \"justification\": \"These frames show '
    'the relevant action.\"}}'
)


def _build_selection_prompt(
    frame_ids: List[int],
    question: str,
    answer_choices: List[str],
) -> str:
    frame_labels = "\n".join(f"FrameID {fid}: <image>" for fid in frame_ids)
    letters = "ABCDE"
    choices_str = " ".join(
        f"({letters[i]}) {c}" for i, c in enumerate(answer_choices)
    )
    return SELECTION_PROMPT.format(
        num_choices=len(answer_choices),
        frame_labels=frame_labels,
        question=question,
        answer_choices=choices_str,
    )


# ─── Response parsing ────────────────────────────────────────────────────

def _parse_selection(
    raw: str,
    valid_ids: List[int],
) -> Tuple[List[int], str]:
    """Parse VLM JSON output → (selected_ids, justification)."""
    valid_set = set(valid_ids)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            ids = data.get("frame_ids", [])
            justification = data.get("justification", "")
            selected = sorted(int(i) for i in ids if int(i) in valid_set)
            if selected:
                return selected, justification
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: return all frames (parse failure)
    logger.warning("  Selection parse failed — using all %d frames", len(valid_ids))
    return valid_ids, "parse_fallback"


# ─── Public API ───────────────────────────────────────────────────────────

def vlm_selection_call(
    model,
    frames: FrameBundle,
    question: str,
    answer_choices: List[str],
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Run one VLM selection call on a set of frames.

    Returns:
        {
          "selected_ids"  : List[int],
          "justification" : str,
          "raw_response"  : str,
        }
    """
    frame_ids = [fid for fid, _ in frames]
    images = [img for _, img in frames]

    prompt = _build_selection_prompt(frame_ids, question, answer_choices)
    raw = model.call_selection(images, prompt)
    selected_ids, justification = _parse_selection(raw, frame_ids)

    return {
        "selected_ids": selected_ids,
        "justification": justification,
        "raw_response": raw,
    }
