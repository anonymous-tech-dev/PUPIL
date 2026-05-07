"""
stages/stage4_answering.py — Stage 4: Answering (H function, Eq. 3).

Given the curated context c = ˆx ∪ x[u] assembled by Stage 3, pass it
to the VLM to produce the final answer.

Paper Eq. 3:  a = H(c, q) = f(c, q)
Paper Fig. 14/15/16: answering prompts.
"""

import logging
from typing import List, Dict, Any, Optional

from stages.stage0_video_loading import FrameBundle, get_frame_images, get_frame_ids
from stages.stage1_prompts import build_answering_prompt
from stages.stage2_selection_parsing import extract_answer_letter
import config

logger = logging.getLogger(__name__)


def answer_question(
    model,
    context_bundle  : FrameBundle,
    question        : str,
    answer_choices  : List[str],
    style           : str = None,
) -> Dict[str, Any]:
    """
    Run the answering call H(c, q) (Eq. 3 in paper).

    Args:
        model           : loaded VLM (BaseVLM subclass)
        context_bundle  : curated context frames c
        question        : QA question string
        answer_choices  : list of answer option strings (empty → open-ended)
        style           : "gpt" | "qwen" → selects answer prompt template.
                          Auto-detected from config.MODEL if None.

    Returns:
        {
          "raw_response"    : str,   full text output from the VLM
          "predicted_letter": str,   extracted letter (A/B/C/D/E) or ""
          "frame_ids_used"  : List[int],
        }
    """
    if style is None:
        style = "qwen" if "qwen" in config.MODEL.lower() else "gpt"

    frame_ids = get_frame_ids(context_bundle)
    images    = get_frame_images(context_bundle)

    prompt = build_answering_prompt(
        frame_ids=frame_ids,
        question=question,
        answer_choices=answer_choices,
        style=style,
    )

    logger.debug("Answering with %d frames.", len(images))
    raw = model.call_answering(images, prompt)

    predicted_letter = ""
    if answer_choices:
        predicted_letter = extract_answer_letter(raw)

    return {
        "raw_response"    : raw,
        "predicted_letter": predicted_letter,
        "frame_ids_used"  : frame_ids,
    }