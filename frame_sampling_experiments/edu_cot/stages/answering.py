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

# Per-option elimination style: forces the model to engage every option
# before committing, breaking the positional letter-bias that favors C.
ANSWERING_MC_PEROPT = (
    "Frames: {frame_label_list}\n\n"
    "Carefully watch the video and pay attention to the cause and sequence "
    "of events, the detail and movement of objects and the action and pose "
    "of persons.\n"
    "Question: {question}\n\n"
    "Options:\n{answer_choices}\n\n"
    "For EACH option (A, B, C, D), state in one short sentence whether the "
    "video evidence SUPPORTS, CONTRADICTS, or is SILENT on that option. "
    "Consider every option independently and do not skip any.\n"
    "After evaluating all options, output your final choice on a new line "
    "in EXACTLY this format: ANSWER: <letter>"
)

# v2 — universal answer-only prompt with two general nudges:
#   (1) anti-rush / coverage: evidence may appear briefly across the video
#   (2) anti-default: compare partially-supported options instead of
#       defaulting to the most familiar-looking one.
# Output format unchanged (single letter), so latency/parsing match v1.
ANSWERING_MC_V2 = (
    "Frames: {frame_label_list}\n\n"
    "Carefully watch the video and pay attention to the cause and sequence "
    "of events, the detail and movement of objects and the action and pose "
    "of persons. Evidence relevant to a choice may appear only briefly or "
    "in a few moments of the video, so consider the full set of moments "
    "shown before deciding. If more than one choice seems partially "
    "supported, compare them against each other rather than defaulting to "
    "the one that feels most familiar.\n"
    "Question: {question}\n"
    "Options: {answer_choices}\n"
    "Answer with the option's letter from the given choices directly and "
    "only give the best option."
)

# v3 — minimal grounding: one short evidence sentence, then the letter.
# Forces the model to commit to *what* it saw before *which* option, but
# without the heavy per-option enumeration that destroyed PEROPT.
ANSWERING_MC_V3 = (
    "Frames: {frame_label_list}\n\n"
    "Carefully watch the video and pay attention to the cause and sequence "
    "of events, the detail and movement of objects and the action and pose "
    "of persons. Based on your observations, select the best option that "
    "accurately addresses the question.\n"
    "Question: {question}\n"
    "Options: {answer_choices}\n\n"
    "Respond in EXACTLY two lines and nothing else:\n"
    "Line 1 — one short sentence (<= 25 words) describing the specific "
    "visual evidence that decides the answer.\n"
    "Line 2 — your final choice in EXACTLY this format: ANSWER: <letter>\n"
    "You MUST include Line 2. Do not omit it under any circumstances."
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

def extract_answer_letter(raw: str, answer_choices: List[str] = None) -> str:
    """Extract a single answer letter (A-E) from raw VLM output.

    If a letter cannot be parsed and `answer_choices` is supplied, fall
    back to substring-matching choice text inside the raw response.
    Useful for prompts that ask for an explanation alongside the letter
    (e.g. v3) where the model may forget the letter line but still name
    the correct choice content.
    """
    raw = raw.strip()
    # Highest priority: explicit "ANSWER: X" marker (per-option prompt format).
    # Use the LAST match so the final committed letter wins over any
    # per-option enumeration that may also contain the marker text.
    matches = list(re.finditer(
        r"ANSWER\s*[:\-]?\s*\(?([A-E])\)?",
        raw, re.IGNORECASE,
    ))
    if matches:
        return matches[-1].group(1).upper()
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
    # Content-match fallback: look for any answer-choice text inside raw.
    # Pick the longest unique match to avoid false positives on short
    # numeric overlaps (e.g. choice "3" matching "1633").
    if answer_choices:
        letters = "ABCDE"
        hits = []
        raw_low = raw.lower()
        for i, c in enumerate(answer_choices):
            ct = c.strip().lower()
            if not ct:
                continue
            # Word-boundary search so "3" doesn't match inside "1633".
            pat = r"(?<![A-Za-z0-9])" + re.escape(ct) + r"(?![A-Za-z0-9])"
            if re.search(pat, raw_low):
                hits.append((len(ct), i))
        if hits:
            # Prefer longest match; break ties by first occurrence.
            hits.sort(key=lambda t: (-t[0], t[1]))
            return letters[hits[0][1]]
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
    prompt_style: str = "direct",
) -> Dict[str, Any]:
    """Run the answering call and return predicted letter + raw output.

    prompt_style:
      "direct"   — original single-letter answer prompt (TCoT style).
      "per_option" — forces the model to evaluate every option before
                     committing; breaks positional letter-bias.
    """
    frame_ids = [fid for fid, _ in context_bundle]
    images = [img for _, img in context_bundle]

    frame_label_list = ", ".join(str(fid) for fid in frame_ids)

    if answer_choices:
        letters = "ABCDE"
        if prompt_style == "per_option":
            # Render each choice on its own line for clearer per-option
            # enumeration in the model's response.
            choices_str = "\n".join(
                f"({letters[i]}) {c}" for i, c in enumerate(answer_choices)
            )
            prompt = ANSWERING_MC_PEROPT.format(
                frame_label_list=frame_label_list,
                question=question,
                answer_choices=choices_str,
            )
        elif prompt_style in ("direct_v2", "direct_v3"):
            choices_str = " ".join(
                f"({letters[i]}) {c}" for i, c in enumerate(answer_choices)
            )
            tmpl = ANSWERING_MC_V2 if prompt_style == "direct_v2" else ANSWERING_MC_V3
            prompt = tmpl.format(
                frame_label_list=frame_label_list,
                question=question,
                answer_choices=choices_str,
            )
        else:
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
    predicted = extract_answer_letter(raw, answer_choices) if answer_choices else ""

    return {
        "raw_response": raw,
        "predicted_letter": predicted,
        "frame_ids_used": frame_ids,
    }
