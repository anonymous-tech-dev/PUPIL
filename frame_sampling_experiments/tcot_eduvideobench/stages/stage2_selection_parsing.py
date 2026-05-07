"""
stages/stage2_selection_parsing.py — Stage 2: Parse & Validate Model Selection Output.

The model's selection call (Eq. 4 in paper) should return JSON:
  {"frame_ids": [int, ...], "justification": "..."}

This stage handles:
  1. JSON extraction from the raw model string (strip markdown fences, etc.)
  2. Validation: ascending order, no duplicates, within-bounds checks.
  3. Fallback: if parsing fails → return all frames (paper §3.2 fallback).

Paper §3.2:
  'We validate S to ensure that the frame indices are in ascending order,
   contain no duplicates and are within bounds.
   For simple parsing, we prompt the model to predict in the JSON format,
   and for responses that fail to parse, we assume that S = [1, …, N]
   is all frames within the video.'
"""

import json
import re
from typing import List, Optional, Tuple


def parse_selection_response(
    raw_response: str,
    valid_frame_ids: List[int],
) -> Tuple[List[int], str]:
    """
    Parse the model's JSON selection response.

    Args:
        raw_response   : raw string output from the VLM
        valid_frame_ids: the complete ordered list of frame IDs that were
                         presented to the model (used for fallback + bounds check)

    Returns:
        (selected_ids, justification)
          selected_ids  : validated, sorted, deduplicated frame IDs
          justification : the model's textual justification (empty if parse fails)
    """
    parsed = _try_parse_json(raw_response)
    if parsed is None:
        return list(valid_frame_ids), "[PARSE FAILED — using all frames as fallback]"

    # Model sometimes returns a bare list instead of {"frame_ids": [...], ...}
    if isinstance(parsed, list):
        parsed = {"frame_ids": parsed, "justification": ""}

    if not isinstance(parsed, dict):
        return list(valid_frame_ids), "[PARSE FAILED — unexpected JSON type, using all frames as fallback]"

    raw_ids       = parsed.get("frame_ids", [])
    justification = parsed.get("justification", "")

    # Validate
    valid_set = set(valid_frame_ids)
    selected  = []
    seen      = set()
    for fid in raw_ids:
        if not isinstance(fid, int):
            try:
                fid = int(fid)
            except (ValueError, TypeError):
                continue
        if fid in valid_set and fid not in seen:
            selected.append(fid)
            seen.add(fid)

    if not selected:
        # Model returned valid JSON but selected no valid frames.
        # This is NOT the same as a parse failure — the model is saying
        # this segment has no relevant frames. Return empty rather than
        # injecting all 64 segment frames as distractors.
        # Paper §3.2 fallback (S=[1..N]) applies only to parse failures.
        return [], "[EMPTY SELECTION — segment has no relevant frames]"

    # Ensure ascending order
    selected.sort()

    return selected, str(justification)


# ─── JSON extraction helpers ───────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt several strategies to extract a JSON object from `text`."""

    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def extract_answer_letter(raw_response: str) -> str:
    """
    Extract the final answer letter (A/B/C/D/E) from the answering call output.

    Handles:
      - "Final Answer: (B)"
      - "Final Answer: B"
      - "The answer is (C)"
      - Single letter at the start
    """
    # Pattern: "Final Answer: (X)" or "Final Answer: X"
    m = re.search(r"[Ff]inal\s+[Aa]nswer\s*[:：]\s*\(?([A-Ea-e])\)?", raw_response)
    if m:
        return m.group(1).upper()

    # Pattern: "(X)" as the last standalone option letter
    m = re.search(r"\(([A-Ea-e])\)", raw_response)
    if m:
        return m.group(1).upper()

    # Pattern: just a single letter on a line
    m = re.search(r"^\s*([A-Ea-e])\s*$", raw_response, re.MULTILINE)
    if m:
        return m.group(1).upper()

    # Last resort: first capital letter A–E
    m = re.search(r"\b([A-E])\b", raw_response)
    if m:
        return m.group(1).upper()

    return ""