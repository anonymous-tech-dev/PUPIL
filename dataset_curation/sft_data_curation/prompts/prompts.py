"""
Prompt templates for CGBench → SFT data generation.

Four strategies
---------------
1. Transcript + Q + A  →  better_answer
2. Transcript + Q + A + [video frames]  →  better_answer
3. Transcript + Q + A  →  reasoning_trace + better_answer
4. Transcript + Q + A + [video frames]  →  reasoning_trace + better_answer

Strategies 1 & 2 are text-only expansions (no CoT chain in output).
Strategies 3 & 4 produce an explicit reasoning trace followed by the answer.

All prompts share the same informational context block; only the
output-format instruction differs.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
#  Shared context builder                                                      #
# --------------------------------------------------------------------------- #

def _build_context(
    question: str,
    choices: list[str],
    correct_answer: str,
    transcript: str,
    clue_intervals: list[list[float]],
    domain: str,
    sub_category: str,
) -> str:
    """Assemble the shared factual context section."""
    formatted_choices = "\n".join(
        f"  {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
    )
    interval_str = "; ".join(
        f"[{s:.2f}s – {e:.2f}s]" for s, e in clue_intervals
    )
    return f"""\
## Video Context
Domain   : {domain}
Category : {sub_category}
Relevant clue window(s): {interval_str}

## Transcript (clue segment)
{transcript.strip()}

## Multiple-Choice Question
{question}

Choices:
{formatted_choices}

## Correct Answer
{correct_answer}"""


# --------------------------------------------------------------------------- #
#  Output-format instructions                                                  #
# --------------------------------------------------------------------------- #

_BETTER_ANSWER_FORMAT = """\
## Your Task
Produce a **better answer** – a concise, well-grounded, complete response to \
the question that:
- Confirms the correct answer with a clear explanation.
- Cites specific visual or textual evidence from the video/transcript that \
  supports the answer (e.g. timestamps, objects seen, words spoken).
- Briefly explains why the other choices are wrong, if that adds clarity.
- Is written as a standalone answer a viewer could read without the original \
  question (i.e. answer in complete sentences).

Respond with ONLY the improved answer text. No preamble, no meta-commentary."""

_REASONING_AND_BETTER_ANSWER_FORMAT = """\
## Your Task
First, produce a **step-by-step reasoning trace** that works through the \
question using evidence from the transcript and video.  Then give the final \
**better answer**.

Output format (follow exactly):
<reasoning>
[Your multi-step reasoning here. Reference the transcript and visual evidence. \
Consider and rule out distractors explicitly.]
</reasoning>
<answer>
[A concise, complete, standalone answer confirming the correct choice with \
evidence. Written as a direct response to the question.]
</answer>

No text outside the XML tags."""


# --------------------------------------------------------------------------- #
#  Public prompt builders                                                      #
# --------------------------------------------------------------------------- #

def build_prompt(
    strategy: int,
    question: str,
    choices: list[str],
    correct_answer: str,
    transcript: str,
    clue_intervals: list[list[float]],
    domain: str,
    sub_category: str,
) -> str:
    """
    Return the full prompt string for the given strategy (1–4).

    For strategies 2 & 4 the caller is expected to pass the clue video path
    to generate_response() – the prompt text itself is identical to strategies
    1 & 3, since the video frames are injected as image tokens by the model
    wrapper, not embedded in the prompt string.
    """
    if strategy not in (1, 2, 3, 4):
        raise ValueError(f"Strategy must be 1–4, got {strategy}.")

    context = _build_context(
        question=question,
        choices=choices,
        correct_answer=correct_answer,
        transcript=transcript,
        clue_intervals=clue_intervals,
        domain=domain,
        sub_category=sub_category,
    )

    if strategy in (2, 4):
        video_note = (
            "\n\n## Video Frames\n"
            "The frames below are sampled uniformly from the clue segment of the video. "
            "Use them together with the transcript as visual evidence.\n"
        )
    else:
        video_note = ""

    if strategy in (1, 2):
        output_instruction = _BETTER_ANSWER_FORMAT
    else:
        output_instruction = _REASONING_AND_BETTER_ANSWER_FORMAT

    return f"{context}{video_note}\n\n{output_instruction}"


# --------------------------------------------------------------------------- #
#  Response parsers                                                            #
# --------------------------------------------------------------------------- #

def parse_response(strategy: int, raw: str) -> dict[str, str]:
    """
    Parse the model's raw response into a dict of SFT keys.

    Strategies 1 & 2  →  {"better_answer": "..."}
    Strategies 3 & 4  →  {"reasoning_trace": "...", "better_answer": "..."}
    """
    raw = raw.strip()

    if strategy in (1, 2):
        return {"better_answer": raw}

    # Strategy 3 / 4 – extract <reasoning> and <answer> tags
    import re
    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL | re.IGNORECASE
    )
    answer_match = re.search(
        r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE
    )

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else raw  # fallback

    if not reasoning:
        # Graceful degradation: if XML tags missing, treat everything before
        # the last paragraph as reasoning and the last paragraph as answer.
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(parts) > 1:
            reasoning = "\n\n".join(parts[:-1])
            answer = parts[-1]
        else:
            reasoning = ""
            answer = raw

    return {"reasoning_trace": reasoning, "better_answer": answer}