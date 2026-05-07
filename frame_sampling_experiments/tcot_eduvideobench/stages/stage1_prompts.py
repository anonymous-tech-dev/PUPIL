"""
stages/stage1_prompts.py — Stage 1: Prompt Construction.

This module contains the *exact* prompts from the paper:
  - Figure 3  → Selection prompt S(x, q)
  - Figure 14 → Multiple-choice answering prompt (Gemini / GPT style)
  - Figure 15 → Multiple-choice answering prompt (Qwen style)
  - Figure 16 → Open-ended answering prompt (OpenEQA)

The selection prompt asks the model to return JSON with frame_ids and a
justification string.
"""

from typing import List


# ─── Selection Prompt (Fig. 3 in paper) ────────────────────────────────────────

SELECTION_PROMPT_TEMPLATE = """You will be given a question about a video and {num_choices} possible answer options.

{frame_labels}
Question: {question}
Possible answer choices: {answer_choices}

Return the frame ids which can answer the given question.

Please use the following JSON format for your output:
{{"frame_ids": [List of integer frame IDs], "justification": "Justification about your output"}}"""

# SELECTION_PROMPT_TEMPLATE_QWEN = """You will be given a question about a video and {num_choices} possible answer options.

# {frame_labels}
# Question: {question}
# Possible answer choices: {answer_choices}

# Return the frame ids which can answer the given question.
# Respond with ONLY a JSON object and nothing else, and where frame_ids is a valid list of integer frame IDs. 

# Example format:
# {{"frame_ids": [1, 5, 12...], "justification": "These frames show the relevant action."}}"""

SELECTION_PROMPT_TEMPLATE_QWEN = """You will be given a question about a video and {num_choices} possible answer options.
{frame_labels}
Question: {question}
Possible answer choices: {answer_choices}
Return the frame ids which are most relevant to answering the given question. You MUST select at least one frame. If no frames are clearly relevant, select the frames most likely to contain useful information.
Respond with ONLY a JSON object and nothing else. Example format:
{{"frame_ids": [1, 5, 12], "justification": "These frames show the relevant action."}}"""


def build_selection_prompt(
    frame_ids: List[int],
    question: str,
    answer_choices: List[str],
    style="qwen",
) -> str:
    """
    Build the frame-selection prompt (Figure 3 of paper).

    The frames themselves are passed separately to the VLM as image tokens;
    here we only embed the FrameID labels so the model can refer to them in
    the JSON output.

    Args:
        frame_ids     : list of 1-indexed frame IDs, in order
        question      : the QA question string
        answer_choices: list of answer option strings (4 or 5 items)

    Returns:
        Formatted prompt string.
    """
    template = (SELECTION_PROMPT_TEMPLATE_QWEN if style == "qwen" 
                else SELECTION_PROMPT_TEMPLATE)

    num_choices = len(answer_choices)

    # Build "FrameID 1: [image], FrameID 2: [image], ..." labels.
    # The actual image content is injected by the model backend.
    # We emit a placeholder text — real content comes from interleaved images.
    
    frame_labels = "\n".join(
        [f"FrameID {fid}: <image>" for fid in frame_ids]
    )

    # frame_labels = ", ".join([f"FrameID {fid}: <image>" for fid in frame_ids])

    # Format choices as "(A) text (B) text …"
    choice_letters = "ABCDE"
    choices_str = " ".join(
        f"({choice_letters[i]}) {c}"
        for i, c in enumerate(answer_choices)
    )

    # return SELECTION_PROMPT_TEMPLATE.format(
    #     num_choices=num_choices,
    #     frame_labels=frame_labels,
    #     question=question,
    #     answer_choices=choices_str,
    # )
    return template.format(
        num_choices=num_choices,
        frame_labels=frame_labels,
        question=question,
        answer_choices=choices_str,
    )


# ─── Answering Prompt — Multiple Choice (Fig. 14, Gemini / GPT style) ──────────

ANSWERING_MC_PROMPT_TEMPLATE_GPT = """You will be given a question about a video and {num_choices} possible answer options. You are provided frames from the video, retrieved by an intelligent agent.

Frames: {frame_label_list}
Question: {question}
Possible answer choices: {answer_choices}

After explaining your reasoning, output the final answer in the format "Final Answer: (X)" where X is the correct letter choice. Never say "unknown" or "unsure", or "None", instead provide your most likely guess."""


# ─── Answering Prompt — Multiple Choice (Fig. 15, Qwen style) ──────────────────

ANSWERING_MC_PROMPT_TEMPLATE_QWEN = """Frames: {frame_label_list}

Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.
Question: {question}
Options: {answer_choices}
Answer with the option's letter from the given choices directly and only give the best option."""


# ─── Answering Prompt — Open-Ended (Fig. 16, OpenEQA style) ───────────────────

ANSWERING_OPENENDED_PROMPT_TEMPLATE = """Frames: {frame_label_list}

You will be given a question about a video. You will be provided frames from the video, retrieved by an intelligent agent. It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response.

Question: {question}"""


def build_answering_prompt(
    frame_ids: List[int],
    question: str,
    answer_choices: List[str],
    style: str = "gpt",
) -> str:
    """
    Build the answering prompt (Figures 14/15/16 of paper).

    Args:
        frame_ids     : list of frame IDs included in the curated context
        question      : question string
        answer_choices: list of answer strings. If empty → open-ended format.
        style         : "gpt" | "qwen" — selects which MC template to use
                        (GPT/Gemini vs Qwen).  Ignored for open-ended.

    Returns:
        Formatted prompt string.
    """
    if not answer_choices:
        # Open-ended (OpenEQA)
        frame_label_list = ", ".join(str(fid) for fid in frame_ids)
        return ANSWERING_OPENENDED_PROMPT_TEMPLATE.format(
                    frame_label_list=frame_label_list,
                    question=question
                )

    num_choices = len(answer_choices)
    choice_letters = "ABCDE"
    choices_str = " ".join(
        f"({choice_letters[i]}) {c}"
        for i, c in enumerate(answer_choices)
    )

    frame_label_list = ", ".join(str(fid) for fid in frame_ids)

    if style == "qwen":
        return ANSWERING_MC_PROMPT_TEMPLATE_QWEN.format(
            frame_label_list=frame_label_list,
            question=question,
            answer_choices=choices_str,
        )
    else:
        return ANSWERING_MC_PROMPT_TEMPLATE_GPT.format(
            num_choices=num_choices,
            frame_label_list=frame_label_list,
            question=question,
            answer_choices=choices_str,
        )