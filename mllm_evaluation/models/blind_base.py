"""
Shared utilities for *blind* evaluators (no video frames AND no transcript).

This is the strictest baseline in the ablation matrix: it asks the question
purely against the model's parametric / pre-training knowledge, with no
video-side conditioning of any kind.  Used to answer:

    "How much of Pupil can a model solve from prior knowledge alone,
     without ever seeing the video or its transcript?"

Every blind evaluator imports the same system prompt and user-prompt builder
from this file so the only thing that varies across models is the LLM
backbone — making the resulting numbers directly comparable.

Design notes
------------
* The system prompt is intentionally explicit that no video / transcript /
  retrieval context will be provided, so the model doesn't hallucinate having
  watched anything.  We still ask it to give its best answer (rather than
  refusing), otherwise judges score every row "incorrect" just because the
  model said "I cannot answer".
* The user prompt deliberately echoes back the same wording as the
  transcript-only and video baselines (`QUESTION: …`) so the only delta in
  the prompt is the absence of the conditioning context.  This keeps the
  ablation as clean as possible.
"""

# ── Prompt template ──────────────────────────────────────────────────────────
BLIND_SYSTEM = (
    "You are a helpful assistant. You will be asked a question that originates "
    "from an educational video, but you will NOT receive the video itself, any "
    "of its frames, its transcript, or any retrieved context. Answer the "
    "question using ONLY your prior knowledge. If the question depends on "
    "specific visual content, examples, or numbers shown in the video that "
    "you cannot know without seeing it, say so explicitly and then give your "
    "best inference based on what you know about the topic."
)


def build_blind_prompt(question: str) -> str:
    """The user-side message for a blind run.

    No transcript, no frames — just the question, clearly framed so the model
    knows it is a blind / no-context setting.
    """
    return (
        "You will be asked a question about an educational video. No video, "
        "frames, or transcript are attached — answer from prior knowledge only.\n\n"
        f"QUESTION: {question}"
    )
