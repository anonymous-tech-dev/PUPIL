"""
Shared utilities for transcript-only evaluators.

Used by ablation runs that ask: "does our benchmark actually require the video,
or is the transcript enough?"  All transcript-only evaluators import the same
loader/prompt builder so the only thing that differs across models is the
underlying LLM backbone.

SRT discovery rule
------------------
For a video at:   <…>/dataset_curation/dataset/videos_db/final_1k/foo_clean.mp4
we look for SRT:  <…>/dataset_curation/dataset/transcripts_db/foo_clean_transcript.srt

Override directories with the env vars:
    EVAL_TRANSCRIPT_DIR   (defaults to <data>/transcripts_db)
    EVAL_TRANSCRIPT_SUFFIX (defaults to "_transcript.srt")
"""

import os
import re

import config


def _transcript_dir() -> str:
    return os.environ.get(
        "EVAL_TRANSCRIPT_DIR",
        os.path.join(config.base_data_dir, "transcripts_db"),
    )


def _transcript_suffix() -> str:
    return os.environ.get("EVAL_TRANSCRIPT_SUFFIX", "_transcript.srt")


def transcript_path_for(video_path: str) -> str | None:
    """Return the SRT path for a given video, or None if not found."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    cand = os.path.join(_transcript_dir(), base + _transcript_suffix())
    return cand if os.path.isfile(cand) else None


_SRT_INDEX_RE = re.compile(r"^\d+\s*$")
_SRT_TS_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*$"
)


def load_transcript(
    video_path: str,
    *,
    keep_timestamps: bool = True,
    max_chars: int | None = None,
) -> str:
    """Load and lightly clean an SRT transcript for prompting.

    keep_timestamps=True   →  keep the "HH:MM:SS --> HH:MM:SS" header lines so
                              models can ground temporal questions.  We drop
                              the per-cue index numbers (1, 2, 3, …) since
                              they're noise.
    keep_timestamps=False  →  strip headers entirely; just newline-joined text.

    max_chars              →  if set, truncate from the end with a "[...
                              transcript truncated]" marker.  None = no cap.
    """
    sp = transcript_path_for(video_path)
    if not sp:
        raise FileNotFoundError(f"No transcript found for {video_path}")

    with open(sp, "r", encoding="utf-8") as f:
        raw = f.read()

    out_lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            # collapse blank lines on the boundary
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue
        if _SRT_INDEX_RE.match(s):
            continue
        if _SRT_TS_RE.match(s):
            if keep_timestamps:
                # Trim the milliseconds — they're rarely useful for grounding.
                short = re.sub(r",\d{3}", "", s)
                short = re.sub(r"\.\d{3}", "", short)
                out_lines.append(f"[{short}]")
            continue
        out_lines.append(s)

    text = "\n".join(out_lines).strip() + "\n"

    if max_chars is not None and len(text) > max_chars:
        head_room = max_chars - 64
        text = text[:head_room].rstrip() + "\n[... transcript truncated ...]\n"

    return text


# ── Prompt template ──────────────────────────────────────────────────────────
# Kept conservative & unbiased: we make it explicit that this is a *text-only*
# baseline so downstream judges aren't surprised by missing visual content.
TRANSCRIPT_SYSTEM = (
    "You are a helpful assistant. You will be given the transcript of an "
    "educational video and a question about it. Answer the question using "
    "ONLY the transcript text. If the transcript does not contain enough "
    "information to answer (e.g., the question requires visual or non-verbal "
    "details), say so explicitly and then give your best inference."
)


def build_user_prompt(transcript: str, question: str) -> str:
    return (
        "VIDEO TRANSCRIPT (SRT format with [HH:MM:SS --> HH:MM:SS] headers):\n"
        "===== BEGIN TRANSCRIPT =====\n"
        f"{transcript}"
        "===== END TRANSCRIPT =====\n\n"
        f"QUESTION: {question}"
    )
