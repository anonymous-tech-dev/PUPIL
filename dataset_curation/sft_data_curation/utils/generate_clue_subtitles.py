#!/usr/bin/env python3
"""
Hot-resumable transcription of CGBench clue_vids using native OpenAI Whisper large-v3.
- float32 precision for maximum accuracy
- no_speech_threshold + condition_on_previous_text=False to minimise hallucinations on short clips
- BEST_OF=1 → normal greedy decoding; BEST_OF>1 → sample N times and pick most confident
- Skips already-transcribed files (hot-resume safe)
- Saves each transcript as .srt + .txt
- Prints (done/total) progress at all times

Usage:
    pip install openai-whisper
    python transcribe_clue_vids.py
"""

import os
import sys
import time
import glob
import traceback
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_DIR  = "/data/Pupil/CGBench/clue_vids"
OUTPUT_DIR = "/data/Pupil/CGBench/clue_vids_subtitles"

MODEL_SIZE = "large-v3"
DEVICE     = "cuda:3"
FP16       = False   # False = float32, maximum accuracy

# ── best_of ────────────────────────────────────────────────────────────────
# Set to 1 for normal greedy decoding (faster, same quality when temperature=0).
# Set to N>1 to sample N candidates and pick the one with lowest avg log-prob.
# Note: best_of only has an effect when temperature > 0 (i.e. during fallback
# decoding). At temperature=0.0 (greedy), the model always picks the same token
# deterministically, so best_of=5 behaves identically to best_of=1.
# Whisper automatically raises temperature during fallback if the greedy pass
# produces low-confidence or repetitive output — that's when best_of kicks in.
BEST_OF = 5   # set to 1 to disable

# ── Anti-hallucination settings ────────────────────────────────────────────
TRANSCRIBE_KWARGS = dict(
    language                    = None,   # auto-detect per file (handles Chinese/English mix)
    beam_size                   = 5,
    temperature                 = 0.0,    # greedy first; whisper falls back automatically
    compression_ratio_threshold = 2.4,
    no_speech_threshold         = 0.6,    # suppress hallucinations on silence / short clips
    condition_on_previous_text  = False,  # each clip is independent, no context bleeding
    word_timestamps             = True,
    verbose                     = False,
)
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_timestamp(seconds: float) -> str:
    hours  = int(seconds // 3600)
    mins   = int((seconds % 3600) // 60)
    secs   = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def result_to_srt(result: dict) -> str:
    lines = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        start = format_timestamp(seg["start"])
        end   = format_timestamp(seg["end"])
        text  = seg["text"].strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def result_to_txt(result: dict) -> str:
    return " ".join(
        seg["text"].strip()
        for seg in result.get("segments", [])
        if seg["text"].strip()
    )


def get_all_videos():
    return sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))


def is_done(video_path: str) -> bool:
    """True only if both .srt and .txt exist and are non-empty."""
    stem    = Path(video_path).stem
    srt_out = os.path.join(OUTPUT_DIR, f"{stem}.srt")
    txt_out = os.path.join(OUTPUT_DIR, f"{stem}.txt")
    return (
        os.path.exists(srt_out) and os.path.getsize(srt_out) > 0 and
        os.path.exists(txt_out) and os.path.getsize(txt_out) > 0
    )


def transcribe_video(model, video_path: str, global_idx: int, total: int):
    stem    = Path(video_path).stem
    srt_out = os.path.join(OUTPUT_DIR, f"{stem}.srt")
    txt_out = os.path.join(OUTPUT_DIR, f"{stem}.txt")

    try:
        kwargs = dict(TRANSCRIBE_KWARGS)
        if BEST_OF > 1:
            kwargs["best_of"] = BEST_OF

        result = model.transcribe(video_path, fp16=FP16, **kwargs)

        srt_content = result_to_srt(result)
        txt_content = result_to_txt(result)

        with open(srt_out, "w", encoding="utf-8") as f:
            f.write(srt_content)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(txt_content)

        lang = result.get("language", "?")
        segs = len(result.get("segments", []))
        print(f"  [{global_idx}/{total}] ✓  {stem}  lang={lang}  segs={segs}")

    except Exception as e:
        print(f"  [{global_idx}/{total}] ✗  FAILED {stem}: {e}")
        traceback.print_exc()
        # Do NOT write output files on failure — job will retry this file on resume


def main():
    try:
        import whisper
    except ImportError:
        print("openai-whisper not installed. Run:  pip install openai-whisper")
        sys.exit(1)

    all_videos = get_all_videos()
    total      = len(all_videos)

    if total == 0:
        print(f"No .mp4 files found in {INPUT_DIR}")
        sys.exit(0)

    # ── Resume logic ────────────────────────────────────────────────────────
    todo   = [v for v in all_videos if not is_done(v)]
    n_done = total - len(todo)

    print(f"\n{'='*60}")
    print(f"  CGBench clue_vids transcription  (native OpenAI whisper, fp32)")
    print(f"  Total videos  : {total}")
    print(f"  Already done  : {n_done}")
    print(f"  Remaining     : {len(todo)}")
    print(f"  Model         : {MODEL_SIZE}  |  device={DEVICE}  |  fp16={FP16}")
    print(f"  best_of       : {BEST_OF}  {'(sampling active)' if BEST_OF > 1 else '(greedy, best_of disabled)'}")
    print(f"{'='*60}\n")

    if not todo:
        print("All videos already transcribed. Nothing to do.")
        return

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading whisper {MODEL_SIZE} (float32) onto {DEVICE}...")
    t0    = time.time()
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # ── Main loop ───────────────────────────────────────────────────────────
    start_time = time.time()
    for i, video_path in enumerate(todo, start=1):
        global_idx = n_done + i
        transcribe_video(model, video_path, global_idx, total)

        # ETA every 50 videos
        if i % 50 == 0 or i == len(todo):
            elapsed   = time.time() - start_time
            per_vid   = elapsed / i
            remaining = per_vid * (len(todo) - i)
            pct       = 100.0 * global_idx / total
            print(
                f"\n  ── [{global_idx}/{total}]  {pct:.1f}%  |  "
                f"Elapsed: {elapsed/60:.1f}m  |  "
                f"ETA: {remaining/60:.1f}m  |  "
                f"{per_vid:.2f}s/vid ──\n"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Done! {len(todo)} videos in {elapsed/60:.1f} minutes.")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()