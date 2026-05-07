"""
ablation_seg_baseline.py — Segment-Count Ablation: Native Video Clip Inference.

MOTIVATION
──────────
Your overlap analysis (N=12 segments) showed:
  Single Segment (No Overlap) : 800
  Spans Exactly 2 Segments    : 129
  Spans Exactly 3 Segments    :  15
  Spans 4 or More Segments    :  94

This ablation isolates the value of TCoT's selection call by asking:
"If we give Qwen the right number of segments directly — with zero selection
logic — how well does it do?"

HOW IT WORKS
────────────
For each question:
  1. Get the video's native duration and fps (via ffprobe / decord).
  2. Divide total duration into NUM_SEGMENTS equal time windows (same l as TCoT).
  3. Choose which window(s) to use (first_N or oracle mode).
  4. Use ffmpeg to trim the video to exactly that time window → temp clip.
  5. Pass the temp clip directly into Qwen's {"type": "video"} input.
     Qwen handles all internal frame sampling — we do nothing.

This is the correct vanilla comparison: same segment boundaries as TCoT,
but Qwen sees a raw clip instead of TCoT-curated frames.

TWO SUB-MODES (set ABLATION_MODE below)
────────────────────────────────────────
  "first_N"  — always use the first NUM_ABLATION_SEGMENTS time windows.
               Sweep NUM_ABLATION_SEGMENTS = 1, 2, 3, 4 to match your
               overlap distribution.

  "oracle"   — use the LVBench time_reference to pick exactly the window(s)
               that contain the answer. Upper-bound on answerer quality.

RESULTS SAVED TO
────────────────
  results/<dataset>/Qwen2.5-VL-7B_seg_{N}_baseline_segs{N}_k<k>_results.jsonl

NOTE: ffmpeg must be available on PATH for video trimming.
      `pip install ffmpeg-python` is NOT needed — we shell out directly.

HOT-RESUME: already-processed UIDs are skipped automatically.

══════════════════════════════════════════════════════════════════════════
  KNOBS
══════════════════════════════════════════════════════════════════════════
"""

# "first_N" → always use the first NUM_ABLATION_SEGMENTS windows
# "oracle"  → use time_reference to pick the right window(s)
ABLATION_MODE = "oracle"

# How many segments to pass to Qwen. Sweep 1, 2, 3, 4 across runs.
NUM_ABLATION_SEGMENTS = 1

# ── How the chosen segment(s) are fed to Qwen ─────────────────────────────────
# "direct" → ffmpeg trims the video to the chosen time window(s) and passes
#            the raw clip file via Qwen's {"type": "video"} input.
#            Qwen handles all internal frame sampling. Requires ffmpeg on PATH.
#
# "frames" → frames are extracted at VIDEO_FPS (decord/cv2), the chosen
#            segment frames are uniformly subsampled to k=CONTEXT_BUDGET_FRAMES
#            and passed as PIL images via answer_question() — identical to
#            baseline.py but restricted to the chosen segment(s) only.
#            No ffmpeg needed.
SEGMENT_INPUT_MODE = "frames"

# Resolution cap for Qwen's video tokeniser ("direct" mode only).
NATIVE_MAX_PIXELS = 360 * 420

# FPS hint for Qwen ("direct" mode only; None = let Qwen decide).
NATIVE_FPS = 1.0

# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcot.ablation_seg")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config
from utils.dataset_loaders import load_egoschema, load_lvbench
from utils.results_io import (
    load_completed_uids, save_result, results_filepath
)
from stages.stage2_selection_parsing import extract_answer_letter
from stages.stage0_video_loading import (
    load_video_frames, segment_bundle, uniform_subsample, get_frame_ids,
)
from stages.stage4_answering import answer_question


# ─────────────────────────────────────────────────────────────────────────────
# Variant name & extra tag (drives the results filename)
# ─────────────────────────────────────────────────────────────────────────────

def _variant_name() -> str:
    # e.g. "seg_1_direct_baseline" or "seg_1_frames_baseline"
    return f"seg_{NUM_ABLATION_SEGMENTS}_{SEGMENT_INPUT_MODE}_baseline"

def _extra() -> Dict[str, Any]:
    return {"segs": NUM_ABLATION_SEGMENTS}


# ─────────────────────────────────────────────────────────────────────────────
# Video duration helper
# ─────────────────────────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """
    Return duration in seconds using ffprobe (fast, no decoding).
    Falls back to decord if ffprobe is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        pass

    # Fallback: decord
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        return len(vr) / vr.get_avg_fps()
    except Exception:
        pass

    # Last resort: cv2
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


# ─────────────────────────────────────────────────────────────────────────────
# Segment time-window helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_segment_windows(
    duration: float,
    num_segments: int,
) -> List[Tuple[float, float]]:
    """
    Return list of (start_sec, end_sec) for each of `num_segments` equal windows.
    """
    seg_dur = duration / num_segments
    return [(i * seg_dur, (i + 1) * seg_dur) for i in range(num_segments)]


def _parse_time_ref(time_reference: str) -> Optional[Tuple[float, float]]:
    """Parse '04:19-08:41' → (259.0, 521.0) seconds."""
    m = re.match(r"(\d+):(\d+)-(\d+):(\d+)", time_reference.strip())
    if not m:
        return None
    m1, s1, m2, s2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return float(m1 * 60 + s1), float(m2 * 60 + s2)


def choose_segment_indices(
    mode           : str,
    n_segs_to_use  : int,
    num_segments   : int,
    windows        : List[Tuple[float, float]],
    time_reference : str,
) -> List[int]:
    """
    Return the 0-indexed segment indices to pass to Qwen.
    """
    if mode == "oracle" and time_reference:
        parsed = _parse_time_ref(time_reference)
        if parsed:
            ref_start, ref_end = parsed
            idxs = []
            for i, (ws, we) in enumerate(windows):
                # Any overlap between [ws, we) and [ref_start, ref_end]
                if ws < ref_end and we > ref_start:
                    idxs.append(i)
            if idxs:
                return idxs
        logger.warning("  Oracle: could not parse time_reference %r — using first segment.", time_reference)

    # first_N (or oracle fallback)
    return list(range(min(n_segs_to_use, num_segments)))


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg clip extraction
# ─────────────────────────────────────────────────────────────────────────────

def trim_video_clip(
    video_path : str,
    start_sec  : float,
    end_sec    : float,
    out_path   : str,
) -> bool:
    """
    Trim video_path to [start_sec, end_sec] and write to out_path using ffmpeg.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", video_path,
        "-c", "copy",          # stream-copy: fast, no re-encoding
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.error("  ffmpeg failed: %s", result.stderr.decode()[-300:])
        return False
    return True


def merge_clip_windows(
    video_path    : str,
    windows       : List[Tuple[float, float]],
    chosen_idxs   : List[int],
    tmp_dir       : str,
) -> Optional[str]:
    """
    Trim the video to each chosen window and concatenate them into one clip.
    Returns path to the merged clip, or None on failure.
    """
    if not chosen_idxs:
        return None

    clip_paths = []
    for idx in chosen_idxs:
        start, end = windows[idx]
        clip_path  = os.path.join(tmp_dir, f"seg_{idx:03d}.mp4")
        if not trim_video_clip(video_path, start, end, clip_path):
            return None
        clip_paths.append(clip_path)

    if len(clip_paths) == 1:
        return clip_paths[0]

    # Concatenate multiple clips with ffmpeg concat demuxer
    list_file = os.path.join(tmp_dir, "concat_list.txt")
    with open(list_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    merged_path = os.path.join(tmp_dir, "merged.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        merged_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.error("  ffmpeg concat failed: %s", result.stderr.decode()[-300:])
        return None
    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# Qwen native video inference (mirrors baseline_native.py)
# ─────────────────────────────────────────────────────────────────────────────

class NativeQwenInference:
    def __init__(self):
        self.model     = None
        self.processor = None

    def load(self):
        logger.info("[NativeQwen] Loading %s …", config.QWEN_MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTN_IMPL,
            device_map=config.QWEN_DEVICE,
        )
        self.processor = AutoProcessor.from_pretrained(config.QWEN_MODEL_ID)
        logger.info("[NativeQwen] Ready.")

    def infer(self, clip_path: str, prompt: str) -> str:
        video_content = {
            "type"       : "video",
            "video"      : clip_path,
            "max_pixels" : NATIVE_MAX_PIXELS,
        }
        if NATIVE_FPS is not None:
            video_content["fps"] = NATIVE_FPS

        messages = [{
            "role": "user",
            "content": [video_content, {"type": "text", "text": prompt}],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(config.QWEN_DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=config.ANSWER_MAX_TOKENS
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def unload(self):
        import gc
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt (Qwen Fig. 15 style)
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(question: str, answer_choices: List[str]) -> str:
    choice_letters = "ABCDE"
    choices_str = " ".join(
        f"({choice_letters[i]}) {c}" for i, c in enumerate(answer_choices)
    )
    if answer_choices:
        return (
            "Carefully watch the video and pay attention to the cause and "
            "sequence of events, the detail and movement of objects and the "
            "action and pose of persons. Based on your observations, select "
            "the best option that accurately addresses the question.\n\n"
            f"Question: {question}\n"
            f"Options: {choices_str}\n\n"
            "Answer with the option's letter from the given choices directly "
            "and only give the best option."
        )
    return (
        "Carefully watch the video. Imagine the visual scene as vividly as "
        "possible to enhance the accuracy of your response.\n\n"
        f"Question: {question}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_iterator():
    n = config.NUM_SAMPLES
    if config.DATASET == "egoschema":
        return load_egoschema(num_samples=n)
    elif config.DATASET in ("lvbench_v1", "lvbench_v2"):
        return load_lvbench(num_samples=n)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET!r}. "
                         "Choose 'egoschema', 'lvbench_v1', or 'lvbench_v2'.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_direct(model, video_path, windows, chosen_idxs, chosen_windows,
                  question, answer_choices, tmp_dir):
    """Trim chosen windows to a clip and pass natively to Qwen."""
    t0 = time.time()
    clip_path = merge_clip_windows(video_path, windows, chosen_idxs, tmp_dir)
    if clip_path is None:
        raise RuntimeError("ffmpeg clip extraction failed")
    logger.info("  Clip ready in %.1fs", time.time() - t0)

    prompt = _build_prompt(question, answer_choices)
    raw    = model.infer(clip_path, prompt)
    predicted = extract_answer_letter(raw) if answer_choices else ""
    return raw, predicted, []   # no frame ids (Qwen internal)


def _infer_frames(model, video_path, windows, chosen_idxs, chosen_windows,
                  question, answer_choices):
    """Extract frames from chosen segments, uniformly subsample to k, answer."""
    # Decode full video at VIDEO_FPS then restrict to chosen segment frames
    full_bundle = load_video_frames(video_path, fps=config.VIDEO_FPS)
    total_frames = len(full_bundle)
    duration_approx = total_frames / config.VIDEO_FPS

    # Map chosen time windows → frame indices in full_bundle
    seg_bundles = segment_bundle(full_bundle, config.NUM_SEGMENTS)
    context = []
    for idx in chosen_idxs:
        if idx < len(seg_bundles):
            context.extend(seg_bundles[idx])

    # Deduplicate & sort
    seen, deduped = set(), []
    for fid, img in sorted(context, key=lambda x: x[0]):
        if fid not in seen:
            seen.add(fid)
            deduped.append((fid, img))

    # Uniformly subsample to k — same as baseline.py
    k = config.CONTEXT_BUDGET_FRAMES
    context_bundle = uniform_subsample(deduped, k)
    logger.info("  Frames mode: %d seg frames → subsampled to %d (k=%d)",
                len(deduped), len(context_bundle), k)

    ans = answer_question(
        model=model,
        context_bundle=context_bundle,
        question=question,
        answer_choices=answer_choices,
    )
    return ans["raw_response"], ans["predicted_letter"], get_frame_ids(context_bundle)


def run_ablation_sample(
    model      : NativeQwenInference,
    item       : Dict[str, Any],
    tmp_dir    : str,
) -> Dict[str, Any]:
    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]
    time_reference = item.get("time_reference", "")

    logger.info("Ablation uid=%s | input=%s ablation=%s segs=%d | %s",
                uid, SEGMENT_INPUT_MODE, ABLATION_MODE, NUM_ABLATION_SEGMENTS,
                os.path.basename(video_path))

    # Duration + segment windows (needed by both modes)
    duration = get_video_duration(video_path)
    logger.info("  Duration: %.1fs", duration)
    windows = compute_segment_windows(duration, config.NUM_SEGMENTS)

    chosen_idxs = choose_segment_indices(
        mode          = ABLATION_MODE,
        n_segs_to_use = NUM_ABLATION_SEGMENTS,
        num_segments  = config.NUM_SEGMENTS,
        windows       = windows,
        time_reference= time_reference,
    )
    chosen_windows = [windows[i] for i in chosen_idxs]
    total_clip_dur = sum(e - s for s, e in chosen_windows)
    logger.info("  Segments %s → %.1fs of video (%.1f%%)",
                chosen_idxs, total_clip_dur,
                100.0 * total_clip_dur / duration if duration else 0.0)

    # ── Branch on SEGMENT_INPUT_MODE ──────────────────────────────────────
    t1 = time.time()
    if SEGMENT_INPUT_MODE == "direct":
        raw, predicted, context_ids = _infer_direct(
            model, video_path, windows, chosen_idxs, chosen_windows,
            question, answer_choices, tmp_dir,
        )
    else:  # "frames"
        raw, predicted, context_ids = _infer_frames(
            model, video_path, windows, chosen_idxs, chosen_windows,
            question, answer_choices,
        )
    logger.info("  predicted=%r  gt=%r (%.1fs)", predicted, ground_truth,
                time.time() - t1)

    return {
        "uid"                  : uid,
        "predicted_letter"     : predicted,
        "ground_truth"         : ground_truth,
        "raw_answer"           : raw,
        "chosen_segment_idxs"  : chosen_idxs,
        "chosen_windows_sec"   : chosen_windows,
        "total_clip_duration_s": total_clip_dur,
        "video_duration_s"     : duration,
        "pct_video_used"       : 100.0 * total_clip_dur / duration if duration else 0.0,
        "context_ids"          : context_ids,
        "num_segments_used"    : len(chosen_idxs),
        "segment_input_mode"   : SEGMENT_INPUT_MODE,
        "ablation_mode"        : ABLATION_MODE,
        "num_ablation_segments": NUM_ABLATION_SEGMENTS,
        "total_segments"       : config.NUM_SEGMENTS,
        "native_fps"           : NATIVE_FPS,
        "native_max_pixels"    : NATIVE_MAX_PIXELS,
        "selected_ids"         : [],
        "justifications"       : [],
        "raw_responses"        : [],
        "stage"                : _variant_name(),
        "video_path"           : video_path,
        "question"             : question,
        "answer_choices"       : answer_choices,
        "question_type"        : item.get("question_type", []),
        "time_reference"       : time_reference,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if SEGMENT_INPUT_MODE == "direct" and "qwen" not in config.MODEL.lower():
        logger.error(
            "SEGMENT_INPUT_MODE='direct' uses Qwen's native video input. "
            "Set config.MODEL = 'Qwen2.5-VL-7B' or switch to SEGMENT_INPUT_MODE='frames'."
        )
        sys.exit(1)

    variant = _variant_name()
    extra   = _extra()

    logger.info("=" * 70)
    logger.info("Segment Ablation Baseline")
    logger.info("  Model           : %s", config.QWEN_MODEL_ID)
    logger.info("  Dataset         : %s", config.DATASET)
    logger.info("  Segment input   : %s", SEGMENT_INPUT_MODE)
    logger.info("  Ablation mode   : %s", ABLATION_MODE)
    logger.info("  Segs used       : %d / %d total (l=%d)",
                NUM_ABLATION_SEGMENTS, config.NUM_SEGMENTS, config.NUM_SEGMENTS)
    logger.info("  Native FPS      : %s  (direct mode only)", NATIVE_FPS)
    logger.info("  Max pixels      : %d  (direct mode only)", NATIVE_MAX_PIXELS)
    logger.info("  Results file    : %s", results_filepath(variant=variant, extra=extra))
    logger.info("=" * 70)

    # Hot-resume
    completed = load_completed_uids(variant=variant, extra=extra)
    logger.info("Hot-resume: %d items already done — skipping.", len(completed))

    # Load the right model class depending on mode:
    #   "direct" → NativeQwenInference (raw video clip via {"type":"video"})
    #   "frames" → BaseVLM subclass via factory (PIL images → call_answering)
    if SEGMENT_INPUT_MODE == "direct":
        model = NativeQwenInference()
        model.load()
    else:
        from models.factory import get_model
        model = get_model()
        model.load()

    total   = 0
    correct = 0
    skipped = 0

    # Use a single shared temp dir per run (cleaned up on exit)
    with tempfile.TemporaryDirectory(prefix="tcot_ablation_") as tmp_dir:
        for item in get_dataset_iterator():
            uid = str(item["uid"])
            if uid in completed:
                skipped += 1
                continue

            try:
                result = run_ablation_sample(model, item, tmp_dir)
                save_result(result, variant=variant, extra=extra)

                total += 1
                if (result["predicted_letter"]
                        and result["ground_truth"]
                        and result["predicted_letter"] == result["ground_truth"]):
                    correct += 1

                acc = 100.0 * correct / total if total else 0.0
                logger.info("  Running accuracy: %.1f%% (%d/%d) [skipped=%d]",
                            acc, correct, total, skipped)

            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "  OOM on uid=%s — clip may be too long. "
                    "Try reducing NATIVE_MAX_PIXELS.", uid
                )
                torch.cuda.empty_cache()
                continue

            except Exception as exc:
                logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
                continue

    logger.info("=" * 70)
    logger.info("Ablation complete. Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results: %s", results_filepath(variant=variant, extra=extra))

    model.unload()


if __name__ == "__main__":
    main()