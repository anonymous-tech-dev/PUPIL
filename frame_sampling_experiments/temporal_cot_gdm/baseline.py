"""
baseline_native.py — True Baseline: Native Video Inference (No Frame Extraction).

This is the most direct possible baseline: the raw video file is passed
straight into Qwen2.5-VL's native {"type": "video"} input.  No frame
extraction, no uniform subsampling, no context aggregation — the model
handles everything internally using its own video tokeniser.

This answers the question: "How well does Qwen do if we just throw the
whole video at it and do nothing clever?"

Comparison ladder:
  baseline_native.py  ← you are here: raw video file → model
  baseline.py         ← our uniform-sampled k=120 frames → model
  main.py (TCoT)      ← TCoT-curated context → model

Results saved to:
  results/<dataset>/<model>_baseline_native_results.jsonl

HOT-RESUME: Already-processed UIDs are skipped automatically.

══════════════════════════════════════════════════════════════════════════════
  KNOBS
══════════════════════════════════════════════════════════════════════════════

  MODEL          — must be "Qwen2.5-VL-7B" (GPT-Azure doesn't support
                   raw video file input via the Azure API)
  DATASET        — egoschema | lvbench
  NUM_SAMPLES    — -1 = all
  NATIVE_FPS     — fps passed to Qwen's video tokeniser (default: 1.0,
                   matching the paper). Lower = fewer tokens = fits longer
                   videos. Set to None to let Qwen choose automatically.
  NATIVE_MAX_PIXELS — resolution cap per frame (default: 360*420 = 151 200).
                   Reduce if you hit OOM on very long videos.
"""

import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tcot.baseline_native")

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config
from utils.dataset_loaders import load_egoschema, load_lvbench
from utils.results_io import load_completed_uids, save_result
from stages.stage2_selection_parsing import extract_answer_letter

# ─────────────────────────────────────────────────────────────────────────────
# Native-video knobs (not in config.py to keep things clean)
# ─────────────────────────────────────────────────────────────────────────────

# FPS passed to Qwen's video tokeniser. Paper samples at 1 fps — match that.
NATIVE_FPS = 1.0

# Max pixels per frame (width × height). Reduce for OOM on long videos.
# 360*420 = 151 200  (Qwen default for video is often 360*420 or lower)
NATIVE_MAX_PIXELS = 360 * 420

# Variant tag for the results file
NATIVE_VARIANT = "baseline_native"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt (Qwen Fig. 15 style — same as main.py answering call)
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
    else:
        # Open-ended
        return (
            "Carefully watch the video. It is crucial that you imagine the "
            "visual scene as vividly as possible to enhance the accuracy of "
            "your response.\n\n"
            f"Question: {question}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Native video inference
# ─────────────────────────────────────────────────────────────────────────────

class NativeQwenInference:
    """
    Loads Qwen2.5-VL once and runs native video-file inference.
    Uses {"type": "video", "video": path, "fps": ..., "max_pixels": ...}
    so Qwen's own video tokeniser does all the frame sampling.
    """

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
        logger.info("[NativeQwen] Model loaded.")

    def infer(self, video_path: str, prompt: str) -> str:
        """
        Pass the video file directly to Qwen — no preprocessing on our side.
        """
        video_content: dict = {
            "type"       : "video",
            "video"      : video_path,
            "max_pixels" : NATIVE_MAX_PIXELS,
        }
        if NATIVE_FPS is not None:
            video_content["fps"] = NATIVE_FPS

        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

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

        num_visual_tokens = inputs["input_ids"].shape[1]
        logger.debug("  Input tokens: %d", num_visual_tokens)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.ANSWER_MAX_TOKENS,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
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

def run_native_sample(
    model: NativeQwenInference,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    sample_start_time = time.time()

    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("Native uid=%s | %s", uid, os.path.basename(video_path))

    prompt = _build_prompt(question, answer_choices)

    t0 = time.time()
    raw = model.infer(video_path, prompt)
    elapsed = time.time() - t0

    predicted = extract_answer_letter(raw) if answer_choices else ""
    logger.info("  predicted=%r  gt=%r  (%.1fs)", predicted, ground_truth, elapsed)

    sample_total_time = time.time() - sample_start_time

    return {
        "uid"              : uid,
        "predicted_letter" : predicted,
        "ground_truth"     : ground_truth,
        "raw_answer"       : raw,
        # No frame extraction — these fields are N/A for native inference
        "selected_ids"     : [],
        "num_selected"     : 0,
        "context_ids"      : [],
        "num_context"      : 0,
        "total_frames"     : -1,   # unknown — Qwen handles internally
        "pct_selected"     : -1.0,
        "justifications"   : [],
        "raw_responses"    : [],
        "stage"            : NATIVE_VARIANT,
        "native_fps"       : NATIVE_FPS,
        "native_max_pixels": NATIVE_MAX_PIXELS,
        "video_path"       : video_path,
        "question"         : question,
        "answer_choices"   : answer_choices,
        "question_type"    : item.get("question_type", []),
        "time_reference"   : item.get("time_reference", ""),
        "time_taken_secs"  : round(sample_total_time, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if "qwen" not in config.MODEL.lower():
        logger.error(
            "baseline_native.py only supports Qwen2.5-VL (native video input). "
            "Set config.MODEL = 'Qwen2.5-VL-7B' and re-run."
        )
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("True Baseline — Native Qwen2.5-VL Video Inference")
    logger.info("  Model       : %s", config.QWEN_MODEL_ID)
    logger.info("  Dataset     : %s", config.DATASET)
    logger.info("  Native FPS  : %s", NATIVE_FPS)
    logger.info("  Max pixels  : %s", NATIVE_MAX_PIXELS)
    logger.info("  Num samples : %s",
                config.NUM_SAMPLES if config.NUM_SAMPLES != -1 else "all")
    logger.info("  NOTE: No frame extraction. Qwen tokenises the video itself.")
    logger.info("=" * 70)

    # ── Hot-resume ────────────────────────────────────────────────────────
    completed = load_completed_uids(variant=NATIVE_VARIANT)
    logger.info("Hot-resume: %d items already done — skipping.", len(completed))

    # ── Load model ────────────────────────────────────────────────────────
    model = NativeQwenInference()
    model.load()

    # ── Main loop ─────────────────────────────────────────────────────────
    total   = 0
    correct = 0
    skipped = 0

    for item in get_dataset_iterator():
        uid = str(item["uid"])

        if uid in completed:
            skipped += 1
            continue

        try:
            result = run_native_sample(model, item)
            save_result(result, variant=NATIVE_VARIANT)

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
                "  OOM on uid=%s — video may be too long for current "
                "NATIVE_MAX_PIXELS=%d. Try reducing it.",
                uid, NATIVE_MAX_PIXELS,
            )
            torch.cuda.empty_cache()
            continue

        except Exception as exc:
            logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
            continue

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Native baseline complete. Processed=%d  Skipped=%d",
                total, skipped)
    if total > 0:
        logger.info("Final native baseline accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    logger.info("Results: results/%s/..._baseline_native_results.jsonl",
                config.DATASET)

    model.unload()


if __name__ == "__main__":
    main()