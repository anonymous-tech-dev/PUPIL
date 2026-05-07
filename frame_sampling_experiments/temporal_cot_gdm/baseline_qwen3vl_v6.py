"""
baseline_qwen3vl_v6.py — Deterministic Vanilla Qwen3-VL-8B Baseline (v6).

Raw video file → Qwen3-VL native video input → answer.
No frame extraction, no TCoT, no context curation.
Fully deterministic: fixed seeds, do_sample=False.

Results:  results/<dataset>/Qwen3-VL-8B_baseline_native_v6_k256_results_v6.jsonl
"""

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import random
from typing import Dict, Any, List

# ── Determinism ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("baseline.qwen3vl_v6")

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config

from utils.dataset_loaders import load_egoschema, load_lvbench
from utils.results_io import load_completed_uids, save_result
from stages.stage2_selection_parsing import extract_answer_letter

# ─────────────────────────────────────────────────────────────────────────────
# Knobs
# ─────────────────────────────────────────────────────────────────────────────
NATIVE_FPS        = 1.0
NATIVE_MAX_PIXELS = 360 * 420       # 151,200
NATIVE_VARIANT    = "baseline_native_v6"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
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
        return (
            "Carefully watch the video. It is crucial that you imagine the "
            "visual scene as vividly as possible to enhance the accuracy of "
            "your response.\n\n"
            f"Question: {question}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class NativeQwen3VLInference:
    _MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self):
        self.model     = None
        self.processor = None

    def load(self):
        logger.info("[Qwen3VL] Loading %s …", self._MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation=config.ATTN_IMPL,
            device_map=config.QWEN_DEVICE,
        )
        self.processor = AutoProcessor.from_pretrained(self._MODEL_ID, use_fast=False)
        logger.info("[Qwen3VL] Model loaded.")

    def infer(self, video_path: str, prompt: str) -> str:
        video_content: dict = {
            "type"      : "video",
            "video"     : video_path,
            "max_pixels": NATIVE_MAX_PIXELS,
        }
        if NATIVE_FPS is not None:
            video_content["fps"] = NATIVE_FPS

        messages = [{"role": "user", "content": [
            video_content,
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Qwen3-VL requires the metadata/kwargs dance
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_tensors, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_tensors)
            video_kwargs["video_metadata"] = list(video_metadatas)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(config.QWEN_DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.ANSWER_MAX_TOKENS,
                do_sample=False,
                temperature=None,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def unload(self):
        import gc
        del self.model; del self.processor
        self.model = self.processor = None
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
        raise ValueError(f"Unknown dataset: {config.DATASET!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────
def run_native_sample(model, item: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("uid=%s | %s", uid, os.path.basename(video_path))

    prompt = _build_prompt(question, answer_choices)
    raw = model.infer(video_path, prompt)

    predicted = extract_answer_letter(raw) if answer_choices else ""
    elapsed = time.time() - t0
    logger.info("  predicted=%r  gt=%r  (%.1fs)", predicted, ground_truth, elapsed)

    return {
        "uid"              : uid,
        "predicted_letter" : predicted,
        "ground_truth"     : ground_truth,
        "raw_answer"       : raw,
        "selected_ids"     : [],
        "num_selected"     : 0,
        "context_ids"      : [],
        "num_context"      : 0,
        "total_frames"     : -1,
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
        "time_taken_secs"  : round(time.time() - t0, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("Deterministic Baseline v6 — Qwen3-VL-8B (native video)")
    logger.info("  Model       : %s", NativeQwen3VLInference._MODEL_ID)
    logger.info("  Dataset     : %s", config.DATASET)
    logger.info("  Native FPS  : %s", NATIVE_FPS)
    logger.info("  Max pixels  : %s", NATIVE_MAX_PIXELS)
    logger.info("  Seed        : %d", SEED)
    logger.info("  Num samples : %s",
                config.NUM_SAMPLES if config.NUM_SAMPLES != -1 else "all")
    logger.info("=" * 70)

    completed = load_completed_uids(
        variant=NATIVE_VARIANT, model="Qwen3-VL-8B",
    )
    logger.info("Hot-resume: %d already done — skipping.", len(completed))

    model = NativeQwen3VLInference()
    model.load()

    total = correct = skipped = 0

    for item in get_dataset_iterator():
        uid = str(item["uid"])
        if uid in completed:
            skipped += 1
            continue

        try:
            result = run_native_sample(model, item)
            save_result(result, variant=NATIVE_VARIANT, model="Qwen3-VL-8B")
            total += 1
            if (result["predicted_letter"]
                    and result["ground_truth"]
                    and result["predicted_letter"] == result["ground_truth"]):
                correct += 1
            acc = 100.0 * correct / total if total else 0.0
            logger.info("  Running accuracy: %.1f%% (%d/%d) [skipped=%d]",
                        acc, correct, total, skipped)
        except torch.cuda.OutOfMemoryError:
            logger.error("  OOM on uid=%s — skipping.", uid)
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            logger.error("  ERROR on uid=%s: %s", uid, exc, exc_info=True)
            continue

    logger.info("=" * 70)
    logger.info("Done. Processed=%d  Skipped=%d", total, skipped)
    if total > 0:
        logger.info("Final accuracy: %.2f%% (%d/%d)",
                    100.0 * correct / total, correct, total)
    model.unload()


if __name__ == "__main__":
    main()
