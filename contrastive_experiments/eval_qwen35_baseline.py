"""
Qwen3.5-9B Baseline Evaluation
===============================
Standalone script — no shell wrapper needed.

Usage:
    python eval_qwen35_baseline.py
    python eval_qwen35_baseline.py --test_data_path /path/to/test.json --output_dir /path/to/out
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Optional metric imports ──────────────────────────────────────────────
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False


# =========================================================================
# Args
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Baseline eval for Qwen3.5-9B")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--test_data_path", type=str,
                    default=os.path.join(os.path.dirname(__file__), "final_sft_data", "test.json"))
    p.add_argument("--video_dir", type=str, default="")
    p.add_argument("--output_dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "outputs", "baseline_qwen35_9b"))
    p.add_argument("--nframes", type=int, default=16)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--sampling_mode", type=str, default="native",
                    choices=["native", "fps", "nframes"])
    p.add_argument("--max_new_tokens", type=int, default=32768,
                    help="Max output tokens. Qwen recommends 32768 for most tasks.")
    p.add_argument("--enable_thinking", action="store_true", default=True,
                    help="Enable Qwen3.5 thinking mode (default: on, per Qwen best practices).")
    p.add_argument("--disable_thinking", action="store_true", default=False,
                    help="Disable thinking mode (instruct mode).")
    p.add_argument("--start_idx", type=int, default=0,
                    help="Start index into test data (inclusive).")
    p.add_argument("--end_idx", type=int, default=None,
                    help="End index into test data (exclusive). None = all.")
    p.add_argument("--use_full_video", action="store_true", default=False,
                    help="Use full-length videos instead of pre-trimmed clue_vids.")
    p.add_argument("--full_video_dir", type=str,
                    default="/data/Pupil/CGBench/train_vids",
                    help="Directory containing full-length CGBench videos ({video_uid}.mp4).")
    p.add_argument("--shard_index", type=int, default=0,
                    help="Index of this shard (0-based).")
    p.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards for data-parallel eval.")
    return p.parse_args()


# =========================================================================
# Helpers
# =========================================================================

THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return THINK_PATTERN.sub("", text).strip()


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_messages(sample, video_dir, nframes, sampling_mode="native", fps=2.0,
                   use_full_video=False, full_video_dir=""):
    video_path = sample.get("video", "")
    if video_path and not os.path.isabs(video_path):
        video_path = os.path.join(video_dir, video_path)

    # Resolve to full-length video if requested
    if use_full_video and full_video_dir:
        video_uid = sample.get("metadata", {}).get("video_uid", "")
        if video_uid:
            full_path = os.path.join(full_video_dir, f"{video_uid}.mp4")
            if os.path.exists(full_path):
                video_path = full_path

    conversations = sample.get("conversations", [])
    question, reference = "", ""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            question = value.replace("<video>", "").replace("<image>", "").strip()
        elif role in ("gpt", "assistant"):
            reference = value
            break

    messages = [{"role": "user", "content": []}]
    if video_path and os.path.exists(video_path):
        video_content = {"type": "video", "video": f"file://{video_path}"}
        if sampling_mode == "nframes":
            video_content["nframes"] = nframes
        elif sampling_mode == "fps":
            video_content["fps"] = fps
        messages[0]["content"].append(video_content)
    messages[0]["content"].append({"type": "text", "text": question})
    return messages, reference, sample.get("id", ""), sample.get("metadata", {})


# =========================================================================
# Metrics
# =========================================================================

def compute_bleu(prediction: str, reference: str) -> dict:
    if not _HAS_NLTK:
        return {}
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not pred_tokens or not ref_tokens:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    smooth = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"bleu_{n}"] = sentence_bleu(
            [ref_tokens], pred_tokens, weights=weights, smoothing_function=smooth
        )
    return scores


def compute_rouge(prediction: str, reference: str) -> dict:
    if not _HAS_ROUGE:
        return {}
    scorer = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }


def aggregate_metrics(all_sample_metrics, metadata_list):
    if not all_sample_metrics:
        return {}
    keys = [k for k in all_sample_metrics[0] if isinstance(all_sample_metrics[0][k], (int, float))]
    if not keys:
        return {}

    overall = {}
    for k in keys:
        vals = [m[k] for m in all_sample_metrics if k in m]
        overall[k] = sum(vals) / len(vals) if vals else 0.0

    per_source = defaultdict(lambda: defaultdict(list))
    for m, meta in zip(all_sample_metrics, metadata_list):
        src = (meta or {}).get("source", "unknown")
        for k in keys:
            if k in m:
                per_source[src][k].append(m[k])

    per_source_agg = {}
    for src, kv in per_source.items():
        per_source_agg[src] = {k: sum(v) / len(v) for k, v in kv.items()}
        per_source_agg[src]["count"] = len(next(iter(kv.values())))

    return {"overall": overall, "per_source": per_source_agg, "total": len(all_sample_metrics)}


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not _HAS_NLTK:
        print("WARNING: nltk not installed — BLEU skipped.")
    if not _HAS_ROUGE:
        print("WARNING: rouge-score not installed — ROUGE skipped.")

    # Resolve thinking flag
    enable_thinking = args.enable_thinking and not args.disable_thinking
    print(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")

    model, processor = load_model(args.model_id)

    with open(args.test_data_path) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} total test samples")

    # Shard the test data
    if args.num_shards > 1:
        n = len(test_data)
        shard_size = (n + args.num_shards - 1) // args.num_shards
        start = args.shard_index * shard_size
        end = min(start + shard_size, n)
        test_data = test_data[start:end]
        print(f"Shard {args.shard_index}/{args.num_shards}: samples [{start}:{end}] → {len(test_data)} samples")
    elif args.start_idx > 0 or args.end_idx is not None:
        end_idx = args.end_idx if args.end_idx is not None else len(test_data)
        test_data = test_data[args.start_idx:end_idx]
        print(f"Running on samples [{args.start_idx}:{end_idx}] → {len(test_data)} samples")

    sample_results = []
    all_sample_metrics = []
    metadata_list = []
    start_time = time.time()

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        messages, reference, sample_id, metadata = build_messages(
            sample, args.video_dir, args.nframes, args.sampling_mode, args.fps,
            use_full_video=args.use_full_video, full_video_dir=args.full_video_dir
        )
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True
            )
            if video_inputs is not None:
                video_tensors, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_tensors)
                video_kwargs["video_metadata"] = list(video_metadatas)

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt", padding=True,
                **video_kwargs,
            ).to(model.device)

            # Qwen3.5 recommended sampling params
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
            if enable_thinking:
                # Thinking mode: temperature=1.0, top_p=0.95, top_k=20
                gen_kwargs.update(do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
            else:
                # Instruct mode: temperature=0.7, top_p=0.8, top_k=20
                gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_prediction = processor.decode(generated, skip_special_tokens=True)

            # Strip thinking tokens if they leaked through
            prediction = strip_thinking(raw_prediction)
        except Exception as e:
            print(f"Error on sample {i} ({sample_id}): {e}")
            prediction = ""
            raw_prediction = ""

        metadata_list.append(metadata)
        sm = {}
        sm.update(compute_bleu(prediction, reference))
        sm.update(compute_rouge(prediction, reference))
        all_sample_metrics.append(sm)

        sample_results.append({
            "id": sample_id,
            "prediction": prediction,
            "reference": reference,
            "metadata": metadata,
            "metrics": sm,
        })

    elapsed = time.time() - start_time

    metrics = aggregate_metrics(all_sample_metrics, metadata_list)
    metrics["elapsed_seconds"] = elapsed
    metrics["samples_per_second"] = len(test_data) / elapsed if elapsed > 0 else 0
    metrics["model_id"] = args.model_id
    metrics["enable_thinking"] = enable_thinking

    # Save (with shard suffix if sharding)
    shard_suffix = f".shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    metrics_path = os.path.join(args.output_dir, f"metrics{shard_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_path = os.path.join(args.output_dir, f"predictions{shard_suffix}.json")
    with open(predictions_path, "w") as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    overall = metrics.get("overall", {})
    for k, v in sorted(overall.items()):
        print(f"  {k}: {v:.4f}")
    print(f"\n  Time: {elapsed:.1f}s ({metrics.get('samples_per_second', 0):.1f} samples/s)")

    per_source = metrics.get("per_source", {})
    if per_source:
        print("\nPer-Source:")
        for src, vals in sorted(per_source.items()):
            count = vals.pop("count", "?")
            summary = ", ".join(f"{k}={v:.4f}" for k, v in sorted(vals.items()))
            print(f"  {src} (n={count}): {summary}")

    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
