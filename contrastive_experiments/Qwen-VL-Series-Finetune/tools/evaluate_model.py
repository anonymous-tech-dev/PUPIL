"""
===============================================================================
Model Evaluation Tool — Base Model or LoRA-Adapted
===============================================================================
Runs inference on a test set and computes:
  • BLEU (1–4), ROUGE-1/2/L  (always)
  • LLM-as-judge via Azure GPT  (--use_llm_judge, default off)

Usage:
    # Base model
    python tools/evaluate_model.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --test_data_path /path/to/test.json \
        --output_dir /path/to/results

    # With LoRA adapters
    python tools/evaluate_model.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --adapter_path /path/to/lora_checkpoint \
        --test_data_path /path/to/test.json \
        --output_dir /path/to/results

    # With LLM-as-judge (requires `az login` first)
    python tools/evaluate_model.py \
        --test_data_path /path/to/test.json \
        --output_dir /path/to/results \
        --use_llm_judge
===============================================================================
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

# ── Lazy metric imports (installed on demand) ────────────────────────────
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
# Argument parsing
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Qwen3-VL on QA test data")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--adapter_path", type=str, default=None,
                    help="Path to LoRA adapter checkpoint. None → base model.")
    p.add_argument("--test_data_path", type=str, required=True)
    p.add_argument("--video_dir", type=str, default="",
                    help="Video directory. Optional if paths in data are absolute.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--nframes", type=int, default=16)
    p.add_argument("--fps", type=float, default=2.0,
                    help="FPS for frame extraction (only used when --sampling_mode fps). Qwen default is 2.0.")
    p.add_argument("--sampling_mode", type=str, default="native",
                    choices=["native", "fps", "nframes"],
                    help="native: let Qwen use its default (fps=2.0), fps: explicit fps, nframes: fixed frame count")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")

    # LLM-as-judge
    p.add_argument("--use_llm_judge", action="store_true", default=False,
                    help="Enable LLM-as-judge scoring via Azure GPT (requires az login).")
    p.add_argument("--judge_model", type=str, default="gpt-5.1_2025-11-13",
                    help="Azure deployment name for the judge model.")
    p.add_argument("--judge_endpoint", type=str,
                    default="https://<AZURE_OPENAI_ENDPOINT>",
                    help="Azure OpenAI endpoint for the judge.")
    p.add_argument("--judge_api_version", type=str, default="2024-10-21")
    p.add_argument("--judge_max_samples", type=int, default=None,
                    help="Max samples to judge (None → all). Useful to cap cost.")

    # Video pixel budget (match training settings)
    p.add_argument("--max_seq_length", type=int, default=None,
                    help="If set, auto-compute video_max_pixels = (max_seq_length/1000)*32*32")
    p.add_argument("--video_max_pixels", type=int, default=None,
                    help="Max pixels per video frame. Overrides max_seq_length-based auto-compute.")
    p.add_argument("--video_min_pixels", type=int, default=None,
                    help="Min pixels per video frame.")

    # Full-length video evaluation (CGBench realistic setting)
    p.add_argument("--use_full_video", action="store_true", default=False,
                    help="Use full-length videos instead of pre-trimmed clue_vids.")
    p.add_argument("--full_video_dir", type=str,
                    default="/data/Pupil/CGBench/train_vids",
                    help="Directory containing full-length CGBench videos ({video_uid}.mp4).")

    # Sharding (for data-parallel multi-GPU eval)
    p.add_argument("--shard_index", type=int, default=0,
                    help="Index of this shard (0-based).")
    p.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards. >1 enables data-parallel sharding.")
    return p.parse_args()


# =========================================================================
# Model loading
# =========================================================================

def load_model(model_id, adapter_path=None, device="cuda"):
    print(f"Loading model: {model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    )
    if adapter_path is not None:
        if not _HAS_PEFT:
            raise ImportError("peft required for LoRA adapters: pip install peft")
        print(f"Loading LoRA adapters from: {adapter_path}")

        # Strip adapter_config.json fields unknown to the installed peft
        # version (e.g. corda_config, eva_config added in newer peft).
        import inspect
        from peft import LoraConfig
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            allowed = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            unknown = [k for k in cfg.keys() if k not in allowed and k != "peft_type" and k != "task_type"]
            if unknown:
                print(f"  Stripping unknown LoraConfig fields: {unknown}")
                for k in unknown:
                    cfg.pop(k, None)
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapters merged.")
        non_lora_path = os.path.join(adapter_path, "non_lora_state_dict.bin")
        if os.path.exists(non_lora_path):
            non_lora = torch.load(non_lora_path, map_location="cpu")
            model.load_state_dict(non_lora, strict=False)
            print(f"Loaded non-LoRA state dict ({len(non_lora)} tensors).")
    model.eval()
    return model


# =========================================================================
# Message building
# =========================================================================

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
        # else: native — don't set fps/nframes, qwen_vl_utils defaults to FPS=2.0
        messages[0]["content"].append(video_content)
    messages[0]["content"].append({"type": "text", "text": question})
    return messages, reference, sample.get("id", ""), sample.get("metadata", {})


# =========================================================================
# Metrics: BLEU + ROUGE
# =========================================================================

def compute_bleu(prediction: str, reference: str) -> dict:
    """Compute BLEU-1..4 for a single (prediction, reference) pair."""
    if not _HAS_NLTK:
        return {}
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
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
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 for a single pair."""
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
    """Aggregate per-sample BLEU/ROUGE into overall + per-source means."""
    if not all_sample_metrics:
        return {}

    # Collect all metric keys
    keys = [k for k in all_sample_metrics[0] if isinstance(all_sample_metrics[0][k], (int, float))]
    if not keys:
        return {}

    # Overall
    overall = {}
    for k in keys:
        vals = [m[k] for m in all_sample_metrics if k in m]
        overall[k] = sum(vals) / len(vals) if vals else 0.0

    # Per-source
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
# LLM-as-Judge
# =========================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for video question answering. 
You will be given a question, a reference (ground-truth) answer, and a model's predicted answer.
Rate the predicted answer on a scale of 1-5:
  1 = Completely wrong or irrelevant
  2 = Partially relevant but mostly incorrect  
  3 = Partially correct, captures some key points
  4 = Mostly correct with minor omissions
  5 = Fully correct and comprehensive

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<brief explanation>"}"""


def judge_with_llm(predictions_data, args):
    """Score predictions using Azure GPT as judge. Returns list of {score, reason}."""
    try:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AzureOpenAI
    except ImportError:
        print("ERROR: azure-identity and openai packages required for LLM judge.")
        print("  pip install azure-identity openai")
        return None

    print("\n══ LLM-as-Judge ══")
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "api://azure/.default")
    client = AzureOpenAI(
        azure_endpoint=args.judge_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=args.judge_api_version,
    )

    samples = predictions_data
    if args.judge_max_samples and args.judge_max_samples < len(samples):
        samples = samples[:args.judge_max_samples]

    judge_results = []
    for item in tqdm(samples, desc="LLM judging"):
        question = ""
        convs = item.get("conversations", [])
        for t in convs:
            if t.get("from", t.get("role", "")) in ("human", "user"):
                question = t.get("value", t.get("content", "")).replace("<video>", "").strip()
                break

        user_msg = (
            f"Question: {question}\n\n"
            f"Reference Answer: {item['reference']}\n\n"
            f"Model Prediction: {item['prediction']}"
        )
        try:
            response = client.chat.completions.create(
                model=args.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            judge_results.append({
                "id": item.get("id", ""),
                "score": result.get("score", 0),
                "reason": result.get("reason", ""),
            })
        except Exception as e:
            print(f"  Judge error for {item.get('id', '?')}: {e}")
            judge_results.append({"id": item.get("id", ""), "score": 0, "reason": f"error: {e}"})

    # Aggregate
    valid_scores = [r["score"] for r in judge_results if r["score"] > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    print(f"  LLM Judge: avg score = {avg_score:.2f} / 5.0  ({len(valid_scores)} valid)")

    return {"avg_score": avg_score, "max_score": 5.0, "num_judged": len(judge_results),
            "num_valid": len(valid_scores), "details": judge_results}


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Check metric dependencies
    if not _HAS_NLTK:
        print("WARNING: nltk not installed. BLEU scores will be skipped. pip install nltk")
    if not _HAS_ROUGE:
        print("WARNING: rouge-score not installed. ROUGE scores will be skipped. pip install rouge-score")

    # Compute video pixel budget
    vid_max_px = args.video_max_pixels
    vid_min_px = args.video_min_pixels
    if vid_max_px is None and args.max_seq_length is not None:
        vid_max_px = (args.max_seq_length // 1000) * 32 * 32
        print(f"Auto video_max_pixels from max_seq_length={args.max_seq_length}: {vid_max_px}")
    if vid_min_px is None and vid_max_px is not None:
        vid_min_px = vid_max_px // 4

    # Load model
    model = load_model(args.model_id, args.adapter_path, args.device)

    # Build processor with matching pixel budget
    proc_kwargs = {}
    if vid_min_px is not None:
        proc_kwargs["min_pixels"] = vid_min_px
    if vid_max_px is not None:
        proc_kwargs["max_pixels"] = vid_max_px
    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    if proc_kwargs:
        print(f"Processor pixel budget: {proc_kwargs}")

    # Load test data
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)
    total_samples = len(test_data)
    print(f"Loaded {total_samples} test samples")

    # Shard the test data (contiguous slice for this shard)
    if args.num_shards > 1:
        n = total_samples
        per = (n + args.num_shards - 1) // args.num_shards  # ceil division
        start = args.shard_index * per
        end = min(start + per, n)
        test_data = test_data[start:end]
        print(f"[Shard {args.shard_index}/{args.num_shards}] "
              f"Processing {len(test_data)} samples (indices {start}:{end})")

    # Run inference
    predictions = []
    references = []
    metadata_list = []
    sample_results = []
    all_sample_metrics = []
    start_time = time.time()

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        messages, reference, sample_id, metadata = build_messages(
            sample, args.video_dir, args.nframes, args.sampling_mode, args.fps,
            use_full_video=args.use_full_video, full_video_dir=args.full_video_dir
        )
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Canonical Qwen3-VL pattern: get video_kwargs + metadata
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

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            prediction = processor.decode(generated, skip_special_tokens=True)
        except Exception as e:
            print(f"Error on sample {i} ({sample_id}): {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(reference)
        metadata_list.append(metadata)

        # Per-sample metrics
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

    # Aggregate metrics
    metrics = aggregate_metrics(all_sample_metrics, metadata_list)
    metrics["elapsed_seconds"] = elapsed
    metrics["samples_per_second"] = len(test_data) / elapsed if elapsed > 0 else 0
    metrics["model_id"] = args.model_id
    metrics["adapter_path"] = args.adapter_path

    # ── LLM-as-judge (optional) ──────────────────────────────────────────
    if args.use_llm_judge:
        # Enrich sample_results with original conversations for the judge
        for sr, sample in zip(sample_results, test_data):
            sr["conversations"] = sample.get("conversations", [])
        judge_results = judge_with_llm(sample_results, args)
        if judge_results:
            metrics["llm_judge"] = {
                k: v for k, v in judge_results.items() if k != "details"
            }
            # Save detailed judge results separately
            judge_path = os.path.join(args.output_dir, "llm_judge_details.json")
            with open(judge_path, "w") as f:
                json.dump(judge_results["details"], f, indent=2)
            print(f"Judge details saved to: {judge_path}")

    # ── Save results ─────────────────────────────────────────────────────
    shard_suffix = f".shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    metrics_path = os.path.join(args.output_dir, f"metrics{shard_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_path = os.path.join(args.output_dir, f"predictions{shard_suffix}.json")
    with open(predictions_path, "w") as f:
        json.dump(sample_results, f, indent=2, ensure_ascii=False)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    overall = metrics.get("overall", {})
    if overall:
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

    if "llm_judge" in metrics:
        j = metrics["llm_judge"]
        print(f"\nLLM Judge: {j['avg_score']:.2f}/5.0 ({j['num_valid']} valid)")

    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
