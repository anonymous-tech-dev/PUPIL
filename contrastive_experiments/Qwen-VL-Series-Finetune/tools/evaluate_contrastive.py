"""
Evaluation tool for Contrastive SFT models.
Computes accuracy, BLEU, ROUGE, and other metrics on test set.
"""
import os
import json
import torch
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trainer.contrastive_sft_trainer import GenerativeEvalPrediction

import numpy as np
import decord
from PIL import Image

def get_video_frames(video_path, nframes):
    """Uniformly samples 'nframes' from a video and returns them as PIL Images."""
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    
    # Generate uniform indices across the video
    indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
    
    # Extract frames and convert to list of PIL Images for the processor
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def load_model_and_processor(model_path: str, device: str = "cuda"):
    """Load trained model and processor correctly (supports LoRA + merged)."""
    from transformers import (
        Qwen3VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        AutoConfig
    )
    print(f"Loading model from {model_path}...")
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    # -------------------------------------------------
    # CASE 1: LoRA checkpoint
    # -------------------------------------------------
    if os.path.exists(adapter_config_path):
        print("LoRA adapter detected. Loading clean base model...")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError("adapter_config missing base_model_name_or_path")
        print(f"Base model: {base_model_name}")
        config = AutoConfig.from_pretrained(base_model_name)
        if config.model_type == "qwen3_vl_moe":
            base_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        elif config.model_type == "qwen3_vl":
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        elif config.model_type == "qwen2_5_vl":
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
        print("Applying LoRA weights...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()

    # -------------------------------------------------
    # CASE 2: Fully merged model
    # -------------------------------------------------
    else:
        print("No LoRA adapter found. Loading merged model.")
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "qwen3_vl_moe":
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        elif config.model_type == "qwen3_vl":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        elif config.model_type == "qwen2_5_vl":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )

    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    print("Model loaded successfully!")
    return model, processor


def load_test_data(test_data_path: str) -> List[Dict[str, Any]]:
    """Load test dataset."""
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")
    return data


def compute_exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute exact match accuracy (case-insensitive)."""
    correct = sum(
        pred.strip().lower() == ref.strip().lower()
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions) if predictions else 0.0


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute token-level accuracy (for short answers)."""
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if ref_tokens:
            correct += len(pred_tokens & ref_tokens)
            total += len(ref_tokens)
    return correct / total if total > 0 else 0.0


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
    except ImportError:
        print("Warning: nltk not installed. Skipping BLEU score.")
        return 0.0
    smoothie = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Warning: rouge-score not installed. Skipping ROUGE scores.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {
        "exact_match": compute_exact_match_accuracy(predictions, references),
        "token_accuracy": compute_token_accuracy(predictions, references),
        "bleu": compute_bleu_score(predictions, references),
    }
    rouge_scores = compute_rouge_scores(predictions, references)
    metrics.update(rouge_scores)
    return metrics


def evaluate_model(
    model,
    processor,
    test_data: List[Dict[str, Any]],
    model_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    nframes: int = 16,
) -> Dict[str, Any]:
    """Evaluate model on test data."""
    predictions = []
    references = []

    print("\nGenerating predictions...")

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]

        batch_messages = []
        batch_references = []

        for sample in batch:
            conversations = sample["conversations"]
            question = conversations[0]["value"]
            answer = conversations[1]["value"]

            message = {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample["video"], "nframes": nframes},
                    {"type": "text", "text": question.replace("<video>\n", "").strip()},
                ]
            }

            batch_messages.append([message])
            batch_references.append(answer)

        # texts = [
        #     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        #     for msg in batch_messages
        # ]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Explicitly extract the frames
        batch_video_frames = [
            get_video_frames(sample["video"], nframes) 
            for sample in batch
        ]

        # Pass the extracted PIL Images to the processor
        inputs = processor(
            text=texts,
            videos=batch_video_frames,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # inputs = processor(
        #     text=texts,
        #     videos=[sample["video"] for sample in batch],
        #     padding=True,
        #     return_tensors="pt",
        # ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_predictions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        predictions.extend(batch_predictions)
        references.extend(batch_references)

    print("\nComputing metrics...")
    metrics = compute_all_metrics(predictions, references)

    results = {
        "model": model_path,                        # mirrors baseline's "model" key
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "samples": [
            {
                "id": test_data[i]["id"],
                "prediction": predictions[i],
                "reference": references[i],
            }
            for i in range(min(10, len(predictions)))
        ]
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Contrastive SFT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--do_sample", type=lambda x: x.lower() == 'true', default=False, help="Whether to use sampling")
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample from video")
    parser.add_argument("--compute_metrics", type=lambda x: x.lower() == 'true', default=True, help="Compute metrics")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_model_and_processor(args.model_path)
    test_data = load_test_data(args.test_data_path)

    results = evaluate_model(
        model,
        processor,
        test_data,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        nframes=args.nframes,
    )

    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results["metrics"], f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for metric, value in results["metrics"].items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*80)
    print(f"\nFull results saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()