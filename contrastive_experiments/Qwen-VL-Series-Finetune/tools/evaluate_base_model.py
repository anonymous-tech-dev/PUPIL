"""
Evaluation of base Qwen3-VL model (no fine-tuning) for baseline numbers.
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.evaluate_contrastive import (
    load_test_data,
    compute_all_metrics,
)

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


def load_base_model(model_id: str):
    print(f"Loading base model: {model_id}")
    config = AutoConfig.from_pretrained(model_id)

    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")

    if config.model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif config.model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    elif config.model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    print("Base model loaded!")
    return model, processor


def evaluate_base_model(
    model,
    processor,
    test_data,
    batch_size: int = 4,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    nframes: int = 16,
):
    predictions = []
    references = []

    print("\nGenerating predictions with base model...")

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

        # Generate the text templates
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]

        # Explicitly sample the exact number of frames for each video in the batch
        batch_video_frames = [
            get_video_frames(sample["video"], nframes) 
            for sample in batch
        ]

        # Pass the extracted frames directly to the processor
        inputs = processor(
            text=texts,
            videos=batch_video_frames,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        batch_predictions = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        predictions.extend(batch_predictions)
        references.extend(batch_references)

    print("\nComputing metrics...")
    metrics = compute_all_metrics(predictions, references)

    results = {
        "model": "base (no fine-tuning)",
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
        ],
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen3-VL model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--nframes", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = load_base_model(args.model_id)
    test_data = load_test_data(args.test_data_path)

    results = evaluate_base_model(
        model, processor, test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        nframes=args.nframes,
    )

    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results["metrics"], f, indent=2)

    print("\n" + "="*80)
    print("BASE MODEL EVALUATION RESULTS")
    print("="*80)
    for metric, value in results["metrics"].items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*80)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()