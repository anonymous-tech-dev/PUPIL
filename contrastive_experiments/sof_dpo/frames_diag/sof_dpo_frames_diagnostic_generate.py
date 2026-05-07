"""
sof_dpo_frames_diagnostic_generate.py

For the AUDIO axis only, sample a small set of train rows and generate the
ablated (frames-only, no transcript) negative TWICE per row:
    * once with N_FRAMES_FULL=24  (current default)
    * once with N_FRAMES_FULL=64  (high-budget reference)

We later pairwise-judge "which one's wrongness is more clearly attributable
to lacking the spoken transcript" (vs. lacking visual detail) so we know
whether 24 frames is contaminating the audio-axis signal with frame-starvation.

Output: one JSONL file with {query_id, question, ground_truth, neg_24, neg_64}.
Designed to run on a single GPU in <10 minutes for ~40 rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._io_utils import iter_train_rows, resolve_video_path  # noqa: E402
from build_pairs.sof_dpo_generate_negatives import (  # noqa: E402
    SYS_TEXT, sample_frames,
)


def build_audio_messages(question: str, video_path: str):
    """Audio-axis ablation = frames only, no transcript."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYS_TEXT}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": f"Question: {question}"},
        ]},
    ]


@torch.no_grad()
def generate_one(model, processor, row: dict, video_path: str, n_frames: int,
                 max_new_tokens: int) -> str:
    msgs = build_audio_messages(row["question"], video_path)
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    frames = sample_frames(video_path, n=n_frames)
    inputs = processor(text=[text], videos=[frames], padding=True,
                       return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None,
    )
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--n-rows", type=int, default=40)
    ap.add_argument("--n-frames-low", type=int, default=24)
    ap.add_argument("--n-frames-high", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    rng = __import__("random").Random(args.seed)
    rows = [r for r in iter_train_rows()
            if r["annotations"]["pipeline_mode"] == "audio"]
    rng.shuffle(rows)
    keep: list[tuple[dict, str]] = []
    for r in rows:
        vp = resolve_video_path(r)
        if vp:
            keep.append((r, vp))
        if len(keep) >= args.n_rows:
            break
    print(f"selected {len(keep)} audio rows with resolvable videos", flush=True)

    cfg = AutoConfig.from_pretrained(args.model_id)
    assert cfg.model_type == "qwen3_vl"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                pass

    fout = open(out_path, "a", buffering=1)
    for k, (row, vp) in enumerate(keep):
        if row["query_id"] in done:
            continue
        t0 = time.time()
        try:
            n_lo = generate_one(model, processor, row, vp,
                                args.n_frames_low, args.max_new_tokens)
            n_hi = generate_one(model, processor, row, vp,
                                args.n_frames_high, args.max_new_tokens)
        except Exception as e:
            print(f"  [err] {row['query_id']}: {e!r}", flush=True)
            continue
        rec = {
            "query_id": row["query_id"],
            "video_path": vp,
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "neg_low":  n_lo,
            "neg_high": n_hi,
            "n_frames_low":  args.n_frames_low,
            "n_frames_high": args.n_frames_high,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {k+1}/{len(keep)}  ({time.time()-t0:.1f}s)  "
              f"|low|={len(n_lo)}  |high|={len(n_hi)}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
