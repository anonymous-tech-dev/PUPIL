"""
sof_dpo_score_margins.py — Forward-pass the *reference* Qwen3-VL-8B over each
filtered pair and compute the implicit-reward margin under the FULL multimodal
context that DPO will actually see at training time.

Why this matters
----------------
Our negatives were generated under MODALITY-ABLATED contexts.  But during DPO
they will be scored under the FULL context (frames + transcript + question).
If under the full context the reference model already considers the rejected
response far less likely than the chosen one, the DPO loss saturates and that
pair contributes essentially zero gradient — wasted compute and possibly noise.

This script computes
    margin_ref = log p_ref(chosen | full_ctx) - log p_ref(rejected | full_ctx)
for each pair, writes it back into the JSONL, and (optionally) drops pairs
above a saturation threshold.  Following Tulu-3 / RLHF-V practice we keep the
distribution and the threshold cut as separate operations so the appendix
table can show the histogram.

CLI
---
    python sof_dpo_score_margins.py \\
        --in-jsonl  ../data/pairs_after_filter.jsonl \\
        --out-jsonl ../data/pairs_with_margin.jsonl \\
        --shard-id 0 --num-shards 8 \\
        --n-frames 24 --max-seq 16384 \\
        --max-rows -1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import decord
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._io_utils import transcript_text  # noqa: E402


SYS_TEXT = (
    "You are an expert tutor. Answer the question concisely and accurately, "
    "using ONLY the evidence available to you in this prompt. If the question "
    "cannot be answered from the available evidence, give your best guess in "
    "one or two sentences."
)


def sample_uniform_frames(video_path: str, n: int) -> list[Image.Image]:
    vr = decord.VideoReader(video_path)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    return [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]


def build_full_context_messages(question: str, transcript: str,
                                video_path: str | None) -> list[dict]:
    user_content: list[dict] = []
    if video_path:
        user_content.append({"type": "video", "video": video_path})
    if transcript:
        user_content.append({
            "type": "text",
            "text": f"Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\nQuestion: {question}",
        })
    else:
        user_content.append({"type": "text", "text": f"Question: {question}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYS_TEXT}]},
        {"role": "user", "content": user_content},
    ]


@torch.no_grad()
def completion_logp(model, processor, prompt_ids: torch.Tensor,
                    completion_text: str,
                    vision_inputs: dict) -> tuple[float, int]:
    """Return (sum_token log p(completion | prompt, vision), n_tokens)."""
    tok = processor.tokenizer
    comp = tok(completion_text + tok.eos_token, add_special_tokens=False,
               return_tensors="pt")["input_ids"][0].to(prompt_ids.device)
    full = torch.cat([prompt_ids, comp], dim=0).unsqueeze(0)
    attn = torch.ones_like(full)
    inputs = {"input_ids": full, "attention_mask": attn}
    inputs.update({k: v.to(prompt_ids.device) if torch.is_tensor(v) else v
                   for k, v in vision_inputs.items()})
    out = model(**inputs)
    logits = out.logits[0, :-1, :]            # predict positions 1..N
    targets = full[0, 1:]
    n_comp = comp.shape[0]
    logits = logits[-n_comp:, :]
    targets = targets[-n_comp:]
    logp = F.log_softmax(logits.float(), dim=-1)
    chosen_lp = logp.gather(1, targets.unsqueeze(1)).squeeze(1).sum()
    return float(chosen_lp.cpu()), int(n_comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--shard-id", type=int,
                    default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--max-rows", type=int, default=-1)
    ap.add_argument("--transcript-max-chars", type=int, default=6000)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    rows = [json.loads(l) for l in open(in_path) if l.strip()]
    rows = [r for i, r in enumerate(rows)
            if i % args.num_shards == args.shard_id]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Always per-shard suffix so glob patterns downstream are uniform.
    out_path = out_path.with_suffix("").with_suffix(f".shard{args.shard_id:02d}.jsonl")
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(f"{r['query_id']}::{r['axis']}")
                except Exception:
                    pass

    print(f"[shard {args.shard_id}/{args.num_shards}] in={len(rows)}  "
          f"already_done={len(done)}  out={out_path}", flush=True)

    cfg = AutoConfig.from_pretrained(args.model_id)
    assert cfg.model_type == "qwen3_vl"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    # Pre-load training rows to fetch transcripts (id-keyed map of question→transcript
    # is rebuilt on-the-fly per pair, since pairs already carry the question text).
    fout = open(out_path, "a", buffering=1)
    for k, rec in enumerate(rows):
        qid_ax = f"{rec['query_id']}::{rec['axis']}"
        if qid_ax in done:
            continue
        t0 = time.time()

        # Build the FULL-context prompt for both completions.
        # We re-resolve transcript from the .srt file using the row's video_path.
        # (For pairs without a usable video, fall back to text-only.)
        # We synthesise a stub `row` dict for transcript_text():
        stub = {"video_path": rec.get("video_path") or ""}
        tx = transcript_text(stub, max_chars=args.transcript_max_chars) if stub["video_path"] else ""
        vp = rec.get("video_path")

        messages = build_full_context_messages(rec["question"], tx, vp)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)

        proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if vp:
            try:
                frames = sample_uniform_frames(vp, args.n_frames)
                proc_kwargs["videos"] = [frames]
            except Exception as e:
                print(f"  [warn] frame sampling failed for {qid_ax}: {e!r}", flush=True)
        try:
            inputs = processor(**proc_kwargs).to(model.device)
        except Exception as e:
            print(f"  [skip] processor failed for {qid_ax}: {e!r}", flush=True)
            continue

        prompt_ids = inputs["input_ids"][0]
        vision_kwargs: dict = {}
        for k_ in ("pixel_values_videos", "video_grid_thw",
                   "second_per_grid_ts", "pixel_values", "image_grid_thw"):
            if k_ in inputs:
                vision_kwargs[k_] = inputs[k_]

        try:
            lp_c, n_c = completion_logp(model, processor, prompt_ids,
                                        rec["chosen"], vision_kwargs)
            lp_r, n_r = completion_logp(model, processor, prompt_ids,
                                        rec["rejected"], vision_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [oom] skipping {qid_ax}", flush=True)
            continue

        margin = lp_c - lp_r
        margin_norm = (lp_c / max(1, n_c)) - (lp_r / max(1, n_r))
        rec["ref_logp_chosen"] = round(lp_c, 4)
        rec["ref_logp_rejected"] = round(lp_r, 4)
        rec["ref_n_tok_chosen"] = n_c
        rec["ref_n_tok_rejected"] = n_r
        rec["ref_margin"] = round(margin, 4)
        rec["ref_margin_per_tok"] = round(margin_norm, 4)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 10 == 0:
            print(f"  [shard {args.shard_id}] {k+1}/{len(rows)}  "
                  f"axis={rec['axis']:8s}  margin={margin:+.2f} "
                  f"({margin_norm:+.3f}/tok)  ({time.time()-t0:.1f}s)",
                  flush=True)

    fout.close()
    print(f"[shard {args.shard_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
