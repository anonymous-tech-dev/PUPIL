"""
sof_dpo_generate_negatives.py — Stage-0 of the SoF-DPO pipeline.

For each train sample, generate ONE *rejected* response by ablating the modality
that the sample's `pipeline_mode` (= SoF axis) is designed to test.  The
*chosen* side is always the dataset's `ground_truth` verbatim (rationale: zero
generation noise on the positive side; SFT warm-start handles the style match).

Per-axis ablations
------------------
visual    -> ASR-only:   no frames, full transcript text in the prompt.
audio     -> Frames-only: full video frames, no transcript at all.
time      -> Single-segment-only: only the first `timestamp_segments` window's
             frames + the ASR text from that same window are exposed.
priority  -> Text-only:   no frames, no transcript; pure language-prior answer.

Sharding
--------
The script supports multi-GPU sharding via env vars:
    NUM_SHARDS, SHARD_ID, CUDA_VISIBLE_DEVICES
Each shard handles `[i for i in range(N) if i % NUM_SHARDS == SHARD_ID]`,
appends to its own JSONL, and is safe to resume (rows whose query_id is
already present are skipped).

CLI
---
    python sof_dpo_generate_negatives.py \\
        --axes visual audio time priority \\
        --shard-id 0 --num-shards 8 \\
        --out-dir ../data/negatives_qwen3vl8b \\
        --max-new-tokens 384 --max-rows -1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
import numpy as np
import decord
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._io_utils import (  # noqa: E402
    SOF_AXES,
    iter_train_rows,
    resolve_video_path,
    transcript_text,
    transcript_text_in_range,
    first_segment_seconds,
)


# ───────────────────────────────────────────────────────────────────────────
# Frame loading
# ───────────────────────────────────────────────────────────────────────────
def sample_frames(video_path: str, n: int = 24,
                  t0: float | None = None, t1: float | None = None) -> list[Image.Image]:
    """Uniform sampling, optionally clipped to a time window."""
    vr = decord.VideoReader(video_path)
    fps = float(vr.get_avg_fps()) or 25.0
    total = len(vr)
    if t0 is not None and t1 is not None and t1 > t0:
        i0 = max(0, int(t0 * fps))
        i1 = min(total - 1, int(t1 * fps))
        if i1 <= i0:
            i0, i1 = max(0, i1 - 1), min(total - 1, i0 + 1)
        idx = np.linspace(i0, i1, n, dtype=int)
    else:
        idx = np.linspace(0, total - 1, n, dtype=int)
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in frames]


# ───────────────────────────────────────────────────────────────────────────
# Per-axis prompt construction
# ───────────────────────────────────────────────────────────────────────────
SYS_TEXT = (
    "You are an expert tutor. Answer the question concisely and accurately, "
    "using ONLY the evidence available to you in this prompt. If the question "
    "cannot be answered from the available evidence, give your best guess in "
    "one or two sentences."
)


def build_messages_for_axis(row: dict, axis: str, video_path: str,
                            n_frames_full: int, n_frames_clip: int,
                            n_frames_audio: int | None = None) -> tuple[list, list[Image.Image] | None]:
    """Returns (chat-messages, frames-or-None).  Frames are passed separately
    to the processor in the same way as evaluate_base_model.py does."""
    q = row["question"].strip()
    user_content = []
    frames = None

    if axis == "visual":
        # ASR-only: no frames, transcript inserted into the user turn.
        tx = transcript_text(row, max_chars=8000) or "(no transcript available)"
        user_content.append({
            "type": "text",
            "text": f"Transcript of the video:\n\"\"\"\n{tx}\n\"\"\"\n\nQuestion: {q}",
        })

    elif axis == "audio":
        # Frames-only: full video frames, no transcript.
        # Audio axis can use a higher frame budget than other axes (the
        # frames-diagnostic showed N_FRAMES_FULL=24 is borderline-contaminated
        # by frame-starvation failures rather than missing-audio failures).
        n_audio = n_frames_audio if n_frames_audio is not None else n_frames_full
        frames = sample_frames(video_path, n=n_audio)
        user_content.append({"type": "video", "video": video_path})  # placeholder; frames passed below
        user_content.append({"type": "text", "text": f"Question: {q}"})

    elif axis == "time":
        seg = first_segment_seconds(row)
        if seg is None:
            # No segment info — fall back to full video.
            frames = sample_frames(video_path, n=n_frames_full)
            window_tx = ""
        else:
            t0, t1 = seg
            frames = sample_frames(video_path, n=n_frames_clip, t0=t0, t1=t1)
            window_tx = transcript_text_in_range(row, t0, t1, max_chars=2000)
        user_content.append({"type": "video", "video": video_path})
        if window_tx:
            user_content.append({
                "type": "text",
                "text": (
                    f"Transcript for THIS clip:\n\"\"\"\n{window_tx}\n\"\"\"\n\n"
                    f"Question: {q}"
                ),
            })
        else:
            user_content.append({"type": "text", "text": f"Question: {q}"})

    elif axis == "priority":
        # Text-only: no frames, no transcript — pure prior.
        user_content.append({"type": "text", "text": f"Question: {q}"})

    else:
        raise ValueError(f"Unknown SoF axis: {axis}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS_TEXT}]},
        {"role": "user", "content": user_content},
    ]
    return messages, frames


# ───────────────────────────────────────────────────────────────────────────
# Generation
# ───────────────────────────────────────────────────────────────────────────
def run_one(model, processor, row: dict, axis: str, video_path: str,
            n_frames_full: int, n_frames_clip: int, max_new_tokens: int,
            n_frames_audio: int | None = None) -> str:
    messages, frames = build_messages_for_axis(
        row, axis, video_path, n_frames_full, n_frames_clip,
        n_frames_audio=n_frames_audio,
    )
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if frames is not None:
        proc_kwargs["videos"] = [frames]
    inputs = processor(**proc_kwargs).to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return out.strip()


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────
def already_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                pass
    return done


def _load_model(model_id: str, sft_adapter_path: str | None):
    """Load base Qwen3-VL, optionally apply an SFT PEFT adapter + non_lora
    weights, then return a merged eager model in bf16.

    Mirrors the eval-loader patch in
    mllm_evaluation/models/qwen3_vl_finetuned.py — without loading
    `non_lora_state_dict.bin`, the SFT-trained merger (~160M params) is
    silently dropped and the "on-policy" negatives end up half on-policy.
    """
    cfg = AutoConfig.from_pretrained(model_id)
    assert cfg.model_type == "qwen3_vl", f"Expected qwen3_vl, got {cfg.model_type}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if sft_adapter_path:
        from peft import PeftModel
        print(f"  🔌 loading SFT adapter: {sft_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        non_lora_path = os.path.join(sft_adapter_path, "non_lora_state_dict.bin")
        if os.path.isfile(non_lora_path):
            print(f"  🔧 loading non-LoRA weights: {non_lora_path}", flush=True)
            sd = torch.load(non_lora_path, map_location="cpu", weights_only=False)
            current_keys = {n for n, _ in model.named_parameters()}
            ren = {}
            for k, v in sd.items():
                if k in current_keys:
                    ren[k] = v
                elif k.replace("base_model.model.", "") in current_keys:
                    ren[k.replace("base_model.model.", "")] = v
                else:
                    ren[k] = v
            missing, unexpected = model.load_state_dict(ren, strict=False)
            print(f"     loaded {len(ren)} tensors  unexpected={len(unexpected)}",
                  flush=True)
        else:
            print(f"  ℹ️  no non_lora_state_dict.bin in {sft_adapter_path} "
                  f"(merger uses base weights — only safe if freeze_merger=True at SFT)",
                  flush=True)
        model = model.merge_and_unload()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=list(SOF_AXES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct",
                    help="Base HF model id. Always loaded first.")
    ap.add_argument("--sft-adapter-path", default=None,
                    help="Optional PEFT adapter dir from SFT. If given, the "
                         "adapter is merged into the base BEFORE generation, "
                         "and non_lora_state_dict.bin (if present) is loaded "
                         "on top so the SFT-trained merger is honored. This "
                         "produces ON-POLICY negatives — required by the "
                         "Tulu-3 / 'What Matters in DPO Data' recipes.")
    ap.add_argument("--shard-id", type=int,
                    default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--n-frames-full", type=int, default=24)
    ap.add_argument("--n-frames-clip", type=int, default=8)
    ap.add_argument("--n-frames-audio", type=int, default=None,
                    help="Frame budget for the audio axis (frames-only condition). "
                         "Defaults to --n-frames-full. Set higher (e.g. 48) to avoid "
                         "contaminating the audio axis with frame-starvation failures.")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-rows", type=int, default=-1,
                    help="Cap rows per axis (debug).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ALL train rows once, then filter per-axis & per-shard ──
    all_rows = list(iter_train_rows())
    print(f"[shard {args.shard_id}/{args.num_shards}] total train rows = {len(all_rows)}",
          flush=True)

    # ── Load model once; reuse across axes ──
    print(f"[shard {args.shard_id}] loading {args.model_id} "
          f"(sft_adapter={args.sft_adapter_path}) ...", flush=True)
    model = _load_model(args.model_id, args.sft_adapter_path)
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    print(f"[shard {args.shard_id}] model loaded.", flush=True)

    for axis in args.axes:
        rows_axis = [r for r in all_rows
                     if r["annotations"]["pipeline_mode"] == axis]
        # Shard
        rows_axis = [r for i, r in enumerate(rows_axis)
                     if i % args.num_shards == args.shard_id]
        if args.max_rows > 0:
            rows_axis = rows_axis[: args.max_rows]
        out_path = out_dir / f"negatives_{axis}.shard{args.shard_id:02d}.jsonl"
        done = already_done(out_path)
        rows_axis = [r for r in rows_axis if r["query_id"] not in done]
        print(f"[shard {args.shard_id}] axis={axis:8s}  to_do={len(rows_axis):4d}  "
              f"already_done={len(done)}  out={out_path}", flush=True)

        if not rows_axis:
            continue

        with open(out_path, "a", buffering=1) as fout:
            for k, row in enumerate(rows_axis):
                vp = resolve_video_path(row)
                if vp is None and axis in ("audio", "time"):
                    print(f"  [skip] missing video for {row['query_id']}", flush=True)
                    continue
                t0 = time.time()
                try:
                    rejected = run_one(
                        model, processor, row, axis, vp or "",
                        n_frames_full=args.n_frames_full,
                        n_frames_audio=args.n_frames_audio,
                        n_frames_clip=args.n_frames_clip,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    print(f"  [err] {row['query_id']} ({axis}): {e!r}", flush=True)
                    continue
                rec = {
                    "query_id": row["query_id"],
                    "axis": axis,
                    "cognitive_category": row["annotations"]["cognitive_category"],
                    "video_path": vp,
                    "question": row["question"],
                    "chosen": row["ground_truth"],
                    "rejected": rejected,
                    "timestamp_segments": row.get("timestamp_segments", []),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if k % 10 == 0:
                    print(f"  [shard {args.shard_id}] {axis} {k+1}/{len(rows_axis)}"
                          f"  ({time.time()-t0:.1f}s)  rej_len={len(rejected)}",
                          flush=True)

    print(f"[shard {args.shard_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
