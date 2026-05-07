"""
sof_dpo_generate_negatives_v2.py — Stage-0 of the SoF-DPO **v2** pipeline.

Key differences vs v1 (sof_dpo_generate_negatives.py):

  * Stronger anti-abstention system prompt ("pretend you can see / hear /
    take your single best guess"). Without this, the modality-ablated negative
    is most often "I can't tell" — which then teaches DPO to NEVER abstain,
    which is the opposite of what the benchmark rewards (especially on the
    visual / priority axes).

  * Per-axis prompt construction matches the user's spec for v2:
      - visual   : transcript ON,  no video frames     (+ anti-abstain)
      - audio    : transcript OFF, video frames        (+ anti-abstain)
      - priority : transcript OFF, no video frames     (+ anti-abstain)
      - time     : transcript OFF, single-segment-only frames (+ anti-abstain)
                   (no per-window transcript anymore)

  * Hot-resumable retry loop. For each (query_id, axis) we try up to
    `--max-attempts` generations with escalating sampling (greedy, then T=0.7,
    then T=1.0). The first non-abstention is kept. If all attempts abstain,
    the LAST attempt is kept with `final_abstain=True` so the downstream
    filter can drop it cleanly while preserving the row count for diagnostics.

  * Each shard writes two JSONL files:
      attempts_<axis>.shard<NN>.jsonl  - one row per generation (full audit)
      final_<axis>.shard<NN>.jsonl     - one row per query_id  (used by stage 1)
    Resume: skip any (query_id, axis) that already has a row in `final_*`.

CLI
---
    NUM_SHARDS=8 SHARD_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python sof_dpo_generate_negatives_v2.py \
        --axes visual audio time priority \
        --out-dir ../old_dpo_revised_data_8b/negatives_v2 \
        --max-attempts 3 --max-new-tokens 384 --max-rows -1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import decord
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pairs._io_utils import (  # noqa: E402
    SOF_AXES,
    iter_train_rows,
    resolve_video_path,
    transcript_text,
    first_segment_seconds,
)
from build_pairs._abstain_utils import is_abstain, abstain_reason  # noqa: E402


# ───────────────────────────────────────────────────────────────────────────
# Frame loading (single-threaded decord, mirrors decord_only_guard)
# ───────────────────────────────────────────────────────────────────────────
def sample_frames(video_path: str, n: int = 24,
                  t0: float | None = None, t1: float | None = None) -> list[Image.Image]:
    vr = decord.VideoReader(video_path, num_threads=1)
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
# v2 anti-abstention system prompt
# ───────────────────────────────────────────────────────────────────────────
SYS_TEXT_BASE = (
    "You are an expert tutor confidently answering a student's question about "
    "a recorded video lecture.\n\n"
    "ABSOLUTE RULES — violating these makes your answer useless:\n"
    "  1. You MUST give a concrete, specific, committed answer to the question.\n"
    "  2. You are FORBIDDEN from refusing or hedging. Do NOT say things like\n"
    "     \"I cannot see\", \"I cannot tell\", \"I cannot determine\",\n"
    "     \"I don't have enough information\", \"the transcript does not\n"
    "     mention\", \"the video does not show\", \"without seeing the video\",\n"
    "     \"based on the transcript alone\", or any similar abstention.\n"
    "  3. If your evidence is partial, take your single best guess and STATE\n"
    "     IT DIRECTLY as if you were certain. A confident wrong answer is\n"
    "     more useful than a refusal.\n"
    "  4. Pretend the missing modality is fully accessible to you and produce\n"
    "     the most plausible answer a knowledgeable tutor would give.\n"
    "  5. Match the length and style of a typical short tutor's reply\n"
    "     (1-4 sentences, plain prose; markdown bold sparingly for emphasis).\n"
)

# Per-axis nudges that get appended to the system message — they tell the
# model exactly what to "pretend" so it doesn't fall back to a generic refusal.
AXIS_NUDGES = {
    "visual": (
        "\nFor THIS question you have only the lecture transcript, NOT the\n"
        "video frames. Pretend you can fully see the visuals on screen.\n"
        "Reconstruct the most plausible visual scene the speaker is referring\n"
        "to from the transcript context, your knowledge of the topic, and\n"
        "common conventions in this kind of lecture, and answer as if you\n"
        "were watching it. Never mention the missing video."
    ),
    "audio": (
        "\nFor THIS question you have only the video frames, NOT the audio /\n"
        "transcript. Pretend you can fully hear the speaker. Reconstruct the\n"
        "most plausible thing the speaker would say at the relevant moment\n"
        "from the visual context and your knowledge of the topic, and answer\n"
        "as if you had heard it. Never mention the missing audio."
    ),
    "priority": (
        "\nFor THIS question you have ONLY the question text — no video, no\n"
        "transcript. Answer purely from your prior knowledge as a subject-matter\n"
        "expert tutor. Pretend you watched the lecture and give the most\n"
        "plausible specific answer a confident expert would give. Never\n"
        "mention that you lack the video / transcript."
    ),
    "time": (
        "\nFor THIS question you have only a SHORT clip of the lecture — not\n"
        "the full video — and NO transcript. Pretend you have already watched\n"
        "the entire lecture from start to finish. Answer using the visible\n"
        "context plus your reconstruction of the rest of the lecture; never\n"
        "say the segment / video is too short or that you can't see the\n"
        "earlier or later parts."
    ),
}


def build_messages_for_axis(row: dict, axis: str, video_path: str,
                            n_frames_full: int, n_frames_clip: int,
                            n_frames_audio: int | None,
                            transcript_max_chars: int = 8000):
    """Returns (chat-messages, frames-or-None)."""
    q = row["question"].strip()
    sys_text = SYS_TEXT_BASE + AXIS_NUDGES[axis]
    user_content: list[dict] = []
    frames = None

    if axis == "visual":
        tx = transcript_text(row, max_chars=transcript_max_chars) \
             or "(transcript unavailable)"
        user_content.append({
            "type": "text",
            "text": (f"Lecture transcript:\n\"\"\"\n{tx}\n\"\"\"\n\n"
                     f"Question: {q}"),
        })

    elif axis == "audio":
        n_audio = n_frames_audio if n_frames_audio is not None else n_frames_full
        frames = sample_frames(video_path, n=n_audio)
        user_content.append({"type": "video", "video": video_path})  # placeholder
        user_content.append({"type": "text", "text": f"Question: {q}"})

    elif axis == "time":
        seg = first_segment_seconds(row)
        if seg is None:
            frames = sample_frames(video_path, n=n_frames_full)
        else:
            t0, t1 = seg
            frames = sample_frames(video_path, n=n_frames_clip, t0=t0, t1=t1)
        user_content.append({"type": "video", "video": video_path})
        user_content.append({"type": "text", "text": f"Question: {q}"})

    elif axis == "priority":
        user_content.append({"type": "text", "text": f"Question: {q}"})

    else:
        raise ValueError(f"Unknown SoF axis: {axis}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": user_content},
    ]
    return messages, frames


# ───────────────────────────────────────────────────────────────────────────
# Generation with escalating sampling
# ───────────────────────────────────────────────────────────────────────────
def _gen_kwargs_for_attempt(attempt_idx: int, max_new_tokens: int) -> dict:
    """attempt 0 = greedy, attempt 1 = T=0.7, attempt 2+ = T=1.0."""
    base = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.05)
    if attempt_idx == 0:
        return {**base, "do_sample": False, "temperature": None,
                "top_p": None, "top_k": None}
    if attempt_idx == 1:
        return {**base, "do_sample": True, "temperature": 0.7,
                "top_p": 0.9, "top_k": 50}
    return {**base, "do_sample": True, "temperature": 1.0,
            "top_p": 0.95, "top_k": 50}


def run_one(model, processor, row, axis, video_path,
            n_frames_full, n_frames_clip, n_frames_audio,
            attempt_idx: int, max_new_tokens: int) -> str:
    messages, frames = build_messages_for_axis(
        row, axis, video_path, n_frames_full, n_frames_clip, n_frames_audio,
    )
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if frames is not None:
        proc_kwargs["videos"] = [frames]
    inputs = processor(**proc_kwargs).to(model.device)

    gen_kwargs = _gen_kwargs_for_attempt(attempt_idx, max_new_tokens)
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return out.strip()


# ───────────────────────────────────────────────────────────────────────────
# Resume helpers
# ───────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _final_qids(path: Path) -> set[str]:
    return {r["query_id"] for r in _read_jsonl(path)}


def _attempt_count_per_qid(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in _read_jsonl(path):
        out[r["query_id"]] = out.get(r["query_id"], 0) + 1
    return out


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="+", default=list(SOF_AXES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--shard-id", type=int,
                    default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--n-frames-full", type=int, default=24)
    ap.add_argument("--n-frames-clip", type=int, default=8)
    ap.add_argument("--n-frames-audio", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Max generations per (query_id, axis). "
                         "Greedy → T=0.7 → T=1.0.")
    ap.add_argument("--max-rows", type=int, default=-1,
                    help="Cap rows per axis (debug).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all train rows once, partition per-axis & per-shard.
    all_rows = list(iter_train_rows())
    print(f"[shard {args.shard_id}/{args.num_shards}] total train rows = "
          f"{len(all_rows)}", flush=True)

    print(f"[shard {args.shard_id}] loading {args.model_id} ...", flush=True)
    cfg = AutoConfig.from_pretrained(args.model_id)
    assert cfg.model_type == "qwen3_vl"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    print(f"[shard {args.shard_id}] model loaded.", flush=True)

    grand_kept = 0
    grand_abstain = 0
    grand_total = 0

    for axis in args.axes:
        rows_axis = [r for r in all_rows
                     if r["annotations"]["pipeline_mode"] == axis]
        rows_axis = [r for i, r in enumerate(rows_axis)
                     if i % args.num_shards == args.shard_id]
        if args.max_rows > 0:
            rows_axis = rows_axis[: args.max_rows]

        attempts_path = out_dir / f"attempts_{axis}.shard{args.shard_id:02d}.jsonl"
        final_path = out_dir / f"final_{axis}.shard{args.shard_id:02d}.jsonl"

        done = _final_qids(final_path)
        prev_attempts = _attempt_count_per_qid(attempts_path)
        rows_todo = [r for r in rows_axis if r["query_id"] not in done]

        print(f"[shard {args.shard_id}] axis={axis:8s}  to_do={len(rows_todo):4d}  "
              f"already_done={len(done)}  "
              f"prev_attempts_no_final={sum(1 for q in prev_attempts if q not in done)}  "
              f"final={final_path}", flush=True)

        if not rows_todo:
            continue

        f_attempts = open(attempts_path, "a", buffering=1)
        f_final = open(final_path, "a", buffering=1)

        for k, row in enumerate(rows_todo):
            qid = row["query_id"]
            vp = resolve_video_path(row)
            if vp is None and axis in ("audio", "time"):
                # Cannot generate without frames for these axes.
                f_attempts.write(json.dumps({
                    "query_id": qid, "axis": axis, "attempt": -1,
                    "error": "missing video", "rejected": "",
                    "is_abstain": True,
                }, ensure_ascii=False) + "\n")
                continue

            t0 = time.time()
            chosen_attempt = None  # the one we'll write to final_*
            seen_attempts = prev_attempts.get(qid, 0)
            for attempt_idx in range(seen_attempts, args.max_attempts):
                try:
                    rejected = run_one(
                        model, processor, row, axis, vp or "",
                        n_frames_full=args.n_frames_full,
                        n_frames_clip=args.n_frames_clip,
                        n_frames_audio=args.n_frames_audio,
                        attempt_idx=attempt_idx,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    f_attempts.write(json.dumps({
                        "query_id": qid, "axis": axis, "attempt": attempt_idx,
                        "error": repr(e)[:300], "rejected": "",
                        "is_abstain": True,
                    }, ensure_ascii=False) + "\n")
                    continue

                abst = is_abstain(rejected)
                reason = abstain_reason(rejected) if abst else None
                f_attempts.write(json.dumps({
                    "query_id": qid, "axis": axis, "attempt": attempt_idx,
                    "rejected": rejected,
                    "is_abstain": abst,
                    "abstain_match": reason,
                    "len_chars": len(rejected),
                }, ensure_ascii=False) + "\n")

                if not abst:
                    chosen_attempt = (attempt_idx, rejected, False)
                    break
                # Otherwise loop to next attempt (escalates temperature).

            if chosen_attempt is None:
                # All attempts abstained. Keep the LAST one with the abstain
                # flag set so the filter drops it but we know it happened.
                chosen_attempt = (args.max_attempts - 1, rejected, True)

            attempt_idx, rejected, final_abstain = chosen_attempt
            rec = {
                "query_id": qid,
                "axis": axis,
                "cognitive_category": row["annotations"]["cognitive_category"],
                "video_path": vp,
                "question": row["question"],
                "chosen": row["ground_truth"],
                "rejected": rejected,
                "timestamp_segments": row.get("timestamp_segments", []),
                "v2_attempt_used": attempt_idx,
                "v2_final_abstain": final_abstain,
            }
            f_final.write(json.dumps(rec, ensure_ascii=False) + "\n")

            grand_total += 1
            if final_abstain:
                grand_abstain += 1
            else:
                grand_kept += 1

            if k % 5 == 0:
                ab = "ABS" if final_abstain else "OK "
                print(f"  [shard {args.shard_id}] {axis} {k+1}/{len(rows_todo)} "
                      f"att={attempt_idx} {ab} ({time.time()-t0:.1f}s) "
                      f"running: kept={grand_kept} abstain={grand_abstain}",
                      flush=True)

        f_attempts.close()
        f_final.close()

    print(f"[shard {args.shard_id}] DONE  total={grand_total}  "
          f"kept={grand_kept}  abstain={grand_abstain}", flush=True)


if __name__ == "__main__":
    main()
