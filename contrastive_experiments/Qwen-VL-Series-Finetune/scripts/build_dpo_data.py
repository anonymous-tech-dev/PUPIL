#!/usr/bin/env python3
"""
build_dpo_data.py
-----------------
Convert the Pupil training queries (and optionally CGBench/FineVideo)
into a flat DPO JSON file consumable by ``src/train/train_dpo.py``.

Each output sample::

    {
      "id":       "<query_id>",
      "source":   "edubench" | "cgbench" | "finevideo",
      "video":    "/abs/path/to/video.mp4",
      "prompt":   "<video>\\n<question>",
      "chosen":   "<ground-truth answer>",
      "rejected": "<a worse / incorrect answer>"
    }

Sources of `rejected`
=====================

``--mode shuffle``      (cheap, default for smoke tests):
    Pick the ground-truth answer of *another* query as the rejected response.
    Useful for plumbing tests — you'll see DPO loss drop because the
    reference model also strongly disprefers off-topic answers.

``--mode predictions``  (recommended for real runs):
    Read a model-predictions JSON (the format produced by
    ``tools/evaluate_model.py``) and use the model's own (typically wrong)
    output as the rejected answer.  Pass ``--predictions PATH``.

``--mode truncate``     (no model needed, gives natural negatives):
    Use a heavily truncated / paraphrased version of the GT as rejected.
    Currently implemented as ``first-N-words`` truncation.

Examples
--------
Smoke-test build (1 video, 5 samples)::

    python scripts/build_dpo_data.py \\
        --edubench_dir /workspace/Pupil/dataset_curation/dataset/queries_db/final_train_1k \\
        --edubench_video_dir /datadisk/edubench_train_vids/train_vids \\
        --output /workspace/.../dpo_smoke.json \\
        --mode shuffle --max_samples 5 --seed 0

Full build from baseline predictions::

    python scripts/build_dpo_data.py \\
        --edubench_dir .../final_train_1k \\
        --edubench_video_dir .../train_vids \\
        --predictions .../baseline_qwen3vl_8b_full_video/predictions.json \\
        --output .../dpo_train.json --mode predictions
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_cgbench(cgbench_json: str, video_dir: str,
                 video_kind: str = "clue") -> List[Dict[str, Any]]:
    """Load CGBench MCQ entries.

    `video_kind` selects which video to use:
        - "clue": short clip per qid (``{qid}.mp4`` under video_dir).
                   Fast, recommended for smoke tests.
        - "full": full-length video per video_uid (``{video_uid}.mp4``).
                   Heavier but matches eval distribution.
    """
    with open(cgbench_json) as f:
        raw = json.load(f)
    out: List[Dict[str, Any]] = []
    for q in raw:
        qid = q.get("qid")
        question = q.get("question", "").strip()
        choices = q.get("choices") or q.get("options") or []
        right_letter = q.get("right_answer") or q.get("answer_idx")
        right_text = q.get("answer") or q.get("answer_text")
        if not (qid is not None and question and choices and right_letter):
            continue

        if video_kind == "clue":
            video_path = os.path.join(video_dir, f"{qid}.mp4")
        elif video_kind == "full":
            video_path = os.path.join(video_dir, f"{q['video_uid']}.mp4")
        else:
            raise ValueError(f"Unknown video_kind: {video_kind}")

        # MCQ-style prompt with lettered options (matches the eval format
        # used elsewhere in the repo).
        labelled = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
        )
        question_full = f"{question}\n\nOptions:\n{labelled}\n\nAnswer with the letter of the correct option."

        # Index of the correct choice
        try:
            right_idx = ord(right_letter.upper()) - ord("A")
        except Exception:
            right_idx = None

        out.append({
            "id": f"cgbench_{qid}",
            "source": "cgbench",
            "video": video_path,
            "question": question_full,
            "ground_truth": f"{right_letter}. {right_text}" if right_text else right_letter,
            "_choices": choices,
            "_right_idx": right_idx,
            "_right_letter": right_letter,
            "_duration_sec": float(q.get("duration") or 0),
            "_sub_category": q.get("sub_category", "?"),
            "_domain": q.get("domain", "?"),
        })
    return out


def load_edubench(edubench_dir: str, video_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Walk all ``*_queries.json`` files and emit one record per query."""
    files = sorted(glob.glob(os.path.join(edubench_dir, "**", "*_queries.json"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_queries.json under {edubench_dir}")
    out: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        for raw_video_path, queries in data.items():
            video_basename = os.path.basename(raw_video_path)
            video_path = (
                os.path.join(video_dir, video_basename) if video_dir else raw_video_path
            )
            for q in queries:
                if not q.get("question") or not q.get("ground_truth"):
                    continue
                out.append({
                    "id": q["query_id"],
                    "source": "edubench",
                    "video": video_path,
                    "question": q["question"].strip(),
                    "ground_truth": q["ground_truth"].strip(),
                })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# token-budget estimation (mirrors src/dataset/dpo_dataset.py math)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_video_tokens(duration_sec: float, fps: float, max_seq_length: int,
                          video_max_pixels: int, video_min_pixels: int,
                          image_patch_size: int = 16) -> int:
    """Estimate #video tokens after the Qwen3-VL processor's smart_resize +
    temporal merge.  Mirrors the formula in dpo_dataset.py:108-140 so we can
    filter out samples that will overflow max_seq_length BEFORE training."""
    MERGE = 2
    factor_sq = (image_patch_size * MERGE) ** 2  # 1024
    FRAME_FACTOR = 2

    video_token_budget = int(max_seq_length * 0.85)
    min_tok_per_frame = max(1, video_min_pixels // factor_sq)
    video_total_pixels = video_token_budget * factor_sq // FRAME_FACTOR
    video_max_frames = max(
        FRAME_FACTOR,
        (video_token_budget // min_tok_per_frame) // FRAME_FACTOR * FRAME_FACTOR,
    )

    # Effective frame count after FPS sampling and the dataset's frame cap
    raw_nframes = max(FRAME_FACTOR, int(fps * duration_sec))
    nframes = min(raw_nframes, video_max_frames)

    # Per-frame pixel budget (smart_resize: min(max_pixels, total_pixels/nframes))
    per_frame_px = min(video_max_pixels, video_total_pixels // max(1, nframes))
    tokens_per_frame_after_spatial_merge = per_frame_px // factor_sq

    # Temporal merge factor of 2: every 2 consecutive frames share tokens
    return (nframes * tokens_per_frame_after_spatial_merge) // FRAME_FACTOR


# ─────────────────────────────────────────────────────────────────────────────
# rejected-answer strategies
# ─────────────────────────────────────────────────────────────────────────────
def rejected_shuffle(records: List[Dict[str, Any]], seed: int) -> List[str]:
    """Pair each record with another record's GT (different video preferred)."""
    rng = random.Random(seed)
    rejected: List[str] = [""] * len(records)
    indices = list(range(len(records)))
    for i, rec in enumerate(records):
        # try up to 8 times to find a record from a different video
        for _ in range(8):
            j = rng.choice(indices)
            if j != i and records[j].get("video") != rec.get("video"):
                rejected[i] = records[j]["ground_truth"]
                break
        if not rejected[i]:
            j = (i + 1) % len(records)
            rejected[i] = records[j]["ground_truth"]
    return rejected


def rejected_truncate(records: List[Dict[str, Any]], min_words: int = 4,
                      max_words: int = 12) -> List[str]:
    rng = random.Random(0)
    out = []
    for rec in records:
        words = re.findall(r"\S+", rec["ground_truth"])
        if not words:
            out.append("I don't know.")
            continue
        n = min(len(words), rng.randint(min_words, max_words))
        out.append(" ".join(words[:n]).rstrip(".,;:") + ".")
    return out


def rejected_predictions(records: List[Dict[str, Any]],
                         predictions_path: str) -> Tuple[List[str], List[bool]]:
    """Look up model predictions by query id; missing → empty (record dropped)."""
    with open(predictions_path) as f:
        preds = json.load(f)
    pred_lookup: Dict[str, str] = {}
    for p in preds:
        # Several formats observed: "id" / "query_id" / "qid"
        pid = p.get("id") or p.get("query_id") or p.get("qid")
        text = p.get("prediction") or p.get("output") or p.get("response")
        if pid and text:
            pred_lookup[str(pid)] = text.strip()

    rejected, keep = [], []
    for rec in records:
        text = pred_lookup.get(str(rec["id"]))
        if text and text.strip() != rec["ground_truth"].strip():
            rejected.append(text)
            keep.append(True)
        else:
            rejected.append("")
            keep.append(False)
    return rejected, keep


def rejected_mcq_wrong(records: List[Dict[str, Any]], seed: int) -> List[str]:
    """For CGBench MCQs: pick a random WRONG option as the rejected answer.

    Falls back to ``rejected_shuffle`` for any record without ``_choices``.
    """
    rng = random.Random(seed)
    out: List[str] = []
    for rec in records:
        choices = rec.get("_choices") or []
        right_idx = rec.get("_right_idx")
        if choices and right_idx is not None and len(choices) > 1:
            wrong_idxs = [i for i in range(len(choices)) if i != right_idx]
            j = rng.choice(wrong_idxs)
            wrong_letter = chr(ord("A") + j)
            out.append(f"{wrong_letter}. {choices[j]}")
        else:
            out.append("")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default="edubench",
                    choices=["edubench", "cgbench"],
                    help="Which dataset to load")
    # EduBench knobs
    ap.add_argument("--edubench_dir", default=None,
                    help="Root with sof_*/*_queries.json (e.g. final_train_1k)")
    ap.add_argument("--edubench_video_dir", default=None,
                    help="Override the video directory (replaces file basename only)")
    # CGBench knobs
    ap.add_argument("--cgbench_json", default=None,
                    help="Path to cgbench.json (or cgbench_mini.json)")
    ap.add_argument("--cgbench_video_dir", default=None,
                    help="Directory containing CGBench videos")
    ap.add_argument("--cgbench_video_kind", default="clue",
                    choices=["clue", "full"],
                    help="Which CGBench video to use: per-qid clue clip or full video")
    # Output / mode
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--mode", default="shuffle",
                    choices=["shuffle", "truncate", "predictions", "mcq_wrong"],
                    help="Strategy for picking the 'rejected' answer. "
                         "'mcq_wrong' is only valid for CGBench (picks a wrong option).")
    ap.add_argument("--predictions", default=None,
                    help="JSON of model predictions (used when --mode predictions)")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--val_fraction", type=float, default=0.0,
                    help="If >0, write {output}.val.json with this fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_video_exists", action="store_true",
                    help="Drop samples whose video file is missing on disk")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.source == "edubench":
        if not args.edubench_dir:
            sys.exit("--edubench_dir required for --source edubench")
        print(f"[build_dpo_data] Loading Pupil from {args.edubench_dir}")
        records = load_edubench(args.edubench_dir, args.edubench_video_dir)
    elif args.source == "cgbench":
        if not args.cgbench_json or not args.cgbench_video_dir:
            sys.exit("--cgbench_json and --cgbench_video_dir required for --source cgbench")
        print(f"[build_dpo_data] Loading CGBench from {args.cgbench_json} "
              f"(videos: {args.cgbench_video_dir}, kind={args.cgbench_video_kind})")
        records = load_cgbench(args.cgbench_json, args.cgbench_video_dir,
                               args.cgbench_video_kind)
    else:
        raise AssertionError(args.source)
    print(f"  → {len(records)} raw queries")

    if args.require_video_exists:
        before = len(records)
        records = [r for r in records if os.path.exists(r["video"])]
        print(f"  → {len(records)} after video-existence filter ({before-len(records)} dropped)")

    rng.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]
        print(f"  → {len(records)} after --max_samples cap")

    if args.mode == "shuffle":
        rejected = rejected_shuffle(records, args.seed)
        keep = [bool(r) for r in rejected]
    elif args.mode == "truncate":
        rejected = rejected_truncate(records)
        keep = [bool(r) for r in rejected]
    elif args.mode == "predictions":
        if not args.predictions:
            sys.exit("--predictions PATH required when --mode predictions")
        rejected, keep = rejected_predictions(records, args.predictions)
    elif args.mode == "mcq_wrong":
        rejected = rejected_mcq_wrong(records, args.seed)
        keep = [bool(r) for r in rejected]
    else:
        raise AssertionError(args.mode)

    samples: List[Dict[str, Any]] = []
    for rec, rej, k in zip(records, rejected, keep):
        if not k:
            continue
        samples.append({
            "id": rec["id"],
            "source": rec["source"],
            "video": rec["video"],
            "prompt": "<video>\n" + rec["question"],
            "chosen": rec["ground_truth"],
            "rejected": rej,
        })
    print(f"  → {len(samples)} usable DPO pairs")

    # Optional train/val split
    if args.val_fraction > 0:
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * args.val_fraction))
        val_samples, train_samples = samples[:n_val], samples[n_val:]
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        val_path = os.path.splitext(args.output)[0] + ".val.json"
        with open(val_path, "w") as f:
            json.dump(val_samples, f, ensure_ascii=False, indent=2)
        print(f"  ✓ wrote {len(train_samples)} train → {args.output}")
        print(f"  ✓ wrote {len(val_samples)} val   → {val_path}")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"  ✓ wrote {len(samples)} samples → {args.output}")


if __name__ == "__main__":
    main()
