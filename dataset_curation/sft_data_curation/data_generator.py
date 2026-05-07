"""
data_generator.py
-----------------
Orchestrates CGBench → SFT data generation across all 4 strategies.

Handles:
  - Transcript loading
  - Checkpoint resume (atomic per-item saves)
  - Exponential-backoff retry
  - tqdm progress bar
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from models.base import BaseGenerator
from prompts.prompts import build_prompt, parse_response


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _load_transcript(subtitle_dir: str, qid: int) -> str:
    """Read the subtitle .txt for a given qid.  Returns '' if missing."""
    path = Path(subtitle_dir) / f"{qid}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _clue_video_path(clue_vid_dir: str, qid: int) -> str:
    return str(Path(clue_vid_dir) / f"{qid}.mp4")


def _needs_video(strategy: int) -> bool:
    return strategy in (2, 4)


def _retry_generate(
    generator: BaseGenerator,
    prompt: str,
    video_path: Optional[str],
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Call generate_response with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return generator.generate_response(prompt, video_path=video_path)
        except Exception as e:
            wait = backoff_base ** attempt
            print(
                f"\n  [retry {attempt+1}/{max_retries}] Error: {e}. "
                f"Waiting {wait:.0f}s…"
            )
            time.sleep(wait)
    raise RuntimeError(f"Generation failed after {max_retries} attempts.")


# --------------------------------------------------------------------------- #
#  Checkpoint helpers                                                          #
# --------------------------------------------------------------------------- #

def _load_checkpoint(ckpt_path: Path) -> dict[int, dict]:
    """Load existing checkpoint as {qid: enriched_record}."""
    if not ckpt_path.exists():
        return {}
    with open(ckpt_path, "r") as f:
        records = json.load(f)
    return {r["qid"]: r for r in records}


def _save_checkpoint(ckpt_path: Path, records: dict[int, dict]):
    """Atomic write – write to .tmp then rename."""
    tmp = ckpt_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(records.values()), f, indent=2, ensure_ascii=False)
    tmp.replace(ckpt_path)


# --------------------------------------------------------------------------- #
#  Main generation loop                                                        #
# --------------------------------------------------------------------------- #

def generate_sft_data(
    cgbench_path: str,
    clue_vid_dir: str,
    subtitle_dir: str,
    output_path: str,
    strategy: int,
    generator: BaseGenerator,
    max_retries: int = 3,
    save_every: int = 5,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> list[dict]:
    """
    Generate SFT-enriched records for a slice of cgbench.json.

    Parameters
    ----------
    cgbench_path:  Path to cgbench.json.
    clue_vid_dir:  Directory containing <qid>.mp4 clue videos.
    subtitle_dir:  Directory containing <qid>.txt subtitles.
    output_path:   Where to write the final JSON file.
    strategy:      1–4 (see prompts/prompts.py for description).
    generator:     Loaded BaseGenerator instance.
    max_retries:   Per-item retry attempts on failure.
    save_every:    Checkpoint every N items.
    start_idx:     First item index to process (inclusive, 0-based).
    end_idx:       Last item index to process (exclusive). None = end of list.

    Returns
    -------
    List of enriched records.
    """
    if strategy not in (1, 2, 3, 4):
        raise ValueError(f"Strategy must be 1–4, got {strategy}.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a .ckpt.json alongside the final output for resume
    ckpt_path = out_path.with_suffix(".ckpt.json")
    done: dict[int, dict] = _load_checkpoint(ckpt_path)
    if done:
        print(f"[checkpoint] Resuming – {len(done)} items already done.")

    with open(cgbench_path, "r") as f:
        all_items: list[dict] = json.load(f)

    # Apply index slice
    shard = all_items[start_idx:end_idx]
    print(f"[shard] Processing items {start_idx}–{(end_idx or len(all_items))-1} "
          f"({len(shard)} items total).")

    needs_vid = _needs_video(strategy)
    errors: list[dict] = []

    pbar = tqdm(shard, desc=f"Strategy {strategy}", unit="item", dynamic_ncols=True)
    pending_since_save = 0

    for item in pbar:
        qid = item["qid"]

        if qid in done:
            pbar.set_postfix(status="skip (cached)")
            continue

        # ---- gather inputs ------------------------------------------------
        transcript = _load_transcript(subtitle_dir, qid)
        if not transcript:
            tqdm.write(f"  [warn] No transcript for qid={qid}, using empty string.")

        vid_path: Optional[str] = None
        if needs_vid:
            vid_path = _clue_video_path(clue_vid_dir, qid)
            if not os.path.exists(vid_path):
                tqdm.write(f"  [warn] Clue video missing for qid={qid}: {vid_path}")
                vid_path = None  # degrade gracefully

        # ---- build prompt -------------------------------------------------
        prompt = build_prompt(
            strategy=strategy,
            question=item["question"],
            choices=item["choices"],
            correct_answer=item["answer"],
            transcript=transcript,
            clue_intervals=item.get("clue_intervals", []),
            domain=item.get("domain", ""),
            sub_category=item.get("sub_category", ""),
        )

        # ---- generate -----------------------------------------------------
        pbar.set_postfix(status=f"generating qid={qid}")
        try:
            raw = _retry_generate(generator, prompt, vid_path, max_retries)
            parsed = parse_response(strategy, raw)
        except Exception as e:
            tqdm.write(f"  [ERROR] qid={qid} — {e}")
            errors.append({"qid": qid, "error": str(e)})
            continue

        # ---- assemble record ----------------------------------------------
        enriched = {**item, **parsed}
        done[qid] = enriched
        pending_since_save += 1

        # ---- checkpoint ---------------------------------------------------
        if pending_since_save >= save_every:
            _save_checkpoint(ckpt_path, done)
            pending_since_save = 0
            pbar.set_postfix(status="checkpoint saved")

    # Final save
    _save_checkpoint(ckpt_path, done)
    records = list(done.values())

    # Write final output
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {len(records)} records written to {out_path}")

    if errors:
        err_path = out_path.with_suffix(".errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"[done] {len(errors)} errors logged to {err_path}")

    # Clean up checkpoint after successful full run of this shard
    if not errors and len(records) == len(shard):
        ckpt_path.unlink(missing_ok=True)
        print("[done] Checkpoint cleaned up.")

    return records