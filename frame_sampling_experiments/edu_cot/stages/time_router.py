"""
stages/time_router.py — VLM time-range routing.

Idea: show the VLM a small uniform overview of the video (e.g., 32 frames
labeled with their timestamps), the question, and the answer choices.
Ask it to return a list of time ranges (in seconds) that look most relevant.
Then map those ranges back to scene-detect segments and use those segments
(plus neighbor hops) as the candidate pool for context aggregation.

Why this might beat scene_zoom v1:
  - Overview frames are well-spaced, low-noise gist of the whole video
    (vs scene_zoom's 1-frame-per-scene which is a noisy 100+ frame collage).
  - Scene boundaries do the heavy lifting for *which exact frames* to pull.
    The VLM only has to express *roughly when*, in seconds.
  - Single VLM call (cheap), JSON output (easy to parse).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from stages.segmentation import Segment
from stages.video_loading import VideoMeta, decode_frames

logger = logging.getLogger("educot.time_router")


TIME_ROUTER_PROMPT = (
    "You will be given a question about a video and {num_choices} possible "
    "answer options. Below are {n_overview} frames sampled uniformly from "
    "the video, labeled with their timestamps in MM:SS format.\n"
    "{frame_labels}\n"
    "Question: {question}\n"
    "Possible answer choices: {answer_choices}\n\n"
    "Identify the time ranges in the video that are most relevant to "
    "answering the question. Each range should be expressed in seconds "
    "as [start_sec, end_sec]. Return between 1 and {max_ranges} ranges, "
    "favoring tighter ranges when the relevant moment is clear and wider "
    "ones when uncertain. The video is {duration_sec:.0f} seconds long.\n"
    "Respond with ONLY a JSON object and nothing else. Example format:\n"
    '{{"time_ranges": [[12, 28], [95, 130]], "justification": "the events '
    'at 0:15 and around 1:40 contain the answer."}}'
)


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _parse_time_ranges(
    raw: str, duration_sec: float
) -> Tuple[List[Tuple[float, float]], str]:
    """Parse VLM JSON output → (list of (start, end), justification)."""
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return [], "parse_no_json"
    try:
        data = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError):
        return [], "parse_bad_json"

    raw_ranges = data.get("time_ranges", [])
    justification = data.get("justification", "")

    out: List[Tuple[float, float]] = []
    for item in raw_ranges:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            s = float(item[0])
            e = float(item[1])
        except (TypeError, ValueError):
            continue
        if e < s:
            s, e = e, s
        s = max(0.0, min(s, duration_sec))
        e = max(0.0, min(e, duration_sec))
        if e <= s:
            continue
        out.append((s, e))
    return out, justification


def time_router_select(
    model,
    video_path: str,
    segments: List[Segment],
    question: str,
    answer_choices: List[str],
    meta: VideoMeta,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    1. Sample U uniform frames across the whole video.
    2. VLM call → JSON of relevant time ranges.
    3. Map ranges → segments that overlap → optional neighbor expansion.
    4. Return all frames in chosen segments.
    """
    tr_cfg = cfg.time_router
    n_overview = max(4, int(tr_cfg.overview_frames))
    neighbor_hops = max(0, int(tr_cfg.neighbor_hops))
    max_ranges = max(1, int(tr_cfg.max_ranges))
    fallback_keep_frac = float(getattr(tr_cfg, "fallback_keep_frac", 0.5))

    nonempty_segments = [seg for seg in segments if seg.frame_ids]
    if not nonempty_segments:
        return {
            "selected_ids": [],
            "justification": "time_router_empty_segments",
            "raw_response": "",
            "selected_scene_indices": [],
            "time_ranges": [],
        }

    duration_sec = max(meta.duration, 1.0)

    # ── Step 1: build uniform overview (frame_ids in target_fps space) ──
    # All-frame ID set comes from the segments (which already use target_fps).
    all_fids: List[int] = []
    for seg in nonempty_segments:
        all_fids.extend(seg.frame_ids)
    all_fids = sorted(set(all_fids))
    if not all_fids:
        return {
            "selected_ids": [],
            "justification": "time_router_no_frames",
            "raw_response": "",
            "selected_scene_indices": [],
            "time_ranges": [],
        }

    if len(all_fids) <= n_overview:
        overview_ids = all_fids
    else:
        idx = np.linspace(0, len(all_fids) - 1, n_overview).round().astype(int)
        overview_ids = [all_fids[i] for i in np.unique(idx)]

    target_fps = max(1, int(meta.target_fps))
    overview_secs = [fid / target_fps for fid in overview_ids]

    # ── Step 2: VLM call ──
    t_dec = time.time()
    overview_frames = decode_frames(
        video_path, overview_ids, meta.native_fps, meta.target_fps,
    )
    logger.info("  TimeRouter: decoded %d overview frames (%.1fs)",
                len(overview_frames), time.time() - t_dec)

    frame_labels = "\n".join(
        f"Frame at {_fmt_mmss(s)} ({s:.0f}s): <image>"
        for s in overview_secs
    )
    letters = "ABCDE"
    choices_str = " ".join(
        f"({letters[i]}) {c}" for i, c in enumerate(answer_choices)
    )
    prompt = TIME_ROUTER_PROMPT.format(
        num_choices=len(answer_choices),
        n_overview=len(overview_frames),
        frame_labels=frame_labels,
        question=question,
        answer_choices=choices_str,
        max_ranges=max_ranges,
        duration_sec=duration_sec,
    )

    images = [img for _, img in overview_frames]
    raw = model.call_selection(images, prompt)
    ranges, justification = _parse_time_ranges(raw, duration_sec)

    # ── Step 3: map ranges → segments ──
    if ranges:
        # Cap at max_ranges
        ranges = ranges[:max_ranges]
        chosen: set[int] = set()
        for (s, e) in ranges:
            for seg in segments:
                if not seg.frame_ids:
                    continue
                # overlap test [seg.start, seg.end) with [s, e]
                if seg.start_sec < e and seg.end_sec > s:
                    chosen.add(seg.index)
        if not chosen:
            # Ranges had no segment overlap — fall back to nearest segment
            # for each range midpoint
            for (s, e) in ranges:
                mid = (s + e) / 2.0
                best = min(
                    nonempty_segments,
                    key=lambda sg: abs((sg.start_sec + sg.end_sec) / 2.0 - mid),
                )
                chosen.add(best.index)
        # Step 3b: neighbor expansion
        if neighbor_hops > 0:
            expanded = set(chosen)
            for idx in chosen:
                for j in range(max(0, idx - neighbor_hops),
                               min(len(segments), idx + neighbor_hops + 1)):
                    if segments[j].frame_ids:
                        expanded.add(j)
            chosen = expanded
        chosen_indices = sorted(chosen)
    else:
        # Fallback: keep a uniform fraction of the video so downstream still works
        n_keep = max(1, int(round(len(nonempty_segments) * fallback_keep_frac)))
        idx = np.linspace(0, len(nonempty_segments) - 1, n_keep).round().astype(int)
        chosen_indices = sorted({nonempty_segments[i].index for i in np.unique(idx)})
        justification = justification or "time_router_fallback_uniform"

    selected_ids: List[int] = []
    for idx in chosen_indices:
        selected_ids.extend(segments[idx].frame_ids)
    selected_ids = sorted(set(selected_ids))

    return {
        "selected_ids": selected_ids,
        "justification": justification,
        "raw_response": raw,
        "selected_scene_indices": chosen_indices,
        "time_ranges": [list(r) for r in ranges],
    }
