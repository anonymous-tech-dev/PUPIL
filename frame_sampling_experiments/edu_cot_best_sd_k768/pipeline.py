"""
pipeline.py — EduCoT pipeline orchestrator.

Pipeline flow:
  Video → Segment → [Keyframe Filter] → [VLM Selection] → Aggregate → Answer

Every bracketed stage is independently toggleable via config.

Per-video caching: segmentation and keyframe filtering results are cached
so that multiple questions about the same video don't redundantly re-process
it.  LVBench averages 15 questions per video — caching gives ~15x speedup
on these stages.
"""

import logging
import os
import time
from typing import Dict, Any, List

import numpy as np
from omegaconf import DictConfig

from stages.video_loading import (
    open_video, decode_frames, FrameBundle, VideoMeta,
)
from stages.segmentation import segment_video, Segment
from stages.keyframe_filter import filter_keyframes, filter_keyframes_batch
from stages.frame_selection import vlm_selection_call
from stages.clip_selection import clip_batch_selection
from stages.context_aggregation import assemble_context
from stages.answering import answer_question

logger = logging.getLogger("educot.pipeline")


# ─── Per-video cache ─────────────────────────────────────────────────────
# Keyed by (video_path, segmentation_mode) → (meta, segments)
# Keyed by (video_path, segmentation_mode, kf_flag) → candidate_ids

_video_meta_cache: Dict[str, VideoMeta] = {}
_segment_cache: Dict[str, List[Segment]] = {}
_candidate_cache: Dict[str, List[int]] = {}


def _cache_key_seg(video_path: str, cfg: DictConfig) -> str:
    return f"{video_path}|{cfg.pipeline.segmentation}"


def _cache_key_cand(video_path: str, cfg: DictConfig) -> str:
    return f"{video_path}|{cfg.pipeline.segmentation}|kf={cfg.pipeline.keyframe_filter}"


def run_pipeline(model, item: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    End-to-end inference for one QA sample.

    Returns a result dict ready for JSONL serialisation.
    """
    t_start = time.time()

    uid            = item["uid"]
    video_path     = item["video_path"]
    question       = item["question"]
    answer_choices = item["answer_choices"]
    ground_truth   = item["ground_truth"]

    logger.info("uid=%s | %s", uid, os.path.basename(video_path))

    # ── Stage 0: Video metadata (cached per video) ────────────────────
    if video_path in _video_meta_cache:
        meta = _video_meta_cache[video_path]
    else:
        meta = open_video(video_path, fps=cfg.video_fps)
        _video_meta_cache[video_path] = meta
    logger.info("  Video: %.0fs, %d frames @ %dfps",
                meta.duration, meta.total_frames, cfg.video_fps)

    # ── Stage 1: Segmentation (cached per video+mode) ────────────────
    seg_key = _cache_key_seg(video_path, cfg)
    if seg_key in _segment_cache:
        segments = _segment_cache[seg_key]
        logger.info("  Segmentation (%s): %d segments (cached)",
                    cfg.pipeline.segmentation, len(segments))
    else:
        t1 = time.time()
        segments = segment_video(video_path, meta, cfg)
        _segment_cache[seg_key] = segments
        logger.info("  Segmentation (%s): %d segments (%.1fs)",
                    cfg.pipeline.segmentation, len(segments), time.time() - t1)

    # ── Stage 2: Keyframe filtering (cached per video+mode+kf) ──────
    cand_key = _cache_key_cand(video_path, cfg)
    if cand_key in _candidate_cache:
        all_candidate_ids = _candidate_cache[cand_key]
        logger.info("  Candidate pool: %d frames (cached)",
                    len(all_candidate_ids))
    else:
        all_candidate_ids: List[int] = []
        if cfg.pipeline.keyframe_filter:
            t_kf = time.time()
            all_candidate_ids = filter_keyframes_batch(
                video_path, segments, meta.native_fps, cfg,
            )
            logger.info("  Keyframe filter: %d frames (%.1fs)",
                        len(all_candidate_ids), time.time() - t_kf)
        else:
            for seg in segments:
                all_candidate_ids.extend(seg.frame_ids)
            all_candidate_ids = sorted(set(all_candidate_ids))
        _candidate_cache[cand_key] = all_candidate_ids
        logger.info("  Candidate pool after segmentation+filter: %d frames",
                    len(all_candidate_ids))

    # ── Stage 3: VLM selection (batched into fixed rounds) ───────────
    # Regardless of how many segments scene-detect produced, we split
    # the candidate pool into a fixed number of VLM selection rounds
    # (= segmentation.num_segments for uniform parity, default 12).
    # This keeps the number of expensive VLM calls constant.
    all_selected_ids: List[int] = []
    all_justifications: List[str] = []
    all_raw_responses: List[str] = []

    s = cfg.selection.frames_per_segment
    selection_mode = getattr(cfg.pipeline, 'selection_mode', 'vlm')  # 'vlm' | 'clip' | 'none'
    # Backward compat: if vlm_selection is explicitly set, derive mode
    if not cfg.pipeline.vlm_selection:
        selection_mode = 'none'

    if selection_mode == 'clip' and all_candidate_ids:
        t_clip = time.time()
        clip_result = clip_batch_selection(
            video_path, all_candidate_ids, question, answer_choices, meta, cfg,
        )
        all_selected_ids = clip_result['selected_ids']
        all_justifications = [clip_result['justification']]
        all_raw_responses = [clip_result['raw_response']]
        logger.info('  CLIP selection: %d frames in %.1fs',
                    len(all_selected_ids), time.time() - t_clip)

    elif selection_mode == 'vlm' and cfg.pipeline.vlm_selection and all_candidate_ids:
        num_rounds = cfg.selection.num_rounds      # default 12 in config.yaml
        round_size = max(1, len(all_candidate_ids) // num_rounds)

        # Split candidate pool into rounds
        rounds: List[List[int]] = []
        for i in range(0, len(all_candidate_ids), round_size):
            rounds.append(all_candidate_ids[i : i + round_size])

        # If we got one extra short tail round, merge it into the last
        if len(rounds) > num_rounds and len(rounds[-1]) < round_size // 2:
            rounds[-2].extend(rounds[-1])
            rounds.pop()

        logger.info("  VLM selection: %d rounds of ~%d frames each",
                    len(rounds), round_size)

        for r_idx, round_ids in enumerate(rounds):
            # Subsample to s if round is still too large
            if len(round_ids) > s:
                indices = np.unique(
                    np.linspace(0, len(round_ids) - 1, s).round().astype(int)
                )
                round_ids = [round_ids[i] for i in indices]

            frames = decode_frames(
                video_path, round_ids, meta.native_fps, meta.target_fps,
            )
            sel = vlm_selection_call(model, frames, question, answer_choices, cfg)
            all_selected_ids.extend(sel["selected_ids"])
            all_justifications.append(sel["justification"])
            all_raw_responses.append(sel["raw_response"])
    else:
        # No VLM selection — pass all candidate frames through
        all_selected_ids = all_candidate_ids

    logger.info("  Selection: %d frames from %d candidates",
                len(all_selected_ids), len(all_candidate_ids))

    # ── Stage 4: Context aggregation ─────────────────────────────────────
    k = cfg.aggregation.context_budget_frames
    u = cfg.aggregation.uniform_context_frames

    context_bundle = assemble_context(video_path, all_selected_ids, meta, k, u)
    logger.info("  Context: %d frames (k=%d, u=%d)", len(context_bundle), k, u)

    # ── Stage 5: Answering ───────────────────────────────────────────────
    t_ans = time.time()
    ans = answer_question(model, context_bundle, question, answer_choices)
    logger.info("  Answer: predicted=%r  gt=%r (%.1fs)",
                ans["predicted_letter"], ground_truth, time.time() - t_ans)

    # ── Result dict ──────────────────────────────────────────────────────
    return {
        "uid":               uid,
        "predicted_letter":  ans["predicted_letter"],
        "ground_truth":      ground_truth,
        "raw_answer":        ans["raw_response"],
        "selected_ids":      all_selected_ids,
        "num_selected":      len(all_selected_ids),
        "context_ids":       ans["frame_ids_used"],
        "num_context":       len(ans["frame_ids_used"]),
        "total_frames":      meta.total_frames,
        "num_segments":      len(segments),
        "justifications":    all_justifications,
        "raw_responses":     all_raw_responses,
        "pipeline": {
            "segmentation":   cfg.pipeline.segmentation,
            "keyframe_filter": cfg.pipeline.keyframe_filter,
            "vlm_selection":  cfg.pipeline.vlm_selection,
            "selection_mode": selection_mode,
        },
        "video_path":        video_path,
        "question":          question,
        "answer_choices":    answer_choices,
        "question_type":     item.get("question_type", []),
        "time_reference":    item.get("time_reference", ""),
        "time_taken_secs":   round(time.time() - t_start, 2),
    }
