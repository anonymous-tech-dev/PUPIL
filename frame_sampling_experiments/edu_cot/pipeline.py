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
from stages.time_router import time_router_select

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


def _scene_zoom_select(
    model,
    video_path: str,
    segments: List[Segment],
    question: str,
    answer_choices: List[str],
    meta: VideoMeta,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """Route to a small set of scenes via representative frames, then zoom in."""
    sz_cfg = cfg.scene_zoom
    max_scene_reps = max(1, int(sz_cfg.max_scene_reps))
    top_scenes = max(1, int(sz_cfg.top_scenes))
    neighbor_hops = max(0, int(sz_cfg.neighbor_hops))
    frames_per_rep = max(1, int(getattr(sz_cfg, 'frames_per_rep', 1)))

    nonempty_segments = [seg for seg in segments if seg.frame_ids]
    if not nonempty_segments:
        return {
            "selected_ids": [],
            "justification": "scene_zoom_empty_segments",
            "raw_response": "",
            "selected_scene_indices": [],
        }

    # Cap number of scenes shown to router so total frames stay manageable
    effective_max = max(1, max_scene_reps // frames_per_rep)
    rep_segments = nonempty_segments
    if len(rep_segments) > effective_max:
        rep_idx = np.unique(
            np.linspace(0, len(rep_segments) - 1, effective_max).round().astype(int)
        )
        rep_segments = [rep_segments[i] for i in rep_idx]

    # Build rep frame list — frames_per_rep evenly-spaced inside each scene
    rep_frame_ids: List[int] = []
    rep_to_segment: Dict[int, int] = {}
    for seg in rep_segments:
        n = len(seg.frame_ids)
        if frames_per_rep == 1 or n < 2:
            picks = [seg.frame_ids[n // 2]]
        else:
            # quantile positions: 1/(fpr+1), 2/(fpr+1), ..., fpr/(fpr+1)
            offs = np.linspace(0, n - 1, frames_per_rep + 2)[1:-1]
            picks_idx = np.unique(offs.round().astype(int)).tolist()
            picks = [seg.frame_ids[i] for i in picks_idx]
        for fid in picks:
            rep_frame_ids.append(fid)
            rep_to_segment[fid] = seg.index

    rep_frames = decode_frames(video_path, rep_frame_ids, meta.native_fps, meta.target_fps)
    routed = vlm_selection_call(model, rep_frames, question, answer_choices, cfg)
    ranked_scene_indices: List[int] = []
    for frame_id in routed["selected_ids"]:
        seg_idx = rep_to_segment.get(frame_id)
        if seg_idx is not None and seg_idx not in ranked_scene_indices:
            ranked_scene_indices.append(seg_idx)
        if len(ranked_scene_indices) >= top_scenes:
            break

    if not ranked_scene_indices:
        ranked_scene_indices = [rep_segments[0].index]

    chosen_indices = set()
    for seg_idx in ranked_scene_indices:
        for idx in range(max(0, seg_idx - neighbor_hops), min(len(segments), seg_idx + neighbor_hops + 1)):
            if segments[idx].frame_ids:
                chosen_indices.add(idx)

    expanded_ids: List[int] = []
    for idx in sorted(chosen_indices):
        expanded_ids.extend(segments[idx].frame_ids)

    return {
        "selected_ids": sorted(set(expanded_ids)),
        "justification": routed["justification"],
        "raw_response": routed["raw_response"],
        "selected_scene_indices": sorted(chosen_indices),
    }


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
    selection_mode = getattr(cfg.pipeline, 'selection_mode', 'vlm')  # 'vlm' | 'clip' | 'scene_zoom' | 'vlm_prune' | 'none'
    # Backward compat: if vlm_selection is explicitly set, derive mode
    if not cfg.pipeline.vlm_selection and selection_mode not in ('vlm_prune', 'scene_zoom'):
        selection_mode = 'none'

    # ── Adaptive-k: override context budget based on video length ─────
    if getattr(cfg.aggregation, 'adaptive_k', False):
        tf = meta.total_frames
        ver = getattr(cfg.aggregation, 'adaptive_k_version', 1)
        if ver == 2:
            if tf < 1500:
                cfg.aggregation.context_budget_frames = 384
            elif tf < 3500:
                cfg.aggregation.context_budget_frames = 768
            else:
                cfg.aggregation.context_budget_frames = 1024
        elif ver == 3:  # no 1024 — cap at 768
            if tf < 2000:
                cfg.aggregation.context_budget_frames = 512
            else:
                cfg.aggregation.context_budget_frames = 768
        else:  # v1
            if tf < 2000:
                cfg.aggregation.context_budget_frames = 512
            elif tf < 4000:
                cfg.aggregation.context_budget_frames = 768
            else:
                cfg.aggregation.context_budget_frames = 1024
        logger.info("  Adaptive-k v%d: total_frames=%d → k=%d",
                    ver, tf, cfg.aggregation.context_budget_frames)

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

    elif selection_mode == 'scene_zoom' and all_candidate_ids:
        t_zoom = time.time()
        zoom_result = _scene_zoom_select(
            model, video_path, segments, question, answer_choices, meta, cfg,
        )
        all_selected_ids = zoom_result['selected_ids']
        all_justifications = [zoom_result['justification']]
        all_raw_responses = [zoom_result['raw_response']]
        logger.info(
            '  Scene zoom: routed %d scenes → %d frames in %.1fs',
            len(zoom_result['selected_scene_indices']),
            len(all_selected_ids),
            time.time() - t_zoom,
        )

    elif selection_mode == 'time_router' and all_candidate_ids:
        t_tr = time.time()
        tr_result = time_router_select(
            model, video_path, segments, question, answer_choices, meta, cfg,
        )
        all_selected_ids = tr_result['selected_ids']
        all_justifications = [tr_result['justification']]
        all_raw_responses = [tr_result['raw_response']]
        logger.info(
            '  Time router: %d ranges → %d scenes → %d frames in %.1fs',
            len(tr_result['time_ranges']),
            len(tr_result['selected_scene_indices']),
            len(all_selected_ids),
            time.time() - t_tr,
        )

    elif selection_mode == 'vlm_prune' and all_candidate_ids:
        # 1-round VLM pruning: subsample candidates → show to VLM → keep selected
        t_prune = time.time()
        prune_budget = getattr(cfg.selection, 'prune_show_frames', 64)
        # Subsample to prune_budget frames for the VLM call
        if len(all_candidate_ids) > prune_budget:
            indices = np.unique(
                np.linspace(0, len(all_candidate_ids) - 1, prune_budget).round().astype(int)
            )
            show_ids = [all_candidate_ids[i] for i in indices]
        else:
            show_ids = all_candidate_ids
        frames = decode_frames(
            video_path, show_ids, meta.native_fps, meta.target_fps,
        )
        sel = vlm_selection_call(model, frames, question, answer_choices, cfg)
        # The VLM picks a subset; these become the selected set
        all_selected_ids = sel["selected_ids"]
        all_justifications = [sel["justification"]]
        all_raw_responses = [sel["raw_response"]]
        logger.info("  VLM prune: %d→%d frames in %.1fs",
                    len(show_ids), len(all_selected_ids), time.time() - t_prune)

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
    prompt_style = getattr(cfg.generation, "prompt_style", "direct")

    # Optional deterministic per-uid shuffle of answer choices to debias the
    # model's positional letter prior (CircularEval-style; standard MCQ-LMM
    # debiasing primitive). The predicted letter is translated back to the
    # original option order so ground_truth comparison is unchanged.
    shuffle_seed = getattr(cfg.pipeline, "shuffle_choices_seed", None)
    if shuffle_seed is not None and answer_choices and len(answer_choices) >= 2:
        import random as _rnd
        rng = _rnd.Random(f"{shuffle_seed}::{uid}")
        n_opt = len(answer_choices)
        perm = list(range(n_opt))
        rng.shuffle(perm)                       # perm[new_idx] = orig_idx
        shuffled_choices = [answer_choices[i] for i in perm]
        ans = answer_question(
            model, context_bundle, question, shuffled_choices,
            prompt_style=prompt_style,
        )
        # Translate predicted letter from shuffled space back to original.
        pred_letter_shuf = ans["predicted_letter"]
        if pred_letter_shuf and pred_letter_shuf in "ABCDE":
            new_idx = ord(pred_letter_shuf) - ord('A')
            if 0 <= new_idx < n_opt:
                orig_idx = perm[new_idx]
                ans["predicted_letter"] = chr(ord('A') + orig_idx)
        ans["_shuffle_perm"] = perm
        ans["_predicted_letter_shuffled"] = pred_letter_shuf
    else:
        ans = answer_question(
            model, context_bundle, question, answer_choices,
            prompt_style=prompt_style,
        )
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
