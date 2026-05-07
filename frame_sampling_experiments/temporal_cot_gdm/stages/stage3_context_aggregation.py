"""
stages/stage3_context_aggregation.py — Stage 3: Context Aggregation (G function).

Implements all three TCoT variants from the paper:

  1. Single-Step TCoT (§3.2, Fig. 2 left)
     - Given up to N frames that fit the context limit, ask the VLM which
       frame IDs are relevant, return those + u uniform context frames.

  2. Dynamic-Segment TCoT (§3.2, Fig. 2 right) ← **default**
     - Partition video into l segments.
     - Subsample s frames per segment.
     - Run Single-Step TCoT independently on each segment.
     - Concatenate selected frames across segments.
     - If result > k, uniformly subsample down to k-u + add u uniform frames.

  3. Hierarchical TCoT (§4.2)
     - Iteratively zoom in: coarse → find relevant → sample neighbourhood →
       repeat until convergence or max iters.

All variants:
  - Use the *same* VLM for selection and answering (single model throughout).
  - Return c = ˆx ∪ x[u] (selected frames ∪ uniform context frames).

Paper equations:
  Eq. 4  — ˆx, S, j = S(x, q)        (single-step selection call)
  Eq. 5  — ˆx = [S(x1[s], q), ..., S(xl[s], q)]  (dynamic-segment)

──────────────────────────────────────────────────────────────────────────────
PERFORMANCE FIX for dynamic_segment (zero accuracy impact):

  Old flow:
    1. load_video_frames() → decode ALL N frames into RAM
    2. segment_bundle()    → split into l segments (still all in RAM)
    3. uniform_subsample() → keep s per segment, discard the rest
    4. selection_call()    → use only the s frames

  New flow (dynamic_segment only):
    1. open_video()        → read metadata only (no pixel decode)
    2. Compute which frame IDs are needed: l segments × s frames = l×s IDs
    3. fetch_frames()      → decode ONLY those l×s frames from disk
    4. selection_call()    → same as before

  For a 68-min LVBench video (N≈4080 frames, l=12, s=64):
    Old: decode 4080 frames, use 768 (19%)
    New: decode 768 frames, use 768 (100%)

  The frames selected are mathematically identical — we just skip decoding
  the ones we'd throw away immediately after.  Accuracy is unaffected.

  single_step and hierarchical still use load_video_frames() (full decode)
  because they need the full bundle for neighbourhood expansion / fallback.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from stages.stage0_video_loading import (
    FrameBundle, uniform_subsample, uniform_subsample_ids,
    segment_bundle, segment_ids, frames_from_ids,
    get_frame_ids, get_frame_images,
    open_video, fetch_frames,
)
from stages.stage1_prompts import build_selection_prompt
from stages.stage2_selection_parsing import parse_selection_response
import config

logger = logging.getLogger(__name__)


# ─── Selection call (Eq. 4) ────────────────────────────────────────────────────

def selection_call(
    model,
    segment_bundle_: FrameBundle,
    question: str,
    answer_choices: List[str],
) -> Dict[str, Any]:
    """
    Run one VLM selection call on `segment_bundle_`.

    Returns a dict:
      {
        "selected_ids"  : List[int],  # validated, sorted
        "justification" : str,
        "raw_response"  : str,
      }
    """
    frame_ids = get_frame_ids(segment_bundle_)
    images    = get_frame_images(segment_bundle_)
    style = "qwen" if "qwen" in config.MODEL.lower() else "gpt"

    prompt = build_selection_prompt(
        frame_ids=frame_ids,
        question=question,
        answer_choices=answer_choices,
        style=style
    )

    raw = model.call_selection(images, prompt)

    selected_ids, justification = parse_selection_response(
        raw_response=raw,
        valid_frame_ids=frame_ids,
    )

    return {
        "selected_ids"  : selected_ids,
        "justification" : justification,
        "raw_response"  : raw,
    }


# ─── Helper: merge + clip to budget ───────────────────────────────────────────

def _assemble_context(
    full_bundle    : FrameBundle,
    selected_ids   : List[int],
    k              : int,
    u              : int,
) -> FrameBundle:
    """
    Build the final context c = ˆx[m] ∪ x[u].

    Steps (paper §3.2 Dynamic-Segment section):
      1. If |selected_ids| > k-u, uniformly subsample to k-u from selected.
      2. Pick u frames uniformly from the full video.
      3. Merge + deduplicate + sort by frame_id.
    """
    m = k - u

    # (1) Selected frames
    sel_bundle = frames_from_ids(full_bundle, selected_ids)
    if len(sel_bundle) > m:
        sel_bundle = uniform_subsample(sel_bundle, m)

    # (2) Uniform context
    uni_bundle = uniform_subsample(full_bundle, u) if u > 0 else []

    # (3) Merge + dedup + sort
    seen   = set()
    merged = []
    for fid, img in sorted(sel_bundle + uni_bundle, key=lambda x: x[0]):
        if fid not in seen:
            seen.add(fid)
            merged.append((fid, img))

    return merged


def _assemble_context_from_meta(
    video_path     : str,
    selected_ids   : List[int],
    k              : int,
    u              : int,
) -> FrameBundle:
    """
    Variant of _assemble_context that works without a pre-loaded full_bundle.

    Used by dynamic_segment: after the selection calls we know which IDs were
    selected; we fetch only those + u uniform frames from disk.

    Steps:
      1. If |selected_ids| > k-u, uniformly subsample the ID list to k-u
         (arithmetic only — no images loaded yet for the discarded ones).
      2. Compute u uniform frame IDs across the full video.
      3. Union the two ID sets.
      4. Decode only those frames from disk in a single batch.
    """
    from stages.stage0_video_loading import open_video, fetch_frames, uniform_subsample_ids

    meta = open_video(video_path)
    m    = k - u

    # # (1) Clip selected IDs if needed (arithmetic, no images)
    # if len(selected_ids) > m:
    #     # Build a dummy positional list so uniform_subsample_ids logic applies
    #     step  = max(1.0, (len(selected_ids) - 1) / (m - 1)) if m > 1 else len(selected_ids)
    #     idxs  = [round(i * step) for i in range(m)]
    #     idxs  = sorted(set(min(i, len(selected_ids) - 1) for i in idxs))
    #     sel_ids_clipped = [selected_ids[i] for i in idxs]
    # else:
    #     sel_ids_clipped = list(selected_ids)

    # # (2) Uniform context IDs
    # uni_ids = uniform_subsample_ids(meta.total_frames, u) if u > 0 else []

    # # (3) Union + sort
    # all_ids = sorted(set(sel_ids_clipped) | set(uni_ids))

    # # (4) Decode only what we need
    # return fetch_frames(meta, all_ids)

    # REPLACE the manual clipping with uniform_subsample_ids logic
    if len(selected_ids) > m:
        # Use linspace over the index positions, same as uniform_subsample
        float_idx = np.linspace(0, len(selected_ids) - 1, m)
        idxs = np.unique(np.round(float_idx).astype(int))
        idxs = np.clip(idxs, 0, len(selected_ids) - 1)
        sel_ids_clipped = [selected_ids[i] for i in idxs]
    else:
        sel_ids_clipped = list(selected_ids)

    uni_ids = uniform_subsample_ids(meta.total_frames, u) if u > 0 else []
    all_ids = sorted(set(sel_ids_clipped) | set(uni_ids))
    
    return fetch_frames(meta, all_ids)


# ─── Variant 1: Single-Step TCoT ───────────────────────────────────────────────

def single_step_tcot(
    model,
    full_bundle    : FrameBundle,
    question       : str,
    answer_choices : List[str],
    k              : int = None,
    u              : int = None,
) -> Dict[str, Any]:
    """
    Single-Step TCoT (Fig. 2 left, §3.2).

    1. Subsample `k` frames uniformly from full_bundle.
    2. Run one selection call.
    3. Assemble context.

    Returns:
      {
        "context_bundle"  : FrameBundle,
        "selected_ids"    : List[int],
        "justifications"  : List[str],   # one per segment (only 1 here)
        "raw_responses"   : List[str],
        "stage"           : "single_step",
      }
    """
    k = k or config.CONTEXT_BUDGET_FRAMES
    u = u if u is not None else config.UNIFORM_CONTEXT_FRAMES

    # Subsample input to fit context limit
    input_bundle = uniform_subsample(full_bundle, k)

    result  = selection_call(model, input_bundle, question, answer_choices)
    context = _assemble_context(full_bundle, result["selected_ids"], k, u)

    return {
        "context_bundle"  : context,
        "selected_ids"    : result["selected_ids"],
        "justifications"  : [result["justification"]],
        "raw_responses"   : [result["raw_response"]],
        "stage"           : "single_step",
    }


# ─── Variant 2: Dynamic-Segment TCoT (default) ─────────────────────────────────

def dynamic_segment_tcot(
    model,
    full_bundle    : FrameBundle,   # may be None when called via the fast path
    question       : str,
    answer_choices : List[str],
    l              : int = None,
    s              : int = None,
    k              : int = None,
    u              : int = None,
    # Fast-path: supply video_path instead of a pre-loaded full_bundle
    video_path     : str = None,
) -> Dict[str, Any]:
    """
    Dynamic-Segment TCoT (Fig. 2 right, §3.2, Eq. 5).

    Fast path (video_path supplied, full_bundle=None):
      - Reads only metadata to learn total frame count.
      - Computes segment frame ID ranges arithmetically.
      - Decodes only the l×s frames needed for selection calls.
      - After selection, decodes only the final context frames (≤k).
      - Total frames decoded: l×s + k  (vs. N in the original).

    Slow path (full_bundle supplied, video_path=None):
      - Identical logic but operates on the pre-loaded bundle.
      - Kept for backwards compatibility / unit tests.

    Returns same dict structure as single_step_tcot with per-segment
    justifications and responses.
    """
    l = l or config.NUM_SEGMENTS
    s = s or config.FRAMES_PER_SEGMENT
    k = k or config.CONTEXT_BUDGET_FRAMES
    u = u if u is not None else config.UNIFORM_CONTEXT_FRAMES

    # ── Choose fast path vs. slow path ────────────────────────────────────────
    if video_path is not None and full_bundle is None:
        return _dynamic_segment_fast(
            model, video_path, question, answer_choices, l, s, k, u
        )
    else:
        return _dynamic_segment_slow(
            model, full_bundle, question, answer_choices, l, s, k, u
        )


def _dynamic_segment_fast(
    model,
    video_path     : str,
    question       : str,
    answer_choices : List[str],
    l              : int,
    s              : int,
    k              : int,
    u              : int,
) -> Dict[str, Any]:
    """
    Fast dynamic_segment: decode only the frames we actually need.

    For each of l segments we compute the s frame IDs arithmetically,
    then batch-decode all l×s frames in a single vr.get_batch() call.
    After selection, we decode only the ≤k context frames for answering.
    """
    meta = open_video(video_path)
    N    = meta.total_frames

    # ── 1. Compute which frame IDs to decode for each segment ─────────────
    seg_ranges = segment_ids(N, l)   # [(start_fid, end_fid), ...] 1-indexed, end exclusive

    # For each segment, compute the s uniformly-spaced frame IDs.
    # This mirrors uniform_subsample(seg, s) but operates on ID ranges only.
    per_segment_ids: List[List[int]] = []
    for (start, end) in seg_ranges:
        seg_total = end - start          # number of logical frames in this segment
        if seg_total <= 0:
            per_segment_ids.append([])
            continue
        if seg_total <= s:
            ids = list(range(start, end))
        else:
            float_idx = np.linspace(0, seg_total - 1, s)
            rel_idx   = np.round(float_idx).astype(int)
            rel_idx   = np.clip(rel_idx, 0, seg_total - 1)
            ids       = sorted(set(int(start + i) for i in rel_idx))
        per_segment_ids.append(ids)

    # Collect all unique IDs needed across segments
    all_needed_ids = sorted(set(fid for seg in per_segment_ids for fid in seg))

    logger.info(
        "  [fast] Decoding %d/%d frames for %d segment selection calls",
        len(all_needed_ids), N, l,
    )

    # ── 2. Batch-decode all needed frames in one shot ──────────────────────
    all_frames_bundle = fetch_frames(meta, all_needed_ids)
    fid_to_frame      = dict(all_frames_bundle)   # {frame_id: PIL.Image}

    # ── 3. Run selection call on each segment ──────────────────────────────
    all_selected_ids : List[int] = []
    justifications   : List[str] = []
    raw_responses    : List[str] = []

    for seg_i, seg_ids_list in enumerate(per_segment_ids):
        if not seg_ids_list:
            continue

        seg_sampled = [(fid, fid_to_frame[fid])
                       for fid in seg_ids_list if fid in fid_to_frame]
        if not seg_sampled:
            continue

        result = selection_call(model, seg_sampled, question, answer_choices)
        
        if result["selected_ids"]:  # only extend if non-empty
            all_selected_ids.extend(result["selected_ids"])

        justifications.append(
            f"[Segment {seg_i+1}/{len(seg_ranges)}] {result['justification']}"
        )
        raw_responses.append(result["raw_response"])

        logger.debug(
            "Segment %d/%d: selected %d frames → %s",
            seg_i + 1, len(seg_ranges),
            len(result["selected_ids"]),
            result["selected_ids"][:10],
        )

    # ── 4. Deduplicate across segments (preserve order) ────────────────────
    seen   = set()
    deduped = []
    for fid in all_selected_ids:
        if fid not in seen:
            seen.add(fid)
            deduped.append(fid)

    # ── 5. Assemble final context — decode only those frames ───────────────
    # _assemble_context_from_meta decodes only the ≤k frames needed for answering
    context = _assemble_context_from_meta(video_path, sorted(deduped), k, u)

    return {
        "context_bundle"  : context,
        "selected_ids"    : sorted(deduped),
        "justifications"  : justifications,
        "raw_responses"   : raw_responses,
        "stage"           : "dynamic_segment",
    }


def _dynamic_segment_slow(
    model,
    full_bundle    : FrameBundle,
    question       : str,
    answer_choices : List[str],
    l              : int,
    s              : int,
    k              : int,
    u              : int,
) -> Dict[str, Any]:
    """
    Original dynamic_segment logic — operates on a pre-loaded full_bundle.
    Kept for backwards compatibility.
    """
    segments         = segment_bundle(full_bundle, l)
    all_selected_ids : List[int] = []
    justifications   : List[str] = []
    raw_responses    : List[str] = []

    for seg_i, seg in enumerate(segments):
        seg_sampled = uniform_subsample(seg, s)
        if not seg_sampled:
            continue

        result = selection_call(model, seg_sampled, question, answer_choices)

        all_selected_ids.extend(result["selected_ids"])
        justifications.append(
            f"[Segment {seg_i+1}/{len(segments)}] {result['justification']}"
        )
        raw_responses.append(result["raw_response"])

        logger.debug(
            "Segment %d/%d: selected %d frames → %s",
            seg_i + 1, len(segments),
            len(result["selected_ids"]),
            result["selected_ids"][:10],
        )

    seen   = set()
    deduped = []
    for fid in all_selected_ids:
        if fid not in seen:
            seen.add(fid)
            deduped.append(fid)

    context = _assemble_context(full_bundle, deduped, k, u)

    return {
        "context_bundle"  : context,
        "selected_ids"    : sorted(deduped),
        "justifications"  : justifications,
        "raw_responses"   : raw_responses,
        "stage"           : "dynamic_segment",
    }


# ─── Variant 3: Hierarchical TCoT ─────────────────────────────────────────────

def hierarchical_tcot(
    model,
    full_bundle     : FrameBundle,
    question        : str,
    answer_choices  : List[str],
    k               : int = None,
    u               : int = None,
    neighbourhood   : int = None,
    max_iters       : int = None,
) -> Dict[str, Any]:
    """
    Hierarchical TCoT (§4.2, App. D).

    Uses fid_to_pos dict for O(1) neighbourhood lookup instead of O(N)
    list.index() per frame.
    """
    k             = k or config.CONTEXT_BUDGET_FRAMES
    u             = u if u is not None else config.UNIFORM_CONTEXT_FRAMES
    neighbourhood = neighbourhood or config.HIER_NEIGHBOURHOOD
    max_iters     = max_iters or config.HIER_MAX_ITERS

    all_frame_ids = get_frame_ids(full_bundle)
    fid_to_img    = dict(full_bundle)
    fid_to_pos    = {fid: pos for pos, fid in enumerate(all_frame_ids)}

    current_input = uniform_subsample(full_bundle, k)

    all_justifications : List[str] = []
    all_raw_responses  : List[str] = []
    prev_selected_ids  : Optional[List[int]] = None

    for iteration in range(max_iters):
        result       = selection_call(model, current_input, question, answer_choices)
        selected_ids = sorted(result["selected_ids"])

        all_justifications.append(f"[Iter {iteration}] {result['justification']}")
        all_raw_responses.append(result["raw_response"])

        # Convergence check
        if selected_ids == prev_selected_ids:
            logger.debug("Hierarchical TCoT converged at iteration %d", iteration)
            break

        prev_selected_ids = selected_ids

        if iteration == max_iters - 1:
            break

        # Expand neighbourhood
        expanded_set: set = set()
        for fid in selected_ids:
            pos = fid_to_pos.get(fid)
            if pos is None:
                continue
            lo = max(0, pos - neighbourhood)
            hi = min(len(all_frame_ids), pos + neighbourhood + 1)
            for p in range(lo, hi):
                expanded_set.add(all_frame_ids[p])

        expanded_ids  = sorted(expanded_set)
        current_input = [
            (fid, fid_to_img[fid]) for fid in expanded_ids if fid in fid_to_img
        ]

        if len(current_input) > k:
            current_input = uniform_subsample(current_input, k)

        logger.debug(
            "Iter %d: %d selected → %d expanded frames for next iter",
            iteration, len(selected_ids), len(current_input),
        )

    context = _assemble_context(full_bundle, prev_selected_ids or [], k, u)

    return {
        "context_bundle"  : context,
        "selected_ids"    : prev_selected_ids or [],
        "justifications"  : all_justifications,
        "raw_responses"   : all_raw_responses,
        "stage"           : "hierarchical",
    }


# ─── Dispatcher ────────────────────────────────────────────────────────────────

def aggregate_context(
    model,
    full_bundle    : FrameBundle,
    question       : str,
    answer_choices : List[str],
    variant        : str = None,
    video_path     : str = None,   # NEW: enables fast path for dynamic_segment
) -> Dict[str, Any]:
    """
    Dispatcher: run the configured TCoT variant.

    Args:
        model          : loaded VLM (BaseVLM subclass)
        full_bundle    : all decoded frames at 1 fps (None for dynamic_segment fast path)
        question       : QA question string
        answer_choices : list of answer option strings
        variant        : override config.TCOT_VARIANT if provided
        video_path     : if supplied with variant='dynamic_segment', uses the
                         fast decode path (no full video load needed)

    Returns:
        dict with keys: context_bundle, selected_ids, justifications,
                        raw_responses, stage
    """
    variant = variant or config.TCOT_VARIANT

    if variant == "single_step":
        return single_step_tcot(model, full_bundle, question, answer_choices)

    elif variant == "dynamic_segment":
        if video_path is not None:
            # Fast path: no full_bundle needed
            return dynamic_segment_tcot(
                model, None, question, answer_choices, video_path=video_path
            )
        else:
            return dynamic_segment_tcot(
                model, full_bundle, question, answer_choices
            )

    elif variant == "hierarchical":
        return hierarchical_tcot(model, full_bundle, question, answer_choices)

    else:
        raise ValueError(f"Unknown TCoT variant: {variant!r}. "
                         "Choose 'single_step', 'dynamic_segment', or 'hierarchical'.")