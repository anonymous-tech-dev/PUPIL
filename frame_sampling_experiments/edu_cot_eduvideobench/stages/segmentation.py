"""
stages/segmentation.py — Video segmentation strategies.

  "uniform"      — TCoT-style: divide duration into l equal windows.
  "scene_detect"  — Content-aware: decord + mean-absolute-diff scene detection.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig

from stages.video_loading import VideoMeta

logger = logging.getLogger("educot.segmentation")


@dataclass
class Segment:
    index: int
    start_sec: float
    end_sec: float
    frame_ids: List[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ─── Helpers ──────────────────────────────────────────────────────────────

def _frame_ids_for_window(start_sec: float, end_sec: float, fps: int) -> List[int]:
    """Return list of frame IDs (at target fps) that fall within [start, end)."""
    first = int(start_sec * fps)
    last = int(end_sec * fps)       # exclusive
    return list(range(first, last))


# ─── Uniform segmentation ────────────────────────────────────────────────

def uniform_segments(meta: VideoMeta, cfg: DictConfig) -> List[Segment]:
    n = cfg.segmentation.num_segments
    fps = cfg.video_fps
    seg_dur = meta.duration / n

    segments = []
    for i in range(n):
        s = i * seg_dur
        e = (i + 1) * seg_dur
        segments.append(Segment(
            index=i,
            start_sec=s,
            end_sec=e,
            frame_ids=_frame_ids_for_window(s, e, fps),
        ))
    return segments


# ─── Scene-detect segmentation ───────────────────────────────────────────

def scene_detect_segments(
    video_path: str,
    meta: VideoMeta,
    cfg: DictConfig,
) -> List[Segment]:
    """
    Content-aware scene detection using decord + mean-absolute-diff.

    Decodes at low resolution (configurable, default 320×180) and low
    FPS (default 2), computes per-frame MAD scores, and places scene
    boundaries where the score exceeds threshold.

    5-9× faster than PySceneDetect on 30-100 min videos because:
      - decord skips native frames via seek (vs PyAV decoding all)
      - low-res decode (320×180 vs 1920×1080)
    """
    import numpy as np
    from decord import VideoReader, cpu

    sd_cfg = cfg.segmentation.scene_detect
    fps = cfg.video_fps

    threshold      = sd_cfg.threshold         # MAD threshold
    sample_fps     = sd_cfg.sample_fps        # FPS for analysis
    frame_w        = sd_cfg.frame_width       # decode width
    frame_h        = sd_cfg.frame_height      # decode height
    min_scene_sec  = sd_cfg.min_scene_length  # merge cuts closer than this
    max_scene_sec  = sd_cfg.max_scene_length  # split scenes longer than this

    # ── Decode at low res + low fps ──────────────────────────────────
    vr = VideoReader(video_path, ctx=cpu(0), width=frame_w, height=frame_h)
    native_fps = vr.get_avg_fps()
    n_native = len(vr)
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, n_native, step))

    if len(indices) < 2:
        return [Segment(
            index=0, start_sec=0.0, end_sec=meta.duration,
            frame_ids=_frame_ids_for_window(0.0, meta.duration, fps),
        )]

    batch = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8

    # ── Compute MAD scores ───────────────────────────────────────────
    scores = np.zeros(len(batch) - 1, dtype=np.float32)
    for i in range(len(batch) - 1):
        scores[i] = np.mean(np.abs(
            batch[i + 1].astype(np.float32) - batch[i].astype(np.float32)
        ))

    # ── Find cut points above threshold ──────────────────────────────
    cut_sample_indices = np.where(scores > threshold)[0] + 1  # +1: cut is *at* the new frame

    # Convert sample indices → seconds
    cut_secs = [indices[int(ci)] / native_fps for ci in cut_sample_indices]

    # Enforce min_scene_length: merge cuts that are too close
    filtered_cuts = []
    for t in cut_secs:
        if not filtered_cuts or (t - filtered_cuts[-1]) >= min_scene_sec:
            filtered_cuts.append(t)

    # ── Build segment list ───────────────────────────────────────────
    boundaries = [0.0] + filtered_cuts + [meta.duration]
    segments: List[Segment] = []

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]

        # Split scenes longer than max_scene_length
        while s < e:
            seg_end = min(e, s + max_scene_sec)
            segments.append(Segment(
                index=len(segments),
                start_sec=s,
                end_sec=seg_end,
                frame_ids=_frame_ids_for_window(s, seg_end, fps),
            ))
            s = seg_end

    if not segments:
        segments = [Segment(
            index=0, start_sec=0.0, end_sec=meta.duration,
            frame_ids=_frame_ids_for_window(0.0, meta.duration, fps),
        )]

    logger.info("  MAD scene detect: %d cuts → %d segments (threshold=%.1f, %dx%d@%dfps)",
                len(filtered_cuts), len(segments), threshold, frame_w, frame_h, sample_fps)
    return segments


# ─── Dispatcher ───────────────────────────────────────────────────────────

def segment_video(
    video_path: str,
    meta: VideoMeta,
    cfg: DictConfig,
) -> List[Segment]:
    mode = cfg.pipeline.segmentation

    if mode == "scene_detect":
        return scene_detect_segments(video_path, meta, cfg)
    else:
        return uniform_segments(meta, cfg)
