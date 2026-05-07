"""
stages/stage0_video_loading.py — Stage 0: Video Loading & Frame Extraction.

Responsibilities:
  1. Decode a video file at 1 fps (configurable via config.VIDEO_FPS).
  2. Return frames as a list of PIL Images with their global frame IDs
     (1-indexed, matching the paper's convention).
  3. Provide helpers for uniform subsampling used throughout TCoT.

Paper reference: §3.1 — "the frames of the video typically need to be
  subsampled to fit within the context-limit k of the model. Current models ...
  can typically fit videos of up to one hour at 1 fps."

──────────────────────────────────────────────────────────────────────────────
PERFORMANCE FIX (zero accuracy impact):
  The original implementation decoded ALL frames into RAM as PIL Images before
  any subsampling. For a 68-min LVBench video at 1fps this means decoding
  ~4080 full-resolution frames when dynamic_segment TCoT only uses
  l×s = 12×64 = 768 of them (≈19%).

  Fix: VideoReader (a thin metadata wrapper) now records the total frame count
  and native fps without decoding pixels. Frame *indices* for each TCoT variant
  are computed arithmetically upfront, and only those native frame indices are
  decoded from disk. The identical frames are selected — just without paying for
  the 81% that get discarded.

  The public API is unchanged:
    - load_video_frames()          → still works, decodes everything (used by
                                     single_step / hierarchical which need the
                                     full bundle for neighbourhood expansion)
    - VideoMeta + fetch_frames()   → new selective path used by dynamic_segment
──────────────────────────────────────────────────────────────────────────────
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu as decord_cpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False

import config

# ─── Type alias ────────────────────────────────────────────────────────────────
# A "frame_bundle" is a list of (frame_id, PIL.Image) pairs.
# frame_id is 1-indexed, consistent with the paper's FrameID convention.
FrameBundle = List[Tuple[int, Image.Image]]


# ─── VideoMeta: lightweight handle (no pixel decoding) ─────────────────────────

@dataclass
class VideoMeta:
    """
    Thin wrapper that holds video metadata and a cached VideoReader handle.
    Pixels are decoded *on demand* via fetch_frames().

    Attributes
    ----------
    video_path   : path to the video file
    total_frames : number of 1-fps logical frames in the video
    native_fps   : original frame rate of the video file
    step         : native frame step corresponding to 1-fps sampling
    _vr          : internal decord VideoReader (None if decord unavailable)
    """
    video_path   : str
    total_frames : int
    native_fps   : float
    step         : int
    _vr          : object = None   # decord VideoReader or None

    def get_frame_ids(self) -> List[int]:
        """Return the full list of 1-indexed frame IDs (no decoding)."""
        return list(range(1, self.total_frames + 1))


def open_video(video_path: str, fps: float = None) -> VideoMeta:
    """
    Open a video and return a VideoMeta without decoding any pixels.

    This replaces the old pattern of calling load_video_frames() just to
    know the frame count.  For dynamic_segment TCoT the frame indices needed
    can be computed from VideoMeta alone.
    """
    fps = fps or config.VIDEO_FPS

    if _DECORD_AVAILABLE:
        vr         = VideoReader(video_path, ctx=decord_cpu(0))
        native_fps = vr.get_avg_fps()
        step       = max(1, int(round(native_fps / fps)))
        total      = len(range(0, len(vr), step))
        return VideoMeta(video_path=video_path, total_frames=total,
                         native_fps=native_fps, step=step, _vr=vr)
    else:
        import cv2
        cap        = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_native   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        step  = max(1, int(round(native_fps / fps)))
        total = len(range(0, n_native, step))
        return VideoMeta(video_path=video_path, total_frames=total,
                         native_fps=native_fps, step=step, _vr=None)


def fetch_frames(meta: VideoMeta, frame_ids: List[int]) -> FrameBundle:
    """
    Decode only the specified 1-indexed frame IDs from disk.

    This is the key performance fix: instead of decoding all frames and
    discarding most, we compute exactly which native indices we need and
    call vr.get_batch() (or cv2 seek) on only those.

    Args:
        meta      : VideoMeta returned by open_video()
        frame_ids : list of 1-indexed logical frame IDs to decode

    Returns:
        FrameBundle with exactly the requested frames, in ascending order.
    """
    if not frame_ids:
        return []

    # Sort and deduplicate
    ids_sorted = sorted(set(frame_ids))

    # Clamp to valid range
    ids_sorted = [fid for fid in ids_sorted if 1 <= fid <= meta.total_frames]
    if not ids_sorted:
        return []

    # Convert 1-indexed logical IDs → 0-indexed native frame indices
    native_indices = [(fid - 1) * meta.step for fid in ids_sorted]

    if _DECORD_AVAILABLE and meta._vr is not None:
        frames_np = meta._vr.get_batch(native_indices).asnumpy()  # (N, H, W, 3) RGB
        return [(fid, Image.fromarray(arr))
                for fid, arr in zip(ids_sorted, frames_np)]
    else:
        return _fetch_cv2(meta.video_path, ids_sorted, native_indices)


def _fetch_cv2(video_path: str,
               frame_ids: List[int],
               native_indices: List[int]) -> FrameBundle:
    """cv2 fallback: seek to each required native index individually."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    bundle = []
    for fid, native_idx in zip(frame_ids, native_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)
        ret, frame = cap.read()
        if ret:
            rgb = frame[:, :, ::-1]
            bundle.append((fid, Image.fromarray(rgb)))
    cap.release()
    return bundle


# ─── Original full-decode loader (used by single_step / hierarchical) ──────────

def load_video_frames(video_path: str, fps: float = None) -> FrameBundle:
    """
    Decode *all* frames of video_path at `fps` frames-per-second.
    Returns a list of (frame_id, PIL.Image) where frame_id is 1-indexed.

    Still used by single_step and hierarchical variants which need the full
    bundle for neighbourhood expansion.  dynamic_segment now uses
    open_video() + fetch_frames() instead.
    """
    fps = fps or config.VIDEO_FPS

    if _DECORD_AVAILABLE:
        return _load_with_decord(video_path, fps)
    else:
        return _load_with_cv2_full(video_path, fps)


def _load_with_decord(video_path: str, fps: float) -> FrameBundle:
    vr           = VideoReader(video_path, ctx=decord_cpu(0))
    total_frames = len(vr)
    native_fps   = vr.get_avg_fps()
    step         = max(1, int(round(native_fps / fps)))
    native_indices = list(range(0, total_frames, step))
    frames_np    = vr.get_batch(native_indices).asnumpy()  # (N, H, W, 3) RGB

    bundle: FrameBundle = []
    for local_i, arr in enumerate(frames_np):
        img      = Image.fromarray(arr)
        frame_id = local_i + 1   # 1-indexed
        bundle.append((frame_id, img))
    return bundle


def _load_with_cv2_full(video_path: str, fps: float) -> FrameBundle:
    import cv2
    cap        = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step       = max(1, int(round(native_fps / fps)))

    bundle: FrameBundle = []
    frame_id   = 1
    native_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if native_idx % step == 0:
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            bundle.append((frame_id, img))
            frame_id += 1
        native_idx += 1

    cap.release()
    return bundle


# ─── Subsampling helpers ────────────────────────────────────────────────────────

def uniform_subsample(bundle: FrameBundle, n: int) -> FrameBundle:
    """
    Uniformly select exactly `n` frames from `bundle`.
    If len(bundle) <= n, return bundle as-is.
    """
    if n <= 0 or len(bundle) == 0:
        return []
    if len(bundle) <= n:
        return bundle

    float_indices = np.linspace(0, len(bundle) - 1, n)
    indices       = np.round(float_indices).astype(int)
    indices       = np.clip(indices, 0, len(bundle) - 1)
    return [bundle[i] for i in indices]


def uniform_subsample_ids(total: int, n: int) -> List[int]:
    """
    Compute n uniformly-spaced 1-indexed frame IDs from a video with
    `total` logical frames.  Pure arithmetic — no frame decoding.

    Used by dynamic_segment to plan which frames to decode before loading.
    """
    if n <= 0 or total == 0:
        return []
    if total <= n:
        return list(range(1, total + 1))

    float_indices = np.linspace(0, total - 1, n)
    indices       = np.round(float_indices).astype(int)
    indices       = np.clip(indices, 0, total - 1)
    # Convert 0-indexed positions → 1-indexed frame IDs
    return [int(i) + 1 for i in indices]


def segment_ids(total: int, num_segments: int) -> List[Tuple[int, int]]:
    """
    Return (start_frame_id, end_frame_id) ranges for each segment.
    Frame IDs are 1-indexed, end is exclusive (Python slice convention).

    Pure arithmetic — used to plan decode indices without loading frames.
    """
    seg_size = max(1, total // num_segments)
    ranges   = []
    for i in range(num_segments):
        start = i * seg_size + 1            # 1-indexed
        end   = (i + 1) * seg_size + 1 if i < num_segments - 1 else total + 1
        if start <= total:
            ranges.append((start, end))
    return ranges


def segment_bundle(bundle: FrameBundle, num_segments: int) -> List[FrameBundle]:
    """
    Split bundle into `num_segments` non-overlapping equal-length segments.
    Last segment may be slightly longer if not evenly divisible.

    Paper §3.2: 'we divide it into l non-overlapping segments of equal length.'
    """
    N        = len(bundle)
    seg_size = max(1, N // num_segments)
    segments = []
    for i in range(num_segments):
        start = i * seg_size
        end   = (i + 1) * seg_size if i < num_segments - 1 else N
        seg   = bundle[start:end]
        if seg:
            segments.append(seg)
    return segments


def frames_from_ids(bundle: FrameBundle, ids: List[int]) -> FrameBundle:
    """
    Return the sub-bundle for a list of frame_ids (1-indexed).
    IDs outside range are silently ignored.
    """
    id_set = set(ids)
    return [(fid, img) for (fid, img) in bundle if fid in id_set]


def get_frame_images(bundle: FrameBundle) -> List[Image.Image]:
    """Strip frame IDs and return only PIL images."""
    return [img for (_, img) in bundle]


def get_frame_ids(bundle: FrameBundle) -> List[int]:
    """Return only the frame IDs."""
    return [fid for (fid, _) in bundle]