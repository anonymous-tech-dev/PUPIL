"""
stages/keyframe_filter.py — Keyframe extraction with multiple backends.

Methods:
  "farneback"  — Farnebäck dense optical-flow (accurate but SLOW, offline).
  "mad"        — Mean Absolute Difference (fast, online — mirrors scene-detect).
  "histogram"  — Chi-squared histogram distance (fast, online).
  "dhash"      — Perceptual difference-hash Hamming distance (fastest, online).

The MAD/histogram/dhash methods are 10-50× faster than Farnebäck because
they avoid per-pixel optical-flow computation entirely.  They operate on
the same downscaled frames and produce the same interface: a boolean
"keep this frame" decision per candidate.
"""

import logging
from typing import List

import cv2
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger("educot.keyframe_filter")


# ═══════════════════════════════════════════════════════════════════════════
# Scoring functions — each returns a float "change score" between two frames
# ═══════════════════════════════════════════════════════════════════════════

def _score_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean optical-flow magnitude (Farnebäck). Expensive."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=12,
        iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def _score_mad(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean Absolute Difference — same idea as MAD scene-detect."""
    return float(np.mean(np.abs(
        curr_gray.astype(np.float32) - prev_gray.astype(np.float32)
    )))


def _score_histogram(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Chi-squared distance between 256-bin grayscale histograms."""
    h1 = cv2.calcHist([prev_gray], [0], None, [256], [0, 256]).flatten()
    h2 = cv2.calcHist([curr_gray], [0], None, [256], [0, 256]).flatten()
    # Normalise to probability distributions
    h1 = h1 / (h1.sum() + 1e-8)
    h2 = h2 / (h2.sum() + 1e-8)
    return float(cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CHISQR
    ))


def _dhash(gray: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Compute a difference hash (binary vector)."""
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def _score_dhash(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Hamming distance between perceptual difference hashes (0-1 normalised)."""
    h1 = _dhash(prev_gray)
    h2 = _dhash(curr_gray)
    return float(np.count_nonzero(h1 != h2)) / len(h1)


_SCORE_FN = {
    "farneback": _score_farneback,
    "mad": _score_mad,
    "histogram": _score_histogram,
    "dhash": _score_dhash,
}


def _decode_native_frames(video_path: str, native_indices: List[int]) -> List[np.ndarray]:
    """Batch-decode specific native frame indices using decord (FFmpeg-backed)."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    clamped = [min(idx, n - 1) for idx in native_indices]
    batch = vr.get_batch(clamped).asnumpy()  # (N, H, W, 3) RGB
    return [batch[i] for i in range(len(batch))]


def _filter_ids_with_frames(
    candidate_ids: List[int],
    rgb_frames: List[np.ndarray],
    threshold: float,
    max_width: int,
    min_keep: int,
    method: str = "farneback",
) -> List[int]:
    """Core filtering logic given pre-decoded frames."""
    if len(candidate_ids) <= min_keep:
        return candidate_ids

    score_fn = _SCORE_FN.get(method, _score_farneback)

    # Get dimensions from first decoded frame
    height, width = rgb_frames[0].shape[:2]
    longest = max(width, height)
    scale = min(1.0, max_width / longest)
    sw, sh = int(width * scale), int(height * scale)

    kept: List[int] = [candidate_ids[0]]  # always keep first frame
    prev_gray = None

    for i, fid in enumerate(candidate_ids):
        frame = rgb_frames[i]

        if scale < 1.0:
            frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        score = score_fn(prev_gray, gray)
        if score >= threshold:
            kept.append(fid)

        prev_gray = gray

    # Guarantee minimum keyframes
    if len(kept) < min_keep and len(candidate_ids) >= min_keep:
        indices = np.unique(
            np.linspace(0, len(candidate_ids) - 1, min_keep).round().astype(int)
        )
        kept = [candidate_ids[i] for i in indices]

    return sorted(set(kept))


def filter_keyframes_batch(
    video_path: str,
    segments,                   # List[Segment] — each has .frame_ids
    native_fps: float,
    cfg: DictConfig,
) -> List[int]:
    """
    Batch keyframe filtering across ALL segments with a single video open.

    Instead of opening the video once per segment (expensive with 400+
    segments from scene-detect), we:
      1. Collect all unique candidate frame IDs across all segments.
      2. Open the video ONCE and batch-decode all candidate frames.
      3. Run per-segment optical-flow filtering using the pre-decoded frames.

    Returns:
        Flat sorted list of surviving frame IDs.
    """
    kf = cfg.keyframe_filter
    min_keep = kf.min_keyframes_per_segment
    threshold = kf.motion_threshold
    max_width = kf.max_frame_width
    method = getattr(kf, 'method', 'farneback')
    target_fps = cfg.video_fps
    fps_scale = native_fps / target_fps if target_fps > 0 else 1.0

    # 1. Collect all unique candidate IDs and their native indices
    all_ids: List[int] = []
    for seg in segments:
        if seg.frame_ids:
            all_ids.extend(seg.frame_ids)
    all_ids_unique = sorted(set(all_ids))

    if not all_ids_unique:
        return []

    native_indices = [int(fid * fps_scale) for fid in all_ids_unique]

    # 2. Single video open, single batch decode
    try:
        rgb_frames = _decode_native_frames(video_path, native_indices)
    except Exception as e:
        logger.warning("Cannot decode video for keyframe filter: %s — %s", video_path, e)
        return all_ids_unique

    # Build lookup: frame_id → decoded frame
    fid_to_frame = {fid: rgb_frames[i] for i, fid in enumerate(all_ids_unique)}

    # 3. Per-segment filtering using pre-decoded frames
    kept_all: List[int] = []
    for seg in segments:
        seg_ids = seg.frame_ids
        if not seg_ids:
            continue

        seg_frames = [fid_to_frame[fid] for fid in seg_ids if fid in fid_to_frame]
        seg_ids_valid = [fid for fid in seg_ids if fid in fid_to_frame]

        if not seg_ids_valid:
            continue

        kept = _filter_ids_with_frames(
            seg_ids_valid, seg_frames, threshold, max_width, min_keep, method,
        )
        kept_all.extend(kept)

    result = sorted(set(kept_all))
    logger.info(
        "  Keyframe filter (batch, method=%s): %d → %d frames (threshold=%.4f)",
        method, len(all_ids_unique), len(result), threshold,
    )
    return result


def filter_keyframes(
    video_path: str,
    candidate_ids: List[int],
    native_fps: float,
    cfg: DictConfig,
) -> List[int]:
    """
    Run Farnebäck optical flow on candidate frames and keep only those
    with motion score ≥ threshold.

    Uses decord for frame decoding (works on headless clusters where
    OpenCV is built without FFmpeg).  Optical-flow computation still
    uses cv2.calcOpticalFlowFarneback (CPU-only, no codec needed).

    Args:
        video_path    : path to the video file
        candidate_ids : frame IDs at target_fps (e.g. 1fps)
        native_fps    : the video's actual fps (for seeking)
        cfg           : full DictConfig (reads cfg.keyframe_filter.*)

    Returns:
        Filtered list of frame IDs (sorted, deduplicated).
    """
    kf = cfg.keyframe_filter
    min_keep = kf.min_keyframes_per_segment

    if len(candidate_ids) <= min_keep:
        return candidate_ids

    threshold = kf.motion_threshold
    max_width = kf.max_frame_width
    method = getattr(kf, 'method', 'farneback')
    target_fps = cfg.video_fps

    # Map target-fps IDs → native frame indices
    fps_scale = native_fps / target_fps if target_fps > 0 else 1.0
    native_indices = [int(fid * fps_scale) for fid in candidate_ids]

    # Batch-decode all candidate frames via decord
    try:
        rgb_frames = _decode_native_frames(video_path, native_indices)
    except Exception as e:
        logger.warning("Cannot decode video for keyframe filter: %s — %s", video_path, e)
        return candidate_ids

    if not rgb_frames:
        return candidate_ids

    return _filter_ids_with_frames(
        candidate_ids, rgb_frames, threshold, max_width, min_keep, method,
    )
