"""
stages/video_loading.py — Video metadata & frame decoding.

Uses decord for efficient random-access frame reading.
Frames are identified by their index at the *target* FPS (default 1 fps).
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("educot.video")

FrameBundle = List[Tuple[int, Image.Image]]


@dataclass
class VideoMeta:
    path: str
    total_frames: int      # at target_fps
    native_fps: float
    duration: float        # seconds
    target_fps: int


def open_video(path: str, fps: int = 1) -> VideoMeta:
    """Read video metadata (no pixel decode)."""
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    n_native = len(vr)
    duration = n_native / native_fps if native_fps > 0 else 0.0
    total_at_fps = max(1, int(duration * fps))

    return VideoMeta(
        path=path,
        total_frames=total_at_fps,
        native_fps=native_fps,
        duration=duration,
        target_fps=fps,
    )


def decode_frames(
    path: str,
    frame_ids: List[int],
    native_fps: float,
    target_fps: int = 1,
) -> FrameBundle:
    """
    Decode specific frame IDs (at target_fps) from the video file.
    Returns list of (frame_id, PIL.Image) sorted by frame_id.
    """
    if not frame_ids:
        return []

    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    n_native = len(vr)
    scale = native_fps / target_fps if target_fps > 0 else 1.0

    # Map target-fps IDs → native frame indices
    native_indices = []
    valid_fids = []
    for fid in sorted(set(frame_ids)):
        nat = min(int(fid * scale), n_native - 1)
        native_indices.append(nat)
        valid_fids.append(fid)

    if not native_indices:
        return []

    frames_np = vr.get_batch(native_indices).asnumpy()
    bundle = []
    for i, fid in enumerate(valid_fids):
        img = Image.fromarray(frames_np[i])
        bundle.append((fid, img))

    return bundle


# ─── Subsampling helpers ──────────────────────────────────────────────────

def uniform_subsample(bundle: FrameBundle, n: int) -> FrameBundle:
    """Uniformly subsample a FrameBundle down to n entries."""
    if len(bundle) <= n:
        return bundle
    indices = np.unique(np.linspace(0, len(bundle) - 1, n).round().astype(int))
    return [bundle[i] for i in indices]


def uniform_subsample_ids(ids: List[int], n: int) -> List[int]:
    """Uniformly subsample a list of integer IDs down to n entries."""
    if len(ids) <= n:
        return list(ids)
    indices = np.unique(np.linspace(0, len(ids) - 1, n).round().astype(int))
    return [ids[i] for i in indices]


def get_frame_ids(bundle: FrameBundle) -> List[int]:
    return [fid for fid, _ in bundle]


def get_frame_images(bundle: FrameBundle) -> List[Image.Image]:
    return [img for _, img in bundle]
