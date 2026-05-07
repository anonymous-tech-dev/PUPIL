"""
stages/context_aggregation.py — Assemble final answering context.

  c = selected_frames[clipped to k-u] ∪ uniform_frames[u]

Decodes only the final set of frames from disk.
"""

import logging
from typing import List

import numpy as np
from omegaconf import DictConfig

from stages.video_loading import (
    FrameBundle, VideoMeta, decode_frames, uniform_subsample_ids,
)

logger = logging.getLogger("educot.aggregation")


def assemble_context(
    video_path: str,
    selected_ids: List[int],
    meta: VideoMeta,
    k: int,
    u: int,
) -> FrameBundle:
    """
    Build final context  c = ˆx[m] ∪ x[u]  (paper §3.2).

    Steps:
      1. If |selected| > k-u, uniformly subsample to k-u.
      2. Pick u frames uniformly from the full video.
      3. Union + deduplicate + sort.
      4. Decode only those frames from disk.
    """
    m = k - u

    # (1) Clip selected
    sel = sorted(set(selected_ids))
    if len(sel) > m:
        sel = uniform_subsample_ids(sel, m)

    # (2) Uniform context
    uni: List[int] = []
    if u > 0 and meta.total_frames > 0:
        uni_arr = np.unique(
            np.linspace(0, meta.total_frames - 1, u).round().astype(int)
        )
        uni = [int(i) for i in uni_arr]

    # (3) Merge + deduplicate + sort
    all_ids = sorted(set(sel) | set(uni))

    # (4) Decode
    logger.debug(
        "  Context: %d selected + %d uniform → %d unique frames",
        len(sel), len(uni), len(all_ids),
    )
    return decode_frames(video_path, all_ids, meta.native_fps, meta.target_fps)
