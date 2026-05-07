"""Decord-only video reader guard for qwen-vl-utils.

Two patches applied at import time:
1. Force qwen-vl-utils to use decord as the primary backend.
2. Disable the silent torchvision fallback that would otherwise
   full-decode multi-minute lecture videos into CPU RAM (OOM).
   The dataset's __getitem__ already catches Exception and resamples
   the next index, so failed reads are skipped cleanly.
3. Belt-and-suspenders: monkey-patch decord.VideoReader to default
   to num_threads=1, eliminating the threaded_decoder.cc races we
   observed (Check failed: run_.load(), avcodec_send_packet -11).
"""
import os, sys

# ---- 1. force decord as the primary backend ----
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

# ---- 2. disable torchvision fallback in qwen_vl_utils ----
import qwen_vl_utils.vision_process as _vp

def _no_fallback(ele):
    raise RuntimeError(
        f"decord failed to read {ele.get('video')!r} and torchvision "
        f"fallback is disabled (it would full-decode the video and OOM). "
        f"Sample will be skipped by the dataset."
    )

_vp.VIDEO_READER_BACKENDS["torchvision"] = _no_fallback

# ---- 3. force decord to single-threaded decoding ----
try:
    import decord
    _orig_VR = decord.VideoReader

    def _SingleThreadVR(*args, **kwargs):
        kwargs.setdefault("num_threads", 1)
        return _orig_VR(*args, **kwargs)

    decord.VideoReader = _SingleThreadVR
    _decord_status = "patched (num_threads=1)"
except Exception as e:
    _decord_status = f"NOT patched: {e!r}"

print(
    f"[decord_only_guard] FORCE_QWENVL_VIDEO_READER=decord  "
    f"torchvision fallback DISABLED  decord {_decord_status}",
    file=sys.stderr, flush=True,
)
