"""Quick calibration: compute score distributions for each keyframe method on a sample video."""
import sys, os, json
import numpy as np
import cv2

# Find a sample video
meta_file = "/workspace/Pupil/frame_sampling_experiments/temporal_cot_gdm/video_meta.jsonl"
video_dir = "/data/Pupil/lvbench_v2"

with open(meta_file) as f:
    items = [json.loads(line) for line in f]

# Pick first 3 videos
test_videos = []
for item in items[:5]:
    key = item.get("key", item.get("video_id", ""))
    vp = os.path.join(video_dir, f"{key}_clean.mp4")
    if not os.path.exists(vp):
        vp = os.path.join(video_dir, f"{key}.mp4")
    if os.path.exists(vp):
        test_videos.append(vp)
    if len(test_videos) >= 3:
        break

from stages.keyframe_filter import _score_mad, _score_histogram, _score_dhash, _score_farneback
from decord import VideoReader, cpu

for vp in test_videos:
    print(f"\n=== {os.path.basename(vp)} ===")
    vr = VideoReader(vp, ctx=cpu(0), width=320, height=180)
    native_fps = vr.get_avg_fps()
    n = len(vr)
    # Sample at 1fps
    step = max(1, int(native_fps))
    indices = list(range(0, min(n, int(120*native_fps)), step))[:120]
    if len(indices) < 2:
        continue
    batch = vr.get_batch(indices).asnumpy()
    
    scores = {"mad": [], "histogram": [], "dhash": []}
    for i in range(len(batch)-1):
        g1 = cv2.cvtColor(batch[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(batch[i+1], cv2.COLOR_RGB2GRAY)
        scores["mad"].append(_score_mad(g1, g2))
        scores["histogram"].append(_score_histogram(g1, g2))
        scores["dhash"].append(_score_dhash(g1, g2))
    
    for method, vals in scores.items():
        arr = np.array(vals)
        print(f"  {method:12s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"p25={np.percentile(arr,25):.4f}  p50={np.percentile(arr,50):.4f}  "
              f"p75={np.percentile(arr,75):.4f}  p90={np.percentile(arr,90):.4f}  "
              f"max={arr.max():.4f}")
