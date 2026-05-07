import itertools
from build_pairs._io_utils import iter_train_rows, resolve_video_path, transcript_text, first_segment_seconds, transcript_text_in_range
n=ok_v=ok_t=ok_seg=0
sample=None
for r in itertools.islice(iter_train_rows(), 200):
    n+=1
    if resolve_video_path(r): ok_v+=1
    if transcript_text(r,200): ok_t+=1
    if first_segment_seconds(r): ok_seg+=1
print(f"first {n}: video={ok_v} transcript={ok_t} segments={ok_seg}")
