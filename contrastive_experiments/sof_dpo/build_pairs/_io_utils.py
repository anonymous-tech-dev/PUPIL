"""
Shared helpers for the SoF-DPO data-construction pipeline.

Conventions:
* Every script reads the train JSONL produced by the dataset team:
  /workspace/Pupil/dataset_curation/dataset/queries_db/final_train/all_Pupil_train_queries_v0.jsonl
* Each row carries: query_id, question, ground_truth, annotations.{pipeline_mode, cognitive_category},
  timestamp_segments, source_folder, video_path.
* `video_path` in the JSONL points at /datadisk/edubench_train_vids/... which is NOT mounted
  in this container.  We re-resolve to /data/Pupil/Pupil_train/{vids,vids2}.
* Transcripts live as .srt next to the videos: transcripts_db/<basename>_transcript.srt
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_JSONL = Path(
    "/workspace/Pupil/dataset_curation/dataset/queries_db/final_train/all_Pupil_train_queries_v0.jsonl"
)
VID_DIRS = [
    Path("/data/Pupil/Pupil_train/vids"),
    Path("/data/Pupil/Pupil_train/vids2"),
]
TRANSCRIPT_DIR = Path("/data/Pupil/Pupil_train/transcripts_db")

SOF_AXES = ("visual", "audio", "time", "priority")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def iter_train_rows(path: Path = TRAIN_JSONL) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_video_path(row: dict) -> str | None:
    """Map the JSONL's /datadisk path -> our actual mount; return None if missing."""
    name = os.path.basename(row["video_path"])
    for d in VID_DIRS:
        p = d / name
        if p.exists():
            return str(p)
    return None


def video_basename(row: dict) -> str:
    return os.path.basename(row["video_path"]).rsplit(".", 1)[0]


# ---------------------------------------------------------------------------
# Transcript loading (SRT -> plain text, with optional time slicing)
# ---------------------------------------------------------------------------
_SRT_TS = re.compile(
    r"^(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)\s*$"
)


def _hms_to_sec(h: str, m: str, s: str, ms: str = "0") -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def transcript_path_for(row: dict) -> Path | None:
    base = video_basename(row)
    # transcripts use the raw basename (e.g. "..._clean") + "_transcript.srt"
    p = TRANSCRIPT_DIR / f"{base}_transcript.srt"
    if p.exists():
        return p
    # some transcripts drop the trailing _clean; try a fallback.
    base2 = re.sub(r"_clean$", "", base)
    p2 = TRANSCRIPT_DIR / f"{base2}_transcript.srt"
    return p2 if p2.exists() else None


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """Return list of (start_sec, end_sec, text)."""
    out: list[tuple[float, float, str]] = []
    if path is None or not path.exists():
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    blocks = re.split(r"\n\s*\n", text.strip())
    for blk in blocks:
        lines = blk.strip().splitlines()
        if len(lines) < 2:
            continue
        # First non-numeric line is the timestamp
        ts_idx = 0
        if lines[0].strip().isdigit():
            ts_idx = 1
        if ts_idx >= len(lines):
            continue
        m = _SRT_TS.match(lines[ts_idx])
        if not m:
            continue
        start = _hms_to_sec(*m.group(1, 2, 3, 4))
        end = _hms_to_sec(*m.group(5, 6, 7, 8))
        body = " ".join(s.strip() for s in lines[ts_idx + 1 :]).strip()
        if body:
            out.append((start, end, body))
    return out


def transcript_text(row: dict, max_chars: int = 8000) -> str:
    cues = parse_srt(transcript_path_for(row))
    if not cues:
        return ""
    text = " ".join(c[2] for c in cues)
    return text[:max_chars]


def transcript_text_in_range(row: dict, t0: float, t1: float, max_chars: int = 4000) -> str:
    cues = parse_srt(transcript_path_for(row))
    if not cues:
        return ""
    keep = [c[2] for c in cues if not (c[1] < t0 or c[0] > t1)]
    return (" ".join(keep))[:max_chars]


# ---------------------------------------------------------------------------
# Timestamp parsing for `time` axis ablation
# ---------------------------------------------------------------------------
_HMS = re.compile(r"^(\d+):(\d+):(\d+)(?:[,.](\d+))?$")
_MS = re.compile(r"^(\d+):(\d+)(?:[,.](\d+))?$")


def _ts_to_sec(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    m = _HMS.match(s)
    if m:
        h, mi, se, ms = m.group(1), m.group(2), m.group(3), m.group(4) or "0"
        return _hms_to_sec(h, mi, se, ms)
    m = _MS.match(s)
    if m:
        mi, se, ms = m.group(1), m.group(2), m.group(3) or "0"
        return _hms_to_sec("0", mi, se, ms)
    try:
        return float(s)
    except Exception:
        return 0.0


def first_segment_seconds(row: dict) -> tuple[float, float] | None:
    segs = row.get("timestamp_segments") or []
    if not segs:
        return None
    s = segs[0]
    return (_ts_to_sec(s.get("start", "")), _ts_to_sec(s.get("end", "")))


# ---------------------------------------------------------------------------
# Sharding helper (8-GPU friendly)
# ---------------------------------------------------------------------------
def shard(rows: Iterable[dict], shard_id: int, num_shards: int) -> Iterator[dict]:
    for i, r in enumerate(rows):
        if i % num_shards == shard_id:
            yield r
