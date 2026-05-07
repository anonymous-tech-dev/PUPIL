"""
make_sft_with_clips.py — produce SFT data variants that point at clipped videos
instead of (or in addition to) full lectures.

Design
------
Three modes, controlled by --video-type:

    full   — leave every record's `video` field unchanged (full lecture mp4)
    clip   — replace every record's `video` with the corresponding clip from
             --clips-dir; records without a valid clip are DROPPED.
    mixed  — per-record stratified-by-axis seeded random assignment.  With
             p = --clip-frac (default 0.8) a record is assigned the clip; with
             p = 1 - clip-frac it is left as full.  Records whose clip is bad
             or missing are forced to full (no drops).

Each output record gets a `video_type` field ("full" | "clip") so we can
audit the actual mix and slice training metrics by video type later.

Audit handling
--------------
The clip directory ships with three audit files:

    _BAD_CLIPS.txt          # decord-broken outputs (must NEVER be used)
    _AUDIT_MUST_DROP.txt    # clips judged structurally wrong (skip)
    _AUDIT_SUSPECT.txt      # clips judged borderline (warn but allow)

`clip` mode drops records on _BAD_CLIPS or _AUDIT_MUST_DROP.
`mixed` mode falls back to full for those.
SUSPECT records are kept by default (override with --drop-suspect).

CLI
---
    python make_sft_with_clips.py \\
        --in-train  data/sof_sft_warmstart.no_transcript.json \\
        --in-val    data/sof_sft_warmstart.val.no_transcript.json \\
        --clips-dir /data/Pupil/clips_v1 \\
        --out-tag   mix80 \\
        --video-type mixed \\
        --clip-frac 0.8 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Audit / clip discovery helpers
# ──────────────────────────────────────────────────────────────────────────────
def _read_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # _BAD_CLIPS.txt rows look like "<id>.mp4\t<size>\t<error>"
            # _AUDIT_*.txt rows are bare ids
            tok = line.split()[0]
            if tok.endswith(".mp4"):
                tok = tok[:-4]
            out.add(tok)
    return out


def _build_clip_index(clips_dir: Path) -> dict[str, Path]:
    """id -> /path/to/<id>.mp4 for every .mp4 in clips_dir (top-level only)."""
    idx: dict[str, Path] = {}
    for entry in os.listdir(clips_dir):
        if entry.endswith(".mp4"):
            idx[entry[:-4]] = clips_dir / entry
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Per-mode assignment
# ──────────────────────────────────────────────────────────────────────────────
def _assign(records: list[dict], mode: str, clip_frac: float, seed: int,
            clip_index: dict[str, Path], drop_set: set[str]) -> list[dict]:
    """
    Returns a new list of records (deep-shallow copies, fresh `video` and
    `video_type` fields).  Drops are applied for `clip` mode only.

    `mixed` mode uses per-axis stratified seeding so the 80/20 ratio is
    *exact* per axis (not just in expectation), which keeps the eval-time
    SoF-axis decomposition fair.
    """
    out: list[dict] = []
    n_drop = 0
    n_force_full = 0
    n_clip = 0
    n_full = 0

    if mode == "mixed":
        # Stratify by axis: shuffle each axis bucket with its own seed,
        # take the first ceil(N*clip_frac) as "clip" (if clip exists,
        # else fall back to full).
        by_axis: dict[str, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(records):
            by_axis[r.get("axis", "_unknown")].append(i)

        clip_idx_set: set[int] = set()
        for ax, idxs in sorted(by_axis.items()):
            rnd = random.Random(f"{seed}:{ax}")
            shuffled = list(idxs)
            rnd.shuffle(shuffled)
            n_clip_target = int(round(len(shuffled) * clip_frac))
            for i in shuffled[:n_clip_target]:
                clip_idx_set.add(i)

        for i, r in enumerate(records):
            assigned = "clip" if i in clip_idx_set else "full"
            new = dict(r)
            if assigned == "clip":
                clip_path = clip_index.get(r["id"])
                if clip_path is None or r["id"] in drop_set:
                    # Fall back to full
                    new["video_type"] = "full"
                    n_force_full += 1
                    n_full += 1
                else:
                    new["video"] = str(clip_path)
                    new["video_type"] = "clip"
                    n_clip += 1
            else:
                new["video_type"] = "full"
                n_full += 1
            out.append(new)

    elif mode == "clip":
        for r in records:
            if r["id"] in drop_set:
                n_drop += 1
                continue
            clip_path = clip_index.get(r["id"])
            if clip_path is None:
                n_drop += 1
                continue
            new = dict(r)
            new["video"] = str(clip_path)
            new["video_type"] = "clip"
            out.append(new)
            n_clip += 1

    elif mode == "full":
        for r in records:
            new = dict(r)
            new["video_type"] = "full"
            out.append(new)
            n_full += 1

    else:
        raise ValueError(f"Unknown video-type mode: {mode}")

    return out, dict(n_clip=n_clip, n_full=n_full,
                     n_drop=n_drop, n_force_full=n_force_full)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
def _process_split(in_path: Path, mode: str, clip_frac: float, seed: int,
                   clip_index: dict[str, Path], drop_set: set[str],
                   out_path: Path, label: str) -> None:
    records = json.load(open(in_path))
    out_recs, stats = _assign(records, mode, clip_frac, seed, clip_index, drop_set)

    # Per-axis breakdown for sanity
    axis_breakdown = collections.Counter(
        (r["axis"], r["video_type"]) for r in out_recs
    )

    print(f"\n[{label}] mode={mode}  in={len(records)}  out={len(out_recs)}")
    print(f"          clip={stats['n_clip']}  full={stats['n_full']}  "
          f"force_full(missing/dropped clip)={stats['n_force_full']}  "
          f"drop={stats['n_drop']}")
    print(f"          per-(axis, video_type):")
    for (ax, vt), n in sorted(axis_breakdown.items()):
        print(f"            {ax:9s} {vt:5s}  {n}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_recs, open(out_path, "w"), indent=2)
    print(f"          -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--in-train", required=True)
    ap.add_argument("--in-val", required=True)
    ap.add_argument("--clips-dir", required=True,
                    help="Directory holding <id>.mp4 clips and _AUDIT_*.txt files")
    ap.add_argument("--out-dir", default=None,
                    help="Default = parent of --in-train")
    ap.add_argument("--out-tag", required=True,
                    help="Suffix inserted before `.json`. e.g. 'mix80' -> "
                         "<basename>.mix80.json")
    ap.add_argument("--video-type", choices=["full", "clip", "mixed"],
                    required=True)
    ap.add_argument("--clip-frac", type=float, default=0.8,
                    help="Fraction of records assigned to clip in `mixed` "
                         "mode. Ignored otherwise.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-suspect", action="store_true",
                    help="Also drop ids listed in _AUDIT_SUSPECT.txt "
                         "(default: keep them).")
    args = ap.parse_args()

    in_train = Path(args.in_train)
    in_val = Path(args.in_val)
    clips_dir = Path(args.clips_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_train.parent

    # Build clip index + drop set
    clip_index = _build_clip_index(clips_dir)
    drop_set = (
        _read_id_set(clips_dir / "_BAD_CLIPS.txt") |
        _read_id_set(clips_dir / "_AUDIT_MUST_DROP.txt")
    )
    if args.drop_suspect:
        drop_set |= _read_id_set(clips_dir / "_AUDIT_SUSPECT.txt")

    print(f"clips: {len(clip_index)}   drop_set: {len(drop_set)}   "
          f"mode={args.video_type}   clip_frac={args.clip_frac}   seed={args.seed}")

    # Output paths: insert .<tag> before .json
    def _tagged(p: Path) -> Path:
        # foo.no_transcript.json  ->  foo.no_transcript.<tag>.json
        return out_dir / (p.stem + f".{args.out_tag}" + p.suffix)

    _process_split(in_train, args.video_type, args.clip_frac, args.seed,
                   clip_index, drop_set, _tagged(in_train), label="train")
    _process_split(in_val, args.video_type, args.clip_frac, args.seed,
                   clip_index, drop_set, _tagged(in_val), label="val")


if __name__ == "__main__":
    main()
