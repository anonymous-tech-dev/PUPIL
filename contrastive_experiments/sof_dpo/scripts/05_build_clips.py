#!/usr/bin/env python3
"""
05_build_clips.py
-----------------
Cut SoF-DPO training videos down to their ground-truth time windows so DPO sees
the relevant content instead of being asked to find a needle in a 30-min haystack.

Behaviour per row in the input json (which must already have `timestamp_segments`,
attached by 04_attach_timestamps.py):

  * 1 segment  -> single ffmpeg cut, re-encoded.
  * N segments (N>=2, only ever the `time` axis) -> for every pair of consecutive
    segments we splice a 1.5s black-screen "X minutes Y seconds later" interstitial
    between them, then concat. The gap text is computed from the actual elapsed
    wall-clock between seg[i].end and seg[i+1].start.

ALL clips are re-encoded to a single canonical spec so the concat demuxer is
happy and decord seeking is rock-solid:
    1280x720, 30fps, yuv420p, libx264 crf 20, AAC 44.1kHz stereo, +faststart

Outputs are written to:  <out-dir>/<query_id>.mp4
A manifest is written to: <out-dir>/_MANIFEST.jsonl
Failures are written to:  <out-dir>/_FAILED.jsonl
The script is fully resumable -- existing outputs are skipped.

Usage:
    python3 05_build_clips.py \
        --in-train data_with_timestamps_v1/sof_dpo_train.json \
        --in-val   data_with_timestamps_v1/sof_dpo_train.val.json \
        --out-dir  /data/Pupil/clips_v1 \
        --workers  16
"""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -------- canonical output spec ------------------------------------------------
W, H, FPS = 1280, 720, 30
CARD_SECS = 1.5
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Letterbox/pad real video cuts to the canonical canvas. Cards are already 1280x720.
SCALE_VF = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
# Codec args that must NOT contain another -vf (so make_card can supply drawtext).
VCODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
          "-pix_fmt", "yuv420p", "-r", str(FPS)]
ACODEC = ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
MOV    = ["-movflags", "+faststart"]
FFCOMMON = ["-y", "-hide_banner", "-loglevel", "error", "-nostdin"]

# -------- helpers --------------------------------------------------------------

def hms_to_secs(t: str) -> float:
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    else:
        h, m, s = 0.0, 0.0, parts[0]
    return h * 3600 + m * 60 + s


def gap_text(secs: float) -> str:
    secs = max(0, int(round(secs)))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} hours {m} minutes {s} seconds later"
    if m:
        return f"{m} minutes {s} seconds later"
    return f"{s} seconds later"


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stderr or p.stdout)


def cut_segment(src: str, start_s: float, end_s: float, dst: str) -> tuple[int, str]:
    dur = max(0.05, end_s - start_s)
    # -ss BEFORE -i = fast seek; we re-encode anyway so it's accurate enough.
    cmd = ["ffmpeg", *FFCOMMON,
           "-ss", f"{start_s:.3f}", "-i", src, "-t", f"{dur:.3f}",
           "-vf", SCALE_VF, *VCODEC, *ACODEC, *MOV, dst]
    return run(cmd)


def make_card(text: str, dst: str) -> tuple[int, str]:
    # drawtext filter parsing: ':' separates options, '\\' escapes. We pass argv via
    # subprocess (no shell), so DO NOT wrap the text in single quotes -- they'd
    # become literal characters in the rendered text. Just escape backslashes and
    # colons. Our gap_text() never produces ':' or '\\' but we stay defensive.
    safe = text.replace("\\", "\\\\").replace(":", r"\:")
    vf = (f"drawtext=fontfile={FONT}:text={safe}:"
          f"fontcolor=white:fontsize=56:x=(w-text_w)/2:y=(h-text_h)/2")
    cmd = ["ffmpeg", *FFCOMMON,
           "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={CARD_SECS}",
           "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
           "-shortest", "-vf", vf,
           *VCODEC, *ACODEC, *MOV, dst]
    return run(cmd)


def concat_files(parts: list[str], dst: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file {shlex.quote(os.path.abspath(p))}\n")
        listfile = f.name
    try:
        # parts already share the canonical codec/spec, so stream-copy concat works.
        cmd = ["ffmpeg", *FFCOMMON, "-f", "concat", "-safe", "0",
               "-i", listfile, "-c", "copy", *MOV, dst]
        return run(cmd)
    finally:
        os.unlink(listfile)


# -------- worker --------------------------------------------------------------

def process(row: dict, out_dir: Path, work_root: Path) -> dict:
    qid  = row["id"]
    src  = row["video"]
    segs = row.get("timestamp_segments") or []
    out  = out_dir / f"{qid}.mp4"
    rec  = {"id": qid, "src": src, "out": str(out),
            "n_segments": len(segs), "axis": row.get("axis"),
            "ok": False, "err": None}

    if out.exists() and out.stat().st_size > 0:
        rec["ok"] = True
        rec["skipped"] = True
        return rec

    if not segs:
        rec["err"] = "no timestamp_segments"
        return rec
    if not Path(src).exists():
        rec["err"] = f"missing source: {src}"
        return rec

    work = work_root / qid
    work.mkdir(parents=True, exist_ok=True)

    try:
        parts: list[str] = []
        # cut each segment
        seg_secs = [(hms_to_secs(s["start"]), hms_to_secs(s["end"])) for s in segs]
        for i, (a, b) in enumerate(seg_secs):
            piece = str(work / f"seg{i:02d}.mp4")
            rc, err = cut_segment(src, a, b, piece)
            if rc != 0 or not Path(piece).exists():
                raise RuntimeError(f"cut seg{i} failed: {err.strip()[:400]}")
            parts.append(piece)
            # interstitial card between this and next segment
            if i < len(seg_secs) - 1:
                gap = seg_secs[i + 1][0] - b
                card = str(work / f"card{i:02d}.mp4")
                rc, err = make_card(gap_text(gap), card)
                if rc != 0 or not Path(card).exists():
                    raise RuntimeError(f"card{i} failed: {err.strip()[:400]}")
                parts.append(card)

        if len(parts) == 1:
            os.replace(parts[0], out)
        else:
            rc, err = concat_files(parts, str(out))
            if rc != 0 or not out.exists():
                raise RuntimeError(f"concat failed: {err.strip()[:400]}")

        rec["ok"] = True
        rec["duration_s"] = round(sum(b - a for a, b in seg_secs)
                                  + (len(seg_secs) - 1) * CARD_SECS, 3)
    except Exception as e:
        rec["err"] = str(e)
    finally:
        # nuke scratch
        for p in work.glob("*"):
            try: p.unlink()
            except OSError: pass
        try: work.rmdir()
        except OSError: pass
    return rec


# -------- driver --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-train", required=True, type=Path)
    ap.add_argument("--in-val",   required=True, type=Path)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--limit",    type=int, default=0,
                    help="if >0, only process this many rows (smoke test)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.out_dir / "_work"
    work_root.mkdir(exist_ok=True)

    if not Path(FONT).exists():
        print(f"!! font missing: {FONT}  (apt-get install fonts-dejavu-core)", file=sys.stderr)
        sys.exit(2)

    # de-dup on id (train and val sometimes overlap in id-space — we want one clip per id)
    seen, rows = set(), []
    for src in (args.in_train, args.in_val):
        for r in json.load(src.open()):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append(r)

    if args.limit:
        rows = rows[: args.limit]

    print(f"[plan] {len(rows)} unique clips -> {args.out_dir}  (workers={args.workers})")

    manifest_path = args.out_dir / "_MANIFEST.jsonl"
    failed_path   = args.out_dir / "_FAILED.jsonl"
    n_ok = n_skip = n_fail = 0
    t0 = time.time()

    with manifest_path.open("a") as mf, failed_path.open("a") as ff, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, r, args.out_dir, work_root) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            (mf if rec["ok"] else ff).write(json.dumps(rec) + "\n")
            if rec["ok"]:
                if rec.get("skipped"): n_skip += 1
                else: n_ok += 1
            else:
                n_fail += 1
            if i % 50 == 0 or i == len(futs):
                rate = i / max(1e-3, time.time() - t0)
                print(f"  [{i}/{len(futs)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"({rate:.1f} clips/s)")

    # cleanup scratch root
    try: work_root.rmdir()
    except OSError: pass

    print(f"[done] ok={n_ok}  already-existed={n_skip}  failed={n_fail}")
    print(f"  manifest: {manifest_path}")
    print(f"  failures: {failed_path}")


if __name__ == "__main__":
    main()
