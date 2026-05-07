    """
visualizer.py — TCoT Visualizer / Debugger.

Runs TCoT on a SINGLE video and saves a detailed visual + textual debug
report of every step:

  viz_output/
  └── <video_stem>_<timestamp>/
      ├── 00_full_timeline.png        — all frames as a filmstrip
      ├── 01_segment_overview.png     — segmentation map
      ├── 02_segment_<i>_input.png    — frames fed to selector for segment i
      ├── 03_segment_<i>_selected.png — frames selected within segment i
      ├── 04_final_context.png        — final curated context passed to answerer
      ├── 05_answer_response.txt      — raw answering call output
      ├── report.md                   — full human-readable Markdown report
      └── selections.json             — machine-readable selection trace

HOW TO USE:
  1. Set VIZ_VIDEO_PATH, VIZ_QUESTION, VIZ_ANSWER_CHOICES in config.py
  2. Run:  python visualizer.py
"""

import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from models.factory import get_model
from stages.stage0_video_loading import (
    load_video_frames, FrameBundle, uniform_subsample,
    segment_bundle, get_frame_ids, get_frame_images,
)
from stages.stage1_prompts import build_selection_prompt, build_answering_prompt
from stages.stage2_selection_parsing import parse_selection_response, extract_answer_letter
from stages.stage3_context_aggregation import (
    selection_call, _assemble_context,
    single_step_tcot, dynamic_segment_tcot, hierarchical_tcot,
)
from stages.stage4_answering import answer_question

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tcot.viz")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────────────────────────────────────────

THUMB_W = 160
THUMB_H = 90
FONT_SIZE = 12
COLS_PER_ROW = 16
PAD = 4
HIGHLIGHT_COLOR = (50, 200, 50)    # green for selected
UNIFORM_COLOR   = (50, 120, 255)   # blue for uniform context
NORMAL_COLOR    = (180, 180, 180)  # grey border for non-selected


def _load_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def make_filmstrip(
    bundle     : FrameBundle,
    title      : str = "",
    highlight  : set = None,   # frame_ids to highlight in green
    secondary  : set = None,   # frame_ids to highlight in blue (uniform ctx)
    cols       : int = COLS_PER_ROW,
    thumb_w    : int = THUMB_W,
    thumb_h    : int = THUMB_H,
) -> Image.Image:
    """
    Render `bundle` as a filmstrip image.
    `highlight` → green border (selected frames).
    `secondary` → blue border (uniform context frames).
    """
    highlight = highlight or set()
    secondary = secondary or set()
    font = _load_font()

    n_frames = len(bundle)
    rows = math.ceil(n_frames / cols)

    cell_w = thumb_w + 2 * PAD
    cell_h = thumb_h + 2 * PAD + 16   # 16px for label

    img_w = cols * cell_w
    img_h = rows * cell_h + (30 if title else 0)

    canvas = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    if title:
        draw.text((4, 4), title, font=font, fill=(220, 220, 220))

    y_offset = 30 if title else 0
    for i, (fid, img) in enumerate(bundle):
        col = i % cols
        row = i // cols
        x = col * cell_w + PAD
        y = y_offset + row * cell_h + PAD

        # Border colour
        if fid in highlight:
            border_color = HIGHLIGHT_COLOR
            bw = 3
        elif fid in secondary:
            border_color = UNIFORM_COLOR
            bw = 2
        else:
            border_color = NORMAL_COLOR
            bw = 1

        # Draw border rect
        draw.rectangle([x - bw, y - bw,
                         x + thumb_w + bw, y + thumb_h + bw],
                        outline=border_color, width=bw)

        # Paste thumbnail
        thumb = img.copy().resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(thumb, (x, y))

        # Frame ID label
        label = f"F{fid}"
        draw.text((x, y + thumb_h + 2), label,
                  font=font,
                  fill=(HIGHLIGHT_COLOR if fid in highlight
                        else UNIFORM_COLOR if fid in secondary
                        else (180, 180, 180)))

    return canvas


def make_selection_timeline(
    full_bundle    : FrameBundle,
    selected_ids   : List[int],
    uniform_ids    : List[int],
    title          : str = "Selection Timeline",
    height         : int = 60,
) -> Image.Image:
    """
    A horizontal bar showing the full video duration with coloured ticks
    for selected frames.
    """
    if not full_bundle:
        return Image.new("RGB", (800, height), (30, 30, 30))

    total_fids = get_frame_ids(full_bundle)
    min_fid = total_fids[0]
    max_fid = total_fids[-1]
    width = max(800, len(total_fids) * 2)

    canvas = Image.new("RGB", (width, height), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)
    font   = _load_font()

    # Background bar
    draw.rectangle([0, 20, width - 1, 40], fill=(60, 60, 60))

    sel_set = set(selected_ids)
    uni_set = set(uniform_ids)

    for fid in total_fids:
        x = int((fid - min_fid) / max(1, max_fid - min_fid) * (width - 1))
        if fid in sel_set:
            draw.rectangle([x - 1, 15, x + 1, 45], fill=HIGHLIGHT_COLOR)
        elif fid in uni_set:
            draw.rectangle([x - 1, 18, x + 1, 42], fill=UNIFORM_COLOR)

    draw.text((2, 2), title, font=font, fill=(220, 220, 220))
    draw.text((2, height - 14),
              f"Total: {len(total_fids)} | Selected: {len(selected_ids)} | "
              f"Uniform: {len(uniform_ids)}",
              font=font, fill=(200, 200, 200))

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Detailed per-segment trace (Dynamic-Segment variant)
# ─────────────────────────────────────────────────────────────────────────────

class VizTracer:
    """Collects step-by-step trace data during a TCoT run."""

    def __init__(self):
        self.segments       : List[dict] = []   # per-segment info
        self.final_context  : FrameBundle = []
        self.selected_ids   : List[int] = []
        self.uniform_ids    : List[int] = []
        self.justifications : List[str] = []
        self.raw_responses  : List[str] = []
        self.answer_result  : dict = {}
        self.full_bundle    : FrameBundle = []


def run_tcot_with_trace(
    model,
    full_bundle    : FrameBundle,
    question       : str,
    answer_choices : List[str],
    tracer         : VizTracer,
) -> dict:
    """
    Run Dynamic-Segment TCoT while populating tracer with intermediate state.
    """
    l = config.NUM_SEGMENTS
    s = config.FRAMES_PER_SEGMENT
    k = config.CONTEXT_BUDGET_FRAMES
    u = config.UNIFORM_CONTEXT_FRAMES

    tracer.full_bundle = full_bundle
    segments = segment_bundle(full_bundle, l)

    all_selected_ids : List[int] = []

    for seg_i, seg in enumerate(segments):
        seg_sampled = uniform_subsample(seg, s)
        if not seg_sampled:
            continue

        result = selection_call(model, seg_sampled, question, answer_choices)

        seg_info = {
            "segment_idx"      : seg_i,
            "total_segments"   : len(segments),
            "seg_frame_ids"    : get_frame_ids(seg),
            "input_frame_ids"  : get_frame_ids(seg_sampled),
            "selected_ids"     : result["selected_ids"],
            "justification"    : result["justification"],
            "raw_response"     : result["raw_response"],
            "input_bundle"     : seg_sampled,
            "selected_bundle"  : [
                (fid, img) for (fid, img) in seg_sampled
                if fid in set(result["selected_ids"])
            ],
        }
        tracer.segments.append(seg_info)
        all_selected_ids.extend(result["selected_ids"])
        tracer.justifications.append(
            f"[Segment {seg_i+1}] {result['justification']}"
        )
        tracer.raw_responses.append(result["raw_response"])

        logger.info("  Segment %d/%d: %d/%d frames selected",
                    seg_i + 1, len(segments),
                    len(result["selected_ids"]),
                    len(seg_sampled))

    # Deduplicate
    seen = set()
    deduped = []
    for fid in all_selected_ids:
        if fid not in seen:
            seen.add(fid)
            deduped.append(fid)

    tracer.selected_ids = sorted(deduped)
    context = _assemble_context(full_bundle, deduped, k, u)
    tracer.final_context = context

    # Uniform context IDs (context frames NOT in selected_ids)
    sel_set = set(deduped)
    tracer.uniform_ids = [fid for (fid, _) in context if fid not in sel_set]

    # Answering
    ans = answer_question(model, context, question, answer_choices)
    tracer.answer_result = ans

    return {
        "context_bundle"  : context,
        "selected_ids"    : tracer.selected_ids,
        "justifications"  : tracer.justifications,
        "raw_responses"   : tracer.raw_responses,
        "stage"           : "dynamic_segment",
        "answer"          : ans,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save all visualizations
# ─────────────────────────────────────────────────────────────────────────────

def save_visualizations(tracer: VizTracer, out_dir: str,
                         question: str, answer_choices: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    font = _load_font()

    sel_set  = set(tracer.selected_ids)
    uni_set  = set(tracer.uniform_ids)

    # 00 — Full timeline filmstrip
    logger.info("  Saving full timeline filmstrip …")
    fs = make_filmstrip(
        tracer.full_bundle,
        title=f"Full Video ({len(tracer.full_bundle)} frames @ {config.VIDEO_FPS}fps)",
        highlight=sel_set,
        secondary=uni_set,
    )
    fs.save(os.path.join(out_dir, "00_full_timeline.png"))

    # 01 — Selection timeline bar
    tl = make_selection_timeline(
        tracer.full_bundle,
        tracer.selected_ids,
        tracer.uniform_ids,
        title="Selection Timeline (green=model-selected, blue=uniform context)",
    )
    tl.save(os.path.join(out_dir, "01_selection_timeline.png"))

    # 02 / 03 — Per-segment input and selection
    for seg in tracer.segments:
        seg_i   = seg["segment_idx"]
        seg_sel = set(seg["selected_ids"])

        inp = make_filmstrip(
            seg["input_bundle"],
            title=f"Segment {seg_i+1}/{seg['total_segments']} — Input "
                  f"({len(seg['input_bundle'])} frames, "
                  f"IDs {seg['input_frame_ids'][0]}–{seg['input_frame_ids'][-1]})",
            highlight=seg_sel,
        )
        inp.save(os.path.join(out_dir, f"02_segment_{seg_i:02d}_input.png"))

        if seg["selected_bundle"]:
            sel_img = make_filmstrip(
                seg["selected_bundle"],
                title=f"Segment {seg_i+1} — Selected "
                      f"({len(seg['selected_ids'])} frames)",
                highlight=set(seg["selected_ids"]),
            )
        else:
            sel_img = Image.new("RGB", (400, 100), (50, 30, 30))
            d = ImageDraw.Draw(sel_img)
            d.text((10, 40), f"Segment {seg_i+1}: No frames selected.", fill=(200, 200, 200))

        sel_img.save(os.path.join(out_dir, f"03_segment_{seg_i:02d}_selected.png"))

    # 04 — Final context
    logger.info("  Saving final context filmstrip …")
    ctx_img = make_filmstrip(
        tracer.final_context,
        title=f"Final Context c ({len(tracer.final_context)} frames) "
              f"= Model-selected (green) ∪ Uniform (blue)",
        highlight=sel_set,
        secondary=uni_set,
    )
    ctx_img.save(os.path.join(out_dir, "04_final_context.png"))

    # 05 — Raw answer text
    ans_path = os.path.join(out_dir, "05_answer_response.txt")
    with open(ans_path, "w") as f:
        f.write(f"Question: {question}\n\n")
        f.write("Answer Choices:\n")
        for i, c in enumerate(answer_choices):
            f.write(f"  ({chr(65+i)}) {c}\n")
        f.write("\n")
        f.write(f"Predicted: {tracer.answer_result.get('predicted_letter', '?')}\n\n")
        f.write("Raw VLM Response:\n")
        f.write(tracer.answer_result.get("raw_response", "") + "\n")

    # selections.json
    trace_data = {
        "question"        : question,
        "answer_choices"  : answer_choices,
        "predicted"       : tracer.answer_result.get("predicted_letter", ""),
        "total_frames"    : len(tracer.full_bundle),
        "selected_ids"    : tracer.selected_ids,
        "uniform_ids"     : tracer.uniform_ids,
        "context_ids"     : get_frame_ids(tracer.final_context),
        "pct_selected"    : (
            100.0 * len(tracer.selected_ids) / len(tracer.full_bundle)
            if tracer.full_bundle else 0.0
        ),
        "segments"        : [
            {
                "segment_idx"    : s["segment_idx"],
                "input_ids"      : s["input_frame_ids"],
                "selected_ids"   : s["selected_ids"],
                "justification"  : s["justification"],
            }
            for s in tracer.segments
        ],
        "final_justification": tracer.answer_result.get("raw_response", ""),
    }
    with open(os.path.join(out_dir, "selections.json"), "w") as f:
        json.dump(trace_data, f, indent=2)

    # Markdown report
    _write_markdown_report(tracer, out_dir, question, answer_choices)

    logger.info("All visualizations saved to: %s", out_dir)


def _write_markdown_report(tracer: VizTracer, out_dir: str,
                            question: str, answer_choices: List[str]):
    lines = []
    lines.append("# TCoT Visualization Report\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Model:** {config.MODEL}  ")
    lines.append(f"**Variant:** {config.TCOT_VARIANT}  ")
    lines.append(f"**Video FPS:** {config.VIDEO_FPS}  ")
    lines.append(f"**l (segments):** {config.NUM_SEGMENTS}  ")
    lines.append(f"**s (frames/seg):** {config.FRAMES_PER_SEGMENT}  ")
    lines.append(f"**k (context budget):** {config.CONTEXT_BUDGET_FRAMES}  ")
    lines.append(f"**u (uniform ctx):** {config.UNIFORM_CONTEXT_FRAMES}  \n")

    lines.append("## Question\n")
    lines.append(f"> {question}\n")
    lines.append("### Answer Choices\n")
    for i, c in enumerate(answer_choices):
        lines.append(f"- **({chr(65+i)})** {c}")
    lines.append("")

    lines.append("## Video Summary\n")
    lines.append(f"- Total frames (@ {config.VIDEO_FPS}fps): **{len(tracer.full_bundle)}**")
    lines.append(f"- Model-selected frames: **{len(tracer.selected_ids)}**  "
                 f"({100.0 * len(tracer.selected_ids) / max(1, len(tracer.full_bundle)):.1f}%)")
    lines.append(f"- Uniform context frames: **{len(tracer.uniform_ids)}**")
    lines.append(f"- Total context frames: **{len(tracer.final_context)}**\n")

    lines.append("## Segment-by-Segment Trace\n")
    for seg in tracer.segments:
        n = seg["total_segments"]
        i = seg["segment_idx"]
        ids = seg["seg_frame_ids"]
        inp = seg["input_frame_ids"]
        sel = seg["selected_ids"]
        lines.append(f"### Segment {i+1}/{n}  "
                     f"(frames {ids[0]}–{ids[-1]}, input={len(inp)}, selected={len(sel)})\n")
        lines.append(f"**Selected IDs:** {sel}\n")
        lines.append(f"**Justification:**  \n> {seg['justification']}\n")
        lines.append(f"![Segment {i+1} Input](02_segment_{i:02d}_input.png)\n")
        lines.append(f"![Segment {i+1} Selected](03_segment_{i:02d}_selected.png)\n")

    lines.append("## Final Context\n")
    lines.append(f"![Final Context](04_final_context.png)\n")
    lines.append(f"![Selection Timeline](01_selection_timeline.png)\n")

    lines.append("## Answering Call\n")
    ans = tracer.answer_result
    lines.append(f"**Predicted:** `{ans.get('predicted_letter', '?')}`\n")
    lines.append("**Raw VLM Response:**\n```\n"
                 + ans.get("raw_response", "") + "\n```\n")

    lines.append("## Failure Mode Indicators\n")
    if len(tracer.selected_ids) / max(1, len(tracer.full_bundle)) > 0.5:
        lines.append("⚠️ **High recall / low precision**: model selected >50% of "
                     "frames — possibly over-eager.")
    elif len(tracer.selected_ids) < 5:
        lines.append("⚠️ **Low recall**: fewer than 5 frames selected — "
                     "may have missed key moments.")
    else:
        lines.append("✅ Selection proportion looks reasonable.")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    video_path     = config.VIZ_VIDEO_PATH
    question       = config.VIZ_QUESTION
    answer_choices = config.VIZ_ANSWER_CHOICES

    if not video_path:
        print("ERROR: Set config.VIZ_VIDEO_PATH to the video you want to visualize.")
        sys.exit(1)
    if not question:
        print("ERROR: Set config.VIZ_QUESTION.")
        sys.exit(1)
    if not answer_choices:
        print("WARNING: config.VIZ_ANSWER_CHOICES is empty — open-ended mode.")

    # Output directory
    stem = os.path.splitext(os.path.basename(video_path))[0]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config.VIZ_OUTPUT_DIR, f"{stem}_{ts}")

    logger.info("TCoT Visualizer")
    logger.info("  Video   : %s", video_path)
    logger.info("  Question: %s", question)
    logger.info("  Output  : %s", out_dir)

    # Load model
    model = get_model()
    model.load()

    # Stage 0: Load frames
    logger.info("Stage 0: Loading frames …")
    full_bundle = load_video_frames(video_path, fps=config.VIDEO_FPS)
    logger.info("  Loaded %d frames.", len(full_bundle))

    # Run TCoT with trace
    tracer = VizTracer()
    logger.info("Stage 3+4: Running TCoT with trace …")
    result = run_tcot_with_trace(model, full_bundle, question,
                                  answer_choices, tracer)

    predicted = result["answer"].get("predicted_letter", "?")
    logger.info("Prediction: %s", predicted)

    # Save visualizations
    logger.info("Saving visualizations …")
    save_visualizations(tracer, out_dir, question, answer_choices)

    print(f"\n✓ Done. Open {out_dir}/report.md for the full report.")
    model.unload()


if __name__ == "__main__":
    main()