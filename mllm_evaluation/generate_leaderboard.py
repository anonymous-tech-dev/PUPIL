#!/usr/bin/env python3
"""
Leaderboard Generator for Pupil
========================================
Scans the results directory, computes accuracy breakdowns, and writes a
leaderboard.md that auto-updates as new model runs come in.

Usage:
    python generate_leaderboard.py                  # uses default paths
    python generate_leaderboard.py --results_dir /path/to/results --query_file /path/to/queries.json
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_QUERY_FILE = os.path.join(
    SCRIPT_DIR, "..", "dataset_curation", "dataset",
    "queries_db", "final_1k", "final_consolidated_1k_final_v0.json",
)
OUTPUT_MD = os.path.join(SCRIPT_DIR, "leaderboard.md")

# ── Canonical ordering ───────────────────────────────────────────────────────
ALL_MODELS = [
    "gpt5_Azure",
    "claude_sonnet_46",
    "claude_opus_46",
    "qwen3_vl",
    "qwen3_vl_ft",
    "qwen32_vl",
    "qwen2_5_vl",
    "intern_3_vl",
    "intern_35_vl",
    "videollama3",
    "videosalmonn_2",
    "videosalmonn_2plus",
]

MODEL_DISPLAY = {
    "gpt5_Azure":          "GPT-5 (Azure)",
    "claude_sonnet_46":    "Claude Sonnet 4.6",
    "claude_opus_46":      "Claude Opus 4.6",
    "qwen3_vl":            "Qwen3-VL",
    "qwen3_vl_ft":         "Qwen3-VL (ft)",
    "qwen32_vl":           "Qwen3.2-VL",
    "qwen2_5_vl":          "Qwen2.5-VL",
    "intern_3_vl":         "InternVL-3",
    "intern_35_vl":        "InternVL-3.5",
    "videollama3":         "VideoLLaMA-3",
    "videosalmonn_2":      "VideoSALMONN-2",
    "videosalmonn_2plus":  "VideoSALMONN-2+",
}

PIPELINE_MODES = ["audio", "visual", "time", "priority"]
PIPELINE_DISPLAY = {
    "audio":    "Audio",
    "visual":   "Visual",
    "time":     "Temporal",
    "priority": "Priority",
}

COGNITIVE_CATEGORIES = [
    "Transcript Comprehension",
    "Symbols in Videos",
    "Spatial",
    "Physical Action",
    "Fine-Grained Inspection",
]

# SOF subfolder name → pipeline_mode value in the query file
SOF_DIR_MAP = {
    "sof_aud":          "audio",
    "sof_audio":        "audio",
    "sof_vid":          "visual",
    "sof_visual":       "visual",
    "sof_rec_time":     "time",
    "sof_time":         "time",
    "sof_rec_priority": "priority",
    "sof_priority":     "priority",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_query_index(query_file):
    """Build query_id → {pipeline_mode, cognitive_category} lookup from the
    consolidated query file.  Also builds a fallback index keyed on
    (question_text, ground_truth) for older results that lack query_id."""
    with open(query_file) as f:
        data = json.load(f)
    by_id = {}
    by_text = {}  # (question, ground_truth) → annotations
    for video_key, queries in data.items():
        for q in queries:
            meta = {
                "pipeline_mode": q["annotations"]["pipeline_mode"],
                "cognitive_category": q["annotations"]["cognitive_category"],
            }
            by_id[q["query_id"]] = meta
            by_text[(q["question"].strip(), q["ground_truth"].strip())] = meta
    return by_id, by_text


def _collect_model_results(results_dir, experiment="exp_v3"):
    """Return {model_name: list_of_result_dicts} for the given experiment."""
    model_results = {}
    if not os.path.isdir(results_dir):
        return model_results

    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        exp_path = os.path.join(model_path, experiment)
        if not os.path.isdir(exp_path):
            # Maybe results sit directly in model_path (flat layout)
            exp_path = model_path
            if not any(f.endswith("_results.json") for f in os.listdir(exp_path) if os.path.isfile(os.path.join(exp_path, f))):
                continue

        records = []
        for json_file in glob.glob(os.path.join(exp_path, "**", "*_results.json"), recursive=True):
            # skip parity folders
            if "parity" in json_file:
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
            except (json.JSONDecodeError, IOError):
                continue

        if records:
            model_results[model_name] = records
    return model_results


def _lookup(r, query_index):
    """Resolve annotations for a result record using id or text fallback."""
    by_id, by_text = query_index
    qid = r.get("query_id", "")
    if qid and qid in by_id:
        return by_id[qid]
    key = (r.get("question", "").strip(), r.get("ground_truth", "").strip())
    return by_text.get(key, {})


def _compute_accuracy(records, query_index, group_key):
    """Compute accuracy grouped by `group_key` (pipeline_mode | cognitive_category).
    Returns {group_value: (correct, total)}."""
    buckets = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in records:
        verdict = r.get("judge_verdict")
        if verdict is None:
            continue
        meta = _lookup(r, query_index)
        if group_key == "pipeline_mode":
            group_val = r.get("source_of_fact") or meta.get("pipeline_mode")
        else:
            group_val = r.get("category") or meta.get("cognitive_category")
        if group_val is None:
            continue
        buckets[group_val][1] += 1
        if verdict:
            buckets[group_val][0] += 1
    return dict(buckets)


def _overall_accuracy(records):
    """Return (correct, total) for judged records."""
    correct = sum(1 for r in records if r.get("judge_verdict") is True)
    total = sum(1 for r in records if r.get("judge_verdict") is not None)
    return correct, total


def _pct(correct, total):
    if total == 0:
        return "—"
    return f"{correct / total * 100:.1f}"


# ── Markdown generation ─────────────────────────────────────────────────────

def _md_table(headers, rows):
    """Return a markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_leaderboard(results_dir, query_file, output_path, experiment="exp_v3"):
    query_index = _load_query_index(query_file)  # (by_id, by_text)
    by_id, by_text = query_index
    model_results = _collect_model_results(results_dir, experiment)

    lines = []
    lines.append("# 🏆 Pupil Leaderboard")
    lines.append("")
    lines.append(f"> Auto-generated on **{datetime.now().strftime('%Y-%m-%d %H:%M')}**  ")
    lines.append(f"> Experiment: `{experiment}` &nbsp;|&nbsp; Judged queries from `final_1k` ({len(by_id)} total)")
    lines.append("")

    # ── Table 1: Overall Accuracy ────────────────────────────────────────
    lines.append("## 📊 Overall Accuracy")
    lines.append("")
    headers = ["#", "Model", "Correct", "Total", "Accuracy (%)"]
    rows = []
    rank = 1
    scored = []
    for m in ALL_MODELS:
        display = MODEL_DISPLAY.get(m, m)
        if m in model_results:
            c, t = _overall_accuracy(model_results[m])
            scored.append((m, display, c, t))
        else:
            scored.append((m, display, None, None))

    # sort: models with results first (by accuracy desc), then pending
    has_results = [(m, d, c, t) for m, d, c, t in scored if t is not None and t > 0]
    pending = [(m, d, c, t) for m, d, c, t in scored if t is None or t == 0]
    has_results.sort(key=lambda x: x[2] / x[3], reverse=True)

    for m, display, c, t in has_results:
        rows.append([rank, f"**{display}**", c, t, f"**{_pct(c, t)}**"])
        rank += 1
    for m, display, c, t in pending:
        rows.append([rank, display, "—", "—", "—"])
        rank += 1

    lines.append(_md_table(headers, rows))
    lines.append("")

    # ── Table 2: Accuracy by Pipeline Mode (Source of Fact) ──────────────
    lines.append("## 🔬 Accuracy by Pipeline Mode (Source of Fact)")
    lines.append("")
    pm_headers = ["Model"] + [PIPELINE_DISPLAY[p] for p in PIPELINE_MODES]
    pm_rows = []
    ordered_models = [x[0] for x in has_results] + [x[0] for x in pending]
    for m in ordered_models:
        display = MODEL_DISPLAY.get(m, m)
        if m not in model_results:
            pm_rows.append([display] + ["—"] * len(PIPELINE_MODES))
            continue
        buckets = _compute_accuracy(model_results[m], query_index, "pipeline_mode")
        row = [f"**{display}**"]
        for p in PIPELINE_MODES:
            if p in buckets:
                c, t = buckets[p]
                row.append(f"{_pct(c, t)} ({c}/{t})")
            else:
                row.append("—")
        pm_rows.append(row)

    lines.append(_md_table(pm_headers, pm_rows))
    lines.append("")

    # ── Table 3: Accuracy by Cognitive Category ──────────────────────────
    lines.append("## 🧠 Accuracy by Cognitive Category")
    lines.append("")
    cc_headers = ["Model"] + COGNITIVE_CATEGORIES
    cc_rows = []
    for m in ordered_models:
        display = MODEL_DISPLAY.get(m, m)
        if m not in model_results:
            cc_rows.append([display] + ["—"] * len(COGNITIVE_CATEGORIES))
            continue
        buckets = _compute_accuracy(model_results[m], query_index, "cognitive_category")
        row = [f"**{display}**"]
        for cat in COGNITIVE_CATEGORIES:
            if cat in buckets:
                c, t = buckets[cat]
                row.append(f"{_pct(c, t)} ({c}/{t})")
            else:
                row.append("—")
        cc_rows.append(row)

    lines.append(_md_table(cc_headers, cc_rows))
    lines.append("")

    # ── Table 4: Pipeline Mode × Cognitive Category (best model highlight) ─
    lines.append("## 🗺️ Best Model per Pipeline Mode × Cognitive Category")
    lines.append("")
    cross_headers = ["Pipeline \\ Category"] + COGNITIVE_CATEGORIES
    cross_rows = []
    for p in PIPELINE_MODES:
        row = [f"**{PIPELINE_DISPLAY[p]}**"]
        for cat in COGNITIVE_CATEGORIES:
            best_name, best_acc = None, -1
            for m in ordered_models:
                if m not in model_results:
                    continue
                c_cnt, t_cnt = 0, 0
                for r in model_results[m]:
                    v = r.get("judge_verdict")
                    if v is None:
                        continue
                    meta = _lookup(r, query_index)
                    rp = r.get("source_of_fact") or meta.get("pipeline_mode")
                    rc = r.get("category") or meta.get("cognitive_category")
                    if rp == p and rc == cat:
                        t_cnt += 1
                        if v:
                            c_cnt += 1
                if t_cnt > 0:
                    acc = c_cnt / t_cnt
                    if acc > best_acc:
                        best_acc = acc
                        best_name = MODEL_DISPLAY.get(m, m)
            if best_name:
                row.append(f"{best_name} ({best_acc*100:.0f}%)")
            else:
                row.append("—")
        cross_rows.append(row)

    lines.append(_md_table(cross_headers, cross_rows))
    lines.append("")

    # ── Write ────────────────────────────────────────────────────────────
    md_content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(md_content)
    print(f"✅ Leaderboard written to {output_path}")
    print(f"   Models with results: {len(model_results)}/{len(ALL_MODELS)}")
    print(f"   Models pending (shown as '—'): {len(ALL_MODELS) - len(model_results)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pupil leaderboard")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                        help="Path to results directory containing model folders")
    parser.add_argument("--query_file", default=DEFAULT_QUERY_FILE,
                        help="Path to consolidated query JSON with annotations")
    parser.add_argument("--output", default=OUTPUT_MD,
                        help="Output markdown file path")
    parser.add_argument("--experiment", default="exp_v3",
                        help="Experiment subfolder to look for inside each model dir")
    args = parser.parse_args()
    generate_leaderboard(args.results_dir, args.query_file, args.output, args.experiment)
