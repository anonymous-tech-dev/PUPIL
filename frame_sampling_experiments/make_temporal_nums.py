#!/usr/bin/env python3
"""
make_temporal_nums.py — generate /workspace/Pupil/temporal_nums.txt

Single-source-of-truth dump of every frame-sampling experiment we report
(or could report) for the temporal-test-time-scaling section of the paper.
Numbers are recomputed from the raw result jsonl/json files under
``frame_sampling_experiments/`` and ``mllm_evaluation/results_v2/`` so that
the output is always consistent with the data on disk.

Layout of the txt:
    PART 1 — Paper-relevant tables (Native / TCoT / SD-CoT on LVBench v2 +
             PUPIL final_1k, overall + per-axis breakdowns).
    PART 2 — Appendix dump: LVBench leaderboard (all edu_cot variants),
             TCoT k-sweep, selection-method coverage collapse, C-bias and
             "how many" diagnostics, PUPIL SFT comparison.

Usage:
    cd /workspace/Pupil
    python frame_sampling_experiments/make_temporal_nums.py
        # writes ./temporal_nums.txt
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSE  = os.path.join(ROOT, "frame_sampling_experiments")

# LVBench v2 (Qwen3-VL-8B, MCQ)
LVB_NATIVE_F = os.path.join(
    FSE, "temporal_cot_gdm/results/lvbench_v2",
    "Qwen3-VL-8B_baseline_native_v6_l12_s64_k256_u0_results_v7.jsonl",
)
LVB_TCOT_F = os.path.join(
    FSE, "temporal_cot_gdm/results/lvbench_v2",
    "Qwen3-VL-8B_dynamic_segment_l12_s64_k256_u32_results_v7.jsonl",
)
LVB_SDCOT_F = os.path.join(
    FSE, "edu_cot/results/lvbench_v2",
    "Qwen3-VL-8B_seg_scene_detect_nosel_k768_u96_results.jsonl",
)

# PUPIL final_1k (Qwen3-VL-8B, judge-verdict)
PUPIL_NATIVE_DIR = os.path.join(
    ROOT, "mllm_evaluation/results_v2/qwen3_vl/final_1k_benchmark",
)
PUPIL_TCOT_F = os.path.join(
    FSE, "tcot_Pupil/results/Pupil/judge/base_tcot",
    "Qwen3-VL-8B_dynamic_segment_l12_s128_k512_u128_results.json",
)
PUPIL_SDCOT_F = os.path.join(
    FSE, "edu_cot_Pupil/results/Pupil/judge/base_educot",
    "Qwen3-VL-8B_seg_scene_detect_nosel_k768_u96_openended_results.json",
)
PUPIL_SDCOT_SFT_F = os.path.join(
    FSE,
    "edu_cot_Pupil/results/Pupil/judge/sft_educot/"
    "sft_clip_32fr_fps2_lr2e-5_ep3",
    "Qwen3-VL-8B_ft_sft_clip_32fr_fps2_lr2e-5_ep3_seg_scene_detect_"
    "nosel_k768_u96_openended_results.json",
)
PUPIL_TCOT_SFT_F = os.path.join(
    FSE,
    "tcot_Pupil/results/Pupil/judge/sft_tcot/"
    "sft_clip_32fr_fps2_lr2e-5_ep3",
    "Qwen3-VL-8B_ft_sft_clip_32fr_fps2_lr2e-5_ep3_dynamic_segment_"
    "l12_s128_k512_u128_results.json",
)

# Appendix sweeps
EDUCOT_LVB_DIR = os.path.join(FSE, "edu_cot/results/lvbench_v2")
TCOT_LVB_DIR   = os.path.join(FSE, "temporal_cot_gdm/results/lvbench_v2")
TCOT_LVB_OLD   = os.path.join(FSE, "temporal_cot_gdm/results/lvbench_v2/old")

OUT_TXT = os.path.join(ROOT, "temporal_nums.txt")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_json(path: str) -> List[dict]:
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    return [data]


def _load_pupil_dir(dir_path: str) -> List[dict]:
    """mllm_evaluation results_v2/<model>/<exp>/<video>_results.json layout."""
    recs: List[dict] = []
    for fn in os.listdir(dir_path):
        if not fn.endswith("_results.json"):
            continue
        recs.extend(_load_json(os.path.join(dir_path, fn)))
    # collapse on (query_id, question) — matches print_leaderboard.py
    seen = {}
    for r in recs:
        qid = r.get("query_id") or ""
        q = (r.get("question") or "").strip()
        seen[(qid, q) if qid or q else id(r)] = r
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Accuracy helpers
# --------------------------------------------------------------------------- #
def _is_correct_mcq(rec: dict) -> bool:
    return rec.get("predicted_letter") == rec.get("ground_truth")


def _is_correct_judge(rec: dict) -> bool | None:
    v = rec.get("judge_verdict")
    if v is None:
        return None
    return bool(v)


def _accuracy(recs: Sequence[dict], judge: bool):
    """Return (correct, judged, total)."""
    if judge:
        correct = sum(1 for r in recs if _is_correct_judge(r) is True)
        judged  = sum(1 for r in recs if _is_correct_judge(r) is not None)
        return correct, judged, len(recs)
    correct = sum(1 for r in recs if _is_correct_mcq(r))
    return correct, len(recs), len(recs)


def _bucket(recs: Sequence[dict], key_fn, judge: bool):
    """key_fn(rec) -> str | iterable[str]; multi-label tolerated."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, n]
    for r in recs:
        if judge:
            ok = _is_correct_judge(r)
            if ok is None:
                continue
        else:
            ok = _is_correct_mcq(r)
        keys = key_fn(r)
        if keys is None:
            keys = ["unknown"]
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            buckets[k][1] += 1
            if ok:
                buckets[k][0] += 1
    return dict(buckets)


# --------------------------------------------------------------------------- #
# Pretty-print helpers
# --------------------------------------------------------------------------- #
def _hr(ch: str = "=", n: int = 100) -> str:
    return ch * n


def _section(title: str, ch: str = "=") -> str:
    return f"{_hr(ch)}\n{title}\n{_hr(ch)}"


def _fmt_pct(c: int, n: int) -> str:
    if n == 0:
        return "  --   "
    return f"{100 * c / n:6.2f}"


def _fmt_signed(d: float) -> str:
    return f"{d:+6.2f}"


def _fmt_table(headers: List[str], rows: List[List[str]],
               aligns: List[str] | None = None) -> str:
    if not rows:
        return "  (no rows)\n"
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    def _fmt_cell(v, w, a):
        return f"{v:<{w}}" if a == "l" else f"{v:>{w}}"
    lines = []
    lines.append("  " + "  ".join(_fmt_cell(h, w, a)
                                  for h, w, a in zip(headers, widths, aligns)))
    lines.append("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  " + "  ".join(_fmt_cell(v, w, a)
                                      for v, w, a in zip(row, widths, aligns)))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# PART 1 — paper-relevant tables
# --------------------------------------------------------------------------- #
def part1_lvbench(out: list, native, tcot, sdcot):
    out.append(_section("[1.1] LVBench v2 — Overall accuracy (n=825, MCQ)", "-"))
    rows = []
    for tag, recs in [("Native",        native),
                      ("TCoT  k=256",  tcot),
                      ("SD-CoT k=768", sdcot)]:
        c, j, n = _accuracy(recs, judge=False)
        rows.append([tag, str(n), str(c), _fmt_pct(c, n)])
    base_acc = 100 * sum(_is_correct_mcq(r) for r in native) / len(native)
    tcot_acc = 100 * sum(_is_correct_mcq(r) for r in tcot)   / len(tcot)
    sdcot_acc= 100 * sum(_is_correct_mcq(r) for r in sdcot)  / len(sdcot)
    rows[0].extend(["  --  ", "  --  "])
    rows[1].extend([_fmt_signed(tcot_acc - base_acc), "  --  "])
    rows[2].extend([_fmt_signed(sdcot_acc - base_acc),
                    _fmt_signed(sdcot_acc - tcot_acc)])
    out.append(_fmt_table(
        ["Regime", "n", "correct", "acc(%)", "Δ_Native", "Δ_TCoT"],
        rows))

    out.append(_section("[1.2] LVBench v2 — Per question_type "
                       "(primary label only, matches paper spider plot)",
                       "-"))
    def qkey(r):
        qts = r.get("question_type") or []
        if isinstance(qts, str):
            return [qts]
        return [qts[0]] if qts else None
    nb = _bucket(native, qkey, judge=False)
    tb = _bucket(tcot,   qkey, judge=False)
    sb = _bucket(sdcot,  qkey, judge=False)
    qts_order = ["entity recognition", "event understanding",
                 "key information retrieval", "reasoning",
                 "summarization", "temporal grounding"]
    rows = []
    for qt in qts_order:
        if qt not in nb:
            continue
        n_n, n_t, n_s = nb[qt][1], tb[qt][1], sb[qt][1]
        a_n = 100 * nb[qt][0] / n_n
        a_t = 100 * tb[qt][0] / n_t
        a_s = 100 * sb[qt][0] / n_s
        rows.append([qt, str(n_n),
                     f"{a_n:6.2f}", f"{a_t:6.2f}", f"{a_s:6.2f}",
                     _fmt_signed(a_s - a_n), _fmt_signed(a_t - a_n)])
    out.append(_fmt_table(
        ["question_type", "n_qt", "Native", "TCoT", "SD-CoT",
         "ΔSDCoT-Nat", "ΔTCoT-Nat"],
        rows))


def part1_pupil(out: list, native, tcot, sdcot):
    out.append(_section("[1.3] PUPIL final_1k — Overall accuracy "
                       "(judge_verdict == True)", "-"))
    rows = []
    for tag, recs in [("Native",        native),
                      ("TCoT  k=512",  tcot),
                      ("SD-CoT k=768", sdcot)]:
        c, j, n = _accuracy(recs, judge=True)
        rows.append([tag, str(n), str(j), str(c), _fmt_pct(c, j)])
    nb_acc = 100 * sum(1 for r in native if _is_correct_judge(r) is True) / \
             max(sum(1 for r in native if _is_correct_judge(r) is not None), 1)
    tb_acc = 100 * sum(1 for r in tcot   if _is_correct_judge(r) is True) / \
             max(sum(1 for r in tcot   if _is_correct_judge(r) is not None), 1)
    sb_acc = 100 * sum(1 for r in sdcot  if _is_correct_judge(r) is True) / \
             max(sum(1 for r in sdcot  if _is_correct_judge(r) is not None), 1)
    rows[0].extend(["  --  ", "  --  "])
    rows[1].extend([_fmt_signed(tb_acc - nb_acc), "  --  "])
    rows[2].extend([_fmt_signed(sb_acc - nb_acc),
                    _fmt_signed(sb_acc - tb_acc)])
    out.append(_fmt_table(
        ["Regime", "n_total", "n_judged", "correct", "acc(%)",
         "Δ_Native", "Δ_TCoT"],
        rows))

    # axes
    sof_order = ["audio", "visual", "priority", "time"]
    cat_order = ["Symbols in Videos", "Spatial", "Transcript Comprehension",
                 "Physical Action", "Fine-Grained Inspection"]
    for tag, key_fn, order in [
        ("[1.4] PUPIL — Per source_of_fact (Source-of-Fact axes)",
         lambda r: r.get("source_of_fact"), sof_order),
        ("[1.5] PUPIL — Per category (Pedagogical categories)",
         lambda r: r.get("category"), cat_order),
    ]:
        out.append(_section(tag, "-"))
        nb = _bucket(native, key_fn, judge=True)
        tb = _bucket(tcot,   key_fn, judge=True)
        sb = _bucket(sdcot,  key_fn, judge=True)
        rows = []
        for k in order:
            if k not in nb:
                continue
            a_n = 100 * nb[k][0] / nb[k][1]
            a_t = 100 * tb[k][0] / tb[k][1] if k in tb and tb[k][1] else 0.0
            a_s = 100 * sb[k][0] / sb[k][1] if k in sb and sb[k][1] else 0.0
            rows.append([k, str(nb[k][1]),
                         f"{a_n:6.2f}", f"{a_t:6.2f}", f"{a_s:6.2f}",
                         _fmt_signed(a_s - a_n), _fmt_signed(a_t - a_n)])
        out.append(_fmt_table(
            [tag.split(" — ")[1].split(" (")[0],
             "n", "Native", "TCoT", "SD-CoT", "ΔSDCoT-Nat", "ΔTCoT-Nat"],
            rows))


# --------------------------------------------------------------------------- #
# PART 2 — appendix
# --------------------------------------------------------------------------- #
def _parse_edu_cot_tag(fn: str) -> str:
    """
    Strip ``Qwen3-VL-8B_`` prefix and ``_results.jsonl`` suffix from an
    edu_cot LVBench result filename. Mirrors the variant tags emitted by
    ``utils/results_io.py::build_variant_tag``.
    """
    base = os.path.basename(fn)
    base = re.sub(r"^Qwen3-VL-8B_", "", base)
    base = re.sub(r"_results\.jsonl$", "", base)
    return base


def part2_lvbench_leaderboard(out: list):
    out.append(_section(
        "[2.1] LVBench v2 — Full edu_cot leaderboard "
        "(scene_detect / uniform / kf / sel / prompt / etc., n=825 marked)",
        "-",
    ))
    rows = []
    for fn in sorted(os.listdir(EDUCOT_LVB_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        recs = _load_jsonl(os.path.join(EDUCOT_LVB_DIR, fn))
        if not recs:
            continue
        c, _, n = _accuracy(recs, judge=False)
        rows.append((100 * c / n, n, c, _parse_edu_cot_tag(fn)))
    rows.sort(key=lambda x: -x[0])
    table = []
    for acc, n, c, tag in rows:
        marker = "★" if n >= 825 else "·"
        table.append([f"{acc:6.2f}", str(n), str(c), marker, tag])
    out.append(_fmt_table(
        ["acc(%)", "n", "correct", "full?", "variant_tag"],
        table,
    ))
    out.append("    full? = ★ : run covers all 825 LVBench v2 questions; "
               "· : partial / killed run (compare with care)\n")


def part2_tcot_ksweep(out: list):
    out.append(_section(
        "[2.2] LVBench v2 — TCoT-family k-sweep on Qwen3-VL-8B "
        "(temporal_cot_gdm)", "-"))
    rows = []
    for sub_dir in [TCOT_LVB_DIR, TCOT_LVB_OLD]:
        if not os.path.isdir(sub_dir):
            continue
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.startswith("Qwen3-VL-8B") or not fn.endswith(".jsonl"):
                continue
            fp = os.path.join(sub_dir, fn)
            recs = _load_jsonl(fp)
            if not recs:
                continue
            c, _, n = _accuracy(recs, judge=False)
            rel = "(latest)" if sub_dir == TCOT_LVB_DIR else "(old)"
            rows.append((100 * c / n, n, c, rel, fn))
    rows.sort(key=lambda x: -x[0])
    table = [[f"{acc:6.2f}", str(n), str(c), rel, fn]
             for acc, n, c, rel, fn in rows]
    out.append(_fmt_table(
        ["acc(%)", "n", "correct", "vintage", "filename"], table))


def part2_coverage(out: list):
    out.append(_section(
        "[2.3] LVBench v2 — Selection-method coverage collapse "
        "(median num_context shown to answering VLM)", "-"))
    rows = []
    for fn in sorted(os.listdir(EDUCOT_LVB_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        recs = _load_jsonl(os.path.join(EDUCOT_LVB_DIR, fn))
        if len(recs) < 50:  # skip tiny/partial
            continue
        nc = [r.get("num_context") or 0 for r in recs]
        nc = [x for x in nc if x > 0]
        if not nc:
            continue
        c, _, n = _accuracy(recs, judge=False)
        rows.append((statistics.median(nc), n, 100 * c / n,
                     _parse_edu_cot_tag(fn)))
    rows.sort(key=lambda x: -x[0])
    table = [[f"{int(med):6d}", str(n), f"{acc:6.2f}", tag]
             for med, n, acc, tag in rows]
    out.append(_fmt_table(
        ["med(num_ctx)", "n", "acc(%)", "variant_tag"], table))


def part2_cbias_and_counting(out: list):
    out.append(_section(
        "[2.4] LVBench v2 — C-bias diagnostic "
        "(pred=C rate − GT=C rate, full-n edu_cot runs only)", "-"))
    rows = []
    for fn in sorted(os.listdir(EDUCOT_LVB_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        recs = _load_jsonl(os.path.join(EDUCOT_LVB_DIR, fn))
        if len(recs) != 825:
            continue
        pc = sum(1 for r in recs if r.get("predicted_letter") == "C")
        gc = sum(1 for r in recs if r.get("ground_truth") == "C")
        n = len(recs)
        rows.append((100 * pc / n - 100 * gc / n,
                     100 * pc / n, 100 * gc / n,
                     _parse_edu_cot_tag(fn)))
    rows.sort(key=lambda x: -x[0])
    table = [[_fmt_signed(d), f"{p:6.2f}", f"{g:6.2f}", tag]
             for d, p, g, tag in rows]
    out.append(_fmt_table(
        ["bias(pp)", "pred=C(%)", "GT=C(%)", "variant_tag"], table))

    out.append(_section(
        "[2.5] LVBench v2 — \"How many\" counting bucket "
        "(vision-side bottleneck; full-n edu_cot runs only)", "-"))
    pat = re.compile(r"\bhow many\b", re.I)
    rows = []
    for fn in sorted(os.listdir(EDUCOT_LVB_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        recs = _load_jsonl(os.path.join(EDUCOT_LVB_DIR, fn))
        if len(recs) != 825:
            continue
        sub = [r for r in recs if pat.search(r.get("question") or "")]
        if not sub:
            continue
        c = sum(1 for r in sub if _is_correct_mcq(r))
        rows.append((100 * c / len(sub), len(sub), c,
                     _parse_edu_cot_tag(fn)))
    rows.sort(key=lambda x: -x[0])
    table = [[f"{acc:6.2f}", str(n), str(c), tag]
             for acc, n, c, tag in rows]
    out.append(_fmt_table(
        ["acc(%)", "n_howmany", "correct", "variant_tag"], table))


def part2_pupil_sft(out: list, native, tcot, sdcot,
                    sdcot_sft, tcot_sft):
    out.append(_section(
        "[2.6] PUPIL final_1k — SFT-on-CLIP (Qwen3-VL-8B + LoRA) variants "
        "vs base", "-"))
    sof_order = ["audio", "visual", "priority", "time"]
    cat_order = ["Symbols in Videos", "Spatial", "Transcript Comprehension",
                 "Physical Action", "Fine-Grained Inspection"]
    runs = [
        ("Native (no localiser)", native),
        ("TCoT  k=512  (base)",   tcot),
        ("TCoT  k=512  + SFT",    tcot_sft),
        ("SD-CoT k=768 (base)",   sdcot),
        ("SD-CoT k=768 + SFT",    sdcot_sft),
    ]
    # overall
    rows = []
    for tag, recs in runs:
        c, j, n = _accuracy(recs, judge=True)
        rows.append([tag, str(n), str(j), str(c), _fmt_pct(c, j)])
    out.append(_fmt_table(
        ["Run", "n_total", "n_judged", "correct", "acc(%)"], rows))

    # SoF
    out.append("\n  Per source_of_fact (% judged correct):\n")
    rows = []
    for tag, recs in runs:
        b = _bucket(recs, lambda r: r.get("source_of_fact"), judge=True)
        rows.append([tag] + [
            f"{100*b[k][0]/b[k][1]:6.2f}" if k in b and b[k][1] else "  --  "
            for k in sof_order])
    out.append(_fmt_table(
        ["Run", *sof_order], rows,
        aligns=["l"] + ["r"] * len(sof_order)))

    # category
    out.append("\n  Per category (% judged correct):\n")
    short = {
        "Symbols in Videos":         "Symbols",
        "Spatial":                   "Spatial",
        "Transcript Comprehension":  "Transcript",
        "Physical Action":           "PhysAction",
        "Fine-Grained Inspection":   "FineGrained",
    }
    rows = []
    for tag, recs in runs:
        b = _bucket(recs, lambda r: r.get("category"), judge=True)
        rows.append([tag] + [
            f"{100*b[k][0]/b[k][1]:6.2f}" if k in b and b[k][1] else "  --  "
            for k in cat_order])
    out.append(_fmt_table(
        ["Run", *(short[c] for c in cat_order)], rows,
        aligns=["l"] + ["r"] * len(cat_order)))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    # Load everything once.
    lvb_native = _load_jsonl(LVB_NATIVE_F)
    lvb_tcot   = _load_jsonl(LVB_TCOT_F)
    lvb_sdcot  = _load_jsonl(LVB_SDCOT_F)

    pupil_native    = _load_pupil_dir(PUPIL_NATIVE_DIR)
    pupil_tcot      = _load_json(PUPIL_TCOT_F)
    pupil_sdcot     = _load_json(PUPIL_SDCOT_F)
    pupil_sdcot_sft = _load_json(PUPIL_SDCOT_SFT_F)
    pupil_tcot_sft  = _load_json(PUPIL_TCOT_SFT_F)

    out: list[str] = []

    # ------------------------------------------------------------------- #
    # Header
    # ------------------------------------------------------------------- #
    out.append(_hr("="))
    out.append("Pupil — Frame-Sampling Experiments — Numbers Dump")
    out.append("Auto-generated by frame_sampling_experiments/"
               "make_temporal_nums.py")
    out.append(f"Generated:    "
               f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    out.append("Backbone:     Qwen3-VL-8B-Instruct "
               "(no fine-tuning unless noted)")
    out.append("")
    out.append("Regimes")
    out.append("  Native      : baseline_native_v6 — 1 fps native sampler, "
               "max_pixels=151200, no localiser")
    out.append("  TCoT        : 1-pass per-segment VLM frame-selector "
               "(temporal Chain-of-Thought)")
    out.append("                  LVBench: l=12 s=64  k=256 u=32  "
               "(dynamic_segment_l12_s64_k256_u32_v7.jsonl)")
    out.append("                  PUPIL  : l=12 s=128 k=512 u=128 "
               "(dynamic_segment_l12_s128_k512_u128_results.json)")
    out.append("                  NB: PUPIL TCoT was sharded at k=512 instead "
               "of LVBench's k=256 — the paper")
    out.append("                      label \"k=256 transferred unchanged\" "
               "is approximate (longer-budget setting).")
    out.append("  SD-CoT      : PySceneDetect 320×180 @ 2 fps, MAD τ=50, "
               "segs ∈ [3,60] s, k=768, u=96, no second-pass selection")
    out.append("                  (edu_cot scene_detect_nosel_k768_u96)")
    out.append("")
    out.append("Datasets")
    out.append(f"  LVBench v2     : {len(lvb_native)} multiple-choice items "
               f"— predicted_letter == ground_truth")
    out.append(f"  PUPIL final_1k : {len(pupil_native)} open-ended items "
               f"judged by external LLM judge → judge_verdict == True")
    out.append("")

    # ------------------------------------------------------------------- #
    # PART 1
    # ------------------------------------------------------------------- #
    out.append(_section("PART 1 — PAPER-RELEVANT TABLES "
                       "(Native vs TCoT vs SD-CoT)", "="))
    out.append("")
    part1_lvbench(out, lvb_native, lvb_tcot, lvb_sdcot)
    out.append("")
    part1_pupil(out, pupil_native, pupil_tcot, pupil_sdcot)
    out.append("")

    # ------------------------------------------------------------------- #
    # PART 2
    # ------------------------------------------------------------------- #
    out.append(_section("PART 2 — APPENDIX DUMP "
                       "(everything else we ran on this regime)", "="))
    out.append("")
    part2_lvbench_leaderboard(out)
    out.append("")
    part2_tcot_ksweep(out)
    out.append("")
    part2_coverage(out)
    out.append("")
    part2_cbias_and_counting(out)
    out.append("")
    part2_pupil_sft(out, pupil_native, pupil_tcot, pupil_sdcot,
                    pupil_sdcot_sft, pupil_tcot_sft)
    out.append("")

    # ------------------------------------------------------------------- #
    # Footer
    # ------------------------------------------------------------------- #
    out.append(_hr("="))
    out.append("End of dump. Re-run `python frame_sampling_experiments/"
               "make_temporal_nums.py` to refresh.")
    out.append(_hr("="))

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(out))
    print(f"Wrote {OUT_TXT} ({sum(len(s) for s in out):,} chars).")


if __name__ == "__main__":
    main()
