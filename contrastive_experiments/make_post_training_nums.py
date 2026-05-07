#!/usr/bin/env python3
"""make_post_training_nums.py

Auto-generates /workspace/Pupil/post_training_nums.txt — a structured
text dump of all SFT / DPO / SFT-warmstart+DPO experiments and their data
counterparts on Pupil (PUPIL final_1k, judged by external LLM).

Companion to frame_sampling_experiments/make_temporal_nums.py. All numbers are
re-derived from the result jsons under mllm_evaluation/results_v2/<model>/<exp>/
and the dataset jsons under contrastive_experiments/. Run this whenever you
add a new run so the appendix dump in the paper stays in sync.

Usage:
    python contrastive_experiments/make_post_training_nums.py
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from collections import defaultdict
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "mllm_evaluation", "results_v2")
OUT_PATH = os.path.join(ROOT, "post_training_nums.txt")


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def _load_records(exp_dir: str) -> list[dict]:
    records: list[dict] = []
    for jf in glob.glob(os.path.join(exp_dir, "**", "*_results.json"), recursive=True):
        if "parity" in jf or "_shard" in jf:
            continue
        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
    seen: dict = {}
    for r in records:
        qid = r.get("query_id") or ""
        q = (r.get("question") or "").strip()
        seen[(qid, q) if qid or q else id(r)] = r
    return list(seen.values())


def _acc(recs: Iterable[dict]) -> tuple[int, int, int]:
    recs = list(recs)
    correct = sum(1 for r in recs if r.get("judge_verdict") is True)
    judged = sum(1 for r in recs if r.get("judge_verdict") is not None)
    return correct, judged, len(recs) - judged


def _pct(c: int, j: int) -> str:
    return f"{100*c/j:.2f}" if j else "  —  "


def _by(recs: Iterable[dict], key: str) -> dict:
    out: dict = defaultdict(lambda: [0, 0])
    for r in recs:
        v = r.get("judge_verdict")
        if v is None:
            continue
        k = r.get(key) or "unknown"
        out[k][1] += 1
        if v:
            out[k][0] += 1
    return dict(out)


def _row_runs() -> list[dict]:
    """Walk results_v2 and emit one row per (model, experiment)."""
    rows: list[dict] = []
    if not os.path.isdir(RESULTS_DIR):
        return rows
    for model in sorted(os.listdir(RESULTS_DIR)):
        mdir = os.path.join(RESULTS_DIR, model)
        if not os.path.isdir(mdir):
            continue
        for exp in sorted(os.listdir(mdir)):
            edir = os.path.join(mdir, exp)
            if not os.path.isdir(edir):
                continue
            recs = _load_records(edir)
            if not recs:
                continue
            c, j, p = _acc(recs)
            rows.append(dict(model=model, exp=exp, recs=recs, c=c, j=j, pending=p))
    return rows


def _find(rows, model: str, *needles: str):
    """Return first row matching model and ALL needles (substring) in exp name."""
    for r in rows:
        if r["model"] != model:
            continue
        if all(n in r["exp"] for n in needles):
            return r
    return None


# ----------------------------------------------------------------------------
# Dataset stat helpers
# ----------------------------------------------------------------------------
def _len_json(path: str) -> int | None:
    try:
        with open(path) as f:
            d = json.load(f)
        return len(d) if isinstance(d, list) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------------
# Pretty-print helpers
# ----------------------------------------------------------------------------
HR = "=" * 100
HR2 = "-" * 100


def _table(headers: list[str], rows: list[list[str]], gap: int = 2) -> str:
    cols = list(zip(*([headers] + rows))) if rows else [list(h) for h in headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    sep = " " * gap
    lines = [sep.join(str(h).ljust(w) for h, w in zip(headers, widths))]
    lines.append(sep.join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(str(v).ljust(w) for v, w in zip(row, widths)))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    rows = _row_runs()
    out: list[str] = []
    pr = out.append

    # ------- header -------
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pr(HR)
    pr("Pupil — Post-Training (SFT / DPO) Experiments — Numbers Dump")
    pr("Auto-generated by contrastive_experiments/make_post_training_nums.py")
    pr(f"Generated:    {now}")
    pr("Scoring:      PUPIL final_1k (n=974), judge_verdict == True (external LLM judge)")
    pr("Backbone:     Qwen3-VL-8B-Instruct (and Qwen3-VL-32B-Instruct where noted)")
    pr("")
    pr("Conventions")
    pr("  CLIP-judged   : SFT/DPO data filtered by per-clip judge (visual+audio+priority+time)")
    pr("  NOTX          : 'No-TX' — same training pipeline but with the merger-fix patch applied")
    pr("  matched       : eval-time settings frozen to match what the model was trained on")
    pr("                  (pixel budget + frame count) so SFT/DPO deltas are not confounded")
    pr("                  by sampler differences.")
    pr("  curriculum v2 : 2-stage SFT (warmstart → judged-clip fine-tune), then DPO on top")
    pr("")

    # ------- reference baselines -------
    ref = {}
    ref["8B native"] = _find(rows, "qwen3_vl", "final_1k_benchmark") and _find(rows, "qwen3_vl", "final_1k_benchmark")
    ref["8B native"] = next((r for r in rows if r["model"] == "qwen3_vl" and r["exp"] == "final_1k_benchmark"), None)
    ref["32B native"] = next((r for r in rows if r["model"] == "qwen32_vl"), None)
    ref["8B matched 32fr"] = _find(rows, "qwen3_vl_matched", "matched_32fr_fps2")
    # the above also matches 32B; refine:
    for r in rows:
        if r["model"] == "qwen3_vl_matched" and r["exp"] == "final_1k_benchmark_matched_32fr_fps2":
            ref["8B matched 32fr"] = r
        if r["model"] == "qwen3_vl_matched" and r["exp"] == "final_1k_benchmark_matched_32B_32fr_fps2":
            ref["32B matched 32fr"] = r
        if r["model"] == "qwen3_vl_matched" and "matched_256fr_NOTX" in r["exp"]:
            ref["8B matched 256fr"] = r
        if r["model"] == "qwen3_vl_matched" and "matched_32fr_sof_sft_step50_premortem" in r["exp"]:
            ref["8B matched 32fr (sft step50 premortem ref)"] = r

    pr(HR)
    pr("PART 0 — REFERENCE BASELINES (no post-training)")
    pr(HR)
    pr("")
    tbl_rows = []
    for label, r in ref.items():
        if not r:
            continue
        tbl_rows.append([label, str(len(r["recs"])), str(r["j"]), str(r["c"]),
                         _pct(r["c"], r["j"]), r["exp"]])
    pr(_table(["Reference baseline", "n_total", "n_judged", "correct", "acc(%)", "experiment_tag"], tbl_rows))
    pr("")

    # quick getter for native-8B baseline
    base8 = ref.get("8B native")
    base8_acc = (100 * base8["c"] / base8["j"]) if base8 and base8["j"] else None
    base32 = ref.get("32B native")
    base32_acc = (100 * base32["c"] / base32["j"]) if base32 and base32["j"] else None

    def _delta(r, ref_acc):
        if not r or not r["j"] or ref_acc is None:
            return "  —  "
        return f"{100*r['c']/r['j'] - ref_acc:+.2f}"

    # ------- PART 1 -------
    pr(HR)
    pr("PART 1 — PAPER-RELEVANT TABLES (SFT / DPO / Curriculum / Warmstart-then-DPO)")
    pr(HR)
    pr("")

    # ---- 1.1 SFT on PUPIL/CGBench-derived data ----
    pr(HR2)
    pr("[1.1] SFT — judged-clip data (Qwen3-VL-8B & 32B, single-stage)")
    pr(HR2)
    sft_specs = [
        # (label, model, needles, ref_acc)
        ("SFT 8B  CLIP-judged 32fr",                 "qwen3_vl_ft", ("ft_sof_sft_NOTX_32fr_CLIP",), base8_acc),
        ("SFT 8B  CLIP-judged 32fr basesettings",    "qwen3_vl_ft", ("ft_sof_sft_NOTX_32fr_basesettings",), base8_acc),
        ("SFT 8B  unjudged 32fr",                    "qwen3_vl_ft", ("ft_sof_sft_NOTX_32fr",), base8_acc),  # generic NOTX — last wins below
        ("SFT 8B  unjudged 32fr (matched MERGERFIX)","qwen3_vl_ft", ("sof_sft_NOTX_32fr_final_matched_MERGERFIX",), base8_acc),
        ("SFT 8B  CLIP-judged 32fr (no NOTX)",       "qwen3_vl_ft", ("ft_sft_clip_32fr_fps2_lr2e-5_ep3_eval32fr",), base8_acc),
        ("SFT 32B CLIP-judged 32fr  ep3",            "qwen3_vl_ft", ("ft_sft_clip_32B_32fr_fps2_lr2e-5_ep3_eval32fr",), base32_acc),
        ("SFT 32B CLIP-judged 32fr  ep5 ckpt120",    "qwen3_vl_ft", ("ft_sft_clip_32B_32fr_fps2_lr2e-5_ep5_ckpt120",), base32_acc),
        ("SFT 32B CLIP-judged 32fr  ep3 → 128fr eval","qwen3_vl_ft", ("ft_sft_clip_32B_32fr_fps2_lr2e-5_ep3_eval128fr_fps2_px131k",), base32_acc),
        ("SFT 32B mix80 32fr ep3",                    "qwen3_vl_ft", ("qwen32b_mix80","eval32fr_fps2"), base32_acc),
        ("SFT 32B mix80 32fr ep3 (eval px524k)",      "qwen3_vl_ft", ("qwen32b_mix80","eval32fr_fps2_px524k"), base32_acc),
    ]
    # de-dup the generic "SFT 8B unjudged 32fr": pick the exact exp ending in _32fr (not _CLIP/_basesettings/_final/_warmstart)
    seen_exps = set()
    sft_rows = []
    for label, model, needles, ref_acc in sft_specs:
        candidates = [r for r in rows if r["model"] == model and all(n in r["exp"] for n in needles)]
        # exclusion logic for the generic NOTX_32fr line
        if label == "SFT 8B  unjudged 32fr":
            candidates = [r for r in candidates if r["exp"] == "final_1k_benchmark_ft_sof_sft_NOTX_32fr"]
        if not candidates:
            continue
        r = candidates[0]
        if r["exp"] in seen_exps:
            continue
        seen_exps.add(r["exp"])
        sft_rows.append([label, str(len(r["recs"])), str(r["j"]), str(r["c"]),
                         _pct(r["c"], r["j"]), _delta(r, ref_acc), r["exp"][:60]])
    pr(_table(
        ["Run", "n_total", "n_judged", "correct", "acc(%)", "Δ_base", "experiment_tag (truncated)"],
        sft_rows))
    pr("  Δ_base is vs the matching native baseline (8B vs Qwen3-VL-8B, 32B vs Qwen3-VL-32B).")
    pr("")

    # ---- 1.2 DPO on top of base / SFT ----
    pr(HR2)
    pr("[1.2] DPO — preference fine-tune on judge-derived chosen/rejected pairs")
    pr(HR2)
    dpo_specs = [
        ("DPO 8B  from base 32fr (judged)",      "qwen3_vl_ft", ("sof_dpo_NOTX32fr_MERGERFIX_judged_beta0.1",), base8_acc),
        ("DPO 8B  from CLIP-SFT 32fr",           "qwen3_vl_ft", ("sof_dpo_NOTX32fr_CLIPSFT_judged_beta0.1",), base8_acc),
        ("DPO 8B  fullpairs from SFT-CLIP",      "qwen3_vl_ft", ("ft_dpo_from-sft-clip_fullpairs_32fr",), base8_acc),
        ("DPO 32B from base",                    "qwen3_vl_ft", ("dpo-from-base_32fr_fps2_b0.1",), base32_acc),
        ("DPO 32B from CLIP-SFT-3 (mix80 ckpt)", "qwen3_vl_ft", ("dpo-from-clip3_32fr_fps2_b0.1","20260503_043105"), base32_acc),
        ("DPO 32B from CLIP-SFT-3 off-policy balanced",
                                                  "qwen3_vl_ft", ("dpo-from-clip3_offpol-balanced",), base32_acc),
    ]
    dpo_rows = []
    for label, model, needles, ref_acc in dpo_specs:
        cands = [r for r in rows if r["model"] == model and all(n in r["exp"] for n in needles)]
        if not cands:
            continue
        r = cands[0]
        dpo_rows.append([label, str(len(r["recs"])), str(r["j"]), str(r["c"]),
                         _pct(r["c"], r["j"]), _delta(r, ref_acc), r["exp"][:60]])
    pr(_table(
        ["Run", "n_total", "n_judged", "correct", "acc(%)", "Δ_base", "experiment_tag (truncated)"],
        dpo_rows))
    pr("")

    # ---- 1.3 256fr SFT-warmstart → DPO ----
    pr(HR2)
    pr("[1.3] SFT-warmstart (256fr) and warmstart→DPO chain")
    pr(HR2)
    ws_specs = [
        ("8B SFT-warmstart 256fr (ckpt102)",                  "qwen3_vl_ft", ("sof_sft_warmstart_NOTX_lr2e-5_ep3_bs64_24576seq_256fr__step102",), base8_acc),
        ("8B 256fr-matched eval (warmstart base)",            "qwen3_vl_matched", ("matched_256fr_NOTX_lr2e-5_ep3_bs64_24576seq",), base8_acc),
        ("8B SFT-warmstart 256fr → DPO 256fr",                "qwen3_vl_ft", ("matched_256fr","sof_dpo_NOTX256fr_MERGERFIX_judged"), base8_acc),
        ("8B SFT-warmstart 256fr → DPO 32fr (down-projected)","qwen3_vl_ft", ("matched_256fr","sof_dpo_NOTX32fr_MERGERFIXv2_judged"), base8_acc),
    ]
    ws_rows = []
    for label, model, needles, ref_acc in ws_specs:
        cands = [r for r in rows if r["model"] == model and all(n in r["exp"] for n in needles)]
        if not cands:
            continue
        r = cands[0]
        ws_rows.append([label, str(len(r["recs"])), str(r["j"]), str(r["c"]),
                        _pct(r["c"], r["j"]), _delta(r, ref_acc), r["exp"][:65]])
    pr(_table(
        ["Run", "n_total", "n_judged", "correct", "acc(%)", "Δ_base", "experiment_tag (truncated)"],
        ws_rows))
    pr("  Note: 256fr eval costs ~8x more frames; ablation is whether warmstart→DPO recovers")
    pr("        what the 32fr CLIP-judged DPO already achieves.")
    pr("")

    # ---- 1.4 Curriculum v2 (warmstart → judged-clip → DPO) ----
    pr(HR2)
    pr("[1.4] Curriculum v2 — warmstart → judged-clip SFT → DPO  (strongest pipeline)")
    pr(HR2)
    cur_specs = [
        ("8B  curriculum SFT  (px524k)",        "qwen3_vl_ft", ("sof_sft_v2_curriculum_8b_NOTX","px524k"), base8_acc),
        ("8B  curriculum SFT  (px786k eval)",   "qwen3_vl_ft", ("v2_sft_curriculum_8b_NOTX","px786k_evalmatched_OLD"), base8_acc),
        ("8B  curriculum SFT → DPO (random_mix80judged)",
                                                "qwen3_vl_ft", ("sof_dpo_v2_run1_random_mix80judged_8b","onCurrSFTckpt84"), base8_acc),
        ("32B curriculum SFT  (clip-judged)",   "qwen3_vl_ft", ("sof_sft_v2_curriculum_clipjudged_32B","sft20260505_045744"), base32_acc),
    ]
    cur_rows = []
    for label, model, needles, ref_acc in cur_specs:
        cands = [r for r in rows if r["model"] == model and all(n in r["exp"] for n in needles)]
        if not cands:
            continue
        r = cands[0]
        cur_rows.append([label, str(len(r["recs"])), str(r["j"]), str(r["c"]),
                         _pct(r["c"], r["j"]), _delta(r, ref_acc), r["exp"][:65]])
    pr(_table(
        ["Run", "n_total", "n_judged", "correct", "acc(%)", "Δ_base", "experiment_tag (truncated)"],
        cur_rows))
    pr("")

    # ---- 1.5 Per-axis breakdowns for the headline runs ----
    pr(HR2)
    pr("[1.5] Per source_of_fact / category breakdowns — headline runs vs 8B native")
    pr(HR2)
    headline_keys = [
        ("8B native",                  base8),
        ("SFT 8B CLIP",                next((r for r in rows if r["model"]=="qwen3_vl_ft" and r["exp"]=="final_1k_benchmark_ft_sof_sft_NOTX_32fr_CLIP"), None)),
        ("SFT 8B curr (px524k)",       next((r for r in rows if r["model"]=="qwen3_vl_ft" and "sof_sft_v2_curriculum_8b_NOTX" in r["exp"] and "px524k" in r["exp"]), None)),
        ("DPO 8B from CLIP-SFT",       next((r for r in rows if r["model"]=="qwen3_vl_ft" and "sof_dpo_NOTX32fr_CLIPSFT" in r["exp"]), None)),
        ("DPO 8B v2 on currSFT ckpt84",next((r for r in rows if r["model"]=="qwen3_vl_ft" and "sof_dpo_v2_run1_random_mix80judged_8b" in r["exp"]), None)),
        ("SFT 32B CLIP ep3",           next((r for r in rows if r["model"]=="qwen3_vl_ft" and "ft_sft_clip_32B_32fr_fps2_lr2e-5_ep3_eval32fr" in r["exp"]), None)),
        ("SFT 32B curr (clip-judged)", next((r for r in rows if r["model"]=="qwen3_vl_ft" and "sof_sft_v2_curriculum_clipjudged_32B" in r["exp"]), None)),
        ("DPO 32B from CLIP-SFT-3",    next((r for r in rows if r["model"]=="qwen3_vl_ft" and "dpo-from-clip3_32fr_fps2_b0.1" in r["exp"] and "20260503_043105" in r["exp"]), None)),
    ]
    sof_keys = ["audio", "visual", "priority", "time"]
    pr("Per source_of_fact (% judged correct):")
    pr("")
    sof_rows = []
    for label, r in headline_keys:
        if not r:
            continue
        d = _by(r["recs"], "source_of_fact")
        cells = [label]
        for k in sof_keys:
            if k in d and d[k][1]:
                cc, tt = d[k]
                cells.append(f"{100*cc/tt:5.2f}")
            else:
                cells.append("  —  ")
        sof_rows.append(cells)
    pr(_table(["Run"] + sof_keys, sof_rows))
    pr("")

    cat_keys_pref = [
        "Symbols in Videos", "Spatial",
        "Transcript Comprehension", "Physical Action", "Fine-Grained Inspection",
    ]
    short = {"Symbols in Videos":"Symbols","Spatial":"Spatial",
             "Transcript Comprehension":"Transcript","Physical Action":"PhysAction",
             "Fine-Grained Inspection":"FineGrain"}
    pr("Per category (% judged correct):")
    pr("")
    cat_rows = []
    for label, r in headline_keys:
        if not r:
            continue
        d = _by(r["recs"], "category")
        cells = [label]
        for k in cat_keys_pref:
            if k in d and d[k][1]:
                cc, tt = d[k]
                cells.append(f"{100*cc/tt:5.2f}")
            else:
                cells.append("  —  ")
        cat_rows.append(cells)
    pr(_table(["Run"] + [short[k] for k in cat_keys_pref], cat_rows))
    pr("")

    # ---- 1.6 SFT/DPO data summary ----
    pr(HR2)
    pr("[1.6] SFT / DPO data counterparts (sizes; judge-filter funnel for SoF-DPO)")
    pr(HR2)
    data_paths = [
        ("SFT  final_sft_data (mix80: PUPIL train ⊕ CGBench)", "contrastive_experiments/final_sft_data/train.json"),
        ("SFT  final_sft_data  val",                            "contrastive_experiments/final_sft_data/val.json"),
        ("SFT  final_sft_data_cgb_only train",                  "contrastive_experiments/final_sft_data_cgb_only/train.json"),
        ("SFT  final_sft_data_cgb_only val",                    "contrastive_experiments/final_sft_data_cgb_only/val.json"),
        ("SFT  sof_sft_warmstart (256fr warmstart)",            "contrastive_experiments/sof_dpo/data/sof_sft_warmstart.json"),
        ("DPO  sof_dpo_train  (raw pairs)",                     "contrastive_experiments/sof_dpo/data/sof_dpo_train.json"),
        ("DPO  sof_dpo_train  val",                             "contrastive_experiments/sof_dpo/data/sof_dpo_train.val.json"),
        ("DPO  judged.clips (post judge filter)",               "contrastive_experiments/sof_dpo/data/sof_dpo_train.judged.clips.json"),
        ("DPO  judged.clips.balanced (final used set)",         "contrastive_experiments/sof_dpo/data/sof_dpo_train.judged.clips.balanced.json"),
        ("DPO  cgbench_dpo_clean",                              "contrastive_experiments/dpo_data/cgbench_dpo_clean.json"),
        ("DPO  cgbench_dpo_full",                               "contrastive_experiments/dpo_data/cgbench_dpo_full.json"),
        ("DPO  cgbench_dpo_fullvids",                           "contrastive_experiments/dpo_data/cgbench_dpo_fullvids.json"),
    ]
    drows = []
    for label, p in data_paths:
        n = _len_json(os.path.join(ROOT, p))
        drows.append([label, str(n) if n is not None else " — ", p])
    pr(_table(["Dataset", "n_examples", "path (relative to repo root)"], drows))
    pr("")

    stats = _read_json(os.path.join(ROOT, "contrastive_experiments/sof_dpo/data/sof_dpo_train.judged.json.stats.json"))
    if stats:
        pr("SoF-DPO judge-filter funnel (from sof_dpo_train.judged.json.stats.json):")
        order = ["in_total", "out_total", "verdict_YES", "verdict_PARTIAL", "verdict_NO",
                 "axis_visual_seen", "axis_visual_dropped",
                 "axis_audio_seen",  "axis_audio_dropped",
                 "axis_priority_seen", "axis_priority_dropped",
                 "axis_time_seen", "axis_time_dropped"]
        sr = [[k, str(stats.get(k, "—"))] for k in order if k in stats]
        pr(_table(["field", "count"], sr))
        pr("  in_total → out_total : raw pairs surviving judge filter (verdict_NO retained as 'rejected' label).")
        pr("")

    # ---- PART 2: Appendix dump ----
    pr(HR)
    pr("PART 2 — APPENDIX DUMP (full leaderboard + every post-training run we ran)")
    pr(HR)
    pr("")

    pr(HR2)
    pr("[2.1] Full final_1k_benchmark leaderboard (all models, all matched/ft variants)")
    pr(HR2)
    lb_rows = sorted(rows, key=lambda r: -(r["c"]/r["j"]) if r["j"] else 1.0)
    table_rows = []
    for i, r in enumerate(lb_rows, 1):
        acc = f"{100*r['c']/r['j']:.2f}" if r["j"] else "—"
        table_rows.append([str(i), r["model"], acc, f"{r['c']}/{r['j']}", str(r["pending"]), r["exp"][:80]])
    pr(_table(["#", "model", "acc(%)", "correct", "pending", "experiment"], table_rows))
    pr("")

    pr(HR2)
    pr("[2.2] All qwen3_vl_ft (post-training) runs only — sorted by acc")
    pr(HR2)
    ft_rows = sorted([r for r in rows if r["model"] == "qwen3_vl_ft"],
                     key=lambda r: -(r["c"]/r["j"]) if r["j"] else 1.0)
    ft_table = []
    for i, r in enumerate(ft_rows, 1):
        acc = f"{100*r['c']/r['j']:.2f}" if r["j"] else "—"
        ft_table.append([str(i), acc, f"{r['c']}/{r['j']}", r["exp"]])
    pr(_table(["#", "acc(%)", "correct", "experiment"], ft_table))
    pr("")

    # ---- 2.3 every run's per-source / per-category dump (compact) ----
    pr(HR2)
    pr("[2.3] Per source_of_fact for every qwen3_vl_ft / qwen3_vl_matched run")
    pr(HR2)
    sof_keys = ["audio", "visual", "priority", "time"]
    rows_for_dump = [r for r in rows if r["model"] in ("qwen3_vl_ft", "qwen3_vl_matched", "qwen3_vl", "qwen32_vl")]
    rows_for_dump = sorted(rows_for_dump, key=lambda r: -(r["c"]/r["j"]) if r["j"] else 1.0)
    dump_tab = []
    for r in rows_for_dump:
        d = _by(r["recs"], "source_of_fact")
        cells = [r["model"]]
        for k in sof_keys:
            if k in d and d[k][1]:
                cc, tt = d[k]
                cells.append(f"{100*cc/tt:5.2f}")
            else:
                cells.append("  —  ")
        cells.append(r["exp"][:70])
        dump_tab.append(cells)
    pr(_table(["model"] + sof_keys + ["experiment"], dump_tab))
    pr("")

    pr(HR)
    pr("End of dump. Re-run `python contrastive_experiments/make_post_training_nums.py` to refresh.")
    pr(HR)

    text = "\n".join(out) + "\n"
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"✅ wrote {OUT_PATH}  ({len(text):,} chars, {text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
