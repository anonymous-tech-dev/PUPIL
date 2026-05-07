"""
lvbench_v2_dataset.py — VLMEvalKit-compatible VideoBaseDataset for LVBench v2.

This registers our local lvbench_v2 benchmark as a proper VLMEvalKit dataset
so it flows through the standard inference + evaluation pipeline (same
prompting, answer extraction, and exact-match scoring as all other benchmarks).

The TSV must be generated first by running:
    python prepare_tsv.py
"""

import os
import os.path as osp
import json
import warnings
import pandas as pd
from vlmeval.smp import load, get_intermediate_file_path, get_file_extension
from vlmeval.dataset.video_base import VideoBaseDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class LVBenchV2(VideoBaseDataset):
    """Local LVBench v2 — 4-option video MCQ dataset."""

    MD5 = ""  # no integrity check for local data
    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"

    def __init__(self, dataset="LVBench_v2_MCQ", nframe=0, fps=-1):
        # Store before calling super().__init__ since it will call prepare_dataset
        self._tsv_path = osp.join(SCRIPT_DIR, "LVBench_v2_MCQ.tsv")
        assert osp.exists(self._tsv_path), (
            f"TSV not found at {self._tsv_path}. Run  python prepare_tsv.py  first."
        )
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ["LVBench_v2_MCQ"]

    def prepare_dataset(self, dataset_name="LVBench_v2_MCQ", **kwargs):
        """Point VLMEvalKit at our pre-built TSV. No download needed."""
        return dict(data_file=self._tsv_path, root=osp.dirname(self._tsv_path))

    def build_prompt(self, line, video_llm):
        """Construct the VLMEvalKit message list for one QA item."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []

        # ── Video ────────────────────────────────────────────────────────
        video_path = line["video_path"]
        if video_llm:
            message.append(dict(type="video", value=video_path))
        else:
            # Fall back to extracted frames
            frame_paths = self.save_video_frames(line)
            for fp in frame_paths:
                message.append(dict(type="image", value=fp))

        # ── Question + options ───────────────────────────────────────────
        question = line["question"]
        options = {}
        for key in "ABCDE":
            if key in line and not pd.isna(line[key]) and str(line[key]).strip():
                options[key] = line[key]

        prompt = f"Question: {question}\n"
        if options:
            prompt += "Options:\n"
            for k, v in options.items():
                prompt += f"{k}. {v}\n"
            prompt += (
                "Answer with the option's letter from the given choices directly "
                "and only give the best option."
            )
        message.append(dict(type="text", value=prompt))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Exact-match evaluation (same as other MCQ video benchmarks).
        Returns a DataFrame with overall + per-question-type accuracy.
        """
        from vlmeval.smp import dump
        from collections import defaultdict

        suffix = get_file_extension(eval_file)
        assert suffix in ["xlsx", "json", "tsv", "csv", "jsonl"], (
            f"Unsupported result format: {suffix}"
        )
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        score_file = get_intermediate_file_path(eval_file, "_acc")

        total = 0
        correct = 0
        per_type = defaultdict(lambda: {"total": 0, "correct": 0})

        for _, row in data.iterrows():
            gt = str(row.get("answer", "")).strip().upper()
            pred = str(row.get("prediction", "")).strip().upper()
            if not gt:
                continue

            # Extract single letter from prediction
            pred_letter = ""
            for ch in pred:
                if ch in "ABCDE":
                    pred_letter = ch
                    break

            total += 1
            hit = pred_letter == gt
            if hit:
                correct += 1

            # Per question type
            qt_raw = row.get("question_type", "[]")
            try:
                qtypes = json.loads(qt_raw) if isinstance(qt_raw, str) else qt_raw
            except (json.JSONDecodeError, TypeError):
                qtypes = []
            for qt in (qtypes if isinstance(qtypes, list) else [qtypes]):
                per_type[qt]["total"] += 1
                if hit:
                    per_type[qt]["correct"] += 1

        acc = 100.0 * correct / total if total else 0.0
        results = {"Overall": {"total": total, "correct": correct, "acc": round(acc, 2)}}
        for qt, v in sorted(per_type.items()):
            qa = 100.0 * v["correct"] / v["total"] if v["total"] else 0.0
            results[qt] = {"total": v["total"], "correct": v["correct"], "acc": round(qa, 2)}

        print(f"\n{'='*60}")
        print(f"  LVBench v2 Results — {total} items")
        print(f"  Overall Accuracy: {acc:.2f}%  ({correct}/{total})")
        print(f"{'='*60}")
        for qt, v in results.items():
            if qt != "Overall":
                print(f"  {qt:<35} {v['acc']:>6.2f}%  ({v['correct']}/{v['total']})")
        print(f"{'='*60}\n")

        result_df = pd.DataFrame(results).T
        dump(result_df, score_file)
        return result_df
