"""
lvbench.py — LVBench dataset class for VLMEvalKit
==================================================
Implements LVBench as a VideoMCQ benchmark inside VLMEvalKit.

Design references:
  • VLMEvalKit custom benchmark guide:
      https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md
  • VLMEvalKit video dataset config pattern (video_dataset_config.py):
      https://vlmevalkit.readthedocs.io/en/latest/ConfigSystem.html
  • LVBench dataset format (HuggingFace THUDM/LVBench):
      https://huggingface.co/datasets/THUDM/LVBench
  • LVBench GitHub (ICCV 2025):
      https://github.com/zai-org/LVBench

LVBench annotation format (from HF dataset):
  Each row has: video_id, question, options (list), answer, question_type, ...
  We map this onto VLMEvalKit's video MCQ TSV schema:
    index | video | question | A | B | C | D | answer | category

VLMEvalKit's video pipeline:
  • All video files are resolved from LMUData/videos/<video_filename>
    (set env var LMUData to override the default $HOME/LMUData)
  • For LVBench we symlink/set LMUData so that
    LMUData/LVBench/videos → /data/Pupil/lvbench_v2
  • The dataset TSV is placed in LMUData/LVBench/LVBench.tsv
"""

import os
import os.path as osp
import pandas as pd
import numpy as np
from collections import defaultdict

from .video_base import VideoBaseDataset  # VLMEvalKit's VideoBaseDataset
from ..smp import get_logger, listinstr


logger = get_logger(__name__)


class LVBench(VideoBaseDataset):
    """
    LVBench: An Extreme Long Video Understanding Benchmark (ICCV 2025).

    Paper:   https://arxiv.org/abs/2406.08035
    Dataset: https://huggingface.co/datasets/THUDM/LVBench
    GitHub:  https://github.com/zai-org/LVBench

    This class integrates LVBench into VLMEvalKit's standard video evaluation
    pipeline.  It inherits VideoBaseDataset, which handles:
      - Video frame extraction (decord / av backend)
      - Per-question prompt construction
      - Distributed inference across GPUs

    Usage in VLMEvalKit config JSON (see ConfigSystem docs):
        "LVBench_1fps": {
            "class": "LVBench",
            "dataset": "LVBench",
            "fps": 1.0,
            "nframe": null
        }

    Or fixed frame count:
        "LVBench_32frame": {
            "class": "LVBench",
            "dataset": "LVBench",
            "nframe": 32
        }
    """

    # Huggingface dataset identifier — used only by prepare_lvbench_tsv.py
    # to download annotations.
    HF_DATASET = "THUDM/LVBench"

    # Dataset filename inside LMUData/LVBench/
    TSV_FILENAME = "LVBench.tsv"

    # VLMEvalKit expects this to be set for video datasets.
    # "pack" means all questions for a video are sent in one model call;
    # "nopack" = one call per question.  LVBench is nopack by default
    # because videos are very long and we want isolated grading.
    TYPE = "MCQ"
    MODALITY = "VIDEO"

    # Six capability categories defined in the LVBench paper (Table 1):
    #   https://arxiv.org/abs/2406.08035
    CATEGORIES = [
        "Counting",
        "Entity Recognition",
        "Event Sequence",
        "Relation Inference",
        "Summarization",
        "Temporal Grounding",
    ]

    def __init__(self, dataset="LVBench", nframe=0, fps=-1, **kwargs):
        """
        Args:
            dataset  : dataset name string (used for TSV file lookup)
            nframe   : fixed number of frames to sample per video.
                       If None and fps<=0, defaults to 16.
                       Community practice for long-video benchmarks is
                       fps=1.0 or nframe=32/64.  See VLMEvalKit issue #876:
                       https://github.com/open-compass/VLMEvalKit/issues/876
            fps      : frames per second.  Takes precedence over nframe
                       when > 0.
        """
        self.dataset_name = dataset
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, **kwargs)

    # ── Dataset preparation (called by VideoBaseDataset.__init__) ───────────

    def prepare_dataset(self, dataset):
        """
        Return {'root': <video_dir>, 'data_file': <tsv_path>}.
        VideoBaseDataset.__init__ calls this and loads the TSV itself.

        VLMEvalKit convention (Development.md):
          TSV lives in $LMUData/<dataset>/<dataset>.tsv
          Videos live in $LMUData/<dataset>/videos/
        """
        lmu_root = os.environ.get("LMUData", osp.expanduser("~/LMUData"))
        tsv_path = osp.join(lmu_root, self.dataset_name, self.TSV_FILENAME)
        video_dir = osp.join(lmu_root, self.dataset_name, "videos")

        if not osp.exists(tsv_path):
            raise FileNotFoundError(
                f"LVBench TSV not found at {tsv_path}.\n"
                f"Run  02_prepare_lvbench_tsv.py  first."
            )

        return {"root": video_dir, "data_file": tsv_path}

    # ── Prompt construction ───────────────────────────────────────────────────

    def build_prompt(self, line, video_llm=True):
        """
        Build the multi-modal message for one LVBench question.

        VLMEvalKit message format (Development.md):
            [
                dict(type='video', value=<path>),
                dict(type='text',  value=<prompt_str>),
            ]

        The prompt template follows the original LVBench eval script:
          https://github.com/zai-org/LVBench/blob/main/scripts/test_acc.py
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Resolve local video path
        lmu_root = os.environ.get("LMUData", osp.expanduser("~/LMUData"))
        video_path = osp.join(lmu_root, self.dataset_name, "videos", str(line["video"]))

        if not osp.exists(video_path):
            logger.warning(f"Video not found: {video_path}")

        question = str(line["question"])
        options = {
            "A": str(line.get("A", "")),
            "B": str(line.get("B", "")),
            "C": str(line.get("C", "")),
            "D": str(line.get("D", "")),
        }

        # Build option text — mirrors the LVBench official prompt format
        option_str = "\n".join(f"{k}. {v}" for k, v in options.items() if v.strip())

        prompt = (
            f"{question}\n\n"
            f"{option_str}\n\n"
            "Answer with the option letter from the given choices directly."
        )

        msgs = [
            dict(type="video", value=video_path),
            dict(type="text", value=prompt),
        ]
        return msgs

    # ── Evaluation / accuracy calculation ─────────────────────────────────────

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Compute overall accuracy and per-category accuracy for LVBench.

        VLMEvalKit evaluation pipeline (Development.md):
          eval_file is a .xlsx with columns: index, prediction, (gold answer)

        The LVBench paper reports accuracy per capability category.
        """
        assert eval_file.endswith(".xlsx"), "eval_file must be an .xlsx file"

        data = load_table(eval_file)  # VLMEvalKit utility
        assert "index" in data and "prediction" in data, (
            "eval_file must have 'index' and 'prediction' columns"
        )

        # Merge predictions with ground truth from self.data
        merged = data.merge(
            self.data[["index", "answer", "category"]],
            on="index",
            how="left",
        )

        # Exact-match accuracy (upper-case single letter)
        merged["pred_clean"] = merged["prediction"].apply(self._clean_pred)
        merged["correct"] = merged["pred_clean"] == merged["answer"].str.upper()

        overall_acc = merged["correct"].mean() * 100

        # Per-category accuracy
        cat_results = {}
        for cat, grp in merged.groupby("category"):
            cat_results[cat] = grp["correct"].mean() * 100

        results = {
            "Overall": round(overall_acc, 2),
            **{k: round(v, 2) for k, v in sorted(cat_results.items())},
        }

        logger.info(f"\nLVBench Results ({eval_file}):")
        logger.info(f"  Overall accuracy: {overall_acc:.2f}%")
        for cat, acc in sorted(cat_results.items()):
            logger.info(f"  {cat}: {acc:.2f}%")

        return results

    @staticmethod
    def _clean_pred(pred_str):
        """
        Extract single-letter answer from free-form model output.
        Mirrors the answer extraction in VLMEvalKit's exact-match mode.
        Reference: VLMEvalKit docs on judge/extractor modes:
          https://vlmevalkit.readthedocs.io/en/latest/Quickstart.html
        """
        if not isinstance(pred_str, str):
            return ""
        pred_str = pred_str.strip().upper()
        for letter in ["A", "B", "C", "D"]:
            if pred_str.startswith(letter):
                return letter
        # Fallback: look for first occurrence of A/B/C/D
        for char in pred_str:
            if char in "ABCD":
                return char
        return ""


# ── Helper to load VLMEvalKit's .xlsx output ─────────────────────────────────

def load_table(path):
    """Load a .xlsx or .tsv result file into a DataFrame."""
    import pandas as pd
    if path.endswith(".xlsx"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, sep="\t", dtype=str)
