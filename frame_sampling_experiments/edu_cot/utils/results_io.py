"""
utils/results_io.py — JSONL results saving, loading, and hot-resume.

Filename format:
  results/<dataset>/<model>_<variant_tag>_results.jsonl

The variant tag encodes which pipeline stages are active so results from
different experiment configurations never collide.
"""

import json
import logging
import os
from typing import Any, Dict, Set

from omegaconf import DictConfig

logger = logging.getLogger("educot.results")


def build_variant_tag(cfg: DictConfig) -> str:
    """
    Compact tag encoding the experiment configuration.

    Examples:
      seg_uniform_sel_k256_u32
      seg_scene_detect_kf_sel_k256_u32
      seg_uniform_nosel_k256_u0
    """
    parts = [f"seg_{cfg.pipeline.segmentation}"]

    # Include num_segments in tag for uniform segmentation variants
    ns = cfg.segmentation.num_segments
    if ns != 12:
        parts.append(f"ns{ns}")

    # Include scene_detect threshold if non-default
    if cfg.pipeline.segmentation == "scene_detect":
        thresh = cfg.segmentation.scene_detect.threshold
        if thresh != 50.0:
            parts.append(f"t{int(thresh)}")

    if cfg.pipeline.keyframe_filter:
        method = getattr(cfg.keyframe_filter, 'method', 'farneback')
        parts.append(f"kf_{method}")

    sel_mode = getattr(cfg.pipeline, 'selection_mode', 'vlm' if cfg.pipeline.vlm_selection else 'none')
    if sel_mode == 'clip':
        parts.append('clipsel')
    elif sel_mode == 'scene_zoom':
        top_scenes = getattr(cfg.scene_zoom, 'top_scenes', 3)
        neighbor_hops = getattr(cfg.scene_zoom, 'neighbor_hops', 1)
        fpr = getattr(cfg.scene_zoom, 'frames_per_rep', 1)
        tag = f'scenezoom_s{top_scenes}n{neighbor_hops}'
        if fpr != 1:
            tag += f'fpr{fpr}'
        parts.append(tag)
    elif sel_mode == 'time_router':
        ov = getattr(cfg.time_router, 'overview_frames', 32)
        mr = getattr(cfg.time_router, 'max_ranges', 4)
        nh = getattr(cfg.time_router, 'neighbor_hops', 1)
        parts.append(f'tr_ov{ov}r{mr}n{nh}')
    elif sel_mode == 'vlm_prune':
        parts.append('vlmprune')
    elif sel_mode == 'vlm' and cfg.pipeline.vlm_selection:
        nr = cfg.selection.num_rounds
        parts.append(f'sel_r{nr}' if nr != 12 else 'sel')
    else:
        parts.append('nosel')

    if getattr(cfg.aggregation, 'adaptive_k', False):
        ver = getattr(cfg.aggregation, 'adaptive_k_version', 1)
        parts.append(f'kadaptive_v{ver}' if ver != 1 else 'kadaptive')
    else:
        k = cfg.aggregation.context_budget_frames
        parts.append(f'k{k}')
    u = cfg.aggregation.uniform_context_frames
    parts.append(f'u{u}')

    prompt_style = getattr(getattr(cfg, 'generation', None), 'prompt_style', 'direct')
    if prompt_style and prompt_style != 'direct':
        # short tag for non-default answer prompts
        _ps_map = {
            'per_option': 'peropt',
            'direct_v2':  'promptv2',
            'direct_v3':  'promptv3',
        }
        parts.append(_ps_map.get(prompt_style, prompt_style))

    shuffle_seed = getattr(cfg.pipeline, 'shuffle_choices_seed', None)
    if shuffle_seed is not None:
        parts.append(f'shuf{shuffle_seed}')

    return "_".join(parts)


def results_filepath(cfg: DictConfig) -> str:
    """Return the full path to the results JSONL file."""
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # edu_cot/
        cfg.results_dir,
        cfg.dataset.name,
    )
    os.makedirs(results_dir, exist_ok=True)

    model_tag = cfg.model.name.replace("/", "-").replace(" ", "_")
    variant = build_variant_tag(cfg)
    return os.path.join(results_dir, f"{model_tag}_{variant}_results.jsonl")


def load_completed_uids(cfg: DictConfig) -> Set[str]:
    """Return UIDs already present in the results file (hot-resume)."""
    path = results_filepath(cfg)
    if not os.path.exists(path):
        return set()

    uids: Set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                uids.add(str(json.loads(line)["uid"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return uids


def save_result(result: Dict[str, Any], cfg: DictConfig) -> None:
    """Append one result dict as a JSON line."""
    path = results_filepath(cfg)
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
