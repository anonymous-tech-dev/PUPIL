"""
==============================================================================
Contrastive SFT Dataset
==============================================================================
Extends the vanilla SupervisedDataset to pass through metadata needed for
contrastive learning (timestamps, duration, source, full-video paths).

Stage: Data Loading & Preparation

Key design decisions:
  - Does NOT generate negatives at dataset level — that happens in the
    Trainer/collator to enable efficient in-batch contrastive learning.
  - Passes through raw metadata so the Trainer can decide at runtime
    what negatives to construct.
  - FineVideo samples (no timestamps) get flagged so λ=0 is applied.
  - CGBench samples carry both clue_vid and full_vid paths for temporal
    negative generation.
==============================================================================
"""

import copy
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    LLAVA_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from src.dataset.data_utils import (
    get_image_info,
    get_video_info,
    llava_to_openai,
    pad_sequence,
)


# ═══════════════════════════════════════════════════════════════════════
# Path resolution: map clue_vid → full_vid for temporal negatives
# ═══════════════════════════════════════════════════════════════════════

def build_uid_to_full_video_map(
    train_vids_dir: str,
) -> Dict[str, str]:
    """
    Build a mapping from video_uid → full video path.
    
    CGBench train_vids are named {video_uid}.mp4 and contain the full-length
    video. We need these for temporal negative sampling (V-04, V-05)
    because the clue_vids are already trimmed to the relevant segment.
    
    Args:
        train_vids_dir: Path to /data/.../CGBench/train_vids/
    Returns:
        Dict mapping video_uid string to full absolute path.
    """
    mapping = {}
    if not os.path.isdir(train_vids_dir):
        return mapping
    for fname in os.listdir(train_vids_dir):
        if fname.endswith(".mp4"):
            uid = fname[:-4]  # strip .mp4
            mapping[uid] = os.path.join(train_vids_dir, fname)
    return mapping


class ContrastiveSFTDataset(Dataset):
    """
    Dataset for contrastive SFT fine-tuning.
    
    Extends the vanilla SupervisedDataset logic but additionally returns
    metadata needed for contrastive learning:
      - source: "cgbench", "finevideo", or "edubench"
      - has_timestamps: whether temporal negatives are possible
      - timestamps_sec: ground truth timestamp segments
      - duration_sec: full video duration
      - full_video_path: path to the full-length video (for temporal clip extraction)
      - video_uid: CGBench video UID
    
    The actual negative generation (blackening, Gaussian noise, temporal
    shifting) happens in the ContrastiveSFTTrainer at collation/training
    time for memory efficiency.
    """

    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args,
        model_id: str,
        max_seq_length: int = 32768,
        # --- Contrastive-specific args ---
        cgbench_train_vids_dir: str = "",
        max_samples_cgbench: int = -1,
        max_samples_finevideo: int = -1,
        max_samples_edubench: int = -1,
        use_reasoning_traces: bool = False,
        cgbench_anchors_path: str = "",
    ):
        super().__init__()
        
        # ── Load and optionally subsample data ──
        if isinstance(data_path, str):
            with open(data_path, "r") as f:
                all_data = json.load(f)
        else:
            all_data = data_path

        # Stage: Subsample per source if requested (knobs for sample counts)
        self.list_data_dict = self._subsample_by_source(
            all_data,
            max_cgbench=max_samples_cgbench,
            max_finevideo=max_samples_finevideo,
            max_edubench=max_samples_edubench,
        )

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.max_seq_length = max_seq_length
        self.use_reasoning_traces = use_reasoning_traces

        # Video/image pixel settings (same as vanilla SFT)
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

        # ── Compute video token budget (same logic as vanilla SFT) ──
        MERGE = 2
        factor = self.image_patch_size * MERGE
        factor_sq = factor * factor
        FRAME_FACTOR = 2
        QWEN_MIN_TOKEN_MUL = 256
        QWEN_MAX_TOKEN_MUL = 768
        model_min_pixels = QWEN_MIN_TOKEN_MUL * factor_sq
        model_max_pixels = QWEN_MAX_TOKEN_MUL * factor_sq

        # Only clamp UP to model defaults when user hasn't explicitly set
        # smaller values (for FPS-based long-video training).
        _default_min = 100352
        _default_max = 602112
        user_set_min = (data_args.video_min_pixels != _default_min)
        user_set_max = (data_args.video_max_pixels != _default_max)

        if not user_set_min and self.video_min_pixel < model_min_pixels:
            self.video_min_pixel = model_min_pixels
        if not user_set_max and self.video_max_pixel < model_max_pixels:
            self.video_max_pixel = model_max_pixels

        video_token_budget = int(max_seq_length * 0.85)
        min_tok_per_frame = max(1, self.video_min_pixel // factor_sq)

        self.video_total_pixels = getattr(data_args, 'video_total_pixels', None)
        if self.video_total_pixels is None:
            self.video_total_pixels = video_token_budget * factor_sq // FRAME_FACTOR

        self.video_max_frames = getattr(data_args, 'video_max_frames', None)
        if self.video_max_frames is None:
            self.video_max_frames = max(
                FRAME_FACTOR,
                (video_token_budget // min_tok_per_frame) // FRAME_FACTOR * FRAME_FACTOR,
            )

        # ── Build UID → full video path mapping for temporal negatives ──
        self.uid_to_full_video = build_uid_to_full_video_map(cgbench_train_vids_dir)

        # ── (V-07) Build CGBench qid → gold MCQ answer text lookup ──
        # Used by anchor-weighted scoring to identify which tokens of the
        # rephrased SFT label correspond to the actual ground-truth content.
        # (T-04) Also build qid → list of WRONG choice texts (distractors)
        # used to construct MCQ-distractor answer negatives.
        self.cgbench_anchors = {}
        self.cgbench_distractors = {}
        if cgbench_anchors_path and os.path.exists(cgbench_anchors_path):
            try:
                with open(cgbench_anchors_path, "r") as f:
                    cg_data = json.load(f)
                for item in cg_data:
                    qid = str(item.get("qid", ""))
                    answer = item.get("answer", "")
                    if qid and answer:
                        self.cgbench_anchors[qid] = answer
                    # Distractors = all choices except the gold one
                    choices = item.get("choices", []) or []
                    if qid and choices and answer:
                        wrongs = [c for c in choices
                                  if isinstance(c, str) and c.strip()
                                  and c.strip().lower() != answer.strip().lower()]
                        if wrongs:
                            self.cgbench_distractors[qid] = wrongs
                print(
                    f"[ContrastiveSFTDataset] loaded {len(self.cgbench_anchors)} "
                    f"CGBench anchor texts and "
                    f"{len(self.cgbench_distractors)} distractor pools "
                    f"from {cgbench_anchors_path}"
                )
            except Exception as e:
                print(
                    f"[ContrastiveSFTDataset] warning: failed to load anchors "
                    f"from {cgbench_anchors_path}: {e}"
                )

    @staticmethod
    def _subsample_by_source(
        data: List[dict],
        max_cgbench: int = -1,
        max_finevideo: int = -1,
        max_edubench: int = -1,
    ) -> List[dict]:
        """
        Stage: Subsampling — apply per-source sample count knobs.
        -1 means use all samples from that source.
        """
        by_source = {}
        for item in data:
            src = item.get("source", "unknown")
            by_source.setdefault(src, []).append(item)

        limits = {
            "cgbench": max_cgbench,
            "finevideo": max_finevideo,
            "edubench": max_edubench,
        }

        result = []
        for src, items in by_source.items():
            limit = limits.get(src, -1)
            if limit > 0 and limit < len(items):
                random.shuffle(items)
                items = items[:limit]
            result.extend(items)

        random.shuffle(result)
        return result

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Returns the standard SFT tensors PLUS a 'cl_metadata' dict containing
        everything the ContrastiveSFTTrainer needs for negative generation.
        """
        sources = self.list_data_dict[i]
        metadata = sources.get("metadata", {})
        source_name = sources.get("source", "unknown")

        # ── Determine contrastive eligibility ──
        # FineVideo has no timestamps → contrastive λ should be 0 for it
        has_timestamps = metadata.get("has_timestamps", False)
        timestamps_sec = metadata.get("timestamps_sec", [])
        duration_sec = metadata.get("duration_sec", 0.0)
        video_uid = metadata.get("video_uid", "")

        # Resolve full-length video path for temporal negatives
        full_video_path = self.uid_to_full_video.get(video_uid, "")

        # ── Build the standard SFT input (reuse vanilla logic) ──
        is_video = False
        processor = self.processor

        if "video" in sources:
            is_video = True
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http") and video_folder:
                        video_file = os.path.join(video_folder, video_file)
                try:
                    video_input, video_kwargs = get_video_info(
                        video_file,
                        self.video_min_pixel,
                        self.video_max_pixel,
                        self.video_resized_w,
                        self.video_resized_h,
                        self.fps,
                        self.nframes,
                        self.image_patch_size,
                        return_video_metadata=self.return_video_metadata,
                        total_pixels=self.video_total_pixels,
                        max_frames=self.video_max_frames,
                    )
                except Exception as e:
                    print(
                        f"WARNING: Failed to load video {video_file}: {e}. "
                        f"Returning next sample instead."
                    )
                    return self.__getitem__((i + 1) % len(self.list_data_dict))
                videos.append(video_input)
        elif "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http") and image_folder:
                        image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                    image_file,
                    self.image_min_pixel,
                    self.image_max_pixel,
                    self.image_resized_w,
                    self.image_resized_h,
                    self.image_patch_size,
                )
                images.append(image_input)
        else:
            grid_key = None
            pixel_key = None
            images = None
            videos = None

        # ── Process conversations (same as vanilla SFT) ──
        conversations = sources["conversations"]
        if is_video and not any(
            LLAVA_VIDEO_TOKEN in c.get("value", "") for c in conversations
        ):
            conversations = copy.deepcopy(conversations)
            for c in conversations:
                if c.get("from") == "human":
                    c["value"] = LLAVA_VIDEO_TOKEN + "\n" + c["value"]
                    break

        openai_convs = copy.deepcopy(llava_to_openai(conversations, is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0

        if len(SYSTEM_MESSAGE) > 0 and "Qwen3" not in self.model_id:
            system_message = (
                f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}"
                f"{DEFAULT_IM_END_TOKEN}\n"
            )
            system_ids = processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_ids, IGNORE_INDEX)
            all_input_ids.append(system_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(openai_convs), 2)):
            user_input = openai_convs[j]
            gpt_response = openai_convs[j + 1]

            user_input_text = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n"
                f"{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            )
            gpt_response_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input_text:
                num_images = user_input_text.count(DEFAULT_IMAGE_TOKEN)
                images_for_turn = images[
                    image_curr_count : image_curr_count + num_images
                ]
                inputs = processor(
                    text=[user_input_text],
                    images=images_for_turn,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input_text:
                num_videos = user_input_text.count(DEFAULT_VIDEO_TOKEN)
                videos_for_turn = videos[
                    video_curr_count : video_curr_count + num_videos
                ]
                if "Qwen2.5" in self.model_id:
                    inputs = processor(
                        text=[user_input_text],
                        images=images,
                        videos=videos_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif "Qwen3" in self.model_id:
                    video_datas, video_metas = zip(*videos_for_turn)
                    video_datas = list(video_datas)
                    video_metas = list(video_metas)
                    inputs = processor(
                        text=[user_input_text],
                        images=images,
                        videos=video_datas,
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                        **video_kwargs,
                        video_metadata=video_metas,
                    )
                else:
                    inputs = processor(
                        text=[user_input_text],
                        images=images,
                        videos=videos_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors="pt",
                    )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos
            else:
                prompt_input_ids = processor.tokenizer(
                    user_input_text,
                    add_special_tokens=False,
                    padding=False,
                    return_tensors="pt",
                )["input_ids"]

            response_input_ids = processor.tokenizer(
                gpt_response_text,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"]

            input_ids = torch.cat(
                [prompt_input_ids, response_input_ids], dim=1
            ).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # ── Enforce max_seq_length (same truncation as vanilla SFT) ──
        if input_ids.shape[0] > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

            if pixel_key and len(all_pixel_values) > 0:
                _pad_id = self.processor.tokenizer.convert_tokens_to_ids(
                    DEFAULT_VIDEO_TOKEN if is_video else DEFAULT_IMAGE_TOKEN
                )
                remaining_tokens = int((input_ids == _pad_id).sum().item())
                pv_cat = torch.cat(all_pixel_values, dim=0)
                thw_cat = torch.cat(all_image_grid_thw, dim=0)
                orig_tokens = int(
                    (thw_cat[:, 0] * thw_cat[:, 1] * thw_cat[:, 2]).sum().item()
                )

                if remaining_tokens == 0:
                    all_pixel_values.clear()
                    all_image_grid_thw.clear()
                elif remaining_tokens < orig_tokens:
                    merge_size = 2
                    new_pv, new_thw = [], []
                    budget = remaining_tokens
                    pv_off = 0

                    for idx in range(thw_cat.shape[0]):
                        tm = int(thw_cat[idx, 0])
                        hm = int(thw_cat[idx, 1])
                        wm = int(thw_cat[idx, 2])
                        spatial = hm * wm
                        vid_tok = tm * spatial
                        raw_patches = tm * (merge_size * hm) * (merge_size * wm)

                        if budget <= 0:
                            pv_off += raw_patches
                            continue

                        if budget >= vid_tok:
                            new_pv.append(pv_cat[pv_off : pv_off + raw_patches])
                            new_thw.append(thw_cat[idx : idx + 1])
                            budget -= vid_tok
                        else:
                            keep_t = budget // spatial
                            if keep_t > 0:
                                keep_raw = (
                                    keep_t * (merge_size * hm) * (merge_size * wm)
                                )
                                new_pv.append(pv_cat[pv_off : pv_off + keep_raw])
                                new_thw.append(
                                    torch.tensor(
                                        [[keep_t, hm, wm]], dtype=thw_cat.dtype
                                    )
                                )
                                budget -= keep_t * spatial
                        pv_off += raw_patches

                    all_pixel_values = (
                        [torch.cat(new_pv, dim=0)] if new_pv else []
                    )
                    all_image_grid_thw = new_thw if new_thw else []

                    new_feat = (
                        sum(
                            int(t[0, 0] * t[0, 1] * t[0, 2])
                            for t in all_image_grid_thw
                        )
                        if all_image_grid_thw
                        else 0
                    )
                    excess = remaining_tokens - new_feat
                    if excess > 0:
                        positions = (input_ids == _pad_id).nonzero(as_tuple=True)[0]
                        for p in positions[-excess:]:
                            input_ids[p] = self.processor.tokenizer.pad_token_id
                            labels[p] = IGNORE_INDEX

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key and len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                pixel_values = torch.nan_to_num(
                    pixel_values, nan=0.0, posinf=0.0, neginf=0.0
                )
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird

        # ══════════════════════════════════════════════════════════════
        # Stage: CL Metadata — pass through for trainer to use
        # ══════════════════════════════════════════════════════════════
        # (V-07) Look up the gold MCQ anchor for CGBench samples so the
        # trainer can build anchor-weighted token weights at scoring time.
        # (T-04) Also attach the FULL list of wrong MCQ choices — every
        # distractor becomes a contrastive negative.
        gold_anchor_text = ""
        distractor_anchor_texts: List[str] = []
        if source_name == "cgbench" and self.cgbench_anchors:
            original_id = str(metadata.get("original_id", ""))
            gold_anchor_text = self.cgbench_anchors.get(original_id, "")
            distractor_anchor_texts = list(
                self.cgbench_distractors.get(original_id, [])
            )

        data_dict["cl_metadata"] = {
            "source": source_name,
            "has_timestamps": has_timestamps,
            "timestamps_sec": timestamps_sec,
            "duration_sec": duration_sec,
            "video_uid": video_uid,
            "full_video_path": full_video_path,
            "original_video_path": sources.get("video", ""),
            "sample_index": i,
            # FineVideo has no timestamps → CL loss weight should be 0
            "cl_eligible": source_name != "finevideo" and is_video,
            # V-07 anchor (empty string for non-CGBench samples)
            "gold_anchor_text": gold_anchor_text,
            # T-04 distractors (empty list when no MCQ choices available)
            "distractor_anchor_texts": distractor_anchor_texts,
        }

        return data_dict


class ContrastiveDataCollator:
    """
    Collator for contrastive SFT.
    
    Same padding logic as the vanilla DataCollatorForSupervisedDataset,
    but additionally collects cl_metadata from each sample and attaches
    it to the batch for the trainer to use.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_cl_metadata = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            # Collect CL metadata (not a tensor — will be popped by trainer)
            if "cl_metadata" in keys:
                batch_cl_metadata.append(example["cl_metadata"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side="right", padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX
        )

        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_video_values) > 0:
            data_dict["pixel_values_videos"] = torch.cat(
                batch_pixel_video_values, dim=0
            )
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        # Attach CL metadata list (trainer will pop this before forward pass)
        if batch_cl_metadata:
            data_dict["cl_metadata"] = batch_cl_metadata

        return data_dict


def make_contrastive_data_module(
    model_id: str,
    processor: transformers.ProcessorMixin,
    data_args,
    max_seq_length: int = 32768,
    cgbench_train_vids_dir: str = "",
    max_samples_cgbench: int = -1,
    max_samples_finevideo: int = -1,
    max_samples_edubench: int = -1,
    use_reasoning_traces: bool = False,
    max_val_samples: int = -1,
    cgbench_anchors_path: str = "",
):
    """
    Factory function: creates train dataset, eval dataset, and collator
    for contrastive SFT training.
    
    Args:
        max_val_samples: Knob to limit validation set size for faster evals.
    """
    train_dataset = ContrastiveSFTDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
        max_seq_length=max_seq_length,
        cgbench_train_vids_dir=cgbench_train_vids_dir,
        max_samples_cgbench=max_samples_cgbench,
        max_samples_finevideo=max_samples_finevideo,
        max_samples_edubench=max_samples_edubench,
        use_reasoning_traces=use_reasoning_traces,
        cgbench_anchors_path=cgbench_anchors_path,
    )

    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = ContrastiveSFTDataset(
            data_path=data_args.eval_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            max_seq_length=max_seq_length,
            cgbench_train_vids_dir=cgbench_train_vids_dir,
            # For validation, apply max_val_samples uniformly
            max_samples_cgbench=max_val_samples,
            max_samples_finevideo=max_val_samples,
            max_samples_edubench=max_val_samples,
            use_reasoning_traces=use_reasoning_traces,
            cgbench_anchors_path=cgbench_anchors_path,
        )

    collator = ContrastiveDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
