"""
DPO dataset & collator for Qwen2/2.5/3 VL.

Each input sample is a dict with at least::

    {
        "video": "/abs/path/to/video.mp4",   # OR "image": "/abs/path/img.png"
        "prompt": "<video>\\nWhat is going on?",   # raw user-side prompt
        "chosen": "Better answer text...",
        "rejected": "Worse answer text..."
    }

Optional keys: ``id``, ``source``, ``conversations`` (LLaVA-style — only the
first human turn is used as the prompt; the assistant turn is ignored in favour
of the explicit ``chosen``/``rejected`` fields).

The collator emits a *flat* batch ready for a Trainer-style DPO compute_loss::

    input_ids        : (2*B, L)   chosen-then-rejected concatenated
    attention_mask   : (2*B, L)
    completion_mask  : (2*B, L)   1 for response tokens, 0 for prompt tokens
    pixel_values_videos / video_grid_thw / second_per_grid_ts (or
    pixel_values / image_grid_thw)  — duplicated 2× so each half of the batch
    has its own visual features.

Designed to mirror :mod:`src.dataset.sft_dataset` for video-budget logic so
that DPO sees the same tokenized inputs the SFT/contrastive runs use.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)
from src.params import DataArguments

from .data_utils import (
    get_image_info,
    get_video_info,
    pad_sequence,
    replace_image_tokens,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    """Pairwise-preference dataset for video/image VLM DPO.

    Returns a *single* ``__getitem__`` dict with the prompt tokens, two
    completion tensors (chosen / rejected), and the visual features.  The
    collator below splits this into the flat 2*B batch the trainer wants.
    """

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        max_seq_length: int = 32768,
    ) -> None:
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, "r") as f:
                self.list_data_dict: List[Dict[str, Any]] = json.load(f)
        else:
            self.list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        self.max_seq_length = max_seq_length

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

        # Resolve model family from the actual config (NOT from substring matching
        # the model_id path). When MODEL_ID points at a local merged-SFT dir whose
        # path doesn't contain the literal string "Qwen3", the old substring check
        # silently fell through to the Qwen2 branch (patch_size=14) and produced
        # video reshape errors at training time.
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=False)
            _mt = getattr(_cfg, "model_type", "") or ""
        except Exception:
            _mt = ""
        if _mt.startswith("qwen3_vl"):
            self._model_family = "qwen3"
        elif _mt == "qwen2_5_vl":
            self._model_family = "qwen2.5"
        else:
            self._model_family = "qwen2"
        _path_says_q3 = "Qwen3" in self.model_id
        if (self._model_family == "qwen3") != _path_says_q3:
            print(
                f"[DPODataset] model_id path={self.model_id!r} substring-says-Qwen3={_path_says_q3} "
                f"but AutoConfig.model_type={_mt!r} → using family={self._model_family!r}"
            )

        if self._model_family == "qwen3":
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

        # ── Replicate sft_dataset's video-budget math so DPO and SFT see
        # ── identically-sized video tensors for a given max_seq_length.
        MERGE = 2
        factor = self.image_patch_size * MERGE
        factor_sq = factor * factor
        FRAME_FACTOR = 2
        QWEN_MIN_TOKEN_MUL = 256
        QWEN_MAX_TOKEN_MUL = 768
        model_min_pixels = QWEN_MIN_TOKEN_MUL * factor_sq
        model_max_pixels = QWEN_MAX_TOKEN_MUL * factor_sq

        _default_min, _default_max = 100352, 602112
        user_set_min = data_args.video_min_pixels != _default_min
        user_set_max = data_args.video_max_pixels != _default_max
        if not user_set_min and self.video_min_pixel < model_min_pixels:
            self.video_min_pixel = model_min_pixels
        if not user_set_max and self.video_max_pixel < model_max_pixels:
            self.video_max_pixel = model_max_pixels

        video_token_budget = int(max_seq_length * 0.85)
        min_tok_per_frame = max(1, self.video_min_pixel // factor_sq)

        self.video_total_pixels = getattr(data_args, "video_total_pixels", None)
        if self.video_total_pixels is None:
            self.video_total_pixels = video_token_budget * factor_sq // FRAME_FACTOR

        self.video_max_frames = getattr(data_args, "video_max_frames", None)
        if self.video_max_frames is None:
            self.video_max_frames = max(
                FRAME_FACTOR,
                (video_token_budget // min_tok_per_frame) // FRAME_FACTOR * FRAME_FACTOR,
            )

    def __len__(self) -> int:
        return len(self.list_data_dict)

    # ─── prompt extraction ──────────────────────────────────────────────
    @staticmethod
    def _extract_prompt(sample: Dict[str, Any]) -> str:
        if "prompt" in sample:
            return sample["prompt"]
        # Fall back to first human turn from a LLaVA-style conversation
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") == "human":
                    return turn["value"]
        raise KeyError(f"Sample {sample.get('id')} lacks 'prompt' or 'conversations'.")

    # ─── video / image loaders ──────────────────────────────────────────
    def _load_visual(self, sample):
        """Returns (images, videos, video_kwargs, grid_key, pixel_key, is_video)."""
        images, videos, video_kwargs = None, None, {}
        grid_key, pixel_key, is_video = None, None, False

        if "image" in sample:
            grid_key, pixel_key = "image_grid_thw", "pixel_values"
            files = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
            folder = self.data_args.image_folder
            images = []
            for f in files:
                if not os.path.exists(f) and not f.startswith("http") and folder:
                    f = os.path.join(folder, f)
                images.append(get_image_info(
                    f, self.image_min_pixel, self.image_max_pixel,
                    self.image_resized_w, self.image_resized_h, self.image_patch_size,
                ))

        elif "video" in sample:
            is_video = True
            grid_key, pixel_key = "video_grid_thw", "pixel_values_videos"
            files = sample["video"] if isinstance(sample["video"], list) else [sample["video"]]
            folder = self.data_args.image_folder
            videos = []
            for f in files:
                if not os.path.exists(f) and not f.startswith("http") and folder:
                    f = os.path.join(folder, f)
                video_input, video_kwargs = get_video_info(
                    f, self.video_min_pixel, self.video_max_pixel,
                    self.video_resized_w, self.video_resized_h,
                    self.fps, self.nframes, self.image_patch_size,
                    return_video_metadata=self.return_video_metadata,
                    total_pixels=self.video_total_pixels,
                    max_frames=self.video_max_frames,
                )
                videos.append(video_input)

        return images, videos, video_kwargs, grid_key, pixel_key, is_video

    # ─── prompt tokenisation (with vision insertion) ────────────────────
    def _tokenise_prompt(self, prompt_text, images, videos, video_kwargs, is_video):
        """Run the processor on the *prompt only*, returning (input_ids, vision_dict)."""
        proc = self.processor
        # Strip any leading <video>/<image> tokens out of the user text and inject
        # the vision-pad blocks Qwen expects.
        prompt_text = replace_image_tokens(prompt_text, is_video=is_video)

        # Build the prompt sequence ending right before the assistant tokens
        sys_prefix = ""
        if len(SYSTEM_MESSAGE) > 0 and self._model_family != "qwen3":
            sys_prefix = (
                f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            )
        full_prompt = (
            sys_prefix
            + f"{DEFAULT_IM_START_TOKEN}user\n{prompt_text}{DEFAULT_IM_END_TOKEN}\n"
            + f"{DEFAULT_IM_START_TOKEN}assistant\n"
        )

        vision = {}
        if DEFAULT_IMAGE_TOKEN in full_prompt and images:
            inputs = proc(text=[full_prompt], images=images, videos=None,
                          padding=False, do_resize=False, return_tensors="pt")
            vision["pixel_values"] = inputs["pixel_values"]
            vision["image_grid_thw"] = inputs["image_grid_thw"]
            input_ids = inputs["input_ids"][0]
        elif DEFAULT_VIDEO_TOKEN in full_prompt and videos:
            if self._model_family == "qwen3":
                video_datas, video_metas = zip(*videos)
                inputs = proc(text=[full_prompt], images=None,
                              videos=list(video_datas),
                              padding=False, do_resize=False, return_tensors="pt",
                              video_metadata=list(video_metas), **video_kwargs)
            elif self._model_family == "qwen2.5":
                inputs = proc(text=[full_prompt], images=None, videos=videos,
                              padding=False, do_resize=False, return_tensors="pt",
                              **video_kwargs)
                if "second_per_grid_ts" in inputs:
                    vision["second_per_grid_ts"] = list(inputs["second_per_grid_ts"])
            else:
                inputs = proc(text=[full_prompt], images=None, videos=videos,
                              padding=False, do_resize=False, return_tensors="pt")
            vision["pixel_values_videos"] = inputs["pixel_values_videos"]
            vision["video_grid_thw"] = inputs["video_grid_thw"]
            input_ids = inputs["input_ids"][0]
        else:
            input_ids = proc.tokenizer(full_prompt, add_special_tokens=False,
                                       return_tensors="pt")["input_ids"][0]

        return input_ids.long(), vision

    # ─── completion tokenisation ────────────────────────────────────────
    def _tokenise_completion(self, response_text: str) -> torch.Tensor:
        """Tokenise an assistant response with the closing <|im_end|>."""
        text = f"{response_text}{DEFAULT_IM_END_TOKEN}\n"
        ids = self.processor.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        return ids.long()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self.list_data_dict[idx]
            prompt_text = self._extract_prompt(sample)
            chosen_text = sample["chosen"]
            rejected_text = sample["rejected"]

            images, videos, video_kwargs, _gk, _pk, is_video = self._load_visual(sample)
            prompt_ids, vision = self._tokenise_prompt(
                prompt_text, images, videos, video_kwargs, is_video,
            )
            chosen_ids = self._tokenise_completion(chosen_text)
            rejected_ids = self._tokenise_completion(rejected_text)

            # ── Hard-enforce max_seq_length on (prompt + completion) ──────
            # CRITICAL: SFT truncates final input_ids to max_seq_length, but
            # DPO does TWO forwards (chosen + rejected) so an over-budget
            # prompt produces NaN logits and corrupts gradients.
            #
            # We CANNOT truncate the prompt itself because that would cut
            # into the <|video_pad|> token block, desynchronising it from
            # pixel_values_videos / video_grid_thw and producing a "video
            # features != video tokens" error in the model forward.
            #
            # Strategy:
            #   1. Truncate over-long completions to fit the remaining budget
            #   2. If the prompt ALONE already exceeds budget, skip this
            #      sample (rare — only ~30% of CGBench long videos)
            MIN_COMPLETION = 32
            SLACK = 8
            if prompt_ids.shape[0] + MIN_COMPLETION + SLACK > self.max_seq_length:
                # Prompt too long even with minimum completion — skip
                raise ValueError(
                    f"prompt_len={prompt_ids.shape[0]} exceeds max_seq_length="
                    f"{self.max_seq_length} - {MIN_COMPLETION} - {SLACK}; "
                    f"skipping (build dataset with stricter video_max_frames)"
                )

            max_completion = self.max_seq_length - prompt_ids.shape[0] - SLACK
            if chosen_ids.shape[0] > max_completion:
                chosen_ids = chosen_ids[:max_completion]
            if rejected_ids.shape[0] > max_completion:
                rejected_ids = rejected_ids[:max_completion]

            item = {
                "prompt_input_ids": prompt_ids,
                "chosen_input_ids": chosen_ids,
                "rejected_input_ids": rejected_ids,
            }
            item.update(vision)
            return item
        except Exception as e:  # noqa: BLE001
            print(f"[DPODataset] Skipping sample {idx} due to: {e!r}")
            return self.__getitem__((idx + 1) % len(self.list_data_dict))


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────
class DataCollatorForDPODataset:
    """Build the flat (2*B, L) batch DPOTrainer expects.

    The chosen half is stacked first, then the rejected half.  Visual features
    are duplicated row-wise so each half indexes its own copy.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_seqs: List[torch.Tensor] = []
        rejected_seqs: List[torch.Tensor] = []
        chosen_comp_masks: List[torch.Tensor] = []
        rejected_comp_masks: List[torch.Tensor] = []

        # vision buckets
        pv_img, thw_img = [], []
        pv_vid, thw_vid = [], []
        sec_per_grid: List = []
        has_video = False
        has_image = False

        for ex in examples:
            p = ex["prompt_input_ids"]
            c = ex["chosen_input_ids"]
            r = ex["rejected_input_ids"]

            chosen_seqs.append(torch.cat([p, c], dim=0))
            rejected_seqs.append(torch.cat([p, r], dim=0))

            chosen_comp_masks.append(torch.cat(
                [torch.zeros(p.shape[0], dtype=torch.long),
                 torch.ones(c.shape[0], dtype=torch.long)], dim=0))
            rejected_comp_masks.append(torch.cat(
                [torch.zeros(p.shape[0], dtype=torch.long),
                 torch.ones(r.shape[0], dtype=torch.long)], dim=0))

            if "pixel_values_videos" in ex:
                has_video = True
                pv_vid.append(ex["pixel_values_videos"])
                thw_vid.append(ex["video_grid_thw"])
                if "second_per_grid_ts" in ex:
                    sec_per_grid.extend(ex["second_per_grid_ts"])
            elif "pixel_values" in ex:
                has_image = True
                pv_img.append(ex["pixel_values"])
                thw_img.append(ex["image_grid_thw"])

        # Pad chosen and rejected separately, then stack
        # Stack [chosen; rejected] on the batch dim — pad to the max length
        # across BOTH halves so they share the same L.
        chosen_padded = pad_sequence(chosen_seqs, padding_side="right",
                                     padding_value=self.pad_token_id)
        rejected_padded = pad_sequence(rejected_seqs, padding_side="right",
                                       padding_value=self.pad_token_id)
        comp_chosen = pad_sequence(chosen_comp_masks, padding_side="right",
                                   padding_value=0)
        comp_rejected = pad_sequence(rejected_comp_masks, padding_side="right",
                                     padding_value=0)

        # Equalise sequence length across halves
        L = max(chosen_padded.shape[1], rejected_padded.shape[1])

        def _right_pad(t: torch.Tensor, length: int, val: int) -> torch.Tensor:
            if t.shape[1] >= length:
                return t
            pad = t.new_full((t.shape[0], length - t.shape[1]), val)
            return torch.cat([t, pad], dim=1)

        chosen_padded = _right_pad(chosen_padded, L, self.pad_token_id)
        rejected_padded = _right_pad(rejected_padded, L, self.pad_token_id)
        comp_chosen = _right_pad(comp_chosen, L, 0)
        comp_rejected = _right_pad(comp_rejected, L, 0)

        input_ids = torch.cat([chosen_padded, rejected_padded], dim=0)
        completion_mask = torch.cat([comp_chosen, comp_rejected], dim=0)
        attention_mask = (input_ids != self.pad_token_id).long()

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

        # Duplicate visuals so they line up with [chosen; rejected]
        if has_video:
            pv = torch.cat(pv_vid, dim=0)
            thw = torch.cat(thw_vid, dim=0)
            batch["pixel_values_videos"] = torch.cat([pv, pv], dim=0)
            batch["video_grid_thw"] = torch.cat([thw, thw], dim=0)
            if sec_per_grid:
                batch["second_per_grid_ts"] = sec_per_grid + sec_per_grid
        if has_image:
            pv = torch.cat(pv_img, dim=0)
            thw = torch.cat(thw_img, dim=0)
            batch["pixel_values"] = torch.cat([pv, pv], dim=0)
            batch["image_grid_thw"] = torch.cat([thw, thw], dim=0)

        return batch


def make_dpo_data_module(model_id: str, processor, data_args: DataArguments,
                         max_seq_length: int = 32768) -> Dict[str, Any]:
    """Build train + (optional) eval DPODataset and a collator."""
    train_ds = DPODataset(
        data_path=data_args.data_path, processor=processor,
        data_args=data_args, model_id=model_id, max_seq_length=max_seq_length,
    )
    eval_ds = None
    if getattr(data_args, "eval_path", None):
        eval_ds = DPODataset(
            data_path=data_args.eval_path, processor=processor,
            data_args=data_args, model_id=model_id, max_seq_length=max_seq_length,
        )

    collator = DataCollatorForDPODataset(pad_token_id=processor.tokenizer.pad_token_id)
    return dict(train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
