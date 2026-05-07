import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        max_seq_length: int = 32768,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
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
        # video reshape errors at training time (qwen-vl-utils resized to multiples
        # of 28 instead of 32, causing patches.view() in the Qwen3-VL processor to
        # reject the tensor with a 'shape ... invalid for input of size N' error).
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
                f"[SupervisedDataset] model_id path={self.model_id!r} substring-says-Qwen3={_path_says_q3} "
                f"but AutoConfig.model_type={_mt!r} \u2192 using family={self._model_family!r}"
            )

        if self._model_family == "qwen3":
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

        # --- Compute video token budget from max_seq_length ---------------
        # qwen_vl_utils.fetch_video uses `total_pixels` to dynamically scale
        # per-frame resolution, and `max_frames` to cap the frame count.
        # The defaults inside fetch_video assume 128k inference context; for
        # training we must override them to match max_seq_length.
        MERGE = 2  # spatial merge factor (all Qwen-VL variants)
        factor = self.image_patch_size * MERGE          # 32 for Qwen3, 28 for Qwen2.x
        factor_sq = factor * factor                      # 1024 / 784
        FRAME_FACTOR = 2  # qwen-vl-utils constant

        # -- Model-aware pixel bounds (Qwen recommends 256× min, 768× max) --
        # DataArguments defaults (100352, 602112) are calibrated for Qwen2.5-VL
        # (28² = 784).  For Qwen3-VL (32² = 1024) we must scale up so the
        # *token* counts per frame stay equivalent.
        QWEN_MIN_TOKEN_MUL = 256   # Qwen-recommended minimum tokens per frame
        QWEN_MAX_TOKEN_MUL = 768   # Qwen-recommended maximum tokens per frame
        model_min_pixels = QWEN_MIN_TOKEN_MUL * factor_sq  # 262144 for Qwen3
        model_max_pixels = QWEN_MAX_TOKEN_MUL * factor_sq  # 786432 for Qwen3

        # Only clamp UP to model defaults when the user hasn't explicitly set
        # smaller values via --video_min_pixels / --video_max_pixels.
        # The default values in DataArguments are 100352 / 602112 (Qwen2.5-VL
        # calibrated).  If the user passed something smaller than the Qwen3
        # model defaults, they're intentionally constraining the budget (e.g.
        # for long-video FPS-based training) — respect that.
        _default_min = 100352   # DataArguments default
        _default_max = 602112   # DataArguments default
        user_set_min = (data_args.video_min_pixels != _default_min)
        user_set_max = (data_args.video_max_pixels != _default_max)

        if not user_set_min and self.video_min_pixel < model_min_pixels:
            self.video_min_pixel = model_min_pixels
        if not user_set_max and self.video_max_pixel < model_max_pixels:
            self.video_max_pixel = model_max_pixels

        video_token_budget = int(max_seq_length * 0.85)  # leave ~15 % for text
        min_tok_per_frame = max(1, self.video_min_pixel // factor_sq)

        self.video_total_pixels = getattr(data_args, 'video_total_pixels', None)
        if self.video_total_pixels is None:
            # total_tokens ≈ total_pixels * FRAME_FACTOR / factor_sq
            self.video_total_pixels = video_token_budget * factor_sq // FRAME_FACTOR

        self.video_max_frames = getattr(data_args, 'video_max_frames', None)
        if self.video_max_frames is None:
            # even at min resolution, total tokens must fit the budget
            self.video_max_frames = max(
                FRAME_FACTOR,
                (video_token_budget // min_tok_per_frame) // FRAME_FACTOR * FRAME_FACTOR
            )

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
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
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
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
                    print(f"WARNING: Failed to load video {video_file}: {e}. "
                          f"Returning next sample instead.")
                    return self.__getitem__((i + 1) % len(self.list_data_dict))
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        conversations = sources['conversations']
        # Auto-inject <video> token if data has video but conversations lack it
        if is_video and not any(LLAVA_VIDEO_TOKEN in c.get('value', '') for c in conversations):
            conversations = copy.deepcopy(conversations)
            for c in conversations:
                if c.get('from') == 'human':
                    c['value'] = LLAVA_VIDEO_TOKEN + '\n' + c['value']
                    break
        sources = copy.deepcopy(llava_to_openai(conversations, is_video=is_video))

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen2-VL uses a default system message so I've added this.
        # Qwen3-Vl does not use a system message by default.
        if len(SYSTEM_MESSAGE) > 0 and self._model_family != "qwen3":
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)

            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                if self._model_family == "qwen2.5":
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt', 
                        **video_kwargs
                    )
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                elif self._model_family == "qwen3":

                    videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                    video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                    video_datas_for_turn = list(video_datas_for_turn)
                    video_metadatas_for_turn = list(video_metadatas_for_turn)

                    inputs = processor(
                        text=[user_input],
                        images=images,
                        videos=video_datas_for_turn,
                        padding=False,
                        do_resize=False,
                        return_tensors='pt',
                        **video_kwargs,
                        video_metadata=video_metadatas_for_turn,
                    )
                else:
                    inputs = processor(
                        text=[user_input], 
                        images=images, 
                        videos=videos_for_this_turn, 
                        padding=False, 
                        do_resize=False, 
                        return_tensors='pt'
                    )
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # Enforce max_seq_length to prevent OOM on abnormally long samples
        if input_ids.shape[0] > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

            # ---- Trim pixel data to match truncated tokens ----
            # Truncation may have removed <|video_pad|> or <|image_pad|>
            # tokens, creating a mismatch with the visual features the
            # encoder will produce.  We trim pixel data and grid metadata
            # so they agree with the remaining placeholder tokens.
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
                    # Every visual token was truncated away
                    all_pixel_values.clear()
                    all_image_grid_thw.clear()
                elif remaining_tokens < orig_tokens:
                    merge_size = 2  # spatial merge factor for all Qwen-VL models
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
                            # keep this video / image fully
                            new_pv.append(pv_cat[pv_off : pv_off + raw_patches])
                            new_thw.append(thw_cat[idx : idx + 1])
                            budget -= vid_tok
                        else:
                            # partially trim temporal dimension
                            keep_t = budget // spatial
                            if keep_t > 0:
                                keep_raw = keep_t * (merge_size * hm) * (merge_size * wm)
                                new_pv.append(pv_cat[pv_off : pv_off + keep_raw])
                                new_thw.append(
                                    torch.tensor([[keep_t, hm, wm]], dtype=thw_cat.dtype)
                                )
                                budget -= keep_t * spatial

                        pv_off += raw_patches

                    all_pixel_values = [torch.cat(new_pv, dim=0)] if new_pv else []
                    all_image_grid_thw = new_thw if new_thw else []

                    # If temporal rounding left more pad tokens than features,
                    # neutralise the excess so counts match exactly.
                    new_feat = (
                        sum(int(t[0, 0] * t[0, 1] * t[0, 2]) for t in all_image_grid_thw)
                        if all_image_grid_thw else 0
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
                pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

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

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args, max_seq_length=32768):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args,
        model_id=model_id, max_seq_length=max_seq_length,
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id,
              max_seq_length=max_seq_length,
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
