import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)

    return image_input[0]

def get_video_info(
    video_path, min_pixels, max_pixels, width, height, fps, nframes,
    image_patch_size, return_video_metadata=False,
    total_pixels=None, max_frames=None,
):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    # Clamp nframes to the video's actual frame count to avoid
    # "nframes should in interval [2, total_frames]" errors from qwen-vl-utils
    if nframes is not None:
        total_frames = None
        try:
            import decord
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            del vr
        except Exception:
            pass
        if total_frames is None:
            try:
                import torchvision
                video, _, info = torchvision.io.read_video(
                    video_path, pts_unit="sec", output_format="TCHW",
                )
                total_frames = video.size(0)
                del video
            except Exception:
                pass
        if total_frames is None:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if total_frames <= 0:
                    total_frames = None
            except Exception:
                pass
        if total_frames is not None:
            FRAME_FACTOR = 2  # qwen-vl-utils minimum
            nframes = min(nframes, max(total_frames, FRAME_FACTOR))

    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

    # total_pixels tells fetch_video the whole-video pixel budget so it can
    # dynamically scale per-frame resolution (critical for training where
    # max_seq_length << 128k inference context).
    if total_pixels is not None:
        content["total_pixels"] = total_pixels

    # max_frames hard-caps the number of sampled frames so that even at the
    # minimum per-frame resolution the total tokens stay within budget.
    if max_frames is not None:
        content["max_frames"] = max_frames

    # Conditionally add fps or nframes
    if fps is not None:
        content["fps"] = fps
    elif nframes is not None:
        content["nframes"] = nframes

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(
        messages, 
        return_video_kwargs=True, 
        image_patch_size=image_patch_size, 
        return_video_metadata=return_video_metadata
    )

    return video_input[0], video_kwargs

def samples_per_class_from_ids(label_ids, num_classes):
    
    counts = torch.bincount(
        torch.as_tensor(label_ids, dtype=torch.long),
        minlength=num_classes
    )
    
    return counts.tolist()