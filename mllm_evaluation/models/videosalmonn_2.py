import os
import sys
import torch
import numpy as np
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoConfig, WhisperFeatureExtractor

import logging
import warnings

# 1. Python Warnings (Deprecation, UserWarning, etc.)
warnings.filterwarnings("ignore")

# --- PATH SETUP ---
# Ensure this points to the root of the cloned repo
repo_root = os.path.abspath("/home/video-SALMONN-2")
if repo_root not in sys.path:
    sys.path.append(repo_root)
# ------------------

try:
    from llava.model import VideoSALMONN2ForCausalLM
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except ImportError as e:
    print(f"Error importing LLaVA modules: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

from models.base import BaseEvaluator
import config

class VideoSALMONN2Evaluator(BaseEvaluator):
    def load(self):
        # Update this to the actual path of your downloaded checkpoint
        model_path = "tsinghua-ee/video-SALMONN-2" 
        
        print(f"🔵 Loading config from {model_path}...")
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.config.attn_implementation = "sdpa" 

        # Audio config required by the model class
        audio_config = {
            "audio_visual": True,
            "video_fps": 1,
            "whisper_path": "openai/whisper-large-v3",
            "num_speech_query_token": 1,
            "window_level_Qformer": True,
            "second_per_window": 0.5,
            "second_stride": 0.5,
            "use_final_linear": True,
        }

        print("🔵 Loading Model (this may take a moment)...")
        self.model = VideoSALMONN2ForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            device_map=self.device,
            torch_dtype=self.dtype,
            **audio_config
        )
        self.model.eval()

        print("🔵 Loading Tokenizer...")
        # CRITICAL FIX: Qwen2 models require use_fast=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        self.image_processor = self.model.get_vision_tower().image_processor
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(audio_config['whisper_path'])
        
        self.max_frames = 30 
        self.fps = 1

    def _extract_audio_spectrogram(self, video_path):
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            
            if audio_clip is None:
                audio_data = np.zeros(16000)
                duration = video_clip.duration if video_clip.duration else 1.0
            else:
                # Resample to 16k
                audio_data = audio_clip.to_soundarray(fps=16000)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                duration = video_clip.duration

            # Pad / Truncate
            if len(audio_data) < 16000:
                pad = np.zeros(16000 - len(audio_data))
                audio_data = np.concatenate((audio_data, pad))
            
            max_audio_len = 110 * 16000 
            if len(audio_data) > max_audio_len:
                audio_data = audio_data[:max_audio_len]

            # Chunking for Whisper
            chunk_size = 30 * 16000
            audio_lst = [audio_data[k: k + chunk_size] for k in range(0, len(audio_data), chunk_size)]
            
            spectrograms = [
                self.audio_processor(a, sampling_rate=16000, return_tensors="pt")["input_features"].squeeze() 
                for a in audio_lst
            ]
            
            stacked_spectrograms = torch.stack(spectrograms, dim=0).to(self.device, dtype=self.dtype)
            org_groups = [len(spectrograms)]
            
            return stacked_spectrograms, org_groups, duration

        except Exception as e:
            print(f"⚠️ Audio processing failed for {video_path}: {e}")
            dummy_spec = self.audio_processor(np.zeros(16000), sampling_rate=16000, return_tensors="pt")["input_features"].squeeze()
            return torch.stack([dummy_spec], dim=0).to(self.device, dtype=self.dtype), [1], 1.0

    def generate_response(self, video_path, prompt):
        # Video Processing
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        ori_fps = vr.get_avg_fps()
        
        frame_idx = [k for k in range(0, total_frame_num, round(ori_fps / self.fps))]
        if len(frame_idx) > self.max_frames:
             frame_idx = np.linspace(0, total_frame_num - 1, self.max_frames, dtype=int).tolist()
             
        video = vr.get_batch(frame_idx).asnumpy()
        image_tensor = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values']
        image_tensor = [image_tensor.to(self.device, dtype=self.dtype)]

        # Audio Processing
        spectrogram, org_groups, duration = self._extract_audio_spectrogram(video_path)

        # Prompt Construction
        conv = conv_templates["qwen_1_5"].copy()
        if DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        # Inference
        input_ids = tokenizer_image_token(prompt_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stopping_criteria = KeywordsStoppingCriteria([conv.sep], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                spectrogram=spectrogram,
                modalities=["audio-video"],
                org_groups=org_groups,
                real_time=[duration],
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()