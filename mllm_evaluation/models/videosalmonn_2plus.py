import sys
import os
import torch
import numpy as np
import math
from decord import VideoReader, cpu
from models.base import BaseEvaluator

# --- 1. Setup Local Imports ---
# Adjust this path to where your repo sits in the docker
# We need to access the 'qwenvl' module inside 'video_SALMONN2_plus'
# sys.path.insert(0, "/workspace/local_repos/video-SALMONN-2/video_SALMONN2_plus")
REPO_ROOT = "/workspace/local_repos/video-SALMONN-2/video_SALMONN2_plus"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from torchcodec.decoders import AudioDecoder
    from peft import PeftModel
except ImportError as e:
    raise ImportError(f"Could not import required modules: {e}")

class VideoSalmonn2PlusEvaluator(BaseEvaluator):
    def load(self):
        # --- Configuration ---
        self.adapter_path = "tsinghua-ee/video-SALMONN-2_plus_7B"
        self.base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.whisper_path = "openai/whisper-large-v3"
        
        print(f"Loading Base: {self.base_model_path}")
        print(f"Loading Adapter: {self.adapter_path}")

        # 1. Load Tokenizer & Processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            use_fast=False, 
            padding_side="right",
            trust_remote_code=True
        )
        self.image_processor = Qwen2VLImageProcessorFast.from_pretrained(self.base_model_path)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.whisper_path)

        # 2. Load Base Model
        self.model = video_SALMONN2_plus.from_pretrained(
            self.base_model_path,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2"
        )

        # 3. Load & Merge Adapter
        try:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        except Exception as e:
            print(f"Standard LoRA load failed, detaching audio module to retry: {e}")
            audio_module = self.model.audio
            del self.model.audio
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.base_model.model.audio = audio_module
        
        self.model = self.model.merge_and_unload()
        self.model.eval()

    def _process_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        # Using 16 frames aligns with the feature count seen in your error (features=9568)
        max_frames = 16 
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        video_data = vr.get_batch(frame_indices).asnumpy()
        return video_data

    def _process_audio(self, video_path):
        try:
            decoder = AudioDecoder(video_path, sample_rate=16000, num_channels=1)
            audio_tensor = decoder.get_all_samples()
            audio_data = audio_tensor.cpu().numpy().squeeze(0)
            
            target_sr = 16000
            if audio_data.shape[0] < target_sr:
                padding = target_sr - audio_data.shape[0]
                audio_data = np.pad(audio_data, (0, padding), mode="constant")
            
            chunk_size = 30 * target_sr
            audio_lst = [audio_data[k: k + chunk_size] for k in range(0, len(audio_data), chunk_size)]
            
            spectrogram_lst = [
                self.audio_processor(a, sampling_rate=target_sr, return_tensors="pt")["input_features"].squeeze() 
                for a in audio_lst
            ]
            
            audio_feature = torch.stack(spectrogram_lst, dim=0)
            total_tokens = math.ceil(audio_data.shape[0] / chunk_size) * 60
            return audio_feature, [total_tokens]

        except Exception as e:
            return None, None

    def generate_response(self, video_path: str, prompt: str) -> str:
        # 1. Process Video
        video_frames = self._process_video(video_path)
        video_inputs = self.image_processor.preprocess(images=None, videos=video_frames, return_tensors="pt")
        
        pixel_values_videos = video_inputs["pixel_values_videos"].to(self.device, self.dtype)
        video_grid_thw = video_inputs["video_grid_thw"].to(self.device)

        # 2. Process Audio
        audio_feature, audio_lengths = self._process_audio(video_path)
        if audio_feature is not None:
            audio_feature = audio_feature.to(self.device, self.dtype)

        # 3. Construct Prompt (Corrected Token Calc)
        grid_t = video_grid_thw[0, 0].item()
        grid_h = video_grid_thw[0, 1].item()
        grid_w = video_grid_thw[0, 2].item()
        merge_size = 2 
        
        # FIX: Multiply by grid_t (Time dimension)
        spatial_tokens_per_frame = (grid_h * grid_w) // (merge_size ** 2)
        total_video_tokens = grid_t * spatial_tokens_per_frame
        
        video_token_str = "<|vision_start|>" + "<|video_pad|>" * total_video_tokens + "<|vision_end|>"
        
        if audio_lengths:
            audio_token_str = "<|vision_start|>" + ("<|audio_pad|>" * audio_lengths[0]) + "<|vision_end|>"
            final_content = f"{video_token_str}\n{audio_token_str}\n{prompt}"
        else:
            final_content = f"{video_token_str}\n{prompt}"

        # 4. Tokenize
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 5. Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, # Passes input_ids and attention_mask automatically
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                audio_feature=audio_feature,
                audio_lengths=audio_lengths if audio_lengths else None,
                max_new_tokens=512,
                use_cache=True
            )

        # 6. Decode
        generated_ids = [
            output_ids[len(inputs.input_ids[0]):] for input_ids, output_ids in zip([inputs.input_ids], output_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        return output_text