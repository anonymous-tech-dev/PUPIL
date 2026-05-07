import os
import cv2
import base64
import math
from azure.identity import (
    DefaultAzureCredential, 
    ChainedTokenCredential, 
    AzureCliCredential, 
    get_bearer_token_provider
)
from openai import AzureOpenAI
from models.base import BaseEvaluator

class GPTAzureEvaluator(BaseEvaluator):
    def load(self):
        # Configuration
        # self.deployment_name = "gpt-4o_2024-11-20" # Or change to gpt-5 if available
        self.deployment_name = "gpt-5.1_2025-11-13" # Or change to gpt-5 if available
        self.frames_to_extract = 10  # Increased for better video context
        
        print(f"🔑 Authenticating with Azure Azure ({self.deployment_name})...")
        self.client = self._initialize_client()

    def _initialize_client(self):
        scope = "api://azure/.default"
        credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(exclude_interactive_browser_credential=True)
            ),
            scope
        )
        
        return AzureOpenAI(
            azure_endpoint='https://<AZURE_OPENAI_ENDPOINT>',
            azure_ad_token_provider=credential,
            api_version='2024-12-01-preview',
        )

    def extract_frames(self, video_path: str) -> list:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
            
        interval = max(1, math.floor(total_frames / self.frames_to_extract))
        base64_frames = []

        for i in range(0, total_frames, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success or len(base64_frames) >= self.frames_to_extract:
                break
            
            # Resize logic (preserve aspect ratio, max 512 width)
            h, w = frame.shape[:2]
            scale = 512 / w
            new_dim = (512, int(h * scale))
            resized = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
            
            _, buffer = cv2.imencode(".jpg", resized)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        video.release()
        return base64_frames

    def generate_response(self, video_path, prompt):
        frames = self.extract_frames(video_path)
        
        content_payload = [{"type": "text", "text": prompt}]
        for b64_frame in frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_frame}",
                    "detail": "low"
                }
            })

        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing educational video content."},
            {"role": "user", "content": content_payload}
        ]

        # Synchronous call (matching main.py flow)
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            # temperature=0.7, # Slightly higher for descriptive tasks
            max_completion_tokens=512,
        )
        return response.choices[0].message.content