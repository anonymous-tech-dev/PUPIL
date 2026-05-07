import os
import cv2
import base64
import numpy as np
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from models.base import BaseEvaluator
from decord import VideoReader, cpu


class GPTAzureEvaluator(BaseEvaluator):
    """
    GPT-5.1 video-frame baseline.

    Azure accepts up to ~50 image inputs per request before rejection, so we
    sample 50 frames by default. Reasoning effort is left at "auto" so the
    model can scale thinking budget per question.

    Knobs (env-overridable for ablations):
      GPT_FRAMES_TO_EXTRACT  default 50
      GPT_IMAGE_DETAIL       default "auto"
      GPT_IMAGE_MAX_DIM      default 768
      GPT_REASONING_EFFORT   default "auto"  ({"minimal","low","medium","high","auto"})
      GPT_MAX_COMPLETION_TOKENS  default 1024
    """

    def load(self):
        self.deployment_name        = "gpt-5.1_2025-11-13"
        self.frames_to_extract      = int(os.environ.get("GPT_FRAMES_TO_EXTRACT", "50"))
        self.image_detail           = os.environ.get("GPT_IMAGE_DETAIL", "auto")
        self.max_dim                = int(os.environ.get("GPT_IMAGE_MAX_DIM", "768"))
        # Azure accepts {none, minimal, low, medium, high, xhigh}.
        # gpt-5.1 defaults to 'none' when omitted (no reasoning at all!),
        # so we explicitly default to 'medium' for a fair baseline.
        self.reasoning_effort       = os.environ.get("GPT_REASONING_EFFORT", "medium").strip().lower()
        if self.reasoning_effort in {"", "auto", "default"}:
            self.reasoning_effort = "medium"
        self.max_completion_tokens  = int(os.environ.get("GPT_MAX_COMPLETION_TOKENS", "1024"))

        print(f"🔑 Authenticating with Azure Azure ({self.deployment_name})...")
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, "api://azure/.default")
        return AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )

    def extract_frames(self, video_path: str) -> list:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"Failed to load video {video_path}: {e}")
            return []

        total_frames = len(vr)
        if total_frames <= 0:
            return []

        # Evenly-spaced sampling that always includes first and last frames.
        n = min(self.frames_to_extract, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)
        frames = vr.get_batch(indices).asnumpy()

        base64_frames = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if max(h, w) > self.max_dim:
                scale = self.max_dim / max(h, w)
                frame = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            _, buf = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buf).decode("utf-8"))
        return base64_frames

    def generate_response(self, video_path, prompt):
        frames = self.extract_frames(video_path)

        content_payload = [{"type": "text", "text": prompt}]
        for b64_frame in frames:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_frame}",
                    "detail": self.image_detail,
                },
            })

        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing educational video content."},
            {"role": "user", "content": content_payload},
        ]

        kwargs = dict(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
