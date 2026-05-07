import os
import cv2
import base64
import time
import numpy as np
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
from models.base import BaseEvaluator
from decord import VideoReader, cpu


class GPT54AzureEvaluator(BaseEvaluator):
    """
    GPT-5.4 video-frame baseline.

    Azure accepts up to ~50 image inputs per request, so we sample 50 frames
    by default. Reasoning effort is left at "auto" so the model can scale
    thinking budget per question.
    """

    def load(self):
        self.deployment_name        = "gpt-5.4_2026-03-05"
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
        self.max_retries            = 3
        self.retry_delay            = 5

        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, "api://azure/.default")
        self.client_azure_ai = AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )
        self.client_azure_ai = AzureOpenAI(
            azure_endpoint="https://<AZURE_OPENAI_ENDPOINT>",
            azure_ad_token_provider=token_provider,
            api_version="2024-12-01-preview",
        )
        self.active_client = self.client_azure_ai
        self._consecutive_rate_limits = 0

    def _call_with_fallback(self, messages):
        kwargs = dict(
            model=self.deployment_name,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        for attempt in range(self.max_retries):
            try:
                response = self.active_client.chat.completions.create(**kwargs)
                self._consecutive_rate_limits = 0
                return response.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err or "throttl" in err:
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits >= 3:
                        self.active_client = (
                            self.client_azure_ai if self.active_client is self.client_azure_ai
                            else self.client_azure_ai
                        )
                        self._consecutive_rate_limits = 0
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise
        return "ERROR: All retries exhausted"

    def extract_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            return []
        n = min(self.frames_to_extract, total)
        indices = np.linspace(0, total - 1, n, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        out = []
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
            out.append(base64.b64encode(buf).decode("utf-8"))
        return out

    def generate_response(self, video_path, prompt):
        frames = self.extract_frames(video_path)
        content = [{"type": "text", "text": prompt}]
        for b in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b}",
                    "detail": self.image_detail,
                },
            })
        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing educational video content."},
            {"role": "user", "content": content},
        ]
        return self._call_with_fallback(messages)
