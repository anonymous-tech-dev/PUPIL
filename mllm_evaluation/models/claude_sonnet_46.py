"""Claude Sonnet 4.6 via local GH Copilot Server (OpenAI-compatible proxy).

Mirrors GPT54AzureEvaluator: samples N frames, downscales, base64 JPEG-encodes,
sends as image_url parts in a single chat-completion request.

Requires the GH Copilot Server VS Code extension to be running locally
(or reverse-port-forwarded to the box this script runs on).
"""
import os
import cv2
import base64
import time
import numpy as np
import requests
from models.base import BaseEvaluator
from decord import VideoReader, cpu


class ClaudeSonnet46CopilotEvaluator(BaseEvaluator):
    """
    Claude Sonnet 4.6 video-frame baseline via the local Copilot proxy server.

    The proxy exposes an OpenAI-compatible endpoint at
        http://127.0.0.1:3141/v1/chat/completions
    No API key is needed. Models are listed at /v1/models.

    Note: Anthropic's image limit through the Copilot wrapper is generous
    (50+ frames have been observed), but very large per-image payloads
    will be rejected. Default max_dim=512 keeps payloads safe.
    """

    def load(self):
        self.model_name             = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4.6")
        self.base_url               = os.environ.get(
            "COPILOT_PROXY_URL", "http://127.0.0.1:3141/v1"
        )
        self.frames_to_extract      = int(os.environ.get("CLAUDE_FRAMES_TO_EXTRACT", "50"))
        self.max_dim                = int(os.environ.get("CLAUDE_IMAGE_MAX_DIM", "512"))
        self.jpeg_quality           = int(os.environ.get("CLAUDE_JPEG_QUALITY", "75"))
        self.max_completion_tokens  = int(os.environ.get("CLAUDE_MAX_COMPLETION_TOKENS", "1024"))
        self.timeout_s              = int(os.environ.get("CLAUDE_TIMEOUT_S", "300"))
        self.max_retries            = 3
        self.retry_delay            = 5

        self.endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        self.session = requests.Session()

    def _post(self, payload):
        return self.session.post(self.endpoint, json=payload, timeout=self.timeout_s)

    def _call(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_completion_tokens,
        }
        for attempt in range(self.max_retries):
            try:
                resp = self._post(payload)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                # Retry on transient server errors / rate limits
                body = resp.text
                err = body.lower()
                if resp.status_code in (429, 500, 502, 503, 504) or "rate" in err or "throttl" in err:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                return f"ERROR: {resp.status_code} {body[:500]}"
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return f"ERROR: {e}"
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
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue
            out.append(base64.b64encode(buf).decode("utf-8"))
        return out

    def generate_response(self, video_path, prompt):
        frames = self.extract_frames(video_path)
        content = [{"type": "text", "text": prompt}]
        for b in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"},
            })
        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing educational video content."},
            {"role": "user", "content": content},
        ]
        return self._call(messages)
