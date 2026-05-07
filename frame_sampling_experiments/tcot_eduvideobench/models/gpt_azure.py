"""
models/gpt_Azure.py — GPT-4o-mini via Azure Azure backend for TCoT.

Accepts lists of PIL images (individual frames) rather than video file paths.
Frames are base64-encoded and sent as image_url content blocks, interleaved
with the text prompt.  This mirrors the Gemini Flash approach in the paper.
"""

import base64
import io
from typing import List

from PIL import Image
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from models.base import BaseVLM
import config


def _pil_to_b64(img: Image.Image, max_dim: int = 768) -> str:
    """Resize to max_dim on the long edge, then base64-encode as JPEG."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class GPTAzureModel(BaseVLM):
    """GPT-4o-mini via Azure OpenAI."""

    def load(self):
        print(f"[GPT-Azure] Authenticating with deployment={config.GPT_DEPLOYMENT} …")
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(
            credential, "api://azure/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=config.GPT_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=config.GPT_API_VERSION,
        )
        print("[GPT-Azure] Client ready.")

    def _build_content(self, frames: List[Image.Image], prompt: str) -> list:
        content = [{"type": "text", "text": prompt}]
        for img in frames:
            b64 = _pil_to_b64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "auto",
                },
            })
        return content

    def _infer(self, frames: List[Image.Image], prompt: str,
               max_tokens: int) -> str:
        content = self._build_content(frames, prompt)
        messages = [
            {
                "role": "system",
                "content": "You are an expert video understanding assistant.",
            },
            {"role": "user", "content": content},
        ]
        response = self.client.chat.completions.create(
            model=config.GPT_DEPLOYMENT,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def call_selection(self, frames: List[Image.Image], prompt: str) -> str:
        return self._infer(frames, prompt,
                           max_tokens=config.SELECTION_MAX_TOKENS)

    def call_answering(self, frames: List[Image.Image], prompt: str) -> str:
        return self._infer(frames, prompt,
                           max_tokens=config.ANSWER_MAX_TOKENS)

    def unload(self):
        self.client = None